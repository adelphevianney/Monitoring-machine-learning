"""
DAG 1 : training_pipeline
─────────────────────────
Orchestre le cycle complet d'entraînement du modèle.

QUAND S'EXÉCUTE-T-IL ?
  Toutes les 24 heures par défaut (configurable).
  Peut aussi être déclenché manuellement depuis l'UI Airflow
  ou par le DAG de monitoring quand un ré-entraînement est nécessaire.

CE QU'IL FAIT (dans l'ordre) :
  1. generate_data        — génère un nouveau dataset synthétique
  2. save_to_minio        — sauvegarde le dataset dans MinIO (raw-datasets)
  3. train_model          — entraîne le modèle, calcule les métriques
  4. log_to_mlflow        — logue params + métriques + artefacts dans MLflow
  5. save_reference_stats — calcule et stocke les stats de distribution
                            dans Postgres ET dans MinIO (feature-stats)
  6. register_model       — enregistre le modèle dans le MLflow Model Registry
                            et le promeut en Staging si les métriques sont OK

COMMENT LES TÂCHES SE PASSENT DES DONNÉES ?
  Airflow ne permet pas de passer des DataFrames entre tâches directement.
  On utilise XCom pour passer les métadonnées légères (URIs, run_ids, métriques).
  Les données lourdes (datasets) transitent via MinIO.

  Flux XCom :
    generate_data   → XCom : dataset_uri (chemin MinIO)
    train_model     → XCom : mlflow_run_id, metrics (dict)
    log_to_mlflow   → XCom : mlflow_run_id confirmé
    register_model  → XCom : model_version

VARIABLES AIRFLOW REQUISES (à définir dans Admin → Variables) :
  MLFLOW_TRACKING_URI   : http://mlflow:5000
  MINIO_ENDPOINT        : minio:9000
  MINIO_ACCESS_KEY      : minioadmin
  MINIO_SECRET_KEY      : minioadmin
  POSTGRES_HOST         : postgres
  MODEL_NAME            : drift_regressor
  MIN_R2_FOR_REGISTRY   : 0.70   (seuil minimum pour enregistrer le modèle)
"""

import json
import logging
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import TriggerRule

#from airflow.operators.python import PythonOperator
#from airflow.operators.trigger_dagrun import TriggerDagRunOperator
#from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ── Paramètres par défaut des tâches ─────────────────────────────────────────
# Ces paramètres s'appliquent à toutes les tâches du DAG sauf surcharge locale.
# On_failure_callback : branchement possible vers Slack/PagerDuty

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,           # chaque run est indépendant du précédent
    "retries": 1,                       # 1 retry automatique en cas d'échec
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# ── Définition du DAG ─────────────────────────────────────────────────────────
with DAG(
    dag_id="training_pipeline",
    description="Entraîne le modèle, logue dans MLflow, enregistre dans le registry",
    schedule="@daily",         # tous les jours à minuit UTC
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,                      # ne pas rejouer les runs passés au démarrage
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training"],
    # max_active_runs=1 : un seul run actif à la fois pour éviter les conflits
    # sur le Model Registry MLflow
    max_active_runs=1,
    doc_md=__doc__,
) as dag:

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 1 — Générer les données
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_data(**context) -> dict:
        """
        Génère un nouveau dataset synthétique et le sauvegarde dans MinIO.

        Retourne via XCom un dict avec :
          - dataset_uri   : URI MinIO du dataset sauvegardé
          - n_rows        : nombre de lignes générées
          - dataset_version : identifiant lisible de la version
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.config.settings import DATA_CFG
        from src.data_generation.generator import generate_reference_data
        from src.storage.minio_client import (
            ensure_buckets_exist,
            make_dataset_object_name,
            upload_dataframe,
        )

        # Assurer que les buckets existent (idempotent)
        ensure_buckets_exist()

        # La seed change à chaque run pour avoir de la variabilité
        # On utilise l'execution_date pour la reproductibilité
        exec_date = context["execution_date"]
        seed = int(exec_date.timestamp()) % 100_000

        n_rows = int(Variable.get("N_ROWS_TRAINING", default_var=5000))

        log.info("Génération de %d lignes (seed=%d)…", n_rows, seed)
        df = generate_reference_data(n_rows=n_rows, seed=seed)

        # Nommage horodaté pour traçabilité
        dataset_version = exec_date.strftime("v%Y%m%d")
        object_name = make_dataset_object_name("reference", dataset_version)

        uri = upload_dataframe(df, bucket="raw-datasets", object_name=object_name)
        log.info("Dataset sauvegardé : %s", uri)

        return {
            "dataset_uri": uri,
            "n_rows": n_rows,
            "dataset_version": dataset_version,
            "seed": seed,
        }

    generate_data = PythonOperator(
        task_id="generate_data",
        python_callable=_generate_data,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 2 — Entraîner le modèle
    # ─────────────────────────────────────────────────────────────────────────
    def _train_model(**context) -> dict:
        """
        Charge le dataset depuis MinIO et entraîne le modèle.

        Retourne via XCom :
          - mlflow_run_id : ID du run MLflow créé
          - metrics       : dict des métriques (val_rmse, val_r2, etc.)
          - dataset_uri   : URI du dataset (propagé pour les tâches suivantes)
          - n_rows        : nombre de lignes (propagé)
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        import mlflow
        import mlflow.sklearn
        from src.storage.minio_client import download_dataframe_from_uri
        from src.training.train import train

        # Récupérer les infos du dataset depuis la tâche précédente
        ti = context["ti"]
        upstream = ti.xcom_pull(task_ids="generate_data")
        dataset_uri = upstream["dataset_uri"]
        n_rows = upstream["n_rows"]
        dataset_version = upstream["dataset_version"]

        log.info("Chargement du dataset : %s", dataset_uri)
        df = download_dataframe_from_uri(dataset_uri)

        # Configuration MLflow
        tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var="http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("mlops_drift_monitoring")

        model_type = Variable.get("MODEL_TYPE", default_var="random_forest")

        # Paramètres du modèle (stockés comme variable JSON dans Airflow)
        model_params_json = Variable.get(
            "MODEL_PARAMS",
            default_var='{"n_estimators": 100, "max_depth": 6, '
                        '"min_samples_leaf": 4, "random_state": 42, "n_jobs": -1}'
        )
        model_params = json.loads(model_params_json)

        log.info("Entraînement du modèle %s…", model_type)

        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id

            mlflow.set_tags({
                "model_type": model_type,
                "dataset_version": dataset_version,
                "dataset_uri": dataset_uri,
                "airflow_run_id": context["run_id"],
                "dag_id": context["dag"].dag_id,
            })
            mlflow.log_params(model_params)

            model, metrics, X_val, y_val = train(
                df, model_type=model_type, model_params=model_params
            )
            mlflow.log_metrics(metrics)

            # Signature du modèle pour la validation des inputs en production
            import pandas as pd
            features = [c for c in df.columns if c != "y"]
            signature = mlflow.models.infer_signature(
                pd.DataFrame(X_val), model.predict(X_val)
            )
            mlflow.sklearn.log_model(
                model, artifact_path="model", signature=signature
            )

        log.info(
            "Entraînement terminé : run_id=%s | R²=%.4f | RMSE=%.4f",
            mlflow_run_id[:8], metrics["val_r2"], metrics["val_rmse"]
        )

        return {
            "mlflow_run_id": mlflow_run_id,
            "metrics": metrics,
            "dataset_uri": dataset_uri,
            "n_rows": n_rows,
            "model_type": model_type,
        }

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 3 — Sauvegarder les stats de référence
    # ─────────────────────────────────────────────────────────────────────────
    def _save_reference_stats(**context) -> dict:
        """
        Calcule les statistiques de distribution du dataset d'entraînement
        et les persiste dans DEUX endroits :
          - Postgres (table reference_feature_stats) : pour le monitoring
          - MinIO (bucket feature-stats)             : pour l'audit et le replay

        Pourquoi deux endroits ?
          Postgres → requêtes rapides pendant le monitoring (lecture SQL)
          MinIO    → conservation longue durée, replay d'analyse historique
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import compute_feature_stats
        from src.storage.minio_client import (
            download_dataframe_from_uri,
            make_stats_object_name,
            upload_json,
        )
        from src.storage.postgres_client import (
            insert_reference_stats,
            insert_training_run,
        )

        ti = context["ti"]
        upstream = ti.xcom_pull(task_ids="train_model")
        mlflow_run_id = upstream["mlflow_run_id"]
        metrics = upstream["metrics"]
        dataset_uri = upstream["dataset_uri"]
        n_rows = upstream["n_rows"]

        # Charger le dataset pour calculer les stats
        df = download_dataframe_from_uri(dataset_uri)
        features = [c for c in df.columns if c != "y"]
        stats_df = compute_feature_stats(df, features)

        # Persister dans Postgres
        # On crée d'abord l'entrée dans training_runs pour la clé étrangère
        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")
        insert_training_run(
            mlflow_run_id=mlflow_run_id,
            model_name=model_name,
            dataset_uri=dataset_uri,
            n_rows=n_rows,
            val_rmse=metrics.get("val_rmse"),
            val_r2=metrics.get("val_r2"),
            status="success",
        )
        insert_reference_stats(mlflow_run_id=mlflow_run_id, stats_df=stats_df)
        log.info("Stats de référence insérées dans Postgres pour run %s", mlflow_run_id[:8])

        # Persister dans MinIO
        stats_dict = stats_df.round(6).to_dict(orient="index")
        object_name = make_stats_object_name(mlflow_run_id)
        stats_uri = upload_json(stats_dict, bucket="feature-stats", object_name=object_name)
        log.info("Stats de référence sauvegardées dans MinIO : %s", stats_uri)

        return {"stats_uri": stats_uri, "mlflow_run_id": mlflow_run_id}

    save_reference_stats = PythonOperator(
        task_id="save_reference_stats",
        python_callable=_save_reference_stats,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 4 — Enregistrer le modèle dans le MLflow Model Registry
    # ─────────────────────────────────────────────────────────────────────────
    def _register_model(**context) -> dict:
        """
        Enregistre le modèle dans le MLflow Model Registry si les métriques
        dépassent le seuil minimum.

        Logique de promotion :
          - Si R² >= MIN_R2_FOR_REGISTRY → enregistrer + promouvoir en Staging
          - Si R² < seuil → logguer un warning, ne pas enregistrer
            (le modèle en production actuel reste inchangé)

        La transition Staging → Production reste manuelle par défaut.
        Elle peut être automatisée en ajoutant une validation sur un
        jeu de données holdout.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        import mlflow
        from mlflow.tracking import MlflowClient

        ti = context["ti"]
        upstream_train = ti.xcom_pull(task_ids="train_model")
        mlflow_run_id = upstream_train["mlflow_run_id"]
        metrics = upstream_train["metrics"]
        val_r2 = metrics.get("val_r2", 0.0)

        min_r2 = float(Variable.get("MIN_R2_FOR_REGISTRY", default_var="0.70"))
        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")

        tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var="http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        if val_r2 < min_r2:
            log.warning(
                "R²=%.4f < seuil=%.2f — modèle non enregistré dans le registry.",
                val_r2, min_r2
            )
            return {"registered": False, "reason": f"R²={val_r2:.4f} < {min_r2}"}

        # Enregistrer le modèle
        model_uri = f"runs:/{mlflow_run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        model_version = int(mv.version)

        log.info(
            "Modèle enregistré : %s v%d (run=%s)",
            model_name, model_version, mlflow_run_id[:8]
        )

        # Promouvoir en Staging (pour validation avant prod)
        client.transition_model_version_stage(
            name=model_name,
            version=str(model_version),
            stage="Staging",
            archive_existing_versions=False,
        )
        log.info("Modèle %s v%d promu en Staging", model_name, model_version)

        # Mettre à jour la version dans Postgres
        from src.storage.postgres_client import update_training_run_status
        update_training_run_status(
            mlflow_run_id=mlflow_run_id,
            status="success",
            model_version=model_version,
        )

        return {
            "registered": True,
            "model_name": model_name,
            "model_version": model_version,
            "mlflow_run_id": mlflow_run_id,
        }

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 5 — Notifier la fin du pipeline (optionnel)
    # ─────────────────────────────────────────────────────────────────────────
    def _notify_success(**context) -> None:
        """
        Log de fin de pipeline. Point d'extension pour :
          - Notification Slack
          - Email
          - Webhook vers un système de supervision
        """
        ti = context["ti"]
        reg_result = ti.xcom_pull(task_ids="register_model")
        train_result = ti.xcom_pull(task_ids="train_model")
        metrics = train_result["metrics"]

        log.info("=" * 55)
        log.info("TRAINING PIPELINE TERMINÉ AVEC SUCCÈS")
        log.info("  Val R²   : %.4f", metrics["val_r2"])
        log.info("  Val RMSE : %.4f", metrics["val_rmse"])
        if reg_result.get("registered"):
            log.info(
                "  Modèle   : %s v%d → Staging",
                reg_result["model_name"], reg_result["model_version"]
            )
        else:
            log.warning("  Modèle NON enregistré : %s", reg_result.get("reason"))
        log.info("=" * 55)

    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=_notify_success,
        # S'exécute même si register_model a produit un résultat "registered=False"
        # car c'est un cas normal, pas une erreur
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DÉPENDANCES — ordre d'exécution
    # ─────────────────────────────────────────────────────────────────────────
    #
    #  generate_data → train_model → save_reference_stats → register_model
    #                                                              ↓
    #                                                       notify_success
    #
    generate_data >> train_model >> save_reference_stats >> register_model >> notify_success
