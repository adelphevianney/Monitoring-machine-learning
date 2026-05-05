"""
DAG 1 : training_pipeline
─────────────────────────
Orchestre le cycle complet d'entraînement du modèle.

QUAND S'EXÉCUTE-T-IL ?
  Toutes les 24 heures par défaut.
  Peut aussi être déclenché manuellement ou par le DAG de monitoring.

CE QU'IL FAIT (dans l'ordre) :
  1. generate_data        — génère un dataset synthétique → MinIO (raw-datasets)
  2. train_model          — entraîne le modèle → artefacts dans MinIO sous runs/<run_id>/
  3. save_reference_stats — calcule les stats de distribution → Postgres + MinIO (feature-stats)
  4. notify_success       — log de fin (extensible Slack/email)

FLUX XCom :
  generate_data → dataset_uri, n_rows, dataset_version, seed
  train_model   → run_id, metrics, dataset_uri, n_rows, dataset_version, model_type

VARIABLES AIRFLOW (Admin → Variables) :
  MODEL_NAME       : drift_regressor
  MODEL_TYPE       : random_forest
  N_ROWS_TRAINING  : 5000
  MIN_R2           : 0.70
"""

import logging
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="training_pipeline",
    description="Entraîne le modèle et sauvegarde les artefacts dans MinIO",
    schedule="@daily",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training"],
    max_active_runs=1,
    doc_md=__doc__,
) as dag:

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 1 — Générer les données
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_data(**context) -> dict:
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import generate_reference_data, generate_drifted_data, generate_partial_drift
        from src.storage.minio_client import (
            ensure_buckets_exist,
            make_dataset_object_name,
            upload_dataframe,
        )

        ensure_buckets_exist()

        exec_date = context["execution_date"]
        seed = int(exec_date.timestamp()) % 100_000
        n_rows = int(Variable.get("N_ROWS_TRAINING", default_var=5000))

        log.info("Génération de %d lignes (seed=%d)…", n_rows, seed)
        df = generate_reference_data(n_rows=n_rows, seed=seed)

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
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.minio_client import download_dataframe_from_uri, download_json
        from src.training.train import run_training_pipeline
        from src.config.settings import MINIO_CFG

        ti = context["ti"]
        upstream = ti.xcom_pull(task_ids="generate_data")
        dataset_uri = upstream["dataset_uri"]
        n_rows = upstream["n_rows"]
        dataset_version = upstream["dataset_version"]

        log.info("Chargement du dataset : %s", dataset_uri)
        df = download_dataframe_from_uri(dataset_uri)

        model_type = Variable.get("MODEL_TYPE", default_var="random_forest")

        log.info("Lancement du pipeline d'entraînement (%s)…", model_type)
        run_id = run_training_pipeline(
            df=df,
            dataset_version=dataset_version,
            dataset_uri=dataset_uri,
            model_type=model_type,
        )

        # Lire les métriques depuis MinIO pour les propager en XCom
        metrics = download_json(
            bucket=MINIO_CFG.bucket,
            object_name=f"runs/{run_id}/metrics.json",
        )

        log.info(
            "Entraînement terminé : run_id=%s | R²=%.4f | RMSE=%.4f",
            run_id, metrics.get("val_r2", 0), metrics.get("val_rmse", 0),
        )

        return {
            "run_id": run_id,
            "metrics": metrics,
            "dataset_uri": dataset_uri,
            "n_rows": n_rows,
            "dataset_version": dataset_version,
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
        1. Insère le run dans training_runs (Postgres) avec status='success'
        2. Calcule les stats de distribution et les persiste dans :
             - Postgres : reference_feature_stats (requêtes rapides au monitoring)
             - MinIO    : bucket feature-stats (audit et replay historique)
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
        run_id = upstream["run_id"]
        metrics = upstream["metrics"]
        dataset_uri = upstream["dataset_uri"]
        n_rows = upstream["n_rows"]
        dataset_version = upstream["dataset_version"]
        model_type = upstream["model_type"]

        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")

        # ── Insérer le run dans training_runs ──────────────────────────────
        # insert_training_run() attend ces paramètres précis (voir postgres_client.py)
        insert_training_run(
            run_id=run_id,
            model_name=model_name,
            dataset_uri=dataset_uri,
            n_rows=n_rows,
            val_rmse=metrics.get("val_rmse"),
            val_r2=metrics.get("val_r2"),
            train_duration_sec=metrics.get("train_duration_sec"),
            dataset_version=dataset_version,
            status="success",
        )
        log.info("training_run inséré : run_id=%s", run_id[:8])

        # ── Calcul et persistance des stats ───────────────────────────────
        df = download_dataframe_from_uri(dataset_uri)
        features = [c for c in df.columns if c != "y"]
        stats_df = compute_feature_stats(df, features)

        # Postgres
        insert_reference_stats(run_id=run_id, stats_df=stats_df)
        log.info("Stats insérées dans Postgres pour run_id=%s", run_id[:8])

        # MinIO (copie indexée par date dans le bucket feature-stats)
        stats_dict = stats_df.round(6).to_dict(orient="index")
        object_name = make_stats_object_name(run_id)
        stats_uri = upload_json(stats_dict, bucket="feature-stats", object_name=object_name)
        log.info("Stats sauvegardées dans MinIO : %s", stats_uri)

        return {"stats_uri": stats_uri, "run_id": run_id}

    save_reference_stats = PythonOperator(
        task_id="save_reference_stats",
        python_callable=_save_reference_stats,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 4 — Notifier la fin du pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _notify_success(**context) -> None:
        ti = context["ti"]
        train_result = ti.xcom_pull(task_ids="train_model")
        stats_result = ti.xcom_pull(task_ids="save_reference_stats")
        metrics = train_result["metrics"]

        log.info("=" * 55)
        log.info("TRAINING PIPELINE TERMINÉ AVEC SUCCÈS")
        log.info("  Run ID   : %s", train_result["run_id"])
        log.info("  Modèle   : %s", train_result["model_type"])
        log.info("  Val R²   : %.4f", metrics.get("val_r2", 0))
        log.info("  Val RMSE : %.4f", metrics.get("val_rmse", 0))
        log.info("  Stats    : %s", stats_result["stats_uri"])
        log.info("=" * 55)

    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=_notify_success,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DÉPENDANCES
    # ─────────────────────────────────────────────────────────────────────────
    generate_data >> train_model >> save_reference_stats >> notify_success