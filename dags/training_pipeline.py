"""
DAG 1 : training_pipeline (v2)
───────────────────────────────
Orchestre le cycle complet d'entraînement + tracking MLflow.

NOUVEAUTÉS v2
─────────────
  • MLflow tracking : chaque run est loggué avec métriques, params, artefacts
  • MLflow Model Registry : promotion automatique en "Production" si R² ≥ seuil
  • Le monitoring_pipeline reste indépendant de MLflow (résolution via Postgres)

FLUX
────
  generate_data → train_model → save_reference_stats → notify_success

VARIABLES AIRFLOW (Admin → Variables)
  MODEL_NAME            : drift_regressor
  MODEL_TYPE            : random_forest
  N_ROWS_TRAINING       : 5000
  MIN_R2_FOR_REGISTRY   : 0.70
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
    description="Entraîne le modèle, sauvegarde dans MinIO et tracke avec MLflow",
    schedule="@daily",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training", "mlflow"],
    max_active_runs=1,
    doc_md=__doc__,
) as dag:

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 1 — Générer les données
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_data(**context) -> dict:
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import generate_reference_data
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
    # TÂCHE 2 — Entraîner le modèle + tracker avec MLflow
    # ─────────────────────────────────────────────────────────────────────────
    def _train_model(**context) -> dict:
        """
        Entraîne via run_training_pipeline() et logue tout dans MLflow.

        STRUCTURE MLFLOW
        ────────────────
          Experiment : MODEL_NAME
          Run :
            params/   → model_type, n_rows, dataset_version, hyperparams
            metrics/  → val_r2, val_rmse, val_mae, train_r2, train_rmse, duration
            tags/     → minio_run_id, dag_run_id, dataset_uri
            artifacts → model.joblib, feature_importance.csv

        REGISTRY (si val_r2 ≥ MIN_R2_FOR_REGISTRY)
        ────────────────────────────────────────────
          Nouvelle version → "Production"
          Versions précédentes → "Archived"
        """
        import sys, io, os
        sys.path.insert(0, "/opt/airflow/project")

        import joblib
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        from src.storage.minio_client import (
            download_dataframe_from_uri,
            download_json,
            download_bytes,
        )
        from src.training.train import run_training_pipeline
        from src.config.settings import MINIO_CFG

        ti = context["ti"]
        upstream = ti.xcom_pull(task_ids="generate_data")
        dataset_uri     = upstream["dataset_uri"]
        n_rows          = upstream["n_rows"]
        dataset_version = upstream["dataset_version"]

        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")
        model_type = Variable.get("MODEL_TYPE", default_var="random_forest")
        min_r2     = float(Variable.get("MIN_R2_FOR_REGISTRY", default_var=0.70))

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlops-mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(model_name)

        log.info("Chargement du dataset : %s", dataset_uri)
        df = download_dataframe_from_uri(dataset_uri)

        with mlflow.start_run() as mlflow_run:
            mlflow_run_id = mlflow_run.info.run_id
            log.info("MLflow run : %s", mlflow_run_id[:8])

            # ── Entraînement + sauvegarde MinIO ───────────────────────────
            minio_run_id = run_training_pipeline(
                df=df,
                dataset_version=dataset_version,
                dataset_uri=dataset_uri,
                model_type=model_type,
            )

            # ── Charger métriques + params depuis MinIO ───────────────────
            metrics = download_json(
                bucket=MINIO_CFG.bucket,
                object_name=f"runs/{minio_run_id}/metrics.json",
            )
            params = download_json(
                bucket=MINIO_CFG.bucket,
                object_name=f"runs/{minio_run_id}/params.json",
            )

            # ── Logger params + métriques dans MLflow ─────────────────────
            mlflow.log_params({
                "model_type":       model_type,
                "dataset_version":  dataset_version,
                "n_rows":           n_rows,
                **{k: str(v) for k, v in params.items()},
            })
            mlflow.log_metrics({
                "val_r2":             metrics.get("val_r2", 0),
                "val_rmse":           metrics.get("val_rmse", 0),
                "val_mae":            metrics.get("val_mae", 0),
                "train_r2":           metrics.get("train_r2", 0),
                "train_rmse":         metrics.get("train_rmse", 0),
                "train_duration_sec": metrics.get("train_duration_sec", 0),
            })
            mlflow.set_tags({
                "minio_run_id": minio_run_id,
                "dag_run_id":   context["run_id"],
                "dataset_uri":  dataset_uri,
            })

            # ── Artefact : modèle sklearn ──────────────────────────────────
            try:
                model_bytes = download_bytes(
                    bucket=MINIO_CFG.bucket,
                    object_name=f"runs/{minio_run_id}/model.joblib",
                )
                model_obj = joblib.load(io.BytesIO(model_bytes))
                mlflow.sklearn.log_model(model_obj, artifact_path="model")
                log.info("Modèle loggué dans MLflow")
            except Exception as exc:
                log.warning("Log modèle MLflow échoué (non bloquant) : %s", exc)

            # ── Artefact : feature importance ──────────────────────────────
            try:
                fi_bytes = download_bytes(
                    bucket=MINIO_CFG.bucket,
                    object_name=f"runs/{minio_run_id}/reports/feature_importance.csv",
                )
                with open("/tmp/feature_importance.csv", "wb") as f:
                    f.write(fi_bytes)
                mlflow.log_artifact("/tmp/feature_importance.csv", artifact_path="reports")
            except Exception:
                pass  # Ridge n'a pas de feature importance

            val_r2 = metrics.get("val_r2", 0)

            # ── Model Registry ─────────────────────────────────────────────
            mlflow_model_version = None
            if val_r2 >= min_r2:
                try:
                    client = MlflowClient(tracking_uri=tracking_uri)
                    mv = mlflow.register_model(f"runs:/{mlflow_run_id}/model", model_name)
                    mlflow_model_version = mv.version

                    # Archiver les versions précédentes en Production
                    for old_v in client.get_latest_versions(model_name, stages=["Production"]):
                        if old_v.version != mv.version:
                            client.transition_model_version_stage(
                                name=model_name, version=old_v.version, stage="Archived"
                            )
                            log.info("Version %s → Archived", old_v.version)

                    # Promouvoir la nouvelle version
                    client.transition_model_version_stage(
                        name=model_name, version=mv.version, stage="Production"
                    )
                    log.info(
                        "Modèle '%s' v%s → Production (R²=%.4f ≥ %.2f)",
                        model_name, mv.version, val_r2, min_r2,
                    )
                except Exception as exc:
                    log.warning("Registry MLflow échoué (non bloquant) : %s", exc)
            else:
                log.warning(
                    "R²=%.4f < seuil %.2f → modèle non enregistré dans le Registry",
                    val_r2, min_r2,
                )

        log.info(
            "Run terminé : minio=%s | mlflow=%s | R²=%.4f",
            minio_run_id[:8], mlflow_run_id[:8], val_r2,
        )
        return {
            "run_id":               minio_run_id,
            "mlflow_run_id":        mlflow_run_id,
            "mlflow_model_version": mlflow_model_version,
            "metrics":              metrics,
            "dataset_uri":          dataset_uri,
            "n_rows":               n_rows,
            "dataset_version":      dataset_version,
            "model_type":           model_type,
        }

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 3 — Sauvegarder les stats de référence
    # ─────────────────────────────────────────────────────────────────────────
    def _save_reference_stats(**context) -> dict:
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import compute_feature_stats
        from src.storage.minio_client import (
            download_dataframe_from_uri,
            make_stats_object_name,
            upload_json,
        )
        from src.storage.postgres_client import insert_reference_stats, insert_training_run

        ti = context["ti"]
        upstream = ti.xcom_pull(task_ids="train_model")
        run_id          = upstream["run_id"]
        metrics         = upstream["metrics"]
        dataset_uri     = upstream["dataset_uri"]
        n_rows          = upstream["n_rows"]
        dataset_version = upstream["dataset_version"]
        model_type      = upstream["model_type"]
        mlflow_run_id   = upstream.get("mlflow_run_id", "")

        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")

        # Postgres : training_runs
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
        log.info(
            "training_run inséré : run_id=%s | mlflow=%s",
            run_id[:8], mlflow_run_id[:8] if mlflow_run_id else "N/A",
        )

        # Postgres + MinIO : stats de référence
        df = download_dataframe_from_uri(dataset_uri)
        features = [c for c in df.columns if c != "y"]
        stats_df = compute_feature_stats(df, features)

        insert_reference_stats(run_id=run_id, stats_df=stats_df)

        stats_dict = stats_df.round(6).to_dict(orient="index")
        object_name = make_stats_object_name(run_id)
        stats_uri = upload_json(stats_dict, bucket="feature-stats", object_name=object_name)
        log.info("Stats : Postgres ✓ | MinIO : %s", stats_uri)

        return {
            "stats_uri":            stats_uri,
            "run_id":               run_id,
            "mlflow_run_id":        mlflow_run_id,
            "mlflow_model_version": upstream.get("mlflow_model_version"),
        }

    save_reference_stats = PythonOperator(
        task_id="save_reference_stats",
        python_callable=_save_reference_stats,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 4 — Notifier
    # ─────────────────────────────────────────────────────────────────────────
    def _notify_success(**context) -> None:
        ti = context["ti"]
        train_result = ti.xcom_pull(task_ids="train_model")
        stats_result = ti.xcom_pull(task_ids="save_reference_stats")
        metrics = train_result["metrics"]
        mv = train_result.get("mlflow_model_version")

        log.info("=" * 60)
        log.info("TRAINING PIPELINE v2 — SUCCÈS")
        log.info("  MinIO run_id     : %s", train_result["run_id"])
        log.info("  MLflow run_id    : %s", train_result.get("mlflow_run_id", "N/A"))
        log.info("  Registry version : %s", f"v{mv}" if mv else "non enregistré (R² trop bas)")
        log.info("  Modèle           : %s", train_result["model_type"])
        log.info("  Val R²           : %.4f", metrics.get("val_r2", 0))
        log.info("  Val RMSE         : %.4f", metrics.get("val_rmse", 0))
        log.info("  Stats MinIO      : %s", stats_result["stats_uri"])
        log.info("=" * 60)

    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=_notify_success,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DÉPENDANCES
    # ─────────────────────────────────────────────────────────────────────────
    generate_data >> train_model >> save_reference_stats >> notify_success