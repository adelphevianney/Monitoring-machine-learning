"""
DAG : drift_monitoring_pipeline (v2)
─────────────────────────────────────
Monitoring du drift — MinIO + Postgres + Elasticsearch + Kibana.

NOUVEAUTÉS v2
─────────────
  • Snapshots JSON dans MinIO en plus du parquet (lisibles, archivables)
  • Indexation dans Elasticsearch → visualisation Kibana en temps réel
  • Nouveaux tests de drift : Wasserstein + Chi² (x4) + Jensen-Shannon
  • DRIFT_SIMULATION_FACTOR pilotable depuis Airflow Admin > Variables
    0.0 = stable | 0.5 = drift partiel | 1.0 = drift complet
  • Tâche dédiée index_elasticsearch (découplée, non bloquante)

FLUX
────
  1. load_reference_run   → Postgres : dernier run réussi
  2. collect_current_data → snapshot parquet + JSON metadata → MinIO
  3. compute_drift        → PSI + KS/Chi² + Wasserstein + JS
  4. store_postgres       → drift_monitoring_runs + drift_feature_metrics
  5. index_elasticsearch  → ES (non bloquant) + snapshot JSON complet → MinIO
  6. evaluate_alert       → branch : log_alert_only | trigger_retraining
  7. pipeline_done

VARIABLES AIRFLOW (Admin → Variables)
  MODEL_NAME                  : drift_regressor
  DRIFT_SIMULATION_FACTOR     : 0.0  (0=stable → 1=drift total)
  N_ROWS_MONITORING           : 1000
"""

import logging
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=20),
}

with DAG(
    dag_id="drift_monitoring_pipeline",
    description="Monitoring du drift v2 — MinIO + Postgres + Elasticsearch",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "monitoring", "drift", "elasticsearch"],
    max_active_runs=1,
) as dag:

    # ─────────────────────────────────────────────────────────
    # 1. LOAD REFERENCE RUN
    # ─────────────────────────────────────────────────────────
    def _load_reference_run(**context):
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.postgres_client import get_latest_successful_run
        from src.storage.minio_client import download_json
        from src.config.settings import MINIO_CFG

        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")
        run_info = get_latest_successful_run(model_name)

        if not run_info:
            raise AirflowSkipException(f"Pas de run réussi pour '{model_name}'")

        run_id = run_info["run_id"]
        try:
            tags = download_json(bucket=MINIO_CFG.bucket, object_name=f"runs/{run_id}/tags.json")
            features = tags.get("features", [])
        except Exception as exc:
            log.warning("tags.json introuvable : %s", exc)
            features = []

        log.info("Référence : run_id=%s | %d features", run_id[:8], len(features))
        return {"run_id": run_id, "dataset_uri": run_info["dataset_uri"], "features": features}

    load_reference_run = PythonOperator(
        task_id="load_reference_run",
        python_callable=_load_reference_run,
    )

    # ─────────────────────────────────────────────────────────
    # 2. COLLECT CURRENT DATA
    # ─────────────────────────────────────────────────────────
    def _collect_current_data(**context):
        """
        Génère le snapshot courant et l'upload dans MinIO.

        DRIFT_SIMULATION_FACTOR (Airflow Variable) :
          0.0 → données stables   (tests de non-régression)
          0.5 → drift partiel     (simulation réaliste)
          1.0 → drift complet     (test des alertes)
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import generate_partial_drift
        from src.storage.minio_client import (
            upload_dataframe, upload_json, make_snapshot_object_name,
        )

        exec_date = context["execution_date"]
        window_start = exec_date - timedelta(hours=1)
        window_end = exec_date

        drift_factor = float(Variable.get("DRIFT_SIMULATION_FACTOR", default_var=1.0))
        n_rows = int(Variable.get("N_ROWS_MONITORING", default_var=1000))
        seed = int(exec_date.timestamp()) % 100_000

        df_current = generate_partial_drift(drift_factor=drift_factor, n_rows=n_rows, seed=seed)

        # Parquet (données brutes pour calcul du drift)
        object_name = make_snapshot_object_name(window_start, window_end)
        snapshot_uri = upload_dataframe(df_current, bucket="monitoring-snapshots", object_name=object_name)

        # JSON metadata (v2 — lisible, archivable)
        json_name = object_name.replace(".parquet", "_metadata.json")
        upload_json(
            {
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "drift_factor": drift_factor,
                "n_rows": n_rows,
                "seed": seed,
                "snapshot_uri": snapshot_uri,
            },
            bucket="monitoring-snapshots",
            object_name=json_name,
        )

        log.info("Snapshot : %s (%d lignes, factor=%.2f)", snapshot_uri, n_rows, drift_factor)
        return {
            "snapshot_uri": snapshot_uri,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "drift_factor": drift_factor,
        }

    collect_current_data = PythonOperator(
        task_id="collect_current_data",
        python_callable=_collect_current_data,
    )

    # ─────────────────────────────────────────────────────────
    # 3. COMPUTE DRIFT
    # ─────────────────────────────────────────────────────────
    def _compute_drift(**context):
        """
        Calcule PSI + KS/Chi² + Wasserstein + Jensen-Shannon.
        Sélection automatique du test selon le type de variable.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.monitoring.drift_metrics import compute_drift_report, feature_results_to_rows
        from src.monitoring.alerting import build_monitoring_verdict
        from src.storage.minio_client import download_dataframe_from_uri
        from src.storage.postgres_client import count_consecutive_alerts

        ti = context["ti"]
        ref_info = ti.xcom_pull(task_ids="load_reference_run")
        data_info = ti.xcom_pull(task_ids="collect_current_data")

        df_ref = download_dataframe_from_uri(ref_info["dataset_uri"])
        df_cur = download_dataframe_from_uri(data_info["snapshot_uri"])

        features = ref_info.get("features") or [c for c in df_ref.columns if c != "y"]

        feature_results = compute_drift_report(df_ref, df_cur, features)
        n_consecutive = count_consecutive_alerts(ref_info["run_id"])
        verdict = build_monitoring_verdict(feature_results, consecutive_alerts=n_consecutive)

        log.info("Drift : alert=%s | score=%.4f | %d drifted",
                 verdict.alert_level, verdict.drift_score_global, verdict.n_features_drifted)

        return {
            "feature_metrics": feature_results_to_rows(feature_results),
            "global_score": verdict.drift_score_global,
            "alert_level": verdict.alert_level,
            "drift_detected": verdict.drift_detected,
            "n_features_drifted": verdict.n_features_drifted,
            "should_retrain": verdict.should_retrain,
            "retrain_reason": verdict.retrain_reason,
        }

    compute_drift = PythonOperator(
        task_id="compute_drift",
        python_callable=_compute_drift,
    )

    # ─────────────────────────────────────────────────────────
    # 4. STORE IN POSTGRES
    # ─────────────────────────────────────────────────────────
    def _store_postgres(**context):
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.postgres_client import insert_monitoring_run, insert_drift_feature_metrics

        ti = context["ti"]
        ref_info  = ti.xcom_pull(task_ids="load_reference_run")
        data_info = ti.xcom_pull(task_ids="collect_current_data")
        drift_info = ti.xcom_pull(task_ids="compute_drift")

        monitoring_run_id = insert_monitoring_run(
            ref_run_id=ref_info["run_id"],
            observation_start=datetime.fromisoformat(data_info["window_start"]),
            observation_end=datetime.fromisoformat(data_info["window_end"]),
            drift_score_global=drift_info["global_score"],
            drift_detected=drift_info["drift_detected"],
            alert_level=drift_info["alert_level"],
            n_features_drifted=drift_info["n_features_drifted"],
            input_dataset_uri=data_info["snapshot_uri"],
        )

        insert_drift_feature_metrics(monitoring_run_id, drift_info["feature_metrics"])

        log.info("Postgres OK : %s | alert=%s", monitoring_run_id[:8], drift_info["alert_level"])
        return {"monitoring_run_id": monitoring_run_id}

    store_postgres = PythonOperator(
        task_id="store_postgres",
        python_callable=_store_postgres,
    )

    # ─────────────────────────────────────────────────────────
    # 5. INDEX ELASTICSEARCH (non bloquant)
    # ─────────────────────────────────────────────────────────
    def _index_elasticsearch(**context):
        """
        Indexe dans ES et uploade le snapshot JSON complet dans MinIO.
        Les erreurs ES ne font pas échouer le DAG.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.es_client import (
            ensure_indices_exist,
            index_monitoring_run,
            index_feature_metrics,
            build_monitoring_snapshot_json,
        )
        from src.storage.minio_client import upload_json, make_snapshot_object_name

        ti = context["ti"]
        ref_info   = ti.xcom_pull(task_ids="load_reference_run")
        data_info  = ti.xcom_pull(task_ids="collect_current_data")
        drift_info = ti.xcom_pull(task_ids="compute_drift")
        store_info = ti.xcom_pull(task_ids="store_postgres")

        monitoring_run_id = store_info["monitoring_run_id"]
        obs_start = datetime.fromisoformat(data_info["window_start"])
        obs_end   = datetime.fromisoformat(data_info["window_end"])

        # ── Elasticsearch ──────────────────────────────────────────────────
        try:
            ensure_indices_exist()
            index_monitoring_run(
                monitoring_run_id=monitoring_run_id,
                ref_run_id=ref_info["run_id"],
                observation_start=obs_start,
                observation_end=obs_end,
                drift_score_global=drift_info["global_score"],
                drift_detected=drift_info["drift_detected"],
                alert_level=drift_info["alert_level"],
                n_features_drifted=drift_info["n_features_drifted"],
                input_dataset_uri=data_info["snapshot_uri"],
                feature_metrics=drift_info["feature_metrics"],
            )
            n_ok = index_feature_metrics(
                monitoring_run_id=monitoring_run_id,
                ref_run_id=ref_info["run_id"],
                observation_start=obs_start,
                alert_level=drift_info["alert_level"],
                feature_metrics=drift_info["feature_metrics"],
            )
            log.info("ES : %d features indexées pour %s", n_ok, monitoring_run_id[:8])
        except Exception as exc:
            log.error("ES indexation échouée (non bloquant) : %s", exc)

        # ── Snapshot JSON complet dans MinIO ───────────────────────────────
        try:
            snapshot_json = build_monitoring_snapshot_json(
                monitoring_run_id=monitoring_run_id,
                ref_run_id=ref_info["run_id"],
                observation_start=obs_start,
                observation_end=obs_end,
                drift_score_global=drift_info["global_score"],
                drift_detected=drift_info["drift_detected"],
                alert_level=drift_info["alert_level"],
                n_features_drifted=drift_info["n_features_drifted"],
                input_dataset_uri=data_info["snapshot_uri"],
                feature_metrics=drift_info["feature_metrics"],
            )
            json_name = make_snapshot_object_name(obs_start, obs_end).replace(
                ".parquet", f"_drift_{monitoring_run_id[:8]}.json"
            )
            uri = upload_json(snapshot_json, bucket="monitoring-snapshots", object_name=json_name)
            log.info("Snapshot JSON : %s", uri)
        except Exception as exc:
            log.error("Upload snapshot JSON échoué (non bloquant) : %s", exc)

    index_elasticsearch = PythonOperator(
        task_id="index_elasticsearch",
        python_callable=_index_elasticsearch,
    )

    # ─────────────────────────────────────────────────────────
    # 6. EVALUATE ALERT
    # ─────────────────────────────────────────────────────────
    def _evaluate_alert(**context):
        ti = context["ti"]
        drift_info = ti.xcom_pull(task_ids="compute_drift")
        if drift_info["should_retrain"]:
            log.info("Retrain déclenché : %s", drift_info.get("retrain_reason", ""))
            return "trigger_retraining"
        return "log_alert_only"

    evaluate_alert = BranchPythonOperator(
        task_id="evaluate_alert",
        python_callable=_evaluate_alert,
    )

    def _log_alert_only(**context):
        ti = context["ti"]
        d = ti.xcom_pull(task_ids="compute_drift")
        log.info("ALERTE: %s | score=%.4f | drifted=%d",
                 d["alert_level"], d["global_score"], d["n_features_drifted"])

    log_alert_only = PythonOperator(task_id="log_alert_only", python_callable=_log_alert_only)

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        conf={"triggered_by": "drift_monitoring_v2"},
        wait_for_completion=False,
    )

    pipeline_done = PythonOperator(
        task_id="pipeline_done",
        python_callable=lambda **kw: log.info("Pipeline monitoring v2 terminé."),
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────
    # FLOW
    # ─────────────────────────────────────────────────────────
    (
        load_reference_run
        >> collect_current_data
        >> compute_drift
        >> store_postgres
        >> index_elasticsearch
        >> evaluate_alert
        >> [log_alert_only, trigger_retraining]
        >> pipeline_done
    )