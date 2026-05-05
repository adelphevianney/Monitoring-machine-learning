"""
DAG : drift_monitoring_pipeline
────────────────────────────────
Monitoring du drift basé sur MinIO + Postgres.
Pas de MLflow : le modèle de référence est résolu depuis Postgres (training_runs).

FLUX DES DONNÉES
────────────────
  1. load_reference_run   — charge le dernier run réussi depuis Postgres
                            (run_id, dataset_uri, features)
  2. collect_current_data — génère un snapshot de données courantes,
                            l'upload dans MinIO (monitoring-snapshots)
  3. compute_drift        — calcule les métriques de drift (PSI, KS)
                            via drift_metrics.compute_drift_report()
  4. store_results        — persiste dans Postgres :
                              • drift_monitoring_runs (résultat global)
                              • drift_feature_metrics (détail par feature)
  5. evaluate_alert       — BranchOperator : 'log_alert_only' ou 'trigger_retraining'
  6a. log_alert_only      — log du niveau d'alerte
  6b. trigger_retraining  — déclenche le DAG training_pipeline
  7. pipeline_done        — fin

VARIABLES AIRFLOW REQUISES (Admin → Variables)
  MODEL_NAME                  : drift_regressor
  DRIFT_SIMULATION_FACTOR     : 0.0  (0 = stable, 1 = drift total)
  N_ROWS_MONITORING           : 1000

CORRECTION PAR RAPPORT À LA VERSION PRÉCÉDENTE
───────────────────────────────────────────────
- Plus de métadonnées MLflow (model_version, metadata.json)
- La référence est résolue via postgres_client.get_latest_successful_run()
- insert_monitoring_run() reçoit ref_run_id (pas model_version)
- count_consecutive_alerts() reçoit ref_run_id (pas model_version)
- insert_drift_feature_metrics() reçoit une liste de dicts
  (produite par drift_metrics.feature_results_to_rows)
- alerting.build_monitoring_verdict() consomme des FeatureDriftResult
  directement (plus de conversion Dict intermédiaire)
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
    description="Monitoring du drift (MinIO + Postgres, sans MLflow)",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "monitoring", "drift"],
    max_active_runs=1,
) as dag:

    # ─────────────────────────────────────────────────────────
    # 1. LOAD REFERENCE RUN FROM POSTGRES
    # ─────────────────────────────────────────────────────────
    def _load_reference_run(**context):
        """
        Charge le dernier run d'entraînement réussi depuis Postgres.

        Retourne via XCom :
          - run_id       : identifiant du run (préfixe MinIO)
          - dataset_uri  : URI MinIO du dataset de référence
          - features     : liste des features utilisées à l'entraînement
          - model_name   : nom du modèle
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.postgres_client import get_latest_successful_run

        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")
        run_info = get_latest_successful_run(model_name)

        if not run_info:
            log.warning("Aucun run réussi pour '%s'", model_name)
            raise AirflowSkipException(f"Pas de run réussi pour '{model_name}'")

        # Charger les tags du run pour récupérer les features
        from src.storage.minio_client import download_json
        from src.config.settings import MINIO_CFG

        run_id = run_info["run_id"]
        try:
            tags = download_json(
                bucket=MINIO_CFG.bucket,
                object_name=f"runs/{run_id}/tags.json",
            )
            features = tags.get("features", [])
        except Exception as exc:
            log.warning("Impossible de charger tags.json pour run_id=%s : %s", run_id, exc)
            features = []

        log.info(
            "Run de référence : run_id=%s | dataset=%s | %d features",
            run_id, run_info["dataset_uri"], len(features),
        )

        return {
            "run_id": run_id,
            "dataset_uri": run_info["dataset_uri"],
            "features": features,
            "model_name": model_name,
        }

    load_reference_run = PythonOperator(
        task_id="load_reference_run",
        python_callable=_load_reference_run,
    )

    # ─────────────────────────────────────────────────────────
    # 2. COLLECT CURRENT DATA
    # ─────────────────────────────────────────────────────────
    def _collect_current_data(**context):
        """
        Génère un snapshot de données courantes et l'upload dans MinIO.

        Retourne via XCom :
          - snapshot_uri : URI MinIO du snapshot
          - window_start : début de la fenêtre d'observation (ISO)
          - window_end   : fin de la fenêtre d'observation (ISO)
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import generate_partial_drift
        from src.storage.minio_client import upload_dataframe, make_snapshot_object_name

        exec_date = context["execution_date"]
        window_start = exec_date - timedelta(hours=1)
        window_end = exec_date

        drift_factor = float(Variable.get("DRIFT_SIMULATION_FACTOR", default_var=0.0))
        n_rows = int(Variable.get("N_ROWS_MONITORING", default_var=1000))
        seed = int(exec_date.timestamp()) % 100_000

        df_current = generate_partial_drift(
            drift_factor=drift_factor,
            n_rows=n_rows,
            seed=seed,
        )

        object_name = make_snapshot_object_name(window_start, window_end)
        snapshot_uri = upload_dataframe(
            df_current,
            bucket="monitoring-snapshots",
            object_name=object_name,
        )

        log.info(
            "Snapshot uploadé : %s (%d lignes, drift_factor=%.2f)",
            snapshot_uri, n_rows, drift_factor,
        )
        return {
            "snapshot_uri": snapshot_uri,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
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
        Calcule les métriques de drift entre référence et données courantes.

        Utilise drift_metrics.compute_drift_report() qui retourne
        Dict[str, FeatureDriftResult].

        Retourne via XCom un dict sérialisable (pas de dataclasses) :
          - feature_metrics  : {fname: {psi, ks_stat, ...}}
          - global_score     : float
          - alert_level      : 'none' | 'warning' | 'critical'
          - drift_detected   : bool
          - n_features_drifted: int
          - should_retrain   : bool
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.monitoring.drift_metrics import (
            compute_drift_report,
            feature_results_to_rows,
        )
        from src.monitoring.alerting import build_monitoring_verdict
        from src.storage.minio_client import download_dataframe_from_uri
        from src.storage.postgres_client import count_consecutive_alerts

        ti = context["ti"]
        ref_info = ti.xcom_pull(task_ids="load_reference_run")
        data_info = ti.xcom_pull(task_ids="collect_current_data")

        df_reference = download_dataframe_from_uri(ref_info["dataset_uri"])
        df_current = download_dataframe_from_uri(data_info["snapshot_uri"])

        features = ref_info.get("features") or [
            c for c in df_reference.columns if c != "y"
        ]

        # Calcul du drift — retourne Dict[str, FeatureDriftResult]
        feature_results = compute_drift_report(df_reference, df_current, features)

        # Compter les alertes consécutives pour la décision de ré-entraînement
        n_consecutive = count_consecutive_alerts(ref_info["run_id"])

        # Verdict global (alerte + décision retrain)
        verdict = build_monitoring_verdict(feature_results, consecutive_alerts=n_consecutive)

        log.info(
            "Drift calculé : alert=%s | score=%.4f | %d features driftées",
            verdict.alert_level, verdict.drift_score_global, verdict.n_features_drifted,
        )

        # Sérialisation pour XCom (pas de dataclasses)
        feature_metrics_rows = feature_results_to_rows(feature_results)

        return {
            "feature_metrics": feature_metrics_rows,
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
    # 4. STORE RESULTS IN POSTGRES
    # ─────────────────────────────────────────────────────────
    def _store_monitoring_results(**context):
        """
        Persiste dans Postgres :
          - drift_monitoring_runs  : résultat global du run
          - drift_feature_metrics  : détail par feature

        Retourne via XCom :
          - monitoring_run_id : UUID du run de monitoring
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.postgres_client import (
            insert_monitoring_run,
            insert_drift_feature_metrics,
        )

        ti = context["ti"]
        ref_info = ti.xcom_pull(task_ids="load_reference_run")
        data_info = ti.xcom_pull(task_ids="collect_current_data")
        drift_info = ti.xcom_pull(task_ids="compute_drift")

        # insert_monitoring_run attend ref_run_id (pas model_version)
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

        # insert_drift_feature_metrics attend une List[Dict]
        # (déjà sérialisée par feature_results_to_rows dans la tâche précédente)
        insert_drift_feature_metrics(
            monitoring_run_id,
            drift_info["feature_metrics"],
        )

        log.info(
            "Résultats persistés : monitoring_run_id=%s | alert=%s",
            monitoring_run_id[:8], drift_info["alert_level"],
        )
        return {"monitoring_run_id": monitoring_run_id}

    store_monitoring_results = PythonOperator(
        task_id="store_monitoring_results",
        python_callable=_store_monitoring_results,
    )

    # ─────────────────────────────────────────────────────────
    # 5. EVALUATE ALERT (branch)
    # ─────────────────────────────────────────────────────────
    def _evaluate_alert(**context):
        """
        Décide du chemin à suivre selon le verdict de drift.

        should_retrain=True → 'trigger_retraining'
        sinon               → 'log_alert_only'
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        ti = context["ti"]
        drift_info = ti.xcom_pull(task_ids="compute_drift")

        if drift_info["should_retrain"]:
            log.info(
                "Ré-entraînement déclenché : %s",
                drift_info.get("retrain_reason", ""),
            )
            return "trigger_retraining"

        return "log_alert_only"

    evaluate_alert = BranchPythonOperator(
        task_id="evaluate_alert",
        python_callable=_evaluate_alert,
    )

    # ─────────────────────────────────────────────────────────
    # 6a. LOG ONLY
    # ─────────────────────────────────────────────────────────
    def _log_alert_only(**context):
        ti = context["ti"]
        drift_info = ti.xcom_pull(task_ids="compute_drift")
        log.info(
            "ALERTE: %s | score=%.4f | drifted=%d | retrain=non",
            drift_info["alert_level"],
            drift_info["global_score"],
            drift_info["n_features_drifted"],
        )

    log_alert_only = PythonOperator(
        task_id="log_alert_only",
        python_callable=_log_alert_only,
    )

    # ─────────────────────────────────────────────────────────
    # 6b. TRIGGER RETRAINING
    # ─────────────────────────────────────────────────────────
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        conf={"triggered_by": "drift_monitoring"},
        wait_for_completion=False,
    )

    # ─────────────────────────────────────────────────────────
    # 7. END
    # ─────────────────────────────────────────────────────────
    def _done(**context):
        log.info("Pipeline de monitoring terminé.")

    pipeline_done = PythonOperator(
        task_id="pipeline_done",
        python_callable=_done,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────
    # FLOW
    # ─────────────────────────────────────────────────────────
    (
        load_reference_run
        >> collect_current_data
        >> compute_drift
        >> store_monitoring_results
        >> evaluate_alert
        >> [log_alert_only, trigger_retraining]
        >> pipeline_done
    )