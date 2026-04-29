"""
DAG 2 : drift_monitoring_pipeline
───────────────────────────────────
Surveille en continu si les données reçues par le modèle
ressemblent encore à celles sur lesquelles il a été entraîné.

QUAND S'EXÉCUTE-T-IL ?
  Toutes les heures par défaut.
  Plus fréquent = détection plus rapide, mais plus de charge sur Postgres/MinIO.
  Adapter selon le volume de données en production.

FENÊTRE D'OBSERVATION :
  À chaque run, on observe les données de la dernière heure.
  Ex : run lancé à 15h00 → observe les données reçues entre 14h00 et 15h00.

CE QU'IL FAIT (dans l'ordre) :
  1. load_production_model   — récupère les infos du modèle actuellement en prod
                               (version, mlflow_run_id, stats de référence)
  2. collect_current_data    — collecte les données de la fenêtre courante
                               (simulation : génère des données avec drift partiel)
  3. compute_drift_metrics   — calcule PSI, KS, deltas pour chaque feature
  4. store_monitoring_results — persiste les résultats dans Postgres
  5. evaluate_alert          — applique les règles de décision, détermine l'alerte
  6. trigger_retraining      — déclenche le DAG training_pipeline si nécessaire
                               (seulement si alerte critique persistante)

SIMULATION DU FLUX DE DONNÉES EN PRODUCTION :
  En production, les "nouvelles données" viendraient d'une table Postgres
  ou d'un bucket MinIO alimenté par le service d'inférence.
  Ici on simule ce flux en générant des données avec un drift progressif
  dont l'intensité augmente à chaque heure pour voir le système réagir.

VARIABLES AIRFLOW REQUISES :
  MLFLOW_TRACKING_URI         : http://mlflow:5000
  MODEL_NAME                  : drift_regressor
  DRIFT_SIMULATION_FACTOR     : 0.0 à 1.0 (0=référence, 1=drift complet)
                                Incrémenter manuellement pour tester.
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
    description="Monitore le drift des données en entrée du modèle en production",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "monitoring", "drift"],
    max_active_runs=1,
    doc_md=__doc__,
) as dag:

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 1 — Charger les infos du modèle en production
    # ─────────────────────────────────────────────────────────────────────────
    def _load_production_model(**context) -> dict:
        """
        Récupère depuis Postgres les infos du modèle actuellement en production :
          - model_version  : version dans le registry MLflow
          - mlflow_run_id  : pour retrouver les stats de référence
          - métriques      : val_rmse, val_r2 (pour contextualiser les alertes)

        Si aucun modèle n'est en production, la tâche est skippée
        (AirflowSkipException) → toutes les tâches en aval sont aussi skippées.
        C'est le comportement correct : pas de modèle = rien à monitorer.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.storage.postgres_client import get_production_model_info

        model_name = Variable.get("MODEL_NAME", default_var="drift_regressor")
        model_info = get_production_model_info(model_name)

        if not model_info:
            log.warning("Aucun modèle en production pour '%s'. Pipeline skippé.", model_name)
            raise AirflowSkipException(
                f"Aucun modèle en production pour '{model_name}'. "
                "Lancer d'abord le DAG training_pipeline."
            )

        log.info(
            "Modèle en production : %s v%s (mlflow_run=%s)",
            model_name, model_info["model_version"], model_info["mlflow_run_id"][:8]
        )
        return model_info

    load_production_model = PythonOperator(
        task_id="load_production_model",
        python_callable=_load_production_model,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 2 — Collecter les données courantes
    # ─────────────────────────────────────────────────────────────────────────
    def _collect_current_data(**context) -> dict:
        """
        Collecte les données reçues par le modèle sur la fenêtre courante.

        EN PRODUCTION :
          On lirait depuis une table Postgres ou un bucket MinIO alimenté
          par le service d'inférence qui logue toutes les requêtes.
          Ex :
            SELECT * FROM inference_logs
            WHERE received_at BETWEEN %(start)s AND %(end)s

        ICI (simulation) :
          On génère des données avec un drift partiel contrôlé par la variable
          DRIFT_SIMULATION_FACTOR. Cela permet de tester le pipeline sans
          infrastructure d'inférence.

          drift_factor = 0.0 → données identiques à la référence (pas d'alerte)
          drift_factor = 0.5 → drift modéré (warning possible)
          drift_factor = 1.0 → drift complet (alerte critique attendue)
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.data_generation.generator import generate_partial_drift
        from src.storage.minio_client import (
            make_snapshot_object_name,
            upload_dataframe,
        )

        exec_date = context["execution_date"]
        window_start = exec_date - timedelta(hours=1)
        window_end = exec_date

        drift_factor = float(
            Variable.get("DRIFT_SIMULATION_FACTOR", default_var="0.0")
        )
        n_rows = int(Variable.get("N_ROWS_MONITORING", default_var="1000"))

        log.info(
            "Collecte fenêtre [%s → %s] | drift_factor=%.2f | n=%d",
            window_start.strftime("%H:%M"), window_end.strftime("%H:%M"),
            drift_factor, n_rows
        )

        # Seed basée sur l'heure pour avoir des données différentes à chaque run
        seed = int(exec_date.timestamp()) % 100_000
        df_current = generate_partial_drift(
            drift_factor=drift_factor, n_rows=n_rows, seed=seed
        )

        # Sauvegarder le snapshot dans MinIO pour traçabilité et replay
        object_name = make_snapshot_object_name(window_start, window_end)
        snapshot_uri = upload_dataframe(
            df_current, bucket="monitoring-snapshots", object_name=object_name
        )
        log.info("Snapshot sauvegardé : %s", snapshot_uri)

        return {
            "snapshot_uri": snapshot_uri,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "n_rows": n_rows,
            "drift_factor": drift_factor,
        }

    collect_current_data = PythonOperator(
        task_id="collect_current_data",
        python_callable=_collect_current_data,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 3 — Calculer les métriques de drift
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_drift_metrics(**context) -> dict:
        """
        Compare la distribution des données courantes aux stats de référence
        du modèle en production.

        MÉTRIQUES CALCULÉES PAR FEATURE :
          - PSI   : Population Stability Index (0=stable, >0.2=drift significatif)
          - KS    : test de Kolmogorov-Smirnov (p-value)
          - Δmean : différence de moyenne (absolue et relative)
          - Δstd  : différence d'écart-type

        Pour reconstruire la distribution de référence, on a deux options :
          Option A : charger le dataset de référence complet depuis MinIO
                     → précis mais coûteux (chargement d'un gros fichier)
          Option B : utiliser les stats agrégées depuis Postgres
                     → rapide mais moins précis pour le PSI (pas les données brutes)

        On utilise ici l'option A (dataset complet) pour avoir un PSI précis.
        En production à fort volume, switcher vers l'option B + PSI sur quantiles.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.monitoring.drift_metrics import compute_drift_report, compute_global_score
        from src.storage.minio_client import (
            download_dataframe_from_uri,
        )
        from src.storage.postgres_client import get_reference_stats

        ti = context["ti"]
        model_info = ti.xcom_pull(task_ids="load_production_model")
        data_info = ti.xcom_pull(task_ids="collect_current_data")

        mlflow_run_id = model_info["mlflow_run_id"]
        dataset_uri = model_info["dataset_uri"]
        snapshot_uri = data_info["snapshot_uri"]

        log.info("Chargement dataset de référence : %s", dataset_uri)
        df_reference = download_dataframe_from_uri(dataset_uri)

        log.info("Chargement snapshot courant : %s", snapshot_uri)
        df_current = download_dataframe_from_uri(snapshot_uri)

        features = [c for c in df_reference.columns if c != "y"]
        log.info("Calcul du drift sur %d features : %s", len(features), features)

        # Calcul du drift (PSI + KS + deltas) pour chaque feature
        drift_report = compute_drift_report(df_reference, df_current, features)
        global_score, alert_level, drift_detected = compute_global_score(drift_report)

        # Sérialiser pour XCom (dataclasses → dicts)
        feature_metrics = {}
        for fname, result in drift_report.items():
            feature_metrics[fname] = {
                "psi": result.psi,
                "ks_stat": result.ks_stat,
                "ks_pvalue": result.ks_pvalue,
                "mean_ref": result.mean_ref,
                "mean_cur": result.mean_cur,
                "mean_delta": result.mean_delta,
                "std_ref": result.std_ref,
                "std_cur": result.std_cur,
                "std_delta": result.std_delta,
                "drift_flag": result.drift_flag,
                "drift_reason": result.drift_reason,
            }

        n_drifted = sum(1 for r in drift_report.values() if r.drift_flag)
        log.info(
            "Drift calculé : score_global=%.4f | alert=%s | %d/%d features en drift",
            global_score, alert_level, n_drifted, len(features)
        )

        return {
            "feature_metrics": feature_metrics,
            "global_score": global_score,
            "alert_level": alert_level,
            "drift_detected": drift_detected,
            "n_features_drifted": n_drifted,
        }

    compute_drift_metrics = PythonOperator(
        task_id="compute_drift_metrics",
        python_callable=_compute_drift_metrics,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 4 — Stocker les résultats dans Postgres
    # ─────────────────────────────────────────────────────────────────────────
    def _store_monitoring_results(**context) -> dict:
        """
        Persiste dans Postgres :
          1. Un enregistrement dans drift_monitoring_runs (verdict global)
          2. Un enregistrement par feature dans drift_feature_metrics

        Ces données alimentent :
          - Le dashboard de monitoring
          - La logique de comptage des alertes consécutives
          - Les rapports d'audit
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from datetime import datetime
        from src.monitoring.alerting import build_monitoring_verdict, verdict_to_postgres_rows
        from src.storage.postgres_client import (
            insert_drift_feature_metrics,
            insert_monitoring_run,
        )

        ti = context["ti"]
        model_info = ti.xcom_pull(task_ids="load_production_model")
        data_info = ti.xcom_pull(task_ids="collect_current_data")
        drift_info = ti.xcom_pull(task_ids="compute_drift_metrics")

        model_version = model_info["model_version"]
        feature_metrics = drift_info["feature_metrics"]

        # Construire le verdict complet via le module alerting
        verdict = build_monitoring_verdict(
            drift_results=feature_metrics,
            consecutive_alerts=0,  # sera affiné dans evaluate_alert
        )

        # Insérer le run global
        monitoring_run_id = insert_monitoring_run(
            model_version=model_version,
            observation_start=datetime.fromisoformat(data_info["window_start"]),
            observation_end=datetime.fromisoformat(data_info["window_end"]),
            drift_score_global=drift_info["global_score"],
            drift_detected=drift_info["drift_detected"],
            alert_level=drift_info["alert_level"],
            n_features_drifted=drift_info["n_features_drifted"],
            input_dataset_uri=data_info["snapshot_uri"],
        )

        # Insérer le détail par feature
        rows = verdict_to_postgres_rows(verdict)
        insert_drift_feature_metrics(monitoring_run_id, rows)

        log.info("Résultats de monitoring persistés : monitoring_run_id=%s", monitoring_run_id[:8])
        return {"monitoring_run_id": monitoring_run_id}

    store_monitoring_results = PythonOperator(
        task_id="store_monitoring_results",
        python_callable=_store_monitoring_results,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 5 — Évaluer l'alerte et décider du ré-entraînement
    # ─────────────────────────────────────────────────────────────────────────
    def _evaluate_alert(**context) -> str:
        """
        Branchement conditionnel : décide quelle tâche exécuter ensuite.

        Logique :
          - Compte les alertes critiques consécutives dans Postgres
          - Si >= seuil (3 par défaut) → branche vers 'trigger_retraining'
          - Sinon                       → branche vers 'log_alert_only'

        Returns:
          task_id de la prochaine tâche à exécuter (BranchPythonOperator)
        """
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        from src.config.settings import MONITOR_CFG
        from src.storage.postgres_client import count_consecutive_alerts

        ti = context["ti"]
        model_info = ti.xcom_pull(task_ids="load_production_model")
        drift_info = ti.xcom_pull(task_ids="compute_drift_metrics")

        alert_level = drift_info["alert_level"]
        model_version = model_info["model_version"]

        if alert_level != "critical":
            log.info("Niveau d'alerte : %s — pas de ré-entraînement.", alert_level)
            return "log_alert_only"

        # Compter les alertes consécutives
        n_consecutive = count_consecutive_alerts(model_version)
        threshold = MONITOR_CFG.consecutive_windows_for_retrain

        log.warning(
            "Alerte CRITIQUE : %d/%d fenêtres consécutives",
            n_consecutive, threshold
        )

        if n_consecutive >= threshold:
            log.critical(
                "Seuil de ré-entraînement atteint (%d fenêtres). "
                "Déclenchement du DAG training_pipeline.",
                n_consecutive
            )
            return "trigger_retraining"

        return "log_alert_only"

    evaluate_alert = BranchPythonOperator(
        task_id="evaluate_alert",
        python_callable=_evaluate_alert,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 6a — Logger l'alerte sans ré-entraînement
    # ─────────────────────────────────────────────────────────────────────────
    def _log_alert_only(**context) -> None:
        """
        Logue le résumé du monitoring sans déclencher de ré-entraînement.
        Point d'extension pour envoyer une notification (Slack, email, PagerDuty).
        """
        ti = context["ti"]
        drift_info = ti.xcom_pull(task_ids="compute_drift_metrics")
        data_info = ti.xcom_pull(task_ids="collect_current_data")

        alert_level = drift_info["alert_level"]
        global_score = drift_info["global_score"]
        n_drifted = drift_info["n_features_drifted"]

        icon = {"none": "✅", "warning": "⚠️", "critical": "🔴"}.get(alert_level, "❓")

        log.info("=" * 55)
        log.info("MONITORING — %s %s", icon, alert_level.upper())
        log.info("  Fenêtre   : %s → %s", data_info["window_start"], data_info["window_end"])
        log.info("  PSI global: %.4f", global_score)
        log.info("  Features en drift : %d", n_drifted)
        log.info("=" * 55)

        # TODO : ajouter ici l'envoi d'une notification Slack/email
        # si alert_level in ("warning", "critical")

    log_alert_only = PythonOperator(
        task_id="log_alert_only",
        python_callable=_log_alert_only,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 6b — Déclencher le ré-entraînement
    # ─────────────────────────────────────────────────────────────────────────
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        # conf passé au DAG déclenché (accessible via context["dag_run"].conf)
        conf={
            "triggered_by": "drift_monitoring",
            "reason": "consecutive_critical_alerts",
        },
        wait_for_completion=False,  # ne pas bloquer ce DAG en attendant le training
        reset_dag_run=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TÂCHE 7 — Fin du pipeline (convergence des branches)
    # ─────────────────────────────────────────────────────────────────────────
    def _pipeline_done(**context) -> None:
        """Tâche de convergence — s'exécute quelle que soit la branche prise."""
        log.info("Drift monitoring pipeline terminé pour la fenêtre courante.")

    pipeline_done = PythonOperator(
        task_id="pipeline_done",
        python_callable=_pipeline_done,
        # ONE_SUCCESS : s'exécute si au moins une branche amont a réussi
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DÉPENDANCES — graphe d'exécution
    # ─────────────────────────────────────────────────────────────────────────
    #
    #  load_production_model
    #          ↓
    #  collect_current_data
    #          ↓
    #  compute_drift_metrics
    #          ↓
    #  store_monitoring_results
    #          ↓
    #  evaluate_alert
    #       ↙        ↘
    # log_alert   trigger_retraining
    #       ↘        ↙
    #       pipeline_done
    #
    (
        load_production_model
        >> collect_current_data
        >> compute_drift_metrics
        >> store_monitoring_results
        >> evaluate_alert
        >> [log_alert_only, trigger_retraining]
        >> pipeline_done
    )
