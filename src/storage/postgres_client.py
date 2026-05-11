"""
Client Postgres pour le système MLOps Drift Monitoring.

RÔLE DE CE MODULE
─────────────────
Il encapsule TOUTES les opérations SQL sur les 4 tables applicatives.
Le reste du code ne fait jamais de SQL directement — il passe par ce client.
Cela a deux avantages :
  1. Si on change de base (ex: MySQL), on ne modifie qu'ici.
  2. Chaque fonction est testable indépendamment.

TABLES GÉRÉES
─────────────
  - training_runs            : historique des entraînements
  - reference_feature_stats  : stats de la distribution de référence
  - drift_monitoring_runs    : historique des checks de drift
  - drift_feature_metrics    : détail par feature par check

CONNEXION
─────────
On utilise psycopg2 avec un pool de connexions simple.
Pour Airflow, chaque tâche crée et ferme sa propre connexion.
"""

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor

from src.config.settings import PG_CFG, PostgresConfig

logger = logging.getLogger(__name__)


# ── Gestionnaire de connexion ─────────────────────────────────────────────────

@contextmanager
def get_connection(cfg: PostgresConfig = PG_CFG):
    """
    Context manager qui ouvre une connexion Postgres et la ferme proprement.

    Usage :
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()

    En cas d'exception, la transaction est rollbackée automatiquement.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=cfg.host,
            port=cfg.port,
            dbname=cfg.database,
            user=cfg.user,
            password=cfg.password,
            cursor_factory=RealDictCursor,
        )
        yield conn
        conn.commit()
    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error("Postgres error — rollback effectué : %s", exc)
        raise
    finally:
        if conn:
            conn.close()


def check_connection(cfg: PostgresConfig = PG_CFG) -> bool:
    """Vérifie que Postgres est accessible. Utile au démarrage des DAGs."""
    try:
        with get_connection(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("Connexion Postgres OK (%s:%s/%s)", cfg.host, cfg.port, cfg.database)
        return True
    except Exception as exc:
        logger.error("Postgres inaccessible : %s", exc)
        return False


# ── TABLE : training_runs ─────────────────────────────────────────────────────

def insert_training_run(
    run_id: str,
    model_name: str,
    dataset_uri: str,
    n_rows: int,
    val_rmse: Optional[float] = None,
    val_r2: Optional[float] = None,
    train_duration_sec: Optional[float] = None,
    dataset_version: Optional[str] = None,
    status: str = "running",
) -> str:
    """
    Crée un enregistrement pour un nouveau run d'entraînement.

    Appelé au DÉBUT du pipeline d'entraînement (status='running'),
    puis mis à jour à la fin via update_training_run_status().

    Args:
        run_id : identifiant du run, correspond au préfixe MinIO runs/<run_id>/

    Returns:
        run_id
    """
    sql = """
        INSERT INTO training_runs
            (run_id, model_name, dataset_uri, dataset_version,
             status, n_rows, val_rmse, val_r2, train_duration_sec)
        VALUES
            (%(run_id)s, %(model_name)s, %(dataset_uri)s, %(dataset_version)s,
             %(status)s, %(n_rows)s, %(val_rmse)s, %(val_r2)s, %(train_duration_sec)s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "run_id": run_id,
                "model_name": model_name,
                "dataset_uri": dataset_uri,
                "dataset_version": dataset_version,
                "status": status,
                "n_rows": n_rows,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "train_duration_sec": train_duration_sec,
            })

    logger.info("training_run créé : run_id=%s", run_id[:8])
    return run_id


def update_training_run_status(
    run_id: str,
    status: str,
    val_rmse: Optional[float] = None,
    val_r2: Optional[float] = None,
    train_duration_sec: Optional[float] = None,
) -> None:
    """
    Met à jour le statut d'un run après entraînement.

    status : 'success' | 'failed'
    """
    sql = """
        UPDATE training_runs
        SET
            status             = %(status)s,
            val_rmse           = COALESCE(%(val_rmse)s, val_rmse),
            val_r2             = COALESCE(%(val_r2)s, val_r2),
            train_duration_sec = COALESCE(%(train_duration_sec)s, train_duration_sec),
            updated_at         = NOW()
        WHERE run_id = %(run_id)s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "run_id": run_id,
                "status": status,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "train_duration_sec": train_duration_sec,
            })
    logger.info("training_run mis à jour : run_id=%s → status=%s", run_id[:8], status)


def get_latest_successful_run(model_name: str) -> Optional[Dict]:
    """
    Récupère le run le plus récent avec status='success'.

    Le monitoring a besoin de ces infos pour savoir quelle référence utiliser.
    Les artefacts sont accessibles dans MinIO sous runs/<run_id>/.

    Returns:
        Dict avec run_id, model_name, run_date, dataset_uri, n_rows, val_rmse, val_r2
        ou None si aucun run réussi.
    """
    sql = """
        SELECT run_id, model_name, run_date, dataset_uri,
               dataset_version, n_rows, val_rmse, val_r2, train_duration_sec
        FROM training_runs
        WHERE model_name = %(model_name)s
          AND status = 'success'
        ORDER BY run_date DESC
        LIMIT 1
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"model_name": model_name})
            row = cur.fetchone()

    if row:
        result = dict(row)
        logger.info(
            "Dernier run réussi : %s → run_id=%s",
            model_name, result["run_id"][:8],
        )
        return result

    logger.warning("Aucun run réussi trouvé pour '%s'", model_name)
    return None


# ── TABLE : reference_feature_stats ──────────────────────────────────────────

def insert_reference_stats(
    run_id: str,
    stats_df,  # pandas DataFrame issu de compute_feature_stats()
) -> int:
    """
    Persiste les stats de référence d'un run d'entraînement.

    Le DataFrame doit avoir pour index les noms de features et
    pour colonnes : mean, std, min, q25, median, q75, max.

    Returns:
        Nombre de lignes insérées (= nombre de features).
    """
    sql = """
        INSERT INTO reference_feature_stats
            (run_id, feature_name, mean_value, std_value,
             min_value, max_value, q25, median, q75, n_rows)
        VALUES
            (%(run_id)s, %(feature_name)s, %(mean_value)s, %(std_value)s,
             %(min_value)s, %(max_value)s, %(q25)s, %(median)s, %(q75)s, %(n_rows)s)
        ON CONFLICT (run_id, feature_name) DO UPDATE SET
            mean_value = EXCLUDED.mean_value,
            std_value  = EXCLUDED.std_value,
            min_value  = EXCLUDED.min_value,
            max_value  = EXCLUDED.max_value,
            q25        = EXCLUDED.q25,
            median     = EXCLUDED.median,
            q75        = EXCLUDED.q75
    """
    rows = []
    for feature_name, row in stats_df.iterrows():
        rows.append({
            "run_id": run_id,
            "feature_name": feature_name,
            "mean_value": float(row["mean"]),
            "std_value": float(row["std"]),
            "min_value": float(row["min"]),
            "max_value": float(row["max"]),
            "q25": float(row["q25"]),
            "median": float(row["median"]),
            "q75": float(row["q75"]),
            "n_rows": int(row.get("count", 0)) if "count" in row else None,
        })

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)

    logger.info(
        "Stats de référence insérées : %d features pour run_id=%s",
        len(rows), run_id[:8],
    )
    return len(rows)


def get_reference_stats(run_id: str) -> Dict[str, Dict]:
    """
    Charge les stats de référence associées à un run d'entraînement.

    Returns:
        Dict[feature_name → Dict[stat_name → valeur]]
        Ex: {"x1": {"mean_value": 0.01, "std_value": 0.99, ...}, ...}
    """
    sql = """
        SELECT feature_name, mean_value, std_value, min_value,
               max_value, q25, median, q75, n_rows
        FROM reference_feature_stats
        WHERE run_id = %(run_id)s
        ORDER BY feature_name
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"run_id": run_id})
            rows = cur.fetchall()

    if not rows:
        logger.warning("Aucune stat de référence pour run_id=%s", run_id[:8])
        return {}

    result = {row["feature_name"]: dict(row) for row in rows}
    logger.info(
        "Stats de référence chargées : %d features pour run_id=%s",
        len(result), run_id[:8],
    )
    return result


# ── TABLE : drift_monitoring_runs ─────────────────────────────────────────────

def insert_monitoring_run(
    ref_run_id: str,
    observation_start: datetime,
    observation_end: datetime,
    drift_score_global: float,
    drift_detected: bool,
    alert_level: str,
    n_features_drifted: int,
    input_dataset_uri: str = "",
) -> str:
    """
    Enregistre le résultat global d'un check de drift.

    Appelé UNE FOIS par exécution du DAG drift_monitoring,
    après avoir calculé tous les scores.

    Args:
        ref_run_id : run_id du training run utilisé comme référence
        alert_level: 'none' | 'warning' | 'critical'

    Returns:
        monitoring_run_id (UUID)
    """
    monitoring_run_id = str(uuid.uuid4())

    sql = """
        INSERT INTO drift_monitoring_runs
            (monitoring_run_id, ref_run_id, observation_start, observation_end,
             input_dataset_uri, drift_score_global, drift_detected,
             alert_level, n_features_in_drift)
        VALUES
            (%(monitoring_run_id)s, %(ref_run_id)s, %(observation_start)s,
             %(observation_end)s, %(input_dataset_uri)s, %(drift_score_global)s,
             %(drift_detected)s, %(alert_level)s, %(n_features_in_drift)s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "monitoring_run_id": monitoring_run_id,
                "ref_run_id": ref_run_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "input_dataset_uri": input_dataset_uri,
                "drift_score_global": drift_score_global,
                "drift_detected": drift_detected,
                "alert_level": alert_level,
                "n_features_in_drift": n_features_drifted,
            })

    logger.info(
        "monitoring_run créé : id=%s | alert=%s | drift=%s | score=%.4f",
        monitoring_run_id[:8], alert_level, drift_detected, drift_score_global,
    )
    return monitoring_run_id


# ── TABLE : drift_feature_metrics ─────────────────────────────────────────────

def insert_drift_feature_metrics(
    monitoring_run_id: str,
    feature_metrics: List[Dict],
) -> int:
    """
    Persiste le détail du drift par feature pour un monitoring_run.

    Colonnes v2 incluses :
        is_binary, chi2_stat, chi2_pvalue,
        wasserstein, wasserstein_norm, js_divergence

    feature_metrics : liste de dicts issus de drift_metrics.feature_results_to_rows()

    Returns:
        Nombre de lignes insérées.
    """
    sql = """
        INSERT INTO drift_feature_metrics
            (monitoring_run_id, feature_name,
             is_binary,
             psi,
             ks_stat, ks_pvalue,
             chi2_stat, chi2_pvalue,
             wasserstein, wasserstein_norm,
             js_divergence,
             mean_ref, mean_cur, mean_delta,
             std_ref, std_cur, std_delta,
             drift_flag, drift_reason)
        VALUES
            (%(monitoring_run_id)s, %(feature_name)s,
             %(is_binary)s,
             %(psi)s,
             %(ks_stat)s, %(ks_pvalue)s,
             %(chi2_stat)s, %(chi2_pvalue)s,
             %(wasserstein)s, %(wasserstein_norm)s,
             %(js_divergence)s,
             %(mean_ref)s, %(mean_cur)s, %(mean_delta)s,
             %(std_ref)s, %(std_cur)s, %(std_delta)s,
             %(drift_flag)s, %(drift_reason)s)
        ON CONFLICT (monitoring_run_id, feature_name) DO NOTHING
    """
    # Valeurs par défaut pour les colonnes optionnelles
    # (compatibilité si appelé avec d'anciens dicts sans les champs v2)
    defaults = {
        "is_binary": False,
        "chi2_stat": -1.0,
        "chi2_pvalue": -1.0,
        "wasserstein": None,
        "wasserstein_norm": None,
        "js_divergence": -1.0,
    }
    rows = [
        {"monitoring_run_id": monitoring_run_id, **defaults, **m}
        for m in feature_metrics
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)

    logger.info(
        "drift_feature_metrics insérées : %d features pour monitoring_run=%s",
        len(rows), monitoring_run_id[:8],
    )
    return len(rows)


def get_recent_monitoring_runs(
    ref_run_id: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Récupère les derniers runs de monitoring pour un run de référence donné.

    Utilisé par le module alerting pour compter les alertes consécutives.

    Returns:
        Liste de dicts triée du plus récent au plus ancien.
    """
    sql = """
        SELECT monitoring_run_id, ref_run_id, observation_start,
               observation_end, drift_score_global, drift_detected,
               alert_level, n_features_in_drift, created_at
        FROM drift_monitoring_runs
        WHERE ref_run_id = %(ref_run_id)s
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"ref_run_id": ref_run_id, "limit": limit})
            rows = cur.fetchall()

    return [dict(r) for r in rows]


def count_consecutive_alerts(ref_run_id: str) -> int:
    """
    Compte le nombre d'alertes consécutives les plus récentes pour un run de référence.

    Logique : on remonte les runs du plus récent au plus ancien.
    On s'arrête dès qu'on rencontre un run sans alerte.

    Ex : [alerte, alerte, alerte, pas d'alerte, alerte] → retourne 3

    Cette valeur est utilisée par alerting.py pour décider
    si on doit déclencher un ré-entraînement.
    """
    runs = get_recent_monitoring_runs(ref_run_id, limit=20)

    consecutive = 0
    for run in runs:
        if run["drift_detected"]:
            consecutive += 1
        else:
            break

    logger.info(
        "Alertes consécutives pour ref_run_id=%s : %d",
        ref_run_id[:8], consecutive,
    )
    return consecutive