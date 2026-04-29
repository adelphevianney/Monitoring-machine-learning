"""
Schéma de la BD — migrations idempotentes lancées au début du service.
utilsation de psycopg2
"""
from __future__ import annotations

from venv import logger

import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.config.settings import PG_CFG


# --------------------------------------------------------------------------- #
# DDL                                                                         #
# --------------------------------------------------------------------------- #

_SCHEMA_SQL = """
-- =============================================================================
-- TABLES APPLICATIVES — MLOps Drift Monitoring
-- =============================================================================
-- Ces tables sont SÉPARÉES des tables internes MLflow (qui gèrent elles-mêmes
-- leur propre schéma dans la même base Postgres).
--
-- Pourquoi deux couches de tables ?
--   - MLflow stocke les runs, params, métriques d'entraînement
--   - Ces tables stockent l'état opérationnel du monitoring :
--     qui est en production, quels drifts ont été détectés, etc.
--
-- Le lien entre les deux : mlflow_run_id (UUID du run MLflow)
-- =============================================================================

DROP TABLE IF EXISTS drift_feature_metrics CASCADE;
DROP TABLE IF EXISTS drift_monitoring_runs CASCADE;
DROP TABLE IF EXISTS reference_feature_stats CASCADE;
DROP TABLE IF EXISTS training_runs CASCADE;


-- =============================================================================
-- TABLE 1 : training_runs
-- =============================================================================
CREATE TABLE training_runs (
    id                 SERIAL PRIMARY KEY,
    run_id             VARCHAR(64) UNIQUE NOT NULL,
    mlflow_run_id      VARCHAR(64) UNIQUE,
    model_name         VARCHAR(128) NOT NULL,
    model_version      VARCHAR(32),
    model_stage        VARCHAR(32) DEFAULT 'None',
    run_date           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_uri        TEXT,
    dataset_version    VARCHAR(64),
    status             VARCHAR(32) DEFAULT 'running',
    n_rows             INTEGER,
    val_rmse           FLOAT,
    val_r2             FLOAT,
    train_duration_sec FLOAT,
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_training_runs_stage   ON training_runs(model_stage);
CREATE INDEX idx_training_runs_rundate ON training_runs(run_date DESC);
CREATE INDEX idx_training_runs_mlflow  ON training_runs(mlflow_run_id);

COMMENT ON TABLE training_runs IS
    'Historique de tous les runs d entraînement. Lien vers MLflow via mlflow_run_id.';


-- =============================================================================
-- TABLE 2 : reference_feature_stats
-- =============================================================================
CREATE TABLE reference_feature_stats (
    id             SERIAL PRIMARY KEY,
    mlflow_run_id  VARCHAR(64) NOT NULL REFERENCES training_runs(mlflow_run_id),
    feature_name   VARCHAR(64) NOT NULL,
    mean_value     FLOAT,
    std_value      FLOAT,
    min_value      FLOAT,
    max_value      FLOAT,
    q25            FLOAT,
    median         FLOAT,
    q75            FLOAT,
    n_rows         INTEGER,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (mlflow_run_id, feature_name)
);

CREATE INDEX idx_ref_stats_runid   ON reference_feature_stats(mlflow_run_id);
CREATE INDEX idx_ref_stats_feature ON reference_feature_stats(feature_name);


-- =============================================================================
-- TABLE 3 : drift_monitoring_runs
-- =============================================================================
CREATE TABLE drift_monitoring_runs (
    id                        SERIAL PRIMARY KEY,
    monitoring_run_id         VARCHAR(64) UNIQUE NOT NULL,
    model_version             VARCHAR(32),
    mlflow_run_id_ref         VARCHAR(64),
    observation_start         TIMESTAMP WITH TIME ZONE,
    observation_end           TIMESTAMP WITH TIME ZONE,
    input_dataset_uri         TEXT,
    n_rows_observed           INTEGER,
    drift_score_global        FLOAT,
    n_features_in_drift       INTEGER DEFAULT 0,
    drift_detected            BOOLEAN DEFAULT FALSE,
    alert_level               VARCHAR(32) DEFAULT 'none',
    consecutive_drift_windows INTEGER DEFAULT 0,
    retrain_triggered         BOOLEAN DEFAULT FALSE,
    created_at                TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_monitoring_runs_alertlevel ON drift_monitoring_runs(alert_level);
CREATE INDEX idx_monitoring_runs_created    ON drift_monitoring_runs(created_at DESC);
CREATE INDEX idx_monitoring_runs_drift      ON drift_monitoring_runs(drift_detected, created_at DESC);


-- =============================================================================
-- TABLE 4 : drift_feature_metrics
-- =============================================================================
CREATE TABLE drift_feature_metrics (
    id                SERIAL PRIMARY KEY,
    monitoring_run_id VARCHAR(64) NOT NULL REFERENCES drift_monitoring_runs(monitoring_run_id),
    feature_name      VARCHAR(64) NOT NULL,
    psi               FLOAT,
    ks_stat           FLOAT,
    ks_pvalue         FLOAT,
    mean_ref          FLOAT,
    mean_cur          FLOAT,
    mean_delta        FLOAT,
    std_ref           FLOAT,
    std_cur           FLOAT,
    std_delta         FLOAT,
    drift_flag        BOOLEAN DEFAULT FALSE,
    drift_reason      TEXT,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (monitoring_run_id, feature_name)
);

CREATE INDEX idx_drift_metrics_runid   ON drift_feature_metrics(monitoring_run_id);
CREATE INDEX idx_drift_metrics_feature ON drift_feature_metrics(feature_name);
CREATE INDEX idx_drift_metrics_flag    ON drift_feature_metrics(drift_flag, feature_name);


-- =============================================================================
-- VUE : dernier état du monitoring
-- =============================================================================
CREATE OR REPLACE VIEW v_latest_drift_status AS
SELECT
    mr.monitoring_run_id,
    mr.created_at,
    mr.alert_level,
    mr.drift_score_global,
    mr.n_features_in_drift,
    mr.consecutive_drift_windows,
    mr.retrain_triggered,
    STRING_AGG(
        CASE WHEN fm.drift_flag
            THEN fm.feature_name || '(PSI=' || ROUND(fm.psi::numeric, 3) || ')'
        END,
        ', ' ORDER BY fm.psi DESC
    ) AS drifted_features
FROM drift_monitoring_runs mr
LEFT JOIN drift_feature_metrics fm ON fm.monitoring_run_id = mr.monitoring_run_id
WHERE mr.created_at = (SELECT MAX(created_at) FROM drift_monitoring_runs)
GROUP BY mr.monitoring_run_id, mr.created_at, mr.alert_level,
         mr.drift_score_global, mr.n_features_in_drift,
         mr.consecutive_drift_windows, mr.retrain_triggered;


-- =============================================================================
-- VUE : historique des 30 dernières alertes
-- =============================================================================
CREATE OR REPLACE VIEW v_alert_history AS
SELECT
    monitoring_run_id,
    created_at,
    alert_level,
    drift_score_global,
    n_features_in_drift,
    consecutive_drift_windows,
    retrain_triggered
FROM drift_monitoring_runs
ORDER BY created_at DESC
LIMIT 30;
"""


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def get_connection() -> PgConnection:
    """retourne une connection psycopg2 brut (la fermeture est la responsabilité du service qui l'appelle)."""
    return psycopg2.connect(PG_CFG.db.dsn)


def run_migrations() -> None:
    """applique le schéma, c'est safe de l'appeler à chaque démarrage (à cause de l'idempotence)."""
    logger.info("Running database migrations")
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
        logger.info("Database migrations completed successfully")
    except Exception as exc:
        logger.error("Migration failed")
        raise
    finally:
        conn.close()