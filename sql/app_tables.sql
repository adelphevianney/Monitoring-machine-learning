-- =============================================================================
-- TABLES APPLICATIVES - MLOps Drift Monitoring
-- Base cible : mlops
--
-- CE FICHIER EST EXECUTE PAR init-multiple-dbs.sh via :
--   psql --username postgres --dbname mlops --file 2_app_tables.sql
--
-- PAS DE \connect ICI - la base cible est passee en argument psql.
-- Cela evite les problemes d interpretation de \connect dans
-- docker-entrypoint-initdb.d sur certaines versions de psql/Alpine.
-- =============================================================================

SET search_path TO public;

DROP TABLE IF EXISTS drift_feature_metrics CASCADE;
DROP TABLE IF EXISTS drift_monitoring_runs CASCADE;
DROP TABLE IF EXISTS reference_feature_stats CASCADE;
DROP TABLE IF EXISTS training_runs CASCADE;


-- =============================================================================
-- TABLE 1 : training_runs
-- =============================================================================
CREATE TABLE training_runs (
    id                 SERIAL PRIMARY KEY,
    run_id             VARCHAR(64)  UNIQUE NOT NULL,
    model_name         VARCHAR(128) NOT NULL,
    run_date           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    dataset_uri        TEXT,
    dataset_version    VARCHAR(64),
    status             VARCHAR(32)  NOT NULL DEFAULT 'running',
    n_rows             INTEGER,
    val_rmse           FLOAT,
    val_r2             FLOAT,
    train_duration_sec FLOAT,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_training_runs_model_status ON training_runs(model_name, status, run_date DESC);
CREATE INDEX idx_training_runs_rundate      ON training_runs(run_date DESC);


-- =============================================================================
-- TABLE 2 : reference_feature_stats
-- =============================================================================
CREATE TABLE reference_feature_stats (
    id           SERIAL      PRIMARY KEY,
    run_id       VARCHAR(64) NOT NULL REFERENCES training_runs(run_id) ON DELETE CASCADE,
    feature_name VARCHAR(128) NOT NULL,
    mean_value   FLOAT,
    std_value    FLOAT,
    min_value    FLOAT,
    max_value    FLOAT,
    q25          FLOAT,
    median       FLOAT,
    q75          FLOAT,
    n_rows       INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_ref_stats_run_feature UNIQUE (run_id, feature_name)
);

CREATE INDEX idx_ref_stats_runid   ON reference_feature_stats(run_id);
CREATE INDEX idx_ref_stats_feature ON reference_feature_stats(feature_name);


-- =============================================================================
-- TABLE 3 : drift_monitoring_runs
-- =============================================================================
CREATE TABLE drift_monitoring_runs (
    id                        SERIAL      PRIMARY KEY,
    monitoring_run_id         VARCHAR(64) UNIQUE NOT NULL,
    ref_run_id                VARCHAR(64) REFERENCES training_runs(run_id) ON DELETE RESTRICT,
    observation_start         TIMESTAMPTZ,
    observation_end           TIMESTAMPTZ,
    input_dataset_uri         TEXT        NOT NULL DEFAULT '',
    n_rows_observed           INTEGER,
    drift_score_global        FLOAT       NOT NULL DEFAULT 0.0,
    n_features_in_drift       INTEGER     NOT NULL DEFAULT 0,
    drift_detected            BOOLEAN     NOT NULL DEFAULT FALSE,
    alert_level               VARCHAR(32) NOT NULL DEFAULT 'none',
    consecutive_drift_windows INTEGER     NOT NULL DEFAULT 0,
    retrain_triggered         BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_monitoring_runs_ref_run    ON drift_monitoring_runs(ref_run_id, created_at DESC);
CREATE INDEX idx_monitoring_runs_alertlevel ON drift_monitoring_runs(alert_level, created_at DESC);
CREATE INDEX idx_monitoring_runs_drift      ON drift_monitoring_runs(drift_detected, created_at DESC);


-- =============================================================================
-- TABLE 4 : drift_feature_metrics
-- =============================================================================
CREATE TABLE drift_feature_metrics (
    id                SERIAL      PRIMARY KEY,
    monitoring_run_id VARCHAR(64) NOT NULL
                      REFERENCES drift_monitoring_runs(monitoring_run_id) ON DELETE CASCADE,
    feature_name      VARCHAR(128) NOT NULL,
    is_binary         BOOLEAN     NOT NULL DEFAULT FALSE,
    psi               FLOAT,
    ks_stat           FLOAT,
    ks_pvalue         FLOAT,
    chi2_stat         FLOAT,
    chi2_pvalue       FLOAT,
    wasserstein       FLOAT,
    wasserstein_norm  FLOAT,
    js_divergence     FLOAT,
    mean_ref          FLOAT,
    mean_cur          FLOAT,
    mean_delta        FLOAT,
    std_ref           FLOAT,
    std_cur           FLOAT,
    std_delta         FLOAT,
    drift_flag        BOOLEAN NOT NULL DEFAULT FALSE,
    drift_reason      TEXT    NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_drift_metrics_run_feature UNIQUE (monitoring_run_id, feature_name)
);

CREATE INDEX idx_drift_metrics_runid   ON drift_feature_metrics(monitoring_run_id);
CREATE INDEX idx_drift_metrics_feature ON drift_feature_metrics(feature_name);
CREATE INDEX idx_drift_metrics_flag    ON drift_feature_metrics(drift_flag, feature_name);


-- =============================================================================
-- PERMISSIONS
-- =============================================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON
    training_runs,
    reference_feature_stats,
    drift_monitoring_runs,
    drift_feature_metrics
TO mlops;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mlops;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mlops;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO mlops;


-- =============================================================================
-- VUES
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
            THEN fm.feature_name || ' (PSI=' || ROUND(fm.psi::numeric, 3) || ')'
        END,
        ', ' ORDER BY fm.psi DESC
    ) AS drifted_features
FROM drift_monitoring_runs mr
LEFT JOIN drift_feature_metrics fm ON fm.monitoring_run_id = mr.monitoring_run_id
WHERE mr.created_at = (SELECT MAX(created_at) FROM drift_monitoring_runs)
GROUP BY mr.monitoring_run_id, mr.created_at, mr.alert_level,
         mr.drift_score_global, mr.n_features_in_drift,
         mr.consecutive_drift_windows, mr.retrain_triggered;

GRANT SELECT ON v_latest_drift_status TO mlops;


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

GRANT SELECT ON v_alert_history TO mlops;