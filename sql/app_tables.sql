-- =============================================================================
-- TABLES APPLICATIVES — MLOps Drift Monitoring
-- Base cible : mlops
--
-- Ce fichier est exécuté par docker-entrypoint-initdb.d/ APRÈS que
-- init-multiple-dbs.sh a créé la base "mlops" et son user.
-- Le SET search_path garantit que les tables atterrissent dans le bon schéma.
-- =============================================================================

-- Connexion explicite à la base mlops
\connect mlops

SET search_path TO public;

-- Drop dans l'ordre inverse des FK pour éviter les erreurs de contrainte
DROP TABLE IF EXISTS drift_feature_metrics CASCADE;
DROP TABLE IF EXISTS drift_monitoring_runs CASCADE;
DROP TABLE IF EXISTS reference_feature_stats CASCADE;
DROP TABLE IF EXISTS training_runs CASCADE;


-- =============================================================================
-- TABLE 1 : training_runs
-- Historique de tous les runs d'entraînement.
-- run_id = préfixe MinIO : runs/<run_id>/
-- =============================================================================
CREATE TABLE training_runs (
    id                 SERIAL PRIMARY KEY,
    run_id             VARCHAR(64)  UNIQUE NOT NULL,
    model_name         VARCHAR(128) NOT NULL,
    run_date           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    dataset_uri        TEXT,
    dataset_version    VARCHAR(64),
    status             VARCHAR(32)  NOT NULL DEFAULT 'running',
                       -- 'running' | 'success' | 'failed'
    n_rows             INTEGER,
    val_rmse           FLOAT,
    val_r2             FLOAT,
    train_duration_sec FLOAT,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_training_runs_model_status ON training_runs(model_name, status, run_date DESC);
CREATE INDEX idx_training_runs_rundate      ON training_runs(run_date DESC);

COMMENT ON TABLE training_runs IS
    'Historique des runs d entraînement. Artefacts dans MinIO sous runs/<run_id>/.';
COMMENT ON COLUMN training_runs.run_id IS
    'Identifiant unique du run = préfixe MinIO runs/<run_id>/.';


-- =============================================================================
-- TABLE 2 : reference_feature_stats
-- Stats de distribution du dataset de référence, par feature et par run.
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

COMMENT ON TABLE reference_feature_stats IS
    'Stats de référence par feature. Miroir du JSON MinIO sous runs/<run_id>/reference_stats/.';


-- =============================================================================
-- TABLE 3 : drift_monitoring_runs
-- Résultat global de chaque exécution du DAG drift_monitoring_pipeline.
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
                              -- 'none' | 'warning' | 'critical'
    consecutive_drift_windows INTEGER     NOT NULL DEFAULT 0,
    retrain_triggered         BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_monitoring_runs_ref_run    ON drift_monitoring_runs(ref_run_id, created_at DESC);
CREATE INDEX idx_monitoring_runs_alertlevel ON drift_monitoring_runs(alert_level, created_at DESC);
CREATE INDEX idx_monitoring_runs_drift      ON drift_monitoring_runs(drift_detected, created_at DESC);

COMMENT ON COLUMN drift_monitoring_runs.ref_run_id IS
    'run_id du training run utilisé comme référence pour ce check de drift.';


-- =============================================================================
-- TABLE 4 : drift_feature_metrics
-- Détail du drift par feature pour chaque monitoring_run.
-- =============================================================================
CREATE TABLE drift_feature_metrics (
    id                SERIAL      PRIMARY KEY,
    monitoring_run_id VARCHAR(64) NOT NULL
                      REFERENCES drift_monitoring_runs(monitoring_run_id) ON DELETE CASCADE,
    feature_name      VARCHAR(128) NOT NULL,
    psi               FLOAT,
    ks_stat           FLOAT,
    ks_pvalue         FLOAT,
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

-- =============================================================================
-- PERMISSIONS — accorder tous les droits à l'utilisateur applicatif mlops
--
-- POURQUOI : ce fichier est exécuté par le superuser postgres dans
-- docker-entrypoint-initdb.d/. Les tables sont donc owned par postgres.
-- Sans ces GRANT, l'utilisateur mlops peut se connecter à la base
-- mais ne peut ni lire ni écrire dans les tables.
--
-- ALTER DEFAULT PRIVILEGES couvre les objets créés APRÈS ce script
-- (ex: séquences créées lors des premiers INSERT sur les colonnes SERIAL).
-- =============================================================================

-- Tables
GRANT SELECT, INSERT, UPDATE, DELETE ON
    training_runs,
    reference_feature_stats,
    drift_monitoring_runs,
    drift_feature_metrics
TO mlops;

-- Vues (lecture seule)
GRANT SELECT ON
    v_latest_drift_status,
    v_alert_history
TO mlops;

-- Séquences SERIAL (nécessaire pour les INSERT avec id auto-incrémenté)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mlops;

-- Droits par défaut pour les futurs objets créés par postgres dans cette base
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mlops;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO mlops;