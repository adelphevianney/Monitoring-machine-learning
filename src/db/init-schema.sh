
#-- =============================================================================
#-- Schéma MLOps Drift Monitoring
#-- À exécuter UNE FOIS au démarrage de l'infrastructure.
#-- Compatible PostgreSQL 13+
#-- =============================================================================
#
#-- -----------------------------------------------------------------------------
#-- TABLE 1 : training_runs
#-- Historique de tous les runs d'entraînement.
#-- run_id correspond au préfixe MinIO : runs/<run_id>/
#-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS training_runs (
    run_id              VARCHAR(64)     PRIMARY KEY,
    model_name          VARCHAR(128)    NOT NULL,
    dataset_uri         TEXT            NOT NULL,
    dataset_version     VARCHAR(64),
    status              VARCHAR(32)     NOT NULL DEFAULT 'running',
                        -- 'running' | 'success' | 'failed'
    n_rows              INTEGER,
    val_rmse            DOUBLE PRECISION,
    val_r2              DOUBLE PRECISION,
    train_duration_sec  DOUBLE PRECISION,
    run_date            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_runs_model_name
    ON training_runs (model_name, run_date DESC);

CREATE INDEX IF NOT EXISTS idx_training_runs_status
    ON training_runs (status, run_date DESC);

#-- -----------------------------------------------------------------------------
#-- TABLE 2 : reference_feature_stats
#-- Stats de distribution du dataset de référence, par feature et par run.
#-- Utilisées pendant le monitoring pour calculer le delta.
#-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reference_feature_stats (
    id              SERIAL          PRIMARY KEY,
    run_id          VARCHAR(64)     NOT NULL REFERENCES training_runs(run_id)
                                    ON DELETE CASCADE,
    feature_name    VARCHAR(128)    NOT NULL,
    mean_value      DOUBLE PRECISION,
    std_value       DOUBLE PRECISION,
    min_value       DOUBLE PRECISION,
    max_value       DOUBLE PRECISION,
    q25             DOUBLE PRECISION,
    median          DOUBLE PRECISION,
    q75             DOUBLE PRECISION,
    n_rows          INTEGER,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_reference_stats_run_feature
        UNIQUE (run_id, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_reference_stats_run_id
    ON reference_feature_stats (run_id);

#-- -----------------------------------------------------------------------------
#-- TABLE 3 : drift_monitoring_runs
#-- Résultat global de chaque exécution du DAG drift_monitoring_pipeline.
#-- ref_run_id → FK vers training_runs (run utilisé comme référence).
#-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_monitoring_runs (
    monitoring_run_id   VARCHAR(64)     PRIMARY KEY,
                        -- UUID généré par postgres_client.insert_monitoring_run()
    ref_run_id          VARCHAR(64)     NOT NULL REFERENCES training_runs(run_id)
                        ON DELETE RESTRICT,
    observation_start   TIMESTAMPTZ     NOT NULL,
    observation_end     TIMESTAMPTZ     NOT NULL,
    input_dataset_uri   TEXT            NOT NULL DEFAULT '',
    drift_score_global  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    drift_detected      BOOLEAN         NOT NULL DEFAULT FALSE,
    alert_level         VARCHAR(32)     NOT NULL DEFAULT 'none',
                        -- 'none' | 'warning' | 'critical'
    n_features_in_drift INTEGER         NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_ref_run_id
    ON drift_monitoring_runs (ref_run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_alert_level
    ON drift_monitoring_runs (alert_level, created_at DESC);

#-- -----------------------------------------------------------------------------
#-- TABLE 4 : drift_feature_metrics
#-- Détail du drift par feature pour chaque monitoring_run.
#-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_feature_metrics (
    id                  SERIAL          PRIMARY KEY,
    monitoring_run_id   VARCHAR(64)     NOT NULL
                        REFERENCES drift_monitoring_runs(monitoring_run_id)
                        ON DELETE CASCADE,
    feature_name        VARCHAR(128)    NOT NULL,
    psi                 DOUBLE PRECISION,
    ks_stat             DOUBLE PRECISION,
    ks_pvalue           DOUBLE PRECISION,
    mean_ref            DOUBLE PRECISION,
    mean_cur            DOUBLE PRECISION,
    mean_delta          DOUBLE PRECISION,
    std_ref             DOUBLE PRECISION,
    std_cur             DOUBLE PRECISION,
    std_delta           DOUBLE PRECISION,
    drift_flag          BOOLEAN         NOT NULL DEFAULT FALSE,
    drift_reason        TEXT            NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_drift_metrics_run_feature
        UNIQUE (monitoring_run_id, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_monitoring_run_id
    ON drift_feature_metrics (monitoring_run_id);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature_drift
    ON drift_feature_metrics (feature_name, drift_flag);