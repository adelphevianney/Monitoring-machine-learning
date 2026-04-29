"""
Centralise tous les paramètres du projet MLOps.
Chargé depuis variables d'environnement ou valeurs par défaut.
"""
import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    n_rows: int = 5_000
    n_features: int = 4
    noise_std: float = 0.5
    random_seed: int = 42

    # Distribution de référence
    x1_mean: float = 0.0
    x1_std: float = 1.0
    x2_mean: float = 2.0
    x2_std: float = 1.5
    x3_low: float = -1.0
    x3_high: float = 1.0
    x4_p: float = 0.4

    # Coefficients de la cible
    coef_x1: float = 3.0
    coef_x2: float = -1.5
    coef_x3: float = 2.0
    coef_x4: float = 0.8


@dataclass
class DriftConfig:
    """Paramètres pour simuler un drift de distribution."""
    x2_mean_drifted: float = 4.0   # dérive de x2
    x4_p_drifted: float = 0.7      # dérive de x4


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_seed: int = 42
    model_name: str = "drift_regressor"


@dataclass
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_drift_monitoring")
    artifact_root: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow-artifacts")


@dataclass
class MinIOConfig:
    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure: bool = False
    buckets: list = field(default_factory=lambda: [
        "mlflow-artifacts",
        "raw-datasets",
        "processed-datasets",
        "feature-stats",
        "monitoring-snapshots",
    ])


@dataclass
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "mlops")
    user: str = os.getenv("POSTGRES_USER", "mlops")
    password: str = os.getenv("POSTGRES_PASSWORD", "mlops")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class MonitoringConfig:
    psi_alert_threshold: float = 0.2
    psi_critical_threshold: float = 0.25
    n_critical_features_for_alert: int = 2
    consecutive_windows_for_retrain: int = 3
    # Buckets PSI standard
    n_bins: int = 10


# ── instances globales ──────────────────────────────────────────────────────
DATA_CFG = DataConfig()
DRIFT_CFG = DriftConfig()
TRAIN_CFG = TrainingConfig()
MLFLOW_CFG = MLflowConfig()
MINIO_CFG = MinIOConfig()
PG_CFG = PostgresConfig()
MONITOR_CFG = MonitoringConfig()
