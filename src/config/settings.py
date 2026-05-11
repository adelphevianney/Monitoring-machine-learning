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
    x2_mean_drifted: float = 4.0
    x4_p_drifted: float = 0.7


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_seed: int = 42
    model_name: str = "drift_regressor"


@dataclass
class MinIOConfig:
    # endpoint (SDK natif MinIO, non utilisé — conservé pour compatibilité)
    endpoint: str = os.getenv("MINIO_ENDPOINT", "mlops-minio:9000")
    # endpoint_url (boto3) — doit inclure le schéma http://
    # La variable d'env est MINIO_ENDPOINTS (avec S) côté docker-compose
    endpoint_url: str = os.getenv("MINIO_ENDPOINTS", "http://mlops-minio:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure: bool = False

    # Bucket principal pour les artefacts de training (runs/<run_id>/)
    bucket: str = "models"

    # Liste complète des buckets à créer au démarrage via ensure_buckets_exist()
    # IMPORTANT : "models" doit être dans cette liste
    buckets: list = field(default_factory=lambda: [
        "models",               # artefacts de training (model.joblib, metrics, stats)
        "raw-datasets",         # datasets bruts générés
        "processed-datasets",   # datasets après preprocessing
        "feature-stats",        # stats de référence indexées par date
        "monitoring-snapshots", # snapshots des données de monitoring
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
    # Seuils PSI
    psi_alert_threshold: float = 0.01
    psi_critical_threshold: float = 0.015
    # Nombre de features simultanément en drift pour déclencher une alerte
    n_critical_features_for_alert: int = 2
    # Fenêtres consécutives en alerte critique avant ré-entraînement
    consecutive_windows_for_retrain: int = 2
    # Nombre de buckets pour PSI et Jensen-Shannon
    n_bins: int = 10
    # Seuil Wasserstein normalisé (distance / std_ref)
    wasserstein_threshold: float = 0.2
    # Seuil Jensen-Shannon divergence (0=identique, 1=opposé)
    js_threshold: float = 0.1


# ── Instances globales ────────────────────────────────────────────────────────
DATA_CFG = DataConfig()
DRIFT_CFG = DriftConfig()
TRAIN_CFG = TrainingConfig()
MINIO_CFG = MinIOConfig()
PG_CFG = PostgresConfig()
MONITOR_CFG = MonitoringConfig()