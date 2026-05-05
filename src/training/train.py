"""
Pipeline d'entraînement complet.

Responsabilités :
  1. Charger le dataset (local ou MinIO)
  2. Split train / validation
  3. Entraîner un RandomForestRegressor (ou autre modèle via config)
  4. Évaluer sur le jeu de validation
  5. Sauvegarder le modèle, les métriques et les artefacts dans MinIO
  6. Calculer et sauvegarder les stats de référence

STOCKAGE
────────
Tous les uploads passent par src.storage.minio_client (boto3 uniquement).
Ce module n'instancie plus de client directement.

Structure des artefacts dans MinIO (bucket=MINIO_CFG.bucket) :
    runs/<run_id>/model.joblib
    runs/<run_id>/metrics.json
    runs/<run_id>/params.json
    runs/<run_id>/tags.json
    runs/<run_id>/reference_stats/reference_stats.json
    runs/<run_id>/reports/feature_importance.csv
"""

import io
import json
import logging
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config.settings import MINIO_CFG, TRAIN_CFG, TrainingConfig
from src.data_generation.generator import compute_feature_stats
from src.storage.minio_client import (
    ensure_buckets_exist,
    upload_bytes,
    upload_json,
    make_run_prefix,
)

logger = logging.getLogger(__name__)

# ── Modèles disponibles ───────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "ridge": Ridge,
}


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Calcule les métriques de régression sur un jeu de données.

    Returns:
        dict avec rmse, mae, r2, mse
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


# ── Entraînement ──────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None,
    features: Optional[list] = None,
    target: str = "y",
    cfg: TrainingConfig = TRAIN_CFG,
) -> Tuple[object, Dict[str, float], pd.DataFrame, pd.Series]:
    """
    Entraîne un modèle de régression.

    Args:
        df          : DataFrame d'entraînement
        model_type  : clé dans MODEL_REGISTRY
        model_params: hyperparamètres du modèle (None = valeurs par défaut)
        features    : liste des colonnes features (défaut : tout sauf target)
        target      : nom de la colonne cible
        cfg         : config d'entraînement

    Returns:
        (model, metrics, X_val, y_val)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model_type '{model_type}' inconnu. Disponibles : {list(MODEL_REGISTRY)}"
        )

    cols_features = features or [c for c in df.columns if c != target]
    X = df[cols_features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed
    )

    params = model_params or _default_params(model_type, cfg.random_seed)
    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(**params)

    logger.info("Entraînement %s sur %d lignes…", model_type, len(X_train))
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)

    logger.info(
        "Train  → RMSE=%.4f MAE=%.4f R²=%.4f",
        train_metrics["rmse"], train_metrics["mae"], train_metrics["r2"],
    )
    logger.info(
        "Val    → RMSE=%.4f MAE=%.4f R²=%.4f  (%.2fs)",
        val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"], train_duration,
    )

    metrics = {
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "train_duration_sec": train_duration,
        "n_rows_total": len(df),
        "n_rows_train": len(X_train),
        "n_rows_val": len(X_val),
        "n_features": len(cols_features),
    }

    return model, metrics, X_val, y_val


def _default_params(model_type: str, seed: int) -> Dict:
    defaults = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_leaf": 4,
            "random_state": seed,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": seed,
        },
        "ridge": {"alpha": 1.0},
    }
    return defaults.get(model_type, {})


# ── Pipeline complet ──────────────────────────────────────────────────────────

def run_training_pipeline(
    df: pd.DataFrame,
    dataset_version: str = "v1",
    dataset_uri: str = "",
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None,
) -> str:
    """
    Pipeline complet : entraîne + sauvegarde tous les artefacts dans MinIO.

    Tous les uploads passent par minio_client.upload_bytes / upload_json.
    Aucun client boto3 n'est instancié ici directement.

    Structure dans le bucket MINIO_CFG.bucket :
        runs/<run_id>/model.joblib
        runs/<run_id>/metrics.json
        runs/<run_id>/params.json
        runs/<run_id>/tags.json
        runs/<run_id>/reference_stats/reference_stats.json
        runs/<run_id>/reports/feature_importance.csv     (si disponible)

    Args:
        df              : DataFrame d'entraînement
        dataset_version : identifiant lisible du dataset (ex: "v20240101")
        dataset_uri     : URI MinIO ou chemin local du dataset
        model_type      : type de modèle à entraîner
        model_params    : hyperparamètres (None = valeurs par défaut)

    Returns:
        run_id (str) — préfixe MinIO utilisé pour ce run
    """
    import uuid
    run_id = uuid.uuid4().hex[:12]
    prefix = make_run_prefix(run_id)
    bucket = MINIO_CFG.bucket

    ensure_buckets_exist()
    logger.info("Démarrage du run : %s  (bucket=%s)", run_id, bucket)

    features = [c for c in df.columns if c != "y"]

    # ── tags ──────────────────────────────────────────────────────────────────
    tags = {
        "model_type": model_type,
        "dataset_version": dataset_version,
        "dataset_uri": dataset_uri,
        "features": features,
    }
    upload_json(tags, bucket, f"{prefix}/tags.json")

    # ── params ────────────────────────────────────────────────────────────────
    params = model_params or _default_params(model_type, TRAIN_CFG.random_seed)
    upload_json(params, bucket, f"{prefix}/params.json")

    # ── entraînement ──────────────────────────────────────────────────────────
    model, metrics, X_val, y_val = train(
        df, model_type=model_type, model_params=params
    )

    # ── métriques ─────────────────────────────────────────────────────────────
    upload_json(metrics, bucket, f"{prefix}/metrics.json")
    logger.info(
        "Métriques sauvegardées — val_r2=%.4f val_rmse=%.4f",
        metrics.get("val_r2", 0), metrics.get("val_rmse", 0),
    )

    # ── stats de référence ────────────────────────────────────────────────────
    ref_stats = compute_feature_stats(df, features)
    ref_stats_dict = ref_stats.round(6).to_dict(orient="index")
    upload_json(
        ref_stats_dict,
        bucket,
        f"{prefix}/reference_stats/reference_stats.json",
    )
    logger.info("Stats de référence sauvegardées (%d features)", len(features))

    # ── feature importance (si disponible) ────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        upload_bytes(
            importance.to_csv(index=False).encode(),
            bucket,
            f"{prefix}/reports/feature_importance.csv",
            content_type="text/csv",
        )
        logger.info("Feature importance sauvegardée")

    # ── modèle (joblib) ───────────────────────────────────────────────────────
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    upload_bytes(
        model_buffer.read(),
        bucket,
        f"{prefix}/model.joblib",
        content_type="application/octet-stream",
    )
    logger.info("Modèle sauvegardé → s3://%s/%s/model.joblib", bucket, prefix)

    logger.info("Run terminé : %s", run_id)
    return run_id


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_generation.generator import generate_reference_data, generate_drifted_data

    mode = sys.argv[1] if len(sys.argv) > 1 else "reference"

    if mode == "drift":
        df = generate_drifted_data()
        version = "drifted_v1"
    else:
        df = generate_reference_data()
        version = "reference_v1"

    run_id = run_training_pipeline(df, dataset_version=version)
    print(f"\nRun ID : {run_id}")