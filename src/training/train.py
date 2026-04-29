"""
Pipeline d'entraînement complet.

Responsabilités :
  1. Charger le dataset (local ou MinIO)
  2. Split train / validation
  3. Entraîner un RandomForestRegressor (ou autre modèle via config)
  4. Évaluer sur le jeu de validation
  5. Logger le run dans MLflow (params + métriques + artefacts)
  6. Calculer et sauvegarder les stats de référence
  7. Enregistrer le modèle dans le MLflow Model Registry
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config.settings import MLFLOW_CFG, TRAIN_CFG, TrainingConfig
from src.data_generation.generator import compute_feature_stats

logger = logging.getLogger(__name__)

# ── modèles disponibles ───────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "ridge": Ridge,
}


# ── évaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Calcule les métriques de régression.

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


# ── entraînement ──────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None,
    features: Optional[list] = None,
    target: str = "y",
    cfg: TrainingConfig = TRAIN_CFG,
) -> Tuple[object, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Entraîne un modèle de régression.

    Args:
        df          : DataFrame d'entraînement
        model_type  : clé dans MODEL_REGISTRY
        model_params: hyperparamètres du modèle
        features    : liste des colonnes features (défaut : tout sauf target)
        target      : nom de la colonne cible
        cfg         : config d'entraînement

    Returns:
        (model, metrics, X_val, y_val)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type '{model_type}' inconnu. Disponibles : {list(MODEL_REGISTRY)}")

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


# ── pipeline MLflow ───────────────────────────────────────────────────────────

def run_training_pipeline(
    df: pd.DataFrame,
    dataset_version: str = "v1",
    dataset_uri: str = "",
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None,
    register_model: bool = True,
    transition_to_staging: bool = False,
) -> str:
    """
    Pipeline complet : entraîne + logue dans MLflow + enregistre le modèle.

    Args:
        df                   : DataFrame d'entraînement
        dataset_version      : identifiant lisible du dataset
        dataset_uri          : URI MinIO ou chemin local du dataset
        model_type           : type de modèle à entraîner
        model_params         : hyperparamètres (None = défauts)
        register_model       : enregistrer dans le Model Registry MLflow
        transition_to_staging: promouvoir automatiquement en Staging

    Returns:
        mlflow_run_id (str)
    """
    mlflow.set_tracking_uri(MLFLOW_CFG.tracking_uri)
    mlflow.set_experiment(MLFLOW_CFG.experiment_name)

    features = [c for c in df.columns if c != "y"]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run démarré : %s", run_id)

        # ── tags ──
        mlflow.set_tags({
            "model_type": model_type,
            "dataset_version": dataset_version,
            "dataset_uri": dataset_uri,
            "features": json.dumps(features),
        })

        # ── entraînement ──
        params = model_params or _default_params(model_type, TRAIN_CFG.random_seed)
        mlflow.log_params(params)

        model, metrics, X_val, y_val = train(
            df, model_type=model_type, model_params=params
        )
        mlflow.log_metrics(metrics)

        # ── stats de référence (artefact JSON) ──
        ref_stats = compute_feature_stats(df, features)
        stats_path = Path("/tmp") / f"reference_stats_{run_id[:8]}.json"
        ref_stats_dict = ref_stats.round(6).to_dict(orient="index")
        stats_path.write_text(json.dumps(ref_stats_dict, indent=2))
        mlflow.log_artifact(str(stats_path), artifact_path="reference_stats")
        logger.info("Stats de référence loguées (%d features)", len(features))

        # ── feature importance (si disponible) ──
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_path = Path("/tmp") / f"feature_importance_{run_id[:8]}.csv"
            importance.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path), artifact_path="reports")

        # ── log du modèle ──
        model_signature = mlflow.models.infer_signature(
            pd.DataFrame(X_val), model.predict(X_val)
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=model_signature,
            registered_model_name=TRAIN_CFG.model_name if register_model else None,
        )

        logger.info("Run MLflow terminé : %s", run_id)

        # ── transition automatique vers Staging ──
        if register_model and transition_to_staging:
            _promote_to_staging(TRAIN_CFG.model_name)

    return run_id


def _promote_to_staging(model_name: str) -> None:
    """Promeut la dernière version du modèle en Staging."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_CFG.tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.warning("Aucune version trouvée pour '%s'", model_name)
        return

    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info("Modèle '%s' v%s → Staging", model_name, latest.version)


# ── point d'entrée CLI ────────────────────────────────────────────────────────

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

    run_id = run_training_pipeline(
        df,
        dataset_version=version,
        register_model=False,   # False pour test local sans registry
    )
    print(f"\nRun ID : {run_id}")
