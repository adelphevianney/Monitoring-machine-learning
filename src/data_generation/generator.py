"""
Génère des données synthétiques pour la simulation MLOps.

Deux modes :
  - référence  : distribution d'entraînement stable
  - drifted    : distributions modifiées pour simuler un data drift

Structure des données :
  x1 ~ N(0, 1)
  x2 ~ N(2, 1.5)     → drift : N(4, 1.5)
  x3 ~ U(-1, 1)
  x4 ~ Bernoulli(0.4) → drift : Bernoulli(0.7)
  y  = 3*x1 - 1.5*x2 + 2*x3 + 0.8*x4 + N(0, 0.5)
"""

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import DATA_CFG, DRIFT_CFG, DataConfig, DriftConfig

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_target(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    noise: np.ndarray,
    cfg: DataConfig,
) -> np.ndarray:
    return (
        cfg.coef_x1 * x1
        + cfg.coef_x2 * x2
        + cfg.coef_x3 * x3
        + cfg.coef_x4 * x4
        + noise
    )


# ── générateurs principaux ───────────────────────────────────────────────────

def generate_reference_data(
    n_rows: Optional[int] = None,
    seed: Optional[int] = None,
    cfg: DataConfig = DATA_CFG,
) -> pd.DataFrame:
    """
    Génère un dataset selon la distribution de référence.

    Args:
        n_rows  : nombre de lignes (défaut : cfg.n_rows)
        seed    : graine aléatoire pour reproductibilité
        cfg     : configuration des paramètres de distribution

    Returns:
        DataFrame avec colonnes [x1, x2, x3, x4, y]
    """
    n = n_rows or cfg.n_rows
    rng = _make_rng(seed if seed is not None else cfg.random_seed)

    x1 = rng.normal(cfg.x1_mean, cfg.x1_std, n)
    x2 = rng.normal(cfg.x2_mean, cfg.x2_std, n)
    x3 = rng.uniform(cfg.x3_low, cfg.x3_high, n)
    x4 = rng.binomial(1, cfg.x4_p, n).astype(float)
    noise = rng.normal(0, cfg.noise_std, n)

    y = _build_target(x1, x2, x3, x4, noise, cfg)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})

    logger.info(
        "Référence générée : %d lignes | y_mean=%.3f y_std=%.3f",
        n, df["y"].mean(), df["y"].std(),
    )
    return df


def generate_drifted_data(
    n_rows: Optional[int] = None,
    seed: Optional[int] = None,
    cfg: DataConfig = DATA_CFG,
    drift_cfg: DriftConfig = DRIFT_CFG,
) -> pd.DataFrame:
    """
    Génère un dataset avec drift sur x2 et x4.

    Les drifts simulés :
      - x2 : mean passe de 2.0 à 4.0  (dérive de +2σ)
      - x4 : probabilité passe de 0.4 à 0.7

    Args:
        n_rows    : nombre de lignes
        seed      : graine aléatoire
        cfg       : config de base
        drift_cfg : paramètres de drift

    Returns:
        DataFrame avec colonnes [x1, x2, x3, x4, y]
    """
    n = n_rows or cfg.n_rows
    rng = _make_rng(seed if seed is not None else cfg.random_seed + 999)

    x1 = rng.normal(cfg.x1_mean, cfg.x1_std, n)
    # ── drift x2 ──
    x2 = rng.normal(drift_cfg.x2_mean_drifted, cfg.x2_std, n)
    x3 = rng.uniform(cfg.x3_low, cfg.x3_high, n)
    # ── drift x4 ──
    x4 = rng.binomial(1, drift_cfg.x4_p_drifted, n).astype(float)
    noise = rng.normal(0, cfg.noise_std, n)

    y = _build_target(x1, x2, x3, x4, noise, cfg)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})

    logger.info(
        "Drifted générée : %d lignes | x2_mean=%.2f (expected %.2f) | x4_mean=%.2f (expected %.2f)",
        n, df["x2"].mean(), drift_cfg.x2_mean_drifted,
        df["x4"].mean(), drift_cfg.x4_p_drifted,
    )
    return df


def generate_partial_drift(
    drift_factor: float = 0.5,
    n_rows: Optional[int] = None,
    seed: Optional[int] = None,
    cfg: DataConfig = DATA_CFG,
    drift_cfg: DriftConfig = DRIFT_CFG,
) -> pd.DataFrame:
    """
    Génère un dataset avec drift progressif (interpolation linéaire).

    drift_factor = 0.0  → distribution de référence
    drift_factor = 1.0  → distribution complètement driftée

    Utile pour simuler une dégradation graduelle.
    """
    n = n_rows or cfg.n_rows
    rng = _make_rng(seed if seed is not None else cfg.random_seed + 1000)

    x2_mean = cfg.x2_mean + drift_factor * (drift_cfg.x2_mean_drifted - cfg.x2_mean)
    x4_p = cfg.x4_p + drift_factor * (drift_cfg.x4_p_drifted - cfg.x4_p)

    x1 = rng.normal(cfg.x1_mean, cfg.x1_std, n)
    x2 = rng.normal(x2_mean, cfg.x2_std, n)
    x3 = rng.uniform(cfg.x3_low, cfg.x3_high, n)
    x4 = rng.binomial(1, x4_p, n).astype(float)
    noise = rng.normal(0, cfg.noise_std, n)

    y = _build_target(x1, x2, x3, x4, noise, cfg)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})

    logger.info(
        "Partial drift (factor=%.2f) : x2_mean=%.2f | x4_p=%.2f",
        drift_factor, x2_mean, x4_p,
    )
    return df


# ── utilitaires de sauvegarde locale ─────────────────────────────────────────

def save_dataset(
    df: pd.DataFrame,
    output_dir: str = "data",
    prefix: str = "dataset",
) -> Path:
    """
    Sauvegarde localement au format parquet avec horodatage.

    Returns:
        Path vers le fichier créé.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{prefix}_{ts}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")

    logger.info("Dataset sauvegardé : %s (%d lignes)", path, len(df))
    return path


def load_dataset(path: str) -> pd.DataFrame:
    """Charge un parquet en DataFrame."""
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Dataset chargé : %s (%d lignes)", path, len(df))
    return df


# ── résumé statistique ───────────────────────────────────────────────────────

def compute_feature_stats(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Calcule les statistiques descriptives des features.

    Returns:
        DataFrame indexé par feature_name avec colonnes :
        mean, std, min, q25, median, q75, max
    """
    cols = features or [c for c in df.columns if c != "y"]
    stats = df[cols].agg(["mean", "std", "min", "median", "max"]).T
    stats["q25"] = df[cols].quantile(0.25)
    stats["q75"] = df[cols].quantile(0.75)
    stats.index.name = "feature_name"
    return stats[["mean", "std", "min", "q25", "median", "q75", "max"]]


# ── point d'entrée CLI (debug rapide) ────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    mode = sys.argv[1] if len(sys.argv) > 1 else "reference"

    if mode == "drift":
        df = generate_drifted_data()
        label = "DRIFTED"
    elif mode == "partial":
        factor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        df = generate_partial_drift(drift_factor=factor)
        label = f"PARTIAL DRIFT (factor={factor})"
    else:
        df = generate_reference_data()
        label = "REFERENCE"

    print(f"\n{'='*50}")
    print(f"Mode : {label}")
    print(f"{'='*50}")
    print(df.describe().round(3).to_string())
    print(f"\nCorrelations avec y :")
    print(df.corr()["y"].drop("y").round(3).to_string())

    path = save_dataset(df, "data/raw", prefix=mode)
    print(f"\nSauvegardé dans : {path}")
