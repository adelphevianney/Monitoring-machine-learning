"""
Tests unitaires pour le data generator et le pipeline d'entraînement.
Lance avec : pytest tests/test_core.py -v
"""
import json
import numpy as np
import pandas as pd
import pytest

from src.data_generation.generator import (
    generate_reference_data,
    generate_drifted_data,
    generate_partial_drift,
    compute_feature_stats,
    save_dataset,
    load_dataset,
)
from src.training.train import train, evaluate_model


# ══════════════════════════════════════════════════════
# DATA GENERATOR
# ══════════════════════════════════════════════════════

class TestGenerateReference:
    def test_shape(self):
        df = generate_reference_data(n_rows=1000, seed=0)
        assert df.shape == (1000, 5), "doit avoir 5 colonnes : x1 x2 x3 x4 y"

    def test_columns(self):
        df = generate_reference_data(n_rows=100, seed=0)
        assert set(df.columns) == {"x1", "x2", "x3", "x4", "y"}

    def test_no_nulls(self):
        df = generate_reference_data(n_rows=500, seed=42)
        assert df.isnull().sum().sum() == 0

    def test_x1_distribution(self):
        df = generate_reference_data(n_rows=50_000, seed=1)
        assert abs(df["x1"].mean()) < 0.05, "x1 ~ N(0,1) — mean ≈ 0"
        assert abs(df["x1"].std() - 1.0) < 0.05

    def test_x2_distribution(self):
        df = generate_reference_data(n_rows=50_000, seed=2)
        assert abs(df["x2"].mean() - 2.0) < 0.05, "x2 ~ N(2,1.5) — mean ≈ 2"
        assert abs(df["x2"].std() - 1.5) < 0.1

    def test_x3_uniform(self):
        df = generate_reference_data(n_rows=50_000, seed=3)
        assert df["x3"].min() >= -1.0
        assert df["x3"].max() <= 1.0
        assert abs(df["x3"].mean()) < 0.05, "x3 ~ U(-1,1) — mean ≈ 0"

    def test_x4_bernoulli(self):
        df = generate_reference_data(n_rows=50_000, seed=4)
        assert set(df["x4"].unique()).issubset({0.0, 1.0}), "x4 est binaire"
        assert abs(df["x4"].mean() - 0.4) < 0.02

    def test_reproducibility(self):
        df1 = generate_reference_data(n_rows=100, seed=42)
        df2 = generate_reference_data(n_rows=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_reference_data(n_rows=100, seed=1)
        df2 = generate_reference_data(n_rows=100, seed=2)
        assert not df1.equals(df2), "des graines différentes doivent produire des données différentes"

    def test_target_linear_relationship(self):
        """Vérifie que y est bien une combinaison linéaire des features."""
        df = generate_reference_data(n_rows=100_000, seed=7)
        corr = df.corr()["y"]
        # x1 a le plus grand coefficient positif (3.0)
        assert corr["x1"] > 0.4
        # x2 a un coefficient négatif (-1.5)
        assert corr["x2"] < -0.2


class TestGenerateDrifted:
    def test_x2_mean_higher(self):
        ref = generate_reference_data(n_rows=50_000, seed=0)
        dri = generate_drifted_data(n_rows=50_000, seed=0)
        assert dri["x2"].mean() > ref["x2"].mean() + 1.5, "x2 drifted mean doit être >> référence"

    def test_x4_probability_higher(self):
        ref = generate_reference_data(n_rows=50_000, seed=0)
        dri = generate_drifted_data(n_rows=50_000, seed=0)
        assert dri["x4"].mean() > ref["x4"].mean() + 0.2

    def test_x1_x3_unchanged(self):
        ref = generate_reference_data(n_rows=50_000, seed=0)
        dri = generate_drifted_data(n_rows=50_000, seed=0)
        assert abs(ref["x1"].mean() - dri["x1"].mean()) < 0.1, "x1 ne doit pas drifter"
        assert abs(ref["x3"].mean() - dri["x3"].mean()) < 0.1, "x3 ne doit pas drifter"


class TestPartialDrift:
    def test_factor_0_equals_reference(self):
        ref = generate_reference_data(n_rows=50_000, seed=5)
        par = generate_partial_drift(drift_factor=0.0, n_rows=50_000, seed=5)
        assert abs(ref["x2"].mean() - par["x2"].mean()) < 0.1

    def test_factor_1_equals_full_drift(self):
        dri = generate_drifted_data(n_rows=50_000, seed=5)
        par = generate_partial_drift(drift_factor=1.0, n_rows=50_000, seed=5)
        assert abs(dri["x2"].mean() - par["x2"].mean()) < 0.2

    def test_monotone_x2_drift(self):
        means = [
            generate_partial_drift(drift_factor=f, n_rows=10_000, seed=9)["x2"].mean()
            for f in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        assert all(means[i] < means[i+1] for i in range(len(means)-1)), \
            "x2 mean doit augmenter monotonement avec drift_factor"


class TestFeatureStats:
    def test_output_columns(self):
        df = generate_reference_data(n_rows=1000, seed=0)
        stats = compute_feature_stats(df)
        assert set(stats.columns) == {"mean", "std", "min", "q25", "median", "q75", "max"}

    def test_index_contains_features(self):
        df = generate_reference_data(n_rows=1000, seed=0)
        stats = compute_feature_stats(df)
        assert set(stats.index) == {"x1", "x2", "x3", "x4"}

    def test_no_target_in_stats(self):
        df = generate_reference_data(n_rows=1000, seed=0)
        stats = compute_feature_stats(df)
        assert "y" not in stats.index

    def test_q25_lt_median_lt_q75(self):
        df = generate_reference_data(n_rows=10_000, seed=0)
        stats = compute_feature_stats(df)
        for feat in stats.index:
            assert stats.loc[feat, "q25"] <= stats.loc[feat, "median"] <= stats.loc[feat, "q75"]


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        df = generate_reference_data(n_rows=200, seed=0)
        path = save_dataset(df, output_dir=str(tmp_path), prefix="test")
        df_loaded = load_dataset(str(path))
        pd.testing.assert_frame_equal(df, df_loaded)

    def test_file_created(self, tmp_path):
        df = generate_reference_data(n_rows=50, seed=0)
        path = save_dataset(df, output_dir=str(tmp_path), prefix="check")
        assert path.exists()
        assert path.suffix == ".parquet"


# ══════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════

class TestTrainFunction:
    @pytest.fixture
    def sample_df(self):
        return generate_reference_data(n_rows=2000, seed=42)

    def test_returns_model_and_metrics(self, sample_df):
        model, metrics, X_val, y_val = train(sample_df, model_type="random_forest")
        assert model is not None
        assert isinstance(metrics, dict)
        assert isinstance(X_val, pd.DataFrame)

    def test_metrics_keys(self, sample_df):
        _, metrics, _, _ = train(sample_df)
        required = {"val_rmse", "val_mae", "val_r2", "train_rmse", "n_rows_total", "train_duration_sec"}
        assert required.issubset(set(metrics.keys()))

    def test_r2_positive(self, sample_df):
        _, metrics, _, _ = train(sample_df)
        assert metrics["val_r2"] > 0.5, "R² doit être > 0.5 sur données synthétiques linéaires"

    def test_rmse_reasonable(self, sample_df):
        _, metrics, _, _ = train(sample_df)
        assert metrics["val_rmse"] < 2.0, "RMSE doit être < 2 sur données à bruit std=0.5"

    def test_ridge_model(self, sample_df):
        model, metrics, _, _ = train(sample_df, model_type="ridge")
        assert metrics["val_r2"] > 0.4

    def test_gradient_boosting(self, sample_df):
        model, metrics, _, _ = train(sample_df, model_type="gradient_boosting")
        assert metrics["val_r2"] > 0.5

    def test_invalid_model_type(self, sample_df):
        with pytest.raises(ValueError, match="inconnu"):
            train(sample_df, model_type="unknown_model")

    def test_custom_params(self, sample_df):
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 0, "n_jobs": 1}
        model, _, _, _ = train(sample_df, model_type="random_forest", model_params=params)
        assert model.n_estimators == 10

    def test_train_val_sizes(self, sample_df):
        _, metrics, X_val, _ = train(sample_df)
        expected_val = int(2000 * 0.2)
        assert abs(len(X_val) - expected_val) <= 1  # tolérance arrondi

    def test_feature_subset(self, sample_df):
        _, metrics, X_val, _ = train(sample_df, features=["x1", "x2"])
        assert metrics["n_features"] == 2
        assert list(X_val.columns) == ["x1", "x2"]


class TestEvaluateModel:
    def test_perfect_prediction(self):
        """Un modèle parfait donne R²=1, RMSE=0."""
        class PerfectModel:
            def predict(self, X): return y.values

        X = pd.DataFrame({"x1": np.arange(100)})
        y = pd.Series(np.arange(100) * 2.0)
        m = PerfectModel()
        metrics = evaluate_model(m, X, y)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_metrics_types(self):
        df = generate_reference_data(n_rows=500, seed=0)
        model, _, X_val, y_val = train(df)
        metrics = evaluate_model(model, X_val, y_val)
        for v in metrics.values():
            assert isinstance(v, float)
