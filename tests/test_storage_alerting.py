"""
Tests pour les modules storage et monitoring/alerting.

STRATÉGIE DE TEST
─────────────────
- postgres_client et minio_client se connectent à des services externes.
  On ne peut pas les tester unitairement sans un vrai Postgres/MinIO.
  → On teste la logique pure (nommage, parsing, helpers) sans connexion.
  → Les tests d'intégration réels tournent en CI avec docker-compose.

- alerting.py est de la logique pure Python, sans dépendance externe.
  → Tests unitaires complets, couvrant tous les cas de décision.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ── alerting ──────────────────────────────────────────────────────────────────
from src.monitoring.alerting import (
    evaluate_feature,
    evaluate_global,
    build_monitoring_verdict,
    verdict_to_postgres_rows,
    FeatureVerdict,
    MonitoringVerdict,
)
from src.config.settings import MonitoringConfig


# Configuration de test avec des seuils explicites et prévisibles
TEST_CFG = MonitoringConfig(
    psi_alert_threshold=0.10,
    psi_critical_threshold=0.25,
    n_critical_features_for_alert=2,
    consecutive_windows_for_retrain=3,
    n_bins=10,
)


def _make_drift_result(psi=0.0, ks_stat=0.0, ks_pvalue=1.0,
                       mean_delta=0.0, mean_delta_pct=0.0, std_delta=0.0,
                       current_mean=0.0, current_std=1.0,
                       current_q25=-0.5, current_median=0.0, current_q75=0.5):
    """Fabrique un dict de résultat de drift (simule la sortie de drift_metrics)."""
    return {
        "psi": psi, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue,
        "mean_delta": mean_delta, "mean_delta_pct": mean_delta_pct,
        "std_delta": std_delta,
        "current_mean": current_mean, "current_std": current_std,
        "current_q25": current_q25, "current_median": current_median,
        "current_q75": current_q75,
    }


# ══════════════════════════════════════════════════════
# evaluate_feature
# ══════════════════════════════════════════════════════

class TestEvaluateFeature:

    def test_stable_feature_no_drift(self):
        result = _make_drift_result(psi=0.05, ks_pvalue=0.30)
        verdict = evaluate_feature("x1", result, TEST_CFG)
        assert verdict.drift_flag is False
        assert verdict.drift_reasons == []

    def test_psi_above_threshold_triggers_drift(self):
        result = _make_drift_result(psi=0.15, ks_pvalue=0.30)
        verdict = evaluate_feature("x2", result, TEST_CFG)
        assert verdict.drift_flag is True
        assert any("PSI" in r for r in verdict.drift_reasons)

    def test_ks_pvalue_below_005_triggers_drift(self):
        result = _make_drift_result(psi=0.05, ks_pvalue=0.03)
        verdict = evaluate_feature("x3", result, TEST_CFG)
        assert verdict.drift_flag is True
        assert any("KS" in r for r in verdict.drift_reasons)

    def test_both_criteria_gives_two_reasons(self):
        result = _make_drift_result(psi=0.20, ks_pvalue=0.01)
        verdict = evaluate_feature("x2", result, TEST_CFG)
        assert verdict.drift_flag is True
        assert len(verdict.drift_reasons) == 2

    def test_psi_exactly_at_threshold_triggers(self):
        result = _make_drift_result(psi=0.10, ks_pvalue=1.0)
        verdict = evaluate_feature("x1", result, TEST_CFG)
        assert verdict.drift_flag is True

    def test_psi_just_below_threshold_stable(self):
        result = _make_drift_result(psi=0.099, ks_pvalue=1.0)
        verdict = evaluate_feature("x1", result, TEST_CFG)
        assert verdict.drift_flag is False

    def test_ks_exactly_at_005_is_stable(self):
        # p-value strictement > 0.05 → stable
        result = _make_drift_result(psi=0.05, ks_pvalue=0.05)
        verdict = evaluate_feature("x1", result, TEST_CFG)
        # 0.05 n'est PAS < 0.05 → pas de drift KS
        assert not any("KS" in r for r in verdict.drift_reasons)

    def test_feature_name_preserved(self):
        result = _make_drift_result()
        verdict = evaluate_feature("x4", result, TEST_CFG)
        assert verdict.feature_name == "x4"

    def test_scores_preserved_in_verdict(self):
        result = _make_drift_result(psi=0.18, ks_stat=0.12, ks_pvalue=0.04,
                                    mean_delta=1.5, mean_delta_pct=75.0)
        verdict = evaluate_feature("x2", result, TEST_CFG)
        assert verdict.psi == pytest.approx(0.18)
        assert verdict.ks_stat == pytest.approx(0.12)
        assert verdict.mean_delta == pytest.approx(1.5)
        assert verdict.mean_delta_pct == pytest.approx(75.0)


# ══════════════════════════════════════════════════════
# evaluate_global
# ══════════════════════════════════════════════════════

class TestEvaluateGlobal:

    def _make_verdicts(self, psi_values):
        """Crée une liste de FeatureVerdict avec les PSI donnés."""
        verdicts = []
        for i, psi in enumerate(psi_values):
            verdicts.append(FeatureVerdict(
                feature_name=f"x{i+1}",
                psi=psi, ks_stat=0.0, ks_pvalue=1.0,
                mean_delta=0.0, mean_delta_pct=0.0, std_delta=0.0,
                current_mean=0.0, current_std=1.0,
                current_q25=0.0, current_median=0.0, current_q75=0.0,
                drift_flag=(psi >= TEST_CFG.psi_alert_threshold),
            ))
        return verdicts

    def test_all_stable_returns_none(self):
        verdicts = self._make_verdicts([0.02, 0.03, 0.05, 0.01])
        score, alert, detected = evaluate_global(verdicts, TEST_CFG)
        assert alert == "none"
        assert detected is False

    def test_one_warning_psi_gives_warning(self):
        verdicts = self._make_verdicts([0.02, 0.15, 0.03, 0.01])
        score, alert, detected = evaluate_global(verdicts, TEST_CFG)
        assert alert == "warning"
        assert detected is True
        assert score == pytest.approx(0.15)

    def test_critical_psi_gives_critical(self):
        verdicts = self._make_verdicts([0.02, 0.35, 0.03, 0.01])
        score, alert, detected = evaluate_global(verdicts, TEST_CFG)
        assert alert == "critical"
        assert detected is True

    def test_two_critical_features_gives_critical(self):
        # Deux features dépassent le seuil critique (0.25)
        verdicts = self._make_verdicts([0.02, 0.26, 0.28, 0.01])
        score, alert, detected = evaluate_global(verdicts, TEST_CFG)
        assert alert == "critical"

    def test_score_is_max_psi(self):
        verdicts = self._make_verdicts([0.05, 0.18, 0.12, 0.30])
        score, _, _ = evaluate_global(verdicts, TEST_CFG)
        assert score == pytest.approx(0.30)

    def test_empty_list_returns_zero(self):
        score, alert, detected = evaluate_global([], TEST_CFG)
        assert score == 0.0
        assert alert == "none"
        assert detected is False


# ══════════════════════════════════════════════════════
# build_monitoring_verdict
# ══════════════════════════════════════════════════════

class TestBuildMonitoringVerdict:

    def _stable_drift_results(self):
        return {f"x{i}": _make_drift_result(psi=0.02, ks_pvalue=0.50) for i in range(1, 5)}

    def _warning_drift_results(self):
        results = {f"x{i}": _make_drift_result(psi=0.02) for i in range(1, 5)}
        results["x2"] = _make_drift_result(psi=0.15, ks_pvalue=0.60)
        return results

    def _critical_drift_results(self):
        results = {f"x{i}": _make_drift_result(psi=0.02) for i in range(1, 5)}
        results["x2"] = _make_drift_result(psi=0.30, ks_pvalue=0.01)
        results["x4"] = _make_drift_result(psi=0.28, ks_pvalue=0.02)
        return results

    def test_stable_no_drift_no_retrain(self):
        verdict = build_monitoring_verdict(self._stable_drift_results(), 0, TEST_CFG)
        assert verdict.alert_level == "none"
        assert verdict.drift_detected is False
        assert verdict.should_retrain is False

    def test_warning_drift_detected_no_retrain(self):
        verdict = build_monitoring_verdict(self._warning_drift_results(), 0, TEST_CFG)
        assert verdict.drift_detected is True
        assert verdict.should_retrain is False

    def test_critical_below_consecutive_threshold_no_retrain(self):
        # 1 alerte critique consécutive (seuil = 3) → pas de ré-entraînement
        verdict = build_monitoring_verdict(self._critical_drift_results(), 0, TEST_CFG)
        assert verdict.alert_level == "critical"
        assert verdict.should_retrain is False

    def test_critical_at_consecutive_threshold_triggers_retrain(self):
        # 2 alertes avant + cette alerte = 3 → ré-entraînement
        verdict = build_monitoring_verdict(self._critical_drift_results(), 2, TEST_CFG)
        assert verdict.should_retrain is True
        assert verdict.retrain_reason is not None
        assert "3" in verdict.retrain_reason   # mentionne le nombre de fenêtres

    def test_critical_exceeds_threshold_triggers_retrain(self):
        # 5 alertes avant → déclenche aussi
        verdict = build_monitoring_verdict(self._critical_drift_results(), 5, TEST_CFG)
        assert verdict.should_retrain is True

    def test_warning_never_triggers_retrain(self):
        # Même avec 10 alertes consécutives, un warning ne déclenche pas
        verdict = build_monitoring_verdict(self._warning_drift_results(), 10, TEST_CFG)
        assert verdict.should_retrain is False

    def test_n_features_drifted_count(self):
        verdict = build_monitoring_verdict(self._critical_drift_results(), 0, TEST_CFG)
        assert verdict.n_features_drifted == 2

    def test_feature_verdicts_all_present(self):
        results = self._stable_drift_results()
        verdict = build_monitoring_verdict(results, 0, TEST_CFG)
        assert len(verdict.feature_verdicts) == 4

    def test_summary_not_empty(self):
        verdict = build_monitoring_verdict(self._critical_drift_results(), 0, TEST_CFG)
        assert len(verdict.summary) > 0
        assert "critical" in verdict.summary.lower() or "CRITICAL" in verdict.summary


# ══════════════════════════════════════════════════════
# verdict_to_postgres_rows
# ══════════════════════════════════════════════════════

class TestVerdictToPostgresRows:

    def test_returns_one_row_per_feature(self):
        results = {f"x{i}": _make_drift_result() for i in range(1, 5)}
        verdict = build_monitoring_verdict(results, 0, TEST_CFG)
        rows = verdict_to_postgres_rows(verdict)
        assert len(rows) == 4

    def test_row_contains_required_keys(self):
        results = {"x1": _make_drift_result(psi=0.15, ks_pvalue=0.03)}
        verdict = build_monitoring_verdict(results, 0, TEST_CFG)
        rows = verdict_to_postgres_rows(verdict)
        required = {
            "feature_name", "psi", "ks_stat", "ks_pvalue",
            "mean_delta", "mean_delta_pct", "std_delta",
            "current_mean", "current_std",
            "current_q25", "current_median", "current_q75",
            "drift_flag",
        }
        assert required.issubset(set(rows[0].keys()))

    def test_drift_flag_preserved(self):
        results = {
            "x1": _make_drift_result(psi=0.02),   # stable
            "x2": _make_drift_result(psi=0.25),   # drift
        }
        verdict = build_monitoring_verdict(results, 0, TEST_CFG)
        rows = verdict_to_postgres_rows(verdict)
        by_feature = {r["feature_name"]: r for r in rows}
        assert by_feature["x1"]["drift_flag"] is False
        assert by_feature["x2"]["drift_flag"] is True


# ══════════════════════════════════════════════════════
# storage/minio_client — logique pure (sans connexion)
# ══════════════════════════════════════════════════════

class TestMinioNamingHelpers:
    """
    On teste uniquement les fonctions de nommage qui ne font pas d'appel réseau.
    Les tests d'intégration réels nécessitent un MinIO local.
    """
    from src.storage.minio_client import (
        make_dataset_object_name,
        make_snapshot_object_name,
        make_stats_object_name,
        download_dataframe_from_uri,
    )

    def test_dataset_name_ends_with_parquet(self):
        from src.storage.minio_client import make_dataset_object_name
        name = make_dataset_object_name("reference", "v1")
        assert name.endswith(".parquet")

    def test_dataset_name_contains_prefix_and_version(self):
        from src.storage.minio_client import make_dataset_object_name
        name = make_dataset_object_name("reference", "v1")
        assert "reference" in name
        assert "v1" in name

    def test_snapshot_name_contains_date(self):
        from src.storage.minio_client import make_snapshot_object_name
        start = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        name = make_snapshot_object_name(start, end)
        assert "2024-01-15" in name
        assert name.endswith(".parquet")

    def test_snapshot_name_contains_hours(self):
        from src.storage.minio_client import make_snapshot_object_name
        start = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc)
        name = make_snapshot_object_name(start, end)
        assert "14h30" in name
        assert "15h30" in name

    def test_stats_name_contains_run_id_prefix(self):
        from src.storage.minio_client import make_stats_object_name
        run_id = "abcdef1234567890"
        name = make_stats_object_name(run_id)
        assert "abcdef12" in name   # 8 premiers caractères
        assert name.endswith(".json")

    def test_download_from_uri_invalid_format_raises(self):
        from src.storage.minio_client import download_dataframe_from_uri
        with pytest.raises(ValueError, match="s3://"):
            download_dataframe_from_uri("http://wrong/path")

    def test_download_from_uri_missing_object_raises(self):
        from src.storage.minio_client import download_dataframe_from_uri
        with pytest.raises(ValueError, match="mal formée"):
            download_dataframe_from_uri("s3://bucket-only")
