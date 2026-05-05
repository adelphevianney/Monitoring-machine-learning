"""
Module de décision et d'alerting pour le drift monitoring.

RÔLE DE CE MODULE
─────────────────
Ce module répond à UNE question : que fait-on face aux scores de drift ?

Il travaille UNIQUEMENT avec les types de drift_metrics.py.
Il n'y a plus de structures parallèles (FeatureVerdict, etc.) :
  - drift_metrics.py calcule → retourne Dict[str, FeatureDriftResult]
  - alerting.py décide       → consomme les FeatureDriftResult directement
  - Le résultat est un MonitoringVerdict prêt pour Postgres

RÈGLES DE DÉCISION (configurables dans settings.py)
─────────────────────────────────────────────────────
  Alerte 'warning'  si :
    - PSI moyen > psi_alert_threshold (0.10)
    - OU >= n_critical_features_for_alert features avec drift_flag=True

  Alerte 'critical' si :
    - Au moins une feature avec PSI > psi_critical_threshold (0.25)

  Ré-entraînement si :
    - Alerte 'critical' pendant >= consecutive_windows_for_retrain (3)
      fenêtres consécutives

POURQUOI ATTENDRE 3 FENÊTRES CONSÉCUTIVES ?
────────────────────────────────────────────
Un spike isolé peut être du bruit (anomalie ponctuelle, erreur de collecte).
Une tendance persistante sur 3 fenêtres est un vrai drift structurel.
Ce délai évite les ré-entraînements inutiles qui coûtent cher en compute.

CONTRAT AVEC drift_metrics.py
──────────────────────────────
Ce module importe et utilise FeatureDriftResult directement.
Aucune conversion intermédiaire en Dict n'est nécessaire.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config.settings import MONITOR_CFG, MonitoringConfig
from src.monitoring.drift_metrics import FeatureDriftResult

logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class MonitoringVerdict:
    """
    Verdict global d'un run de monitoring.

    Agrège tous les FeatureDriftResult de drift_metrics.py
    et prend la décision finale (alerte + ré-entraînement).
    C'est cet objet qui est persisté dans drift_monitoring_runs.
    """
    # Résultats détaillés par feature (issus de drift_metrics.py)
    feature_results: Dict[str, FeatureDriftResult]

    # Agrégats globaux
    drift_score_global: float   # PSI moyen sur toutes les features
    drift_detected: bool        # Au moins une feature en drift
    n_features_drifted: int     # Nombre de features en drift

    # Niveau d'alerte : 'none' | 'warning' | 'critical'
    alert_level: str

    # Décision finale
    should_retrain: bool             # Déclencher le ré-entraînement ?
    retrain_reason: Optional[str]    # Explication lisible de la décision

    # Résumé pour les logs
    summary: str = ""


# ── Verdict global ────────────────────────────────────────────────────────────

def build_monitoring_verdict(
    feature_results: Dict[str, FeatureDriftResult],
    consecutive_alerts: int = 0,
    cfg: MonitoringConfig = MONITOR_CFG,
) -> MonitoringVerdict:
    """
    Construit le verdict complet d'un run de monitoring.

    C'est LA fonction principale du module. Elle orchestre :
      1. La lecture des FeatureDriftResult produits par drift_metrics.py
      2. Le calcul du score global et du niveau d'alerte
      3. La décision de ré-entraînement

    Args:
        feature_results     : Dict[feature_name → FeatureDriftResult]
                              issu de drift_metrics.compute_drift_report()
        consecutive_alerts  : nombre d'alertes consécutives AVANT ce run
                              (lu depuis Postgres via postgres_client.count_consecutive_alerts)
        cfg                 : configuration des seuils

    Returns:
        MonitoringVerdict complet, prêt à être persisté dans Postgres

    Exemple d'usage :
        feature_results = compute_drift_report(ref_df, current_df, features)
        n_consecutive = count_consecutive_alerts(ref_run_id)
        verdict = build_monitoring_verdict(feature_results, n_consecutive)

        if verdict.should_retrain:
            trigger_retraining_dag()
    """
    if not feature_results:
        return MonitoringVerdict(
            feature_results={},
            drift_score_global=0.0,
            drift_detected=False,
            n_features_drifted=0,
            alert_level="none",
            should_retrain=False,
            retrain_reason=None,
            summary="Aucune feature à analyser.",
        )

    # ── Score global et niveau d'alerte ──────────────────────────────────────
    drift_score_global, alert_level, drift_detected = _compute_alert_level(
        feature_results, cfg
    )
    n_features_drifted = sum(1 for r in feature_results.values() if r.drift_flag)

    # ── Décision de ré-entraînement ───────────────────────────────────────────
    # On compte cette alerte dans la séquence consécutive
    current_consecutive = consecutive_alerts + (1 if drift_detected else 0)

    should_retrain = False
    retrain_reason = None

    if (
        alert_level == "critical"
        and current_consecutive >= cfg.consecutive_windows_for_retrain
    ):
        should_retrain = True
        drifted_names = [n for n, r in feature_results.items() if r.drift_flag]
        retrain_reason = (
            f"Alerte critique persistante depuis {current_consecutive} fenêtres "
            f"consécutives (seuil={cfg.consecutive_windows_for_retrain}). "
            f"Score global PSI={drift_score_global:.4f}. "
            f"Features en drift : {drifted_names}"
        )
        logger.critical("RÉ-ENTRAÎNEMENT DÉCLENCHÉ : %s", retrain_reason)

    elif alert_level == "critical":
        logger.warning(
            "Alerte critique (%d/%d fenêtres consécutives nécessaires)",
            current_consecutive, cfg.consecutive_windows_for_retrain,
        )
    elif alert_level == "warning":
        logger.warning(
            "Alerte warning : drift modéré détecté (PSI global=%.4f)",
            drift_score_global,
        )
    else:
        logger.info("Monitoring OK — aucun drift détecté (PSI global=%.4f)", drift_score_global)

    # ── Résumé lisible ────────────────────────────────────────────────────────
    drifted_names = [n for n, r in feature_results.items() if r.drift_flag]
    summary = _build_summary(
        alert_level, drift_score_global, n_features_drifted,
        drifted_names, current_consecutive, should_retrain, cfg,
    )
    logger.info("RÉSUMÉ MONITORING :\n%s", summary)

    return MonitoringVerdict(
        feature_results=feature_results,
        drift_score_global=drift_score_global,
        drift_detected=drift_detected,
        n_features_drifted=n_features_drifted,
        alert_level=alert_level,
        should_retrain=should_retrain,
        retrain_reason=retrain_reason,
        summary=summary,
    )


def _compute_alert_level(
    feature_results: Dict[str, FeatureDriftResult],
    cfg: MonitoringConfig,
) -> Tuple[float, str, bool]:
    """
    Calcule le score global et le niveau d'alerte.

    Score global = moyenne des PSI (simple et interprétable).

    Règles :
      "critical" si au moins une feature PSI > psi_critical_threshold
      "warning"  si score_global > psi_alert_threshold
                 OU >= n_critical_features_for_alert features en drift
      "none"     sinon
    """
    import numpy as np

    psi_values = [r.psi for r in feature_results.values()]
    drift_score_global = float(np.mean(psi_values))
    n_in_drift = sum(1 for r in feature_results.values() if r.drift_flag)

    has_critical_feature = any(
        r.psi > cfg.psi_critical_threshold for r in feature_results.values()
    )

    if has_critical_feature:
        alert_level = "critical"
    elif (
        drift_score_global > cfg.psi_alert_threshold
        or n_in_drift >= cfg.n_critical_features_for_alert
    ):
        alert_level = "warning"
    else:
        alert_level = "none"

    drift_detected = alert_level != "none"

    logger.info(
        "Verdict global : score=%.4f | alert=%s | %d/%d features en drift",
        drift_score_global, alert_level, n_in_drift, len(feature_results),
    )
    return drift_score_global, alert_level, drift_detected


# ── Formatage ─────────────────────────────────────────────────────────────────

def _build_summary(
    alert_level: str,
    drift_score_global: float,
    n_features_drifted: int,
    drifted_names: List[str],
    consecutive: int,
    should_retrain: bool,
    cfg: MonitoringConfig,
) -> str:
    """Génère un résumé lisible du verdict pour les logs et notifications."""
    icon = {"none": "✅", "warning": "⚠️", "critical": "🔴"}.get(alert_level, "❓")

    lines = [
        f"{'='*55}",
        f"  DRIFT MONITORING — {icon} {alert_level.upper()}",
        f"{'='*55}",
        f"  Score PSI global     : {drift_score_global:.4f}",
        f"  Seuil warning        : {cfg.psi_alert_threshold}",
        f"  Seuil critical       : {cfg.psi_critical_threshold}",
        f"  Features en drift    : {n_features_drifted}",
    ]

    if drifted_names:
        lines.append(f"  Features concernées  : {', '.join(drifted_names)}")

    lines += [
        f"  Fenêtres consécutives: {consecutive}/{cfg.consecutive_windows_for_retrain}",
        f"  Ré-entraînement      : {'OUI 🚀' if should_retrain else 'non'}",
        f"{'='*55}",
    ]

    return "\n".join(lines)


def format_feature_report(feature_results: Dict[str, FeatureDriftResult]) -> str:
    """
    Formate un rapport détaillé feature par feature.

    Utile pour les logs Airflow et les notifications Slack/email.
    """
    lines = ["\n  DÉTAIL PAR FEATURE", "  " + "-" * 50]

    for name, r in sorted(feature_results.items(), key=lambda x: x[1].psi, reverse=True):
        status = "🔴 DRIFT" if r.drift_flag else "✅ stable"
        lines.append(
            f"  {name:<6} | {status:<12} | "
            f"PSI={r.psi:.4f} | KS_p={r.ks_pvalue:.4f} | "
            f"Δmean={r.mean_delta:+.3f} ({r.mean_delta_pct:+.1f}%)"
        )
        if r.drift_reason and r.drift_reason != "stable":
            lines.append(f"           → {r.drift_reason}")

    return "\n".join(lines)


# ── Conversion vers format Postgres ──────────────────────────────────────────

def verdict_to_postgres_rows(verdict: MonitoringVerdict) -> List[Dict]:
    """
    Convertit les résultats par feature du verdict en liste de dicts
    prêts pour postgres_client.insert_drift_feature_metrics().

    Délègue à drift_metrics.feature_results_to_rows() pour
    centraliser la sérialisation.
    """
    from src.monitoring.drift_metrics import feature_results_to_rows
    return feature_results_to_rows(verdict.feature_results)