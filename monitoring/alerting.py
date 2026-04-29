"""
Module de décision et d'alerting pour le drift monitoring.

RÔLE DE CE MODULE
─────────────────
Ce module répond à UNE question : que fait-on face aux scores de drift ?

Il y a deux niveaux de décision :

  Niveau 1 — Verdict par feature
    Pour chaque feature individuellement :
    "est-ce que cette feature a drifté ?"
    → basé sur PSI et/ou KS p-value

  Niveau 2 — Verdict global + décision de ré-entraînement
    Pour l'ensemble du run de monitoring :
    "est-ce que le modèle est en danger ?"
    "faut-il ré-entraîner maintenant ?"
    → basé sur le verdict global + historique des fenêtres précédentes

RÈGLES DE DÉCISION (configurables dans settings.py)
─────────────────────────────────────────────────────
  Alerte 'warning'  si :
    - PSI global > 0.10 (seuil_warning)
    - OU au moins 1 feature avec PSI > 0.10

  Alerte 'critical' si :
    - PSI global > 0.20 (seuil_critical)
    - OU >= 2 features avec PSI > 0.25

  Ré-entraînement si :
    - Alerte 'critical' pendant >= 3 fenêtres consécutives

POURQUOI ATTENDRE 3 FENÊTRES CONSÉCUTIVES ?
────────────────────────────────────────────
Un spike isolé peut être du bruit (anomalie ponctuelle, erreur de collecte).
Une tendance persistante sur 3 fenêtres est un vrai drift structurel.
Ce délai évite les ré-entraînements inutiles qui coûtent cher en compute.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config.settings import MONITOR_CFG, MonitoringConfig

logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class FeatureVerdict:
    """
    Verdict pour une feature individuelle.

    Regroupe tous les scores calculés par drift_metrics.py
    et ajoute le verdict final (drift ou pas).
    """
    feature_name: str

    # Scores bruts (issus de drift_metrics.py)
    psi: float
    ks_stat: float
    ks_pvalue: float
    mean_delta: float
    mean_delta_pct: float
    std_delta: float

    # Stats courantes (pour le rapport)
    current_mean: float
    current_std: float
    current_q25: float
    current_median: float
    current_q75: float

    # Verdict
    drift_flag: bool = False
    drift_reasons: List[str] = field(default_factory=list)


@dataclass
class MonitoringVerdict:
    """
    Verdict global d'un run de monitoring.

    Agrège tous les verdicts par feature et prend la décision finale.
    C'est cet objet qui est persisté dans drift_monitoring_runs.
    """
    # Verdicts détaillés par feature
    feature_verdicts: List[FeatureVerdict]

    # Agrégats globaux
    drift_score_global: float        # PSI max ou moyen selon config
    drift_detected: bool             # Au moins une feature en drift
    n_features_drifted: int          # Nombre de features en drift

    # Niveau d'alerte : 'none' | 'warning' | 'critical'
    alert_level: str

    # Décision finale
    should_retrain: bool             # Déclencher le ré-entraînement ?
    retrain_reason: Optional[str]    # Explication lisible de la décision

    # Résumé pour les logs
    summary: str = ""


# ── Verdict par feature ───────────────────────────────────────────────────────

def evaluate_feature(
    feature_name: str,
    drift_result: Dict,
    cfg: MonitoringConfig = MONITOR_CFG,
) -> FeatureVerdict:
    """
    Évalue si une feature individuelle est en drift.

    Args:
        feature_name : nom de la feature (ex: 'x2')
        drift_result : dict issu de drift_metrics.compute_feature_drift()
                       avec les clés psi, ks_stat, ks_pvalue, mean_delta, etc.
        cfg          : seuils de décision

    Returns:
        FeatureVerdict avec drift_flag=True si drift détecté

    Logique de décision :
      Une feature est en drift si :
        - PSI ≥ seuil_warning (0.10 par défaut)
        - OU KS p-value < 0.05 (distributions statistiquement différentes)
      Les deux critères sont indépendants.
      On note toutes les raisons pour le rapport.
    """
    psi = drift_result.get("psi", 0.0)
    ks_pvalue = drift_result.get("ks_pvalue", 1.0)
    ks_stat = drift_result.get("ks_stat", 0.0)
    mean_delta = drift_result.get("mean_delta", 0.0)
    mean_delta_pct = drift_result.get("mean_delta_pct", 0.0)
    std_delta = drift_result.get("std_delta", 0.0)

    drift_reasons = []

    # Critère 1 : PSI
    if psi >= cfg.psi_alert_threshold:
        label = "critique" if psi >= cfg.psi_critical_threshold else "modéré"
        drift_reasons.append(
            f"PSI={psi:.4f} ≥ {cfg.psi_alert_threshold} (drift {label})"
        )

    # Critère 2 : Test KS
    if ks_pvalue < 0.05:
        drift_reasons.append(
            f"KS p-value={ks_pvalue:.4f} < 0.05 (distributions différentes, stat={ks_stat:.4f})"
        )

    drift_flag = len(drift_reasons) > 0

    verdict = FeatureVerdict(
        feature_name=feature_name,
        psi=psi,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        mean_delta=mean_delta,
        mean_delta_pct=mean_delta_pct,
        std_delta=std_delta,
        current_mean=drift_result.get("current_mean", 0.0),
        current_std=drift_result.get("current_std", 0.0),
        current_q25=drift_result.get("current_q25", 0.0),
        current_median=drift_result.get("current_median", 0.0),
        current_q75=drift_result.get("current_q75", 0.0),
        drift_flag=drift_flag,
        drift_reasons=drift_reasons,
    )

    if drift_flag:
        logger.warning(
            "DRIFT détecté sur '%s' : %s",
            feature_name, " | ".join(drift_reasons)
        )
    else:
        logger.info("Feature '%s' stable (PSI=%.4f, KS_p=%.4f)", feature_name, psi, ks_pvalue)

    return verdict


# ── Verdict global ────────────────────────────────────────────────────────────

def evaluate_global(
    feature_verdicts: List[FeatureVerdict],
    cfg: MonitoringConfig = MONITOR_CFG,
) -> Tuple[float, str, bool]:
    """
    Calcule le score global et le niveau d'alerte à partir des verdicts par feature.

    Returns:
        (drift_score_global, alert_level, drift_detected)

    Règles :
      score_global = PSI maximum observé sur toutes les features
      (le max est plus conservateur que la moyenne : on détecte les cas extrêmes)

      alert_level = 'none'     si score_global < seuil_warning
      alert_level = 'warning'  si seuil_warning ≤ score_global < seuil_critical
      alert_level = 'critical' si score_global ≥ seuil_critical
                                OU si >= n_critical_features ont PSI > seuil_critical
    """
    if not feature_verdicts:
        return 0.0, "none", False

    # Score global = PSI maximum
    drift_score_global = max(v.psi for v in feature_verdicts)

    # Nombre de features avec PSI critique
    n_critical = sum(
        1 for v in feature_verdicts
        if v.psi >= cfg.psi_critical_threshold
    )

    # Déterminer le niveau d'alerte
    if (
        drift_score_global >= cfg.psi_alert_threshold
        or n_critical >= cfg.n_critical_features_for_alert
    ):
        if (
            drift_score_global >= cfg.psi_alert_threshold * 2   # 0.20 par défaut
            or n_critical >= cfg.n_critical_features_for_alert
        ):
            alert_level = "critical"
        else:
            alert_level = "warning"
        drift_detected = True
    else:
        alert_level = "none"
        drift_detected = False

    logger.info(
        "Verdict global : score=%.4f | alert=%s | %d/%d features en drift",
        drift_score_global, alert_level,
        sum(1 for v in feature_verdicts if v.drift_flag), len(feature_verdicts)
    )
    return drift_score_global, alert_level, drift_detected


# ── Point d'entrée principal ──────────────────────────────────────────────────

def build_monitoring_verdict(
    drift_results: Dict[str, Dict],
    consecutive_alerts: int = 0,
    cfg: MonitoringConfig = MONITOR_CFG,
) -> MonitoringVerdict:
    """
    Construit le verdict complet d'un run de monitoring.

    C'est LA fonction principale du module. Elle orchestre :
      1. L'évaluation de chaque feature individuellement
      2. Le calcul du score global et du niveau d'alerte
      3. La décision de ré-entraînement

    Args:
        drift_results       : Dict[feature_name → drift_metrics_result]
                              issu de drift_metrics.compute_all_features_drift()
        consecutive_alerts  : nombre d'alertes consécutives AVANT ce run
                              (lu depuis Postgres via postgres_client.count_consecutive_alerts)
        cfg                 : configuration des seuils

    Returns:
        MonitoringVerdict complet, prêt à être persisté dans Postgres

    Exemple d'usage :
        results = compute_all_features_drift(ref_df, current_df)
        n_consecutive = count_consecutive_alerts(model_version)
        verdict = build_monitoring_verdict(results, consecutive_alerts=n_consecutive)

        if verdict.should_retrain:
            trigger_retraining_dag()
    """
    # Étape 1 — verdict par feature
    feature_verdicts = [
        evaluate_feature(fname, fresult, cfg)
        for fname, fresult in drift_results.items()
    ]

    # Étape 2 — verdict global
    drift_score_global, alert_level, drift_detected = evaluate_global(feature_verdicts, cfg)
    n_features_drifted = sum(1 for v in feature_verdicts if v.drift_flag)

    # Étape 3 — décision de ré-entraînement
    # On compte cette alerte dans la séquence consécutive
    current_consecutive = consecutive_alerts + (1 if drift_detected else 0)

    should_retrain = False
    retrain_reason = None

    if (
        alert_level == "critical"
        and current_consecutive >= cfg.consecutive_windows_for_retrain
    ):
        should_retrain = True
        retrain_reason = (
            f"Alerte critique persistante depuis {current_consecutive} fenêtres "
            f"consécutives (seuil={cfg.consecutive_windows_for_retrain}). "
            f"Score global PSI={drift_score_global:.4f}. "
            f"Features en drift : {[v.feature_name for v in feature_verdicts if v.drift_flag]}"
        )
        logger.critical("RÉ-ENTRAÎNEMENT DÉCLENCHÉ : %s", retrain_reason)

    elif alert_level == "critical":
        logger.warning(
            "Alerte critique (%d/%d fenêtres consécutives nécessaires)",
            current_consecutive, cfg.consecutive_windows_for_retrain
        )

    elif alert_level == "warning":
        logger.warning(
            "Alerte warning : drift modéré détecté (PSI global=%.4f)",
            drift_score_global
        )

    # Résumé lisible
    drifted_names = [v.feature_name for v in feature_verdicts if v.drift_flag]
    summary = _build_summary(
        alert_level, drift_score_global, n_features_drifted,
        drifted_names, current_consecutive, should_retrain, cfg
    )
    logger.info("RÉSUMÉ MONITORING :\n%s", summary)

    return MonitoringVerdict(
        feature_verdicts=feature_verdicts,
        drift_score_global=drift_score_global,
        drift_detected=drift_detected,
        n_features_drifted=n_features_drifted,
        alert_level=alert_level,
        should_retrain=should_retrain,
        retrain_reason=retrain_reason,
        summary=summary,
    )


# ── Formatage du rapport ──────────────────────────────────────────────────────

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
        f"  Seuil critical       : {cfg.psi_alert_threshold * 2}",
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


def format_feature_report(verdicts: List[FeatureVerdict]) -> str:
    """
    Formate un rapport détaillé feature par feature.

    Utile pour les logs Airflow et les notifications Slack/email.
    """
    lines = ["\n  DÉTAIL PAR FEATURE", "  " + "-" * 50]

    for v in sorted(verdicts, key=lambda x: x.psi, reverse=True):
        status = "🔴 DRIFT" if v.drift_flag else "✅ stable"
        lines.append(
            f"  {v.feature_name:<6} | {status:<12} | "
            f"PSI={v.psi:.4f} | KS_p={v.ks_pvalue:.4f} | "
            f"Δmean={v.mean_delta:+.3f} ({v.mean_delta_pct:+.1f}%)"
        )
        if v.drift_reasons:
            for reason in v.drift_reasons:
                lines.append(f"           → {reason}")

    return "\n".join(lines)


# ── Conversion vers format Postgres ──────────────────────────────────────────

def verdict_to_postgres_rows(verdict: MonitoringVerdict) -> List[Dict]:
    """
    Convertit les verdicts par feature en liste de dicts
    prêts pour postgres_client.insert_drift_feature_metrics().

    Évite que les autres modules aient à manipuler les dataclasses directement.
    """
    rows = []
    for v in verdict.feature_verdicts:
        rows.append({
            "feature_name": v.feature_name,
            "psi": v.psi,
            "ks_stat": v.ks_stat,
            "ks_pvalue": v.ks_pvalue,
            "mean_delta": v.mean_delta,
            "mean_delta_pct": v.mean_delta_pct,
            "std_delta": v.std_delta,
            "current_mean": v.current_mean,
            "current_std": v.current_std,
            "current_q25": v.current_q25,
            "current_median": v.current_median,
            "current_q75": v.current_q75,
            "drift_flag": v.drift_flag,
        })
    return rows
