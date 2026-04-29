"""
Calcul des métriques de drift entre deux distributions.

MÉTRIQUES IMPLÉMENTÉES
──────────────────────
1. PSI (Population Stability Index)
   - Découpe la référence en N buckets de quantiles égaux
   - Mesure la divergence entre les proportions référence vs courantes
   - Formule : PSI = Σ (p_ref - p_cur) * ln(p_ref / p_cur)
   - Seuils standards :
       PSI < 0.10  → distribution stable
       PSI 0.10–0.20 → changement modéré, surveiller
       PSI > 0.20  → drift significatif, alerter

2. Test KS (Kolmogorov-Smirnov)
   - Mesure la distance maximale entre deux CDF (fonctions de répartition cumulée)
   - Produit une stat et une p-value
   - p-value < 0.05 → les deux distributions sont significativement différentes

3. Delta mean / std
   - Comparaison directe des statistiques descriptives
   - Utile pour comprendre dans quel sens la distribution a dérivé

POURQUOI CES DEUX MÉTRIQUES ENSEMBLE ?
──────────────────────────────────────
Le PSI est bon pour détecter des shifts globaux (toute la distribution se déplace).
Le KS est bon pour détecter des changements locaux (une queue de distribution qui grossit).
Ensemble, ils couvrent la plupart des patterns de drift réels.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import MONITOR_CFG, MonitoringConfig

logger = logging.getLogger(__name__)


# ── structures de données ─────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    """
    Résultat du calcul de drift pour une feature individuelle.

    Attributs :
        feature_name : nom de la colonne
        psi          : Population Stability Index
        ks_stat      : statistique du test KS (distance max entre les CDF)
        ks_pvalue    : p-value du test KS
        mean_ref     : moyenne sur la distribution de référence
        mean_cur     : moyenne sur la distribution courante
        mean_delta   : différence absolue des moyennes
        std_ref      : écart-type de référence
        std_cur      : écart-type courant
        std_delta    : différence absolue des écarts-types
        drift_flag   : True si la feature est considérée en drift
        drift_reason : explication textuelle du déclenchement
    """
    feature_name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    mean_ref: float
    mean_cur: float
    mean_delta: float
    std_ref: float
    std_cur: float
    std_delta: float
    drift_flag: bool
    drift_reason: str = ""


@dataclass
class MonitoringRunResult:
    """
    Résultat complet d'un run de monitoring.

    Contient les résultats par feature et un score global agrégé.
    """
    monitoring_run_id: str
    model_version: str
    observation_start: str
    observation_end: str
    input_dataset_uri: str
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    drift_score_global: float = 0.0
    drift_detected: bool = False
    alert_level: str = "none"   # "none" | "warning" | "critical"
    n_features_in_drift: int = 0
    created_at: str = ""


# ── calcul PSI ────────────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calcule le PSI entre une distribution de référence et une distribution courante.

    COMMENT ÇA MARCHE :
      1. On découpe la référence en n_bins buckets de quantiles égaux
         (pas des intervalles fixes — des quantiles, donc chaque bucket
          contient ~10% des données de référence)
      2. On calcule la proportion de données dans chaque bucket
         pour la référence (p_ref) et pour le courant (p_cur)
      3. PSI = Σ (p_ref - p_cur) * ln(p_ref / p_cur)

    POURQUOI DES QUANTILES et pas des intervalles fixes ?
      - Avec des intervalles fixes, certains buckets peuvent être vides
        (ex: si les données sont concentrées dans une petite plage)
      - Les quantiles garantissent une répartition équilibrée sur la référence

    Args:
        reference : tableau numpy de la distribution de référence
        current   : tableau numpy de la distribution courante
        n_bins    : nombre de buckets (10 par défaut = standard industrie)
        epsilon   : petite valeur pour éviter log(0) quand un bucket est vide

    Returns:
        (psi_total, proportions_ref, proportions_cur)
    """
    # Calcul des seuils de buckets sur la référence
    # np.linspace(0, 100, n_bins+1) donne [0, 10, 20, ..., 100] pour 10 bins
    quantile_points = np.linspace(0, 100, n_bins + 1)
    bucket_edges = np.percentile(reference, quantile_points)

    # Dédupliquer les bords (si la distribution est très discrète, certains quantiles
    # peuvent être identiques — ex: pour x4 binaire avec seulement 0 et 1)
    bucket_edges = np.unique(bucket_edges)

    # np.histogram compte combien de valeurs tombent dans chaque bucket
    ref_counts, _ = np.histogram(reference, bins=bucket_edges)
    cur_counts, _ = np.histogram(current, bins=bucket_edges)

    # Conversion en proportions (somme = 1)
    # On ajoute epsilon pour éviter la division par zéro
    ref_props = (ref_counts / len(reference)) + epsilon
    cur_props = (cur_counts / len(current)) + epsilon

    # Normalisation pour que la somme soit exactement 1 même après epsilon
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    # Formule PSI : divergence de Jensen-Shannon modifiée
    psi_per_bucket = (ref_props - cur_props) * np.log(ref_props / cur_props)
    psi_total = float(psi_per_bucket.sum())

    return psi_total, ref_props, cur_props


# ── calcul KS ─────────────────────────────────────────────────────────────────

def compute_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Applique le test de Kolmogorov-Smirnov à deux échantillons.

    COMMENT ÇA MARCHE :
      - On construit les deux fonctions de répartition empiriques (CDF)
      - On mesure la distance maximale D entre ces deux courbes
      - On calcule la p-value : sous H0 (mêmes distributions), quelle est
        la probabilité d'observer une distance ≥ D ?

    INTERPRÉTATION :
      - ks_stat proche de 0 → les deux distributions se ressemblent
      - ks_stat proche de 1 → les deux distributions sont très différentes
      - p_value < 0.05 → on rejette H0, les distributions diffèrent significativement

    Args:
        reference : tableau numpy de référence
        current   : tableau numpy courant

    Returns:
        (ks_statistic, p_value)
    """
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


# ── calcul par feature ────────────────────────────────────────────────────────

def compute_feature_drift(
    feature_name: str,
    reference: np.ndarray,
    current: np.ndarray,
    cfg: MonitoringConfig = MONITOR_CFG,
) -> FeatureDriftResult:
    """
    Calcule toutes les métriques de drift pour une feature donnée.

    La logique de drift_flag :
      Un flag est levé si :
        - PSI > psi_alert_threshold (0.2 par défaut)
        OU
        - KS p-value < 0.05 (rejet de H0 avec confiance 95%)

    Args:
        feature_name : nom de la feature (pour les logs)
        reference    : valeurs de la distribution de référence (numpy array)
        current      : valeurs de la distribution courante (numpy array)
        cfg          : configuration des seuils

    Returns:
        FeatureDriftResult avec toutes les métriques calculées
    """
    psi, _, _ = compute_psi(reference, current, n_bins=cfg.n_bins)
    ks_stat, ks_pvalue = compute_ks(reference, current)

    mean_ref = float(np.mean(reference))
    mean_cur = float(np.mean(current))
    std_ref = float(np.std(reference))
    std_cur = float(np.std(current))

    # Détermination du drift flag et de la raison
    reasons = []
    drift_flag = False

    if psi > cfg.psi_alert_threshold:
        drift_flag = True
        reasons.append(f"PSI={psi:.3f} > seuil {cfg.psi_alert_threshold}")

    if ks_pvalue < 0.05:
        drift_flag = True
        reasons.append(f"KS p-value={ks_pvalue:.4f} < 0.05")

    drift_reason = " | ".join(reasons) if reasons else "stable"

    logger.debug(
        "Feature '%s' : PSI=%.4f | KS=%.4f (p=%.4f) | drift=%s",
        feature_name, psi, ks_stat, ks_pvalue, drift_flag,
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        psi=psi,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        mean_ref=mean_ref,
        mean_cur=mean_cur,
        mean_delta=abs(mean_cur - mean_ref),
        std_ref=std_ref,
        std_cur=std_cur,
        std_delta=abs(std_cur - std_ref),
        drift_flag=drift_flag,
        drift_reason=drift_reason,
    )


# ── calcul global ─────────────────────────────────────────────────────────────

def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    cfg: MonitoringConfig = MONITOR_CFG,
) -> Dict[str, FeatureDriftResult]:
    """
    Calcule le drift pour toutes les features d'un coup.

    Args:
        reference_df : DataFrame de la distribution de référence
        current_df   : DataFrame de la distribution courante
        features     : liste des colonnes à analyser (défaut = toutes sauf 'y')
        cfg          : configuration des seuils

    Returns:
        dict {feature_name: FeatureDriftResult}
    """
    cols = features or [c for c in reference_df.columns if c != "y"]

    # Vérification que les colonnes existent dans les deux datasets
    missing_ref = [c for c in cols if c not in reference_df.columns]
    missing_cur = [c for c in cols if c not in current_df.columns]
    if missing_ref:
        raise ValueError(f"Colonnes absentes de reference_df : {missing_ref}")
    if missing_cur:
        raise ValueError(f"Colonnes absentes de current_df : {missing_cur}")

    results = {}
    for feat in cols:
        results[feat] = compute_feature_drift(
            feature_name=feat,
            reference=reference_df[feat].values,
            current=current_df[feat].values,
            cfg=cfg,
        )

    n_drifted = sum(1 for r in results.values() if r.drift_flag)
    logger.info(
        "Rapport drift : %d/%d features en drift",
        n_drifted, len(cols),
    )

    return results


# ── score global et niveau d'alerte ──────────────────────────────────────────

def compute_global_score(
    feature_results: Dict[str, FeatureDriftResult],
    cfg: MonitoringConfig = MONITOR_CFG,
) -> Tuple[float, str, bool]:
    """
    Agrège les métriques par feature en un score global et un niveau d'alerte.

    LOGIQUE DE DÉCISION :
      Le score global = moyenne pondérée des PSI de chaque feature.
      On pondère davantage les features avec un PSI élevé (x² pour amplifier).

      Niveau d'alerte :
        "none"     → rien à signaler
        "warning"  → score global > psi_alert_threshold (0.2)
                     OU au moins n_critical_features_for_alert features en drift
        "critical" → au moins une feature avec PSI > psi_critical_threshold (0.25)

    Args:
        feature_results : dict {nom: FeatureDriftResult}
        cfg             : configuration des seuils

    Returns:
        (drift_score_global, alert_level, drift_detected)
    """
    if not feature_results:
        return 0.0, "none", False

    psi_values = [r.psi for r in feature_results.values()]
    n_in_drift = sum(1 for r in feature_results.values() if r.drift_flag)

    # Score global = moyenne des PSI (simple et interprétable)
    global_score = float(np.mean(psi_values))

    # Détermination du niveau d'alerte
    has_critical_feature = any(r.psi > cfg.psi_critical_threshold for r in feature_results.values())

    if has_critical_feature:
        alert_level = "critical"
    elif global_score > cfg.psi_alert_threshold or n_in_drift >= cfg.n_critical_features_for_alert:
        alert_level = "warning"
    else:
        alert_level = "none"

    drift_detected = alert_level != "none"

    logger.info(
        "Score global PSI=%.4f | %d features en drift | alerte='%s'",
        global_score, n_in_drift, alert_level,
    )

    return global_score, alert_level, drift_detected


# ── rapport lisible ───────────────────────────────────────────────────────────

def format_drift_report(
    feature_results: Dict[str, FeatureDriftResult],
    global_score: float,
    alert_level: str,
) -> str:
    """
    Formate un rapport texte lisible pour les logs ou alertes.

    Returns:
        Rapport multi-lignes sous forme de string.
    """
    lines = [
        "=" * 60,
        f"  RAPPORT DE DRIFT",
        f"  Score global (PSI moyen) : {global_score:.4f}",
        f"  Niveau d'alerte          : {alert_level.upper()}",
        "=" * 60,
        f"{'Feature':<12} {'PSI':>8} {'KS stat':>10} {'KS p-val':>10} {'Δmean':>10} {'Drift':>8}",
        "-" * 60,
    ]

    for feat, r in sorted(feature_results.items()):
        drift_icon = "⚠ OUI" if r.drift_flag else "  non"
        lines.append(
            f"{feat:<12} {r.psi:>8.4f} {r.ks_stat:>10.4f} {r.ks_pvalue:>10.4f}"
            f" {r.mean_delta:>10.4f} {drift_icon:>8}"
        )

    lines.append("=" * 60)

    # Détails des features en drift
    drifted = [(n, r) for n, r in feature_results.items() if r.drift_flag]
    if drifted:
        lines.append("\nDÉTAIL DES FEATURES EN DRIFT :")
        for name, r in drifted:
            lines.append(f"  {name} :")
            lines.append(f"    mean  : {r.mean_ref:.4f} (ref) → {r.mean_cur:.4f} (cur)  Δ={r.mean_delta:.4f}")
            lines.append(f"    std   : {r.std_ref:.4f} (ref) → {r.std_cur:.4f} (cur)  Δ={r.std_delta:.4f}")
            lines.append(f"    raison: {r.drift_reason}")
    else:
        lines.append("\nAucune feature en drift détectée.")

    return "\n".join(lines)


# ── point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Démo rapide :
      python -m src.monitoring.drift_metrics           # comparer ref vs ref (PSI ~ 0)
      python -m src.monitoring.drift_metrics drift     # comparer ref vs drifted
      python -m src.monitoring.drift_metrics partial 0.5
    """
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_generation.generator import (
        generate_reference_data,
        generate_drifted_data,
        generate_partial_drift,
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "stable"

    ref_df = generate_reference_data(n_rows=10_000, seed=42)

    if mode == "drift":
        cur_df = generate_drifted_data(n_rows=2_000, seed=1)
        label = "DRIFT COMPLET (x2 +2σ, x4 +0.3)"
    elif mode == "partial":
        factor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        cur_df = generate_partial_drift(drift_factor=factor, n_rows=2_000, seed=1)
        label = f"DRIFT PARTIEL (factor={factor})"
    else:
        cur_df = generate_reference_data(n_rows=2_000, seed=99)
        label = "STABLE (ref vs ref)"

    print(f"\nComparaison : {label}")
    print(f"Référence   : {len(ref_df)} lignes")
    print(f"Courant     : {len(cur_df)} lignes")

    feature_results = compute_drift_report(ref_df, cur_df)
    global_score, alert_level, drift_detected = compute_global_score(feature_results)
    report = format_drift_report(feature_results, global_score, alert_level)
    print(report)