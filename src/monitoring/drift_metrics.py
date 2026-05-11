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
   - Mesure la distance maximale entre deux CDF
   - p-value < 0.05 → distributions statistiquement différentes
   - Idéal : variables continues (x1, x2, x3)

3. Wasserstein Distance (Earth Mover's Distance)
   - Mesure le "coût de transport" pour transformer une distribution en l'autre
   - Avantages vs KS : sensible à l'amplitude du shift (pas juste sa présence),
     très robuste sur les petits échantillons (<500 lignes)
   - Normalisée par l'écart-type de référence pour être comparable entre features
   - Seuil : wasserstein_norm > 0.2 → drift détecté

4. Test Chi² (Chi-carré)
   - Test statistique sur variables catégorielles / binaires
   - Utilisé automatiquement pour x4 (Bernoulli) si la feature est binaire (0/1)
   - p-value < 0.05 → les fréquences ont changé significativement
   - Complète KS qui suppose une variable continue

5. Jensen-Shannon Divergence
   - Symétrique, bornée entre 0 et 1
   - Mesure la divergence entre deux distributions discrétisées
   - Seuil : js_divergence > 0.1 → drift modéré

6. Delta mean / std / mean_delta_pct
   - Comparaison directe des statistiques descriptives

SÉLECTION AUTOMATIQUE DES TESTS
────────────────────────────────
  Variable binaire (0/1 uniquement) → PSI + Chi² + Wasserstein
  Variable continue                 → PSI + KS + Wasserstein + Jensen-Shannon

CONTRAT AVEC alerting.py
────────────────────────
compute_drift_report() retourne Dict[str, FeatureDriftResult].
alerting.py consomme ces objets directement.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import MONITOR_CFG, MonitoringConfig

logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    """
    Résultat du calcul de drift pour une feature individuelle.

    Contient TOUTES les métriques nécessaires :
    - aux règles de décision d'alerting.py
    - au stockage Postgres (drift_feature_metrics)
    - au reporting / format_drift_report()
    - à l'indexation Elasticsearch

    Attributs de base :
        feature_name    : nom de la colonne
        is_binary       : True si la feature est binaire (0/1) → guide le choix des tests

    Métriques PSI :
        psi             : Population Stability Index

    Métriques KS (variables continues uniquement) :
        ks_stat         : statistique du test KS
        ks_pvalue       : p-value du test KS (-1.0 si non applicable)

    Métriques Chi² (variables binaires uniquement) :
        chi2_stat       : statistique du test Chi² (-1.0 si non applicable)
        chi2_pvalue     : p-value du test Chi² (-1.0 si non applicable)

    Wasserstein (toutes variables) :
        wasserstein     : distance de Wasserstein brute
        wasserstein_norm: distance normalisée par std_ref (comparable entre features)

    Jensen-Shannon (variables continues uniquement) :
        js_divergence   : divergence Jensen-Shannon (0=identique, 1=opposé)

    Stats descriptives :
        mean_ref / mean_cur / mean_delta / mean_delta_pct
        std_ref / std_cur / std_delta
        current_q25 / current_median / current_q75

    Verdict :
        drift_flag      : True si au moins un test détecte un drift
        drift_reason    : explication textuelle du/des déclenchements
    """
    feature_name: str
    is_binary: bool

    # PSI
    psi: float

    # KS (continues) — -1.0 si non calculé
    ks_stat: float
    ks_pvalue: float

    # Chi² (binaires) — -1.0 si non calculé
    chi2_stat: float
    chi2_pvalue: float

    # Wasserstein (toutes)
    wasserstein: float
    wasserstein_norm: float

    # Jensen-Shannon (continues)
    js_divergence: float

    # Stats descriptives
    mean_ref: float
    mean_cur: float
    mean_delta: float
    mean_delta_pct: float
    std_ref: float
    std_cur: float
    std_delta: float
    current_q25: float
    current_median: float
    current_q75: float

    # Verdict
    drift_flag: bool
    drift_reason: str = ""

    # Histogrammes de distribution (pour visualisation Kibana/Vega)
    # Bins communs ref + cur → superposition directe dans Vega
    # Listes Python (sérialisables JSON/ES/XCom)
    distribution_ref: List[float] = field(default_factory=list)
    distribution_cur: List[float] = field(default_factory=list)
    bin_edges: List[float] = field(default_factory=list)


# ── Calcul PSI ────────────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule le PSI entre une distribution de référence et une distribution courante.

    COMMENT ÇA MARCHE :
      1. On découpe la référence en n_bins buckets de quantiles égaux
      2. On calcule la proportion de données dans chaque bucket
      3. PSI = Σ (p_ref - p_cur) * ln(p_ref / p_cur)

    Args:
        reference : tableau numpy de la distribution de référence
        current   : tableau numpy de la distribution courante
        n_bins    : nombre de buckets (10 par défaut)
        epsilon   : petite valeur pour éviter log(0)

    Returns:
        (psi_total, proportions_ref, proportions_cur, bucket_edges)
        Les proportions et bucket_edges sont utilisés pour construire
        les histogrammes de distribution stockés dans ES et le snapshot JSON.
    """
    quantile_points = np.linspace(0, 100, n_bins + 1)
    bucket_edges = np.percentile(reference, quantile_points)
    bucket_edges = np.unique(bucket_edges)

    ref_counts, _ = np.histogram(reference, bins=bucket_edges)
    cur_counts, _ = np.histogram(current, bins=bucket_edges)

    ref_props = (ref_counts / len(reference)) + epsilon
    cur_props = (cur_counts / len(current)) + epsilon

    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    psi_per_bucket = (ref_props - cur_props) * np.log(ref_props / cur_props)
    psi_total = float(psi_per_bucket.sum())

    return psi_total, ref_props, cur_props, bucket_edges


def _build_histogram(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Construit les histogrammes de distribution sur des bins communs.

    Utilise les mêmes bins pour ref et cur afin que Kibana/Vega puisse
    les superposer directement sur le même axe X.

    Les bins sont calculés sur l'union des deux distributions pour
    couvrir toute la plage de valeurs observées.

    Returns:
        (distribution_ref, distribution_cur, bin_edges)
        Toutes les valeurs sont des listes Python (sérialisables JSON).
        distribution_ref[i] = proportion des valeurs de ref dans le bin i
        distribution_cur[i] = proportion des valeurs de cur dans le bin i
        bin_edges a len(distribution_ref) + 1 éléments (bornes des bins)
    """
    combined_min = float(min(reference.min(), current.min()))
    combined_max = float(max(reference.max(), current.max()))

    # Légère extension des bornes pour inclure les valeurs exactes min/max
    margin = (combined_max - combined_min) * 0.01
    edges = np.linspace(combined_min - margin, combined_max + margin, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_props = (ref_counts / len(reference)).tolist()
    cur_props = (cur_counts / len(current)).tolist()
    bin_edges = [round(float(e), 6) for e in edges]

    return ref_props, cur_props, bin_edges


# ── Calcul KS ─────────────────────────────────────────────────────────────────

def compute_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Applique le test de Kolmogorov-Smirnov à deux échantillons.

    Returns:
        (ks_statistic, p_value)
    """
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


# ── Calcul Wasserstein ────────────────────────────────────────────────────────

def compute_wasserstein(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Calcule la distance de Wasserstein (Earth Mover's Distance) entre deux distributions.

    La distance brute est normalisée par l'écart-type de la référence pour
    la rendre comparable entre features d'échelles différentes.

    Returns:
        (wasserstein_brut, wasserstein_normalisé)
    """
    w = float(stats.wasserstein_distance(reference, current))
    std_ref = float(np.std(reference))
    w_norm = w / std_ref if std_ref > 0 else 0.0
    return w, w_norm


# ── Calcul Chi² ───────────────────────────────────────────────────────────────

def compute_chi2(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Applique le test Chi² d'homogénéité entre deux distributions de fréquences.

    Utilisé pour les variables binaires (0/1) ou catégorielles où KS
    n'est pas adapté (distributions discrètes).

    Construit une table de contingence 2×2 :
        ref_0  ref_1
        cur_0  cur_1

    Returns:
        (chi2_stat, p_value)
        (-1.0, -1.0) si le test n'est pas applicable (une catégorie absente)
    """
    ref_0 = int(np.sum(reference == 0))
    ref_1 = int(np.sum(reference == 1))
    cur_0 = int(np.sum(current == 0))
    cur_1 = int(np.sum(current == 1))

    # Le test nécessite que chaque cellule ait au moins 5 observations
    contingency = np.array([[ref_0, ref_1], [cur_0, cur_1]])
    if np.any(contingency < 5):
        logger.debug("Chi² non applicable : effectifs insuffisants %s", contingency.tolist())
        return -1.0, -1.0

    chi2, p, _, _ = stats.chi2_contingency(contingency)
    return float(chi2), float(p)


# ── Calcul Jensen-Shannon ─────────────────────────────────────────────────────

def compute_js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 50,
    epsilon: float = 1e-10,
) -> float:
    """
    Calcule la divergence Jensen-Shannon entre deux distributions.

    JS est symétrique (contrairement à KL) et bornée entre 0 et 1.
    Les distributions sont discrétisées sur les mêmes bins pour être comparables.

    JS = 0   → distributions identiques
    JS = 1   → distributions totalement différentes (log base 2)

    Seuil pratique : JS > 0.1 → drift modéré, JS > 0.2 → drift significatif

    Returns:
        js_divergence (float, entre 0 et 1)
    """
    # Discrétisation sur les mêmes bins (couvrant les deux distributions)
    combined_min = min(reference.min(), current.min())
    combined_max = max(reference.max(), current.max())
    bins = np.linspace(combined_min, combined_max, n_bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bins, density=False)
    cur_hist, _ = np.histogram(current, bins=bins, density=False)

    # Conversion en probabilités avec epsilon pour éviter log(0)
    p = (ref_hist + epsilon) / (ref_hist + epsilon).sum()
    q = (cur_hist + epsilon) / (cur_hist + epsilon).sum()

    # Mixture M = (P + Q) / 2
    m = (p + q) / 2

    # JS = (KL(P||M) + KL(Q||M)) / 2
    js = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    return float(np.clip(js, 0.0, 1.0))


# ── Calcul par feature ────────────────────────────────────────────────────────

def _is_binary(arr: np.ndarray) -> bool:
    """Détecte si un tableau ne contient que des valeurs 0 et 1."""
    unique = np.unique(arr)
    return set(unique.tolist()).issubset({0.0, 1.0, 0, 1})


def compute_feature_drift(
    feature_name: str,
    reference: np.ndarray,
    current: np.ndarray,
    cfg: MonitoringConfig = MONITOR_CFG,
) -> FeatureDriftResult:
    """
    Calcule toutes les métriques de drift pour une feature donnée.

    Sélection automatique des tests selon le type de variable :
      - Binaire (0/1) : PSI + Chi² + Wasserstein
      - Continue      : PSI + KS + Wasserstein + Jensen-Shannon

    Logique du drift_flag :
      Levé si AU MOINS UN des critères suivants est vrai :
        - PSI > psi_alert_threshold
        - KS p-value < 0.05 (continues)
        - Chi² p-value < 0.05 (binaires)
        - Wasserstein normalisé > wasserstein_threshold (0.2 par défaut)
        - JS divergence > js_threshold (0.1 par défaut)

    Args:
        feature_name : nom de la feature
        reference    : valeurs de la distribution de référence
        current      : valeurs de la distribution courante
        cfg          : configuration des seuils

    Returns:
        FeatureDriftResult complet
    """
    binary = _is_binary(reference)

    # ── PSI (toutes variables) ────────────────────────────────────────────────
    psi, _, _, _ = compute_psi(reference, current, n_bins=cfg.n_bins)

    # ── Histogrammes sur bins communs (pour Kibana/Vega) ──────────────────────
    distribution_ref, distribution_cur, bin_edges = _build_histogram(
        reference, current, n_bins=cfg.n_bins
    )

    # ── Wasserstein (toutes variables) ────────────────────────────────────────
    wasserstein, wasserstein_norm = compute_wasserstein(reference, current)

    # ── Tests spécifiques selon le type ──────────────────────────────────────
    if binary:
        ks_stat, ks_pvalue = -1.0, -1.0
        chi2_stat, chi2_pvalue = compute_chi2(reference, current)
        js_divergence = -1.0
    else:
        ks_stat, ks_pvalue = compute_ks(reference, current)
        chi2_stat, chi2_pvalue = -1.0, -1.0
        js_divergence = compute_js_divergence(reference, current)

    # ── Stats descriptives ────────────────────────────────────────────────────
    mean_ref = float(np.mean(reference))
    mean_cur = float(np.mean(current))
    std_ref = float(np.std(reference))
    std_cur = float(np.std(current))
    mean_delta = abs(mean_cur - mean_ref)
    mean_delta_pct = float((mean_cur - mean_ref) / mean_ref * 100) if mean_ref != 0 else 0.0
    current_q25 = float(np.percentile(current, 25))
    current_median = float(np.median(current))
    current_q75 = float(np.percentile(current, 75))

    # ── Verdict ───────────────────────────────────────────────────────────────
    reasons = []

    if psi > cfg.psi_alert_threshold:
        reasons.append(f"PSI={psi:.3f} > {cfg.psi_alert_threshold}")

    if not binary and ks_pvalue < 0.05:
        reasons.append(f"KS p-value={ks_pvalue:.4f} < 0.05")

    if binary and chi2_pvalue != -1.0 and chi2_pvalue < 0.05:
        reasons.append(f"Chi²={chi2_stat:.3f} p-value={chi2_pvalue:.4f} < 0.05")

    if wasserstein_norm > cfg.wasserstein_threshold:
        reasons.append(f"Wasserstein normalisé={wasserstein_norm:.3f} > {cfg.wasserstein_threshold}")

    if not binary and js_divergence != -1.0 and js_divergence > cfg.js_threshold:
        reasons.append(f"JS divergence={js_divergence:.3f} > {cfg.js_threshold}")

    drift_flag = len(reasons) > 0
    drift_reason = " | ".join(reasons) if reasons else "stable"

    test_used = "Chi²+Wasserstein" if binary else "KS+Wasserstein+JS"
    logger.debug(
        "Feature '%s' [%s] : PSI=%.4f | W_norm=%.4f | drift=%s",
        feature_name, "binaire" if binary else "continue",
        psi, wasserstein_norm, drift_flag,
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        is_binary=binary,
        psi=psi,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        chi2_stat=chi2_stat,
        chi2_pvalue=chi2_pvalue,
        wasserstein=wasserstein,
        wasserstein_norm=wasserstein_norm,
        js_divergence=js_divergence,
        mean_ref=mean_ref,
        mean_cur=mean_cur,
        mean_delta=mean_delta,
        mean_delta_pct=mean_delta_pct,
        std_ref=std_ref,
        std_cur=std_cur,
        std_delta=abs(std_cur - std_ref),
        current_q25=current_q25,
        current_median=current_median,
        current_q75=current_q75,
        drift_flag=drift_flag,
        drift_reason=drift_reason,
        distribution_ref=distribution_ref,
        distribution_cur=distribution_cur,
        bin_edges=bin_edges,
    )


# ── Calcul global ─────────────────────────────────────────────────────────────

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
    logger.info("Rapport drift : %d/%d features en drift", n_drifted, len(cols))

    return results


def compute_global_score(
    feature_results: Dict[str, "FeatureDriftResult"],
    cfg: MonitoringConfig = MONITOR_CFG,
) -> Tuple[float, str, bool]:
    """
    Agrège les métriques par feature en un score global et un niveau d'alerte.

    Score global = moyenne des PSI (simple et interprétable).

    Niveaux d'alerte :
      "none"     → score global < psi_alert_threshold ET pas de feature critique
      "warning"  → score global > psi_alert_threshold
                   OU >= n_critical_features_for_alert features en drift
      "critical" → au moins une feature avec PSI > psi_critical_threshold

    Returns:
        (drift_score_global, alert_level, drift_detected)
    """
    if not feature_results:
        return 0.0, "none", False

    psi_values = [r.psi for r in feature_results.values()]
    n_in_drift = sum(1 for r in feature_results.values() if r.drift_flag)
    global_score = float(np.mean(psi_values))

    has_critical_feature = any(
        r.psi > cfg.psi_critical_threshold for r in feature_results.values()
    )

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


# ── Conversion vers format Postgres ──────────────────────────────────────────

def feature_results_to_rows(
    feature_results: Dict[str, "FeatureDriftResult"],
) -> List[Dict]:
    """
    Convertit les résultats par feature en liste de dicts
    prêts pour postgres_client.insert_drift_feature_metrics(),
    l'indexation ES et le snapshot JSON MinIO.

    Inclut distribution_ref, distribution_cur, bin_edges
    pour la visualisation Kibana/Vega.
    """
    rows = []
    for fname, r in feature_results.items():
        rows.append({
            "feature_name": fname,
            "is_binary": r.is_binary,
            "psi": r.psi,
            "ks_stat": r.ks_stat,
            "ks_pvalue": r.ks_pvalue,
            "chi2_stat": r.chi2_stat,
            "chi2_pvalue": r.chi2_pvalue,
            "wasserstein": r.wasserstein,
            "wasserstein_norm": r.wasserstein_norm,
            "js_divergence": r.js_divergence,
            "mean_ref": r.mean_ref,
            "mean_cur": r.mean_cur,
            "mean_delta": r.mean_delta,
            "std_ref": r.std_ref,
            "std_cur": r.std_cur,
            "std_delta": r.std_delta,
            "drift_flag": r.drift_flag,
            "drift_reason": r.drift_reason,
            # Histogrammes — bins communs ref/cur pour Kibana/Vega
            "distribution_ref": r.distribution_ref,
            "distribution_cur": r.distribution_cur,
            "bin_edges": r.bin_edges,
        })
    return rows


# ── Rapport lisible ───────────────────────────────────────────────────────────

def format_drift_report(
    feature_results: Dict[str, "FeatureDriftResult"],
    global_score: float,
    alert_level: str,
) -> str:
    """Formate un rapport texte lisible pour les logs ou alertes."""
    lines = [
        "=" * 75,
        f"  RAPPORT DE DRIFT",
        f"  Score global (PSI moyen) : {global_score:.4f}",
        f"  Niveau d'alerte          : {alert_level.upper()}",
        "=" * 75,
        f"{'Feature':<8} {'Type':<8} {'PSI':>7} {'KS/χ²p':>9} {'W_norm':>8} {'JS':>7} {'Drift':>7}",
        "-" * 75,
    ]

    for feat, r in sorted(feature_results.items()):
        drift_icon = "⚠ OUI" if r.drift_flag else "  non"
        type_label = "binaire" if r.is_binary else "continu"

        # KS p-value pour continues, Chi² p-value pour binaires
        if r.is_binary:
            test_p = f"{r.chi2_pvalue:.4f}" if r.chi2_pvalue != -1.0 else "  N/A"
        else:
            test_p = f"{r.ks_pvalue:.4f}" if r.ks_pvalue != -1.0 else "  N/A"

        js_str = f"{r.js_divergence:.4f}" if r.js_divergence != -1.0 else "  N/A"

        lines.append(
            f"{feat:<8} {type_label:<8} {r.psi:>7.4f} {test_p:>9} "
            f"{r.wasserstein_norm:>8.4f} {js_str:>7} {drift_icon:>7}"
        )

    lines.append("=" * 75)

    drifted = [(n, r) for n, r in feature_results.items() if r.drift_flag]
    if drifted:
        lines.append("\nDÉTAIL DES FEATURES EN DRIFT :")
        for name, r in drifted:
            lines.append(f"  {name} ({'binaire' if r.is_binary else 'continue'}) :")
            lines.append(
                f"    mean  : {r.mean_ref:.4f} (ref) → {r.mean_cur:.4f} (cur)"
                f"  Δ={r.mean_delta:.4f} ({r.mean_delta_pct:+.1f}%)"
            )
            lines.append(f"    std   : {r.std_ref:.4f} (ref) → {r.std_cur:.4f} (cur)  Δ={r.std_delta:.4f}")
            lines.append(f"    raison: {r.drift_reason}")
    else:
        lines.append("\nAucune feature en drift détectée.")

    return "\n".join(lines)


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
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
        label = "DRIFT COMPLET"
    elif mode == "partial":
        factor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        cur_df = generate_partial_drift(drift_factor=factor, n_rows=2_000, seed=1)
        label = f"DRIFT PARTIEL (factor={factor})"
    else:
        cur_df = generate_reference_data(n_rows=2_000, seed=99)
        label = "STABLE (ref vs ref)"

    print(f"\nComparaison : {label}")
    feature_results = compute_drift_report(ref_df, cur_df)
    global_score, alert_level, _ = compute_global_score(feature_results)
    print(format_drift_report(feature_results, global_score, alert_level))