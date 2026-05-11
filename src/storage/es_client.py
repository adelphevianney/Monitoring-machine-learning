"""
Client Elasticsearch pour le système MLOps Drift Monitoring.

RÔLE DE CE MODULE
─────────────────
Indexe les résultats du monitoring dans Elasticsearch pour visualisation
dans Kibana. Complète Postgres (requêtes opérationnelles) avec une
couche d'observabilité temps réel (dashboards, alertes, tendances).

DEUX INDEX
──────────
  drift-monitoring-runs
    Un document par run de monitoring (fenêtre d'observation).
    Contient le score global, le niveau d'alerte, le nombre de features
    en drift, et un snapshot des métriques PSI par feature.
    Utilisé pour les graphiques temporels dans Kibana.

  drift-feature-metrics
    Un document par (run, feature).
    Contient toutes les métriques détaillées : PSI, KS/Chi², Wasserstein,
    Jensen-Shannon, deltas, et le verdict final.
    Utilisé pour les heatmaps et analyses par feature.

IDEMPOTENCE
───────────
  ensure_indices_exist() crée les index avec leurs mappings s'ils n'existent
  pas encore. Safe à appeler à chaque démarrage du DAG.

  Les documents sont indexés avec un ID déterministe basé sur
  monitoring_run_id (+ feature_name pour les métriques par feature)
  → upsert implicite si le document existe déjà.

ERREURS
───────
  Toutes les fonctions loguent les erreurs mais ne lèvent pas d'exception.
  Un échec d'indexation ES ne doit pas faire échouer le DAG —
  Postgres reste la source de vérité.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# URL Elasticsearch (configurable via variable d'environnement)
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://mlops-elasticsearch:9200")

# Noms des index
INDEX_MONITORING_RUNS = "drift-monitoring-runs"
INDEX_FEATURE_METRICS = "drift-feature-metrics"


# ── Helpers HTTP ──────────────────────────────────────────────────────────────
# On utilise urllib stdlib pour éviter une dépendance supplémentaire
# (elasticsearch-py ou requests). Pour une prod réelle, utiliser
# le client officiel elasticsearch-py.

def _es_request(
    method: str,
    path: str,
    body: Optional[dict] = None,
    timeout: int = 10,
) -> Optional[dict]:
    """
    Effectue une requête HTTP vers Elasticsearch.

    Args:
        method  : GET, PUT, POST, HEAD
        path    : chemin relatif (ex: "/_cluster/health")
        body    : corps JSON optionnel
        timeout : timeout en secondes

    Returns:
        dict de la réponse JSON, ou None en cas d'erreur
    """
    url = f"{ES_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body else None

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8") if exc.fp else ""
        logger.error("ES HTTP %s %s → %d : %s", method, path, exc.code, body_text[:200])
        return None
    except Exception as exc:
        logger.error("ES request failed %s %s : %s", method, path, exc)
        return None


def _index_exists(index_name: str) -> bool:
    """Vérifie si un index Elasticsearch existe."""
    req = urllib.request.Request(f"{ES_URL}/{index_name}", method="HEAD")
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except urllib.error.HTTPError as exc:
        return exc.code != 404
    except Exception:
        return False


# ── Vérification de connexion ─────────────────────────────────────────────────

def check_connection() -> bool:
    """Vérifie que Elasticsearch est accessible."""
    result = _es_request("GET", "/_cluster/health")
    if result:
        status = result.get("status", "unknown")
        logger.info("Elasticsearch OK — cluster status: %s", status)
        return True
    logger.error("Elasticsearch inaccessible (%s)", ES_URL)
    return False


# ── Création des index avec mappings ─────────────────────────────────────────

def ensure_indices_exist() -> None:
    """
    Crée les index Elasticsearch avec leurs mappings s'ils n'existent pas.

    Les mappings définissent les types de champs pour optimiser
    les agrégations Kibana (keyword pour les enums, float pour les métriques).
    Idempotent — safe à appeler plusieurs fois.
    """
    _ensure_monitoring_runs_index()
    _ensure_feature_metrics_index()


def _ensure_monitoring_runs_index() -> None:
    if _index_exists(INDEX_MONITORING_RUNS):
        logger.debug("Index '%s' existe déjà", INDEX_MONITORING_RUNS)
        return

    mapping = {
        "mappings": {
            "properties": {
                "monitoring_run_id":    {"type": "keyword"},
                "ref_run_id":           {"type": "keyword"},
                "observation_start":    {"type": "date"},
                "observation_end":      {"type": "date"},
                "indexed_at":           {"type": "date"},
                "input_dataset_uri":    {"type": "keyword"},
                # Score global
                "drift_score_global":   {"type": "float"},
                "drift_detected":       {"type": "boolean"},
                "alert_level":          {"type": "keyword"},  # none|warning|critical
                "n_features_drifted":   {"type": "integer"},
                # PSI par feature (pour sparklines Kibana)
                "feature_psi":          {"type": "object"},
                # Résumé des features en drift
                "drifted_features":     {"type": "keyword"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,  # single-node → pas de répliques
        }
    }

    result = _es_request("PUT", f"/{INDEX_MONITORING_RUNS}", mapping)
    if result and result.get("acknowledged"):
        logger.info("Index '%s' créé", INDEX_MONITORING_RUNS)
    else:
        logger.error("Échec création index '%s'", INDEX_MONITORING_RUNS)


def _ensure_feature_metrics_index() -> None:
    if _index_exists(INDEX_FEATURE_METRICS):
        logger.debug("Index '%s' existe déjà", INDEX_FEATURE_METRICS)
        return

    mapping = {
        "mappings": {
            "properties": {
                "monitoring_run_id":    {"type": "keyword"},
                "ref_run_id":           {"type": "keyword"},
                "observation_start":    {"type": "date"},
                "indexed_at":           {"type": "date"},
                "feature_name":         {"type": "keyword"},
                "is_binary":            {"type": "boolean"},
                # PSI
                "psi":                  {"type": "float"},
                # KS (continues)
                "ks_stat":              {"type": "float"},
                "ks_pvalue":            {"type": "float"},
                # Chi² (binaires)
                "chi2_stat":            {"type": "float"},
                "chi2_pvalue":          {"type": "float"},
                # Wasserstein
                "wasserstein":          {"type": "float"},
                "wasserstein_norm":     {"type": "float"},
                # Jensen-Shannon
                "js_divergence":        {"type": "float"},
                # Stats descriptives
                "mean_ref":             {"type": "float"},
                "mean_cur":             {"type": "float"},
                "mean_delta":           {"type": "float"},
                "std_ref":              {"type": "float"},
                "std_cur":              {"type": "float"},
                "std_delta":            {"type": "float"},
                # Verdict
                "drift_flag":           {"type": "boolean"},
                "drift_reason":         {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "alert_level":          {"type": "keyword"},
                # Histogrammes de distribution (pour Vega dans Kibana)
                # bin_edges       : N+1 bornes des bins (axe X du graphique)
                # distribution_ref: N proportions de référence
                # distribution_cur: N proportions courantes
                # Type float[] — ES indexe chaque élément du tableau
                "bin_edges":            {"type": "float"},
                "distribution_ref":     {"type": "float"},
                "distribution_cur":     {"type": "float"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    }

    result = _es_request("PUT", f"/{INDEX_FEATURE_METRICS}", mapping)
    if result and result.get("acknowledged"):
        logger.info("Index '%s' créé", INDEX_FEATURE_METRICS)
    else:
        logger.error("Échec création index '%s'", INDEX_FEATURE_METRICS)


# ── Indexation ────────────────────────────────────────────────────────────────

def index_monitoring_run(
    monitoring_run_id: str,
    ref_run_id: str,
    observation_start: datetime,
    observation_end: datetime,
    drift_score_global: float,
    drift_detected: bool,
    alert_level: str,
    n_features_drifted: int,
    input_dataset_uri: str,
    feature_metrics: List[Dict],
) -> bool:
    """
    Indexe le résultat global d'un run de monitoring.

    Le document inclut un snapshot des PSI par feature pour permettre
    des visualisations sparkline dans Kibana sans avoir à joindre
    les deux index.

    Args:
        monitoring_run_id : identifiant unique du run de monitoring
        ref_run_id        : run d'entraînement utilisé comme référence
        observation_start : début de la fenêtre d'observation
        observation_end   : fin de la fenêtre d'observation
        drift_score_global: PSI moyen global
        drift_detected    : True si au moins une feature en drift
        alert_level       : 'none' | 'warning' | 'critical'
        n_features_drifted: nombre de features en drift
        input_dataset_uri : URI MinIO du snapshot observé
        feature_metrics   : liste des dicts de métriques par feature
                            (issus de feature_results_to_rows())

    Returns:
        True si l'indexation a réussi, False sinon
    """
    # Snapshot PSI par feature (objet ES imbriqué)
    feature_psi = {
        row["feature_name"]: round(row["psi"], 4)
        for row in feature_metrics
    }

    # Liste des features en drift pour filtrage Kibana
    drifted_features = [
        row["feature_name"]
        for row in feature_metrics
        if row.get("drift_flag")
    ]

    doc = {
        "monitoring_run_id":  monitoring_run_id,
        "ref_run_id":         ref_run_id,
        "observation_start":  observation_start.isoformat(),
        "observation_end":    observation_end.isoformat(),
        "indexed_at":         datetime.now(timezone.utc).isoformat(),
        "input_dataset_uri":  input_dataset_uri,
        "drift_score_global": round(drift_score_global, 6),
        "drift_detected":     drift_detected,
        "alert_level":        alert_level,
        "n_features_drifted": n_features_drifted,
        "feature_psi":        feature_psi,
        "drifted_features":   drifted_features,
    }

    # ID déterministe → upsert si le document existe déjà
    result = _es_request(
        "PUT",
        f"/{INDEX_MONITORING_RUNS}/_doc/{monitoring_run_id}",
        doc,
    )

    if result and result.get("result") in ("created", "updated"):
        logger.info(
            "ES indexé [monitoring_run] : %s | alert=%s | score=%.4f",
            monitoring_run_id[:8], alert_level, drift_score_global,
        )
        return True

    logger.error("Échec indexation ES [monitoring_run] : %s", monitoring_run_id[:8])
    return False


def index_feature_metrics(
    monitoring_run_id: str,
    ref_run_id: str,
    observation_start: datetime,
    alert_level: str,
    feature_metrics: List[Dict],
) -> int:
    """
    Indexe le détail des métriques par feature pour un run de monitoring.

    Utilise le bulk API d'ES pour indexer toutes les features en une
    seule requête HTTP (plus efficace que N requêtes individuelles).

    Args:
        monitoring_run_id : identifiant du run de monitoring
        ref_run_id        : run d'entraînement de référence
        observation_start : timestamp de début de fenêtre
        alert_level       : niveau d'alerte global du run
        feature_metrics   : liste des dicts de métriques par feature

    Returns:
        Nombre de documents indexés avec succès
    """
    if not feature_metrics:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    obs_start_iso = observation_start.isoformat()

    # Construction du payload bulk (format NDJSON requis par ES)
    bulk_lines = []
    for row in feature_metrics:
        doc_id = f"{monitoring_run_id}_{row['feature_name']}"
        # Ligne d'action
        bulk_lines.append(json.dumps({
            "index": {
                "_index": INDEX_FEATURE_METRICS,
                "_id": doc_id,
            }
        }))
        # Ligne de document
        bulk_lines.append(json.dumps({
            "monitoring_run_id":  monitoring_run_id,
            "ref_run_id":         ref_run_id,
            "observation_start":  obs_start_iso,
            "indexed_at":         now,
            "alert_level":        alert_level,
            "feature_name":       row["feature_name"],
            "is_binary":          row.get("is_binary", False),
            "psi":                row.get("psi"),
            "ks_stat":            row.get("ks_stat"),
            "ks_pvalue":          row.get("ks_pvalue"),
            "chi2_stat":          row.get("chi2_stat"),
            "chi2_pvalue":        row.get("chi2_pvalue"),
            "wasserstein":        row.get("wasserstein"),
            "wasserstein_norm":   row.get("wasserstein_norm"),
            "js_divergence":      row.get("js_divergence"),
            "mean_ref":           row.get("mean_ref"),
            "mean_cur":           row.get("mean_cur"),
            "mean_delta":         row.get("mean_delta"),
            "std_ref":            row.get("std_ref"),
            "std_cur":            row.get("std_cur"),
            "std_delta":          row.get("std_delta"),
            "drift_flag":         row.get("drift_flag", False),
            "drift_reason":       row.get("drift_reason", ""),
            # Histogrammes — bins communs ref/cur pour Kibana Vega
            "bin_edges":          row.get("bin_edges", []),
            "distribution_ref":   row.get("distribution_ref", []),
            "distribution_cur":   row.get("distribution_cur", []),
        }))

    # Requête bulk
    payload = "\n".join(bulk_lines) + "\n"
    data = payload.encode("utf-8")

    req = urllib.request.Request(f"{ES_URL}/_bulk", data=data, method="POST")
    req.add_header("Content-Type", "application/x-ndjson")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("Échec bulk indexation ES [feature_metrics] : %s", exc)
        return 0

    if result.get("errors"):
        errors = [
            item["index"].get("error")
            for item in result.get("items", [])
            if item.get("index", {}).get("error")
        ]
        logger.error("Erreurs bulk ES : %d erreurs — premier : %s", len(errors), errors[0] if errors else "?")
        n_ok = sum(1 for item in result.get("items", []) if item.get("index", {}).get("status") in (200, 201))
    else:
        n_ok = len(feature_metrics)

    logger.info(
        "ES bulk [feature_metrics] : %d/%d features indexées pour run %s",
        n_ok, len(feature_metrics), monitoring_run_id[:8],
    )
    return n_ok


# ── Export JSON pour MinIO (point 1 version 2) ───────────────────────────────

def build_monitoring_snapshot_json(
    monitoring_run_id: str,
    ref_run_id: str,
    observation_start: datetime,
    observation_end: datetime,
    drift_score_global: float,
    drift_detected: bool,
    alert_level: str,
    n_features_drifted: int,
    input_dataset_uri: str,
    feature_metrics: List[Dict],
) -> dict:
    """
    Construit un dict JSON complet du snapshot de monitoring.

    Ce dict est uploadé dans MinIO (monitoring-snapshots) au format JSON
    en plus du parquet des données brutes.
    Lisible humainement, exploitable par ES, archivable pour replay.

    Returns:
        dict sérialisable en JSON
    """
    return {
        "schema_version": "2.0",
        "monitoring_run_id": monitoring_run_id,
        "ref_run_id": ref_run_id,
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dataset_uri": input_dataset_uri,
        "summary": {
            "drift_score_global": round(drift_score_global, 6),
            "drift_detected": drift_detected,
            "alert_level": alert_level,
            "n_features_drifted": n_features_drifted,
        },
        "features": {
            row["feature_name"]: {
                "is_binary": row.get("is_binary", False),
                "psi": row.get("psi"),
                "ks_stat": row.get("ks_stat"),
                "ks_pvalue": row.get("ks_pvalue"),
                "chi2_stat": row.get("chi2_stat"),
                "chi2_pvalue": row.get("chi2_pvalue"),
                "wasserstein": row.get("wasserstein"),
                "wasserstein_norm": row.get("wasserstein_norm"),
                "js_divergence": row.get("js_divergence"),
                "mean_ref": row.get("mean_ref"),
                "mean_cur": row.get("mean_cur"),
                "mean_delta": row.get("mean_delta"),
                "std_ref": row.get("std_ref"),
                "std_cur": row.get("std_cur"),
                "std_delta": row.get("std_delta"),
                "drift_flag": row.get("drift_flag", False),
                "drift_reason": row.get("drift_reason", ""),
                # Histogrammes — bins communs ref/cur pour Kibana Vega
                "bin_edges":        row.get("bin_edges", []),
                "distribution_ref": row.get("distribution_ref", []),
                "distribution_cur": row.get("distribution_cur", []),
            }
            for row in feature_metrics
        },
    }