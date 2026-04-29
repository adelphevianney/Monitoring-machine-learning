"""
Client MinIO pour le système MLOps Drift Monitoring.

RÔLE DE CE MODULE
─────────────────
MinIO est notre "disque partagé" entre tous les composants.
Il joue deux rôles distincts :

  Rôle 1 — Artifact store de MLflow
    MLflow y stocke automatiquement les modèles sérialisés,
    les courbes, les rapports. On ne touche pas à ça directement.

  Rôle 2 — Data lake applicatif
    Notre code y stocke les datasets et snapshots de monitoring.
    C'est ce que gère CE module.

BUCKETS UTILISÉS
────────────────
  raw-datasets        : datasets bruts générés (parquet)
  processed-datasets  : datasets après preprocessing (parquet)
  feature-stats       : stats de référence au format JSON
  monitoring-snapshots: snapshots des données observées lors du monitoring

CONVENTIONS DE NOMMAGE DES OBJETS
──────────────────────────────────
  raw-datasets/reference_v1_20240101T120000Z.parquet
  monitoring-snapshots/2024-01-15/window_14h_15h.parquet

Toujours inclure un horodatage pour pouvoir rejouer les analyses.
"""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from minio import Minio
from minio.error import S3Error

from src.config.settings import MINIO_CFG, MinIOConfig

logger = logging.getLogger(__name__)


# ── Client singleton ──────────────────────────────────────────────────────────

def get_client(cfg: MinIOConfig = MINIO_CFG) -> Minio:
    """
    Crée et retourne un client MinIO.

    On recrée le client à chaque appel pour éviter les problèmes
    de connexions mortes dans les DAGs Airflow longue durée.
    Le client MinIO est léger, il n'y a pas de pool à gérer.
    """
    return Minio(
        endpoint=cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def ensure_buckets_exist(cfg: MinIOConfig = MINIO_CFG) -> None:
    """
    Crée les buckets s'ils n'existent pas encore.

    À appeler au démarrage de l'infrastructure (docker-compose entrypoint
    ou premier run Airflow). Idempotent : safe à appeler plusieurs fois.
    """
    client = get_client(cfg)
    for bucket in cfg.buckets:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("Bucket créé : %s", bucket)
        else:
            logger.debug("Bucket existant : %s", bucket)


def check_connection(cfg: MinIOConfig = MINIO_CFG) -> bool:
    """Vérifie que MinIO est accessible. Utile au démarrage des DAGs."""
    try:
        client = get_client(cfg)
        client.list_buckets()
        logger.info("Connexion MinIO OK (%s)", cfg.endpoint)
        return True
    except Exception as exc:
        logger.error("MinIO inaccessible : %s", exc)
        return False


# ── Upload / Download de DataFrames ──────────────────────────────────────────

def upload_dataframe(
    df: pd.DataFrame,
    bucket: str,
    object_name: str,
    cfg: MinIOConfig = MINIO_CFG,
) -> str:
    """
    Upload un DataFrame pandas vers MinIO au format parquet.

    Pourquoi parquet ?
    - Colonne-orienté : lecture sélective des colonnes très rapide
    - Compressé : ~10x plus léger que CSV sur des données numériques
    - Typé : préserve les types pandas (float64, bool, etc.)

    Args:
        df          : DataFrame à uploader
        bucket      : nom du bucket cible (ex: 'raw-datasets')
        object_name : chemin de l'objet dans le bucket
                      (ex: 'reference_v1_20240101T120000Z.parquet')

    Returns:
        URI complète de l'objet : "s3://bucket/object_name"
    """
    # Sérialiser le DataFrame en mémoire (pas de fichier temporaire)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    size = buffer.getbuffer().nbytes

    client = get_client(cfg)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=size,
        content_type="application/octet-stream",
    )

    uri = f"s3://{bucket}/{object_name}"
    logger.info("Upload réussi : %s (%d lignes, %.1f KB)", uri, len(df), size / 1024)
    return uri


def download_dataframe(
    bucket: str,
    object_name: str,
    cfg: MinIOConfig = MINIO_CFG,
) -> pd.DataFrame:
    """
    Télécharge un objet parquet depuis MinIO et le retourne comme DataFrame.

    Args:
        bucket      : nom du bucket source
        object_name : chemin de l'objet dans le bucket

    Returns:
        DataFrame pandas
    """
    client = get_client(cfg)
    response = client.get_object(bucket_name=bucket, object_name=object_name)

    try:
        buffer = io.BytesIO(response.read())
        df = pd.read_parquet(buffer, engine="pyarrow")
    finally:
        response.close()
        response.release_conn()

    logger.info(
        "Download réussi : s3://%s/%s (%d lignes)", bucket, object_name, len(df)
    )
    return df


def download_dataframe_from_uri(uri: str, cfg: MinIOConfig = MINIO_CFG) -> pd.DataFrame:
    """
    Télécharge depuis une URI complète au format "s3://bucket/object".

    Pratique quand on stocke l'URI dans Postgres et qu'on veut
    recharger le dataset directement depuis l'URI stockée.
    """
    # Parser l'URI : "s3://raw-datasets/reference_v1.parquet"
    if not uri.startswith("s3://"):
        raise ValueError(f"URI invalide, doit commencer par 's3://' : {uri}")

    parts = uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"URI mal formée : {uri}")

    bucket, object_name = parts
    return download_dataframe(bucket, object_name, cfg)


# ── Upload / Download de JSON ─────────────────────────────────────────────────

def upload_json(
    data: dict,
    bucket: str,
    object_name: str,
    cfg: MinIOConfig = MINIO_CFG,
) -> str:
    """
    Upload un dict Python comme JSON vers MinIO.

    Utilisé pour sauvegarder les stats de référence (feature_stats)
    et les résultats de monitoring au format lisible.

    Returns:
        URI "s3://bucket/object_name"
    """
    payload = json.dumps(data, indent=2, default=str).encode("utf-8")
    buffer = io.BytesIO(payload)

    client = get_client(cfg)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=len(payload),
        content_type="application/json",
    )

    uri = f"s3://{bucket}/{object_name}"
    logger.info("JSON uploadé : %s (%.1f KB)", uri, len(payload) / 1024)
    return uri


def download_json(
    bucket: str,
    object_name: str,
    cfg: MinIOConfig = MINIO_CFG,
) -> dict:
    """Télécharge un objet JSON depuis MinIO et retourne un dict Python."""
    client = get_client(cfg)
    response = client.get_object(bucket_name=bucket, object_name=object_name)
    try:
        data = json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()
    return data


# ── Helpers de nommage ────────────────────────────────────────────────────────

def make_dataset_object_name(prefix: str, version: str = "") -> str:
    """
    Génère un nom d'objet horodaté pour un dataset.

    Ex : make_dataset_object_name("reference", "v1")
         → "reference_v1_20240101T120000Z.parquet"
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [p for p in [prefix, version, ts] if p]
    return "_".join(parts) + ".parquet"


def make_snapshot_object_name(window_start: datetime, window_end: datetime) -> str:
    """
    Génère un nom d'objet pour un snapshot de monitoring.

    Ex : "2024-01-15/window_14h00_15h00.parquet"

    On organise par date pour faciliter le parcours chronologique.
    """
    date_prefix = window_start.strftime("%Y-%m-%d")
    start_str = window_start.strftime("%Hh%M")
    end_str = window_end.strftime("%Hh%M")
    return f"{date_prefix}/window_{start_str}_{end_str}.parquet"


def make_stats_object_name(mlflow_run_id: str) -> str:
    """
    Génère un nom d'objet pour un fichier de stats de référence.

    Ex : "stats_a1b2c3d4_20240101T120000Z.json"
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = mlflow_run_id[:8]
    return f"stats_{short_id}_{ts}.json"


# ── Utilitaires ───────────────────────────────────────────────────────────────

def list_objects(
    bucket: str,
    prefix: str = "",
    cfg: MinIOConfig = MINIO_CFG,
) -> list:
    """
    Liste les objets dans un bucket (avec préfixe optionnel).

    Utile pour inspecter le contenu du data lake ou trouver
    le dataset le plus récent d'un préfixe donné.

    Returns:
        Liste de dicts : {"name": ..., "size": ..., "last_modified": ...}
    """
    client = get_client(cfg)
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)

    result = []
    for obj in objects:
        result.append({
            "name": obj.object_name,
            "size": obj.size,
            "last_modified": obj.last_modified,
        })

    logger.debug("Liste bucket '%s' prefix='%s' : %d objets", bucket, prefix, len(result))
    return result


def get_latest_object(
    bucket: str,
    prefix: str = "",
    cfg: MinIOConfig = MINIO_CFG,
) -> Optional[str]:
    """
    Retourne le nom du dernier objet uploadé dans un bucket (par date).

    Pratique pour charger le dataset de référence le plus récent
    sans avoir à stocker le nom explicitement.

    Returns:
        Nom de l'objet le plus récent, ou None si le bucket est vide.
    """
    objects = list_objects(bucket, prefix, cfg)
    if not objects:
        return None

    latest = max(objects, key=lambda o: o["last_modified"])
    logger.info("Dernier objet dans '%s/%s' : %s", bucket, prefix, latest["name"])
    return latest["name"]
