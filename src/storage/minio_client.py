"""
Client MinIO pour le système MLOps Drift Monitoring.

RÔLE DE CE MODULE
─────────────────
MinIO est notre "disque partagé" entre tous les composants.
Il stocke deux catégories d'objets :

  Artefacts de training
    Modèles sérialisés, métriques, params, stats de référence,
    feature importance. Organisés sous runs/<run_id>/.
    Gérés par train.py, ce module fournit les helpers de nommage.

  Data lake applicatif
    Datasets bruts, datasets préprocessés, snapshots de monitoring.

BUCKETS UTILISÉS
────────────────
  models              : artefacts de training (modèle, métriques, stats)
  raw-datasets        : datasets bruts générés (parquet)
  processed-datasets  : datasets après preprocessing (parquet)
  feature-stats       : stats de référence au format JSON
  monitoring-snapshots: snapshots des données observées lors du monitoring

CONVENTIONS DE NOMMAGE DES OBJETS
──────────────────────────────────
  models/runs/<run_id>/model.joblib
  models/runs/<run_id>/metrics.json
  models/runs/<run_id>/reference_stats/reference_stats.json
  raw-datasets/reference_v1_20240101T120000Z.parquet
  monitoring-snapshots/2024-01-15/window_14h_15h.parquet

CLIENT UNIQUE : boto3
─────────────────────
On utilise UNIQUEMENT boto3 (compatible S3) pour éviter toute ambiguïté.
Le SDK MinIO natif (minio.Minio) est abandonné.
"""

import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import boto3
import pandas as pd
from botocore.client import Config

from src.config.settings import MINIO_CFG, MinIOConfig

logger = logging.getLogger(__name__)


# ── Client singleton ──────────────────────────────────────────────────────────

def get_minio_client(cfg: MinIOConfig = MINIO_CFG):
    """
    Crée et retourne un client boto3 pointant vers MinIO.

    On recrée le client à chaque appel pour éviter les problèmes
    de connexions mortes dans les DAGs Airflow longue durée.
    boto3 est léger : pas de pool à gérer.
    """
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_buckets_exist(cfg: MinIOConfig = MINIO_CFG) -> None:
    """
    Crée les buckets s'ils n'existent pas encore.

    À appeler au démarrage de l'infrastructure (docker-compose entrypoint
    ou premier run Airflow). Idempotent : safe à appeler plusieurs fois.
    """
    client = get_minio_client(cfg)
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    for bucket in cfg.buckets:
        if bucket not in existing:
            client.create_bucket(Bucket=bucket)
            logger.info("Bucket créé : %s", bucket)
        else:
            logger.debug("Bucket existant : %s", bucket)


def check_connection(cfg: MinIOConfig = MINIO_CFG) -> bool:
    """Vérifie que MinIO est accessible. Utile au démarrage des DAGs."""
    try:
        client = get_minio_client(cfg)
        client.list_buckets()
        logger.info("Connexion MinIO OK (%s)", cfg.endpoint_url)
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

    Returns:
        URI complète de l'objet : "s3://bucket/object_name"
    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    size = buffer.getbuffer().nbytes

    client = get_minio_client(cfg)
    client.put_object(
        Bucket=bucket,
        Key=object_name,
        Body=buffer,
        ContentLength=size,
        ContentType="application/octet-stream",
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
    """
    client = get_minio_client(cfg)
    response = client.get_object(Bucket=bucket, Key=object_name)
    buffer = io.BytesIO(response["Body"].read())
    df = pd.read_parquet(buffer, engine="pyarrow")
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

    Returns:
        URI "s3://bucket/object_name"
    """
    payload = json.dumps(data, indent=2, default=str).encode("utf-8")
    buffer = io.BytesIO(payload)

    client = get_minio_client(cfg)
    client.put_object(
        Bucket=bucket,
        Key=object_name,
        Body=buffer,
        ContentLength=len(payload),
        ContentType="application/json",
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
    client = get_minio_client(cfg)
    response = client.get_object(Bucket=bucket, Key=object_name)
    data = json.loads(response["Body"].read().decode("utf-8"))
    return data


def upload_bytes(
    data: bytes,
    bucket: str,
    object_name: str,
    content_type: str = "application/octet-stream",
    cfg: MinIOConfig = MINIO_CFG,
) -> str:
    """
    Upload des bytes bruts vers MinIO.

    Utilisé pour les modèles joblib, CSV, et autres binaires.

    Returns:
        URI "s3://bucket/object_name"
    """
    client = get_minio_client(cfg)
    client.put_object(
        Bucket=bucket,
        Key=object_name,
        Body=io.BytesIO(data),
        ContentLength=len(data),
        ContentType=content_type,
    )
    uri = f"s3://{bucket}/{object_name}"
    logger.debug("Bytes uploadés → %s (%.1f KB)", uri, len(data) / 1024)
    return uri


def download_bytes(
    bucket: str,
    object_name: str,
    cfg: MinIOConfig = MINIO_CFG,
) -> bytes:
    """Télécharge un objet binaire depuis MinIO."""
    client = get_minio_client(cfg)
    response = client.get_object(Bucket=bucket, Key=object_name)
    return response["Body"].read()


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
    """
    date_prefix = window_start.strftime("%Y-%m-%d")
    start_str = window_start.strftime("%Hh%M")
    end_str = window_end.strftime("%Hh%M")
    return f"{date_prefix}/window_{start_str}_{end_str}.parquet"


def make_stats_object_name(run_id: str) -> str:
    """
    Génère un nom d'objet pour un fichier de stats de référence.

    Ex : "stats_a1b2c3d4_20240101T120000Z.json"
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = run_id[:8]
    return f"stats_{short_id}_{ts}.json"


def make_run_prefix(run_id: str) -> str:
    """
    Retourne le préfixe MinIO d'un run de training.

    Ex : make_run_prefix("a1b2c3d4e5f6") → "runs/a1b2c3d4e5f6"
    """
    return f"runs/{run_id}"


# ── Utilitaires ───────────────────────────────────────────────────────────────

def list_objects(
    bucket: str,
    prefix: str = "",
    cfg: MinIOConfig = MINIO_CFG,
) -> list:
    """
    Liste les objets dans un bucket (avec préfixe optionnel).

    Returns:
        Liste de dicts : {"name": ..., "size": ..., "last_modified": ...}
    """
    client = get_minio_client(cfg)
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    result = []
    for page in pages:
        for obj in page.get("Contents", []):
            result.append({
                "name": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"],
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

    Returns:
        Nom de l'objet le plus récent, ou None si le bucket est vide.
    """
    objects = list_objects(bucket, prefix, cfg)
    if not objects:
        return None

    latest = max(objects, key=lambda o: o["last_modified"])
    logger.info("Dernier objet dans '%s/%s' : %s", bucket, prefix, latest["name"])
    return latest["name"]