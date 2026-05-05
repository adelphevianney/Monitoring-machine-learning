"""
Schéma de la BD — migrations idempotentes lancées au début du service.
"""
from __future__ import annotations

import logging
from pathlib import Path

import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.config.settings import PG_CFG

logger = logging.getLogger(__name__)

# Chemin absolu vers le SQL, indépendant du répertoire de lancement
_SQL_PATH = Path(__file__).resolve().parent.parent / "sql" / "app_tables.sql"


def get_connection() -> PgConnection:
    """Retourne une connexion psycopg2 brute (la fermeture est la responsabilité de l'appelant)."""
    return psycopg2.connect(PG_CFG.dsn)


def run_migrations() -> None:
    """Applique le schéma. Safe à appeler à chaque démarrage (idempotent grâce aux DROP IF EXISTS)."""
    logger.info("Running database migrations from %s", _SQL_PATH)
    sql = _SQL_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.info("Database migrations completed successfully")
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        conn.close()