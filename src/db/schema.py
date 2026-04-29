"""
Schéma de la BD — migrations idempotentes lancées au début du service.
utilsation de psycopg2
"""
from __future__ import annotations

from venv import logger

import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.config.settings import PG_CFG


# --------------------------------------------------------------------------- #
# DDL                                                                         #
# --------------------------------------------------------------------------- #

tables = "../sql/app_tables.sql"
_SCHEMA_SQL = tables


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def get_connection() -> PgConnection:
    """retourne une connection psycopg2 brut (la fermeture est la responsabilité du service qui l'appelle)."""
    return psycopg2.connect(PG_CFG.db.dsn)


def run_migrations() -> None:
    """applique le schéma, c'est safe de l'appeler à chaque démarrage (à cause de l'idempotence)."""
    logger.info("Running database migrations")
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
        logger.info("Database migrations completed successfully")
    except Exception as exc:
        logger.error("Migration failed")
        raise
    finally:
        conn.close()