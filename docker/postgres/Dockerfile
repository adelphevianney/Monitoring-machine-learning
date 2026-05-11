# =============================================================================
# docker/postgres/Dockerfile
# Image Postgres avec scripts d'initialisation embarqués.
#
# Aucune dépendance Python — Postgres est un service système pur.
#
# ORDRE D'EXÉCUTION docker-entrypoint-initdb.d/ (alphabétique) :
#   1_init_databases.sh  → crée les bases airflow + mlops avec leurs users
#   2_app_tables.sql     → crée les tables applicatives dans la base mlops
#
# Ces scripts ne s'exécutent QU'UNE SEULE FOIS à la création du volume.
# Si le volume postgres-data existe déjà → scripts ignorés.
# Pour forcer la réinitialisation : docker compose down -v
# =============================================================================

FROM postgres:15-alpine

USER root

COPY src/db/init-multiple-dbs.sh /docker-entrypoint-initdb.d/1_init_databases.sh
COPY sql/app_tables.sql           /docker-entrypoint-initdb.d/2_app_tables.sql

RUN chmod +x /docker-entrypoint-initdb.d/1_init_databases.sh

# Variables fournies par docker-compose :
#   POSTGRES_USER                 : postgres
#   POSTGRES_PASSWORD             : postgres
#   POSTGRES_MULTIPLE_DATABASES   : airflow:airflow:airflow,mlops:mlops:mlops