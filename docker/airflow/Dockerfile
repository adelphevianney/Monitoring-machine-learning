# =============================================================================
# docker/airflow/Dockerfile
# Image Airflow custom — MLOps Drift Monitoring v2
#
# Utilisé par : airflow-webserver, airflow-scheduler, airflow-init
#
# CHANGEMENTS vs Dockerfile original
# ───────────────────────────────────
# - psycopg2-binary à la place de psycopg2 compilé
#   → libpq-dev et gcc/g++ ne sont plus nécessaires
#   → couche apt réduite à curl uniquement (healthcheck)
# - requirements/airflow.txt à la place de requirements.txt global
#   → dépendances propres à ce conteneur
# - minio SDK supprimé (migration boto3 uniquement en v2)
# =============================================================================

FROM apache/airflow:2.8.1-python3.11

USER root

RUN mkdir -p /var/lib/apt/lists/partial \
    && chmod -R 755 /var/lib/apt \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# UID 50000 = utilisateur airflow dans l'image apache/airflow officielle
USER 50000

COPY --chown=50000:0 requirements/airflow.txt /tmp/requirements-airflow.txt

RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt