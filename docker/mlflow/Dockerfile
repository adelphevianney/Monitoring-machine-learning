# =============================================================================
# docker/mlflow/Dockerfile
# Image MLflow server — backend Postgres + artefacts MinIO
#
# L'image officielle ghcr.io/mlflow/mlflow:v2.10.0 inclut mlflow core
# mais pas les connecteurs pour Postgres ni pour S3/MinIO.
# Sans psycopg2 → crash au démarrage avec --backend-store-uri postgresql://
# Sans boto3    → impossible de pousser les artefacts vers MinIO
# =============================================================================

FROM ghcr.io/mlflow/mlflow:v2.10.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/mlflow.txt /tmp/requirements-mlflow.txt

RUN pip install --no-cache-dir -r /tmp/requirements-mlflow.txt

# Le CMD est hérité de l'image de base et surchargé dans docker-compose :
#   mlflow server
#     --backend-store-uri postgresql://mlops:mlops@postgres/mlops
#     --default-artifact-root s3://mlflow-artifacts/
#     --host 0.0.0.0 --port 5000