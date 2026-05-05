FROM apache/airflow:2.8.1-python3.11

USER root

RUN mkdir -p /var/lib/apt/lists/partial \
    && chmod -R 755 /var/lib/apt \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        g++ \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# UID 50000 = utilisateur airflow dans l'image apache/airflow officielle
USER 50000

COPY --chown=50000:0 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt