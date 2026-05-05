# MLOps Drift Monitoring — Guide de démarrage

## à modifier
changer la fonction train pour ne plus enregistrer vers mlflow mais vers minio pour l'instant.
ou carrément enregistrer les modèles en local de mem que les sorties html et autres.

## Architecture

Lambda architecture simplifiée avec :
- **Airflow** — orchestration des pipelines batch
- **MLflow** — tracking des expérimentations + model registry
- **MinIO** — data lake objet (datasets, artefacts, snapshots)
- **Postgres** — backend MLflow + tables applicatives de monitoring

```
Airflow (DAGs)
    ├── training_pipeline      [daily]
    │     generate → train → log MLflow → stats référence → registry
    │
    └── drift_monitoring_pipeline   [hourly]
          load modèle prod → collect data → PSI/KS → store → alerte → [ré-entraîner]
```

---

## Démarrage rapide

### 1. Prérequis

- Docker Desktop >= 4.0 avec au moins **4 GB RAM**
- Ports libres : `5432`, `5000`, `8080`, `9000`, `9001`

### 2. Lancer la stack

```bash
# Cloner / se placer dans le dossier du projet
cd monitoring_ML

# Rendre le script postgres exécutable
chmod +x src/db/init-multiple-dbs.sh

# Démarrer tous les services
docker-compose build
docker-compose up -d

# Vérifier que tout est healthy (attendre ~2 min)
docker-compose ps
```

### 3. Accès aux interfaces

| Service    | URL                        | Identifiants          |
|------------|----------------------------|-----------------------|
| Airflow UI | http://localhost:8080      | admin / admin         |
| MLflow UI  | http://localhost:5000      | (pas d'auth)          |
| MinIO UI   | http://localhost:9001      | minioadmin / minioadmin |
| Postgres   | localhost:5432             | mlops / mlops         |

---

## Lancer les pipelines

### Via l'UI Airflow

1. Ouvrir http://localhost:8080
2. Activer le DAG `training_pipeline` (toggle ON)
3. Cliquer sur le bouton ▶️ "Trigger DAG" pour le lancer manuellement
4. Attendre la fin (toutes les tâches en vert)
5. Activer le DAG `drift_monitoring_pipeline`

### Via la CLI

```bash
# Déclencher le training manuellement
docker-compose exec airflow-scheduler \
  airflow dags trigger training_pipeline

# Déclencher le monitoring manuellement
docker-compose exec airflow-scheduler \
  airflow dags trigger drift_monitoring_pipeline
```

---

## Simuler un drift

Le drift est contrôlé par la variable Airflow `DRIFT_SIMULATION_FACTOR` (0.0 à 1.0).

**Via l'UI Airflow :**
1. Admin → Variables
2. Modifier `DRIFT_SIMULATION_FACTOR`
3. Mettre `0.5` pour un drift modéré, `1.0` pour un drift complet
4. Déclencher manuellement `drift_monitoring_pipeline`

**Via la CLI :**
```bash
docker-compose exec airflow-scheduler \
  airflow variables set DRIFT_SIMULATION_FACTOR 1.0
```

**Scénario de test recommandé :**

```bash
# Étape 1 : entraîner un modèle de référence
airflow dags trigger training_pipeline

# Étape 2 : monitoring sans drift (doit rester stable)
airflow variables set DRIFT_SIMULATION_FACTOR 0.0
airflow dags trigger drift_monitoring_pipeline   # → alert=none

# Étape 3 : drift modéré
airflow variables set DRIFT_SIMULATION_FACTOR 0.5
airflow dags trigger drift_monitoring_pipeline   # → alert=warning possible

# Étape 4 : drift complet (répéter 3 fois pour déclencher le ré-entraînement)
airflow variables set DRIFT_SIMULATION_FACTOR 1.0
airflow dags trigger drift_monitoring_pipeline   # → alert=critical (1/3)
airflow dags trigger drift_monitoring_pipeline   # → alert=critical (2/3)
airflow dags trigger drift_monitoring_pipeline   # → alert=critical (3/3) → RETRAIN
```

---

## Structure du projet

```
mlops_drift/
├── dags/
│   ├── training_pipeline.py          # DAG 1 : entraînement
│   └── drift_monitoring_pipeline.py  # DAG 2 : monitoring
├── src/
│   ├── config/settings.py            # configuration centralisée
│   ├── data_generation/generator.py  # génération de données synthétiques
│   ├── training/train.py             # pipeline d'entraînement
│   ├── monitoring/
│   │   ├── drift_metrics.py          # calcul PSI, KS, deltas
│   │   └── alerting.py               # règles de décision
│   └── storage/
│       ├── postgres_client.py        # opérations SQL
│       └── minio_client.py           # upload/download MinIO
├── sql/
│   └── app_tables.sql                # 4 tables + 2 vues
├── tests/
│   ├── test_core.py                  # tests generator + training
│   ├── test_storage_alerting.py      # tests alerting + storage helpers
│   └── test_quick.py                 # test rapide local sans MLflow
├── docker/
│   └── postgres/init-multiple-dbs.sh # init Postgres multi-bases
├── docker-compose.yml
└── requirements.txt
```

---

## Arrêt et nettoyage

```bash
# Arrêter sans supprimer les données
docker-compose down

# Arrêter ET supprimer tous les volumes (repart de zéro)
docker-compose down -v
```

---

## Prochaines étapes possibles

- **Grafana** — dashboard visuel des métriques de drift dans le temps
- **Slack/email** — notifications dans `alerting.py` et `log_alert_only`
- **Couche Kappa** — ajouter Kafka/Redpanda pour monitoring quasi temps réel
- **Serving** — API FastAPI qui logue chaque requête d'inférence dans MinIO
