#!/bin/bash
# =============================================================================
# Crée plusieurs bases de données Postgres au démarrage du conteneur.
#
# Format de la variable POSTGRES_MULTIPLE_DATABASES :
#   "db1:user1:password1,db2:user2:password2"
#
# Utilisé dans docker-compose.yml pour créer :
#   - base "airflow"  avec user "airflow"  (pour Airflow)
#   - base "mlops"    avec user "mlops"    (pour MLflow + tables applicatives)
# =============================================================================

set -e

function create_db_and_user() {
    local db=$1
    local user=$2
    local password=$3

    echo "Création de la base '$db' et de l'utilisateur '$user'..."

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE USER $user WITH PASSWORD '$password';
        CREATE DATABASE $db OWNER $user;
        GRANT ALL PRIVILEGES ON DATABASE $db TO $user;
EOSQL

    echo "  ✅ Base '$db' créée."
}

# Parser POSTGRES_MULTIPLE_DATABASES : "db1:user1:pw1,db2:user2:pw2"
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Initialisation des bases multiples..."
    for entry in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
        db=$(echo "$entry" | cut -d: -f1)
        user=$(echo "$entry" | cut -d: -f2)
        password=$(echo "$entry" | cut -d: -f3)
        create_db_and_user "$db" "$user" "$password"
    done
    echo "Toutes les bases sont créées."
fi
