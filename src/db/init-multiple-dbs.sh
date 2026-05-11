#!/bin/bash
# =============================================================================
# Crée plusieurs bases de données Postgres au démarrage du conteneur,
# puis applique le schéma applicatif sur la base mlops.
#
# Format de POSTGRES_MULTIPLE_DATABASES :
#   "db1:user1:password1,db2:user2:password2"
#
# Bases créées :
#   - airflow : user airflow  (métadonnées internes Airflow)
#   - mlops   : user mlops   (tables applicatives du projet)
#
# POURQUOI NE PAS UTILISER \connect DANS app_tables.sql ?
# ────────────────────────────────────────────────────────
# psql peut mal interpréter \connect dans certains contextes d'exécution
# (docker-entrypoint-initdb.d), surtout avec des commentaires en français
# qui contiennent des mots ressemblant à des options de connexion.
# On contourne ça en passant -d mlops directement à psql ici.
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

# ── Création des bases ────────────────────────────────────────────────────────
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

# ── Application du schéma applicatif sur la base mlops ───────────────────────
# On exécute app_tables.sql directement avec -d mlops pour éviter tout
# problème de \connect. Le fichier est copié dans l'image par le Dockerfile.
SQL_FILE="/docker-entrypoint-initdb.d/2_app_tables.sql"

if [ -f "$SQL_FILE" ]; then
    echo "Application du schéma applicatif sur la base 'mlops'..."
    psql -v ON_ERROR_STOP=1 \
         --username "$POSTGRES_USER" \
         --dbname "mlops" \
         --file "$SQL_FILE"
    echo "  ✅ Schéma applicatif appliqué."
else
    echo "  ⚠️  $SQL_FILE introuvable — schéma non appliqué."
fi