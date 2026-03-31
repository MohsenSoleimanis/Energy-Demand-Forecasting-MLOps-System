#!/bin/bash
# Restore PostgreSQL from backup
set -euo pipefail

BACKUP_FILE="${1:?Usage: restore.sh <backup_file.sql.gz>}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-mlops-postgres}"

echo "Restoring from $BACKUP_FILE..."
gunzip -c "$BACKUP_FILE" | docker exec -i "$POSTGRES_CONTAINER" psql -U "${POSTGRES_USER:-mlflow}" "${POSTGRES_DB:-mlflow}"
echo "Restore complete."
