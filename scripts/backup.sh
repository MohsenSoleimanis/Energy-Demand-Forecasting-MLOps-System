#!/bin/bash
# Database backup script for PostgreSQL (MLflow metadata)
# Schedule via cron: 0 2 * * * /path/to/backup.sh
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-mlops-postgres}"

mkdir -p "$BACKUP_DIR"

echo "Backing up PostgreSQL (MLflow metadata)..."
docker exec "$POSTGRES_CONTAINER" pg_dump -U "${POSTGRES_USER:-mlflow}" "${POSTGRES_DB:-mlflow}" | gzip > "$BACKUP_DIR/mlflow_db_${TIMESTAMP}.sql.gz"

echo "Backing up DuckDB..."
if [ -f "src/data_platform/dbt_project/energy_demand.duckdb" ]; then
    cp "src/data_platform/dbt_project/energy_demand.duckdb" "$BACKUP_DIR/energy_demand_${TIMESTAMP}.duckdb"
fi

# Cleanup backups older than 30 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.duckdb" -mtime +30 -delete

echo "Backup complete: $BACKUP_DIR"
ls -lh "$BACKUP_DIR"
