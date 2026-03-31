-- Create the MLflow database if it does not already exist.
-- Note: PostgreSQL entrypoint already creates POSTGRES_DB, but this
-- script ensures the database and proper permissions are in place.

SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Energy warehouse database for gold table serving (feature store, API)
SELECT 'CREATE DATABASE energy_warehouse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'energy_warehouse')\gexec

GRANT ALL PRIVILEGES ON DATABASE energy_warehouse TO mlflow;

\c energy_warehouse;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS bronze;
