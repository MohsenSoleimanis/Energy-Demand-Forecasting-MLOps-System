-- Create the MLflow database if it does not already exist.
-- Note: PostgreSQL entrypoint already creates POSTGRES_DB, but this
-- script ensures the database and proper permissions are in place.

SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
