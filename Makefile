.PHONY: up down ingest transform train serve test monitor lint validate-bronze dbt-docs

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python -m src.data_platform.ingestion.entsoe
	python -m src.data_platform.ingestion.weather
	python -m src.data_platform.ingestion.weather_forecast
	python -m src.data_platform.ingestion.calendar

transform:
	cd src/data_platform/dbt_project && dbt run
	cd src/data_platform/dbt_project && dbt test

train:
	python -m src.ml.training.pull_gold
	python -m src.ml.training.train

serve:
	uvicorn src.ml.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src

monitor:
	python -m src.ml.monitoring.drift_report
	python -m src.ml.monitoring.performance_report

lint:
	ruff check src/ tests/

validate-bronze:
	great_expectations checkpoint run pipeline_checkpoint

dbt-docs:
	cd src/data_platform/dbt_project && dbt docs generate && dbt docs serve
