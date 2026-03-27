.PHONY: up down ingest transform train serve test monitor lint validate-bronze dbt-docs

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python -m src.ingestion.entsoe_fetcher

transform:
	python -m src.transformation.bronze_to_silver

train:
	python -m src.training.train

serve:
	python -m src.serving.app

test:
	pytest tests/ -v

monitor:
	python -m src.monitoring.drift_detector

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

validate-bronze:
	python -m src.validation.bronze_validation

dbt-docs:
	cd dbt && dbt docs generate && dbt docs serve
