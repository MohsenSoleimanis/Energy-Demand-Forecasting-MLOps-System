.PHONY: install install-dev test test-unit test-integration test-performance lint format type-check clean docker-up docker-down train serve seed help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	pip install -e .

install-dev: ## Install package with all dev dependencies
	pip install -e ".[dev,test]"

test: ## Run all tests
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short -m unit

test-integration: ## Run integration tests only
	pytest tests/integration/ -v --tb=short -m integration

test-performance: ## Run performance benchmark tests
	pytest tests/performance/ -v --tb=short -m performance

test-cov: ## Run tests with coverage report
	pytest tests/ -v --tb=short --cov=src/energy_forecast --cov-report=html --cov-report=term-missing

lint: ## Run linter (ruff)
	ruff check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run type checker (mypy)
	mypy src/energy_forecast/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-up: ## Start all services with Docker Compose
	docker compose up -d --build

docker-down: ## Stop all services
	docker compose down -v

train: ## Run training pipeline locally
	python -m energy_forecast.pipelines.training_pipeline

serve: ## Start the prediction API locally
	uvicorn energy_forecast.serving.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

seed: ## Generate synthetic data and run initial training
	python -c "from energy_forecast.data.synthetic import SyntheticDataGenerator; SyntheticDataGenerator().generate().to_parquet('data/raw/energy_data.parquet')"

monitor: ## Generate monitoring report
	python -c "from energy_forecast.monitoring.drift import DriftDetector; DriftDetector().generate_report()"
