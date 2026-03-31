"""
Cross-platform CLI to run the Belgian Energy Demand Forecasting system.
Works on Windows, macOS, and Linux without Make.

Usage:
    python run.py up                  Start all Docker services
    python run.py down                Stop all Docker services
    python run.py ingest              Run all data ingestion (ENTSO-E + weather + calendar)
    python run.py ingest --source X   Run single ingestion (entsoe|weather|forecast|calendar)
    python run.py transform           Run dbt (bronze -> silver -> gold)
    python run.py train               Run full training pipeline (DVC)
    python run.py serve               Start FastAPI prediction server
    python run.py test                Run all tests
    python run.py monitor             Generate monitoring reports
    python run.py status              Check service health
    python run.py setup               Full setup: up -> ingest -> transform -> train
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DBT_DIR = ROOT / "src" / "data_platform" / "dbt_project"


def run(cmd: str, cwd: str | Path | None = None, check: bool = True) -> int:
    """Run a shell command, streaming output in real time."""
    print(f"\n{'='*60}")
    print(f"  Running: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if check and result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def check_tool(name: str, install_hint: str) -> bool:
    """Check if a CLI tool is available."""
    if shutil.which(name) is None:
        print(f"Error: '{name}' not found. {install_hint}")
        return False
    return True


def ensure_env():
    """Ensure .env exists and load it into os.environ."""
    env_file = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env_file.exists():
        if example.exists():
            shutil.copy(example, env_file)
            print("Created .env from .env.example")
            print("IMPORTANT: Edit .env and add your ENTSOE_API_KEY before running ingestion.")
        else:
            print("Error: No .env or .env.example found.")
            sys.exit(1)
    # Load all env vars using shared config
    from src.shared.config import load_env_file
    load_env_file(str(env_file))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_up(_args):
    """Start all Docker infrastructure services."""
    if not check_tool("docker", "Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"):
        sys.exit(1)
    ensure_env()
    run("docker compose up -d")
    print("\nServices starting. Waiting for health checks...")
    time.sleep(5)
    cmd_status(_args)


def cmd_down(_args):
    """Stop all Docker services."""
    run("docker compose down")


def cmd_status(_args):
    """Check health of all services."""
    import urllib.request

    services = {
        "MLflow":     "http://localhost:5000/health",
        "MinIO":      "http://localhost:9000/minio/health/live",
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana":    "http://localhost:3000/api/health",
    }
    print("\nService Status:")
    print("-" * 40)
    for name, url in services.items():
        try:
            req = urllib.request.urlopen(url, timeout=3)
            status = "UP" if req.status == 200 else f"HTTP {req.status}"
            print(f"  {name:15s} {status}")
        except Exception:
            print(f"  {name:15s} DOWN")

    print()
    print("  MLflow UI:      http://localhost:5000")
    print("  MinIO Console:  http://localhost:9001  (see .env for credentials)")
    print("  Prometheus:     http://localhost:9090")
    print("  Grafana:        http://localhost:3000  (admin/admin)")


def cmd_ingest(args):
    """Run data ingestion scripts."""
    ensure_env()
    source = getattr(args, "source", None)

    if source is None or source == "entsoe":
        print("\n--- Ingesting ENTSO-E data (load, price, generation) ---")
        run(f"{sys.executable} -m src.data_platform.ingestion.ingest_entsoe")

    if source is None or source == "weather":
        print("\n--- Ingesting weather actuals ---")
        run(f"{sys.executable} -m src.data_platform.ingestion.ingest_weather")

    if source is None or source == "forecast":
        print("\n--- Ingesting weather forecast ---")
        run(f"{sys.executable} -m src.data_platform.ingestion.ingest_weather_forecast")

    if source is None or source == "calendar":
        print("\n--- Generating Belgian calendar ---")
        run(f"{sys.executable} -m src.data_platform.ingestion.generate_calendar")

    print("\nIngestion complete.")


def cmd_transform(_args):
    """Run dbt transformations (bronze -> silver -> gold)."""
    ensure_env()
    if not check_tool("dbt", "Install: pip install dbt-core dbt-duckdb"):
        sys.exit(1)
    run("dbt deps", cwd=DBT_DIR)
    run("dbt run", cwd=DBT_DIR)
    run("dbt test", cwd=DBT_DIR)
    print("\nTransformation complete.")


def cmd_train(args):
    """Run the ML training pipeline."""
    ensure_env()
    tune_flag = " --tune" if getattr(args, "tune", False) else ""
    if shutil.which("dvc"):
        run("dvc repro")
    else:
        run(f"{sys.executable} -m src.ml.training.pull_gold", check=False)
        run(f"{sys.executable} -m src.ml.training.train{tune_flag}")
    print("\nTraining complete. Check MLflow at http://localhost:5000")


def cmd_serve(_args):
    """Start the FastAPI prediction server.

    Runs inside Docker so Prometheus can scrape metrics and Grafana
    shows real dashboards. Use 'docker compose logs -f api' to see logs.
    """
    ensure_env()
    # Rebuild and start the API container
    run("docker compose up -d --build api")
    print("\n  API is running in Docker.")
    print("  API docs:    http://localhost:8000/docs")
    print("  Health:      http://localhost:8000/health")
    print("  Metrics:     http://localhost:8000/metrics")
    print("  Grafana:     http://localhost:3000")
    print("  API logs:    docker compose logs -f api")
    print()
    print("  To stop:     docker compose stop api")
    print("  To restart:  docker compose restart api")


def cmd_test(_args):
    """Run all tests."""
    run(f"{sys.executable} -m pytest tests/ -v")


def cmd_monitor(_args):
    """Generate monitoring reports."""
    ensure_env()
    run(f"{sys.executable} -m src.ml.monitoring.build_monitoring_set", check=False)
    run(f"{sys.executable} -m src.ml.monitoring.drift_report", check=False)
    run(f"{sys.executable} -m src.ml.monitoring.performance_report", check=False)


def cmd_setup(args):
    """Full setup: start services, ingest data, transform, train."""
    print("=" * 60)
    print("  FULL SYSTEM SETUP")
    print("  This will: start Docker -> ingest real data -> dbt -> train")
    print("=" * 60)

    cmd_up(args)
    print("\nWaiting for services to be fully ready...")
    time.sleep(10)

    cmd_ingest(args)
    cmd_transform(args)
    cmd_train(args)

    print("\n" + "=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print("\n  Start the API:  python run.py serve")
    print("  MLflow UI:      http://localhost:5000")
    print("  API docs:       http://localhost:8000/docs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Belgian Energy Demand Forecasting - System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("up", help="Start Docker services")
    sub.add_parser("down", help="Stop Docker services")
    sub.add_parser("status", help="Check service health")

    ingest_parser = sub.add_parser("ingest", help="Run data ingestion")
    ingest_parser.add_argument(
        "--source", choices=["entsoe", "weather", "forecast", "calendar"],
        help="Run only one source (default: all)",
    )

    sub.add_parser("transform", help="Run dbt transformations")
    train_parser = sub.add_parser("train", help="Train ML model")
    train_parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning before final training",
    )
    sub.add_parser("serve", help="Start prediction API")
    sub.add_parser("test", help="Run tests")
    sub.add_parser("monitor", help="Generate monitoring reports")
    sub.add_parser("setup", help="Full setup: up -> ingest -> transform -> train")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "up": cmd_up,
        "down": cmd_down,
        "status": cmd_status,
        "ingest": cmd_ingest,
        "transform": cmd_transform,
        "train": cmd_train,
        "serve": cmd_serve,
        "test": cmd_test,
        "monitor": cmd_monitor,
        "setup": cmd_setup,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
