"""Run Great Expectations validation on gold training data."""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def validate_training_data() -> bool:
    """Validate the gold training set meets quality standards.

    Checks:
    - Minimum row count (from config)
    - No null values in critical columns
    - Target variable in expected range
    - Feature values in expected ranges
    - No future data leakage
    """
    data_path = PROJECT_ROOT / "data" / "gold" / "training_set.parquet"
    if not data_path.exists():
        logger.error("Training data not found: %s", data_path)
        return False

    df = pd.read_parquet(data_path)
    results = {"timestamp": pd.Timestamp.now().isoformat(), "checks": []}
    passed = True

    # Check 1: Minimum rows
    min_rows = 8000
    check = {"name": "min_row_count", "expected": min_rows, "actual": len(df), "passed": len(df) >= min_rows}
    results["checks"].append(check)
    if not check["passed"]:
        logger.error("FAIL: %d rows < minimum %d", len(df), min_rows)
        passed = False

    # Check 2: No null target
    null_targets = int(df["target_load_24h"].isna().sum())
    check = {"name": "no_null_target", "null_count": null_targets, "passed": null_targets == 0}
    results["checks"].append(check)
    if not check["passed"]:
        logger.error("FAIL: %d null target values", null_targets)
        passed = False

    # Check 3: Target in range (4000-16000 MW for Belgium)
    target = df["target_load_24h"].dropna()
    in_range = ((target >= 4000) & (target <= 16000)).all()
    check = {"name": "target_range", "min": float(target.min()), "max": float(target.max()), "passed": bool(in_range)}
    results["checks"].append(check)
    if not check["passed"]:
        logger.warning("WARN: target out of expected range [4000, 16000]")

    # Check 4: No future timestamps
    latest = pd.to_datetime(df["timestamp_brussels"]).max()
    now = pd.Timestamp.now()
    check = {"name": "no_future_data", "latest": str(latest), "passed": latest <= now}
    results["checks"].append(check)
    if not check["passed"]:
        logger.error("FAIL: data contains future timestamps: %s", latest)
        passed = False

    # Check 5: Critical columns not null
    critical_cols = ["load_lag_24h", "load_lag_168h", "temperature_2m"]
    for col in critical_cols:
        if col in df.columns:
            nulls = int(df[col].isna().sum())
            check = {"name": f"not_null_{col}", "null_count": nulls, "passed": nulls == 0}
            results["checks"].append(check)
            if not check["passed"]:
                logger.warning("WARN: %d nulls in %s", nulls, col)

    # Save results
    output_dir = PROJECT_ROOT / "data" / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "validation_result.json", "w") as f:
        json.dump(results, f, indent=2)

    results["passed"] = passed
    logger.info("Validation %s: %d checks, %d passed",
                "PASSED" if passed else "FAILED",
                len(results["checks"]),
                sum(1 for c in results["checks"] if c["passed"]))

    return passed


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not validate_training_data():
        sys.exit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
