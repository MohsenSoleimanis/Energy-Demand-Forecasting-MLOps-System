#!/bin/bash
set -euo pipefail
echo "=== Running Initial Model Training ==="
python -m energy_forecast.pipelines.training_pipeline
echo "=== Training Complete ==="
