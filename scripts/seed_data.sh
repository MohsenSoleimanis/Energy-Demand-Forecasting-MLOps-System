#!/bin/bash
set -euo pipefail
echo "=== Generating Synthetic Energy Data ==="
mkdir -p data/raw
python -c "
from energy_forecast.data.synthetic import SyntheticDataGenerator
gen = SyntheticDataGenerator(num_buildings=50, start_date='2020-01-01', end_date='2023-12-31', random_seed=42)
df = gen.generate()
df.to_parquet('data/raw/energy_data.parquet', index=False)
print(f'Generated {len(df):,} rows of data')
print(f'Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')
print(f'Buildings: {df[\"building_id\"].nunique()}')
print(f'Saved to data/raw/energy_data.parquet')
"
