"""Tests for SyntheticDataGenerator."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecast.data.synthetic import SyntheticDataGenerator

pytestmark = pytest.mark.unit


class TestSyntheticDataGenerator:
    """Tests for the SyntheticDataGenerator class."""

    def test_generate_returns_dataframe(self):
        gen = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-01-07",
            random_seed=42,
        )
        result = gen.generate()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_generate_has_required_columns(self):
        gen = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-01-07",
            random_seed=42,
        )
        result = gen.generate()
        required = {
            "timestamp",
            "building_id",
            "energy_demand_kwh",
            "temperature",
            "humidity",
            "is_holiday",
            "day_of_week",
            "month",
            "hour",
        }
        assert required.issubset(set(result.columns))

    def test_generate_correct_date_range(self):
        gen = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-03-01",
            end_date="2023-03-10",
            random_seed=42,
        )
        result = gen.generate()
        ts = pd.to_datetime(result["timestamp"])
        assert ts.min() == pd.Timestamp("2023-03-01 00:00:00")
        assert ts.max() == pd.Timestamp("2023-03-10 00:00:00")

    def test_generate_reproducible_with_seed(self):
        kwargs = dict(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-01-03",
            random_seed=123,
        )
        df1 = SyntheticDataGenerator(**kwargs).generate()
        df2 = SyntheticDataGenerator(**kwargs).generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_positive_energy_values(self, sample_energy_df):
        assert (sample_energy_df["energy_demand_kwh"] > 0).all()

    def test_generate_with_drift(self):
        gen = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-06-30",
            random_seed=42,
        )
        df_clean = gen.generate()
        # Re-create generator with same seed for drift version
        gen2 = SyntheticDataGenerator(
            num_buildings=1,
            start_date="2023-01-01",
            end_date="2023-06-30",
            random_seed=42,
        )
        df_drift = gen2.generate_with_drift(drift_start_frac=0.5, drift_magnitude=0.3)

        assert len(df_drift) == len(df_clean)
        # The second half should have higher demand on average due to drift
        n = len(df_clean)
        second_half = slice(n // 2, n)
        mean_clean = df_clean.iloc[second_half]["energy_demand_kwh"].mean()
        mean_drift = df_drift.iloc[second_half]["energy_demand_kwh"].mean()
        assert mean_drift > mean_clean
