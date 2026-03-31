"""Unit tests for feature engineering module (ML-002)."""

import numpy as np
import pandas as pd

from src.ml.features.engineering import (
    compute_cyclical_features,
    compute_interaction_features,
    compute_lag_features,
    compute_rolling_features,
    get_feature_columns,
    prepare_features,
)


class TestLagFeatures:
    def test_lag_features_use_only_past(self, sample_feature_base_df):
        df = compute_lag_features(sample_feature_base_df.copy())
        # First row should have NaN for all lags
        assert pd.isna(df["load_lag_24h"].iloc[0])
        # The lag_24h at index 24 should equal load_mw at index 0
        assert df["load_lag_24h"].iloc[24] == sample_feature_base_df["load_mw"].iloc[0]

    def test_lag_24h_correct(self, sample_feature_base_df):
        df = compute_lag_features(sample_feature_base_df.copy())
        assert df["load_lag_24h"].iloc[24] == sample_feature_base_df["load_mw"].iloc[0]

    def test_lag_168h_correct(self, sample_feature_base_df):
        df = compute_lag_features(sample_feature_base_df.copy())
        assert df["load_lag_168h"].iloc[168] == sample_feature_base_df["load_mw"].iloc[0]


class TestRollingFeatures:
    def test_rolling_mean_correct(self, sample_feature_base_df):
        df = compute_rolling_features(sample_feature_base_df.copy())
        idx = 30
        expected = sample_feature_base_df["load_mw"].iloc[idx - 23 : idx + 1].mean()
        actual = df["load_rolling_mean_24h"].iloc[idx]
        np.testing.assert_almost_equal(actual, expected, decimal=2)

    def test_rolling_std_non_negative(self, sample_feature_base_df):
        df = compute_rolling_features(sample_feature_base_df.copy())
        valid = df["load_rolling_std_24h"].dropna()
        assert (valid >= 0).all()


class TestCyclicalFeatures:
    def test_cyclical_encoding_range(self, sample_feature_base_df):
        df = compute_cyclical_features(sample_feature_base_df.copy())
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert df[col].min() >= -1.0
            assert df[col].max() <= 1.0

    def test_midnight_encoding(self):
        df = pd.DataFrame({"hour_of_day": [0], "month": [1]})
        df = compute_cyclical_features(df)
        np.testing.assert_almost_equal(df["hour_sin"].iloc[0], 0.0, decimal=5)
        np.testing.assert_almost_equal(df["hour_cos"].iloc[0], 1.0, decimal=5)


class TestInteractionFeatures:
    def test_interaction_computed(self, sample_feature_base_df):
        df = compute_interaction_features(sample_feature_base_df.copy())
        expected = sample_feature_base_df["temperature_2m"] * sample_feature_base_df["hour_of_day"]
        pd.testing.assert_series_equal(df["temp_x_hour"], expected, check_names=False)


class TestPrepareFeatures:
    def test_training_serving_output_parity(self, sample_feature_base_df):
        df_train = prepare_features(sample_feature_base_df.copy(), mode="training")
        df_serve = prepare_features(sample_feature_base_df.copy(), mode="serving")
        assert list(df_train.columns) == list(df_serve.columns)
        assert list(df_train.dtypes) == list(df_serve.dtypes)

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame()
        result = prepare_features(df, mode="training")
        assert len(result) == 0

    def test_feature_names_consistent(self, sample_feature_base_df):
        df = prepare_features(sample_feature_base_df.copy(), mode="training")
        expected_features = get_feature_columns()
        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_handles_nulls_gracefully(self, sample_feature_base_df):
        df = sample_feature_base_df.copy()
        df.loc[5, "temperature_2m"] = np.nan
        df.loc[10, "load_mw"] = np.nan
        result = prepare_features(df, mode="training")
        assert len(result) == len(df)
