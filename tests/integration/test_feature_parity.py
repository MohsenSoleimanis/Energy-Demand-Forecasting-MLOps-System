"""Test that training and serving feature pipelines produce identical outputs (ML-009)."""

import pandas as pd
import pytest

from src.ml.features.feature_engineering import prepare_features


def test_feature_parity(sample_feature_base_df):
    df_train = prepare_features(sample_feature_base_df.copy(), mode="training")
    df_serve = prepare_features(sample_feature_base_df.copy(), mode="serving")

    assert list(df_train.columns) == list(df_serve.columns), "Column names differ"
    assert list(df_train.dtypes) == list(df_serve.dtypes), "Column dtypes differ"
    pd.testing.assert_frame_equal(df_train, df_serve)


def test_feature_parity_single_row():
    df = pd.DataFrame({
        "timestamp_brussels": [pd.Timestamp("2024-06-15 14:00:00")],
        "load_mw": [9500.0],
        "price_eur_mwh": [55.0],
        "temperature_2m": [22.5],
        "feels_like_temp": [23.0],
        "wind_speed_10m": [12.0],
        "wind_direction_10m": [180.0],
        "shortwave_radiation": [450.0],
        "precipitation": [0.0],
        "cloud_cover": [30.0],
        "pressure_msl": [1013.0],
        "renewable_share_pct": [35.0],
        "nuclear_mw": [4000.0],
        "gas_mw": [2000.0],
        "is_belgian_holiday": [False],
        "is_weekend": [False],
        "is_school_vacation": [False],
        "day_of_week": [5],
        "month": [6],
        "hour_of_day": [14],
    })

    train_result = prepare_features(df.copy(), mode="training")
    serve_result = prepare_features(df.copy(), mode="serving")

    pd.testing.assert_frame_equal(train_result, serve_result)
