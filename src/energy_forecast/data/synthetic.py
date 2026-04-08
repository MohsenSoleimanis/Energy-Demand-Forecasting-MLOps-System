"""Realistic synthetic energy-demand data generator.

Produces hourly energy consumption data with proper daily, weekly, and yearly
seasonality, weather correlations, holiday effects, and configurable noise.
All operations are numpy-vectorized for speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import holidays as holidays_pkg
import numpy as np
import pandas as pd


@dataclass
class SeasonalWeights:
    """Relative amplitudes for each seasonality component."""

    daily: float = 1.0
    weekly: float = 0.3
    yearly: float = 0.5


@dataclass
class WeatherParams:
    """Parameters governing synthetic weather generation."""

    temp_mean_winter: float = 2.0  # degC
    temp_mean_summer: float = 28.0  # degC
    temp_daily_range: float = 8.0  # degC peak-to-trough within a day
    temp_noise_std: float = 3.0
    humidity_mean: float = 55.0  # %
    humidity_std: float = 15.0
    humidity_temp_corr: float = -0.4  # negative: hotter → drier


class SyntheticDataGenerator:
    """Generate realistic hourly energy-demand data for multiple buildings.

    Parameters
    ----------
    num_buildings : int
        Number of distinct buildings to simulate.
    start_date : str | datetime
        First timestamp (inclusive).  Strings are parsed as ISO format.
    end_date : str | datetime
        Last timestamp (inclusive).
    frequency : str
        Pandas offset alias for the time index (default ``"h"`` = hourly).
    random_seed : int
        Seed for reproducibility.
    noise_level : float
        Standard deviation of additive Gaussian noise relative to base load
        (default 0.05 = 5 %).
    seasonal_weights : SeasonalWeights | None
        Override default seasonal amplitudes.
    weather_params : WeatherParams | None
        Override default weather-generation parameters.
    """

    def __init__(
        self,
        num_buildings: int = 5,
        start_date: str | datetime = "2022-01-01",
        end_date: str | datetime = "2023-12-31",
        frequency: str = "h",
        random_seed: int = 42,
        noise_level: float = 0.05,
        seasonal_weights: SeasonalWeights | None = None,
        weather_params: WeatherParams | None = None,
    ) -> None:
        self.num_buildings = num_buildings
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.frequency = frequency
        self.random_seed = random_seed
        self.noise_level = noise_level
        self.seasonal_weights = seasonal_weights or SeasonalWeights()
        self.weather = weather_params or WeatherParams()

        self._rng = np.random.default_rng(self.random_seed)

        # Pre-compute US holidays for the full date range.
        years = range(self.start_date.year, self.end_date.year + 1)
        self._us_holidays: set[datetime] = set()
        for yr in years:
            self._us_holidays.update(holidays_pkg.US(years=yr).keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate a clean energy-demand DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, building_id, energy_demand_kwh, temperature,
            humidity, is_holiday, day_of_week, month, hour``
        """
        timestamps = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.frequency
        )
        n_times = len(timestamps)

        # Assign a random base load per building.
        base_loads = self._rng.uniform(10.0, 100.0, size=self.num_buildings)

        all_frames: list[pd.DataFrame] = []

        for b_idx in range(self.num_buildings):
            building_id = f"building_{b_idx + 1:03d}"
            base = base_loads[b_idx]

            # --- Time features (vectorised) ---
            hours = timestamps.hour.to_numpy(dtype=np.float64)
            dow = timestamps.dayofweek.to_numpy(dtype=np.float64)  # 0=Mon
            month = timestamps.month.to_numpy(dtype=np.float64)
            day_of_year = timestamps.dayofyear.to_numpy(dtype=np.float64)

            # --- Weather ---
            temperature, humidity = self._generate_weather(
                n_times, day_of_year, hours
            )

            # --- Seasonality ---
            daily = self._daily_seasonality(hours)
            weekly = self._weekly_seasonality(dow)
            yearly = self._yearly_seasonality(day_of_year)

            sw = self.seasonal_weights
            seasonal = (
                1.0
                + sw.daily * daily
                + sw.weekly * weekly
                + sw.yearly * yearly
            )

            # --- Weather effect on demand ---
            weather_effect = self._weather_demand_effect(temperature, humidity)

            # --- Holiday effect ---
            is_holiday = np.array(
                [ts.date() in self._us_holidays for ts in timestamps], dtype=bool
            )
            holiday_factor = np.where(is_holiday, 0.6, 1.0)

            # --- Compose demand ---
            demand = base * seasonal * weather_effect * holiday_factor

            # --- Additive noise ---
            noise = self._rng.normal(0, self.noise_level * base, size=n_times)
            demand = np.maximum(demand + noise, 0.1)  # floor at 0.1 kWh

            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "building_id": building_id,
                    "energy_demand_kwh": demand,
                    "temperature": temperature,
                    "humidity": humidity,
                    "is_holiday": is_holiday,
                    "day_of_week": dow.astype(int),
                    "month": month.astype(int),
                    "hour": hours.astype(int),
                }
            )
            all_frames.append(df)

        result = pd.concat(all_frames, ignore_index=True)
        result.sort_values(["building_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def generate_with_drift(
        self,
        drift_start_frac: float = 0.5,
        drift_magnitude: float = 0.15,
    ) -> pd.DataFrame:
        """Generate data with gradual concept drift.

        After ``drift_start_frac`` of the time range has elapsed the demand
        begins a linear increase of up to ``drift_magnitude`` times the
        base demand.  Useful for testing monitoring / retraining pipelines.

        Parameters
        ----------
        drift_start_frac : float
            Fraction of the time range after which drift begins (default 0.5).
        drift_magnitude : float
            Maximum relative increase at the end of the series (default 0.15).

        Returns
        -------
        pd.DataFrame
            Same schema as :meth:`generate`.
        """
        df = self.generate()

        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        total_seconds = (ts_max - ts_min).total_seconds()
        drift_start_seconds = drift_start_frac * total_seconds

        elapsed = (df["timestamp"] - ts_min).dt.total_seconds().to_numpy()
        drift_factor = np.where(
            elapsed < drift_start_seconds,
            1.0,
            1.0
            + drift_magnitude
            * (elapsed - drift_start_seconds)
            / (total_seconds - drift_start_seconds + 1e-9),
        )

        df["energy_demand_kwh"] = df["energy_demand_kwh"].to_numpy() * drift_factor
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _daily_seasonality(self, hours: np.ndarray) -> np.ndarray:
        """Model bimodal daily pattern: peaks at ~9 AM and ~6 PM, trough at ~3 AM.

        Uses a sum of two Gaussians centred on 9 and 18 with a baseline dip at
        3 AM.
        """
        peak_morning = np.exp(-0.5 * ((hours - 9.0) / 3.0) ** 2)
        peak_evening = np.exp(-0.5 * ((hours - 18.0) / 2.5) ** 2)
        trough = -0.3 * np.exp(-0.5 * ((hours - 3.0) / 2.0) ** 2)
        return 0.4 * (peak_morning + peak_evening) + trough

    def _weekly_seasonality(self, dow: np.ndarray) -> np.ndarray:
        """Weekend demand is ~25 % lower than weekday demand."""
        return np.where(dow >= 5, -0.25, 0.05)

    def _yearly_seasonality(self, doy: np.ndarray) -> np.ndarray:
        """Higher demand in winter (heating) and summer (cooling).

        Modelled as a cosine with period 182.5 days (peaks near Jan and Jul).
        """
        return 0.3 * np.cos(2 * np.pi * doy / 182.5)

    def _generate_weather(
        self,
        n: int,
        day_of_year: np.ndarray,
        hours: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce correlated temperature and humidity arrays.

        Temperature follows a yearly sinusoid (warm summer, cold winter) with
        a diurnal cycle and noise.  Humidity is negatively correlated with
        temperature.
        """
        wp = self.weather

        # Yearly cycle: coldest ~Jan 15 (doy≈15), warmest ~Jul 15 (doy≈196)
        yearly_frac = (day_of_year - 15.0) / 365.0
        yearly_temp = (
            (wp.temp_mean_winter + wp.temp_mean_summer) / 2.0
            + (wp.temp_mean_summer - wp.temp_mean_winter)
            / 2.0
            * np.cos(2 * np.pi * (yearly_frac - 0.5))
        )

        # Diurnal cycle: warmest at ~14:00
        diurnal = (wp.temp_daily_range / 2.0) * np.sin(
            2 * np.pi * (hours - 8.0) / 24.0
        )

        temp_noise = self._rng.normal(0, wp.temp_noise_std, size=n)
        temperature = np.clip(yearly_temp + diurnal + temp_noise, -40.0, 50.0)

        # Humidity with negative temperature correlation
        hum_base = self._rng.normal(wp.humidity_mean, wp.humidity_std, size=n)
        humidity = np.clip(
            hum_base + wp.humidity_temp_corr * (temperature - yearly_temp), 0.0, 100.0
        )

        return temperature, humidity

    def _weather_demand_effect(
        self,
        temperature: np.ndarray,
        humidity: np.ndarray,
    ) -> np.ndarray:
        """Model how weather drives energy demand.

        Very cold or very hot temperatures increase demand (heating / cooling).
        High humidity in summer further increases cooling load.
        """
        comfort_temp = 20.0  # degC

        # Heating effect: increases below comfort
        heating = np.maximum(comfort_temp - temperature, 0.0) * 0.015

        # Cooling effect: increases above comfort, amplified by humidity
        humidity_factor = 1.0 + 0.005 * np.maximum(humidity - 50.0, 0.0)
        cooling = np.maximum(temperature - comfort_temp, 0.0) * 0.012 * humidity_factor

        return 1.0 + heating + cooling
