"""Simple feature store backed by the DuckDB gold layer.

Provides historical load and price values for computing lag and
rolling features at serving time, eliminating train-serve skew.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default DuckDB path — same one dbt creates
_DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "data_platform"
    / "dbt_project"
    / "energy_demand.duckdb"
)


class FeatureStore:
    """Read-only feature store backed by DuckDB feature_base table.

    Provides historical values for lag and rolling feature computation
    at serving time.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        import duckdb

        self._db_path = str(
            db_path or os.environ.get("DUCKDB_PATH", _DEFAULT_DB_PATH)
        )
        self._con = duckdb.connect(self._db_path, read_only=True)
        # Load S3 extension for httpfs
        try:
            self._con.execute("INSTALL httpfs; LOAD httpfs;")
            endpoint = (
                os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
                .replace("http://", "")
                .replace("https://", "")
            )
            self._con.execute(f"""
                SET s3_region='us-east-1';
                SET s3_endpoint='{endpoint}';
                SET s3_access_key_id='{os.environ.get("AWS_ACCESS_KEY_ID", "")}';
                SET s3_secret_access_key='{os.environ.get("AWS_SECRET_ACCESS_KEY", "")}';
                SET s3_use_ssl=false; SET s3_url_style='path';
            """)
        except Exception:
            pass
        logger.info("Feature store connected to %s", self._db_path)

    def get_recent_load(
        self, before: datetime, hours: int = 168
    ) -> pd.DataFrame:
        """Get recent hourly load data for computing lags and rolling stats.

        Args:
            before: Get data before this timestamp.
            hours: How many hours of history to fetch (default 168 = 7 days).

        Returns:
            DataFrame with timestamp_brussels, load_mw, price_eur_mwh,
            and temperature_2m columns, sorted by time.
        """
        query = """
            SELECT timestamp_brussels, load_mw, price_eur_mwh, temperature_2m
            FROM main_gold.feature_base
            WHERE timestamp_brussels < ?
            ORDER BY timestamp_brussels DESC
            LIMIT ?
        """
        df = self._con.execute(query, [before, hours]).fetchdf()
        return df.sort_values("timestamp_brussels").reset_index(drop=True)

    def get_lag_features(
        self, target_timestamp: datetime
    ) -> dict[str, float | None]:
        """Compute lag features for a specific prediction timestamp.

        Returns dict with load_lag_24h, load_lag_168h, price_lag_24h,
        temp_lag_24h.  Values are ``None`` if historical data is not
        available.
        """
        features: dict[str, float | None] = {}

        for col, lag_hours, key in [
            ("load_mw", 24, "load_lag_24h"),
            ("load_mw", 168, "load_lag_168h"),
            ("price_eur_mwh", 24, "price_lag_24h"),
            ("temperature_2m", 24, "temp_lag_24h"),
        ]:
            query = f"""
                SELECT {col} FROM main_gold.feature_base
                WHERE timestamp_brussels = ? - INTERVAL '{lag_hours} hours'
                LIMIT 1
            """
            result = self._con.execute(query, [target_timestamp]).fetchone()
            features[key] = float(result[0]) if result else None

        return features

    def get_rolling_features(
        self, target_timestamp: datetime
    ) -> dict[str, float | None]:
        """Compute rolling statistics for a prediction timestamp."""
        features: dict[str, float | None] = {}

        # 24h rolling mean and std for load
        query = """
            SELECT AVG(load_mw) as mean_24, STDDEV(load_mw) as std_24
            FROM main_gold.feature_base
            WHERE timestamp_brussels BETWEEN ? - INTERVAL '24 hours'
                                          AND ? - INTERVAL '1 hour'
        """
        result = self._con.execute(
            query, [target_timestamp, target_timestamp]
        ).fetchone()
        features["load_rolling_mean_24h"] = (
            float(result[0]) if result and result[0] else None
        )
        features["load_rolling_std_24h"] = (
            float(result[1]) if result and result[1] else None
        )

        # 168h rolling mean for load
        query = """
            SELECT AVG(load_mw) as mean_168
            FROM main_gold.feature_base
            WHERE timestamp_brussels BETWEEN ? - INTERVAL '168 hours'
                                          AND ? - INTERVAL '1 hour'
        """
        result = self._con.execute(
            query, [target_timestamp, target_timestamp]
        ).fetchone()
        features["load_rolling_mean_168h"] = (
            float(result[0]) if result and result[0] else None
        )

        # 24h rolling mean for temperature
        query = """
            SELECT AVG(temperature_2m) as temp_mean_24
            FROM main_gold.feature_base
            WHERE timestamp_brussels BETWEEN ? - INTERVAL '24 hours'
                                          AND ? - INTERVAL '1 hour'
        """
        result = self._con.execute(
            query, [target_timestamp, target_timestamp]
        ).fetchone()
        features["temp_rolling_mean_24h"] = (
            float(result[0]) if result and result[0] else None
        )

        return features

    def enrich_request(self, request_data: dict) -> dict:
        """Enrich a prediction request with historical features.

        Fills in lag and rolling features from the feature store,
        replacing the zero-fill that previously caused train-serve skew.

        Args:
            request_data: Raw prediction request as a dict.

        Returns:
            Enriched copy of the request dict with lag and rolling
            features populated from historical data.
        """
        ts = request_data.get("timestamp_brussels")
        if ts is None:
            return request_data

        if isinstance(ts, str):
            ts = pd.Timestamp(ts)

        enriched = dict(request_data)

        # Get lag features
        lags = self.get_lag_features(ts)
        for key, value in lags.items():
            if enriched.get(key) is None or enriched.get(key) == 0.0:
                enriched[key] = value if value is not None else 0.0

        # Get rolling features
        rolling = self.get_rolling_features(ts)
        for key, value in rolling.items():
            enriched[key] = value if value is not None else 0.0

        return enriched

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._con.close()
