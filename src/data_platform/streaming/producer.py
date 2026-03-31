"""Kafka producer for ENTSO-E data streaming.

Instead of batch-pulling all historical data, this producer
polls the ENTSO-E API for the latest data and publishes to
Kafka topics for real-time processing.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from confluent_kafka import Producer

from src.shared.config import load_env_file, require_env

logger = logging.getLogger(__name__)

TOPICS = {
    "load": "entsoe.load",
    "price": "entsoe.price",
    "generation": "entsoe.generation",
}


def get_producer() -> Producer:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return Producer({"bootstrap.servers": bootstrap})


def publish_load_data(producer: Producer, df: pd.DataFrame) -> int:
    """Publish load data records to Kafka topic."""
    count = 0
    for _, row in df.iterrows():
        msg = {
            "timestamp_utc": row["timestamp_utc"].isoformat(),
            "load_mw": float(row["load_mw"]),
            "area_code": row.get("area_code", "BE"),
            "published_at": datetime.utcnow().isoformat(),
        }
        producer.produce(
            TOPICS["load"],
            key=msg["timestamp_utc"],
            value=json.dumps(msg).encode("utf-8"),
        )
        count += 1
    producer.flush()
    logger.info("Published %d load records to %s", count, TOPICS["load"])
    return count


def run_streaming_producer(poll_interval_s: int = 300):
    """Continuously poll ENTSO-E and publish new data to Kafka.

    Args:
        poll_interval_s: How often to poll (seconds). Default 5 minutes.
    """
    from src.data_platform.ingestion.entsoe import _get_client, fetch_load

    load_env_file()
    producer = get_producer()
    client = _get_client()

    logger.info("Starting streaming producer (poll every %ds)", poll_interval_s)

    while True:
        try:
            end = pd.Timestamp.now(tz="Europe/Brussels")
            start = end - timedelta(hours=2)  # Get last 2 hours

            df = fetch_load(client, "BE", start, end)
            if len(df) > 0:
                publish_load_data(producer, df)

        except Exception as e:
            logger.error("Streaming poll failed: %s", e)

        time.sleep(poll_interval_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_streaming_producer()
