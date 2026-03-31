"""Kafka consumer that writes ENTSO-E data to Delta Lake.

Reads from Kafka topics and writes to Delta tables in MinIO,
providing a real-time data pipeline.
"""
import json
import logging
import os

import pandas as pd
from confluent_kafka import Consumer, KafkaError

from src.data_platform.delta_writer import write_delta_table
from src.shared.config import load_env_file

logger = logging.getLogger(__name__)


def get_consumer(group_id: str = "delta-writer") -> Consumer:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
    })


def consume_and_write(
    topic: str = "entsoe.load",
    delta_uri: str = "s3://lakehouse/delta/entsoe_load",
    batch_size: int = 100,
    timeout_s: float = 5.0,
):
    """Consume messages from Kafka and write to Delta Lake in batches."""
    load_env_file()
    consumer = get_consumer()
    consumer.subscribe([topic])

    buffer = []

    try:
        while True:
            msg = consumer.poll(timeout_s)

            if msg is None:
                if buffer:
                    _flush_buffer(buffer, delta_uri)
                    buffer.clear()
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Consumer error: %s", msg.error())
                continue

            record = json.loads(msg.value().decode("utf-8"))
            buffer.append(record)

            if len(buffer) >= batch_size:
                _flush_buffer(buffer, delta_uri)
                buffer.clear()

    except KeyboardInterrupt:
        if buffer:
            _flush_buffer(buffer, delta_uri)
    finally:
        consumer.close()


def _flush_buffer(buffer: list[dict], delta_uri: str) -> None:
    """Write buffered records to Delta table."""
    df = pd.DataFrame(buffer)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    write_delta_table(df, delta_uri, mode="append")
    logger.info("Flushed %d records to %s", len(buffer), delta_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    consume_and_write()
