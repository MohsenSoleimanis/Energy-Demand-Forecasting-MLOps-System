import duckdb

c = duckdb.connect()
c.execute("INSTALL httpfs; LOAD httpfs")
c.execute("""
    SET s3_region='us-east-1';
    SET s3_endpoint='localhost:9000';
    SET s3_access_key_id='minioadmin';
    SET s3_secret_access_key='minioadmin';
    SET s3_use_ssl=false;
    SET s3_url_style='path';
""")

print("=== Weather parquet column types ===")
for r in c.execute("DESCRIBE SELECT * FROM read_parquet('s3://lakehouse/bronze/weather_actual/year=2024/month=01/*.parquet')").fetchall():
    print(f"  {r[0]:30s} {r[1]}")

print("\n=== Load parquet column types ===")
for r in c.execute("DESCRIBE SELECT * FROM read_parquet('s3://lakehouse/bronze/entsoe_load/date=2024-01-01/*.parquet')").fetchall():
    print(f"  {r[0]:30s} {r[1]}")

print("\n=== Quick test: timezone convert on weather ===")
try:
    r = c.execute("""
        SELECT typeof(timestamp_utc) as orig_type,
               typeof(cast(timezone('Europe/Brussels', timezone('UTC', timestamp_utc)) as timestamp)) as converted_type
        FROM read_parquet('s3://lakehouse/bronze/weather_actual/year=2024/month=01/*.parquet')
        LIMIT 1
    """).fetchone()
    print(f"  Original: {r[0]}, Converted: {r[1]}")
except Exception as e:
    print(f"  Error: {e}")
