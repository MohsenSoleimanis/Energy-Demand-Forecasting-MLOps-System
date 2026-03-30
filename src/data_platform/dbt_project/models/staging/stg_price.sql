{{
    config(
        materialized='table'
    )
}}

with raw_price as (
    select
        cast(timestamp_utc as timestamp) as timestamp_utc,
        cast(price_eur_mwh as double) as price_eur_mwh,
        cast(ingestion_ts as timestamp) as ingestion_ts
    from read_parquet('s3://lakehouse/bronze/entsoe_price/**/*.parquet', hive_partitioning=true)
),

with_timezone as (
    select
        {{ timezone_convert('timestamp_utc') }} as timestamp_brussels,
        price_eur_mwh,
        ingestion_ts
    from raw_price
),

deduplicated as (
    select
        timestamp_brussels,
        price_eur_mwh,
        ingestion_ts,
        row_number() over (
            partition by timestamp_brussels
            order by ingestion_ts desc
        ) as rn
    from with_timezone
),

final as (
    select
        timestamp_brussels,
        price_eur_mwh,
        case
            when price_eur_mwh > 500 or price_eur_mwh < -100 then true
            else false
        end as is_extreme
    from deduplicated
    where rn = 1
)

select * from final
