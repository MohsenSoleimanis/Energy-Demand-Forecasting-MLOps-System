{{
    config(
        materialized='table'
    )
}}

with raw_load as (
    select
        cast(timestamp_utc as timestamp) as timestamp_utc,
        cast(load_mw as double) as load_mw,
        cast(ingestion_ts as timestamp) as ingestion_ts
    from read_parquet('s3://lakehouse/bronze/entsoe_load/**/*.parquet', hive_partitioning=true)
),

hourly_load as (
    select
        date_trunc('hour', timestamp_utc) as timestamp_utc,
        avg(load_mw) as load_mw,
        max(ingestion_ts) as ingestion_ts
    from raw_load
    group by date_trunc('hour', timestamp_utc)
),

with_timezone as (
    select
        {{ timezone_convert('timestamp_utc') }} as timestamp_brussels,
        load_mw,
        ingestion_ts
    from hourly_load
),

deduplicated as (
    select
        timestamp_brussels,
        load_mw,
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
        load_mw,
        case
            when load_mw between 4000 and 16000 then true
            else false
        end as is_valid,
        case
            when lag(timestamp_brussels) over (order by timestamp_brussels)
                 < timestamp_brussels - interval '1 hour'
                 or lead(timestamp_brussels) over (order by timestamp_brussels)
                 > timestamp_brussels + interval '1 hour'
            then true
            else false
        end as is_interpolated
    from deduplicated
    where rn = 1
)

select * from final
