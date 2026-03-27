{{
    config(
        materialized='view'
    )
}}

with raw_load as (
    select
        cast(timestamp_utc as timestamp) as timestamp_utc,
        cast(load_mw as double) as load_mw,
        cast(ingestion_ts as timestamp) as ingestion_ts
    from {{ source('bronze', 'entsoe_load') }}
),

-- Resample 15-minute data to hourly by averaging
hourly_load as (
    select
        date_trunc('hour', timestamp_utc) as timestamp_utc,
        avg(load_mw) as load_mw,
        max(ingestion_ts) as ingestion_ts
    from raw_load
    group by date_trunc('hour', timestamp_utc)
),

-- Convert UTC to Europe/Brussels
with_timezone as (
    select
        {{ timezone_convert('timestamp_utc') }} as timestamp_brussels,
        load_mw,
        ingestion_ts
    from hourly_load
),

-- Deduplicate by keeping the latest ingestion per timestamp
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
        -- Valid Belgian load is roughly between 4000 and 16000 MW
        case
            when load_mw between 4000 and 16000 then true
            else false
        end as is_valid,
        -- Detect gaps: if the previous or next hour is missing, flag as interpolated
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
