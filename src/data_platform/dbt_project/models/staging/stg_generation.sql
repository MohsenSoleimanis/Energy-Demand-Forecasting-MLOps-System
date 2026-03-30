{{
    config(
        materialized='table'
    )
}}

with raw_generation as (
    select
        cast(timestamp_utc as timestamp) as timestamp_utc,
        cast(fuel_type as varchar) as fuel_type,
        cast(generation_mw as double) as generation_mw,
        cast(ingestion_ts as timestamp) as ingestion_ts
    from read_parquet('s3://lakehouse/bronze/entsoe_generation/**/*.parquet', hive_partitioning=true)
),

with_timezone as (
    select
        {{ timezone_convert('timestamp_utc') }} as timestamp_brussels,
        fuel_type,
        generation_mw,
        ingestion_ts
    from raw_generation
),

deduplicated as (
    select
        *,
        row_number() over (
            partition by timestamp_brussels, fuel_type
            order by ingestion_ts desc
        ) as rn
    from with_timezone
),

clean as (
    select
        timestamp_brussels,
        fuel_type,
        generation_mw
    from deduplicated
    where rn = 1
),

aggregated as (
    select
        timestamp_brussels,
        sum(generation_mw) as total_generation_mw,
        sum(case
            when fuel_type in ('Solar', 'Wind Onshore', 'Wind Offshore')
            then generation_mw else 0
        end) as renewable_mw,
        sum(case
            when fuel_type = 'Nuclear'
            then generation_mw else 0
        end) as nuclear_mw,
        sum(case
            when fuel_type in ('Gas', 'Fossil Gas')
            then generation_mw else 0
        end) as gas_mw
    from clean
    group by timestamp_brussels
),

final as (
    select
        timestamp_brussels,
        total_generation_mw,
        renewable_mw,
        case
            when total_generation_mw > 0
            then round(renewable_mw / total_generation_mw * 100, 2)
            else 0
        end as renewable_share_pct,
        nuclear_mw,
        gas_mw
    from aggregated
)

select * from final
