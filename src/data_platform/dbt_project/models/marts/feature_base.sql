{{
    config(
        materialized='table'
    )
}}

with load_data as (
    select * from {{ ref('stg_load') }}
    where is_valid = true
),

price_data as (
    select * from {{ ref('stg_price') }}
),

weather_data as (
    select * from {{ ref('stg_weather') }}
),

generation_data as (
    select * from {{ ref('stg_generation') }}
),

calendar_data as (
    select * from {{ ref('stg_calendar') }}
),

final as (
    select
        l.timestamp_brussels,
        l.load_mw,
        p.price_eur_mwh,
        w.temperature_2m,
        w.feels_like_temp,
        w.wind_speed_10m,
        w.wind_direction_10m,
        w.shortwave_radiation,
        w.precipitation,
        w.cloud_cover,
        w.pressure_msl,
        g.renewable_share_pct,
        g.nuclear_mw,
        g.gas_mw,
        c.is_belgian_holiday,
        c.is_weekend,
        c.is_school_vacation,
        c.day_of_week,
        c.month,
        extract(hour from l.timestamp_brussels)::int as hour_of_day,
        l.is_valid as load_is_valid,
        w.is_anomalous as weather_is_anomalous

    from load_data l
    left join price_data p
        on l.timestamp_brussels = p.timestamp_brussels
    left join weather_data w
        on l.timestamp_brussels = w.timestamp_brussels
    left join generation_data g
        on l.timestamp_brussels = g.timestamp_brussels
    left join calendar_data c
        on cast(l.timestamp_brussels as date) = c.date
)

select * from final
