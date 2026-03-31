{{
    config(
        materialized='table'
    )
}}

with load_data as (
    select
        timestamp_brussels::timestamp as timestamp_brussels,
        load_mw,
        is_valid,
        is_interpolated,
        row_number() over (
            partition by timestamp_brussels::timestamp
            order by timestamp_brussels
        ) as _rn
    from {{ ref('stg_load') }}
    where is_valid = true
),

load_deduped as (
    select
        timestamp_brussels,
        load_mw,
        is_valid,
        is_interpolated
    from load_data
    where _rn = 1
),

price_data as (
    select
        timestamp_brussels::timestamp as timestamp_brussels,
        price_eur_mwh,
        row_number() over (
            partition by timestamp_brussels::timestamp
            order by timestamp_brussels
        ) as _rn
    from {{ ref('stg_price') }}
),

price_deduped as (
    select timestamp_brussels, price_eur_mwh
    from price_data
    where _rn = 1
),

weather_raw as (
    select
        timestamp_brussels::timestamp as timestamp_brussels,
        temperature_2m,
        feels_like_temp,
        wind_speed_10m,
        wind_direction_10m,
        shortwave_radiation,
        precipitation,
        cloud_cover,
        pressure_msl,
        is_anomalous,
        row_number() over (
            partition by timestamp_brussels::timestamp
            order by timestamp_brussels
        ) as _rn
    from {{ ref('stg_weather') }}
),

weather_deduped as (
    select
        timestamp_brussels,
        temperature_2m,
        feels_like_temp,
        wind_speed_10m,
        wind_direction_10m,
        shortwave_radiation,
        precipitation,
        cloud_cover,
        pressure_msl,
        is_anomalous
    from weather_raw
    where _rn = 1
),

-- Interpolate null weather values using surrounding hours
weather_data as (
    select
        timestamp_brussels,
        coalesce(
            temperature_2m,
            (lag(temperature_2m) over (order by timestamp_brussels)
             + lead(temperature_2m) over (order by timestamp_brussels)) / 2.0
        ) as temperature_2m,
        coalesce(
            feels_like_temp,
            (lag(feels_like_temp) over (order by timestamp_brussels)
             + lead(feels_like_temp) over (order by timestamp_brussels)) / 2.0
        ) as feels_like_temp,
        coalesce(
            wind_speed_10m,
            (lag(wind_speed_10m) over (order by timestamp_brussels)
             + lead(wind_speed_10m) over (order by timestamp_brussels)) / 2.0
        ) as wind_speed_10m,
        coalesce(
            wind_direction_10m,
            (lag(wind_direction_10m) over (order by timestamp_brussels)
             + lead(wind_direction_10m) over (order by timestamp_brussels)) / 2.0
        ) as wind_direction_10m,
        coalesce(
            shortwave_radiation,
            (lag(shortwave_radiation) over (order by timestamp_brussels)
             + lead(shortwave_radiation) over (order by timestamp_brussels)) / 2.0
        ) as shortwave_radiation,
        coalesce(
            precipitation,
            (lag(precipitation) over (order by timestamp_brussels)
             + lead(precipitation) over (order by timestamp_brussels)) / 2.0
        ) as precipitation,
        coalesce(
            cloud_cover,
            (lag(cloud_cover) over (order by timestamp_brussels)
             + lead(cloud_cover) over (order by timestamp_brussels)) / 2.0
        ) as cloud_cover,
        coalesce(
            pressure_msl,
            (lag(pressure_msl) over (order by timestamp_brussels)
             + lead(pressure_msl) over (order by timestamp_brussels)) / 2.0
        ) as pressure_msl,
        is_anomalous
    from weather_deduped
),

generation_data as (
    select
        timestamp_brussels::timestamp as timestamp_brussels,
        renewable_share_pct,
        nuclear_mw,
        gas_mw,
        row_number() over (
            partition by timestamp_brussels::timestamp
            order by timestamp_brussels
        ) as _rn
    from {{ ref('stg_generation') }}
),

generation_deduped as (
    select timestamp_brussels, renewable_share_pct, nuclear_mw, gas_mw
    from generation_data
    where _rn = 1
),

calendar_data as (
    select * from {{ ref('stg_calendar') }}
),

joined as (
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
        w.is_anomalous as weather_is_anomalous,
        row_number() over (
            partition by l.timestamp_brussels
            order by l.timestamp_brussels
        ) as _final_rn

    from load_deduped l
    left join price_deduped p
        on l.timestamp_brussels = p.timestamp_brussels
    left join weather_data w
        on l.timestamp_brussels = w.timestamp_brussels
    left join generation_deduped g
        on l.timestamp_brussels = g.timestamp_brussels
    left join calendar_data c
        on l.timestamp_brussels::date = c.date
),

final as (
    select
        timestamp_brussels,
        load_mw,
        price_eur_mwh,
        temperature_2m,
        feels_like_temp,
        wind_speed_10m,
        wind_direction_10m,
        shortwave_radiation,
        precipitation,
        cloud_cover,
        pressure_msl,
        renewable_share_pct,
        nuclear_mw,
        gas_mw,
        is_belgian_holiday,
        is_weekend,
        is_school_vacation,
        day_of_week,
        month,
        hour_of_day,
        load_is_valid,
        weather_is_anomalous
    from joined
    where _final_rn = 1
)

select * from final
