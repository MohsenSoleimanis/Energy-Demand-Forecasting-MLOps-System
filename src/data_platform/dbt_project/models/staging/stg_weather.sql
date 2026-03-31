{{
    config(
        materialized='table'
    )
}}

with raw_weather as (
    select
        date_trunc('hour', cast(timestamp_utc as timestamp)) as ts_hour_utc,
        cast(temperature_2m as double) as temperature_2m,
        cast(relative_humidity_2m as double) as relative_humidity_2m,
        cast(wind_speed_10m as double) as wind_speed_10m,
        cast(wind_direction_10m as double) as wind_direction_10m,
        cast(shortwave_radiation as double) as shortwave_radiation,
        cast(precipitation as double) as precipitation,
        cast(cloud_cover as double) as cloud_cover,
        cast(pressure_msl as double) as pressure_msl
    from read_parquet('s3://lakehouse/bronze/weather_actual/**/*.parquet', hive_partitioning=true)
),

-- Deduplicate by taking average per hour (handles overlapping ingestions)
hourly as (
    select
        ts_hour_utc,
        avg(temperature_2m) as temperature_2m,
        avg(relative_humidity_2m) as relative_humidity_2m,
        avg(wind_speed_10m) as wind_speed_10m,
        avg(wind_direction_10m) as wind_direction_10m,
        avg(shortwave_radiation) as shortwave_radiation,
        avg(precipitation) as precipitation,
        avg(cloud_cover) as cloud_cover,
        avg(pressure_msl) as pressure_msl
    from raw_weather
    group by ts_hour_utc
),

-- Convert UTC to Brussels time (same pattern as stg_load which works)
final as (
    select
        cast(timezone('Europe/Brussels', timezone('UTC', ts_hour_utc)) as timestamp) as timestamp_brussels,
        temperature_2m,
        relative_humidity_2m,
        wind_speed_10m,
        wind_direction_10m,
        shortwave_radiation,
        precipitation,
        cloud_cover,
        pressure_msl,
        case
            when temperature_2m < 10 and wind_speed_10m > 5 then
                13.12
                + 0.6215 * temperature_2m
                - 11.37 * power(wind_speed_10m, 0.16)
                + 0.3965 * temperature_2m * power(wind_speed_10m, 0.16)
            when temperature_2m > 27 and relative_humidity_2m > 40 then
                temperature_2m
                + 0.33 * (relative_humidity_2m / 100.0
                    * 6.105 * exp(17.27 * temperature_2m / (237.7 + temperature_2m)))
                - 4.0
            else temperature_2m
        end as feels_like_temp,
        case
            when temperature_2m < -30 or temperature_2m > 50 then true
            when relative_humidity_2m < 0 or relative_humidity_2m > 100 then true
            when wind_speed_10m < 0 then true
            when shortwave_radiation < 0 then true
            else false
        end as is_anomalous
    from hourly
)

select * from final
