{{
    config(
        materialized='view'
    )
}}

with raw_weather as (
    select
        cast(timestamp_utc as timestamp) as timestamp_utc,
        cast(temperature_2m as double) as temperature_2m,
        cast(relative_humidity_2m as double) as relative_humidity_2m,
        cast(wind_speed_10m as double) as wind_speed_10m,
        cast(wind_direction_10m as double) as wind_direction_10m,
        cast(shortwave_radiation as double) as shortwave_radiation,
        cast(precipitation as double) as precipitation,
        cast(cloud_cover as double) as cloud_cover,
        cast(pressure_msl as double) as pressure_msl,
        cast(ingestion_ts as timestamp) as ingestion_ts
    from {{ source('bronze', 'weather_actual') }}
),

with_timezone as (
    select
        {{ timezone_convert('timestamp_utc') }} as timestamp_brussels,
        temperature_2m,
        relative_humidity_2m,
        wind_speed_10m,
        wind_direction_10m,
        shortwave_radiation,
        precipitation,
        cloud_cover,
        pressure_msl,
        ingestion_ts
    from raw_weather
),

deduplicated as (
    select
        *,
        row_number() over (
            partition by timestamp_brussels
            order by ingestion_ts desc
        ) as rn
    from with_timezone
),

final as (
    select
        timestamp_brussels,
        temperature_2m,
        relative_humidity_2m,
        wind_speed_10m,
        wind_direction_10m,
        shortwave_radiation,
        precipitation,
        cloud_cover,
        pressure_msl,
        -- Compute feels-like temperature
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
        -- Flag anomalous weather readings
        case
            when temperature_2m < -30 or temperature_2m > 50 then true
            when relative_humidity_2m < 0 or relative_humidity_2m > 100 then true
            when wind_speed_10m < 0 then true
            when shortwave_radiation < 0 then true
            else false
        end as is_anomalous
    from deduplicated
    where rn = 1
)

select * from final
