{{
    config(
        materialized='table'
    )
}}

with recent_features as (
    select *
    from {{ ref('feature_base') }}
    where timestamp_brussels >= current_timestamp - interval '30 days'
),

final as (
    select
        timestamp_brussels,
        load_mw,
        price_eur_mwh,
        temperature_2m,
        relative_humidity_2m,
        wind_speed_10m,
        wind_direction_10m,
        surface_pressure,
        cloud_cover,
        direct_radiation,
        feels_like_temp,
        total_generation_mw,
        renewable_mw,
        renewable_share_pct,
        nuclear_mw,
        gas_mw,
        day_of_week,
        is_weekend,
        is_holiday,
        holiday_name,
        season,
        hour_of_day,
        is_interpolated,
        is_price_extreme,
        is_weather_anomalous,

        -- Placeholder columns for prediction outputs (to be joined once serving is operational)
        cast(null as double) as predicted_load_mw,
        cast(null as double) as prediction_error,
        cast(null as timestamp) as prediction_timestamp

    from recent_features
)

select * from final
