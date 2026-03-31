{{
    config(
        materialized='table'
    )
}}

with recent_actuals as (
    select
        timestamp_brussels,
        load_mw,
        price_eur_mwh,
        temperature_2m,
        wind_speed_10m,
        hour_of_day,
        day_of_week,
        month,
        is_weekend,
        is_belgian_holiday
    from {{ ref('feature_base') }}
    where timestamp_brussels >= current_timestamp - interval '30 days'
)

select
    *,
    cast(null as double) as predicted_load_mw,
    cast(null as double) as prediction_error,
    cast(null as varchar) as prediction_id,
    cast(null as varchar) as model_version,
    cast(null as timestamp) as prediction_timestamp
from recent_actuals
