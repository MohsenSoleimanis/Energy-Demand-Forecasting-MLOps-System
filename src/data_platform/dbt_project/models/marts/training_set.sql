{{
    config(
        materialized='table'
    )
}}

with base as (
    select * from {{ ref('feature_base') }}
),

with_lags as (
    select
        *,

        -- Load lag features
        lag(load_mw, 1) over (order by timestamp_brussels) as load_lag_1h,
        lag(load_mw, 24) over (order by timestamp_brussels) as load_lag_24h,
        lag(load_mw, 168) over (order by timestamp_brussels) as load_lag_168h,

        -- Price lag features
        lag(price_eur_mwh, 24) over (order by timestamp_brussels) as price_lag_24h,

        -- Temperature lag features
        lag(temperature_2m, 24) over (order by timestamp_brussels) as temp_lag_24h

    from base
),

with_rolling as (
    select
        *,

        -- Rolling load statistics (24h window, backward-looking)
        avg(load_mw) over (
            order by timestamp_brussels
            rows between 24 preceding and 1 preceding
        ) as load_rolling_mean_24h,

        stddev(load_mw) over (
            order by timestamp_brussels
            rows between 24 preceding and 1 preceding
        ) as load_rolling_std_24h,

        -- Rolling load statistics (168h / 1 week window)
        avg(load_mw) over (
            order by timestamp_brussels
            rows between 168 preceding and 1 preceding
        ) as load_rolling_mean_168h,

        -- Rolling temperature statistics (24h window)
        avg(temperature_2m) over (
            order by timestamp_brussels
            rows between 24 preceding and 1 preceding
        ) as temp_rolling_mean_24h

    from with_lags
),

with_features as (
    select
        *,

        -- Cyclical encodings for hour of day
        sin(2 * pi() * hour_of_day / 24.0) as hour_sin,
        cos(2 * pi() * hour_of_day / 24.0) as hour_cos,

        -- Cyclical encodings for month
        sin(2 * pi() * month / 12.0) as month_sin,
        cos(2 * pi() * month / 12.0) as month_cos,

        -- Interaction features
        temperature_2m * hour_of_day as temp_x_hour,
        wind_speed_10m * shortwave_radiation as wind_x_radiation,

        -- Target variable: load 24 hours ahead
        lead(load_mw, 24) over (order by timestamp_brussels) as target_load_24h

    from with_rolling
),

final as (
    select * from with_features
    -- Ensure we have enough history for all lag/rolling features
    where load_lag_168h is not null
      and target_load_24h is not null
)

select * from final
