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
        *,
        -- Placeholder columns for prediction outputs (joined once serving is operational)
        cast(null as double) as predicted_load_mw,
        cast(null as double) as prediction_error,
        cast(null as timestamp) as prediction_timestamp
    from recent_features
)

select * from final
