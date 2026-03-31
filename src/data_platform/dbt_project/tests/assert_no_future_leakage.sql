/*
    Custom test: Assert no future data leakage in the training set.

    Verifies that lag features only reference past data and the target
    references future data, by recomputing LAG/LEAD on the ordered
    training set and comparing.

    Only checks features actually used by the model:
    - load_lag_24h = LAG(load_mw, 24) -- 24 rows earlier
    - load_lag_168h = LAG(load_mw, 168) -- 168 rows earlier
    - target_load_24h = LEAD(load_mw, 24) -- 24 rows later

    Uses a tolerance of 1.0 MW to account for floating point and
    DST-related row reordering edge cases.
*/

with training as (
    select
        timestamp_brussels,
        load_mw,
        load_lag_24h,
        load_lag_168h,
        target_load_24h,
        lag(load_mw, 24) over (order by timestamp_brussels) as expected_lag_24h,
        lag(load_mw, 168) over (order by timestamp_brussels) as expected_lag_168h,
        lead(load_mw, 24) over (order by timestamp_brussels) as expected_target_24h
    from {{ ref('training_set') }}
)

select
    timestamp_brussels,
    'lag_24h' as check_type,
    load_lag_24h as actual_value,
    expected_lag_24h as expected_value
from training
where load_lag_24h is not null
  and expected_lag_24h is not null
  and abs(load_lag_24h - expected_lag_24h) > 1.0

union all

select
    timestamp_brussels,
    'lag_168h' as check_type,
    load_lag_168h as actual_value,
    expected_lag_168h as expected_value
from training
where load_lag_168h is not null
  and expected_lag_168h is not null
  and abs(load_lag_168h - expected_lag_168h) > 1.0

union all

select
    timestamp_brussels,
    'target_24h' as check_type,
    target_load_24h as actual_value,
    expected_target_24h as expected_value
from training
where target_load_24h is not null
  and expected_target_24h is not null
  and abs(target_load_24h - expected_target_24h) > 1.0
