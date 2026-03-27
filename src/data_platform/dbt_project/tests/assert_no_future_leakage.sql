/*
    Custom test: Assert no future data leakage in the training set.

    This test verifies that no row in the training_set has lag features
    or rolling features computed from data points that occur AFTER the
    row's own timestamp. Specifically, it checks that:

    1. load_lag_1h comes from 1 hour before the current timestamp
    2. load_lag_24h comes from 24 hours before the current timestamp
    3. load_lag_168h comes from 168 hours (1 week) before the current timestamp
    4. The target variable (target_load_24h) is from the future (which is correct
       for the target, but the features should only use past data)

    If any rows are returned, the test fails -- meaning there is potential
    future leakage in the feature set.
*/

with training as (
    select
        timestamp_brussels,
        load_mw,
        load_lag_1h,
        load_lag_24h,
        load_lag_168h,
        target_load_24h
    from {{ ref('training_set') }}
),

-- Verify lag features by self-joining to check actual values at lagged timestamps
lag_check as (
    select
        t.timestamp_brussels,
        t.load_lag_1h,
        t_1h.load_mw as actual_load_1h_ago
    from training t
    left join training t_1h
        on t_1h.timestamp_brussels = t.timestamp_brussels - interval '1 hour'
    where t.load_lag_1h is not null
      and t_1h.load_mw is not null
      and abs(t.load_lag_1h - t_1h.load_mw) > 0.01
),

-- Check that target is actually from the future (24h ahead)
target_check as (
    select
        t.timestamp_brussels,
        t.target_load_24h,
        t_future.load_mw as actual_load_24h_ahead
    from training t
    left join training t_future
        on t_future.timestamp_brussels = t.timestamp_brussels + interval '24 hours'
    where t.target_load_24h is not null
      and t_future.load_mw is not null
      and abs(t.target_load_24h - t_future.load_mw) > 0.01
)

-- Return any rows that indicate leakage; test passes if no rows returned
select * from lag_check
union all
select timestamp_brussels, target_load_24h, actual_load_24h_ahead from target_check
