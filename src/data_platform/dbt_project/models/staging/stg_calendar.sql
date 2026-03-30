{{
    config(
        materialized='table'
    )
}}

with raw_calendar as (
    select
        cast("date" as date) as date,
        cast(is_belgian_holiday as boolean) as is_belgian_holiday,
        cast(holiday_name as varchar) as holiday_name,
        cast(day_of_week as integer) as day_of_week,
        cast(is_weekend as boolean) as is_weekend,
        cast(is_school_vacation as boolean) as is_school_vacation,
        cast(month as integer) as month,
        cast(week_of_year as integer) as week_of_year
    from read_parquet('s3://lakehouse/bronze/calendar/*.parquet')
)

select * from raw_calendar
