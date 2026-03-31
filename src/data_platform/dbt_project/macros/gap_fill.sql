{% macro gap_fill(timestamp_column, expected_interval='1 hour', partition_column=none) %}
{#
    Detect gaps in a time series by comparing each row's timestamp to
    the previous row's timestamp. Returns TRUE when a gap is detected
    (i.e., the difference between consecutive timestamps exceeds the
    expected interval).

    Args:
        timestamp_column: The column containing timestamps.
        expected_interval: The expected interval between rows (default: '1 hour').
        partition_column: Optional column to partition by before detecting gaps.
#}
case
    when lag({{ timestamp_column }}) over (
        {% if partition_column %}
        partition by {{ partition_column }}
        {% endif %}
        order by {{ timestamp_column }}
    ) is null then false
    when {{ timestamp_column }} - lag({{ timestamp_column }}) over (
        {% if partition_column %}
        partition by {{ partition_column }}
        {% endif %}
        order by {{ timestamp_column }}
    ) > interval '{{ expected_interval }}' then true
    when lead({{ timestamp_column }}) over (
        {% if partition_column %}
        partition by {{ partition_column }}
        {% endif %}
        order by {{ timestamp_column }}
    ) is null then false
    when lead({{ timestamp_column }}) over (
        {% if partition_column %}
        partition by {{ partition_column }}
        {% endif %}
        order by {{ timestamp_column }}
    ) - {{ timestamp_column }} > interval '{{ expected_interval }}' then true
    else false
end
{% endmacro %}
