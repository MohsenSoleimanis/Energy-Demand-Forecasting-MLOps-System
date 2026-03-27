{% macro timezone_convert(timestamp_column, from_tz='UTC', to_tz='Europe/Brussels') %}
{#
    Convert a timestamp from one timezone to another.

    Args:
        timestamp_column: The column or expression containing the UTC timestamp.
        from_tz: Source timezone (default: 'UTC').
        to_tz: Target timezone (default: 'Europe/Brussels').

    Returns:
        A SQL expression that converts the timestamp to the target timezone.
#}
timezone('{{ to_tz }}', timezone('{{ from_tz }}', {{ timestamp_column }}))
{% endmacro %}
