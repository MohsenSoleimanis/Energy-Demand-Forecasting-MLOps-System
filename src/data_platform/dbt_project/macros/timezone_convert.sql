{% macro timezone_convert(timestamp_column, from_tz='UTC', to_tz='Europe/Brussels') %}
{#
    Convert a UTC timestamp to Europe/Brussels local time.
    Returns plain TIMESTAMP (not TIMESTAMPTZ) for consistent JOIN types.
#}
cast(timezone('{{ to_tz }}', timezone('{{ from_tz }}', {{ timestamp_column }})) as timestamp)
{% endmacro %}
