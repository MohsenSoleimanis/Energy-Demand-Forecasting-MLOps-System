{% macro cyclical_encode(column, period) %}
{#
    Generate sin/cos cyclical encodings for a numeric column.

    This is useful for encoding periodic features like hour_of_day (period=24)
    or month (period=12) so that the model understands the cyclical nature
    (e.g., hour 23 is close to hour 0).

    Args:
        column: The column or expression to encode.
        period: The period of the cycle (e.g., 24 for hours, 12 for months).

    Returns:
        Two SQL expressions: one for sin and one for cos encoding.
#}
sin(2 * pi() * ({{ column }}) / {{ period }}) as {{ column | replace('(', '') | replace(')', '') | replace(' ', '_') }}_sin,
cos(2 * pi() * ({{ column }}) / {{ period }}) as {{ column | replace('(', '') | replace(')', '') | replace(' ', '_') }}_cos
{% endmacro %}
