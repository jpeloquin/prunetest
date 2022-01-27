import pint


def fix_pint_registry(ureg: pint.registry.ApplicationRegistry):
    """Update pint ApplicationRegistry with some fixes"""
    # Use "h" for hours, not for the Planck constant.  There's an open debate on this
    # upstream: https://github.com/hgrecco/pint/issues/719
    ureg.define("@alias hour = h")
    # Add greek small letter mu as valid micro prefix.  Eventually this should get
    # fixed upstream.  https://github.com/hgrecco/pint/pull/1347.  We have to use the
    # _registry parameter due to https://github.com/hgrecco/pint/pull/1403.
    on_redefinition = ureg._registry._on_redefinition
    ureg._registry._on_redefinition = "ignore"
    ureg.define("micro- = 1e-6  = µ- = μ- = u-")
    ureg._registry._on_redefinition = on_redefinition
    return ureg


def format_unit(unit):
    long = f"{unit}"
    short = f"{unit:~}"
    if long == "dimensionless":
        return "1"
    return short


# noinspection PyTypeChecker
ureg = fix_pint_registry(pint.get_application_registry())
Quantity = ureg.Quantity
Unit = ureg.Unit
