"""Test interpretation of units

For tests of parsing units (i.e., identifying which text content in a prunetest file
indicates units), see test_parse.py.

"""
from prunetest.units import Quantity


def test_match_unit_distance():
    # µ (micro sign)
    assert (m := Quantity("μm")) is not None
    assert str(m.units) == "micrometer"
    # μ (greek small letter mu)
    assert (m := Quantity("μm")) is not None
    assert str(m.units) == "micrometer"
    assert (m := Quantity("mm")) is not None
    assert str(m.units) == "millimeter"
    assert (m := Quantity("m")) is not None
    assert str(m.units) == "meter"
    assert (m := Quantity("km")) is not None
    assert str(m.units) == "kilometer"


def test_match_unit_time():
    assert (m := Quantity("h")) is not None
    assert str(m.units) == "hour"
    assert Quantity(1, "h") / Quantity(3600, "s") == 1
