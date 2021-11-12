from prunetest import parse


def test_is_blank_line():
    assert parse.is_ws("")
    assert parse.is_ws("  \n")
    assert parse.is_ws("  \n\r")
    assert not parse.is_ws("  abc\n\r")
    assert not parse.is_ws("abc\n\r")


def test_match_assignment():
    # Leading whitespace should not be matched
    assert parse.Assignment.match(" r = 1 mm") is None
    for s in ("r = 1 mm", "r=1 mm", "r= 1 mm", "r =1 mm", "r = 1 mm ", "r = 1 mm"):
        m = parse.Assignment.match(s)
        assert m is not None
        assert m.parameter == "r"
        assert len(m.expression.tokens) == 1
        assert isinstance(m.expression.tokens[0], parse.NumericValue)
        assert m.expression.tokens[0].num == "1"
        assert m.expression.tokens[0].unit == "mm"


def test_match_definition():
    # Example 1
    s = r"f_swell [1] := Free swelling stretch ratio"
    m = parse.Definition.match(s)
    assert m is not None
    assert m.parameter == "f_swell"
    assert m.unit == "1"
    assert m.description == "Free swelling stretch ratio"
    # Example 2
    s = r"r [mm] := Radius"
    m = parse.Definition.match(s)
    assert m is not None
    assert m.parameter == "r"
    assert m.unit == "mm"
    assert m.description == "Radius"


def test_match_unit_blank():
    """Unit should not match whitespace"""
    assert parse.Unit.match("") is None
    assert parse.Unit.match(" ") is None
    assert parse.Unit.match("   ") is None


def test_match_unit_bare_op():
    """Unit should not match a bare operator"""
    for op in parse.operators:
        assert parse.Unit.match(op) is None


def test_match_unit_leading_op():
    for op in parse.operators:
        assert parse.Unit.match(f"{op}x") is None
        assert parse.Unit.match(f"{op} x") is None


def test_match_unit_trailing_space():
    """Unit should ignore trailing spaces"""
    assert (m := parse.Unit.match("x ")) is not None
    assert m == "x"
    assert (m := parse.Unit.match("x\n")) is not None
    assert m == "x"


def test_match_unit_trailing_characters():
    """Unit should ignore trailing characters"""
    assert (m := parse.Unit.match("x foo")) is not None
    assert m == "x"

def test_match_unit_internal_op():
    assert (m := parse.Unit.match("mm/s")) is not None
    assert m == "mm/s"
    assert (m := parse.Unit.match("mm / s")) is not None
    assert m == "mm / s"
    assert (m := parse.Unit.match("mm/ s")) is not None
    assert m == "mm/ s"
    assert (m := parse.Unit.match("mm /s")) is not None
    assert m == "mm /s"


def test_match_unit_unity():
    """Unit should match 1 as 'unitless' indicator"""
    m = parse.Unit.match("1")
    assert m is not None
    assert m == "1"


def test_match_unit_degrees():
    assert parse.Unit.match("°") is not None


def test_match_unit_distance():
    assert (m := parse.Unit.match("μm")) is not None
    assert m == "μm"
    assert (m := parse.Unit.match("mm")) is not None
    assert m == "mm"
    assert (m := parse.Unit.match("m")) is not None
    assert m == "m"
    assert (m := parse.Unit.match("km")) is not None
    assert m == "km"


def test_match_unit_frequency():
    assert (m := parse.Unit.match("Hz")) is not None
    assert m == "Hz"
    assert (m := parse.Unit.match("1/s")) is not None
    assert m == "1/s"


def test_match_unit_permeability():
    assert (m := parse.Unit.match("N·s/mm^4")) is not None
    assert m == "N·s/mm^4"
