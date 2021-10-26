from prunetest import parse


def test_match_unit_blank():
    assert parse.re_unit.match("") is None
    assert parse.re_unit.match(" ") is None
    assert parse.re_unit.match("   ") is None


def test_match_unit_bare_op():
    for op in parse.operators:
        assert parse.re_unit.match(op) is None


def test_match_unit_leading_op():
    for op in parse.operators:
        assert parse.re_unit.match(f"{op}x") is None
        assert parse.re_unit.match(f"{op} x") is None


def test_match_unit_trailing_space():
    assert (m := parse.re_unit.match("x ")) is not None
    assert m.group() == "x"
    assert (m := parse.re_unit.match("x\n")) is not None
    assert m.group() == "x"


def test_match_unit_bare_op():
    assert parse.re_unit.match("/") is None


def test_match_unit_internal_op():
    assert (m := parse.re_unit.match("mm/s")) is not None
    assert m.group() == "mm/s"
    assert (m := parse.re_unit.match("mm / s")) is not None
    assert m.group() == "mm / s"
    assert (m := parse.re_unit.match("mm/ s")) is not None
    assert m.group() == "mm/ s"
    assert (m := parse.re_unit.match("mm /s")) is not None
    assert m.group() == "mm /s"


def test_match_unit_degrees():
    assert parse.re_unit.match("°") is not None


def test_match_unit_distance():
    assert (m := parse.re_unit.match("μm")) is not None
    assert m.group() == "μm"
    assert (m := parse.re_unit.match("mm")) is not None
    assert m.group() == "mm"
    assert (m := parse.re_unit.match("m")) is not None
    assert m.group() == "m"
    assert (m := parse.re_unit.match("km")) is not None
    assert m.group() == "km"


def test_match_unit_frequency():
    assert (m := parse.re_unit.match("Hz")) is not None
    assert m.group() == "Hz"
    assert (m := parse.re_unit.match("1/s")) is not None
    assert m.group() == "1/s"


def test_match_unit_permeability():
    assert (m := parse.re_unit.match("N·s/mm^4")) is not None
    assert m.group() == "N·s/mm^4"
