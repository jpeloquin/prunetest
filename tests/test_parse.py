from prunetest import parse
from prunetest.parse import BinOp, Expression, Number, NumericValue, Unit, Symbol, UnOp


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
        assert m.expression.tokens[0].units == "mm"


def test_match_definition():
    # Example 1
    s = r"f_swell [1] := Free swelling stretch ratio"
    m = parse.Definition.match(s)
    assert m is not None
    assert m.parameter == "f_swell"
    assert m.units == "1"
    assert m.description == "Free swelling stretch ratio"
    # Example 2
    s = r"r [mm] := Radius"
    m = parse.Definition.match(s)
    assert m is not None
    assert m.parameter == "r"
    assert m.units == "mm"
    assert m.description == "Radius"


# Test parsing of units


def test_match_unit_blank():
    """Unit should not match whitespace"""
    assert Unit.match("") is None
    assert Unit.match(" ") is None
    assert Unit.match("   ") is None


def test_match_unit_bare_op():
    """Unit should not match a bare operator"""
    for op in parse.operators:
        assert Unit.match(op) is None


def test_match_unit_leading_op():
    for op in parse.operators:
        assert Unit.match(f"{op}x") is None
        assert Unit.match(f"{op} x") is None


def test_match_unit_trailing_space():
    """Unit should ignore trailing spaces"""
    assert (m := Unit.match("x ")) is not None
    assert m == "x"
    assert (m := Unit.match("x\n")) is not None
    assert m == "x"


def test_match_unit_trailing_characters():
    """Unit should ignore trailing characters"""
    assert (m := Unit.match("x foo")) is not None
    assert m == "x"


def test_match_unit_internal_op():
    assert (m := Unit.match("mm/s")) is not None
    assert m == "mm/s"
    assert (m := Unit.match("mm / s")) is not None
    assert m == "mm / s"
    assert (m := Unit.match("mm/ s")) is not None
    assert m == "mm/ s"
    assert (m := Unit.match("mm /s")) is not None
    assert m == "mm /s"


def test_match_unit_unity():
    """Unit should match 1 as 'unitless' indicator"""
    m = Unit.match("1")
    assert m is not None
    assert m == "1"


def test_match_unit_degrees():
    assert Unit.match("°") is not None


def test_match_unit_distance():
    assert (m := Unit.match("μm")) is not None
    assert m == "μm"
    assert (m := Unit.match("mm")) is not None
    assert m == "mm"
    assert (m := Unit.match("m")) is not None
    assert m == "m"
    assert (m := Unit.match("km")) is not None
    assert m == "km"


def test_match_unit_frequency():
    assert (m := Unit.match("Hz")) is not None
    assert m == "Hz"
    assert (m := Unit.match("1/s")) is not None
    assert m == "1/s"


def test_match_unit_permeability():
    assert (m := Unit.match("N·s/mm^4")) is not None
    assert m == "N·s/mm^4"


# TODO: Test parsing of expressions


def test_match_neg_exp():
    # Without parens
    match = parse.parse_expression("-λ_freeswell^2")
    noparens = Expression(
        [UnOp("-"), Symbol("λ_freeswell"), BinOp("^"), NumericValue("2", Number(2))]
    )
    assert match == noparens
    # With parens
    match = parse.parse_expression("-(λ_freeswell^2)")
    withparens = Expression(
        [
            UnOp("-"),
            Expression(
                [Symbol("λ_freeswell"), BinOp("^"), NumericValue("2", Number(2))]
            ),
        ]
    )
    assert match == withparens
    assert noparens.read() == withparens.read()


def test_match_nested_groups():
    # 1 + (a + b)
    match = parse.parse_expression("1 + (a + b)")
    expected = Expression(
        [
            NumericValue("1", Number(1)),
            BinOp("+"),
            Expression([Symbol("a"), BinOp("+"), Symbol("b")]),
        ]
    )
    assert match == expected
    # (a + b) - 1
    match = parse.parse_expression("(a + b) + 1")
    expected = Expression(
        [
            Expression([Symbol("a"), BinOp("+"), Symbol("b")]),
            BinOp("+"),
            NumericValue("1", Number(1)),
        ]
    )
    assert match == expected
    # (1 + 1)
    match = parse.parse_expression("(1 + 1)")
    expected = Expression(
        [
            Expression(
                [NumericValue("1", Number(1)), BinOp("+"), NumericValue("1", Number(1))]
            )
        ]
    )
    assert match == expected
    # 1 + (a + (1 - b))
    match = parse.parse_expression("1 + (a + (1 - b))")
    expected = Expression(
        [
            NumericValue("1", Number(1)),
            BinOp("+"),
            Expression(
                [
                    Symbol("a"),
                    BinOp("+"),
                    Expression([NumericValue("1", Number(1)), BinOp("-"), Symbol("b")]),
                ]
            ),
        ]
    )
    assert match == expected
