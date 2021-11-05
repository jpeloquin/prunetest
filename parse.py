# Base packages
import pathlib
import re

# Local packages
from warnings import warn
from typing import Optional

from . import protocol, ureg


operators = ("+", "-", "−", "/", "*", "×", "^", "·")
re_bin_op = re.compile("^" + "|".join(["\\" + op for op in operators]))
re_un_op = re.compile(r"^-")

trans_op = ("→", "->")
re_trans = re.compile(r"^→|->")

re_unit = re.compile(
    r"(\w|°)+(\s*" f"[{''.join(op for op in operators)}]" r"\s*(\w|°)+)*"
)

re_ws = re.compile(r"^\s+")

re_num = re.compile(r"-?(\d*\.)?\d+")

keywords = ("set-default", "control", "uncontrol")


def to_number(s):
    """Convert numeric string to int or float as appropriate."""
    try:
        return int(s)
    except ValueError:
        return float(s)


class Token:
    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f"{self.__class__.__name__}({self.v!r})"


class Comment(Token):
    regex = re.compile("^#")

    @classmethod
    def match(self, s: str) -> Optional[str]:
        if s.startswith("#"):
            return s[1:].lstrip()
        return None


class Segment:
    """Parse data for a Segment"""

    def __init__(self, transitions):
        self.transitions = transitions

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transitions!r})"

    def read(self):
        """Return protocol.Segment object from parse data"""
        transitions = []
        for t in self.transitions:
            if t.path is not None:
                raise ValueError(
                    "A transition with path constraint {t.path} was "
                    "provided.  Non-default paths are not yet supported. "
                    "The default is a linear transition."
                )
            transitions.append(t.read())
        return protocol.Segment(transitions)


class Instruction:
    """Parse data for a keyword instruction"""

    def __init__(self, keyword, variable, setting):
        self.keyword = keyword
        self.variable = variable
        self.setting = setting

    def read(self):
        return self.keyword, self.variable, self.setting


class Phase:
    """Parse data for Phase"""

    def __init__(self, name, elements):
        self.name = name
        self.elements = elements

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.elements!r})"

    def read(self):
        """Return protocol.Phase instance from parse data"""
        elements = []
        for e in self.elements:
            elements.append(e.read())
        return protocol.Phase(self.name, elements)


# Classes for parsing expressions


class BinOp:
    def __init__(self, op, lval, rval):
        self.op = op
        self.lval = lval
        self.rval = rval


class Number(Token):
    def read(self):
        return to_number(self.v)


class Operator(Token):
    pass


class Unit(Token):
    def read(self):
        return ureg(self.v)


class Name(Token):
    regex = re.compile(r"^\w+")

    @classmethod
    def match(self, s: str) -> Optional[str]:
        """Return matching start of string or None"""
        if m := self.regex.match(s):
            return m.group()


class NumericValue:
    def __init__(self, num, unit=Unit("1")):
        self.num = num
        self.unit = unit

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num}, {self.unit!r})"

    def read(self):
        return self.num.read() * self.unit.read()


class SymbolicValue:
    def __init__(self, symbol, op=None):
        self.symbol = symbol
        self.operator = op

    def __repr__(self):
        return f"{self.__class__.__name__}({self.operator or ''}{self.symbol})"

    def read(self):
        # How do we handle symbolic values?
        return self


# Classes for parsing parameter definitions and assignments


class Assignment:
    """Assignment; e.g., a = 1 mm

    The right hand side can be an expression, not just a value.

    """
    def __init__(self, parameter, expression):
        self.parameter = parameter
        self.expression = expression

    @classmethod
    def match(cls, s):
        name = None
        expr = None
        i = 0
        i += match_ws(s[i:])
        if m := Name.match(s[i:]):
            i += len(m)
            name = m
        else:
            return False
        i += match_ws(s[i:])
        if not s[i] == "=":
            return False
        i += 1
        i += match_ws(s[i:])
        expr = parse_expression(s[i:])
        return cls(name, expr)


# Classes for parsing protocol statements


class Transition:
    """Store parse data for a transition statement, e.g., t → 1 s"""

    def __init__(self, variable, target: "Target", path=None):
        self.variable = variable
        self.target = target
        # The default path at the model layer should be linear,
        # but since this class is just parse data, we store what we
        # found.
        self.path = path

    def __repr__(self):
        return f"{self.__class__.__name__}({self.variable}, {self.target}, {self.path})"

    def read(self):
        path = self.path if self.path is not None else "linear"
        return protocol.Transition(self.variable, self.target.read(), path)


class Target:
    def __init__(self, expr):
        self.value = expr

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class AbsoluteTarget(Target):
    def read(self):
        # For now, require simple expressions as targets
        return protocol.AbsoluteTarget(self.value[0].read())


class RelativeTarget(Target):
    def read(self):
        # For now, require simple expressions as targets
        return protocol.RelativeTarget(self.value[0].read())


def is_ws(s) -> bool:
    return s.strip() == ""


def match_ws(s) -> int:
    """Return length of whitespace at start of text"""
    if m := re_ws.match(s):
        return m.end()
    else:
        return 0


def match_comment(s) -> bool:
    if s.startswith("#"):
        return True
    else:
        return False


# Parsing functions


def expand_block(elements):
    """Expand blocks of cycles to individual segments"""
    expanded = []
    active = []
    for e in reversed(elements):
        if e[0] == "segment":
            active.append(e)
        elif e[0] == "block":
            active = active * e[1]["n"]
            expanded += active
            active = []
            expanded.append(e)
        else:
            expanded += active
            active = []
            expanded.append(e)
    expanded.reverse()
    return expanded


def parse_expression(s):
    """Return syntax tree for expression"""
    i = 0  # character index; if ≥ len(s), must return immediately

    def match_binary_op():
        nonlocal i
        if m := re_bin_op.match(s[i:]):
            i += m.end()
            return Operator(m.group())

    def match_group():
        nonlocal i
        if s[i] == "(":
            # Open group
            i += 1
            j = s[i:].find(")")
            if j == -1:
                raise ValueError(
                    f"Unmatched open parenthesis.  Next 10 characters after unmatch parenthesis were: {s[i:i+10]}"
                )
            group = parse_expression(s[i:j])
            i = i + j + 1
            return group

    def match_number():
        nonlocal i
        if m := re_num.match(s[i:]):
            i += m.end()
            return Number(m.group())

    def match_unary_op():
        nonlocal i
        if m := re_un_op.match(s[i:]):
            i += m.end()
            return m.group()

    def match_unit():
        nonlocal i
        if m := re_unit.match(s[i:]):
            i += m.end()
            return Unit(m.group())

    def match_name():
        nonlocal i
        if m := Name.match(s[i:]):
            i += len(m)
            return Name(m)

    def match_ws():
        nonlocal i
        if m := re_ws.match(s[i:]):
            i += m.end()
            return m.group()

    def match_value():
        nonlocal i
        if num := match_number():
            match_ws()
            if unit := match_unit():
                return NumericValue(num, unit)
            else:
                return NumericValue(num)
        if un := match_unary_op():
            # No whitespace may separate the unary operator and its operand
            if name := match_name():
                return SymbolicValue(name, op=un)
            else:
                raise ValueError(
                    f"Unary operator `{un}` was not followed by a symbolic reference.  The remaining characters in the problem line were: '{s[i:]}'"
                )
        if name := match_name():
            return SymbolicValue(name)

    stream = []
    while i < len(s):
        match_ws()
        if i == len(s):
            # Have to break manually if everything left was whitespace.
            break
        if group := match_group():
            stream.append(tuple(group))
            continue
        if value := match_value():
            stream.append(value)
            continue
        if binary := match_binary_op():
            stream.append(Operator(binary))
            continue
        raise ValueError(f"Failed to parse the following text as an expression: {s}")
    return stream


def parse_protocol_section(lines):
    """Return list of elements from content of the protocol section"""
    i = 0  # index of next line to match; if ≥ len(lines), return immediately

    def match_phase():
        """Match phase definition & advance line number"""
        nonlocal i
        if not lines[i].startswith("phase"):
            return None
        # Phase
        name = lines[i][len("phase") :].strip()
        elements = []
        i += 1
        while i < len(lines):
            if is_ws(lines[i]):
                i += 1
                continue
            if match_comment(lines[i]):
                i += 1
                continue
            if m := match_instruction():
                elements.append(m)
                continue
            if m := match_segment():
                elements.append(m)
                continue
            break
        return Phase(name, elements)

    def match_instruction():
        """Match keyword-prefixed instruction"""
        nonlocal i
        for kw in keywords:
            if lines[i].startswith(kw):
                kw, var, *rest = lines[i].split()
                if len(rest) == 0:
                    setting = None
                elif len(rest) == 1:
                    setting = rest[0]
                else:  # len(rest) > 1
                    raise ValueError(
                        "Keyword instruction in line '{lines[i]}' has too many parts."
                    )
                i += 1
                return Instruction(kw, var, setting)
        return None

    def match_segment():
        """Match segment definition & advance line number"""
        nonlocal i
        if lines[i].startswith("|"):
            statements = []
            # Make sure opening line has a valid statement
            if statement := match_transition(lines[i][1:]):
                statements.append(statement)
                i += 1
            else:
                raise ValueError(
                    f"The following line is a segment definition but does not contain a valid transition statement: '{lines[i]}'"
                )
            # Collect subsequent lines belonging to the same segment
            while i < len(lines):
                if statement := match_transition(lines[i]):
                    statements.append(statement)
                    i += 1
                else:
                    break
            return Segment(statements)
        else:
            return None

    def match_transition(s: str) -> Optional[str]:
        """Match transition statement & advance line number

        Matches optional trailing comment, but expects the leading "|" for the first
        transition in a segment to already be stripped.

        """
        i = 0

        # TODO: This is a hack; handle comments more properly.  Right now
        #  parse_expression switches to a token stream too early and can't cope with
        #
        #  a trailing comment.
        if (idx := s.find("#")) != -1:
            s = s[:idx]

        def skip_ws(s):
            nonlocal i
            if m := re_ws.match(s):
                i += m.end()

        # Symbol
        skip_ws(s[i:])
        if m := Name.match(s[i:]):
            var = m
            i += len(m)
        else:
            return None
        # Transition operator
        skip_ws(s[i:])
        if m := re_trans.match(s[i:]):
            i += m.end()
        else:
            return None
        # Flag for relative change
        skip_ws(s[i:])
        if s[i : i + 1] == "+":
            relative = True
            i += 1
        else:
            relative = False
        # Target
        skip_ws(s[i:])
        if m := parse_expression(s[i:]):
            expr = m
            i += len(expr)
        else:
            return None
        if relative:
            target = RelativeTarget(expr)
        else:
            target = AbsoluteTarget(expr)
        return Transition(var, target)

    protocol = []  # list of ([children], kind, {param: value}) tuples.
    while i < len(lines):
        # Blank line
        if is_ws(lines[i]):
            i += 1
            continue
        # Comment
        if match_comment(lines[i]):
            i += 1
            continue
        # Instruction
        if m := match_instruction():
            # match_keyword advances line number
            protocol.append(m)
            continue
        # Phase
        if m := match_phase():
            # match_phase advances line number
            protocol.append(m)
            continue
        # Block
        if lines[i].startswith("repeat"):
            ln = lines[i]
            kw, count = ln.strip().split(",")
            n = int(count.rstrip("cycles").strip())
            protocol.append(("block", {"n": n}))
            i += 1
            continue
        # Segment
        if m := match_segment():
            # match_segment increments the line number
            protocol.append(m)
            i += 1
            continue
        # Unrecognized
        warn(f"Did not recognize syntax of line: {lines[i]!r}")
        i += 1
    return protocol


def parse_sections(lines, offset=0):
    sections = {}
    path = []
    contents = None  # contents of section under last seen header
    for i, ln in enumerate(lines):
        if is_ws(ln):
            i += 1
            continue
        if ln.startswith("*"):  # On header line
            # Tokenize this header
            l, _, k = ln.strip().partition(" ")
            level = l.count("*")
            name = k.lower()
            # Validation
            if any([a != "*" for a in l]):
                raise ValueError("Line is not a valid header: {}".format(ln))
            if level > len(path) + 1:
                raise ValueError(
                    "Headers skip from level {} to level {}.  Line {}: {}".format(
                        len(path), level, offset + i + 1, ln
                    )
                )
            # Store contents of last header
            if contents is not None:
                _nested_set(sections, path, contents)
            # Prepare to store contents of this header
            path = path[: level - 1] + [name]
            contents = None
        elif contents is None:
            # Found non-blank contents under current header
            contents = []
            contents.append(ln.strip())
        elif contents is not None:
            # Already found non-blank contents under current header.
            # Now we'll append every line we find.
            contents.append(ln.strip())
    # Store the last section
    if contents is not None:
        _nested_set(sections, path, contents)
    return sections


def read_definition(ln):
    sym = ":="
    i = ln.find(sym)
    if i == -1:
        raise ValueError("Line is not a 'key := value' definition: {}".format(ln))
    k = ln[:i].strip()
    v = ln[i + len(sym) :].strip()
    return k, v


def read_default_rates(lines):
    rates = {}
    for ln in lines:
        if not is_ws(ln):
            m = re.match(
                r"(?P<var>\w+)\((\w+)\)"
                r"\s*=\s*"
                r"\w+\(\w+,\s*"
                r"(?P<rate>[\s\S]+)"
                r")",
                ln,
            )
            rate = ureg.parse_expression(m.group("rate"))
            rates[m.group("var")] = rate
    return rates


def read_reference_state(lines):
    reference_values = {}
    for ln in lines:
        if not is_ws(ln):
            m = re.match(r"(?P<var>\w+)" r"\s*=\s*" r"(?P<expr>[\s\S]+)", ln)
            tokens = parse_expression(m.group("expr"))
            value = ureg.Quantity(to_number(tokens[0].num.v), tokens[0].unit.v)
            reference_values[m.group("var")] = value
    return reference_values


def _nested_set(dic, path, value):
    """Set a value in a dictionary at a path of keys"""
    for k in path[:-1]:
        dic = dic.setdefault(k, {})
    dic[path[-1]] = value


def read_prune(p):
    """Read a file as a prunetest protocol

    :param p: File path or a file-like object with a readlines method.

    """
    if isinstance(p, str) or isinstance(p, pathlib.Path):
        # Assume p is a path
        with open(p, "r") as f:
            lines = f.readlines()
    else:
        # Assume p is an object that supports line-oriented IO
        lines = p.readlines()
    # Read metadata
    metadata = {}
    in_metadata = True
    for i, ln in enumerate(lines):
        if ln.isspace():
            # Empty line
            pass
        elif ln.startswith("*"):
            # Is a header; no longer in metadata section
            break
        else:
            # Is a metadata line
            k, v = read_definition(ln)
            metadata[k] = v
    # Read headers and enclosed data
    sections = parse_sections(lines[i:], offset=i)
    # Parse protocol
    p_protocol = parse_protocol_section(sections["protocol"])
    # Read protocol
    defaults = {}
    defaults["rates"] = read_default_rates(
        sections["declarations"].get("default interpolation", [])
    )
    reference_state = read_reference_state(sections["declarations"]["initialization"])
    return protocol.Protocol(reference_state, [e.read() for e in p_protocol])
