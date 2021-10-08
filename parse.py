# Base packages
import re
# Local packages
from collections import UserString
from enum import Enum, auto
from warnings import warn
from typing import Optional

from .unit import ureg


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


class Segment:
    """Parse data for Segment definition"""
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"{self.__class__.__name__}({self.statements!r})"


# Classes for parsing expressions

class BinOp:
    def __init__(self, op, lval, rval):
        self.op = op
        self.lval = lval
        self.rval = rval


class Number(Token):
    pass


class Operator(Token):
    pass


class Unit(Token):
    pass


class Symbol(Token):
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


class SymbolicValue:
    def __init__(self, symbol, op=None):
        self.symbol = symbol
        self.operator = op

    def __repr__(self):
        return f"{self.__class__.__name__}({self.operator or ''}{self.symbol})"


operators = ("+", "-", "−", "/", "*", "×", "^")
re_bin_op = re.compile("^" + "|".join(["\\" + op for op in operators]))
re_un_op = re.compile(r"^-")

trans_op = ("→", "->")
re_trans = re.compile(r"^→|->")

re_unit = re.compile(f"^[^{''.join(operators)}]" + r"\w·/\-*°]+")

re_ws = re.compile(r"^\s+")

re_num = re.compile(r"-?(\d*\.)?\d+")


# Classes for parsing protocol statements

class Transition:
    """Store parse data for a transition statement, e.g., t → 1 s"""
    def __init__(self, variable, target: "Target", interpolant=None):
        self.variable = variable
        self.target = target
        # The default interpolant at the model layer should be linear,
        # but since this class is just parse data, we store what we
        # found.
        self.interpolant = interpolant

    def __repr__(self):
        return f"{self.__class__.__name__}({self.variable} → {self.target})"


class Target:
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target})"


class AbsoluteTarget(Target):
    pass


class RelativeTarget(Target):
    pass


## Parsing functions

def expand_blocks(elements):
    """Expand blocks of cycles to individual segments"""
    expanded = []
    active = []
    for e in reversed(elements):
        if e[0] == 'segment':
            active.append(e)
        elif e[0] == 'block':
            active = active * e[1]['n']
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
                raise ValueError(f"Unmatched open parenthesis.  Next 10 characters after unmatch parenthesis were: {s[i:i+10]}")
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

    def match_symbol():
        nonlocal i
        if m := Symbol.match(s[i:]):
            i += len(m)
            return Symbol(m)

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
            if sym := match_symbol():
                return SymbolicValue(sym, op=un)
            else:
                raise ValueError(f"Unary operator `{un}` was not followed by a symbolic reference.  The remaining characters in the problem line were: '{s[i:]}'")
        if sym := match_symbol():
            return SymbolicValue(sym)

    stream = []
    while i < len(s):
        match_ws()
        if group := match_group():
            stream.append(tuple(group))
            continue
        if value := match_value():
            stream.append(value)
            continue
        if binary := match_binary_op():
            stream.append(Operator(binary))
            continue
        raise ValueError(f"Could not interpret {s[i:]}")
    return stream


def parse_protocol_section(lines):
    """Return list of elements from content of the protocol section"""
    i = 0  # index of next line to match; if ≥ len(lines), return immediately

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
                raise ValueError(f"The following line is a segment definition but does not contain a valid transition statement: '{lines[i]}'")
            # Collect subsequent lines belonging to the same segment
            while i < len(lines):
                if statement := match_transition(lines[i]):
                    statements.append(statement)
                    i += 1
                else:
                    break
            # still in `if lines[i].startswith("|")`
            return Segment(statements)

    def match_transition(s: str) -> Optional[str]:
        """Match transition statement & advance line number"""
        i = 0

        def consume_ws(s):
            nonlocal i
            if m := re_ws.match(s):
                i += m.end()

        # Symbol
        consume_ws(s[i:])
        if m := Symbol.match(s[i:]):
            var = m
            i += len(m)
        else:
            return None
        # Transition operator
        consume_ws(s[i:])
        if m := re_trans.match(s[i:]):
            i += m.end()
        else:
            return None
        # Flag for relative change
        consume_ws(s[i:])
        if s[i:i+1] == "+":
            relative = True
            i += 1
        else:
            relative = False
        # Target
        consume_ws(s[i:])
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
        ln = lines[i]
        # Phase
        if ln.startswith("phase"):
            kw, lbl = ln.strip().split()
            lbl = lbl.strip('"')
            element = ("phase", {"label": lbl})
            protocol.append(element)
            i += 1
            continue
        # Block
        if ln.startswith("repeat"):
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
        # Blank line separator.  These are important to preserve # because they #
        # terminate blocks.
        if re_ws.match(lines[i]) or lines[i] == "":
            protocol.append(("sep",))
            i += 1
            continue
        # Unrecognized
        warn(f"Did not recognize syntax of line: {lines[i]}")
        i += 1
    return protocol


def parse_sections(lines, offset=0):
    sections = {}
    path = []
    contents = None
    lines = strip_blank_lines(lines)
    if not lines[0].startswith("*"):
        raise ValueError("Input data does not have a header as its first non-blank line.  Line {}: {}".format(offset+1, lines[0]))
    last_nonblank = None  # "header" or "content"
    for i, ln in enumerate(lines):
        if ln.startswith("*"):  # On header line
            # Tokenize this header
            l, _, k = ln.strip().partition(" ")
            level = l.count("*")
            name = k.lower()
            # Validation
            if any([a != "*" for a in l]):
                raise ValueError("Line is not a valid header: {}".format(ln))
            if level > len(path) + 1:
                raise ValueError("Headers skip from level {} to level {}.  Line {}: {}".format(len(path), level, offset+i+1, ln))
            # Store contents of last header
            if contents is not None:
                _nested_set(sections, path, contents)
            # Prepare to store contents of this header
            path = path[:level-1] + [name]
            contents = None
        elif contents is None and not ln.isspace():
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
    v = ln[i+len(sym):].strip()
    return k, v


def read_default_rates(lines):
    rates = {}
    for ln in lines:
        if not _is_blank(ln):
            m = re.match(r'(?P<var>\w+)\((\w+)\)'
                         '\s*=\s*'
                         '\w+\(\w+,\s*'
                         '(?P<rate>[\s\S]+)'
                         '\)', ln)
            rate = ureg.parse_expression(m.group('rate'))
            rates[m.group('var')] = rate
    return rates


def read_reference_state(lines):
    reference_values = {}
    for ln in lines:
        if not _is_blank(ln):
            m = re.match(r'(?P<var>\w+)'
                         '\s*=\s*'
                         '(?P<expr>[\s\S]+)',
                         ln)
            tokens = parse_expression(m.group("expr"))
            value = tokens[0]
            value = ureg.Quantity(value.num.v, value.unit.v)
            reference_values[m.group('var')] = value
    return reference_values


def strip_blank_lines(lines):
    for i0, ln in enumerate(lines):
        if not ln.isspace():
            break
    for i1, ln in enumerate(lines[::-1]):
        if not ln.isspace():
            i1 = len(lines) - i1
            break
    return lines[i0:i1]


def _nested_set(dic, path, value):
    """Set a value in a dictionary at a path of keys"""
    for k in path[:-1]:
        dic = dic.setdefault(k, {})
    dic[path[-1]] = value


def _is_blank(s):
    return s.strip() == ''
