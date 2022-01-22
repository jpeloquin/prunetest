# Base packages
import pathlib
import re

# Local packages
from warnings import warn
from typing import Iterable, Optional

from . import protocol
from .protocol import evaluate
from .units import Quantity

precedence = {"^": 2, "/": 1, "*": 1, "×": 1, "·": 1, "+": 0, "-": 0, "−": 0}
operators = set(op for g in precedence for op in g)

trans_op = ("→", "->")
re_trans = re.compile(r"^→|->")

re_ws = re.compile(r"^\s+")

keywords = ("set-default", "unset-default", "fix", "unfix")


def to_number(s):
    """Convert numeric string to int or float as appropriate."""
    try:
        return int(s)
    except ValueError:
        return float(s)


class Token:
    def __init__(self, text: str):
        self.text: str = text

    def __eq__(self, s):
        return str(self) == s

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.text})"

    def __str__(self):
        return str(self.text)


class ParseError(Exception):
    pass


class Comment(Token):
    regex = re.compile("^#")

    @classmethod
    def match(cls, s: str) -> Optional[str]:
        if s.startswith("#"):
            return s[1:].lstrip()
        return None


class Segment:
    """Parse data for a Segment"""

    def __init__(self, transitions: Iterable["Transition"]):
        self.transitions = transitions

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transitions!r})"

    def read(self, variables, parameters=None):
        """Return protocol.Segment object from parse data"""
        transitions = []
        for t in self.transitions:
            if t.path is not None:
                raise ValueError(
                    "A transition with path constraint {t.path} was "
                    "provided.  Non-default paths are not yet supported. "
                    "The default is a linear transition."
                )
            transitions.append(t.read(variables, parameters))
        return protocol.Segment(transitions)


class Instruction:
    """Parse data for a keyword instruction"""

    def __init__(self, keyword: str, variable: str, value: str):
        self.action = keyword
        self.variable = variable
        self.value = value

    def __str__(self):
        return f"'{self.action} {self.variable} {self.value}'"

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.action}', '{self.variable}', '{self.value}')"

    def read(self, variables, parameters=None):
        return protocol.Instruction(variables[self.variable], self.action, self.value)


class Phase:
    """Parse data for Phase"""

    def __init__(self, name, elements):
        self.name = name
        self.elements = elements

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.elements!r})"

    def iter_expand(self):
        for e in self.elements:
            if isinstance(e, Segment) or isinstance(e, Instruction):
                yield e
            elif hasattr(e, "iter_expand"):
                for e2 in e.iter_expand():
                    yield e2
            elif hasattr(e, "segments"):
                for e2 in e.segments:
                    yield e2

    def read(self, variables, parameters=None):
        """Return protocol.Phase instance from parse data"""
        elements = []
        for e in self.elements:
            elements.append(e.read(variables, parameters))
        return protocol.Phase(self.name, elements)


# Classes for parsing expressions


class Number(Token):
    re = re.compile(r"-?(\d*\.)?\d+")

    @classmethod
    def match(cls, s: str) -> Optional["Number"]:
        if m := cls.re.match(s):
            return cls(m.group())

    def read(self):
        return to_number(self.text)


class Unit(Token):
    re_unit = re.compile(
        r"(\w|°)+(\s*"
        f"[{'|'.join(re.escape(op) for op in operators)}]"
        "{1}"
        r"\s*(\w|°)+)*"
    )

    @classmethod
    def match(cls, s: str) -> Optional["Unit"]:
        if m := cls.re_unit.match(s):
            return cls(m.group())

    def read(self):
        # TODO: It would make more sense to read the value and quantity together one
        #  level above this
        return Quantity(self.text).units


class NumericValue:
    # TODO: This class doesn't follow the pattern I've established with other classes
    #  of separating string matching and data structures.
    def __init__(self, text, num, units=Unit("1")):
        self._match = text
        self.num = num
        self.units = units

    def __repr__(self):
        return f"{self.__class__.__name__}({self._match})"

    def __str__(self):
        return self._match

    def __eq__(self, other):
        return self.num == other.num and self.units == other.units

    def __len__(self):
        return len(self._match)

    @classmethod
    def match(cls, s: str) -> Optional["NumericValue"]:
        i = 0
        if num := Number.match(s[i:]):
            i += len(num)
            i += match_ws(s[i:])
            if unit := Unit.match(s[i:]):
                i += len(unit)
                return cls(s[:i], num, unit)
            else:
                return cls(s[:i], num)

    def read(self):
        return Quantity(self.num.read(), self.units.read())


class Symbol(Token):
    regex = re.compile(r"^\w+")

    @classmethod
    def match(cls, s: str) -> Optional["Symbol"]:
        """Return matching start of string or None"""
        if m := cls.regex.match(s):
            return cls(m.group())

    def read(self):
        # Note: SymbolicValue assumes that the name refers to a parameter; at least,
        # it did when this comment was written.
        return protocol.SymbolicValue(self.text)


class UnOp(Token):
    re_un_op = re.compile(r"^[-−]")

    @classmethod
    def match(cls, s):
        if m := cls.re_un_op.match(s):
            return cls(m.group())


class BinOp(Token):
    re_bin_op = re.compile("^" + "|".join(["\\" + op for op in operators]))

    @classmethod
    def match(cls, s):
        if m := cls.re_bin_op.match(s):
            return cls(m.group())


class Expression:
    def __init__(self, tokens):
        if len(tokens) == 0:
            raise ValueError("An expression must comprise at least one token.")
        self.tokens = tokens

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(repr(v) for v in self.tokens)})"

    def __str__(self):
        return " ".join(str(v) for v in self.tokens)

    def __eq__(self, other):
        return self.tokens == other.tokens

    def read(self):
        """Evaluate expression, substituting parameters"""

        def reduce_stack(stack: list):
            rval = stack.pop()
            while stack:
                op = stack.pop()
                if isinstance(op, UnOp):
                    rval = protocol.UnOp(op.text, rval)
                elif isinstance(op, BinOp):
                    lval = stack.pop()
                    rval = protocol.BinOp(op.text, lval, rval)
            return rval

        stack = []
        precedence_level = 0
        i = 0
        while i < len(self.tokens):
            # Convert any value token to evaluable objects
            if hasattr(self.tokens[i], "read"):
                stack.append(self.tokens[i].read())
                i += 1
                continue
            # If the operand is numeric, apply any unary operator now.  This is the
            # highest-precedence rule because the minus sign is considered part of
            # the number.  Probably it should be parsed as part of the number during
            # matching, instead of waiting for reading.
            if isinstance(self.tokens[i], UnOp) and isinstance(
                self.tokens[i + 1], NumericValue
            ):
                stack.append(
                    protocol.UnOp(self.tokens[i].text, self.tokens[i + 1].read())
                )
                i += 2
                continue
            # Apply other operators in order of precedence.
            if isinstance(self.tokens[i], UnOp) or isinstance(self.tokens[i], BinOp):
                if stack and precedence[self.tokens[i].text] <= precedence_level:
                    # Consume the entire stack
                    rval = reduce_stack(stack)
                    stack.append(rval)
                    stack.append(self.tokens[i])
                    precedence_level = precedence[self.tokens[i].text]
                    i += 1
                else:
                    stack.append(self.tokens[i])
                    i += 1
        # Consume the entire stack
        rval = reduce_stack(stack)
        return rval


# Classes for parsing definitions and assignments for parameters and variables


class ParametersSection:
    """Representation of Parameters section"""

    def __init__(self, definitions, values):
        self.definitions = definitions
        self.values = values
        for p, v in values.items():
            if p not in definitions:
                raise ParseError(
                    f"Parameter '{p}' has the value '{v}' but no definition."
                )

    @classmethod
    def match(cls, lines):
        definitions = {}
        values = {}
        for ln in lines:
            ln = ln.strip()
            if ln == "":
                continue
            if m := Definition.match(ln):
                definitions[m.parameter.text] = m
                continue
            if m := Assignment.match(ln):
                values[m.parameter.text] = m.expression
                continue
            raise ParseError(
                f"Could not parse the following line as part of the 'Parameters' section:\n{ln}"
            )
        return cls(definitions, values)

    def read(self):
        parameters = {}
        for p, d in self.definitions.items():
            if p in self.values:
                value = evaluate(self.values[p].read(), parameters)
            else:
                value = None
            units = d.units.read()
            if not value.is_compatible_with(units):
                raise ValueError(
                    f"Value of parameter '{p}' has units of '{value.units}', but its definition requires units compatible with '{units}'."
                )
            parameters[p] = protocol.Parameter(p, value)
        return parameters


class VariablesSection:
    """Representation of Variables section"""

    def __init__(self, definitions):
        self.definitions = definitions

    @classmethod
    def match(cls, lines):
        definitions = {}
        for ln in lines:
            ln = ln.strip()
            if ln == "":
                continue
            if m := Definition.match(ln):
                definitions[m.parameter.text] = m
                continue
            raise ParseError(
                f"Could not parse the following line as part of the 'Variables' section:\n{ln}"
            )
        return cls(definitions)

    def read(self):
        variables = {}
        for nm, d in self.definitions.items():
            unit = d.units.read()
            variables[nm] = protocol.Variable(nm, unit)
        return variables


class Assignment:
    """Assignment; e.g., a = 1 mm

    The right hand side can be an expression, not just a value.

    """

    def __init__(self, parameter, expression):
        self.parameter = parameter
        self.expression = expression

    def __str__(self):
        return f"{self.parameter} = {self.expression}"

    @classmethod
    def match(cls, s: str) -> Optional["Assignment"]:
        """Match assignment statement

        `s` must not have leading whitespace.  Characters after the match are ignored.

        """
        name = None
        expr = None
        i = 0
        if m := Symbol.match(s[i:]):
            i += len(m)
            name = m
        else:
            return None
        i += match_ws(s[i:])
        if not s[i] == "=":
            return None
        i += 1
        i += match_ws(s[i:])
        expr = parse_expression(s[i:])
        return cls(name, expr)


class Definition:
    """Definition; e.g., h [mm] := specimen height"""

    def __init__(self, parameter, units, description):
        self.parameter = parameter
        self.units = units
        self.description = description

    @classmethod
    def match(cls, s: str) -> Optional["Definition"]:
        """Match definition statement at start of string up to newline"""
        i = 0
        if m := Symbol.match(s[i:]):
            name = m
            i += len(m)
        else:
            return None
        i += match_ws(s[i:])
        if s[i] == "[":
            i += 1
        else:
            return None
        if m := Unit.match(s[i:]):
            unit = m
            i += len(m)
        else:
            return None
        if s[i] == "]":
            i += 1
        else:
            return None
        i += match_ws(s[i:])
        if s[i : i + 2] == ":=":
            i += 2
        else:
            return None
        i += match_ws(s[i:])
        newline = s.find("\n")
        if newline == -1:
            newline = len(s)
        description = s[i:newline]
        return Definition(name, unit, description)


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

    def read(self, variables, parameters=None):
        path = self.path if self.path is not None else "linear"
        return protocol.Transition(variables[self.variable], self.target.read(), path)


class Target:
    def __init__(self, expr: Expression):
        self.value = expr

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def read(self):
        """Read target value

        Override in derived classes

        """
        raise NotImplementedError


class AbsoluteTarget(Target):
    def read(self):
        return self.value.read()


class RelativeTarget(Target):
    def read(self):
        return protocol.RelativeTarget(self.value.read())


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

    def match_binary_op():
        nonlocal i
        if m := BinOp.match(s[i:]):
            i += len(m)
            return m

    def match_group(s):
        if s[0] != "(":
            return None
        n_open = 0
        close_paren = None
        i = 0
        while i < len(s):
            if s[i] == "(":
                n_open += 1
            elif s[i] == ")":
                n_open -= 1
            if n_open == 0:
                close_paren = i
                i += 1
                break
            else:
                i += 1
        if close_paren is None:
            raise ParseError(
                f"Unmatched open parenthesis.  Next 20 characters after unmatched parenthesis were: {s[i:i+20]}"
            )
        return s[: close_paren + 1]

    def match_value(s):
        if m := NumericValue.match(s):
            return m
        if m := Symbol.match(s):
            return m

    def skip_ws():
        nonlocal i
        if m := re_ws.match(s[i:]):
            i += m.end()
            return m.group()

    i = 0  # character index; if ≥ len(s), must return immediately
    stream = []
    while i < len(s):
        skip_ws()
        if i == len(s):
            # Have to break manually if everything left was whitespace.
            break
        # Look for a value-like token
        if m := UnOp.match(s[i:]):
            i += len(m)
            stream.append(m)
            # Unary operator must be adjacent to operand
            if m := match_group(s[i:]):
                i += len(m)
                expr = parse_expression(m[1:-1])  # remove parens
                stream.append(expr)
            elif m := match_value(s[i:]):
                i += len(m)
                stream.append(m)
            else:
                raise ParseError(
                    f"Unary operator was not followed by an operand.  Failed to parse the following text as an expression: {s}"
                )
        elif m := match_group(s[i:]):
            i += len(m)
            expr = parse_expression(m[1:-1])
            stream.append(expr)
            skip_ws()
        elif m := match_value(s[i:]):
            i += len(m)
            stream.append(m)
            skip_ws()
        else:
            raise ParseError(
                f"Failed to parse the following text as an expression: {s}"
            )
        # Look for optional following binary operator
        if m := match_binary_op():
            stream.append(m)
    return Expression(stream)


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
                    raise ParseError(
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
                raise ParseError(
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

    def match_transition(s: str) -> Optional[Transition]:
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
        if m := Symbol.match(s[i:]):
            varname = m.text
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
        # An expression is a terminal statement
        if m := parse_expression(s[i:]):
            expr = m
        else:
            return None
        if relative:
            target = RelativeTarget(expr)
        else:
            target = AbsoluteTarget(expr)
        return Transition(varname, target)

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
                raise ParseError("Line is not a valid header: {}".format(ln))
            if level > len(path) + 1:
                raise ParseError(
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
        raise ParseError("Line is not a 'key := value' definition: {}".format(ln))
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
                r"(?P<rate>[\s\S]+)",
                ln,
            )
            rate = ureg.parse_expression(m.group("rate"))
            rates[m.group("var")] = rate
    return rates


def read_reference_state(lines, variables, parameters):
    reference_values = {}
    for ln in lines:
        if not is_ws(ln):
            m = re.match(r"(?P<var>\w+)" r"\s*=\s*" r"(?P<expr>[\s\S]+)", ln)
            # TODO: A read call probably shouldn't trigger parsing, because that
            #  might force us to pass variable lists, parameter lists, and other
            #  context to the parser.
            expr = parse_expression(m.group("expr"))
            varname = m.group("var")
            reference_values[variables[varname]] = expr.read()
    return protocol.State(reference_values)


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
    i = 0
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
    # Read parameters (and their default values)
    parameters = ParametersSection.match(sections["definitions"]["parameters"]).read()
    # Read variables
    variables = VariablesSection.match(sections["definitions"]["variables"]).read()
    # Parse protocol
    p_protocol = parse_protocol_section(sections["protocol"])
    # Read protocol
    defaults = {
        "rates": read_default_rates(
            sections["definitions"].get("default interpolation", [])
        )
    }
    reference_state = read_reference_state(
        sections["definitions"]["initialization"], variables, parameters
    )
    return protocol.Protocol(
        reference_state,
        [e.read(variables, parameters) for e in p_protocol],
        variables.values(),
        parameters.values(),
    )
