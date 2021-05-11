# Base packages
import re
# Local packages
from .unit import ureg

def expand_blocks(elements):
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


def parse_protocol(lines):
    protocol = []  # list of ([children], kind, {param: value}) tuples.
    for i, ln in enumerate(lines):
        ## Phase
        if ln.startswith("phase"):
            kw, lbl = ln.strip().split()
            lbl = lbl.strip('"')
            element = ("phase", {"label": lbl})
            protocol.append(element)
        ## Block
        elif ln.startswith("repeat"):
            kw, count = ln.strip().split(",")
            n = int(count.rstrip("cycles").strip())
            protocol.append(("block", {"n": n}))
        ## Segment
        elif ln.startswith("|"):
            channel, target = ln.lstrip("| ").strip().split("→")
            channel = channel.strip()
            target = ureg.Quantity(ureg.parse_expression(target.strip()))
            protocol.append(("segment", {"channel": channel,
                                         "target": target}))
        elif ln == '' or ln.isspace():
            ## Blank line separator.  These are important to preserve
            ## because they terminate blocks.
            protocol.append(("sep",))
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
                         '(?P<value>[\s\S]+)',
                         ln)
            value = ureg.Quantity(ureg.parse_expression(m.group('value')))
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
