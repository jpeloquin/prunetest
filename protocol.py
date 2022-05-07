# Base packages
import operator
from pathlib import Path
from collections import OrderedDict, abc

# Third-party packages
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from scipy.optimize import minimize

from .units import Unit, Quantity


class ControlConflictError(ValueError):
    def __init__(self, var, msg):
        msg = f"Control conflict for variable '{var}': {msg}"
        super().__init__(msg)


class InvariantViolation(Exception):
    """Raise this error if an invariant is violated

    If this error is raised, there is a bug in spamneggs.

    """

    pass


def evaluate(obj, parameters=None):
    """Return literal value from expression or value-like object"""
    if hasattr(obj, "eval"):
        return obj.eval(parameters=parameters)
    elif not isinstance(obj, Quantity):
        # This check could be deleted.  Users could in principle use other literals,
        # even though prunetest should always use Quantity.  For now, I'll leave it
        # to help identify bugs in prunetest.
        raise TypeError(
            f"Expected literal value of type {Quantity}, but was given value of type {type(obj)}"
        )
    return obj


def iter_expanded(elements):
    """Return iterator over instructions and segments, expanding phases and blocks"""
    for e in elements:
        if isinstance(e, Segment) or isinstance(e, Instruction):
            yield e
        elif hasattr(e, "iter_expanded"):
            for e2 in e.iter_expanded():
                yield e2
        elif hasattr(e, "segments"):
            for e2 in e.segments:
                yield e2


def label_data(protocol, data, control_var):
    """Label a table of test data based on a protocol.

    The protocol is defined as a table of test segments with at least a
    column named "Time [s]" and a column with name == `control_var`.

    Warning: `protocol` will be mutated to update its initial state to
    match the control channel's data.  This is necessary for fitting of
    the first segment.

    Future versions of this function will support an arbitrarily large
    number of control variables.

    """
    # TODO: ensure units of data match units of reference state
    state = protocol.initial_state.end_state
    state[control_var] = (
        data[control_var].iloc[0] * protocol.initial_state.end_state[control_var].units
    )
    protocol.initial_state = InitialState(state)
    protocol.segments[0].previous = protocol.initial_state
    protocol_change_points = np.cumsum(
        [0] + [seg.duration for seg in protocol.segments]
    )
    # TODO: Standardize variable's units to the unit declaration under
    # Channels section, rather than the first unit encountered.
    unit = protocol.initial_state[control_var].units
    protocol_values = [protocol.initial_state[control_var].m] + [
        seg.end_state[control_var].to(unit).m for seg in protocol.segments
    ]
    tab_protocol = {"Time [s]": protocol_change_points, control_var: protocol_values}
    time_points = tab_protocol["Time [s]"].copy()
    # ↑ times of change points
    def f(p):
        """Fit quality metric for fitting origin point."""
        # Time points for comparison are drawn from the protocol rather
        # than the data because we want the number of residuals to
        # remain constant regardless of how much of the test data ends
        # up spanned by the fitted protocol segments.

        # i is meant to refer to variable in the enclosing environment
        if i == 0:
            s_d = p[1:]  # duration, fit, by segment
            i0 = 0
            i1 = 2
        else:
            i0 = i - 1
            i1 = i + 1
            s_d = p  # duration, fit, by segment
        s_tf = time_points[i0] + np.cumsum(np.hstack([[0], s_d]))
        # ^ time, fit, by segment
        s_dp = np.diff(tab_protocol["Time [s]"][i0 : i1 + 1])
        # ^ duration, protocol, by segment
        tf = np.hstack(
            [np.linspace(s_tf[j], s_tf[j + 1], 10) for j in range(len(s_tf) - 1)]
        )
        # ^ time, fit, dense
        tp = np.hstack(
            [
                np.linspace(
                    tab_protocol["Time [s]"][i0 + j],
                    tab_protocol["Time [s]"][i0 + j + 1],
                    10,
                )
                for j in range(len(s_tf) - 1)
            ]
        )
        # ↑ time, protocol, dense
        yp = np.interp(
            tp,
            tab_protocol["Time [s]"],  # y, protocol, dense
            tab_protocol[control_var],
        )
        yd = np.interp(tf, data["Time [s]"], data[control_var])  # y, data, dense
        yf = np.interp(
            tf, s_tf, tab_protocol[control_var][i0 : i1 + 1]
        )  # y, fit, dense
        r = np.corrcoef(yf, yd)[0, 1]
        # r = np.cov(yf, yd)[0,1] / np.cov(yp, yp)[0,0]
        if np.isnan(r):
            if np.var(yf) == 0 and np.var(yd) == 0:
                reason = "The values of the control variable `{}` have zero variance in both the *provided data* and the *current protocol fit* for {} ≤ t ≤ {}."
            elif np.var(yd) == 0:
                reason = "The values of the control variable `{}` have zero variance in the *provided data* for {} ≤ t ≤ {}."
            elif np.var(yf) == 0:
                reason = "The values of the control variable `{}` have zero variance in the *current protocol fit* for {} ≤ t ≤ {}."
            else:
                reason = "The cause is unknown."
            msg = (
                "The Pearson correlation coefficient between the provided data and the current protocol fit is undefined. "
                + reason.format(control_var, tf[0], tf[-1])
            )
            raise (RuntimeError(msg))
        # Add time dilation penalty
        penalty = np.sum((abs(s_d - s_dp) / s_dp) ** 3.0) / len(s_dp)
        # stdout.write("r = {:.4f}  penalty = {:.4f}  ".format(r, penalty))
        # print("p = {}".format(p))
        return -r + penalty

    for i in range(len(tab_protocol["Time [s]"])):
        if i == 0:
            p0 = np.hstack([[0], np.diff(time_points[i : i + 3])])
        else:
            p0 = np.diff(tab_protocol["Time [s]"][i - 1 : i + 2])
        # print("\ni = {}".format(i))
        bounds = [(0, np.inf) for x in p0]
        result = minimize(f, p0, method="L-BFGS-B", bounds=bounds)
        time_points[i] = time_points[max([0, i - 1])] + result["x"][0]
    labeled = ProtocolData(data, data["Time [s]"], protocol, time_points)
    return labeled


class UnOp:
    fn = {"-": operator.neg, "−": operator.neg}

    def __init__(self, operator, operand):
        if operator == "-":
            operator = "−"
        self.op = operator
        self.rval = operand

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.op}', {self.rval!r})"

    def __str__(self):
        return f"{self.op}{self.rval}"

    def __eq__(self, other):
        return all((self.op == other.op, self.rval == other.rval))

    def eval(self, parameters=None):
        return self.fn[self.op](evaluate(self.rval, parameters))


class BinOp:
    fn = {
        "+": operator.add,
        "-": operator.sub,
        "−": operator.sub,
        "/": operator.truediv,
        "*": operator.mul,
        "×": operator.mul,
        "·": operator.mul,
        "^": operator.pow,
    }

    def __init__(self, op: str, lval, rval):
        self.op = op
        self.lval = lval
        self.rval = rval

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.op}', {self.lval!r}, {self.rval!r})"

    def __str__(self):
        return f"{self.lval} {self.op} {self.rval}"

    def __eq__(self, other):
        return all(
            (self.op == other.op, self.lval == other.lval, self.rval == other.rval)
        )

    def eval(self, parameters=None):
        return self.fn[self.op](
            evaluate(self.lval, parameters), evaluate(self.rval, parameters)
        )


class Parameter:
    def __init__(self, name, value: Union["Unit", "Q"]):
        """Return Parameter object

        May be called two ways:

        `Parameter(name, unit)` returns a parameter with the given units but no
        default value.

        `Parameter(name, quantity)` returns a parameter with a default value and the
        units of that value.

        """
        self.name = name
        if isinstance(value, Unit):
            self.units = value
            self._value = None  # bypass value.setter
        else:
            if not hasattr(value, "units"):
                raise ValueError(
                    f"Parameter '{self.name}': Parameter initialization with `Parameter(name, value)` requires the existence of `value.units`."
                )
            self.units = value.units
            self.value = value

    def __str__(self):
        return f"{self.name} [{self.units}] = {self.value}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.name!r}, {self.units!r}, {self.value!r})"
        )

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, Quantity):
            try:
                value = Quantity(value)
            except ValueError:
                raise ValueError(
                    f"Parameter '{self.name}': The input value must be a Quantity or convertible to a Quantity."
                )
        if not value.is_compatible_with(self.units):
            raise ValueError(
                f"Parameter '{self.name}': The input value {value} has units incompatible with '{self.units}'"
            )
        self._value = value

    def eval(self):
        return self.value


class Variable:
    def __init__(self, name, units: Union[str, Unit]):
        self.name = name
        if isinstance(units, str):
            self.units = Unit(units)
        else:
            self.units = units

    def __str__(self):
        return f"{self.name} [{self.units}]"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.units!r})"

    def __hash__(self):
        return hash(self.name)


class ProtocolData:
    def __init__(self, data, t, protocol, change_points):
        """Construct an object to store protocol labels for a dataset.

        t := array of data point times in (units: seconds).

        data := table or array of data values.  Must be sliceable; e.g.,
        data[i:j].  Usually a pandas.DataFrame object.

        change_points := array of time points (units: seconds) in
        ascending order, specifying each change point in the protocol.
        The length of `change_points` is equal to the number of segments
        + 1.  (The start of the test counts as a change in control and
        thus a change point.)

        """
        self.change_points = change_points

        for k in [
            "segments",
            "blocks",
            "phases",
            "segment_dict",
            "block_dict",
            "phase_dict",
            "phase_of_block",
            "phase_of_segment",
        ]:
            self.__dict__[k] = protocol.__dict__[k]

        self.data_times = t
        self.data = data
        self.initial_state = protocol.initial_state

        for i in range(len(protocol.segments)):
            # i = index of first change point for current segment
            # j = index of second change point for current segment
            j = i + 1
            t0 = change_points[i]
            t1 = change_points[j]
            m = np.logical_and(t0 < self.data_times, self.data_times <= t1)
            self.segments[i].data = self.data[m]
            self.segments[i].times = [t0, t1]


# TODO: Delete?
class InitialState:
    def __init__(self, reference_state):
        self._state = reference_state
        self.end_state = self._state
        self.duration = 0

    def evaluate(self, t):
        if t != 0:
            raise ValueError("t = {} outside of segment bounds.".format(t))
        return self.end_state

    def __getitem__(self, k):
        return self._state[k]


class Instruction:
    """Keyword instruction, mostly for variable defaults"""

    def __init__(self, variable: Variable, action, value=None):
        self.variable = variable
        self.action = action
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.variable}', '{self.action}', '{self.value}')"

    def apply_to_segment(self, segment: "Segment"):
        # We could check if the variable was already controlled, but that's not
        # really a concern for the Instruction itself.
        if self.variable not in segment.constrained_vars:
            if self.value == "hold":
                t = Transition(
                    self.variable, RelativeTarget(Quantity(0, self.variable.units))
                )
                segment.set_constraint(t)
            if self.value == "free":
                segment.set_unconstrained(self.variable)


class State:
    """A collection of variable values

    If the value of a prameter is unknown (unconstrained or "free"), its value is
    represented as None.

    """

    def __init__(self, values: Dict[Variable, Quantity]):
        self.by_name = {v.name: q for v, q in values.items()}
        self.by_var = {v: q for v, q in values.items()}  # copy

    def __eq__(self, other):
        if isinstance(other, State):
            return self.by_var == other.by_var
        else:
            return NotImplemented  # fallback to other object

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.by_name[item]
        else:
            return self.by_var[item]

    def __repr__(self):
        nm = self.__class__.__name__
        s = f"{nm}({{"
        for i, (var, val) in enumerate(self.by_var.items()):
            if i != 0:
                s += f",\n  {' '*len(nm)}"
            s += f"{var}, {val}"
        s += "})"
        return s

    def __str__(self):
        s = "State:"
        for var, val in self.by_var.items():
            s += f"\n  {var.name} [{var.units}] = {val}"
        return s

    def eval(self, parameters):
        return State({k: evaluate(v, parameters) for k, v in self.by_var.items()})


class SymbolicValue:
    """A symbolic value to be substituted with a literal value during evaluation"""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __eq__(self, other):
        return self.name == other.name

    def eval(self, parameters: dict):
        if not parameters:
            raise ValueError(
                "eval() was called on a symbolic value but no parameters were provided."
            )
        return evaluate(parameters[self.name].value)


class RelativeTarget:
    def __init__(self, value):
        self.value: Quantity = value

    def __str__(self):
        return f"+ {self.value}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"

    def eval(self, initial_value, parameters=None):
        # TODO: Try to make this compatible with the general evaluate() function, despite
        #  the need for an initial value.
        if initial_value is None:
            return None
        else:
            return evaluate(initial_value, parameters) + evaluate(
                self.value, parameters
            )


class Transition:
    def __init__(
        self,
        variable: Variable,
        target: Union[Quantity, UnOp, BinOp, RelativeTarget],
        path="linear",
    ):
        self.variable = variable
        self.target = target
        self.path = path

    def __repr__(self):
        return f"{self.__class__.__name__}({self.variable}, {self.target}, {self.path})"

    def __str__(self):
        return f"{self.variable} → {self.target}"


class Segment:
    def __init__(
        self,
        transitions: Iterable[Transition],
        extra_vars: Iterable[Variable] = set(),
    ):
        """Return Segment object"""
        self._transitions: Dict[str, Transition] = {
            t.variable.name: t for t in transitions
        }
        self._free = {v.name: v for v in set(extra_vars) - self.constrained_vars}

    def __repr__(self):
        parts = [f"{self._transitions[v.name]}" for v in self.constrained_vars] + [
            f"{v!r} → free" for v in self.unconstrained_vars
        ]
        prefix = f"{self.__class__.__name__}({{"
        suffix = "})"
        s = prefix + f",\n{' '*len(prefix)}".join(parts) + suffix
        return s

    @property
    def variables(self):
        return self.constrained_vars | self.unconstrained_vars

    @property
    def constrained_vars(self):
        return set(t.variable for t in self._transitions.values())

    @property
    def unconstrained_vars(self):
        return set(self._free.values())

    @property
    def transitions(self):
        return self._transitions

    def set_constraint(self, transition):
        """Constrain a variable in the segment (idempotent, mutates)

        The variable is added to the segment if necessary.
        """
        # TODO: Add test to make sure this overrides a "free" variable
        self._transitions[transition.variable.name] = transition
        if transition.variable.name in self._free:
            del self._free[transition.variable.name]

    def set_unconstrained(self, var):
        """Unconstrain a variable in the segment (idempotent, mutates)

        The variable is added to the segment if necessary.
        """
        if var.name in self._transitions:
            del self._transitions[var.name]
        else:
            self._free[var.name] = var

    def eval_state(
        self,
        xvar: Union[str, Variable],
        value: Quantity,
        initial_state: State,
        parameters=None,
        extra_vars: Iterable[Variable] = set(),
    ):
        """Return the state at a particular value of an independent variable

        The independent variable must be strictly monotonically increasing.
        Typically the independent variable is time or pseudo-time.

        :param extra_vars: Variables that must be included in the calculated state as
        unconstrained variables if they are not already constrained by the segment.
        If any variable in this list is constrained by the segment, it will be
        instead calculated according to those constraints, exactly as if it were not
        in this list.

        """
        if parameters is None:
            parameters = {}
        # Resolve pseudotime variable's name
        if isinstance(xvar, str):
            xvar = self._transitions[xvar].variable
        # Make the initial state concrete
        # initial_state = evaluate(initial_state, parameters)
        # Find where we're going to interpolate the transitions
        try:
            t = self._transitions[xvar.name]
        except KeyError:
            raise ValueError(
                f"Variable '{xvar.name}' is not controlled in this segment"
            )
        # Interpolate the abscissa
        try:
            v0 = initial_state[xvar]
        except KeyError:
            raise ValueError(f"Variable '{xvar.name}' not in provided initial state.")
        # TODO: refactor this to not have to switch call signatures
        if isinstance(t.target, RelativeTarget):
            v1 = t.target.eval(v0, parameters)
        else:
            v1 = evaluate(t.target, parameters)
        if t.path == "linear":
            # Define pseudo-time s; 0 ≤ s ≤ 1, where s = 0 is the start of the segment
            # and s = 1 is the end.
            s_crit = ((value - v0) / (v1 - v0)).m
        else:
            raise NotImplementedError
        # Calculate the state by variable interpolation
        state = {xvar: value}
        for var in self.constrained_vars:
            if var == xvar:
                # We already did the abscissa variable
                continue
            t = self.transitions[var.name]
            v0 = initial_state[t.variable]
            # TODO: When support for time shift operators (referencing past values)
            #  is added, refactor this to store the variable's offset relative to the
            #  last unconstrained value.
            if v0 is None:
                state[var] = None
                continue
            # TODO: refactor this to not have to switch call signatures
            if isinstance(t.target, RelativeTarget):
                Δ = evaluate(t.target.value, parameters)
            else:
                v1 = evaluate(t.target, parameters)
                Δ = v1 - v0
            state[t.variable] = v0 + s_crit * Δ
        free = self.unconstrained_vars | (set(extra_vars) - self.constrained_vars)
        for var in free:
            state[var] = None
        return State(state)

    def target_state(self, initial_state={}, parameters={}, extra_vars=set()):
        """Return target state for all variables

        `target_state` is basically an `eval_state` that returns the segment's final
        state.  As such, there is no need to specify an independent variable.  The
        segment's initial state is required if the segment includes relative
        transitions.  If all transitions are absolutely valued (independent of initial
        state) an empty dict may be provided for the initial state.

        :param extra_vars: Variables that must be included in the target state as
        unconstrained variables if not constrained by the segment's target state. If
        any variable in this list is already constrained by the segment, it will be
        instead calculated according to those constraints, exactly as if it were not
        in this list.

        """
        # Workaround for mutable default.  parameters probably won't be mutated here
        # regardless, but it is passed to other functions so no guarantee.
        if parameters == {}:
            parameters = {}
        state = {}
        for var in self.constrained_vars:
            t = self._transitions[var.name]
            if isinstance(t.target, RelativeTarget):
                state[t.variable] = t.target.eval(initial_state[t.variable], parameters)
            else:
                # Absolute target
                state[t.variable]: Quantity = evaluate(t.target, parameters)
        free = self.unconstrained_vars | (set(extra_vars) - self.constrained_vars)
        for var in free:
            state[var] = None
        return State(state)


class Block:
    def __init__(self, cycle, n):
        self.cycle = cycle
        self.n = n

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cycle}, {self.n})"

    def __iter__(self):
        """Return iterator over segments with cycle repeats"""
        iterator = (
            self.cycle[i] for ncycles in range(self.n) for i in range(len(self.cycle))
        )
        return iterator

    @property
    def cycles(self):
        return tuple(self.cycle for ncycles in range(self.n))

    @property
    def segments(self):
        return tuple(iter(self))


class Phase:
    def __init__(self, name, elements):
        self.name = name
        self.elements = elements

    def __repr__(self):
        return f"{self.__class__}({self.name}, {self.elements})"

    def iter_expanded(self):
        """Return iterator over instructions and segments, expanding phases and blocks"""
        for e in iter_expanded(self.elements):
            yield e

    @property
    def segments(self):
        segments = []
        for e in self.elements:
            if isinstance(e, Segment):
                segments.append(e)
            elif isinstance(e, Block):
                segments.append(seg for seg in e)
            elif isinstance(e, Instruction):
                # Keyword command, not really supported yet
                continue
            else:
                raise NotImplementedError(
                    "{self.__class__} does not know how to interpret {e}"
                )
        return tuple(segments)


class Protocol:
    def __init__(
        self,
        initial_state: State,
        elements,
        variables: Optional[Iterable[Variable]] = None,
        parameters: Optional[Iterable[Parameter]] = None,
    ):
        # Make the protocol data immutable-ish
        self.initial_state = initial_state
        self.elements = tuple(elements)
        # Collect parameters
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = {p.name: p for p in parameters}
        # Collect variables
        if variables is None:
            # A list of variables was not provided.  Build a list of variables that
            # are referenced in the protocol (this is for user convenience).
            self.variables = {}
            for seg in self.segments:
                for nm, var in seg._variables.items():
                    self.variables[nm] = var
        else:
            # A list of variables was provided.  Ensure no undefined variables are used.
            self.variables = {var.name: var for var in variables}
            for seg in self.segments:
                for var in seg.variables:
                    if var.name not in self.variables:
                        raise ValueError(
                            f"{var} was referenced in this Segment, but was not present in the list of variables provided to the Protocol."
                        )
        # Expand defaults and check/populate variables list
        active = {}  # active defaults
        for e in self.iter_expanded():
            if isinstance(e, Instruction):
                # Keep track of active instructions
                new = e
                old = active.get(new.variable, None)
                if new.action == "fix" or new.action == "set-default":
                    if old is not None and old.action == "fix":
                        # You can't adjust a fixed variable except to unfix it.
                        raise ControlConflictError(
                            new.variable,
                            f"{new} was given while a 'fix' instruction was active.  Use 'unfix' to remove this restriction.",
                        )
                    active[new.variable] = new
                    continue
                if new.action == "unfix":
                    if old is None or old.action != "fix":
                        raise ControlConflictError(
                            new.variable,
                            f"Can't 'unfix' without an active 'fix' instruction.",
                        )
                    del active[new.variable]
                    continue
                if new.action == "unset-default":
                    if old is None or old.action != "set-default":
                        raise ControlConflictError(
                            new.variable,
                            f"Can't 'unset-default' without an active 'set-default' instruction.",
                        )
                    del active[new.variable]
                    continue
                active[new.variable] = new
                continue
            if isinstance(e, Segment):
                for var, instr in active.items():
                    instr.apply_to_segment(e)
        # Dictionaries to access phases by name.  Add dicts for segments and blocks
        # if they get names later.
        self.phase_dict = OrderedDict()
        self.phase_of_segment: Dict[Segment, Optional[Phase]] = {}
        self.phase_of_block: Dict[Segment, Optional[Block]] = {}
        for element in self.elements:
            if isinstance(element, Phase):
                phase = element
                self.phase_dict[phase.name] = phase
                # Segments
                for segment in phase.segments:
                    self.phase_of_segment[segment] = phase
                # Blocks
                for e in phase.elements:
                    if isinstance(e, Block):
                        self.phase_of_block[e] = phase
            # Outside of any phase
            if isinstance(element, Segment):
                self.phase_of_segment[element] = None
            if isinstance(element, Block):
                self.phase_of_block[element] = None

    def __repr__(self):
        return f"{self.__class__}({self.initial_state}, {self.elements})"

    def __str__(self):
        s = f"Protocol with {len(self.elements)} elements:\n"
        for e in self.elements:
            s += f"  {e}\n"
        return s

    def iter_expanded(self):
        """Return iterator over instructions and segments, expanding phases and blocks"""
        for e in iter_expanded(self.elements):
            yield e

    # Perhaps this should be cached for performance?  It would help if the protocol were
    # genuinely immutable.
    @property
    def segments(self) -> abc.Sequence:
        """Return sequence of Segments

        Only segments are included in the return value.  Instructions (e.g.,
        set-default, fix, and unfix) are not included.

        """
        segments = tuple()
        for part in self.elements:
            if isinstance(part, Segment):
                segments += (part,)
            elif hasattr(part, "segments"):
                segments += part.segments
        return segments

    def eval_state(self, variable: Union[str, Variable], value: Quantity):
        # Resolve names
        if isinstance(variable, str):
            variable = self.variables[variable]
        if not isinstance(value, Quantity):
            raise ValueError(
                "Must provide a single Quantity to `eval_state`.  To evaluate multiple states, use `eval_states`."
            )
        return self.eval_states(variable, [value])[0]

    def eval_states(
        self,
        variable: Union[str, Variable],
        values: Iterable[Quantity],
        parameters=None,
        extra_vars: Iterable[Variable] = set(),
    ):
        """Return succession of states at an independent variable's values

        :param variable: The independent variable which has values at which the
        protocol's state will be calculated.  The independent variable must be
        strictly monotonically increasing. Typically the independent variable is
        (pseudo-)time.

        :param values: The values of the independent variable at which the protocol's
        state will be calculated.  The values must be monotonically increasing.

        :param parameters: Dictionary of parameter values, used in place of the
        protocol's dictionary of default parameter values when evaluating expressions.

        :returns: Sequence of states.  Each state is a map: variable name → value.

        """
        # Resolve variable name
        if isinstance(variable, str):
            variable = self.variables[variable]
        # Get parameter values
        if parameters is None:
            parameters = self.parameters
        # Get list of all variables
        extra_vars = set(self.variables.values()) | extra_vars
        # Evaluate states
        states: List[State] = []
        initial_state = evaluate(self.initial_state, parameters)
        i = 0  # index into `values`
        segments = self.segments
        j = 0  # index into segments
        last_state = initial_state
        for value in values:
            # Check if the abscissa value is in the initial state
            if value == initial_state[variable]:
                states.append(initial_state)
                i += 1
                continue
            # Find the segment containing the abscissa (given value).  Note that j is
            # carried over from abscissa to abscissa; this is why they must be
            # provided in monotonically increasing order.
            while j < len(segments):
                segment = segments[j]
                t0 = last_state[variable]
                next_state = segment.target_state(
                    initial_state=last_state,
                    parameters=parameters,
                    extra_vars=extra_vars,
                )
                t1 = next_state[variable]
                if t0 < value < t1:
                    state = segment.eval_state(
                        variable, value, last_state, parameters, extra_vars
                    )
                elif value == t1:
                    # Abscissa is at the end of a segment; get exact values
                    state = next_state
                elif value <= t0:
                    raise ValueError(
                        f"Provided values of {variable} are not monotonically increasing"
                    )
                else:
                    # independent variable value not in this segment; check next segment
                    j += 1
                    last_state = next_state
                    continue
                # Ensure all variables are present in the calculated state
                for var in self.variables.values():
                    if not var in state.by_var:
                        raise InvariantViolation(
                            f"Variable '{var}' was defined in {Protocol} but not in a state calculated from that protocol."
                        )
                states.append(state)
                break
        # TODO: Check for out-of-bounds error.  Maybe add a special terminating segment
        # with infinite duration?
        return tuple(states)
