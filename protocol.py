# Base packages
import operator
from collections import OrderedDict, abc
import sys
import warnings

# Third-party packages
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pint
from scipy.optimize import minimize


ureg = pint.UnitRegistry()


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


class Expression:
    """A mathematical expression which may contain symbolic values"""

    def eval(self, parameters=None):
        raise NotImplementedError


class UnOp(Expression):
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

    def eval(self, parameters=None):
        return self.fn[self.op](self.rval.eval(parameters))


class BinOp(Expression):
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

    def __init__(self, op, lval, rval):
        self.op = op
        self.lval = lval
        self.rval = rval

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.op}', {self.lval!r}, {self.rval!r})"

    def __str__(self):
        return f"{self.lval} {self.op} {self.rval}"

    def eval(self, parameters=None):
        return self.fn[self.op](self.lval.eval(parameters), self.rval.eval(parameters))


class Parameter:
    def __init__(self, name, unit, value=None):
        self.name = name
        self.unit = unit
        self.value = value
        if value is not None:
            try:
                value.to(unit)
            except pint.errors.DimensionalityError:
                raise ValueError(
                    f"Value of parameter '{name}' has units of '{value.units}', but its definition requires units compatible with '{unit}'."
                )

    def __str__(self):
        return f"{self.name} [{self.unit}] = {self.value}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.name!r}, {self.unit!r}, {self.value!r})"
        )

    def eval(self):
        return self.value


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


class Q(ureg.Quantity):
    """A literal value, with no parameters

    This class exists for two reasons:

    (1) ureg.Quantity does not have an eval() method, but we want other classes to be
    able to call eval() on any value, without testing if it is parameterized or not.

    (2) Q is shorter than Quantity, and users will type it a lot if they use the
    Python interface.

    """

    def eval(self, *args, **kwargs):
        return self


class SymbolicValue:
    """A symbolic value to be substituted with a literal value during evaluation"""

    def __init__(self, name):
        self.name = name

    def eval(self, parameters: dict):
        if parameters is None:
            raise ValueError(
                "eval() was called on a symbolic value but no parameters were provided."
            )
        return parameters[self.name].eval()


class RelativeTarget:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"+ {self.value}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"

    def eval(self, initial_value, parameters=None):
        return initial_value + self.value.eval(parameters)


class Transition:
    def __init__(
        self,
        variable: str,
        target: Union[Q, Expression, RelativeTarget],
        path="linear",
    ):
        self.variable = variable
        self.target = target
        self.path = path

    def __repr__(self):
        return f"{self.__class__}({self.variable}, {self.target}, {self.path})"

    def __str__(self):
        return f"{self.variable} → {self.target}"


class Segment:
    def __init__(self, transitions: List[Transition]):
        """Return Segment object"""
        self.transitions = {t.variable: t for t in transitions}

    def __repr__(self):
        return f"{self.__class__}({self.transitions})"

    @property
    def variables(self):
        """Return list of this segment's controlled variables"""
        return set(self.transitions.keys())

    def eval_state(self, variable: str, value: Q, initial_state: dict, parameters=None):
        """Return succession of states at an independent variable's values

        The independent variable must be strictly monotonically increasing.
        Typically the independent variable is time or pseudo-time.

        """
        if parameters is None:
            parameters = {}
        # TODO: How do I want to handle relative vs. absolute target_state for eval?
        # Initialize evaluated state
        state = {var: None for var in self.transitions}
        state[variable] = value
        try:
            trans = self.transitions[variable]
        except KeyError:
            raise ValueError(f"Variable '{variable}' is not controlled in this segment")
        try:
            v0 = initial_state[variable]
        except KeyError:
            raise ValueError(f"Variable '{variable}' not in provided initial state.")
        # TODO: refactor this to not have to switch call signatures
        if isinstance(trans.target, RelativeTarget):
            v1 = trans.target.eval(v0, parameters)
        else:
            v1 = trans.target.eval(parameters)
        if trans.path == "linear":
            # Define pseudo-time s; 0 ≤ s ≤ 1, where s = 0 is the start of the segment
            # and s = 1 is the end.
            s_crit = ((value - v0) / (v1 - v0)).m
        else:
            raise NotImplementedError
        for var, trans in self.transitions.items():
            if var == variable:
                # Already handled the abscissa
                continue
            v0 = initial_state[var]
            # TODO: refactor this to not have to switch call signatures
            if isinstance(trans.target, RelativeTarget):
                Δ = trans.target.value.eval(parameters)
            else:
                v1 = trans.target.eval(parameters)
                Δ = v1 - v0
            state[var] = v0 + s_crit * Δ
        return state

    def target_state(self, initial_state, parameters=None):
        """Return target state for all variables

        `target` is basically an `eval` that returns the segment's final state.  As
        such, there is no need to specify an independent variable.  The segment's
        initial state is in general still required to allow calculation of state for
        relative transitions, but if all transitions are absolutely valued (independent
        of initial state) an empty dict may be provided for the initial state.

        """
        state = {}
        for t in self.transitions.values():
            if isinstance(t.target, RelativeTarget):
                state[t.variable] = t.target.eval(initial_state[t.variable], parameters)
            else:
                # Absolute target
                state[t.variable] = t.target.eval(parameters)
        return state


class Block:
    def __init__(self, cycle, n):
        self.cycle = cycle
        self.n = n

    def __repr__(self):
        return f"self.__class__.__name__({self.cycle}, {self.n})"

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

    @property
    def segments(self):
        segments = []
        for e in self.elements:
            if isinstance(e, Segment):
                segments.append(e)
            elif isinstance(e, Block):
                segments.append(seg for seg in e)
            elif isinstance(e, tuple) and e[0] in (
                "set-default",
                "control",
                "uncontrol",
            ):
                # Keyword command, not really supported yet
                continue
            else:
                raise NotImplementedError(
                    "{self.__class__} does not know how to interpret {e}"
                )
        return tuple(segments)


class Protocol:
    def __init__(self, initial_state, elements, parameters: Optional[dict] = None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        # Make the protocol data immutable-ish
        self.initial_state = initial_state
        self.elements = tuple(elements)

        # Dictionaries to access phases by name.  Add dicts for segments and blocks
        # if they get names later.
        self.phase_dict = OrderedDict()

        # Create membership maps
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

    # Perhaps this should be cached for performance?  It would help if the protocol were
    # genuinely immutable.
    @property
    def segments(self) -> abc.Sequence:
        """Return sequence of Segments

        Only segments are included in the return value.  Instructions (e.g.,
        set-default, control, and uncontrol) are not included.

        """
        segments = tuple()
        for part in self.elements:
            if isinstance(part, Segment):
                segments += (part,)
            elif hasattr(part, "segments"):
                segments += part.segments
        return segments

    def eval_state(self, variable: str, value: Q):
        return self.eval_states(variable, [value])[0]

    def eval_states(self, variable: str, values: Iterable[Q]):
        """Return succession of states at an independent variable's values

        :param variable: The independent variable which has values at which the
        protocol's state will be calculated.  The independent variable must be
        strictly monotonically increasing. Typically the independent variable is
        (pseudo-)time.

        :param values: The values of the independent variable at which the protocol's
        state will be calculated.  The values must be monotonically increasing.

        :returns: Sequence of states.  Each state is a map: variable name → value.

        """
        states = []
        i = 0  # index into `values`
        segments = self.segments
        j = 0  # index into segments
        last_state = self.initial_state
        for value in values:
            # Check if the abscissa value is in the initial state
            if value == self.initial_state[variable]:
                states.append(self.initial_state)
                i += 1
                continue
            # Find the segment containing the abscissa (given value).  Note that j is
            # carried over from abscissa to abscissa; this is why they must be
            # provided in monotonically increasing order.
            while j < len(segments):
                segment = segments[j]
                t0 = last_state[variable]
                next_state = segment.target_state(
                    initial_state=last_state, parameters=self.parameters
                )
                t1 = next_state[variable]
                if t0 < value < t1:
                    state = segment.eval_state(
                        variable, value, last_state, self.parameters
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
                # TODO: Found state; ensure all variables are present
                states.append(state)
                break
        return tuple(states)
