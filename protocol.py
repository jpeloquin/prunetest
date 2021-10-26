# Base packages
from collections import OrderedDict, abc
import sys
import warnings

# Third-party packages
import numpy as np
from scipy.optimize import minimize

# Local packages
from .unit import ureg


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


class Target:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class AbsoluteTarget(Target):
    def __str__(self):
        return f"{self.value}"


class RelativeTarget(Target):
    # Should the relative target store a reference to the prior state?
    # If not, the absolute target can only be resolved by walking a
    # sequence of (state, segment, …).  That's true regardless but the
    # interface could hide the fact.
    def __str__(self):
        return f"+ {self.value}"


class Transition:
    def __init__(self, variable: str, target: Target, path="linear"):
        self.variable = variable
        self.target = target
        self.path = path

    def __repr__(self):
        return f"{self.__class__}({self.variable}, {self.target}, {self.path})"

    def __str__(self):
        return f"{self.variable} → {self.target}"


class Segment:
    def __init__(self, transitions: list[Transition]):
        """Return Segment object"""
        self.transitions = {t.variable: t for t in transitions}

    def __repr__(self):
        return f"{self.__class__}({self.transitions})"

    @property
    def variables(self):
        """Return list of this segment's controlled variables"""
        return set(self.transitions.keys())

    def eval(self, variable, value, initial_state):
        """Return succession of states at an independent variable's values

        The independent variable must be strictly monotonically increasing.
        Typically the independent variable is time or pseudo-time.

        """
        # TODO: How do I want to handle relative vs. absolute targets for eval?
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
        v1 = trans.target.value
        if trans.path == "linear":
            # Define pseudo-time s; 0 ≤ s ≤ 1, where s = 0 is the start of the segment
            # and s = 1 is the end.
            s_crit = (value - v0) / (v1 - v0.m)
            for var in self.transitions:
                if var == variable:
                    # Already handled the abscissa
                    continue
                v0 = initial_state[var]
                v1 = self.transitions[var].target
                state[var] = s_crit * (v1 - v0)
        else:
            raise NotImplementedError
        return state

    def targets(self, initial_state):
        """Return target state for all variables

        `target` is basically an `eval` that returns the segment's final state.  As
        such, there is no need to specify an independent variable.  The segment's
        initial state is in general still required to allow calculation of state for
        relative transitions, but if all transitions are absolutely valued (independent
        of initial state) an empty dict may be provided for the initial state.

        """
        state = {}
        for t in self.transitions:
            if isinstance(t, AbsoluteTarget):
                state[t.variable] = t.target.value
            else:
                state[t.variable] = t.target.value + initial_state[t.variable]
        return state


class LinearSegment:

    # Delete this class as soon as the new Segment class has working eval and target
    # methods.

    def __init__(self, previous=None):
        self.channel = ""
        self.target = None
        self.rate = None
        self.previous = previous

    @classmethod
    def from_intermediate(cls, struct, previous=None):
        segment = cls(previous)
        segment.channel = struct[1]["channel"]
        segment.target = struct[1]["target"]
        segment.rate = struct[1]["rate"]
        return segment

    @property
    def end_state(self):
        return {self.channel: self.target}

    @property
    def duration(self):
        try:
            initial = self.previous.end_state[self.channel]
        except KeyError:
            return 0
        change = self.target - initial
        duration = abs(change) / self.rate
        return duration.to("s").m

    def evaluate(self, t):
        """Eval segment at relative time t.

        Returns dict with channels as keys.  Values are None if the
        channel is free.

        """
        # But if the segment is supposed to have values defined relative
        # to the previous segment, how can eval figure out what the
        # absolute values should be?
        if isinstance(t, str):
            t = ureg.parse_expression(t)
        eps = 2 * sys.float_info.epsilon / self.rate.m
        initial = self.previous.end_state[self.channel]
        change = self.target - initial
        duration = abs(change) / self.rate
        if t >= duration * (1 + eps):
            raise (
                ValueError(
                    "t = {} > {}; t exceeds the segment's duration.".format(t, duration)
                )
            )
        return initial + t / duration * change


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
    def __init__(self, initial_state, elements):
        # Make the protocol data immutable-ish
        self.initial_state = initial_state
        self.elements = tuple(elements)

        # Dictionaries to access phases by name.  Add dicts for segments and blocks
        # if they get names later.
        self.phase_dict = OrderedDict()

        # Create membership maps
        self.phase_of_segment = {}
        self.phase_of_block = {}
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
                self.phase_of_segment[e] = None
            if isinstance(element, Block):
                self.phase_of_block[e] = None

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

    def eval(self, variable: str, values):
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
        while i < len(values):
            value = values[i]
            while j < len(segments):
                segment = segments[j]
                t0 = last_state[variable]
                next_state = segment.targets(initial_state=last_state)
                t1 = next_state[variable]
                if t0 < value < t1:
                    s = segment.eval(variable, value, last_state)
                    states.append(s)
                    i += 1
                    break
                elif value == t1:
                    # Special case when abscissa is at the end of a segment so we get
                    # exact values.
                    states.append(next_state)
                    break
                elif value <= t0:
                    raise ValueError(f"Provided values of {variable} are not monotonically increasing")
                else:
                    # independent variable value not in this segment; check next segment
                    j += 1
                    continue
        return tuple(states)
