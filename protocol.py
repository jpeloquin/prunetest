# Base packages
import pathlib
from collections import OrderedDict
import sys
import warnings
# Third-party packages
import numpy as np
from scipy.optimize import minimize
# Local packages
from .unit import ureg
from . import parse

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
    state[control_var] = data[control_var].iloc[0] *\
        protocol.initial_state.end_state[control_var].units
    protocol.initial_state = InitialState(state)
    protocol.segments[0].previous = protocol.initial_state
    protocol_change_points = np.cumsum([0] + [seg.duration for seg in
                                              protocol.segments])
    # TODO: Standardize variable's units to the unit declaration under
    # Channels section, rather than the first unit encountered.
    unit = protocol.initial_state[control_var].units
    protocol_values = [protocol.initial_state[control_var].m] +\
        [seg.end_state[control_var].to(unit).m for seg in protocol.segments]
    tab_protocol = {"Time [s]": protocol_change_points,
                    control_var: protocol_values}
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
        s_dp = np.diff(tab_protocol["Time [s]"][i0:i1+1])
        # ^ duration, protocol, by segment
        tf = np.hstack([np.linspace(s_tf[j], s_tf[j+1], 10)
                        for j in range(len(s_tf)-1)])
        # ^ time, fit, dense
        tp = np.hstack([np.linspace(tab_protocol["Time [s]"][i0+j],
                                    tab_protocol["Time [s]"][i0+j+1],
                                    10)
                        for j in range(len(s_tf)-1)])
        # ↑ time, protocol, dense
        yp = np.interp(tp, tab_protocol["Time [s]"],  # y, protocol, dense
                       tab_protocol[control_var])
        yd = np.interp(tf, data["Time [s]"], data[control_var])  # y, data, dense
        yf = np.interp(tf, s_tf, tab_protocol[control_var][i0:i1+1])  # y, fit, dense
        r = np.corrcoef(yf, yd)[0,1]
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
            msg = "The Pearson correlation coefficient between the provided data and the current protocol fit is undefined. " + reason.format(control_var, tf[0], tf[-1])
            raise(RuntimeError(msg))
        # Add time dilation penalty
        penalty = np.sum((abs(s_d - s_dp) / s_dp)**3.0) / len(s_dp)
        # stdout.write("r = {:.4f}  penalty = {:.4f}  ".format(r, penalty))
        # print("p = {}".format(p))
        return -r + penalty
    for i in range(len(tab_protocol["Time [s]"])):
        if i == 0:
            p0 = np.hstack([[0], np.diff(time_points[i:i+3])])
        else:
            p0 = np.diff(tab_protocol["Time [s]"][i-1:i+2])
        # print("\ni = {}".format(i))
        bounds = [(0, np.inf) for x in p0]
        result = minimize(f, p0, method="L-BFGS-B",
                          bounds=bounds)
        time_points[i] = time_points[max([0, i-1])] + result['x'][0]
    labeled = ProtocolData(data, data["Time [s]"], protocol, time_points)
    return labeled


def read_prune(p):
    """Read a file object as a prunetest protocol."""
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
            k, v = parse.read_definition(ln)
            metadata[k] = v
    # Read headers and enclosed data
    sections = parse.parse_sections(lines[i:], offset=i)
    # Parse protocol
    iprotocol = parse.parse_protocol(sections["protocol"])
    # Read protocol
    defaults = {}
    defaults['rates'] = parse.read_default_rates(sections['declarations'].get('default interpolation', []))
    reference_state = parse.read_reference_state(sections['declarations']['initialization'])
    protocol = read_parsed_protocol(iprotocol, reference_state, defaults)
    return protocol


def read_parsed_protocol(elements, reference_state, defaults=None):
    elements = parse.expand_blocks(elements)
    i = 0  # next segment index (0-indexed)
    segments = []
    blocks = []
    phases = []
    active_block = None
    active_phase = None
    for e in elements:
        if e[0] == 'segment':
            if e[1]['channel'] not in ['Δ.t', 't'] and 'rate' not in e[1]:
                e[1]['rate'] = defaults['rates'][e[1]['channel']]
            segments.append(e)
            if active_block is not None:
                active_block[1].append(i)
            if active_phase is not None:
                active_phase[1].append(i)
            i += 1
        elif e[0] == 'block':
            active_block = ('block', [], e[1])
        elif e[0] == 'phase':
            if active_phase is not None:
                phases.append(active_phase)
            active_phase = ('phase', [], e[1])
        elif e[0] == 'sep':
            if active_block is not None:
                blocks.append(active_block)
                active_block = None
    ## Add non-terminated blocks and phases
    if active_block is not None:
        blocks.append(active_block)
        active_block = None
    if active_phase is not None:
        phases.append(active_phase)
        active_phase = None
    protocol = Protocol.from_intermediate(segments, blocks, phases,
                                          reference_state)
    return protocol


def segment_from_intermediate(struct, previous=None):
    """Return appropriate segment object for intermediate representation."""
    if struct[1]['channel'] == 'Δ.t':
        return HoldSegment.from_intermediate(struct, previous)
    elif struct[1]['channel'] == 't':
        raise NotImplementedError
    else:
        return LinearSegment.from_intermediate(struct, previous)


class Protocol:

    def __init__(self):
        self.segments = []
        self.blocks = []
        self.phases = []

        ## Dictionaries to access blocks and phases by label
        self.segment_dict = OrderedDict()
        self.block_dict = OrderedDict()
        self.phase_dict = OrderedDict()


    @classmethod
    def from_intermediate(cls, segments, blocks, phases, reference_state):
        """Construct a Protocol object from parsed intermediate representation.

        """
        protocol = cls()
        # Initial state
        protocol.initial_state = InitialState(reference_state)
        # Segments
        protocol.segments = []
        previous = protocol.initial_state
        for s in segments:
            segment = segment_from_intermediate(s, previous=previous)
            protocol.segments.append(segment)
            previous = segment
        # Blocks
        for b in blocks:
            block = Block.from_intermediate(protocol, b)
            protocol.blocks.append(block)
            if block.label != '':
                if block.label in protocol.block_dict:
                    warnings.warn("Block label `{}` is not unique.  It will refer to the last phase with this label.".format(block.label))
                protocol.block_dict[block.label] = block
        # Phases
        for p in phases:
            phase = Phase.from_intermediate(protocol, p)
            protocol.phases.append(phase)
            if phase.label != '':
                if phase.label in protocol.phase_dict:
                    warnings.warn("Phase label `{}` is not unique.  It will refer to the last phase with this label.".format(phase.label))
                protocol.phase_dict[phase.label] = phase

        protocol._init_parentage()

        return protocol

    def _init_parentage(self):
        self.phase_of_segment = {}
        self.phase_of_block = {}
        for phase in self.phases:
            for segment in phase.segments:
                self.phase_of_segment[segment] = phase
        for block in self.blocks:
            phase0 = self.phase_of_segment[block.segments[0]]
            phase1 = self.phase_of_segment[block.segments[-1]]
            assert phase0 == phase1
            self.phase_of_block[block] = phase0


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

        for k in ['segments', 'blocks', 'phases',
                  'segment_dict', 'block_dict', 'phase_dict',
                  'phase_of_block', 'phase_of_segment']:
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
            m = np.logical_and(t0 < self.data_times,
                               self.data_times <= t1)
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


class HoldSegment:

    def __init__(self, channel, previous=None):
        self.duration = 0
        self.previous = previous
        self.end_state = self.previous.end_state.copy()

    @classmethod
    def from_intermediate(cls, struct, previous=None):
        segment = cls(struct[1]['channel'], previous=previous)
        segment.duration = struct[1]['target'].to("s").m
        return segment

    def evaluate(self, t):
        # Return values at end of last segment
        raise NotImplementedError


class LinearSegment:

    def __init__(self, previous=None):
        self.channel = ''
        self.target = None
        self.rate = None
        self.previous = previous

    @classmethod
    def from_intermediate(cls, struct, previous=None):
        segment = cls(previous)
        segment.channel = struct[1]['channel']
        segment.target = struct[1]['target']
        segment.rate = struct[1]['rate']
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
            raise(ValueError("t = {} > {}; t exceeds the segment's duration.".format(t, duration)))
        return initial + t / duration * change


class SegmentList:
    def __init__(self, protocol, segment_ids=None, params=None):
        ## Retain reference to parent protocol
        self.protocol = protocol

        if segment_ids is None:
            self._isegments = []
        else:
            self._isegments = segment_ids

        if params is not None and 'label' in params:
            self.label = params['label']
        else:
            self.label = ''

    def append(self, i):
        self._isegments.append(i)

    @property
    def segments(self):
        return [self.protocol.segments[i]
                for i in self._isegments]


class Block(SegmentList):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_intermediate(cls, protocol, struct):
        block = cls(protocol, struct[1], struct[2])
        block.cycle_length = len(struct[1]) // struct[2]['n']
        return block

    @property
    def cycles(self):
        n = len(self._isegments) // self.cycle_length
        return [[self.protocol.segments[j]
                 for j in self._isegments[2*i:2*i+self.cycle_length]]
                for i in range(n)]


class Cycle(SegmentList):

    def __init__(self, *args):
        super().__init__(*args)


class Phase(SegmentList):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def from_intermediate(cls, protocol, struct):
        phase = cls(protocol, struct[1], struct[2])
        return phase
