from math import isclose

from prunetest import Protocol, Segment, Q
from prunetest.protocol import (
    BinOp,
    RelativeTarget,
    SymbolicValue,
    Transition,
)

initial_state = {"t": Q(0, "s"), "εr": Q(0), "εz": Q(0)}
parameters = {"ε_swell": Q(0.09)}
segments = [
    Segment(
        [
            Transition("t", Q(10, "s")),
            Transition(
                "εz",
                BinOp("*", Q(0.9), SymbolicValue("ε_swell")),
            ),
        ]
    ),
    Segment(
        [
            Transition("t", RelativeTarget(Q(90, "s"))),
            Transition("εz", RelativeTarget(Q(0))),
        ]
    ),
]
protocol = Protocol(initial_state, segments, parameters)


def test_eval_state_initial_state():
    state0 = protocol.eval_state("t", Q(0, "s"))
    assert state0["t"] == initial_state["t"]
    assert state0["εz"] == initial_state["εz"]
    assert state0["εr"] == initial_state["εr"]


def test_eval_state_absolute_targets():
    # Test first segment
    state1 = protocol.eval_state("t", Q(0.5, "s"))
    assert state1["t"] == Q(0.5, "s")
    assert isclose(state1["εz"].m, 0.00405)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None
    state2 = protocol.eval_state("t", Q(10, "s"))
    assert state2["t"] == Q(10, "s")
    assert isclose(state2["εz"].m, 0.081)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None


def test_eval_state_relative_targets():
    state = protocol.eval_state("t", Q(51, "s"))
    assert state["t"] == Q(51, "s")
    assert isclose(state["εz"].m, 0.081)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None
