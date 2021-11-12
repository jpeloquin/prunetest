from math import isclose

from prunetest import Protocol, Segment, Quantity
from prunetest.protocol import (
    AbsoluteTarget,
    BinOp,
    RelativeTarget,
    SymbolicValue,
    Transition,
)

initial_state = {"t": Quantity(0, "s"), "εr": Quantity(0), "εz": Quantity(0)}
parameters = {"ε_swell": Quantity(0.09)}
segments = [
    Segment(
        [
            Transition("t", AbsoluteTarget(Quantity(10, "s"))),
            Transition(
                "εz",
                AbsoluteTarget(BinOp("*", Quantity(0.9), SymbolicValue("ε_swell"))),
            ),
        ]
    ),
    Segment(
        [
            Transition("t", RelativeTarget(Quantity(90, "s"))),
            Transition("εz", RelativeTarget(Quantity(0))),
        ]
    ),
]
protocol = Protocol(initial_state, segments, parameters)


def test_eval_state_initial_state():
    state0 = protocol.eval_state("t", Quantity(0, "s"))
    assert state0["t"] == initial_state["t"]
    assert state0["εz"] == initial_state["εz"]
    assert state0["εr"] == initial_state["εr"]


def test_eval_state_absolute_targets():
    # Test first segment
    state1 = protocol.eval_state("t", Quantity(0.5, "s"))
    assert state1["t"] == Quantity(0.5, "s")
    assert isclose(state1["εz"].m, 0.00405)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None
    state2 = protocol.eval_state("t", Quantity(10, "s"))
    assert state2["t"] == Quantity(10, "s")
    assert isclose(state2["εz"].m, 0.081)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None


def test_eval_state_relative_targets():
    state = protocol.eval_state("t", Quantity(51, "s"))
    assert state["t"] == Quantity(51, "s")
    assert isclose(state["εz"].m, 0.081)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None
