from math import isclose
from pathlib import Path

import pytest

from prunetest import Protocol, Segment, Q, read_prune
from prunetest.protocol import (
    BinOp,
    Parameter,
    RelativeTarget,
    State,
    SymbolicValue,
    Transition,
    Variable,
)

DIR_THIS = Path(__file__).parent
DIR_FIXTURES = DIR_THIS / Path("fixtures")


@pytest.fixture(scope="session")
def segments_protocol():
    ε_swell = Parameter("ε_swell", Q(0.09))
    t = Variable("t", "s")
    εz = Variable("εz", "1")
    εr = Variable("εr", "1")
    initial_state = State({t: Q(0, "s"), εr: Q(0), εz: Q(0)})
    segments = [
        Segment(
            [
                Transition(t, Q(10, "s")),
                Transition(
                    εz,
                    BinOp("*", Q(0.9), SymbolicValue("ε_swell")),
                ),
            ]
        ),
        Segment(
            [
                Transition(t, RelativeTarget(Q(90, "s"))),
                Transition(εz, RelativeTarget(Q(0))),
            ]
        ),
    ]
    protocol = Protocol(
        initial_state, segments, variables=[t, εz, εr], parameters=[ε_swell]
    )
    return protocol


def test_eval_state_initial_state(segments_protocol):
    p = segments_protocol
    state0 = p.eval_state("t", Q(0, "s"))
    assert state0 == p.initial_state


def test_eval_state_absolute_targets(segments_protocol):
    p = segments_protocol
    # Test first segment
    state1 = p.eval_state("t", Q(0.5, "s"))
    assert state1["t"] == Q(0.5, "s")
    assert isclose(state1["εz"].m, 0.00405)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None
    state2 = p.eval_state("t", Q(10, "s"))
    assert state2["t"] == Q(10, "s")
    assert isclose(state2["εz"].m, 0.081)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None


def test_eval_state_relative_targets(segments_protocol):
    state = segments_protocol.eval_state("t", Q(51, "s"))
    assert state["t"] == Q(51, "s")
    assert isclose(state["εz"].m, 0.081)
    # TODO: Fill in uncontrolled variables.  Requires variables list in protocol.
    # assert state1["εr"] is None


def test_read_eval_parameterized_PDL():
    p = read_prune(DIR_FIXTURES / "CEP_ccomp_FEA.prune")
    assert len(p.segments) == 9
    states = p.eval_states("t", [Q(0.5, "s"), Q(1.5, "s")])
    assert states[0]["t"] == Q(0.5, "s")
    assert states[0]["λx"] == Q(1.04)
    assert states[0]["λy"] == Q(1.04)
    assert states[0]["λz"] == Q(1.04)
    assert states[0]["f_FCD"] == Q(0)
    assert states[0]["f_k0"] == Q(0)
    assert states[1]["t"] == Q(1.5, "s")
    assert states[1]["λx"] == Q(1.08)
    assert states[1]["λy"] == Q(1.08)
    assert states[1]["λz"] == Q(1.08)
    assert states[1]["f_FCD"] == Q(0.5)
    assert states[1]["f_k0"] == Q(0)
