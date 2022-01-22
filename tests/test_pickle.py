"""Test interoperability with other libraries"""
import dill
import pickle
import pytest

from prunetest import Q, Unit


@pytest.mark.parametrize("version", [0, 1, 2])
def test_pickle_roundtrip_quantity(version):
    q1 = Q(1, "mm")
    assert isinstance(q1, Q)
    q2 = pickle.loads(pickle.dumps(q1, protocol=version))
    assert isinstance(q2, Q)


def test_dill_roundtrip_quantity():
    q1 = Q(1, "mm")
    assert isinstance(q1, Q)
    q2 = dill.loads(dill.dumps(q1))
    assert isinstance(q2, Q)


@pytest.mark.parametrize("version", [0, 1, 2])
def test_pickle_roundtrip_unit(version):
    u1 = Unit("mm")
    assert isinstance(u1, Unit)
    u2 = pickle.loads(pickle.dumps(u1, protocol=version))
    assert isinstance(u2, Unit)


@pytest.mark.parametrize("version", [0, 1, 2])
def test_dill_roundtrip_unit(version):
    u1 = Unit("mm")
    assert isinstance(u1, Unit)
    u2 = dill.loads(dill.dumps(u1, protocol=version))
    assert isinstance(u2, Unit)
