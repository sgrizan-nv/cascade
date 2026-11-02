"""Perception must never reconnect an arm, including during shutdown."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control.lazy_arm import LazyArm


@pytest.mark.parametrize("method", ["body_mask_state", "body_mask_q", "get_state_if_connected"])
def test_mask_reads_do_not_materialize_standby_or_disconnected_arm(method):
    def forbidden():
        pytest.fail("passive mask read attempted to construct an actuator")

    arm = LazyArm(forbidden)
    assert getattr(arm, method)() is None
    arm._arm = SimpleNamespace(disconnect=lambda: None)
    assert arm.connected
    # Reproduce the window between the caller's connected check and lookup.
    arm.disconnect()
    assert getattr(arm, method)() is None
    assert not arm.connected


@pytest.mark.parametrize("method", ["body_mask_state", "body_mask_q", "get_state_if_connected"])
def test_disconnect_waits_for_inflight_mask_read(method):
    entered, finish, disconnected, disconnect_started = [threading.Event() for _ in range(4)]
    state = SimpleNamespace(q=np.zeros(6))

    def read():
        entered.set()
        assert finish.wait(2)
        assert not disconnected.is_set()
        return state

    arm = LazyArm(lambda: pytest.fail("unexpected reconnect"))
    arm._arm = SimpleNamespace(get_state=read, disconnect=disconnected.set)
    results = []
    reader = threading.Thread(target=lambda: results.append(getattr(arm, method)()))

    def disconnect():
        disconnect_started.set()
        arm.disconnect()

    closer = threading.Thread(target=disconnect)
    reader.start()
    try:
        assert entered.wait(2)
        closer.start()
        assert disconnect_started.wait(2)
        assert not disconnected.wait(.05)
    finally:
        finish.set()
        reader.join(2)
        if closer.ident is not None:
            closer.join(2)
    assert not reader.is_alive() and not closer.is_alive()
    assert disconnected.is_set()
    assert len(results) == 1
    assert getattr(arm, method)() is None


def test_drift_reader_uses_measured_feedback_without_waking_arm():
    from cascade.apps.demo import _measured_q_if_connected

    arm = LazyArm(lambda: pytest.fail("drift reader attempted to connect"))
    assert _measured_q_if_connected(arm) is None
    measured = np.full(6, .01)
    arm._arm = SimpleNamespace(get_state=lambda: SimpleNamespace(q=measured),
                               body_mask_q=lambda: np.ones(6), disconnect=lambda: None)
    np.testing.assert_array_equal(_measured_q_if_connected(arm), measured)
    arm.disconnect()
    assert _measured_q_if_connected(arm) is None
