"""A delayed depth image must not be masked at the newest moving target."""
import numpy as np
import pytest

from cascade.apps.demo import _MeshArmState
from cascade.control.lazy_arm import LazyArm
from cascade.control import rebot_rs_arm as rs
from cascade.perception.robot_mask import MeshBodyMask
from cascade.sensing.alignment import SampleBuffer
from test_rebot_stream_body_mask import connected_arm
from test_occupancy_mesh_body_mask import (
    ARM_BOXES, K, SIDE_CAM, _Bridge, _frame, _occupancy, _state,
    render, synth_geometry, synth_poses,
)


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rs.time, "monotonic", lambda: now[0])
    return now


def test_delayed_frame_uses_historical_target_without_reading_can(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    old = np.full(6, .2)
    arm.send_joint_target(old)
    old[:] = 99  # history must own its values
    clock[0] += .1
    arm.send_joint_target(np.full(6, .3))
    state = arm.body_mask_state_at(100.01)
    np.testing.assert_allclose(state.q, .2)
    assert state.t == 100.0 and not state.gripper_valid
    assert reads == []
    # The older capture is still older even after stop; don't use live feedback.
    arm.stop()
    np.testing.assert_allclose(arm.body_mask_state_at(100.01).q, .2)
    assert reads == []


def test_missing_stream_history_refuses_instead_of_using_latest(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    arm.send_joint_target(np.ones(6))
    with pytest.raises(RuntimeError, match="No streaming target"):
        arm.body_mask_state_at(clock[0] - .2)
    assert reads == []
    clock[0] += .5
    np.testing.assert_allclose(arm.body_mask_state_at(clock[0]).q,
                               [.1, .2, .3, .4, .5, .6])
    assert len(reads) == 1  # idle keeps the standard measured state path


def test_pre_start_image_uses_the_hold_that_was_active_at_capture(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    arm.send_joint_target(np.full(6, .2))
    clock[0] = 120.0  # twenty seconds of stationary hold, no repeated sends
    arm.send_joint_target(np.full(6, .2001))
    state = arm.body_mask_state_at(120.0 - .14243427501060069)
    np.testing.assert_allclose(state.q, .2)
    assert state.t == 100.0
    assert reads == []


def test_post_abort_image_uses_the_target_still_being_held(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    arm.send_joint_target(np.full(6, .2))
    clock[0] = 100.02
    arm.send_joint_target(np.full(6, .201))
    arm.stop()
    clock[0] = 100.3
    state = arm.body_mask_state_at(100.02 + .0972115510085132)
    np.testing.assert_allclose(state.q, .201)
    assert state.t == 100.02
    assert reads == []


def test_command_history_never_uses_a_future_or_evicted_target():
    history = SampleBuffer(2)
    history.add(1, "evicted")
    history.add(2, "held")
    history.add(3, "future")
    assert history.at_or_before(1.5) is None
    assert history.at_or_before(2.99) == ("held", 2)
    assert history.at_or_before(3) == ("future", 3)
    assert history.at_or_before(float("nan")) is None


def test_frame_between_targets_uses_preceding_even_if_next_is_nearer(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    arm.send_joint_target(np.zeros(6))
    clock[0] = 100.02
    arm.send_joint_target(np.ones(6))
    np.testing.assert_allclose(arm.body_mask_state_at(100.019).q, 0)
    assert reads == []


def test_timed_passive_reader_never_wakes_standby_or_disconnected_arm():
    lazy = LazyArm(lambda: pytest.fail("actuator was constructed"))
    source = _MeshArmState()
    source.arm = lazy
    assert source.at_time(100) is None
    assert lazy.body_mask_state_at(100) is None


@pytest.mark.parametrize("stride", [1, 2])
def test_capture_time_mask_removes_moving_robot_but_preserves_object(stride):
    old, current = _state(q=(0, 0)), _state(q=(.4, .3))
    depth, labels = render(old, SIDE_CAM)
    frame = _frame(depth)
    frame.t = 100.2  # delivery time differs from exposure
    frame.observed_t = 100.0
    class Source:
        def __call__(self):
            return current
        def at_time(self, t):
            assert t == 100.0
            return old
    mask = MeshBodyMask(synth_geometry(), synth_poses, Source())
    robot = np.isin(labels, list(ARM_BOXES))
    # Pin the premise: using the newest pose really does leak robot pixels.
    assert (robot & ~mask(depth, K, SIDE_CAM)).any()
    bridge = _Bridge()
    occ = _occupancy(bridge, stride)
    occ.add_robot_body_mask(mask)
    occ.refresh(frame, SIDE_CAM)
    assert occ.last_error is None
    sent = bridge.sent("integrate_depth")[-1]["depth"]
    lab = labels[::stride, ::stride]
    assert (sent[np.isin(lab, list(ARM_BOXES))] == 0).all()
    np.testing.assert_allclose(sent[lab == "object"], depth[::stride, ::stride][lab == "object"])


def test_missing_timed_pose_latches_map_and_integrates_nothing():
    depth, _ = render(_state(), SIDE_CAM)
    class Source:
        def __call__(self):
            pytest.fail("must not fall back to current pose")
        def at_time(self, t):
            raise RuntimeError("capture pose unavailable")
    bridge = _Bridge()
    occ = _occupancy(bridge)
    occ.add_robot_body_mask(MeshBodyMask(synth_geometry(), synth_poses, Source()))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert "capture pose unavailable" in occ.last_error
    assert occ._body_error
    assert not bridge.sent("integrate_depth")
