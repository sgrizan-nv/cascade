"""The workstation selects upstream control policy, not a tuned pilot."""
from pathlib import Path

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.control.rebot_rs_arm import RebotRSArm
from conftest import JOINT_SIGNS, URDF, needs_pin


def configurations():
    return [load_demo_config(arm=name, camera="oak_d", llm="inferencehub_astra",
                            _ignore_robot_environment=True)
            for name in ("rebot_rs", "rebot_rs_oak")]


def test_oak_inherits_entire_rs_control_policy():
    parent, local = configurations()
    a, b = parent.arm.as_dict(), local.arm.as_dict()
    equipment_keys = {"name", "model", "body_mask", "check_link_attachments", "overrides", "resolved"}
    assert {k: v for k, v in a.items() if k not in equipment_keys} == {
        k: v for k, v in b.items() if k not in equipment_keys}
    assert a["body_mask"] | {"attachments": b["body_mask"]["attachments"]} == b["body_mask"]
    assert parent.occupancy.as_dict() == local.occupancy.as_dict()
    # Only actual installed tool geometry changes in the safety policy.
    pa, pb = parent.safety.as_dict(), local.safety.as_dict()
    equipment_policy = {"tool_volume", "wrist_camera_up"}
    assert {k: v for k, v in pa.items() if k not in equipment_policy} == {
        k: v for k, v in pb.items() if k not in equipment_policy}
    from cascade.safety.harness import SafetyLimits
    assert SafetyLimits.from_config(parent.safety).tool_up_axis == (0., 0., 1.)
    assert SafetyLimits.from_config(local.safety).tool_up_axis is None
    assert local.arm.check_link_attachments is True
    assert local.safety.min_clearance_m == parent.safety.min_clearance_m
    raw_a, raw_b = RebotRSArm(parent.arm), RebotRSArm(local.arm)
    assert raw_a.settle_tol == raw_b.settle_tol == .05
    assert raw_a.settle_timeout_s == raw_b.settle_timeout_s == 2.
    np.testing.assert_array_equal(raw_a._gc_tau_scale, raw_b._gc_tau_scale)
    np.testing.assert_array_equal(raw_b._gc_tau_scale, [1, 1.1, 1.1, 1, 1, 1])
    assert b["mit_kp"] is None and b["mit_kd"] is None
    assert set(b["gravity_comp"]) == {"enabled", "tau_scale"}


def test_retired_commission_command_cannot_open_hardware(monkeypatch):
    from scripts import rebot_oak_workstation as launch
    monkeypatch.setattr(launch, "prepare", lambda: pytest.fail("must reject before preparing"))
    with pytest.raises(SystemExit) as exc:
        launch.main(["commission", "--load-test"])
    assert exc.value.code == 2
    assert not (Path(launch.ROOT) / "configs/arms/rebot_rs_oak_load_test.yaml").exists()


@needs_pin
def test_sonix_mount_does_not_follow_the_parents_tcp_camera_axis():
    from cascade.control.kinematics import Kinematics

    kin = Kinematics(str(URDF), "gripper_end", n_controlled=6, joint_signs=JOINT_SIGNS)
    a = np.array([0., 1.2, 1.2, 0., 0., 0.])
    b = a.copy()
    b[5] = 2.0
    np.testing.assert_allclose(kin.frame_poses(a, ["link4"]), kin.frame_poses(b, ["link4"]))
    assert np.linalg.norm(kin.fk(a)[:3, 2] - kin.fk(b)[:3, 2]) > 1.0


def test_run_uses_normal_cli_without_constructing_another_controller(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from cascade.apps import demo
    from cascade.control import arm_hold
    from scripts import rebot_oak_workstation as launch
    _, cfg = configurations()
    key = tmp_path / "key"
    key.write_text("test-placeholder")
    cfg.camera.detector._data["api_key_file"] = str(key)
    monkeypatch.setattr(launch, "config", lambda: cfg)
    monkeypatch.setattr(arm_hold, "agent_environment", lambda: None)
    monkeypatch.setattr(arm_hold, "_interactive", lambda: True)
    # Ensure monkeypatch restores environment writes made by the launcher.
    monkeypatch.setenv("NV_API_KEY", "before")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "5557")
    monkeypatch.setenv("CASCADE_GRASP_EVIDENCE_DIR", "old-run")
    calls = []

    @contextmanager
    def owned(out):
        yield {"port": 45678}

    monkeypatch.setattr(launch, "bridge", owned)
    monkeypatch.setattr(demo, "main", lambda argv: calls.append(argv) or 0)
    assert launch.run(tmp_path) == 0
    import os
    assert os.environ["CASCADE_GRASP_EVIDENCE_DIR"] == str(tmp_path / "grasp_evidence")
    assert calls == [["--arm", "rebot_rs_oak", "--camera", "oak_d",
                      "--llm", "inferencehub_astra", "--interactive",
                      "--no-view", "--no-serve", "--run-dir", str(tmp_path)]]


def test_run_refuses_agent_before_build_or_bridge(monkeypatch, tmp_path):
    from cascade.control import arm_hold
    from scripts import rebot_oak_workstation as launch
    monkeypatch.setattr(arm_hold, "agent_environment", lambda: "test-agent")
    monkeypatch.setattr(launch, "config", lambda: pytest.fail("must refuse first"))
    with pytest.raises(RuntimeError, match="own workstation terminal"):
        launch.run(tmp_path)


def test_owned_bridge_survives_terminal_signal_until_runtime_teardown(monkeypatch, tmp_path):
    import json
    import os
    from types import SimpleNamespace
    from scripts import rebot_oak_workstation as launch

    read_fd, write_fd = os.pipe()
    ready = {"pid": 123, "instance": "rebot-oak-" + tmp_path.name, "port": 45678}
    os.write(write_fd, ("CASCADE_SERVER_READY " + json.dumps(ready) + "\n").encode())
    os.close(write_fd)
    stream = os.fdopen(read_fd, "rb")
    events = []

    def start(command, **kwargs):
        assert kwargs["start_new_session"] is True
        assert command[1].endswith("serve_occupancy_bridge.py")
        return SimpleNamespace(pid=123, stdout=stream,
            terminate=lambda: events.append("terminate"),
            wait=lambda **kw: events.append("wait"))

    monkeypatch.setattr(launch.subprocess, "Popen", start)
    with launch.bridge(tmp_path) as info:
        assert info == ready
        assert not events
        events.append("runtime_shutdown_complete")
    assert events == ["runtime_shutdown_complete", "terminate", "wait"]
    assert stream.closed
