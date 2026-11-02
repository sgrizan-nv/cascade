"""Passive feedback must never use actuator commands or trust bad readings."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from cascade.control.rebot_observer import (
    FeedbackUnavailable,
    read_snapshot,
    validate_snapshot,
)


LIMITS = (
    np.array([-3.0, 0.0, 0.0, -2.0, -2.0, -3.2]),
    np.array([3.0, 3.0, 3.0, 2.0, 2.0, 3.2]),
)


class ReadOnlyController:
    """No enable, disable, stop, send or mode-changing methods exist."""

    def __init__(self, channel, values=None):
        self.channel = channel
        self.values = values or {i: iter([0.0, 0.0]) for i in range(1, 8)}
        self.registered = []
        self.closed_bus = self.closed = False

    def add_robstride_motor(self, *, motor_id, feedback_id, model):
        self.registered.append((motor_id, feedback_id, model))
        controller = self

        class Motor:
            def robstride_get_param_f32(self, index, timeout):
                assert index == 0x7019
                assert timeout == 150
                return next(controller.values[motor_id])

        return Motor()

    def close_bus(self):
        self.closed_bus = True

    def close(self):
        self.closed = True


@pytest.fixture
def sdk_config(tmp_path):
    path = tmp_path / "sdk.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "channel": "can0",
                "joints": [
                    {
                        "motor_id": i,
                        "model": "rs-06" if i <= 3 else "rs-00",
                        "vendor": "robstride",
                        "feedback_id": 253,
                    }
                    for i in range(1, 8)
                ],
            }
        )
    )
    return path


def test_only_parameter_reads_and_host_cleanup(sdk_config):
    ctrl = ReadOnlyController("can0")
    record = read_snapshot(sdk_config, LIMITS, controller_factory=lambda **_: ctrl)
    assert ctrl.closed_bus and ctrl.closed
    assert [m[2] for m in ctrl.registered] == ["rs-06"] * 3 + ["rs-00"] * 4
    assert record["atomic"] is False
    assert record["motion_commands_sent"] == 0
    assert "unknown" in record["torque_state"]
    assert np.array_equal(validate_snapshot(record, LIMITS), np.zeros(6))
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("unavailable", [False, True])
def test_settings_use_typed_reads_and_preserve_errors(sdk_config, unavailable):
    from cascade.control.rebot_observer import SETTINGS_REGISTERS, DIAGNOSTIC_REGISTERS
    calls = []

    class Controller(ReadOnlyController):
        def add_robstride_motor(self, **kwargs):
            position_reader = super().add_robstride_motor(**kwargs)
            mid = kwargs["motor_id"]

            class Motor:
                def read(self, index, timeout, kind):
                    calls.append((mid, index, kind))
                    if index == 0x7019:
                        return position_reader.robstride_get_param_f32(index, timeout)
                    if unavailable and mid == 3 and index == 0x7018:
                        raise RuntimeError("parameter timeout")
                    return 24. if kind == "f32" else 0

                def robstride_get_param_f32(self, index, timeout):
                    return self.read(index, timeout, "f32")

                def robstride_get_param_i8(self, index, timeout):
                    return self.read(index, timeout, "i8")

                def robstride_get_param_u8(self, index, timeout):
                    return self.read(index, timeout, "u8")

                def robstride_get_param_u32(self, index, timeout):
                    return self.read(index, timeout, "u32")

            return Motor()

    ctrl = Controller("can0", values={i: iter([0.0] * 3) for i in range(1, 8)})
    record = read_snapshot(sdk_config, LIMITS, controller_factory=lambda **_: ctrl,
                           include_settings=True, include_diagnostics=True)
    assert ctrl.closed_bus and ctrl.closed
    assert record["settings_complete"] is not unavailable
    assert len(calls) == 14 + 7 * (len(SETTINGS_REGISTERS) + len(DIAGNOSTIC_REGISTERS))
    assert record["diagnostics"][2]["parameters"]["commanded_position_raw"]["index"] == "0x3012"
    assert (3, 0x7005, "i8") in calls and (3, 0x7028, "u32") in calls
    params = record["settings"][2]["parameters"]
    assert params["mechanical_velocity_rad_s"]["index"] == "0x701B"
    assert params["filtered_iq_a"]["index"] == "0x701A"
    if unavailable:
        assert "value" not in params["current_limit_a"]
        assert "parameter timeout" in params["current_limit_a"]["error"]
    assert record["motion_commands_sent"] == 0
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), 2.3e18, -0.5])
def test_bad_read_never_becomes_pose_and_still_releases_bus(sdk_config, bad):
    values = {i: iter([0.0, 0.0]) for i in range(1, 8)}
    values[2] = iter([bad])
    ctrl = ReadOnlyController("can0", values)
    with pytest.raises(FeedbackUnavailable):
        read_snapshot(sdk_config, LIMITS, controller_factory=lambda **_: ctrl)
    assert ctrl.closed_bus and ctrl.closed


def test_moving_snapshot_refused(sdk_config):
    values = {i: iter([0.0, 0.0]) for i in range(1, 8)}
    values[1] = iter([0.0, 0.02])
    ctrl = ReadOnlyController("can0", values)
    with pytest.raises(FeedbackUnavailable, match="changed pose"):
        read_snapshot(sdk_config, LIMITS, controller_factory=lambda **_: ctrl)


def test_snapshot_freshness_completeness_and_time_order(sdk_config):
    record = read_snapshot(sdk_config, LIMITS, controller_factory=ReadOnlyController)
    with pytest.raises(FeedbackUnavailable, match="Stale"):
        validate_snapshot(
            record, LIMITS, now=record["motors"][-1]["read_completed"] + 2
        )
    missing = copy.deepcopy(record)
    missing["motors"].pop()
    with pytest.raises(FeedbackUnavailable, match="Incomplete"):
        validate_snapshot(missing, LIMITS)
    future = copy.deepcopy(record)
    future["motors"][0]["read_started"] += 10
    with pytest.raises(FeedbackUnavailable):
        validate_snapshot(future, LIMITS)


def test_bad_sdk_identity_refused_before_open(sdk_config):
    data = yaml.safe_load(sdk_config.read_text())
    data["joints"][0]["model"] = "rs-unknown"
    sdk_config.write_text(yaml.safe_dump(data))

    def forbidden(**kw):
        pytest.fail("invalid config opened CAN")

    with pytest.raises(FeedbackUnavailable):
        read_snapshot(sdk_config, LIMITS, controller_factory=forbidden)


def test_helper_preserves_venv_path_and_excludes_unmeasured_fingers(
    tmp_path, monkeypatch, sdk_config
):
    import types
    from cascade.config import Cfg
    from cascade.control import kinematics
    from cascade.control import rebot_observer as module

    model = tmp_path / "model.urdf"
    model.write_text("<robot/>")
    target = tmp_path / "system-python"
    target.write_text("")
    python = tmp_path / "venv-python"
    python.symlink_to(target)
    fake = types.SimpleNamespace(
        n=6,
        joint_limits=LIMITS,
        fk=lambda q: np.eye(4),
        link_positions=lambda q: np.zeros((8, 3)),
        model=types.SimpleNamespace(
            names=["universe"] + [f"joint{i}" for i in range(1, 9)]
        ),
    )
    monkeypatch.setattr(kinematics, "Kinematics", lambda *a, **k: fake)
    cfg = Cfg(
        {
            "type": "rebot_rs",
            "n_joints": 6,
            "joint_signs": [-1] * 6,
            "model": str(model),
            "sdk_path": str(tmp_path),
            "hw_yaml": "sdk.yaml",
        }
    )
    observer = module.PassiveRebotObserver(cfg, python)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        r = read_snapshot(sdk_config, LIMITS, controller_factory=ReadOnlyController)
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(r))

    monkeypatch.setattr(module.subprocess, "run", run)
    result = observer.observe()
    assert calls[0][0] == str(python)  # preserve venv, not resolved interpreter
    assert len(result["link_positions_base_m"]) == 6
    assert len(result["link_position_joint_names"]) == 6
    assert result["gripper_width_m"] is None
    assert result["camera_synchronized"] is False
