"""Read-only RobStride parameter snapshots, never a motor-control backend.

This reader cannot enable, stop, zero, change mode or send targets. Closing
it releases the host handle without changing whatever torque state exists.
Snapshots are sequential reads, not atomic encoder/camera observations.
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from .robstride import plausible_position


class FeedbackUnavailable(RuntimeError):
    pass


# Typed read-only subset of MotorBridge v0.4.9's RobStride ParameterId table:
# motor_vendors/robstride/src/registers.rs. The Python package exposes typed
# reads but no RobStride register catalog (its catalog is Damiao-only).
SETTINGS_REGISTERS = (
    ("run_mode", 0x7005, "i8"),
    ("torque_limit_nm", 0x700B, "f32"),
    ("current_limit_a", 0x7018, "f32"),
    ("velocity_limit_rad_s", 0x7017, "f32"),
    ("position_kp", 0x701E, "f32"),
    ("velocity_kp", 0x701F, "f32"),
    ("velocity_ki", 0x7020, "f32"),
    ("filtered_iq_a", 0x701A, "f32"),
    ("mechanical_velocity_rad_s", 0x701B, "f32"),
    ("bus_voltage_v", 0x701C, "f32"),
    ("can_timeout", 0x7028, "u32"),
    ("zero_state", 0x7029, "u8"),
)

DIAGNOSTIC_REGISTERS = (
    ("position_rad", 0x7019, "f32"),
    ("filtered_iq_a", 0x701A, "f32"),
    ("velocity_rad_s", 0x701B, "f32"),
    ("commanded_torque_nm", 0x3011, "f32"),
    ("measured_torque_nm", 0x302C, "f32"),
    ("run_mode", 0x7005, "i8"),
    ("bus_voltage_v", 0x701C, "f32"),
    ("commanded_position_raw", 0x3012, "f32"),
    ("commanded_velocity_raw", 0x3013, "f32"),
    ("commanded_iq_raw", 0x300E, "f32"),
    ("fault_flags_raw", 0x3022, "u32"),
    ("warning_flags_raw", 0x3023, "u32"),
)


def read_parameters(motor, registers, timeout_ms=150, lock=None):
    """Typed parameter reads on an existing handle; no controller ownership.

    A connected driver may share its lock, held for one bounded read only.
    Diagnostic registers are firmware readbacks, not independent torque sensors.
    """
    result = {}
    for name, index, kind in registers:
        reading = {"index": f"0x{index:04X}", "type": kind}
        try:
            with lock if lock is not None else nullcontext():
                reading["read_started"] = time.monotonic()
                value = getattr(motor, f"robstride_get_param_{kind}")(index, timeout_ms)
                reading["read_completed"] = time.monotonic()
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise FeedbackUnavailable("Nonfinite or nonnumeric parameter")
            reading["value"] = value
        except Exception as exc:
            reading["error"] = f"{type(exc).__name__}: {exc}"
            reading["read_completed"] = time.monotonic()
        result[name] = reading
    return result


def read_snapshot(config_path, limits, *, controller_factory=None, timeout_ms=150,
                  include_settings=False, include_diagnostics=False):
    """Read all seven positions twice; reject bad, incomplete or moving data.

    The second sweep is returned. Its interval and the difference between
    sweeps are retained. The 10 mrad stability bound only admits a stationary
    observation; it is NOT a motion-tracking tolerance.
    """
    import yaml

    data = yaml.safe_load(Path(config_path).read_text())
    joints = data["joints"]
    if len(joints) != 7 or {j["motor_id"] for j in joints} != set(range(1, 8)):
        raise FeedbackUnavailable("SDK config must identify exactly motors 1 through 7")
    joints = sorted(joints, key=lambda j: j["motor_id"])
    if any(
        j.get("vendor") != "robstride"
        or j.get("model") not in {"rs-00", "rs-06"}
        or j.get("feedback_id") != 0xFD
        for j in joints
    ):
        raise FeedbackUnavailable("Unsupported motor identity or feedback id")
    lo, hi = (np.asarray(v, dtype=float) for v in limits)
    if (
        lo.shape != (6,)
        or hi.shape != (6,)
        or not np.isfinite([lo, hi]).all()
        or np.any(lo >= hi)
    ):
        raise FeedbackUnavailable("Six finite ordered URDF joint limits are required")
    if controller_factory is None:
        from motorbridge import Controller

        controller_factory = Controller
    controller = controller_factory(channel=str(data["channel"]))
    try:
        motors = [
            controller.add_robstride_motor(
                motor_id=j["motor_id"], feedback_id=j["feedback_id"], model=j["model"]
            )
            for j in joints
        ]
        sweeps = []
        for _ in range(2):
            sweep = []
            for i, motor in enumerate(motors):
                before = time.monotonic()
                value = motor.robstride_get_param_f32(0x7019, timeout_ms)
                after = time.monotonic()
                if not plausible_position(
                    value, lo[i] if i < 6 else None, hi[i] if i < 6 else None
                ):
                    raise FeedbackUnavailable(f"Invalid position from motor {i + 1}")
                sweep.append(
                    {
                        "id": i + 1,
                        "position_rad": float(value),
                        "read_started": before,
                        "read_completed": after,
                    }
                )
            sweeps.append(sweep)
        drift = [
            abs(a["position_rad"] - b["position_rad"])
            for a, b in zip(*sweeps, strict=True)
        ]
        if max(drift) > 0.01:
            raise FeedbackUnavailable(
                "Arm changed pose between reads; stationary snapshot refused"
            )
        result = {
            "schema": 1,
            "source": "robstride_mechPos_parameter_read",
            "channel": str(data["channel"]),
            "motors": sweeps[1],
            "models": [j["model"] for j in joints],
            "repeat_delta_rad": drift,
            "atomic": False,
            "torque_state": "unknown; no enable/disable request sent",
            "motion_commands_sent": 0,
        }
        if include_settings:
            result["settings_complete"] = True
            result["settings"] = []
            for joint, motor in zip(joints, motors, strict=True):
                params = read_parameters(motor, SETTINGS_REGISTERS, timeout_ms)
                row = {"id": joint["motor_id"], "model": joint["model"], "parameters": params}
                if any("error" in p for p in params.values()):
                    result["settings_complete"] = False
                result["settings"].append(row)
        if include_diagnostics:
            result["diagnostics"] = [
                {"id": joint["motor_id"], "model": joint["model"],
                 "parameters": read_parameters(motor, DIAGNOSTIC_REGISTERS, timeout_ms)}
                for joint, motor in zip(joints, motors, strict=True)
            ]
            result["diagnostics_note"] = (
                "Firmware register readbacks, not command acknowledgements. "
                "Unsupported registers may return zero; raw values need live-response validation.")
        return result
    finally:
        try:
            controller.close_bus()
        finally:
            controller.close()


def validate_snapshot(record, limits, *, now=None, max_age_s=1.0):
    """Reject stale/forged-shape/implausible records before FK consumes them."""
    now = time.monotonic() if now is None else now
    if (
        record.get("schema") != 1
        or record.get("source") != "robstride_mechPos_parameter_read"
    ):
        raise FeedbackUnavailable("Unknown encoder snapshot schema/source")
    motors = record.get("motors", [])
    if len(motors) != 7 or [m.get("id") for m in motors] != list(range(1, 8)):
        raise FeedbackUnavailable("Incomplete or unordered encoder snapshot")
    lo, hi = limits
    previous_end = -math.inf
    for i, motor in enumerate(motors):
        value = motor.get("position_rad")
        if not plausible_position(
            value, lo[i] if i < 6 else None, hi[i] if i < 6 else None
        ):
            raise FeedbackUnavailable(f"Invalid position from motor {i + 1}")
        start, end = motor.get("read_started"), motor.get("read_completed")
        if any(
            isinstance(t, bool)
            or not isinstance(t, (float, int))
            or not math.isfinite(t)
            for t in (start, end)
        ):
            raise FeedbackUnavailable("Missing/nonfinite feedback timestamps")
        if not previous_end <= start <= end <= now or now - start > max_age_s:
            raise FeedbackUnavailable("Stale, reversed or future feedback timestamps")
        previous_end = end
    return np.array([m["position_rad"] for m in motors[:6]], dtype=float)


class PassiveRebotObserver:
    """Short-lived helper interpreter owns CAN reads; this process computes FK.

    The helper supports a separate existing motorbridge environment without
    installing it into the main runtime. No shell, caches or fallback poses.
    """

    def __init__(self, arm_cfg, python, *, helper=None):
        from .kinematics import Kinematics

        if (
            arm_cfg.get("type") != "rebot_rs"
            or arm_cfg.get("n_joints") != 6
            or list(arm_cfg.joint_signs) != [-1] * 6
        ):
            raise ValueError(
                "Passive reader requires the signed six-joint rebot_rs profile"
            )
        # Resolving a venv's python symlink loses its pyvenv.cfg and packages.
        self.python = str(Path(python).expanduser().absolute())
        self.helper = (
            Path(helper)
            if helper
            else Path(__file__).resolve().parents[3] / "scripts/read_rebot_state.py"
        )
        self.config_path = Path(arm_cfg.sdk_path) / "config" / arm_cfg.hw_yaml
        self.kin = Kinematics(
            arm_cfg.model,
            arm_cfg.get("ee_frame", "gripper_end"),
            n_controlled=6,
            joint_signs=arm_cfg.joint_signs,
        )
        self.model_sha256 = hashlib.sha256(Path(arm_cfg.model).read_bytes()).hexdigest()
        self.model_path = str(arm_cfg.model)

    def observe(self, frame_t=None):
        limits = self.kin.joint_limits
        command = [
            self.python,
            str(self.helper),
            "--sdk-config",
            str(self.config_path),
            "--limits-json",
            json.dumps([v.tolist() for v in limits]),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=8, check=False
        )
        if completed.returncode:
            raise FeedbackUnavailable(
                "Read-only encoder helper failed: "
                + (completed.stdout or completed.stderr).strip()[-300:]
            )
        try:
            record = json.loads(completed.stdout)
        except ValueError as exc:
            raise FeedbackUnavailable("Invalid encoder helper response") from exc
        q = validate_snapshot(record, limits)
        transform = self.kin.fk(q)
        # The remaining model joints are passive finger slides. Their padded
        # zero pose is NOT measured by these six arm encoders.
        links = self.kin.link_positions(q)[: self.kin.n]
        return {
            **record,
            "q_rad": q.tolist(),
            "tcp_transform_base": transform.tolist(),
            "link_positions_base_m": links.tolist(),
            "gripper_motor_rad": record["motors"][6]["position_rad"],
            "link_position_joint_names": list(self.kin.model.names)[1 : self.kin.n + 1],
            "gripper_width_m": None,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "joint_signs": [-1] * 6,
            "camera_frame_t": frame_t,
            "camera_synchronized": False,
            "camera_to_feedback_offset_s": None
            if frame_t is None
            else record["motors"][0]["read_started"] - frame_t,
            "note": "Measured encoders; FK is model-derived. Sequential reads are not camera-synchronized. "
            "Motor angle is not a measured jaw width. This grants no motion authority.",
        }
