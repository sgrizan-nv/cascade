"""Collect one stationary RGB-D/encoder sample without an arm control backend.

The operator supports the disabled arm. This command neither guides it nor
changes torque. A sample is calibration evidence, never a usable extrinsic by
itself. Keep the camera and robot base fixed across samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cascade.config import load_demo_config
from cascade.control.rebot_observer import PassiveRebotObserver
from cascade.perception.camera_base import make_camera


def validate_static_interval(before, after, frames, tolerance=0.005):
    if not frames or before["model_sha256"] != after["model_sha256"]:
        raise ValueError("Missing frames or changed robot model")
    delta = np.abs(np.asarray(after["q_rad"]) - before["q_rad"])
    if delta.shape != (6,) or not np.isfinite(delta).all() or np.max(delta) > tolerance:
        raise ValueError("Arm changed pose during RGB-D capture; sample refused")
    start = before["motors"][-1]["read_completed"]
    end = after["motors"][0]["read_started"]
    last_t = start
    for f in frames:
        capture_t = f.observed_t
        if capture_t is None or not last_t < capture_t <= f.t <= end:
            raise ValueError("Frames are not bracketed by the encoder observations")
        if f.depth_source != "sensor" or f.depth_m is None:
            raise ValueError("Calibration requires sensor depth")
        if f.size != frames[0].size or not np.allclose(f.K, frames[0].K):
            raise ValueError("Camera geometry changed during capture")
        last_t = capture_t
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing samples are never overwritten",
    )
    parser.add_argument("--feedback-python", required=True)
    parser.add_argument(
        "--sensor-calibration", type=Path,
        help="Optional DepthAI calibration JSON, applied in camera RAM only",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; use a new sample directory")
    cfg = load_demo_config(
        camera="oak_d", arm="rebot_rs", llm="mock", _ignore_robot_environment=True
    )
    if args.sensor_calibration:
        cfg.camera._data["sensor_calibration_json"] = str(args.sensor_calibration.resolve())
    observer = PassiveRebotObserver(cfg.arm, args.feedback_python)
    camera = make_camera(cfg.camera)
    with camera:
        camera.warm_up(int(cfg.camera.warmup_frames))
        before = observer.observe()
        frames = []
        deadline = time.monotonic() + 5
        while len(frames) < 5 and time.monotonic() < deadline:
            frame = camera.get_frame()
            if frame.observed_t > before["motors"][-1]["read_completed"]:
                frames.append(frame)
        after = observer.observe()
        delta = validate_static_interval(before, after, frames)
        if len(frames) != 5:
            raise ValueError("Not enough fresh frames; sample refused")
        stack = np.stack([f.depth_m for f in frames])
        # No synthesized surface in missing pixels. Retain raw frames as evidence.
        supported = np.sum(stack > 0, axis=0) >= 3
        depth = np.zeros(stack.shape[1:], np.float32)
        depth[supported] = np.nanmedian(
            np.where(stack[:, supported] > 0, stack[:, supported], np.nan), axis=0
        )
        report = {
            "schema": "cascade.passive-rgbd-calibration/1",
            "mode": "eye_to_hand",
            "accepted_extrinsic": False,
            "motion_commands_sent": 0,
            "created_at_unix_s": time.time(),
            "camera": camera.device_info,
            "before": before,
            "after": after,
            "max_joint_change_rad": float(delta.max()),
            "image_size": list(frames[-1].size),
            "K": frames[-1].K.tolist(),
            "frames": [
                {"t": f.t, "observed_t": f.observed_t, "capture": f.capture}
                for f in frames
            ],
            "depth_coverage": float((depth > 0).mean()),
            "limitations": "Sequential encoder reads bracket a stationary sample; not hardware synchronized. Multiple diverse poses and accepted fit still required.",
        }
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            args.output / "sample.npz",
            depth_m=depth,
            raw_depth_m=stack,
            rgb=frames[-1].rgb,
            K=frames[-1].K,
            q=np.asarray(before["q_rad"]),
            T_gripper2base=np.asarray(before["tcp_transform_base"]),
        )
        cv2.imwrite(str(args.output / "rgb.jpg"), frames[-1].rgb)
        (args.output / "sample.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.absolute()),
                    "max_joint_change_rad": float(delta.max()),
                    "depth_coverage": report["depth_coverage"],
                    "accepted_extrinsic": False,
                    "motion_commands_sent": 0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
