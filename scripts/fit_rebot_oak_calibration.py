"""Fit passive OAK/reBot samples with CASCADE's existing markerless solver.

Writes a gated record in the run directory. Never installs a calibration,
opens hardware, or changes the physical profile's safety policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cascade.config import load_demo_config
from cascade.control.kinematics import Kinematics
from cascade.calibration.dataset import save_hand_eye
from cascade.calibration.markerless import (
    DepthSample,
    record_from_markerless,
    solve_markerless,
)
from cascade.calibration.robot_surface import RobotSurface, parse_collision_meshes


def prepare_model(cfg, directory):
    """Reuse local SDK STLs only when they match the checkout's exact LFS OIDs."""
    original = Path(cfg.arm.model)
    package = directory / original.parent.parent.name
    model = package / "urdf" / original.name
    model.parent.mkdir(parents=True, exist_ok=True)
    (package / "meshes").mkdir(exist_ok=True)
    shutil.copy2(original, model)
    records = {}
    for parts in parse_collision_meshes(original).values():
        for _, mesh, _ in parts:
            blob = mesh.read_bytes()
            source = mesh
            if blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
                expected = blob.decode().split("oid sha256:")[1].splitlines()[0]
                source = Path(cfg.arm.sdk_path) / "urdf/RS/meshes/shared" / mesh.name
                blob = source.read_bytes()
                if hashlib.sha256(blob).hexdigest() != expected:
                    raise ValueError(f"Local mesh does not match checkout: {mesh.name}")
            digest = hashlib.sha256(blob).hexdigest()
            target = package / "meshes" / mesh.name
            if (
                not target.exists()
                or hashlib.sha256(target.read_bytes()).hexdigest() != digest
            ):
                target.write_bytes(blob)
            records[mesh.name] = {"source": str(source), "sha256": digest}
    (directory / "verified-meshes.json").write_text(json.dumps(records, indent=2))
    return model


def load_samples(directory, *, model_sha256, serial):
    samples, depth_files = [], []
    reference_K = reference_size = None
    reference_sensor_calibration = None
    have_reference = False
    for path in sorted(directory.glob("pose*/sample.json")):
        meta = json.loads(path.read_text())
        sensor_calibration = meta.get("camera", {}).get("sensor_calibration_sha256")
        if have_reference and sensor_calibration != reference_sensor_calibration:
            raise ValueError("Camera sensor calibration changed between samples")
        reference_sensor_calibration = sensor_calibration
        have_reference = True
        if (
            meta.get("schema") != "cascade.passive-rgbd-calibration/1"
            or meta.get("motion_commands_sent") != 0
        ):
            raise ValueError(f"Unknown sample provenance: {path}")
        if (
            meta["camera"]["serial"] != serial
            or meta["before"]["model_sha256"] != model_sha256
        ):
            raise ValueError(f"Camera/model mismatch: {path}")
        if (
            meta["before"]["joint_signs"] != [-1] * 6
            or meta["after"]["model_sha256"] != model_sha256
        ):
            raise ValueError(f"Model/convention changed during capture: {path}")
        with np.load(path.parent / "sample.npz", allow_pickle=False) as data:
            q, depth, K, T = [
                data[key].copy() for key in ("q", "depth_m", "K", "T_gripper2base")
            ]
        if (
            q.shape != (6,)
            or not np.isfinite(q).all()
            or not np.allclose(q, meta["before"]["q_rad"], atol=1e-9, rtol=0)
        ):
            raise ValueError(f"Invalid/inconsistent encoder pose: {path}")
        after = np.asarray(meta["after"]["q_rad"])
        if (
            after.shape != (6,)
            or not np.isfinite(after).all()
            or np.max(np.abs(q - after)) > 0.005
        ):
            raise ValueError(f"Nonstationary capture: {path}")
        if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError(f"Invalid depth: {path}")
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or not np.allclose(K, meta["K"])
        ):
            raise ValueError(f"Invalid/inconsistent intrinsics: {path}")
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError(f"Invalid FK transform: {path}")
        if reference_K is not None and (
            depth.shape != reference_size or not np.allclose(K, reference_K)
        ):
            raise ValueError("Camera intrinsics/resolution changed between samples")
        reference_K, reference_size = K, depth.shape
        samples.append(DepthSample(tuple(q), T, depth, K, label=path.parent.name))
        depth_files.append(str((path.parent / "sample.npz").absolute()))
    if not samples:
        raise ValueError("No pose*/sample.json captures found")
    return samples, depth_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--initial",
        type=Path,
        required=True,
        help="Candidate JSON containing T_cam2base; only an initialization",
    )
    args = parser.parse_args()
    cfg = load_demo_config(
        arm="rebot_rs", camera="oak_d", llm="mock", _ignore_robot_environment=True
    )
    model_hash = hashlib.sha256(Path(cfg.arm.model).read_bytes()).hexdigest()
    samples, files = load_samples(
        args.samples, model_sha256=model_hash, serial=cfg.camera.serial
    )
    model = prepare_model(cfg, args.samples / "model")
    kin = Kinematics(
        str(model), cfg.arm.ee_frame, n_controlled=6, joint_signs=cfg.arm.joint_signs
    )
    for s in samples:
        if not np.allclose(
            kin.fk(np.asarray(s.q)), s.T_gripper2base, atol=1e-8, rtol=0
        ):
            raise ValueError(
                "Stored FK does not match the measured joints and signed model"
            )
    T = np.asarray(json.loads(args.initial.read_text())["T_cam2base"], dtype=float)
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1)
    ):
        raise ValueError("Initial candidate is not a rigid transform")
    surface = RobotSurface.from_kinematics(kin, str(model))
    fit = solve_markerless(samples, surface, T)
    record = record_from_markerless(
        fit,
        samples,
        camera="oak_d",
        camera_serial=cfg.camera.serial,
        arm="rebot_rs",
        ee_frame=cfg.arm.ee_frame,
        K=samples[-1].K,
        image_size=(samples[-1].depth_m.shape[1], samples[-1].depth_m.shape[0]),
        depth_files=files,
        note=f"Passive operator-supported poses; no motor commands. Model sha256 {model_hash}. Camera/base/encoder zeros must remain unchanged. Not installed by this command.",
    )
    destination = args.samples / "candidate.handeye.json"
    save_hand_eye(destination, record)
    print(fit.summary())
    print(
        json.dumps(
            {
                "record": str(destination.absolute()),
                "acceptable": record.acceptable,
                "rejection_reasons": record.rejection_reasons,
                "installed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
