"""Read-only map and route preflight. Contains no actuator construction."""

# Direct execution selects this checkout before package imports.
# ruff: noqa: E402
import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from cascade.config import load_demo_config, Cfg
from cascade.control.rebot_observer import PassiveRebotObserver
from cascade.control import arm_hold
from cascade.control.kinematics import Kinematics
from cascade.safety.trajectory import plan_route
from cascade.perception.camera_base import make_camera
from cascade.perception.grounding import Extrinsics
from cascade.perception.occupancy import OccupancyMap
from cascade.perception.robot_mask import build_mesh_body_mask, BodyMaskUnavailable
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import build_tool_volume
from cascade.safety.link_attachments import build_link_attachments
from cascade.types import RobotState, SafetyViolation, SkillError
from cascade.calibration.markerless import DepthSample, measure_offset
from cascade.calibration.robot_surface import RobotSurface
from cascade.perception.drift_monitor import DriftMonitorConfig
from scripts.fit_rebot_oak_calibration import prepare_model
from scripts.capture_rebot_oak_calibration import validate_static_interval



def check_alignment(frame, q, kin, surface, transform, config):
    """Reuse runtime drift measurement/limits; inconclusive is not readiness."""
    sample = DepthSample(tuple(float(v) for v in q), kin.fk(q),
                         frame.depth_m, frame.K, label="passive_preflight")
    result = measure_offset(sample, surface, transform)
    passed = (result.conclusive and result.offset_m <= config.max_offset_m
              and result.offset_deg <= config.max_rot_deg)
    return {
        "passed": bool(passed), "conclusive": bool(result.conclusive),
        "reason": result.reason if not result.conclusive else (
            "within runtime drift limits" if passed else "outside runtime drift limits"),
        "offset_mm": float(result.offset_m * 1000) if result.conclusive else None,
        "offset_deg": float(result.offset_deg) if result.conclusive else None,
        "max_offset_mm": config.max_offset_m * 1000,
        "max_rot_deg": config.max_rot_deg,
        "n_inliers": result.n_inliers_after,
    }


def startup_hold_blocker(acfg):
    """Report the existing connect gate without clearing it or touching CAN."""
    key = arm_hold.hold_key(acfg.type, acfg)
    marker = arm_hold.read(key) if key is not None else None
    return arm_hold.refusal(marker, key) if marker is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence directory; never overwrites an earlier result",
    )
    parser.add_argument(
        "--feedback-python",
        required=True,
        help="Python containing motorbridge, used for parameter reads only",
    )
    parser.add_argument("--device", default="cuda:0", help="Owned Warp bridge device")
    parser.add_argument("--arm", default="rebot_rs", choices=["rebot_rs", "rebot_rs_oak"])
    parser.add_argument("--map-samples", type=int, default=3,
                        help="Number of fresh maps to check and retain (1–100; no motion)")
    args = parser.parse_args()
    if not 1 <= args.map_samples <= 100:
        parser.error("--map-samples must be between 1 and 100")
    root = ROOT
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_demo_config(
        camera="oak_d", arm=args.arm, llm="mock", _ignore_robot_environment=True
    )
    blocker = startup_hold_blocker(cfg.arm)
    if blocker:
        print(f"STARTUP BLOCKED (map checks may continue): {blocker}", flush=True)
    model = prepare_model(cfg, out / "model")
    acfg = Cfg(cfg.arm.as_dict())
    acfg._data["model"] = str(model)
    # The reference rig's D455F is not installed here. Keep unknown attachments
    # as obstacles. This intentionally masks LESS and cannot authorize motion.
    if args.arm == "rebot_rs":
        acfg.body_mask._data["attachments"] = []
    kin = Kinematics(
        str(model), acfg.ee_frame, n_controlled=6, joint_signs=acfg.joint_signs
    )
    observer = PassiveRebotObserver(cfg.arm, args.feedback_python)
    ext = Extrinsics.from_config(cfg.camera.extrinsics, camera_serial=cfg.camera.serial)
    T = ext.cam_to_base()
    drift_config = DriftMonitorConfig.from_config(cfg.camera.extrinsics.get("drift_monitor"))
    surface = RobotSurface.from_kinematics(kin, str(model)) if drift_config.enabled else None
    slot = [None]

    def state():
        s = slot[0]
        if s is None or time.monotonic() - s.t > 0.5:
            raise BodyMaskUnavailable("Passive mask pose is unavailable/stale")
        return s

    mask = build_mesh_body_mask(acfg, kin, state, name="read-only-rebot")
    occ = None
    proc = None
    log = (out / "bridge.log").open("w")
    try:
        env = dict(os.environ)
        env["XDG_CACHE_HOME"] = str(out / "cache")
        env["PYTHONPATH"] = str(root / "src")
        cmd = [
            sys.executable,
            str(root / "scripts/serve_occupancy_bridge.py"),
            "--port",
            "0",
            "--backend",
            "warp",
            "--device",
            args.device,
            "--instance-id",
            "oak-readonly-preflight",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env
        )
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        startup_deadline = time.monotonic() + 60
        pending = b""
        ready = None
        while time.monotonic() < startup_deadline and ready is None:
            if not selector.select(min(1, max(0, startup_deadline - time.monotonic()))):
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Bridge exited before readiness; see bridge.log")
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.startswith(b"CASCADE_SERVER_READY "):
                    ready = json.loads(line.split(b" ", 1)[1])
                    break
        if ready is None:
            raise RuntimeError("Bridge did not become ready in 60 seconds")
        selector.close()
        print("READY", ready, flush=True)
        ocfg = Cfg(cfg.occupancy.as_dict())
        ocfg._data["port"] = ready["port"]
        occ = OccupancyMap.from_config(
            ocfg,
            workspace_min=cfg.safety.workspace.min,
            workspace_max=cfg.safety.workspace.max,
        )
        probe = occ.probe()
        print("BRIDGE", probe, flush=True)
        if (
            probe.get("pid") != proc.pid
            or probe.get("instance") != "oak-readonly-preflight"
        ):
            raise RuntimeError("Bridge identity mismatch; refusing to modify its map")
        occ.add_robot_body_mask(mask)
        harness = SafetyHarness(
            SafetyLimits.from_config(cfg.safety),
            kinematics=kin,
            occupancy=occ,
            tool_volume=build_tool_volume(cfg, acfg),
            link_attachments=build_link_attachments(acfg, kin),
        )
        camera = make_camera(cfg.camera)
        reports = []
        refusals = []
        with camera:
            camera.warm_up(cfg.camera.warmup_frames)
            # Compile the owned GPU bridge before timing real masked refreshes.
            # This temporary warm-up geometry is cleared, never admitted or reused.
            warm = camera.get_frame()
            occ._client.request(
                {
                    "action": "integrate_depth",
                    "depth": warm.depth_m,
                    "K": warm.K,
                    "T_base_cam": T,
                },
                timeout_ms=30000,
            )
            occ._client.request(
                {
                    "action": "query",
                    "region_min": np.asarray(cfg.safety.workspace.min),
                    "region_max": np.asarray(cfg.safety.workspace.max),
                },
                timeout_ms=30000,
            )
            if not occ.clear_bridge_map():
                raise RuntimeError("Could not clear warm-up geometry")
            print("GPU warm-up complete; temporary geometry cleared", flush=True)
            for i in range(args.map_samples + 5):
                before = observer.observe()
                deadline = time.monotonic() + 3
                while True:
                    f = camera.get_frame()
                    if f.observed_t > before["motors"][-1]["read_completed"]:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("No fresh camera frame")
                after = observer.observe()
                validate_static_interval(before, after, [f])
                q = np.array(after["q_rad"])
                alignment = None
                if surface is not None:
                    alignment = check_alignment(f, q, kin, surface, T, drift_config)
                    if not alignment["passed"]:
                        # A clear route in a misregistered map is not readiness.
                        report = {"mode": "read_only_preflight", "motion_commands_sent": 0,
                                  "motion_authorized": False, "alignment": alignment,
                                  "startup_hold_blocker": blocker,
                                  "route_checked": False, "arm_profile": args.arm,
                                  "q_start": q.tolist(), "camera_capture": f.capture}
                        (out / "result.json").write_text(json.dumps(report, indent=2))
                        cv2.imwrite(str(out / "rgb.jpg"), f.rgb)
                        np.savez_compressed(out / "frame.npz", rgb=f.rgb, depth_m=f.depth_m,
                                            K=f.K, T_cam2base=T, q=q)
                        raise RuntimeError(f"Camera alignment preflight failed: {alignment}")
                slot[0] = RobotState(
                    q=q, gripper_valid=False, t=after["motors"][-1]["read_completed"]
                )
                started = time.monotonic()
                occ.refresh(f, T)
                elapsed = time.monotonic() - started
                status = occ.report()
                reports.append(status)
                print(
                    "MAP",
                    i,
                    "refresh_s",
                    round(elapsed, 3),
                    json.dumps(status),
                    flush=True,
                )
                if status["live"]:
                    goal = np.asarray(cfg.arm.home_q, dtype=float)
                    reason, route = None, None
                    try:
                        # The same route planner called by SafeArm.move_planned.
                        # No custom tiny target, settling test or execution.
                        route = plan_route(harness, q, goal, 3.0, rate_hz=50.0)
                    except (SkillError, SafetyViolation) as exc:
                        reason = str(exc)
                    report = {
                        "mode": "read_only_preflight",
                        "motion_commands_sent": 0,
                        "motion_authorized": False,
                        "startup_hold_blocker": blocker,
                        "alignment": alignment,
                        "map": status,
                        "map_history": reports,
                        "body_mask": mask.last,
                        "unmodeled_attachments_retained_as_obstacles": args.arm == "rebot_rs",
                        "arm_profile": args.arm,
                        "link_attachment_checks": harness.link_attachments is not None,
                        "q_start": q.tolist(),
                        "q_target": goal.tolist(),
                        "tcp_start_m": kin.fk(q)[:3, 3].tolist(),
                        "tcp_target_m": kin.fk(goal)[:3, 3].tolist(),
                        "route_refusal": reason,
                        "clearance_reports": harness.drain_clearance_reports(),
                        "clearance_reports_lost": harness.drain_clearance_lost(),
                        "route_history": refusals + [reason],
                        "candidate_duration_s": 3.0,
                        "planner": "cascade.safety.trajectory.plan_route",
                        "route": None if route is None else [r.tolist() for r in route],
                        "camera_capture": f.capture,
                        "note": "Owned diagnostic bridge stops at script exit. Saved map/route never authorizes execution.",
                    }
                    (out / "result.json").write_text(json.dumps(report, indent=2))
                    # Preserve each map's evidence: a later passing refresh
                    # must not overwrite the depth associated with a refusal.
                    sample_out = out / f"sample-{len(refusals) + 1}"
                    sample_out.mkdir()
                    (sample_out / "result.json").write_text(json.dumps(report, indent=2))
                    np.savez_compressed(sample_out / "frame.npz", rgb=f.rgb,
                                        depth_m=f.depth_m, K=f.K, T_cam2base=T, q=q)
                    cv2.imwrite(str(out / "rgb.jpg"), f.rgb)
                    np.savez_compressed(
                        out / "frame.npz",
                        rgb=f.rgb,
                        depth_m=f.depth_m,
                        K=f.K,
                        T_cam2base=T,
                        q=q,
                    )
                    print(
                        f"PREFLIGHT {len(refusals) + 1}: standard home route; "
                        f"refusal={reason!r}; evidence={out / 'result.json'}",
                        flush=True,
                    )
                    refusals.append(reason)
                    if len(refusals) >= args.map_samples:
                        if any(r is not None for r in refusals):
                            raise RuntimeError(
                                "Candidate refused by at least one fresh map; see result.json"
                            )
                        blocker = startup_hold_blocker(cfg.arm)
                        if blocker:
                            raise RuntimeError(f"Map/route checks passed, but startup remains blocked: {blocker}")
                        break
            else:
                raise RuntimeError(f"Could not establish {args.map_samples} live depth-map updates")
    finally:
        if occ is not None:
            try:
                occ._client.close()
            except Exception:
                pass
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()


if __name__ == "__main__":
    main()
