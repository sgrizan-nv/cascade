"""Workstation launcher for the standard CASCADE CLI.

Only `run` opens the control backend, in the operator's terminal. There is
no commissioning trajectory, alternate agent loop or controller tuning.
"""
# Direct execution selects this checkout before package imports.
# ruff: noqa: E402
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from cascade.config import load_demo_config
from scripts.fit_rebot_oak_calibration import prepare_model


def prepare():
    cfg = load_demo_config(arm="rebot_rs", camera="oak_d", llm="mock",
                          _ignore_robot_environment=True)
    return prepare_model(cfg, ROOT / ".runtime-rebot/model")


def config():
    return load_demo_config(arm="rebot_rs_oak", camera="oak_d", llm="inferencehub_astra",
                            _ignore_robot_environment=True)


@contextmanager
def bridge(out):
    """Own one ephemeral bridge; never clear or stop a preexisting service."""
    instance = "rebot-oak-" + out.name
    with (out / "bridge.log").open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/serve_occupancy_bridge.py"),
             "--port", "0", "--backend", "warp", "--device", "cuda:0",
             "--instance-id", instance], stdout=subprocess.PIPE, stderr=log,
            # Ctrl+C belongs to CASCADE's stop/park handler. Keep its map
            # available until normal shutdown has finished using it.
            start_new_session=True,
        )
        try:
            pending, ready = b"", None
            deadline = time.monotonic() + 60
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while ready is None and time.monotonic() < deadline:
                    if not selector.select(1):
                        continue
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError("Owned obstacle bridge exited before readiness")
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if line.startswith(b"CASCADE_SERVER_READY "):
                            ready = json.loads(line.split(b" ", 1)[1])
                            break
            if ready is None or ready.get("pid") != proc.pid or ready.get("instance") != instance:
                raise RuntimeError("Owned obstacle bridge readiness/identity failed")
            yield ready
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.stdout.close()



def run(out):
    from cascade.control import arm_hold
    if arm_hold.agent_environment() is not None or not arm_hold._interactive():
        raise RuntimeError("Start the arm in your own workstation terminal; an agent cannot confirm physical readiness")
    if any(os.environ.get(key) for key in ("CASCADE_ROBOT", "CASCADE_BASE")):
        raise RuntimeError("Unset CASCADE_ROBOT/CASCADE_BASE before selecting this arm workstation")
    from cascade.apps.demo import main as demo_main
    cfg = config()
    key = Path(cfg.camera.detector.api_key_file).read_text().strip()
    if not key:
        raise ValueError("Configured inference key is empty")
    os.environ["NV_API_KEY"] = key
    # Existing CASCADE attempt recorder: localization, candidates, IK/vetting
    # and configuration. Supplements the map-refusal input snapshots.
    os.environ["CASCADE_GRASP_EVIDENCE_DIR"] = str(out / "grasp_evidence")
    print("Starting the standard CASCADE interactive CLI. This enables the arm. "
          "Shutdown uses CASCADE's normal stop/park/release policy.", flush=True)
    with bridge(out) as owned:
        os.environ["CASCADE_OCCUPANCY_PORT"] = str(owned["port"])
        # Explicit robot selection is handled by CASCADE's normal CLI/config.
        return demo_main(["--arm", "rebot_rs_oak", "--camera", "oak_d",
                          "--llm", "inferencehub_astra", "--interactive",
                          "--no-view", "--no-serve", "--run-dir", str(out)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "status", "check", "run"])
    args = parser.parse_args(argv)
    # Refuse the obsolete `commission`/`--load-test` paths in argparse,
    # before preparing files, starting services or importing the actuator.
    model = prepare()
    cfg = config()
    if args.command == "prepare":
        print(f"Verified model: {model}")
        return 0
    if args.command == "status":
        from cascade.control.rebot_rs_arm import RebotRSArm
        from cascade.perception.grounding import Extrinsics
        # Construction only, never connect(). Calibration is assessed by its
        # existing loader, not just by file existence.
        raw = RebotRSArm(cfg.arm)
        ext = Extrinsics.from_config(cfg.camera.extrinsics, camera_serial=cfg.camera.serial)
        ext.cam_to_base()
        print(json.dumps({"python": sys.executable, "model": str(model),
            "arm": "rebot_rs_oak", "camera": cfg.camera.serial,
            "calibration_loaded": True, "gravity_scale": raw._gc_tau_scale.tolist(),
            "settle_tol_rad": raw.settle_tol, "settle_timeout_s": raw.settle_timeout_s,
            "home_q": cfg.arm.home_q, "mit_gains": "SDK rebotarm_rs.yaml",
            "llm_profile": "inferencehub_astra", "grasp_backend": cfg.grasp.backend,
            "entrypoint": "cascade.apps.demo.main", "hardware_opened": False}, indent=2))
        return 0
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"rebot-oak-{args.command}-", dir=runs))
    print(f"Evidence: {out}", flush=True)
    if args.command == "check":
        return subprocess.call([sys.executable, str(ROOT / "scripts/preflight_rebot_oak.py"),
            "--arm", "rebot_rs_oak", "--output", str(out / "preflight"),
            "--feedback-python", sys.executable])
    return run(out)


if __name__ == "__main__":
    raise SystemExit(main())
