#!/usr/bin/env python3
"""Run a bounded Astra task with a rendered B601-RS MuJoCo scene (no hardware).

.venv/bin/python scripts/fetch_robot_assets.py rebot_rs
.venv/bin/python scripts/run_rebot_mujoco_astra.py
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from scripts.rebot_astra.experiment import cleanup, configure, task_contract, verdict
from scripts.rebot_astra.model_log import RecordingLLM
from scripts.rebot_astra.simulation import NativeSimulation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key-file', type=Path, default=Path.home() / '.config/cascade/inferencehub_api_key')
    p.add_argument('--inspect-only', action='store_true', help='Render the initial scene without an API request')
    p.add_argument('--max-steps', type=int, default=8)
    p.add_argument('--view', action='store_true', help='Open the native MuJoCo viewer on the desktop')
    p.add_argument('--record', action='store_true', help='Record the visible viewer window to MP4 (requires --view)')
    p.add_argument('--keep-open', action='store_true', help='Leave the final scene on screen until the viewer is closed')
    p.add_argument('--plate', action='store_true', help='Add a blue plate and verify pick-and-place onto it')
    p.add_argument('--task')
    p.add_argument('--verify', choices=['lift', 'plate', 'none'],
                   help='Independent goal check; required with --task (none uses model report only)')
    args = p.parse_args(argv)
    args.task, verification_mode = task_contract(args, p)
    if (args.record or args.keep_open) and not args.view:
        p.error('--record and --keep-open require --view')
    if not 1 <= args.max_steps <= 20:
        p.error('--max-steps must be between 1 and 20')
    # RGB-D rendering owns EGL in this process; the optional desktop viewer
    # owns GLFW in a separate process. Mixing their lifetimes broke GLX teardown.
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['CASCADE_MJ_VIEW'] = '0'
    (ROOT / 'runs').mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='rebot-mujoco-astra-', dir=ROOT / 'runs'))
    for name in ['BELIEFS', 'GRASP_MEMORY', 'ENVELOPE', 'EPISODIC']:
        os.environ[f'CASCADE_{name}_PATH'] = str(out / f'{name.lower()}.json')
    print(f'Run directory: {out}', flush=True)

    from cascade.apps.demo import build_runtime
    cfg = configure()
    if args.plate:
        from scripts.rebot_astra.scene import prepare
        prepare(cfg, out)
    runtime = arm = recorder = simulation = None
    exit_code = 1
    try:
        runtime, arm = build_runtime(cfg, out, view=False, serve=False)
        simulation = NativeSimulation(arm)
        if args.view:
            simulation.configure_viewer()
        if args.record:
            from scripts.rebot_astra.recording import WindowRecording
            recorder = WindowRecording(out / 'mujoco-screen.mp4', target_pid=simulation.viewer.pid)
            recorder.start()
            print(f'Recording on-screen MuJoCo window: {recorder.path}', flush=True)
        def snapshot(name):
            return simulation.snapshot(runtime, out / f'{name}.png')
        initial = snapshot('initial')
        if args.inspect_only:
            print(json.dumps(initial), flush=True)
            exit_code = 0
        else:
            exit_code = run_task(args, cfg, runtime, arm, simulation, out, snapshot, initial, verification_mode)
        if recorder is not None:
            time.sleep(2)
            recorder.stop()
            print(f'Video ready: {recorder.path} ({recorder.frames} frames)', flush=True)
            recorder = None
        if args.keep_open:
            print('Viewer remains open at the final pose; close its window to exit.', flush=True)
            while simulation.viewer.is_running():
                simulation.viewer.sync()
                time.sleep(.1)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        receipt = cleanup(out, recorder=recorder, simulation=simulation, runtime=runtime, arm=arm)
        if not receipt['ok']:
            print(f'Cleanup incomplete: {out / "teardown.json"}', file=sys.stderr)
            exit_code = 1
    return exit_code


def run_task(args, cfg, runtime, arm, simulation, out, snapshot, initial, verification_mode):
    from cascade.agent.llm import make_llm
    from cascade.agent.orchestrator import AgentOrchestrator
    import mujoco

    llm = None
    try:
        key = args.key_file.expanduser().read_text().strip()
        if not key:
            raise ValueError('Empty key file')
        os.environ['NV_API_KEY'] = key
        llm = RecordingLLM(make_llm(cfg.llm), out / 'model-inputs')
        agent = AgentOrchestrator(llm, runtime, max_steps=args.max_steps,
                                  decompose=False, fast_planner=None, verify_milestones=False)
        report = agent.run_task(args.task)
        final = snapshot('final')
        # Measure sustained lift independently from model self-report. Hold the
        # actuator targets and advance 0.5 s of native physics before re-reading.
        simulation.hold(0.5)
        settled = snapshot('settled')
        lift = settled['cube_xyz'][2] - initial['cube_xyz'][2]
        placement = None
        if args.plate:
            from scripts.rebot_astra.scene import audit
            placement = audit(simulation.world, arm, cfg.arm.gripper)
            print(f'Independent plate audit: {json.dumps(placement)}', flush=True)
        verification = verdict(verification_mode, report.success, lift, placement)
        result = {'model': llm.model, 'mujoco_version': mujoco.__version__,
                  'asset_commit': 'da76818e269b82289eba39808e2fb91d679d6994',
                  'initial': initial, 'final': final, 'after_half_second_hold': settled,
                  'sustained_cube_lift_m': lift, 'lift_over_3cm': bool(lift > 0.03),
                  'plate_placement': placement,
                  'verification': verification,
                  'requests': llm.requests, 'report': asdict(report),
                  'limitations': ['simulation only', 'analytic OBB grasps',
                                  'no nvblox occupancy map', 'color-threshold cube detector']}
        (out / 'result.json').write_text(json.dumps(result, indent=2, default=str))
        measured_result = f'goal verification={verification}'
        print(f'Model task success={report.success}; {measured_result}', flush=True)
        print(f'Result: {out / "result.json"}', flush=True)
        return 0 if verification['success'] else 1
    finally:
        if llm is not None:
            receipt = cleanup(out / 'model-inputs', llm=llm)
            if not receipt['ok']:
                raise RuntimeError('LLM cleanup failed; see model-inputs/teardown.json')


if __name__ == '__main__':
    raise SystemExit(main())
