#!/usr/bin/env python3
"""Live Astra API smoke check using Cascade's B601-RS kinematic mock and camera.

Run from the checkout: .venv/bin/python scripts/verify_inferencehub_astra.py
This verifies API integration, not physics or physical robot performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.rebot_astra.model_log import RecordingLLM
from scripts.rebot_astra.experiment import cleanup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path,
                        default=Path.home() / ".config/cascade/inferencehub_api_key")
    args = parser.parse_args()
    key = args.key_file.expanduser().read_text().strip()
    if not key:
        parser.error("The key file is empty")
    os.environ["NV_API_KEY"] = key
    import tempfile
    run_dir = Path(tempfile.mkdtemp(prefix="astra-api-", dir=ROOT / "runs"))
    for variable, filename in {
        "CASCADE_BELIEFS_PATH": "beliefs.json",
        "CASCADE_GRASP_MEMORY_PATH": "grasp_memory.json",
        "CASCADE_ENVELOPE_PATH": "envelope.json",
        "CASCADE_EPISODIC_PATH": "episodic.json",
    }.items():
        os.environ[variable] = str(run_dir / filename)

    from cascade.agent.llm import make_llm
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock", arm="mock", llm="inferencehub_astra",
                           _ignore_robot_environment=True)
    cfg.grasp._data["backend"] = "obb"
    llm = RecordingLLM(make_llm(cfg.llm), run_dir / 'model-inputs')
    runtime = arm = None
    try:
        runtime, arm = build_runtime(cfg, run_dir, view=False, serve=False)
        execute = runtime.execute

        def read_only_execute(name, arguments):
            if name not in {"get_observation", "task_done"}:
                return {"ok": False, "error": "This API check only permits get_observation"}
            return execute(name, arguments)

        runtime.execute = read_only_execute
        agent = AgentOrchestrator(llm, runtime, max_steps=4, decompose=False,
                                  fast_planner=None, verify_milestones=False)
        agent.tool_specs = [s for s in agent.tool_specs
                            if s["name"] in {"get_observation", "task_done"}]
        report = agent.run_task(
            "Verify the Astra API connection only. First call get_observation, "
            "then inspect the supplied camera image and tool result. Call task_done "
            "with a short description of the visible scene. This is a static mock "
            "camera and B601-RS kinematic mock; do not claim physical task success."
        )
        names = [call['name'] for request in llm.requests for call in request['calls']]
        passed = (report.success and report.path == 'llm'
                  and names == ['get_observation', 'task_done']
                  and any(r['images'] > 0 for r in llm.requests)
                  and any(t.get('tool') == 'get_observation' and t.get('result', {}).get('ok')
                          for t in report.tool_log))
        result = {"passed": passed, "model": llm.model,
                  "scope": "API and mock observation only; no physics or hardware validation",
                  "requests": llm.requests, "report": asdict(report)}
        (run_dir / "verification.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"{'PASS' if passed else 'FAIL'}: {run_dir / 'verification.json'}")
        return 0 if passed else 1
    except Exception as exc:
        # Provider errors can contain request details; print only class/status.
        print(f"FAIL: {type(exc).__name__}; HTTP status={getattr(exc, 'status_code', None)}")
        raise SystemExit(1) from None
    finally:
        receipt = cleanup(run_dir, runtime=runtime, arm=arm, llm=llm)
        if not receipt['ok']:
            raise RuntimeError(f'Cleanup incomplete: {run_dir / "teardown.json"}')


if __name__ == "__main__":
    (ROOT / "runs").mkdir(exist_ok=True)
    raise SystemExit(main())
