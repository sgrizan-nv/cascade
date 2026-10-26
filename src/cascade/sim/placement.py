"""Fresh read-only kitchen placement evidence, separate from actuator feedback.

The installed checkout supplies the same scene binding and geometry auditors
as the external acceptance proof. Missing optional resources fail unverified.
No prior READY receipt, motion result or authored spawn pose is reused.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import uuid

from .bridge_client import BridgeClient
from .truth import _match_label

ROOT = Path(__file__).resolve().parents[3]
WALL_BUDGET_S = 20.0
REQUEST_TIMEOUT_S = 4.0
INTERVAL_S = 0.1


@lru_cache(maxsize=1)
def _auditor():
    spec = importlib.util.spec_from_file_location(
        "cascade_kitchen_placement_verdict", ROOT / "demo/kitchen/physics/placement_verdict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lab_config():
    name = os.environ.get("CASCADE_SCENE_CONFIG")
    if not name:
        return None
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or path.stat().st_size > 1024 * 1024:
        raise ValueError("Scene configuration must be a bounded local checkout file")
    config = json.loads(path.read_text())
    return config if config.get("lab_scene") else None


LAB_POSE_CODE = """
import json as _lab_j
from isaacsim.core.experimental.prims import RigidPrim as _lab_R
_lab_rp = _lab_R('/World_Props/' + LAB_OBJECT)
_lab_p, _ = _lab_rp.get_world_poses()
_lab_v, _ = _lab_rp.get_velocities()
_lab_p = (_lab_p.numpy() if hasattr(_lab_p, 'numpy') else _lab_p)[0]
_lab_v = (_lab_v.numpy() if hasattr(_lab_v, 'numpy') else _lab_v)[0]
print('LABPOSE ' + _lab_j.dumps({'sim_time': float(SimulationManager.get_simulation_time()),
    'physics_step': int(SimulationManager.get_num_physics_steps()),
    'position_m': [float(_lab_p[0]), float(_lab_p[1]), float(_lab_p[2]) - float(BASE_Z)],
    'speed_m_s': float(sum(float(c) ** 2 for c in _lab_v[:3]) ** .5)}))
"""


def _lab_region(config, destination):
    if destination == "open box" and "open_box" in config:
        box = config["open_box"]
        return {"center_xy_m": box["center_xy_m"],
                "half_m": [d / 2 for d in box["interior_dimensions_m"][:2]],
                "support_top_z_m": box["support_top_z_m"], "footprint_inside": False}
    if destination == "green square" and "target_pad" in config:
        pad = config["target_pad"]
        half = pad["outer_size_m"] / 2 - pad["border_width_m"]
        return {"center_xy_m": pad["center_xy_m"], "half_m": [half, half],
                "support_top_z_m": pad["support_top_z_m"], "footprint_inside": True}
    return None


def _read_lab_placement(endpoint, config, label, destination):
    """Lab scenes: settle window from live PhysX rigid-body pose and speed only."""
    measured = {"destination": destination, "scope": "postplacement only", "channel": "lab rigid-body readback"}
    names = {p["name"]: p["name"] for p in config.get("props", [])}
    names.update({n: n for n in config.get("cube_dimensions", {})})
    object_name = _match_label(label, names)
    if object_name is None:
        # Visitors add materials the prop names omit ("steel hex nut" -> hex_nut).
        known = {t for n in names for t in n.split("_")}
        wanted = [t for t in str(label).lower().replace("-", " ").split() if t in known]
        hits = [n for n in names if wanted and set(wanted) <= set(n.split("_"))]
        object_name = hits[0] if len(hits) == 1 else None
    region = _lab_region(config, destination)
    if object_name is None or region is None:
        return {"status": "unverified", "evidence": "requested object or destination has no lab scene identity",
                "measured": measured}
    dims = next((p["dimensions"] for p in config.get("props", []) if p["name"] == object_name),
                config.get("cube_dimensions", {}).get(object_name))
    records, client = [], BridgeClient(host=endpoint[0], port=endpoint[1], timeout_s=REQUEST_TIMEOUT_S)
    deadline = time.monotonic() + WALL_BUDGET_S
    code = "LAB_OBJECT = " + json.dumps(object_name) + "\n" + LAB_POSE_CODE
    try:
        client.connect()
        while time.monotonic() < deadline:
            reply = client.request({"op": "exec", "code": code}, timeout_s=REQUEST_TIMEOUT_S)
            line = next((l for l in str(reply.get("stdout") or "").splitlines() if l.startswith("LABPOSE ")), None)
            if line is None:
                raise ValueError("lab pose readback returned no sample")
            sample = json.loads(line[8:])
            if records and (sample["sim_time"] < records[-1]["sim_time"]
                            or sample["physics_step"] < records[-1]["physics_step"]):
                raise ValueError("simulation restarted during placement observation")
            records.append(sample)
            if len(records) >= 6 and sample["sim_time"] - records[0]["sim_time"] >= .6:
                break
            time.sleep(INTERVAL_S)
        else:
            raise TimeoutError("placement observation exceeded its wall-time budget")
    except Exception as exc:
        return {"status": "unverified", "evidence": f"placement observation unavailable: {type(exc).__name__}: {exc}",
                "measured": measured}
    finally:
        client.close()
    first, last = records[0]["position_m"], records[-1]["position_m"]
    drift = math.dist(first, last)
    speed = max(r["speed_m_s"] for r in records[-3:])
    cx, cy = region["center_xy_m"]
    reach = max(dims[0], dims[1]) / 2 if region["footprint_inside"] else 0.0
    margin = min(region["half_m"][0] - abs(last[0] - cx) - reach, region["half_m"][1] - abs(last[1] - cy) - reach)
    resting = last[2] <= region["support_top_z_m"] + max(dims) / 2 + 0.01
    measured.update(object=object_name, position_m=[round(v, 4) for v in last], inside_margin_m=round(margin, 4),
                    drift_m=round(drift, 4), max_speed_m_s=round(speed, 4), samples=len(records),
                    window_s=round(records[-1]["sim_time"] - records[0]["sim_time"], 3))
    if speed > 0.02 or drift > 0.003:
        return {"status": "unverified", "evidence": "object still moving in the post-release window",
                "measured": measured}
    if margin < 0 or not resting:
        why = "outside the destination region" if margin < 0 else "not resting on the destination support"
        return {"status": "refuted", "evidence": f"settled {object_name} is {why}", "measured": measured}
    return {"status": "confirmed", "evidence": f"settled {object_name} rests inside the {destination}",
            "measured": measured}


def read_placement(endpoint, robot_id, label, destination, *, evidence_dir=None):
    """Sample only after the caller's motion finishes, using a private socket.

    A failed/timed-out read cannot poison a camera or motion client's stream.
    Both wall time and sample count are bounded. Frozen or restarted simulation
    clocks never become a successful settling window.
    """
    try:
        lab = _lab_config()
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "unverified", "evidence": f"lab scene configuration unavailable: {type(exc).__name__}",
                "measured": {"destination": destination, "scope": "postplacement only"}}
    if lab is not None:
        return _read_lab_placement(endpoint, lab, label, destination)
    report = {"status": "unverified", "evidence": "placement physics unavailable",
              "measured": {"destination": destination, "scope": "postplacement only"}}
    records, client, expected, code = [], None, None, ""
    started = time.monotonic()
    deadline = started + WALL_BUDGET_S
    try:
        auditor = _auditor()
        expected = auditor.gpu.load_expected_scene_geometry(ROOT / "demo/scene/kitchen_config.json")
        names = tuple(expected["prop_dimensions_m"])
        object_name = _match_label(label, {name: name for name in names})
        if object_name is None or destination not in {"green square", "open box"}:
            raise ValueError("requested object or destination has no unambiguous kitchen identity")
        code = auditor.gpu.snapshot_code(object_name=object_name, include_scene_geometry=True,
            include_open_box="open_box" in expected, expected_props=names)
        client = BridgeClient(host=endpoint[0], port=endpoint[1], timeout_s=REQUEST_TIMEOUT_S)
        client.connect()
        for _ in range(200):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("placement observation exceeded its wall-time budget")
            began = time.monotonic()
            reply = client.request({"op": "exec", "code": code},
                                   timeout_s=min(REQUEST_TIMEOUT_S, remaining))
            sample = auditor.gpu.original.parse_snapshot_reply(reply)
            finished = time.monotonic()
            if finished > deadline:
                raise TimeoutError("placement observation exceeded its wall-time budget")
            records.append({"sequence": len(records), "client_started_monotonic": began,
                            "client_finished_monotonic": finished, "physics": sample})
            clock = sample.get("sim_time")
            step = sample.get("physics_step")
            if (type(clock) not in (int, float) or not math.isfinite(clock)
                    or type(step) is not int or step < 0):
                raise ValueError("invalid simulation clock in placement observation")
            if len(records) > 1:
                previous = records[-2]["physics"]
                if clock < previous["sim_time"] or step < previous["physics_step"]:
                    raise ValueError("simulation restarted during placement observation")
            if len(records) >= 6 and clock - records[0]["physics"]["sim_time"] >= .6:
                break
            time.sleep(min(INTERVAL_S, max(0., deadline - time.monotonic())))
        else:
            raise TimeoutError("placement observation exhausted its sample budget")
        report = auditor.audit_placement(records, object_name=object_name,
            destination_name=destination, expected_scene_geometry=expected, robot_id=robot_id)
        report["measured"]["destination"] = destination
    except Exception as exc:
        report = {"status": "unverified",
                  "evidence": f"placement observation unavailable: {type(exc).__name__}: {exc}",
                  "measured": {"destination": destination, "scope": "postplacement only"}}
    finally:
        if client is not None:
            client.close()
    report["measured"].update(observed_samples=len(records),
        observation_wall_s=round(time.monotonic() - started, 3))
    if evidence_dir is not None:
        # Preserve the readings used by this call alongside its normal trace.
        # A receipt write failure cannot erase a physical refutation.
        try:
            out = Path(evidence_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{uuid.uuid4().hex}.json"
            payload = {"version": 1, "requested_object": label, "destination": destination,
                       "robot_id": robot_id, "expected_scene_geometry": expected,
                       "snapshot_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                       "records": records, "verdict": report}
            raw = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
            path.write_text(raw)
            report["measured"].update(evidence_path=str(path),
                evidence_sha256=hashlib.sha256(raw.encode()).hexdigest())
        except (OSError, ValueError, TypeError) as exc:
            report["measured"]["evidence_write_error"] = type(exc).__name__
    return report
