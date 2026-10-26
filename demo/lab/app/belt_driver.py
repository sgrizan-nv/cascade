#!/usr/bin/env python3
"""Cascade Lab recycling, phase 2: moving-belt loop driven from the host (scripted, engine-agnostic).

Honest summary: the belt collider stays static (Newton and PhysX). Every tick this script moves the waste lying on the belt
band a few millimetres toward the tail (-x) through the bridge `exec` op and scrolls the belt texture.
Past the tail drum (x < -0.16) an item is no longer nudged: it falls off the collider into the drop hole
by PhysX gravity. Once below z = -0.2 it is re-placed at the infeed (x ~0.70) with zero velocity (scripted
respawn). The belt pauses while the cell has a robot order running, so grasps never chase a moving item.

    python3 belt_driver.py --speed 0.03 --minutes 10        # run on the Brev host
"""
import argparse
import json
import random
import subprocess
import time

import lab_tests as t

PROPS = {"soda_can": 0.10, "yogurt_cup": 0.066}
BELT_Y = (0.11, 0.3796)
TAIL_X, INFEED_X = -0.16, 0.70

TICK_CODE = """
import json as _j
from isaacsim.core.experimental.prims import RigidPrim as _R
from pxr import Gf as _Gf, UsdGeom as _UG, Sdf as _Sdf
_out = {}
def _place(_rp, _n, _x, _y, _z, _q):
    # Newton: write both state buffers (upright, zero velocity); PhysX: keep the current orientation.
    if '_newton_teleport' in globals() and args.engine == 'newton':
        return _newton_teleport(_n, (_x, _y, _z))
    _qq = np.array([[1.0, 0.0, 0.0, 0.0]]) if _q is None else np.array([[float(v) for v in _q]])
    _rp.set_world_poses(np.array([[_x, _y, _z + BASE_Z]]), _qq)
    _zero_prop_velocity(_rp) if _q is None else None
    return True
for _n, _h in PROPS.items():
    _rp = _R('/World_Props/' + _n)
    _p, _q = _rp.get_world_poses()
    _p = _p.numpy()[0] if hasattr(_p, 'numpy') else _p[0]
    _q = _q.numpy()[0] if hasattr(_q, 'numpy') else _q[0]
    _x, _y, _z = float(_p[0]), float(_p[1]), float(_p[2]) - BASE_Z
    _act = 'stay'
    if _z < -0.2:
        _x, _y, _z, _act = INFEED_X, RESPAWN_Y[_n], _h / 2 + 0.004, 'respawn'
        _place(_rp, _n, _x, _y, _z, None)
    elif RUN and BELT_Y[0] < _y < BELT_Y[1] and _x > TAIL_X - 0.035 and -0.01 < _z < _h / 2 + 0.02:
        _x -= STEP
        _act = 'move'
        _place(_rp, _n, _x, _y, _z, _q)
    _out[_n] = [round(_x, 4), round(_y, 4), round(_z, 4), _act]
if RUN:
    _m = _UG.Mesh(stage.GetPrimAtPath('/Kitchen/SortConveyor/Belt'))
    _pv = _UG.PrimvarsAPI(_m).GetPrimvar('st')
    _st = _pv.Get()
    _pv.Set([_Gf.Vec2f(float(s[0]) + STEP, float(s[1])) for s in _st])
print('BELT ' + _j.dumps(_out))
"""


def cell_busy():
    """Pause the belt while the cell's agent is working (any openclaw-agent alive in the container).

    Checked through docker, not the lab app: the app rate-limits logins for every local script."""
    try:
        out = subprocess.run(["sudo", "-n", "docker", "exec", "cascade-lab-recycling", "pgrep", "-c", "openclaw-agent"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out or 0) > 0
    except (ValueError, OSError, subprocess.SubprocessError):
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.03, help="belt speed in m/s")
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--minutes", type=float, default=10.0)
    a = ap.parse_args()
    ip = t.container_ip("recycling")
    rng = random.Random(7)
    t_end = time.time() + 60 * a.minutes
    last_busy, busy = 0.0, False
    while time.time() < t_end:
        t0 = time.time()
        if t0 - last_busy > 3:
            busy, last_busy = cell_busy(), t0
        respawn = {n: round(rng.uniform(0.16, 0.32), 3) for n in PROPS}
        code = (f"PROPS = {PROPS!r}\nBELT_Y = {BELT_Y!r}\nTAIL_X = {TAIL_X}\nINFEED_X = {INFEED_X}\n"
                f"RESPAWN_Y = {respawn!r}\nRUN = {not busy}\nSTEP = {a.speed / a.hz:.5f}\n" + TICK_CODE)
        r = t.bridge(ip, {"op": "exec", "code": code}, timeout=20)
        line = next((x for x in (r.get("stdout") or "").splitlines() if x.startswith("BELT ")), None)
        if line:
            state = json.loads(line[5:])
            events = {n: v for n, v in state.items() if v[3] == "respawn"}
            if events:
                print(time.strftime("%T"), "respawn", events, flush=True)
        else:
            print(time.strftime("%T"), "exec failed", json.dumps(r)[:300], flush=True)
        time.sleep(max(0.0, 1.0 / a.hz - (time.time() - t0)))


if __name__ == "__main__":
    main()
