#!/usr/bin/env python3
"""Cascade Lab experiments: one interaction at a time, judged on measured physics.

Each experiment resets its cell, records camera snapshots before and after, sends one order to
the agent through the lab server (or drives the bridge directly for physics probes), then reads
the real rigid-body poses from Isaac and applies an explicit pass rule.

    python3 lab_tests.py --cell workshop --exp nut_tray
    python3 lab_tests.py --cell workshop --exp peg_drop      # physics only, no agent
"""
import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OPENER = build_opener(ProxyHandler({}))
SERVER = "http://127.0.0.1:8095"
ADMIN_PW = os.environ.get("CASCADE_LAB_ADMIN_PASSWORD", "")
CAMS = ("kitchen", "worktop", "side")

TRAY = {"center": (0.30, -0.14), "half": 0.064, "floor_z": 0.004}
PAD = {"center": (0.14, -0.27), "half": 0.07}
PEG = {"center": (0.36, -0.30)}
PLASTIC_BIN = {"center": (0.30, -0.14), "half": 0.064}   # recycling v3: blue bin = calibrated open box
METAL_BIN = {"center": (0.13, -0.25), "half": 0.074}     # recycling v3: yellow fixture bin, 148 mm cavity
SORT = {"soda_can": METAL_BIN, "yogurt_cup": PLASTIC_BIN}  # bottle removed 17:45 (visual neck vs cylinder collider)


def inside(xy, region):
    return all(abs(xy[i] - region["center"][i]) <= region["half"] for i in range(2))


EXPERIMENTS = {
    "nut_tray": {"cell": "workshop", "order": "Put the steel hex nut in the blue parts tray.", "object": "hex_nut",
                 "rule": lambda p: inside(p, TRAY) and p[2] < 0.06, "rule_text": "nut centre inside the tray walls"},
    "gear_pad": {"cell": "workshop", "order": "Put the brass gear on the green square.", "object": "brass_gear",
                 "rule": lambda p: inside(p, PAD) and p[2] < 0.04, "rule_text": "gear centre inside the green square"},
    "collar_tray": {"cell": "workshop", "order": "Put the red collar in the blue parts tray.", "object": "red_collar",
                    "rule": lambda p: inside(p, TRAY) and p[2] < 0.06, "rule_text": "collar centre inside the tray"},
    "collar_peg": {"cell": "workshop", "order": "Put the red collar on the steel peg.", "object": "red_collar",
                   "rule": lambda p: ((p[0] - PEG["center"][0]) ** 2 + (p[1] - PEG["center"][1]) ** 2) ** .5 < 0.012
                   and p[2] < 0.03, "rule_text": "collar seated around the peg (axis error < 12 mm, z < 30 mm)"},
    "collar_peg_direct": {"cell": "workshop", "object": "red_collar",
                          "direct": {"tool": "pick_and_place",
                                     "args": {"object": "red collar", "destination": "steel peg"}},
                          "rule": lambda p: ((p[0] - PEG["center"][0]) ** 2 + (p[1] - PEG["center"][1]) ** 2) ** .5
                          < 0.012 and p[2] < 0.03,
                          "rule_text": "collar seated around the peg via the MCP tool, no LLM (axis error < 12 mm)"},
    "peg_drop": {"cell": "workshop", "physics": True, "object": "red_collar",
                 "rule": lambda p: ((p[0] - PEG["center"][0]) ** 2 + (p[1] - PEG["center"][1]) ** 2) ** .5 < 0.012
                 and p[2] < 0.03, "rule_text": "collar dropped 6 cm above the peg slides down to the bench"},
    "can_bin": {"cell": "recycling", "order": "Put the soda can in the blue recycling bin.", "object": "soda_can",
                "rule": lambda p: inside(p, TRAY) and p[2] < 0.1, "rule_text": "can inside the recycling bin"},
    "bottle_bin": {"cell": "recycling", "order": "Put the plastic bottle in the blue recycling bin.",
                   "object": "plastic_bottle", "rule": lambda p: inside(p, TRAY) and p[2] < 0.1,
                   "rule_text": "bottle inside the recycling bin"},
    "paper_pad": {"cell": "recycling", "order": "Put the paper ball on the green square.", "object": "paper_ball",
                  "rule": lambda p: inside(p, PAD) and p[2] < 0.05, "rule_text": "paper ball inside the green square"},
    "can_metal": {"cell": "recycling", "order": "Put the soda can in the yellow metal bin.", "object": "soda_can",
                  "rule": lambda p: inside(p, METAL_BIN) and p[2] < 0.1, "rule_text": "can inside the metal bin"},
    "tin_metal": {"cell": "recycling", "order": "Put the tin can in the yellow metal bin.", "object": "tin_can",
                  "rule": lambda p: inside(p, METAL_BIN) and p[2] < 0.1, "rule_text": "tin inside the metal bin"},
    "bottle_plastic": {"cell": "recycling", "order": "Put the plastic bottle in the blue plastic bin.",
                       "object": "plastic_bottle", "rule": lambda p: inside(p, PLASTIC_BIN) and p[2] < 0.1,
                       "rule_text": "bottle inside the plastic bin"},
    "cup_plastic": {"cell": "recycling", "order": "Put the yogurt cup in the blue plastic bin.", "object": "yogurt_cup",
                    "rule": lambda p: inside(p, PLASTIC_BIN) and p[2] < 0.1, "rule_text": "cup inside the plastic bin"},
    "sort_all": {"cell": "recycling", "order": "Sort the waste.", "objects": SORT, "timeout": 2400,
                 "rule_text": "every item inside its material's bin (metal: yellow, plastic: blue)"},
}


def container_ip(cell):
    out = subprocess.run(["sudo", "-n", "docker", "inspect", "-f",
                          "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", f"cascade-lab-{cell}"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    if not out:
        raise SystemExit(f"cell {cell} is not running")
    return out


def bridge(ip, req, timeout=90.0):
    with socket.create_connection((ip, 8611), timeout=timeout) as sock:
        sock.settimeout(timeout)
        f = sock.makefile("rwb")
        f.write((json.dumps(req) + "\n").encode())
        f.flush()
        return json.loads(f.readline() or b"{}")


POSE_CODE = """
import json as _j
from isaacsim.core.experimental.prims import RigidPrim as _R
_out = {}
for _n in NAMES:
    try:
        _p, _q = _R('/World_Props/' + _n).get_world_poses()
        _v = _p.numpy()[0] if hasattr(_p, 'numpy') else _p[0]
        _out[_n] = [round(float(_v[0]), 4), round(float(_v[1]), 4), round(float(_v[2]) - BASE_Z, 4)]
    except Exception as _e:
        _out[_n] = repr(_e)[:120]
print('POSES ' + _j.dumps(_out))
"""


def poses(ip, names):
    r = bridge(ip, {"op": "exec", "code": "NAMES = " + json.dumps(list(names)) + "\n" + POSE_CODE}, timeout=40)
    for line in (r.get("stdout") or "").splitlines():
        if line.startswith("POSES "):
            return json.loads(line[6:])
    raise RuntimeError("pose readback failed: " + json.dumps(r)[:400])


def snap(ip, out, tag):
    for cam in CAMS:
        try:
            with OPENER.open(f"http://{ip}:8091/snapshot/{cam}.jpg", timeout=6) as r:
                (out / f"{tag}_{cam}.jpg").write_bytes(r.read())
        except Exception as e:
            print("snapshot failed", cam, type(e).__name__)


def server(path, data=None, cookie=None, client="lab-tests"):
    headers = {"Content-Type": "application/json", "X-Client": client}
    if cookie:
        headers["Cookie"] = cookie
    req = Request(SERVER + path, data=None if data is None else json.dumps(data).encode(), headers=headers,
                  method="POST" if data is not None else "GET")
    with OPENER.open(req, timeout=30) as r:
        return json.load(r), r.headers.get("Set-Cookie")


def agent_order(cell, text, timeout=1300):
    # A fresh client id per order = a fresh agent session (the server keys sessions by client[:20]),
    # so earlier experiments do not fill the 16k context of the next one.
    client = "lt" + time.strftime("%d%H%M%S") + format(int(time.time() * 1000) % 1000, "03d")
    _, cookie = server("/api/auth", {"password": ADMIN_PW}, client=client)
    cookie = cookie.split(";")[0]
    job, _ = server(f"/api/chat?s={cell}", {"text": text, "s": cell}, cookie, client=client)
    t0 = time.time()
    hard_stop = t0 + 3 * timeout
    while time.time() - t0 < timeout and time.time() < hard_stop:
        time.sleep(3)
        if job.get("status") == "queued":
            t0 += 3  # time waiting behind other users' orders does not count
        try:
            job, _ = server(f"/api/job/{job['id']}", cookie=cookie, client=client)
        except OSError as e:
            # Other agents restart the lab app; its jobs live in memory, the cell keeps working.
            if getattr(e, "code", None) == 404:
                time.sleep(max(0, min(timeout - (time.time() - t0), 240)))
                return {**job, "status": "lost", "reply": "lab server restarted during the job"}
            if getattr(e, "code", None) in (401, 403):
                try:
                    cookie = server("/api/auth", {"password": ADMIN_PW}, client=client)[1].split(";")[0]
                except OSError:
                    pass
            continue
        if job["status"] in ("done", "error", "cancelled"):
            return job
    return {**job, "status": "timeout"}


def direct_tool(cell, tool, args, timeout=900):
    """One MCP tool call inside the cell (mcp_direct.py), bypassing the LLM and the lab server queue."""
    cmd = ["sudo", "-n", "docker", "exec", "-i", "-e", "LAB_TOOL=" + tool, "-e", "LAB_ARGS=" + json.dumps(args),
           "-e", f"LAB_TIMEOUT={timeout}", f"cascade-lab-{cell}", "/opt/paai/venv/bin/python", "-"]
    p = subprocess.run(cmd, stdin=open(HERE / "mcp_direct.py"), capture_output=True, text=True, timeout=timeout + 300)
    line = next((l for l in reversed((p.stdout + "\n" + p.stderr).splitlines()) if l.startswith("DIRECT ")), None)
    data = json.loads(line[7:]) if line else {"error": (p.stderr or p.stdout)[-600:]}
    result = data.get("result") or {}
    return {"status": "direct", "reply": json.dumps(data)[:3000], "tools": [tool],
            "direct_ok": result.get("ok"), "direct_outcome": result.get("outcome")}


def sync_peg(cell):
    """The peg moved between scene versions: read its position from the running cell's own config."""
    try:
        name = subprocess.run(["sudo", "-n", "docker", "exec", f"cascade-lab-{cell}", "printenv",
                               "PAAI_LAB_SCENE_CONFIG"], capture_output=True, text=True, timeout=15).stdout.strip()
        raw = subprocess.run(["sudo", "-n", "docker", "exec", f"cascade-lab-{cell}", "cat", "/workspace/" + name],
                             capture_output=True, text=True, timeout=15).stdout
        peg = next(f for f in json.loads(raw).get("fixtures", []) if f.get("kind") == "peg")
    except (ValueError, StopIteration, OSError, subprocess.SubprocessError):
        return None
    PEG["center"] = tuple(peg["position"])
    PEG["height"] = float(peg["height"])
    return PEG["center"]


def wait_idle(cell, max_s=1500):
    """Visitors and other agents share the cells: never reset props under someone else's order."""
    _, cookie = server("/api/auth", {"password": ADMIN_PW})
    cookie = cookie.split(";")[0]
    t0 = time.time()
    while time.time() - t0 < max_s:
        st, _ = server(f"/api/status?s={cell}", cookie=cookie)
        if st.get("robot") == "idle" and not st.get("current") and not st.get("queue"):
            return round(time.time() - t0, 1)
        time.sleep(5)
    raise SystemExit(f"cell {cell} stayed busy for {max_s} s")


def run(exp_name, cell=None, note=""):
    exp = EXPERIMENTS[exp_name]
    cell = cell or exp["cell"]
    waited = wait_idle(cell)
    if waited:
        print(f"waited {waited} s for {cell} to be idle")
    ip = container_ip(cell)
    if cell == "workshop":
        sync_peg(cell)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = RUNS / f"{stamp}_{cell}_{exp_name}"
    out.mkdir(parents=True)
    reset = bridge(ip, {"op": "reset_props"}, timeout=90)
    time.sleep(2)
    names = list(exp["objects"]) if "objects" in exp else [exp["object"]]
    before = poses(ip, names)
    snap(ip, out, "before")
    t0 = time.time()
    job = None
    if exp.get("physics"):
        x, y = PEG["center"]
        placed = bridge(ip, {"op": "place_prop", "name": exp["object"],
                             "pos": [x, y, PEG.get("height", 0.08) + 0.06 + 0.015]}, timeout=60)
        time.sleep(3)
        job = {"status": "physics", "reply": json.dumps(placed)[:300], "tools": []}
    elif exp.get("direct"):
        job = direct_tool(cell, exp["direct"]["tool"], exp["direct"]["args"])
    else:
        job = agent_order(cell, exp["order"], timeout=exp.get("timeout", 1300))
    time.sleep(2)
    after = poses(ip, names)
    snap(ip, out, "after")
    per_item = None
    if "objects" in exp:
        per_item = {n: isinstance(after[n], list) and inside(after[n], region) and after[n][2] < 0.1
                    for n, region in exp["objects"].items()}
        verdict = all(per_item.values())
    else:
        p = after[exp["object"]]
        verdict = isinstance(p, list) and bool(exp["rule"](p))
    summary = {"experiment": exp_name, "cell": cell, "order": exp.get("order"), "rule": exp["rule_text"],
               "pass": verdict, "per_item": per_item, "seconds": round(time.time() - t0, 1), "reset_ok": reset.get("ok"),
               "before": before, "after": after, "agent_status": job.get("status"), "agent_reply": job.get("reply"),
               "tools": job.get("tools"), "note": note, "dir": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(RUNS / "ledger.jsonl", "a") as f:
        f.write(json.dumps({k: summary[k] for k in ("experiment", "cell", "pass", "seconds", "agent_status", "dir")})
                + "\n")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--cell")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    run(a.exp, a.cell, a.note)
