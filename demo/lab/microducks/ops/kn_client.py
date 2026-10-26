#!/usr/bin/env python3
"""Test client for the keynote simulator: send an order or an action, wait for the camera check, print one JSON line.

    kn_client.py [--port 18211] order "Everyone, gather in the centre circle." [--wait 60]
    kn_client.py [--port 18211] action reset
    kn_client.py [--port 18211] health
"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=18211)
ap.add_argument("kind", choices=("order", "action", "health"))
ap.add_argument("text", nargs="?", default="")
ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the camera check")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path, timeout=5):
    with OPENER.open(BASE + path, timeout=timeout) as r:
        return json.load(r)


def post(data, timeout=70):
    req = urllib.request.Request(BASE + "/api/control", data=json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:  # noqa: BLE001
            return e.code, {}


out = dict(kind=a.kind, text=a.text, wall=time.strftime("%H:%M:%S"))
if a.kind == "health":
    out.update(get("/api/health"))
    print(json.dumps(out))
    raise SystemExit
t0 = time.time()
try:
    before = max([x["id"] for x in get("/api/info")["agent"]["feed"]] or [0])
except Exception:  # noqa: BLE001
    before = 0
if a.kind == "action":
    code, r = post({"action": a.text, "duck": 3}, timeout=15)
    out.update(code=code, message=r.get("message"), secs=round(time.time() - t0, 2))
else:
    code, r = post({"action": "order", "text": a.text, "duck": 3})
    out.update(code=code, tool=r.get("tool_call"), navigator=r.get("navigator"), message=(r.get("message") or "")[:300],
               latency_s=r.get("latency_s"), perception=r.get("perception_calls"), secs=round(time.time() - t0, 2))
    if r.get("tool_call") and not r["tool_call"].startswith(("resume_wander", "push_duck")):
        end = time.time() + a.wait
        while time.time() < end:
            time.sleep(1.0)
            try:
                feed = get("/api/info")["agent"]["feed"]
            except Exception:  # noqa: BLE001
                continue
            hit = [x for x in feed if x["id"] > before and x["kind"] == "check"]
            if hit:
                out.update(check=hit[-1]["text"], check_after_s=round(time.time() - t0, 1))
                break
        else:
            out["check"] = None
try:
    inf = get("/api/info")
    out.update(up=sum(1 for d in inf["ducks"] if d["upright"]), mode=inf.get("mode"), rtf=inf.get("rtf"),
               sim_time=inf.get("sim_time"))
except Exception as e:  # noqa: BLE001
    out["info_error"] = type(e).__name__
print(json.dumps(out))
