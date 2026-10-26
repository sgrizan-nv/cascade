#!/usr/bin/env python3
"""Booth soak test of the Keynote tab through the Cascade Lab web app (the same calls the page makes).

    kn_soak.py --minutes 60 [--hard-every 0] [--out ~/keynote_night/logs/soak.jsonl]

Each round: soft reset, then the booth script (one-tap orders + typed variants, some in Spanish), waiting for every
camera check. Every step logs one JSON line: tool chosen, camera check, ducks up, camera frame, health lights.
--hard-every N: press Hard reset every N rounds and time it. Finite: stops after --minutes.
"""
import argparse
import http.cookiejar
import json
import os
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=60)
ap.add_argument("--hard-every", type=int, default=0)
ap.add_argument("--out", default=os.path.expanduser("~/keynote_night/logs/soak.jsonl"))
ap.add_argument("--base", default="http://127.0.0.1:8095")
ap.add_argument("--seed", type=int, default=int(time.time()))
a = ap.parse_args()
rng = random.Random(a.seed)
jar = http.cookiejar.CookieJar()
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
HDR = {"Content-Type": "application/json", "X-Client": "soaktest0000000000000000"}

# (order text, tools that count as right)
SCRIPT = [
    ("Everyone, gather in the centre circle.", {"gather_center", "regroup"}),
    ("Five of you, line up.", {"line_up"}),
    ("Look through the north-east camera. Is the north ramp clear?", {None}),
    ("Now walk over the north ramp in single file.", {"cross_ramp"}),
    ("Three of you, go up onto the north ramp and stay there.", {"hold_on_ramp"}),
    ("Everyone, walk around the centre circle.", {"walk_circle"}),
    ("Spin around on the spot!", {"spin"}),
    ("Stop!", {"stop_all"}),
    ("Everyone onto the ramps and freeze.", {"hold_on_ramp"}),
    ("Regroup in the centre circle.", {"regroup", "gather_center"}),
]
EXTRA = [
    ("Subid a la rampa y quedaos quietos", {"hold_on_ramp"}),
    ("dad vueltas en círculo", {"walk_circle"}),
    ("Girad sobre vosotros mismos", {"spin"}),
    ("Scatter!", {"scatter"}),
    ("How many ducks are standing?", {None}),
    ("Give duck 3 a light push", {"push_duck"}),
    ("Go back to wandering", {"resume_wander"}),
    ("Two of you go up the west ramp and wait there", {"hold_on_ramp"}),
    ("Parade around the circle clockwise, two laps", {"walk_circle"}),
    ("Everybody freeze!", {"stop_all"}),
]


def call(method, path, body=None, timeout=75):
    url = a.base + path + ("&" if "?" in path else "?") + "s=keynote"
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers=HDR)
    t0 = time.time()
    try:
        with OP.open(req, timeout=timeout) as r:
            data = r.read()
            ct = r.headers.get("Content-Type", "")
            return r.status, (json.loads(data) if "json" in ct else data), time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e), time.time() - t0
        except Exception:  # noqa: BLE001
            return e.code, {}, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"[:200]}, time.time() - t0


def login():
    pw = os.environ.get("LAB_PW", "")
    code, _, _ = call("POST", "/api/auth", {"password": pw})
    return code == 200


def feed():
    code, inf, _ = call("GET", "/api/sim/info", timeout=8)
    return (inf.get("agent") or {}).get("feed", []) if code == 200 and isinstance(inf, dict) else []


def look_state():
    out = {}
    code, h, _ = call("GET", "/api/health_full", timeout=8)
    if code == 200:
        out["lights"] = {k: (h.get(k) or {}).get("ok") for k in ("sim", "cameras", "nemotron", "lab")}
        s = h.get("sim") or {}
        out.update(up=s.get("up"), rtf=s.get("rtf"), render_ms=(h.get("cameras") or {}).get("render_ms"))
    code, img, dt = call("GET", "/api/cam/keynote/overview.jpg", timeout=10)
    out["frame_bytes"] = len(img) if code == 200 and isinstance(img, bytes) else 0
    out["frame_s"] = round(dt, 2)
    return out


def write(rec):
    rec["wall"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def order(text, want, rnd):
    before = max([x["id"] for x in feed()] or [0])
    code, r, dt = call("POST", "/api/sim", {"action": "order", "text": text, "duck": 3})
    tool = (r.get("tool_call") or "").split("(")[0] or None if isinstance(r, dict) else None
    rec = dict(round=rnd, step="order", text=text, code=code, tool=r.get("tool_call") if isinstance(r, dict) else None,
               right_tool=tool in want, secs=round(dt, 2), message=(r.get("message") or "")[:160] if isinstance(r, dict) else "")
    if tool and tool not in ("resume_wander", "push_duck"):
        end = time.time() + 120
        while time.time() < end:
            time.sleep(1.5)
            hit = [x for x in feed() if x["id"] > before and x["kind"] == "check"]
            if hit:
                txt = hit[-1]["text"]
                rec["check"] = txt
                try:
                    okn = txt.rsplit("[camera check ", 1)[1].rstrip("]").split("/")
                    rec["check_ok"], rec["check_n"] = int(okn[0]), int(okn[1])
                except Exception:  # noqa: BLE001
                    pass
                rec["check_s"] = round(time.time() - (end - 120), 1)
                break
        else:
            rec["check"] = None
    elif tool in ("push_duck", "resume_wander"):
        time.sleep(4)
    else:
        time.sleep(2)
    rec.update(look_state())
    write(rec)


def soft_reset(rnd):
    code, r, dt = call("POST", "/api/sim", {"action": "reset", "duck": 3}, timeout=15)
    rec = dict(round=rnd, step="soft_reset", code=code, secs=round(dt, 2), message=(r.get("message") or "")[:120])
    time.sleep(3)
    rec.update(look_state())
    write(rec)


def hard_reset(rnd):
    t0 = time.time()
    code, r, _ = call("POST", "/api/hard_reset", {}, timeout=30)
    rec = dict(round=rnd, step="hard_reset", code=code, message=(r.get("message") or "")[:160])
    last = None
    while time.time() - t0 < 600:
        time.sleep(4)
        c, h, _ = call("GET", "/api/health_full", timeout=8)
        if c == 200 and h.get("reset"):
            last = h["reset"]
            if last.get("done") and last.get("started", 0) >= t0 - 5:
                break
    rec.update(reset=last, secs=round(time.time() - t0, 1))
    time.sleep(5)
    rec.update(look_state())
    write(rec)


def main():
    if not login():
        write(dict(step="login", ok=False))
        return
    end = time.time() + a.minutes * 60
    rnd = 0
    while time.time() < end:
        rnd += 1
        if a.hard_every and rnd % a.hard_every == 0:
            hard_reset(rnd)
            login()
        soft_reset(rnd)
        steps = list(SCRIPT)
        for x in rng.sample(EXTRA, 3):
            steps.insert(rng.randrange(1, len(steps)), x)
        for text, want in steps:
            if time.time() > end:
                break
            order(text, want, rnd)
    write(dict(step="end", rounds=rnd))


if __name__ == "__main__":
    main()
