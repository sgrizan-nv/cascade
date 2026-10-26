#!/usr/bin/env python3
"""Watchdog for the Keynote tab (runs as root, systemd keynote-watchdog.service).

Every 15 s: simulation (:18211/api/health, sim steps must advance), OVRTX cameras (:18210 snapshot), Nemotron
(:18310, a 2-token chat every minute) and the lab app (:8095/healthz). A part that stays broken is restarted:
simulation and cameras after ~45 s, the lab app after ~45 s, Nemotron after ~3 min. A service is never restarted
while it is still starting, while a hard reset runs, or more than 4 times an hour. Status for the web page in
state/watchdog.json; every restart goes to Telegram.
"""
import fcntl
import json
import os
import subprocess
import time
import urllib.request

K = "/home/ubuntu/keynote_night"
STATE = f"{K}/state"
LOG = f"{K}/logs/watchdog.log"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PERIOD = 15
os.makedirs(STATE, exist_ok=True)
os.makedirs(f"{K}/logs", exist_ok=True)


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg + "\n")


def tg(msg):
    try:
        subprocess.run(["python3", "/home/ubuntu/cascade-lab/ops/tg_send_local.py", msg], timeout=40,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def get(url, timeout=5):
    t0 = time.time()
    with OPENER.open(url, timeout=timeout) as r:
        body = r.read()
    return body, time.time() - t0


def jget(url, timeout=5):
    return json.loads(get(url, timeout)[0])


def nemo_chat():
    body = json.dumps({"model": "nemotron-3.5-bolt", "messages": [{"role": "user", "content": "Say OK."}],
                       "max_tokens": 2, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request("http://127.0.0.1:18310/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with OPENER.open(req, timeout=30) as r:
        json.load(r)
    return time.time() - t0


def active_since(unit):
    """Seconds since the unit (re)started; None when it is not active."""
    try:
        out = subprocess.run(["systemctl", "show", unit, "-p", "ActiveState", "-p", "ActiveEnterTimestampMonotonic"],
                             capture_output=True, text=True, timeout=10).stdout
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        if kv.get("ActiveState") != "active":
            return None
        mono = int(kv.get("ActiveEnterTimestampMonotonic", "0")) / 1e6
        return time.monotonic() - mono
    except Exception:  # noqa: BLE001
        return None


def hard_reset_running():
    try:
        with open(f"{STATE}/hard_reset.lock", "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                return False
            except OSError:
                return True
    except OSError:
        return False


class Part:
    def __init__(self, name, unit, label, fails_needed, grace_s):
        self.name, self.unit, self.label = name, unit, label
        self.fails_needed, self.grace_s = fails_needed, grace_s
        self.fails = 0
        self.restarts = []
        self.last = dict(ok=None, detail="not checked yet")

    def result(self, ok, detail, extra=None):
        extra = {k: v for k, v in (extra or {}).items() if k not in ("ok", "detail", "at")}
        self.last = dict(extra, ok=ok, detail=detail, at=int(time.time()))
        self.fails = 0 if ok else self.fails + 1

    def maybe_restart(self):
        if self.fails < self.fails_needed:
            return
        up = active_since(self.unit)
        if up is not None and up < self.grace_s:
            return                                   # still starting
        now = time.time()
        self.restarts = [t for t in self.restarts if now - t < 3600]
        if len(self.restarts) >= 4:
            if self.fails == self.fails_needed:
                log(f"{self.name}: broken but restarted 4 times this hour; leaving it")
                tg(f"Keynote watchdog: {self.label} is still broken after 4 restarts this hour. Use Hard reset or check it.")
            return
        if os.environ.get("WATCHDOG_DRY"):
            log(f"{self.name}: DRY RUN, would restart {self.unit} ({self.last.get('detail')})")
            self.fails = 0
            return
        log(f"{self.name}: restarting {self.unit} after {self.fails} failed checks ({self.last.get('detail')})")
        subprocess.run(["systemctl", "restart", self.unit], timeout=120)
        self.restarts.append(now)
        self.fails = 0
        tg(f"Keynote watchdog: restarted {self.label} ({self.last.get('detail')}).")


sim = Part("sim", "microducks-keynote-sim", "the simulation", 3, 240)
cam = Part("cameras", "microducks-keynote-ovrtx", "the OVRTX cameras", 3, 150)
nemo = Part("nemotron", "microducks-nemotron-vllm", "Nemotron", 4, 420)
lab = Part("lab", "cascade-lab-app", "the Cascade Lab web app", 3, 30)
PARTS = [sim, cam, nemo, lab]


def check_sim(prev):
    try:
        h = jget("http://127.0.0.1:18211/api/health", 5)
    except Exception as e:  # noqa: BLE001
        sim.result(False, f"no answer ({type(e).__name__})")
        return prev
    seq = h.get("seq", 0)
    if prev is not None and seq == prev:
        sim.result(False, "simulation steps stopped", extra=h)
    elif not h.get("map_ready"):
        sim.result(False, "cameras still mapping the yard", extra=h)
    else:
        sim.result(True, f"{h.get('up')}/12 ducks up, {h.get('rtf')}x real time", extra=h)
    return seq


def check_cam():
    try:
        body, dt = get("http://127.0.0.1:18210/snapshot/overview.jpg", 12)
        h = {}
        try:
            h = jget("http://127.0.0.1:18210/api/health", 5)
        except Exception:  # noqa: BLE001
            pass
        if len(body) < 5000:
            cam.result(False, "empty camera frame")
        elif (h.get("pose_age") or 0) > 20 and sim.last.get("ok"):
            cam.result(False, f"frozen ducks (poses {h['pose_age']:.0f} s old)", extra=h)
        else:
            cam.result(True, f"snapshot in {dt:.2f} s, render {h.get('render_ms')} ms", extra=dict(h, snap_s=round(dt, 2)))
    except Exception as e:  # noqa: BLE001
        cam.result(False, f"no frame ({type(e).__name__})")


def check_nemo(n):
    try:
        get("http://127.0.0.1:18310/v1/models", 10)
        if n % 4 == 0:
            dt = nemo_chat()
            nemo.result(True, f"answers in {dt:.2f} s", extra=dict(latency_s=round(dt, 2)))
        else:
            nemo.last.update(at=int(time.time()))
            nemo.fails = 0
    except Exception as e:  # noqa: BLE001
        nemo.result(False, f"no answer ({type(e).__name__})")


def check_lab():
    try:
        get("http://127.0.0.1:8095/healthz", 5)
        lab.result(True, "web app answers")
    except Exception as e:  # noqa: BLE001
        lab.result(False, f"no answer ({type(e).__name__})")


def main():
    log("watchdog started")
    prev_seq, n = None, 0
    while True:
        t0 = time.time()
        busy = hard_reset_running()
        for fn, arg in ((check_sim, prev_seq), (check_cam, None), (check_nemo, n), (check_lab, None)):
            try:
                r = fn(arg) if arg is not None or fn is check_sim else fn()
                if fn is check_sim:
                    prev_seq = r
            except Exception as e:  # noqa: BLE001
                log(f"check {fn.__name__} crashed: {type(e).__name__}: {e}")
        if not busy:
            for p in PARTS:
                try:
                    p.maybe_restart()
                except Exception as e:  # noqa: BLE001
                    log(f"{p.name}: restart failed {type(e).__name__}: {e}")
        rec = dict(at=int(time.time()), hard_reset_running=busy,
                   parts={p.name: dict(p.last, fails=p.fails, restarts_last_hour=len(
                       [t for t in p.restarts if time.time() - t < 3600])) for p in PARTS})
        tmp = f"{STATE}/watchdog.json.tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.chmod(tmp, 0o644)
        os.replace(tmp, f"{STATE}/watchdog.json")
        n += 1
        time.sleep(max(1.0, PERIOD - (time.time() - t0)))


if __name__ == "__main__":
    main()
