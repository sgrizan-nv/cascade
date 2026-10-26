#!/usr/bin/env python3
"""Booth viewers: N MJPEG streams through the lab app (login, reconnect on error) for M minutes.
Every minute one JSON line per viewer: frames received, fps, reconnects. Output: logs/viewers.jsonl"""
import http.cookiejar
import json
import os
import sys
import threading
import time
import urllib.request

MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 60
VIEWS = [("overview", 1), ("duck", 5)]
OUT = os.path.expanduser("~/keynote_night/logs/viewers.jsonl")
stats = {f"{c}{d}": dict(frames=0, reconnects=0) for c, d in VIEWS}
lock = threading.Lock()


def opener():
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
    req = urllib.request.Request("http://127.0.0.1:8095/api/auth", data=json.dumps({"password": os.environ.get("LAB_PW", "")}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    op.open(req, timeout=5).read()
    return op


def viewer(cam, duck, end):
    key = f"{cam}{duck}"
    while time.time() < end:
        try:
            op = opener()
            r = op.open(f"http://127.0.0.1:8095/api/stream/keynote/{cam}.mjpg?duck={duck}", timeout=15)
            buf = b""
            while time.time() < end:
                ch = r.read1(65536)
                if not ch:
                    break
                buf += ch
                n = buf.count(b"Content-Type: image/jpeg")
                if n:
                    with lock:
                        stats[key]["frames"] += n
                    buf = buf[buf.rfind(b"Content-Type: image/jpeg") + 10:]
        except Exception:  # noqa: BLE001
            pass
        with lock:
            stats[key]["reconnects"] += 1
        time.sleep(2)


end = time.time() + MIN * 60
for c, d in VIEWS:
    threading.Thread(target=viewer, args=(c, d, end), daemon=True).start()
last = {k: 0 for k in stats}
while time.time() < end:
    time.sleep(60)
    with lock:
        rec = {k: dict(fps=round((v["frames"] - last[k]) / 60, 1), reconnects=v["reconnects"]) for k, v in stats.items()}
        last = {k: v["frames"] for k, v in stats.items()}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(dict(wall=time.strftime("%H:%M:%S", time.gmtime()), **rec)) + "\n")
