#!/usr/bin/env python3
"""Count MJPEG frames for a few seconds: straight from OVRTX (:18210) and through the lab app with a login."""
import http.cookiejar, json, os, sys, time, urllib.request
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 6
cam = sys.argv[2] if len(sys.argv) > 2 else "overview"

def count(op, url):
    t0 = time.time(); n = 0
    try:
        r = op.open(url, timeout=10)
        buf = b""
        while time.time() - t0 < secs:
            chunk = r.read1(65536)
            if not chunk:
                break
            buf += chunk
            n += buf.count(b"Content-Type: image/jpeg")
            buf = buf[-64:]
        return dict(frames=n, fps=round(n / secs, 1), status=r.status)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}"[:160], after_s=round(time.time() - t0, 1))

plain = urllib.request.build_opener(urllib.request.ProxyHandler({}))
print("ovrtx", count(plain, f"http://127.0.0.1:18210/stream/{cam}.mjpg?duck=1"))
jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
req = urllib.request.Request("http://127.0.0.1:8095/api/auth", data=json.dumps({"password": os.environ.get("LAB_PW", "")}).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
op.open(req, timeout=5).read()
print("lab", count(op, f"http://127.0.0.1:8095/api/stream/keynote/{cam}.mjpg?duck=1"))
