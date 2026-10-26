#!/usr/bin/env python3
"""Cascade Lab: password gate, scenario switcher, live cameras and one queue per robot cell.

Each scenario is an independent Cascade demo container (one GPU each). The server talks to it
only through its own interfaces: camera snapshots (8091), the OpenClaw CLI inside the container
and the Isaac bridge (8611) for stop/reset. Publishes itself on its own named ngrok endpoint.
"""
import hashlib
import hmac
import json
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STATE = HERE / "state"
PORT = int(os.environ.get("CASCADE_LAB_PORT", "8095"))
# Unset: a random password for this run, printed by main().
PASSWORD = os.environ.get("CASCADE_LAB_PASSWORD") or secrets.token_urlsafe(9)
ADMIN_PASSWORD = os.environ.get("CASCADE_LAB_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
SCENARIOS_FILE = Path(os.environ.get("CASCADE_LAB_SCENARIOS", str(HERE / "scenarios.json")))
CAMERAS = ("kitchen", "worktop", "side")
NGROK_API = "http://127.0.0.1:4040/api/endpoints"
NGROK_NAME = "cascade-lab"
NGROK_URLS = [u for u in os.environ.get("CASCADE_LAB_PUBLIC_URLS", "").split(",") if u]
AGENT_TIMEOUT = int(os.environ.get("CASCADE_LAB_AGENT_TIMEOUT", "1800"))  # a full "Sort the waste" takes ~15 min on a shared GPU
MAX_QUEUE = 8
MAX_TEXT = 280
RESERVED_MSG = "This robot is reserved for a live demo right now. You can keep watching the cameras."
OPENER = build_opener(ProxyHandler({}))

RUNTIME_IMPORT = (
    "import sys,json,os;sys.path.insert(0,next(p for p in ('/workspace/deploy/runtime','/workspace/kitchen/brev-paai')"
    " if os.path.exists(p+'/runtime.py')));import runtime;"
)
AGENT_PY = RUNTIME_IMPORT + (
    "r=runtime.openclaw(['agent','--agent','main','--session-key',sys.argv[1],'--message',sys.argv[2],"
    "'--json','--timeout',sys.argv[3]],timeout=int(sys.argv[3])+60);"
    "sys.stdout.write(json.dumps({'rc':r.returncode,'out':r.stdout,'err':r.stderr[-1500:]}))"
)
ABORT_PY = RUNTIME_IMPORT + (
    "r=runtime.openclaw(['gateway','call','chat.abort','--params',json.dumps({'sessionKey':sys.argv[1]}),'--json'],"
    "timeout=40);sys.stdout.write(json.dumps({'rc':r.returncode,'out':r.stdout[-400:],'err':r.stderr[-300:]}))"
)

STATE.mkdir(exist_ok=True)
secret_file = STATE / "cookie_secret"
if not secret_file.exists():
    secret_file.write_text(secrets.token_hex(32))
    secret_file.chmod(0o600)
SECRET = secret_file.read_text().strip().encode()
TOKEN = hmac.new(SECRET, b"cascade-lab-v1", hashlib.sha256).hexdigest()
ADMIN_TOKEN = hmac.new(SECRET, b"cascade-lab-admin-v1", hashlib.sha256).hexdigest()
SETTINGS_FILE = STATE / "settings.json"

lock = threading.Lock()
jobs = {}
login_attempts = {}
public = {"url": None}
try:
    settings = json.loads(SETTINGS_FILE.read_text())
except Exception:
    settings = {"exclusive": False}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(settings))


class Cell:
    """One robot cell = one Cascade container with its own queue."""

    def __init__(self, spec):
        self.spec = spec
        self.id = spec["id"]
        self.sim = spec.get("kind") == "sim"
        self.cams = tuple(spec.get("cams") or CAMERAS)
        self.sim_cache = (0.0, {})
        self.thumb_cache = {}
        self.container = spec.get("container", "")
        self.docker = ["sudo", "-n", "docker", "exec", self.container]
        self.queue = deque()
        self.feed = deque(maxlen=60)
        self.wake = threading.Event()
        self.current = None
        self.snap_cache = {}
        self.snap_locks = {c: threading.Lock() for c in self.cams}
        self.last_reset = 0.0
        self.ip = None
        self.ip_checked = 0.0

    def address(self):
        if self.spec.get("host"):
            return self.spec["host"]
        if time.time() - self.ip_checked > 30 or not self.ip:
            self.ip_checked = time.time()
            try:
                out = subprocess.run(["sudo", "-n", "docker", "inspect", "-f",
                                      "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", self.container],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                self.ip = out or None
            except Exception:
                self.ip = None
        return self.ip

    def bridge(self, op, timeout=10.0):
        with socket.create_connection((self.address(), 8611), timeout=timeout) as sock:
            sock.settimeout(timeout)
            f = sock.makefile("rwb")
            f.write((json.dumps({"op": op}) + "\n").encode())
            f.flush()
            line = f.readline()
        return json.loads(line or b"{}")

    def sim_get(self, path, timeout=4):
        with OPENER.open(self.spec["stream"] + path, timeout=timeout) as r:
            return r.read()

    def sim_info(self):
        if time.time() - self.sim_cache[0] > 1.0:
            self.sim_cache = (time.time(), json.loads(self.sim_get("/api/info")))
        return self.sim_cache[1]

    def sim_control(self, action, duck=None, text=None):  # ORDERBOX: typed orders go to the Microducks agent
        msg = {"action": action, "duck": duck}
        if text:
            msg["text"] = " ".join(str(text).split())[:200]
        req = Request(self.spec["stream"] + "/api/control", data=json.dumps(msg).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
        with OPENER.open(req, timeout=60 if action == "order" else 8) as r:  # KEYNOTE_TAB: look() can take a while
            return json.load(r)

    def snapshot(self, cam, duck=None, hold=None):
        if self.sim:
            return self.sim_get(f"/snapshot/{cam}.jpg?duck={int(duck or 1)}" +
                                (f"&hold={hold}&max_age=60" if hold else ""))
        with self.snap_locks[cam]:
            hit = self.snap_cache.get(cam)
            if hit and time.time() - hit[0] < 0.2:
                return hit[1]
            with OPENER.open(f"http://{self.address()}:8091/snapshot/{cam}.jpg", timeout=4) as r:
                data = r.read()
            self.snap_cache[cam] = (time.time(), data)
            return data

    def health(self):
        if self.sim:
            try:
                self.sim_info()
                return "healthy"
            except Exception:
                return "missing"
        try:
            out = subprocess.run(["sudo", "-n", "docker", "inspect", "-f", "{{.State.Health.Status}}", self.container],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            return out or "missing"
        except Exception:
            return "unknown"


# LOWDATA_ONECAM: smaller JPEGs for bad connections; one camera per cell when the operator asks for it
QUALITY = {"thumb": (240, 50), "low": (480, 55), "min": (320, 35)}
_variants = {}


def shrink_jpeg(data, quality, key):
    width, q = QUALITY.get(quality, (None, None))
    if width is None:
        return data
    digest = hashlib.sha1(data).hexdigest()[:12]
    hit = _variants.get(key)
    if hit and hit[0] == digest:
        return hit[1]
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        if img.width > width:
            img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
        out = io.BytesIO()
        img.convert("RGB").save(out, "JPEG", quality=q)
    except Exception:
        return data
    _variants[key] = (digest, out.getvalue())
    return out.getvalue()


def only_cam(cell):
    cam = (settings.get("only_cam") or {}).get(cell.id)
    return cam if cam in cell.cams else None


# HARD_RESET: Keynote tab operator resets (state written by keynote_night/ops: hard reset script + watchdog)
KN_STATE = Path(os.environ.get("KEYNOTE_STATE", "/home/ubuntu/keynote_night/state"))
KN_HARD_RESET = os.environ.get("KEYNOTE_HARD_RESET", "/home/ubuntu/keynote_night/ops/keynote_hard_reset.sh")
hard_reset_at = {"t": 0.0}
STREAMS = {}   # open MJPEG proxies: id -> (cell, cam, duck, client ip, start)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def full_health(cell):
    wd = read_json(KN_STATE / "watchdog.json", {}) or {}
    hr = read_json(KN_STATE / "hard_reset.json", None)
    if hr and not hr.get("done") and time.time() - hr.get("updated", 0) > 600:
        hr = dict(hr, done=True, ok=False, message="The last hard reset stopped reporting.")
    parts = wd.get("parts") or {}
    out = {"watchdog_age_s": round(time.time() - wd.get("at", 0), 1) if wd else None, "reset": hr,
           "viewers": sum(1 for s in list(STREAMS.values()) if s[0] == cell.id),
           "resetting": bool(hr and not hr.get("done")) or time.time() - hard_reset_at["t"] < 8}
    for name in ("sim", "cameras", "nemotron", "lab"):
        p = parts.get(name) or {}
        out[name] = {"ok": p.get("ok"), "detail": p.get("detail", "not checked yet")}
    try:
        inf = cell.sim_info()
        out["sim"].update(up=sum(1 for d in inf.get("ducks", []) if d.get("upright")), rtf=inf.get("rtf"),
                          mode=inf.get("mode"), sim_time=inf.get("sim_time"),
                          busy=bool((inf.get("agent") or {}).get("busy")))
        st = inf.get("stats") or {}
        out["cameras"]["render_ms"] = st.get("render_ms")
    except Exception:
        out["sim"]["live"] = False
    if wd and out["watchdog_age_s"] is not None and out["watchdog_age_s"] > 90:
        for name in ("sim", "cameras", "nemotron", "lab"):
            out[name]["ok"] = None
    return out


def start_hard_reset(with_nemotron=False):
    hr = read_json(KN_STATE / "hard_reset.json", None)
    if (hr and not hr.get("done") and time.time() - hr.get("updated", 0) < 600) or time.time() - hard_reset_at["t"] < 8:
        return False, "A hard reset is already running."
    hard_reset_at["t"] = time.time()
    cmd = ["sudo", "-n", "systemd-run", "--unit", f"keynote-hard-reset-{int(time.time())}", "--collect",
           "/bin/bash", KN_HARD_RESET] + (["--nemotron"] if with_nemotron else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        hard_reset_at["t"] = 0.0
        return False, "Could not start the hard reset (" + (r.stderr or "")[-120:].strip() + ")"
    return True, "Hard reset started: simulation, cameras and AI restart. Back in about 2 minutes."


SCENARIOS = json.loads(SCENARIOS_FILE.read_text())
CELLS = {s["id"]: Cell(s) for s in SCENARIOS}
DEFAULT_CELL = SCENARIOS[0]["id"]
health_cache = {}


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def run_agent(cell, job):
    cmd = cell.docker + ["/opt/paai/venv/bin/python", "-c", AGENT_PY, job["session_key"], job["text"],
                         str(AGENT_TIMEOUT)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        job["proc"] = proc
        stdout, _ = proc.communicate(timeout=AGENT_TIMEOUT + 90)
        wrapper = json.loads(stdout or "{}")
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, [], "The robot took too long to answer."
    except Exception as e:
        return None, [], "Could not reach the robot (" + type(e).__name__ + ")."
    finally:
        job.pop("proc", None)
    out = wrapper.get("out") or ""
    try:
        data = json.loads(out[out.index("{"):])
    except Exception:
        return None, [], "The robot answered in an unexpected format."
    reply = find_key(data, "finalAssistantVisibleText")
    if not reply:
        payloads = find_key(data, "payloads") or []
        reply = "\n".join(p.get("text", "") for p in payloads if isinstance(p, dict)).strip()
    summary = find_key(data, "toolSummary") or {}
    tools = [t.replace("cascade__", "") for t in summary.get("tools", [])]
    if wrapper.get("rc") not in (0, None) and not reply:
        return None, tools, "The robot could not finish this request."
    return reply or "(no answer)", tools, None


def abort_running(cell, job):
    try:
        log(cell.id + " cancel-stop " + json.dumps(cell.bridge("stop")))
    except Exception as e:
        log(cell.id + " cancel-stop failed " + type(e).__name__)
    try:
        p = subprocess.run(cell.docker + ["/opt/paai/venv/bin/python", "-c", ABORT_PY, job["session_key"]],
                           capture_output=True, text=True, timeout=60)
        log(cell.id + " chat.abort " + (p.stdout or "")[:300])
    except Exception as e:
        log(cell.id + " chat.abort failed " + type(e).__name__)
    for _ in range(20):
        if not job.get("proc"):
            return
        time.sleep(0.5)
    proc = job.get("proc")
    if proc:
        proc.kill()


def cancel_job(cell, job, reason):
    if job["status"] == "queued":
        if job["id"] in cell.queue:
            cell.queue.remove(job["id"])
        job.update(status="cancelled", reply=reason, finished=time.time())
        return True
    if job["status"] == "running":
        job.update(status="cancelled", reply=reason)
        threading.Thread(target=abort_running, args=(cell, job), daemon=True).start()
        return True
    return False


def worker(cell):
    while True:
        cell.wake.wait(5)
        cell.wake.clear()
        while True:
            with lock:
                if not cell.queue:
                    break
                job = jobs[cell.queue.popleft()]
                job["status"] = "running"
                job["started"] = time.time()
                cell.current = job["id"]
            log(cell.id + " run " + job["id"] + " " + job["text"][:80])
            reply, tools, error = run_agent(cell, job)
            with lock:
                job["finished"] = time.time()
                job["secs"] = round(job["finished"] - job["started"], 1)
                job["tools"] = tools
                if job["status"] != "cancelled":
                    job["status"], job["reply"] = ("error", error) if error else ("done", reply)
                cell.current = None
            log(cell.id + " done " + job["id"] + " " + job["status"] + " " + str(job["secs"]) + "s")
            with open(STATE / "jobs.jsonl", "a") as f:
                f.write(json.dumps({k: job.get(k) for k in ("id", "cell", "text", "status", "reply", "tools",
                                                            "secs", "created", "admin")}) + "\n")


def public_job(job, client, admin=False):
    cell = CELLS[job["cell"]]
    position = None
    if job["status"] == "queued" and job["id"] in cell.queue:
        position = list(cell.queue).index(job["id"]) + 1
    mine = job["client"] == client
    return {"id": job["id"], "cell": job["cell"], "text": job["text"], "status": job["status"],
            "reply": job.get("reply"), "tools": job.get("tools", []), "secs": job.get("secs"),
            "created": job["created"], "started": job.get("started"), "mine": mine, "position": position,
            "by_admin": job["admin"],
            "can_cancel": job["status"] in ("queued", "running") and (admin or (mine and job["status"] == "queued"))}


def ngrok_api(method, data=None):
    req = Request(NGROK_API, data=None if data is None else json.dumps(data).encode(), method=method,
                  headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=10) as r:
        return json.load(r)


def reconcile_ngrok():
    rows = ngrok_api("GET")["endpoints"]
    owned = [r for r in rows if r.get("name") == NGROK_NAME]
    if owned:
        public["url"] = owned[0].get("url")
        return
    for url in NGROK_URLS + [None]:
        config = {"name": NGROK_NAME, "upstream": {"url": f"http://127.0.0.1:{PORT}"},
                  "inspect": False, "pooling_enabled": False}
        if url:
            config["url"] = url
        try:
            row = ngrok_api("POST", config)
            public["url"] = row.get("url") or url
            (STATE / "public_url.txt").write_text((public["url"] or "") + "\n")
            log("ngrok endpoint " + str(public["url"]))
            return
        except Exception as e:
            log("ngrok create failed for " + str(url) + ": " + type(e).__name__)


def ngrok_loop(stop):
    while not stop.is_set():
        try:
            reconcile_ngrok()
        except Exception as e:
            log("ngrok reconcile retry: " + type(e).__name__)
        stop.wait(15)


def health_loop(stop):
    while not stop.is_set():
        for cid, cell in CELLS.items():
            health_cache[cid] = cell.health()
        stop.wait(10)


class Handler(BaseHTTPRequestHandler):
    server_version = "CascadeLab/1"

    def log_message(self, *_):
        pass

    def client_ip(self):
        return (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()

    def role(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "cascade_lab_auth":
                if hmac.compare_digest(v, ADMIN_TOKEN):
                    return "admin"
                if hmac.compare_digest(v, TOKEN):
                    return "guest"
        return None

    def cookie(self, value, max_age):
        c = f"cascade_lab_auth={value}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax"
        if self.headers.get("X-Forwarded-Proto") == "https":
            c += "; Secure"
        return c

    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def static(self, name, ctype):
        self.send(200, (STATIC / name).read_bytes(), ctype)

    def body(self):
        n = min(int(self.headers.get("Content-Length") or 0), 8192)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def cell(self, query, data=None):
        cid = ((data or {}).get("s") or (query.get("s") or [DEFAULT_CELL])[0])
        return CELLS.get(cid) or CELLS[DEFAULT_CELL]

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        client = (self.headers.get("X-Client") or "")[:40]
        role = self.role()
        admin = role == "admin"
        if path == "/healthz":
            return self.send(200, {"ok": True, "cells": health_cache})
        if path in ("/icon.svg", "/manifest.webmanifest"):
            return self.static(path[1:], "image/svg+xml" if path.endswith("svg") else "application/manifest+json")
        if path == "/":
            return self.static("app.html" if role else "login.html", "text/html; charset=utf-8")
        if not role:
            return self.send(401, {"error": "login"})
        if path == "/api/scenarios":
            return self.send(200, {"default": DEFAULT_CELL, "items": [
                {**{k: v for k, v in s.items() if k not in ("container", "host")},
                 "health": health_cache.get(s["id"], "unknown")} for s in SCENARIOS]})
        if path.startswith("/api/cam/") and path.endswith(".jpg"):
            parts = path[len("/api/cam/"):-4].split("/")
            cid, cam = (parts[0], parts[1]) if len(parts) == 2 else (DEFAULT_CELL, parts[0])
            cell = CELLS.get(cid)
            if not cell or cam not in cell.cams:
                return self.send(404, {"error": "camera"})
            cam = only_cam(cell) or cam
            try:
                q = (query.get("q") or [""])[0]
                duck = (query.get("duck") or [1])[0]
                # thumbnails (and pages cached from before 22-10, whose thumbnails send no q): one render per camera
                # every 20 s, not per viewer; the live view streams, or asks for q=full / low / min
                if cell.sim and q in ("thumb", ""):
                    q = "thumb"
                    key = (cam, str(duck))
                    hit = cell.thumb_cache.get(key)
                    if hit and time.time() - hit[0] < 20.0:
                        return self.send(200, hit[1], "image/jpeg")
                    data = shrink_jpeg(cell.snapshot(cam, duck, hold=0.2), q, (cell.id, cam, q))
                    cell.thumb_cache[key] = (time.time(), data)
                    return self.send(200, data, "image/jpeg")
                data = cell.snapshot(cam, duck)
                return self.send(200, shrink_jpeg(data, q, (cell.id, cam, q)), "image/jpeg")
            except Exception:
                return self.send(503, {"error": "camera offline"})
        if path.startswith("/api/stream/") and path.endswith(".mjpg"):
            parts = path[len("/api/stream/"):-5].split("/")
            cell = CELLS.get(parts[0]) if len(parts) == 2 else None
            if not cell or not cell.sim or parts[1] not in cell.cams:
                return self.send(404, {"error": "stream"})
            return self.proxy_stream(cell, only_cam(cell) or parts[1], (query.get("duck") or [1])[0])
        if path == "/api/sim/info":
            cell = self.cell(query)
            if not cell.sim:
                return self.send(404, {"error": "not a sim cell"})
            try:
                return self.send(200, cell.sim_info())
            except Exception:
                return self.send(503, {"error": "simulation offline"})
        if path == "/api/health_full":
            cell = self.cell(query)
            if not cell.spec.get("hard_reset"):
                return self.send(404, {"error": "no health panel for this cell"})
            return self.send(200, full_health(cell))
        if path == "/api/me":
            return self.send(200, {"role": role, "exclusive": settings.get("exclusive", False)})
        if path == "/api/status":
            cell = self.cell(query)
            if cell.sim:
                out = {"cell": cell.id, "health": health_cache.get(cell.id, "unknown"), "robot": "idle", "role": role,
                       "exclusive": settings.get("exclusive", False), "current": None, "queue": 0,
                       "cameras": list(cell.cams), "now": time.time(), "only_cam": only_cam(cell)}
                try:
                    inf = cell.sim_info()
                    up = sum(1 for d in inf.get("ducks", []) if d.get("upright"))
                    mode = {"wander": "wandering", "ramps": "heading for the ramps", "stop": "standing still"}.get(
                        inf.get("mode"), inf.get("mode"))
                    out["sim_title"] = f"{len(inf.get('ducks', []))} ducks live \u00b7 {up} on their feet"
                    out["sim_sub"] = f"Everyone is {mode} \u00b7 sim time {int(inf.get('sim_time', 0))} s"
                except Exception:
                    out["sim_title"], out["sim_sub"] = "Simulation reconnecting", "Try again in a few seconds"
                return self.send(200, out)
            with lock:
                job = jobs.get(cell.current) if cell.current else None
                return self.send(200, {
                    "cell": cell.id, "health": health_cache.get(cell.id, "unknown"),
                    "robot": "busy" if job else "idle", "role": role,
                    "exclusive": settings.get("exclusive", False),
                    "current": public_job(job, client, admin) if job else None,
                    "queue": len(cell.queue), "cameras": list(CAMERAS), "now": time.time(), "only_cam": only_cam(cell)})
        if path == "/api/feed":
            cell = self.cell(query)
            with lock:
                return self.send(200, {"items": [public_job(jobs[i], client, admin) for i in cell.feed if i in jobs]})
        if path.startswith("/api/job/"):
            with lock:
                job = jobs.get(path[len("/api/job/"):])
                return self.send(200, public_job(job, client, admin)) if job else self.send(404, {"error": "job"})
        self.send(404, {"error": "not found"})

    def proxy_stream(self, cell, cam, duck):
        try:
            up = OPENER.open(cell.spec["stream"] + f"/stream/{cam}.mjpg?duck={int(duck)}", timeout=10)
        except Exception:
            return self.send(503, {"error": "stream offline"})
        t0 = time.time()
        STREAMS[id(self)] = (cell.id, cam, int(duck), self.client_ip(), t0)
        log(f"stream {cell.id}/{cam} duck {duck} from {self.client_ip()} open ({len(STREAMS)} open)")
        try:
            return self._pump(up)
        finally:
            STREAMS.pop(id(self), None)
            log(f"stream {cell.id}/{cam} duck {duck} from {self.client_ip()} closed after {time.time() - t0:.0f} s")

    def _pump(self, up):
        try:
            self.send_response(200)
            self.send_header("Content-Type", up.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            while True:
                chunk = up.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            pass
        finally:
            up.close()

    def do_POST(self):
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        client = (self.headers.get("X-Client") or "")[:40] or "anon"
        if path == "/api/auth":
            ip = self.client_ip()
            now = time.time()
            hist = [t for t in login_attempts.get(ip, []) if now - t < 300]
            if len(hist) >= 15:
                return self.send(429, {"retry_after": int(300 - (now - hist[0]))})
            hist.append(now)
            login_attempts[ip] = hist
            pw = (self.body().get("password") or "").strip().lower()
            if pw == ADMIN_PASSWORD.lower():
                log("admin login from " + ip)
                return self.send(200, {"ok": True, "role": "admin"},
                                 extra={"Set-Cookie": self.cookie(ADMIN_TOKEN, 2592000)})
            if pw == PASSWORD.lower():
                return self.send(200, {"ok": True, "role": "guest"}, extra={"Set-Cookie": self.cookie(TOKEN, 2592000)})
            return self.send(403, {"ok": False})
        if path == "/api/logout":
            return self.send(200, {"ok": True}, extra={"Set-Cookie": self.cookie("", 0)})
        role = self.role()
        if not role:
            return self.send(401, {"error": "login"})
        admin = role == "admin"
        data = self.body()
        cell = self.cell(query, data)
        if cell.sim and path in ("/api/sim", "/api/stop", "/api/reset"):
            if settings.get("exclusive") and not admin:
                return self.send(423, {"ok": False, "message": RESERVED_MSG})
            action = {"/api/stop": "stop", "/api/reset": "reset"}.get(path) or str(data.get("action") or "")
            try:
                r = cell.sim_control(action, data.get("duck"), data.get("text") if action == "order" else None)
                cell.sim_cache = (0.0, {})
                log(cell.id + " SIM " + action + " " + str(data.get("duck")) +
                    (" " + json.dumps(str(data.get("text"))[:200]) + " -> " + str(r.get("tool_call")) if action == "order" else ""))
                return self.send(200, r)
            except HTTPError as e:
                try:
                    return self.send(200, json.load(e))
                except Exception:
                    return self.send(503, {"ok": False, "message": "Simulation offline: HTTP " + str(e.code)})
            except Exception as e:
                return self.send(503, {"ok": False, "message": "Simulation offline: " + type(e).__name__})
        if path == "/api/hard_reset":
            if not cell.spec.get("hard_reset"):
                return self.send(404, {"ok": False, "message": "This cell has no hard reset."})
            if settings.get("exclusive") and not admin:
                return self.send(423, {"ok": False, "message": RESERVED_MSG})
            ok, msg = start_hard_reset(bool(data.get("nemotron")) and admin)
            log(cell.id + " HARD RESET " + ("started" if ok else "refused: " + msg))
            cell.sim_cache = (0.0, {})
            return self.send(200 if ok else 409, {"ok": ok, "message": msg})
        if cell.sim and path == "/api/chat":
            return self.send(400, {"error": "The ducks take one-tap commands only (buttons above)."})
        if path == "/api/chat":
            text = " ".join(str(data.get("text") or "").split())[:MAX_TEXT]
            if not text:
                return self.send(400, {"error": "Write something for the robot first."})
            with lock:
                if settings.get("exclusive") and not admin:
                    return self.send(423, {"error": RESERVED_MSG})
                if not admin and any(jobs[i]["client"] == client for i in cell.queue):
                    return self.send(429, {"error": "You already have a request waiting for this robot."})
                if not admin and len(cell.queue) >= MAX_QUEUE:
                    return self.send(429, {"error": "This robot is busy. Try again in a minute."})
                jid = uuid.uuid4().hex[:12]
                jobs[jid] = {"id": jid, "cell": cell.id, "client": client, "text": text, "status": "queued",
                             "created": time.time(), "admin": admin,
                             "session_key": "agent:main:lab-" + cell.id + "-" + client[:20]}
                if admin:
                    pos = sum(1 for i in cell.queue if jobs[i]["admin"])
                    cell.queue.insert(pos, jid)
                else:
                    cell.queue.append(jid)
                cell.feed.appendleft(jid)
                job = public_job(jobs[jid], client, admin)
            cell.wake.set()
            log(cell.id + " queued " + jid + (" [admin] " if admin else " ") + text[:80])
            return self.send(200, job)
        if path == "/api/cancel":
            jid = str(data.get("id") or "")
            with lock:
                job = jobs.get(jid)
                if not job:
                    return self.send(404, {"error": "Request not found."})
                if not admin and not (job["client"] == client and job["status"] == "queued"):
                    return self.send(403, {"error": "Only the admin can cancel this request."})
                ok = cancel_job(CELLS[job["cell"]], job, "Cancelled by the operator." if admin and
                                job["client"] != client else "Cancelled.")
            return self.send(200, {"ok": ok, "message": "Request cancelled." if ok else "Nothing to cancel."})
        if path == "/api/admin/only_cam":
            if not admin:
                return self.send(403, {"error": "Admin only."})
            cam = str(data.get("cam") or "") or None
            if cam is not None and cam not in cell.cams:
                return self.send(400, {"error": "Unknown camera."})
            with lock:
                oc = dict(settings.get("only_cam") or {})
                if cam:
                    oc[cell.id] = cam
                else:
                    oc.pop(cell.id, None)
                settings["only_cam"] = oc
                save_settings()
            log(f"{cell.id} only_cam={cam}")
            label = (cell.spec.get("cam_labels") or {}).get(cam, cam)
            return self.send(200, {"ok": True, "only_cam": cam, "message": (
                f"Only the {label} camera renders now." if cam else "All cameras render again.")})
        if path == "/api/admin/exclusive":
            if not admin:
                return self.send(403, {"error": "Admin only."})
            on = bool(data.get("on"))
            cancelled = 0
            with lock:
                settings["exclusive"] = on
                save_settings()
                if on:
                    for c in CELLS.values():
                        for jid in list(c.queue) + ([c.current] if c.current else []):
                            job = jobs.get(jid)
                            if job and not job["admin"] and cancel_job(c, job, "Cancelled: reserved for a live demo."):
                                cancelled += 1
            return self.send(200, {"ok": True, "exclusive": on, "cancelled": cancelled,
                                   "message": ("Robots reserved for you." if on else "Robots open to everyone again.")})
        if path == "/api/admin/clear":
            if not admin:
                return self.send(403, {"error": "Admin only."})
            with lock:
                n = sum(1 for jid in list(cell.queue) if not jobs[jid]["admin"]
                        and cancel_job(cell, jobs[jid], "Cancelled by the operator."))
            return self.send(200, {"ok": True, "message": f"{n} waiting request(s) cancelled."})
        if path == "/api/stop":
            if settings.get("exclusive") and not admin:
                return self.send(423, {"ok": False, "message": RESERVED_MSG})
            try:
                r = cell.bridge("stop")
                log(cell.id + " STOP " + json.dumps(r))
                return self.send(200, {"ok": bool(r.get("ok")), "message": "Robot stopped."})
            except Exception as e:
                return self.send(503, {"ok": False, "message": "Stop failed: " + type(e).__name__})
        if path == "/api/reset":
            if settings.get("exclusive") and not admin:
                return self.send(423, {"ok": False, "message": RESERVED_MSG})
            with lock:
                busy = cell.current is not None
            if busy:
                return self.send(409, {"ok": False, "message": "Wait until the robot finishes its task."})
            if time.time() - cell.last_reset < 20:
                return self.send(429, {"ok": False, "message": "This cell was just reset."})
            cell.last_reset = time.time()
            try:
                r = cell.bridge("reset_props", timeout=75)
                log(cell.id + " RESET " + json.dumps(r)[:200])
                return self.send(200, {"ok": bool(r.get("ok")), "message": cell.spec.get("reset_message", "Scene reset.")
                                       if r.get("ok") else "Reset did not complete."})
            except Exception as e:
                return self.send(503, {"ok": False, "message": "Reset failed: " + type(e).__name__})
        self.send(404, {"error": "not found"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        pass


def main():
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    server = Server(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for cell in CELLS.values():
        threading.Thread(target=worker, args=(cell,), daemon=True).start()
    threading.Thread(target=health_loop, args=(stop,), daemon=True).start()
    if os.environ.get("CASCADE_NGROK", "1") == "1":
        threading.Thread(target=ngrok_loop, args=(stop,), daemon=True).start()
    log(f"Cascade Lab on 127.0.0.1:{PORT} cells={list(CELLS)}")
    for var, value in (("CASCADE_LAB_PASSWORD", PASSWORD), ("CASCADE_LAB_ADMIN_PASSWORD", ADMIN_PASSWORD)):
        if not os.environ.get(var):
            log(f"{var} not set, generated for this run: {value}")
    stop.wait()
    server.shutdown()


if __name__ == "__main__":
    main()
