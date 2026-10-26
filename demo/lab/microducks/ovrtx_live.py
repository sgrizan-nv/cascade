"""[OVRTX] Live camera server for the Microducks schoolyard.

Poses come from a pose source and are written straight into OVRTX (omni:xform world matrices on the 12 x 15 duck
bodies); only the views somebody requested are rendered. Same HTTP API as the previous live server:
  /snapshot/<cam>.jpg?duck=N   /stream/<cam>.mjpg?duck=N   /api/info   POST /api/control   /  (test page)

Pose sources:
  --source shm     /dev/shm/microducks_pose.bin written by isaac_ducks_sim.py --live (controls go to its :18201)
  --source trace   replays a recorded trace.npz in a loop (testing only)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "assets"))

import ovrtx_scene as SC  # noqa: E402
import yard_layout as Y  # noqa: E402
from kitfilm import PosedDuck  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=("shm", "trace"), default="shm")
ap.add_argument("--trace", default="")
ap.add_argument("--shm", default="/dev/shm/microducks_pose.bin")
ap.add_argument("--sim-url", default="http://127.0.0.1:18201")
ap.add_argument("--port", type=int, default=18200)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--fps", type=float, default=15.0, help="max refresh per streamed view")
ap.add_argument("--updates", type=int, default=1)
ap.add_argument("--quality", type=int, default=80)
ap.add_argument("--mode", default="RealTimePathTracing")
ap.add_argument("--asset", default=str(HERE / "assets" / "microduck_allcollisions.usd"))
ap.add_argument("--tex", default=str(HERE / "textures"))
ap.add_argument("--sky", default=str(HERE / "assets" / "sky_daylift_v2.hdr"))
ap.add_argument("--fonts", default=str(HERE.parent / "fonts"))
ap.add_argument("--look", choices=("v1", "v3"), default="v3")
ap.add_argument("--ph", default=str(HERE / "assets" / "assets_ph"))
args = ap.parse_args()

CAMS = [("overview", "Overview"), ("ramp", "Ramp close-up"), ("orbit", "Orbit"), ("chase", "Follow duck"),
        ("duck", "Duck camera")]
OPENER = build_opener(ProxyHandler({}))
BODY_NAMES = ['trunk_base', 'bearing_roll', 'hip_l_2', 'upper_leg_right', 'leg_2', 'ankle_right', 'neck',
              'neck_pitch_1', 'yaw_roll_motion', 'jaw_soft', 'yaw2roll', 'hip_l', 'upper_leg_left', 'leg', 'ankle_left']
NB = len(BODY_NAMES)
HDR = 16  # float64 header: seq, sim_time, then reserved


# ----------------------------------------------------------------------------- pose sources
class TraceSource:
    def __init__(self, path):
        d = np.load(Path(path) / "trace.npz")
        self.pose = d["body_pose_xyzw"].astype(np.float64)
        names = [str(n) for n in d["body_names"]]
        order = [names.index(n) for n in BODY_NAMES]
        self.pose = self.pose[:, :, order]
        self.dt = float(d["ctrl_dt"])
        self.t0 = time.time()
        self.selected = 2
        self.info_extra = dict(source="trace replay (test)")

    def latest(self):
        t = (time.time() - self.t0) % (len(self.pose) * self.dt)
        return t, self.pose[int(t / self.dt)]

    def info(self):
        t, p = self.latest()
        return dict(sim_time=round(t, 1), mode="replay", stack={}, ducks=[
            dict(id=i + 1, x=round(float(p[i, 0, 0]), 2), y=round(float(p[i, 0, 1]), 2), upright=True)
            for i in range(Y.NDUCK)], **self.info_extra)

    def control(self, data):
        if data.get("duck") is not None:
            self.selected = int(np.clip(int(data["duck"]) - 1, 0, Y.NDUCK - 1))
        return dict(ok=True, message="Test replay: controls are ignored.", selected=self.selected + 1)


class ShmSource:
    def __init__(self, path, url):
        self.path, self.url = path, url
        self.mm = None
        self.selected = 2
        self._info = {}
        self._info_t = 0.0

    def _map(self):
        if self.mm is None and os.path.exists(self.path):
            self.mm = np.memmap(self.path, dtype=np.float64, mode="r", shape=(HDR + Y.NDUCK * NB * 7,))
        return self.mm

    def latest(self):
        mm = self._map()
        if mm is None:
            return 0.0, None
        for _ in range(3):
            s0 = mm[0]
            body = np.array(mm[HDR:]).reshape(Y.NDUCK, NB, 7)
            t = float(mm[1])
            if mm[0] == s0:
                return t, body
        return t, body

    def info(self):
        if time.time() - self._info_t > 0.5:
            try:
                with OPENER.open(self.url + "/api/info", timeout=2) as r:
                    self._info = json.load(r)
                    self.selected = int(self._info.get("selected", self.selected + 1)) - 1
            except Exception:  # noqa: BLE001
                self._info = dict(error="simulation offline")
            self._info_t = time.time()
        return dict(self._info)

    def control(self, data):
        req = Request(self.url + "/api/control", data=json.dumps(data).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
        with OPENER.open(req, timeout=8) as r:
            out = json.load(r)
        self.selected = int(out.get("selected", self.selected + 1)) - 1
        self._info_t = 0.0
        return out


# ----------------------------------------------------------------------------- renderer
class LiveRenderer:
    def __init__(self, source):
        from pxr import Gf, Usd, UsdGeom
        self.src = source
        work = Path(tempfile.mkdtemp(prefix="microducks_ovrtx_"))
        scene = work / "scene.usda"
        stage = Usd.Stage.CreateNew(str(scene))
        stage.GetRootLayer().subLayerPaths.append("render_product.usda")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        if args.look == "v1":
            SC.build_yard(stage, args.tex, args.sky)
        else:
            SC.build_yard_v2(stage, args.tex, args.sky, args.ph, dome_intensity=380.0, sun_intensity=2500.0, v3=True,
                             sun_from=(3.0, -4.2, 2.9))
        self.ducks = [PosedDuck(stage, f"/World/Ducks/Duck{i:02d}", args.asset, BODY_NAMES) for i in range(Y.NDUCK)]
        SC.color_heads(stage, self.ducks)
        self.head_cams = [SC.head_camera(stage, d, f"HeadCam{i:02d}") for i, d in enumerate(self.ducks)]
        self.free_cams = {}
        for c in ("overview", "ramp", "orbit", "chase"):
            cam = UsdGeom.Camera.Define(stage, f"/World/Cams/{c}")
            cam.CreateHorizontalApertureAttr(0.36)
            cam.CreateVerticalApertureAttr(0.36 * args.height / args.width)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
            cam.CreateFocalLengthAttr({"overview": 0.22, "ramp": 0.24, "orbit": 0.24, "chase": 0.28}[c])
            self.free_cams[c] = str(cam.GetPath())
        import usdkit as K
        K.aim(stage.GetPrimAtPath(self.free_cams["overview"]), (0.0, -4.3, 3.3), (0.0, 0.35, 0.0))
        K.aim(stage.GetPrimAtPath(self.free_cams["ramp"]), (2.05, 0.55, 0.42), (0.95, 1.5, 0.02))
        products = [(c, self.free_cams[c], args.width, args.height) for c in ("overview", "ramp", "orbit", "chase")]
        products.append(("duck", self.head_cams[2], args.width, args.height))
        (work / "render_product.usda").write_text(SC.product_layer(products, args.mode, 0))
        stage.GetRootLayer().Save()
        self.paths = [p for d in self.ducks for p in SC.body_paths(d)]
        import ovrtx
        from ovrtx import PrimMode, Semantic
        self.ovrtx, self.PrimMode, self.Semantic = ovrtx, PrimMode, Semantic
        self.r = ovrtx.Renderer(ovrtx.RendererConfig(log_file_path=str(work / "ovrtx.log"), log_level="warn"))
        self.r.open_usd(str(scene))
        self.r.write_attribute(prim_paths=self.paths, attribute_name="omni:resetXformStack",
                               tensor=np.full(len(self.paths), True, dtype=np.bool_))
        moving = [self.free_cams["orbit"], self.free_cams["chase"]]
        self.r.write_attribute(prim_paths=moving, attribute_name="omni:resetXformStack",
                               tensor=np.full(len(moving), True, dtype=np.bool_))
        self.products = {c: f"/{SC.SCOPE}/{c}" for c, _, _, _ in products}
        self.var = f"/{SC.SCOPE}/Vars/LdrColor"
        self.duck_cam = 2
        self.chase_state = None
        self.font = self.small = None
        try:
            from PIL import ImageFont
            self.font = ImageFont.truetype(str(Path(args.fonts) / "NVIDIASans_Bd.ttf"), 19)
        except Exception:  # noqa: BLE001
            pass
        self.write_poses()
        for _ in range(30):
            self.r.step(render_products=set(self.products.values()), delta_time=1 / 30)
        SC.log("live OVRTX scene ready")

    def write_poses(self):
        t, body = self.src.latest()
        if body is None:
            return t, None
        M = SC.pose_to_rowmajor(body.reshape(-1, 7))
        self.r.write_attribute(prim_paths=self.paths, attribute_name="omni:xform", tensor=M,
                               semantic=self.Semantic.XFORM_MAT4x4)
        return t, body

    def set_free(self, name, eye, target):
        from usdkit import look_at_quat_wxyz
        w, x, y, z = look_at_quat_wxyz(eye, target)
        pose = np.array([[*eye, x, y, z, w]], float)
        self.r.write_attribute(prim_paths=[self.free_cams[name]], attribute_name="omni:xform",
                               tensor=SC.pose_to_rowmajor(pose), semantic=self.Semantic.XFORM_MAT4x4)

    def render(self, keys):
        """keys: list of (cam, duck). Renders all requested views in one OVRTX step; returns {key: jpeg bytes}."""
        t, body = self.write_poses()
        want = {}
        for cam, duck in keys:
            if cam == "orbit":
                a = math.radians(270 + 68 * math.sin(t * 0.13))
                self.set_free("orbit", (3.75 * math.cos(a), 3.75 * math.sin(a) + 0.15, 1.85), (0.0, 0.35, 0.05))
            elif cam == "chase" and body is not None:
                root = body[duck, 0]
                q = root[3:7]
                yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
                p = root[:3].copy()
                if self.chase_state is None or self.chase_state[0] != duck:
                    self.chase_state = [duck, p.copy(), yaw]
                st = self.chase_state
                st[1] += 0.15 * (p - st[1])
                st[2] += 0.08 * ((yaw - st[2] + math.pi) % (2 * math.pi) - math.pi)
                c = st[1]
                eye = (c[0] - 0.82 * math.cos(st[2]), c[1] - 0.82 * math.sin(st[2]), 0.30)
                self.set_free("chase", eye, (c[0], c[1], 0.11))
            elif cam == "duck" and duck != self.duck_cam:
                self.r.write_array_attribute(prim_paths=[self.products["duck"]], attribute_name="camera",
                                             tensors=[[self.head_cams[duck]]])
                self.duck_cam = duck
            want[(cam, duck)] = self.products[cam]
        prod = None
        for _ in range(args.updates):
            prod = self.r.step(render_products=set(want.values()), delta_time=1.0 / args.fps)
        out = {}
        from PIL import Image, ImageDraw
        for key, path in want.items():
            fr = prod[path].frames[0]
            with fr.render_vars[self.var].map(device=self.ovrtx.Device.CPU) as mp:
                img = Image.fromarray(np.ascontiguousarray(np.array(np.from_dlpack(mp))[..., :3]))
            self.label(ImageDraw.Draw(img), key, t)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=args.quality)
            out[key] = buf.getvalue()
        return out

    def label(self, d, key, t):
        cam, duck = key
        txt = dict(CAMS)[cam]
        if cam in ("duck", "chase"):
            txt = f"Duck {duck + 1} \u00b7 " + ("head cam" if cam == "duck" else "follow cam")
        credit = credit_line()
        line = f"{txt}  \u00b7  SIMULATION{('  ' + credit) if credit else ''}"
        x0 = 48
        if cam in ("duck", "chase"):
            col = tuple(int(255 * v) for v in Y.DUCK_COLORS[duck])
            d.ellipse([48, args.height - 36, 70, args.height - 14], fill=col, outline=(0, 0, 0), width=3)
            x0 = 78
        d.text((x0, args.height - 37), line, font=self.font, fill=(255, 255, 255), stroke_width=3,
               stroke_fill=(0, 0, 0))


STACK = {}


def credit_line():
    """Engine credit only when the simulation process reports it really runs on Isaac Sim 6.2 with Newton."""
    v = str(STACK.get("isaacsim_dist") or STACK.get("kit_isaacsim") or "")
    if v.startswith("6.2") and STACK.get("physics") == "newton":
        return "\u00b7  Isaac Sim 6.2 \u00b7 Newton \u00b7 RTX"
    return "\u00b7  OVRTX"


class Streamer:
    def __init__(self, rend):
        self.r = rend
        self.want = {}
        self.cache = {}
        self.cv = threading.Condition()
        self.frames = 0
        self.ms = 0.0
        self.ms_per_view = {}

    def key(self, cam, duck):
        return (cam, duck if cam in ("duck", "chase") else 0)

    def request(self, cam, duck, hold):
        k = self.key(cam, duck)
        with self.cv:
            self.want[k] = max(self.want.get(k, 0), time.time() + hold)
            self.cv.notify_all()
        return k

    def snapshot(self, cam, duck):
        k = self.key(cam, duck)
        hit = self.cache.get(k)
        if hit and time.time() - hit[1] < 1.0 / args.fps:
            return hit[0]
        old = hit[1] if hit else 0
        self.request(cam, duck, 0.3)
        end = time.time() + 3.0
        with self.cv:
            while self.cache.get(k, (None, 0))[1] <= old and time.time() < end:
                self.cv.wait(0.05)
            return self.cache.get(k, (None, 0))[0]

    def latest(self, k, wait=3.0):
        end = time.time() + wait
        with self.cv:
            while k not in self.cache and time.time() < end:
                self.cv.wait(0.05)
            return self.cache.get(k, (None, 0))[0]

    def loop(self):
        while True:
            now = time.time()
            with self.cv:
                keys = [k for k, until in self.want.items() if now < until]
            # chase/duck for a different duck than the camera shows: one duck-cam view per step
            seen = set()
            due = []
            for k in keys:
                if now - self.cache.get(k, (None, 0))[1] < 1.0 / args.fps:
                    continue
                if k[0] in seen:
                    continue
                seen.add(k[0])
                due.append(k)
            if not due:
                time.sleep(0.003)
                continue
            t0 = time.perf_counter()
            imgs = self.r.render(due)
            dt = (time.perf_counter() - t0) * 1000
            self.ms = 0.9 * self.ms + 0.1 * dt
            for k in due:
                self.ms_per_view[k[0]] = round(dt, 1)
            with self.cv:
                for k, v in imgs.items():
                    self.cache[k] = (v, time.time())
                self.frames += len(imgs)
                self.cv.notify_all()


PREVIEW = """<!doctype html><html><head><meta name=viewport content="width=device-width"><title>Microducks OVRTX</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:10px}button{margin:3px;padding:8px}img{width:100%;max-width:960px}</style></head>
<body><div id=c></div><div id=d></div><img id=v><div id=k></div><pre id=i></pre><script>
var cam='overview',duck=3;
fetch('api/info').then(r=>r.json()).then(j=>{var c=document.getElementById('c');j.cams.forEach(x=>{var b=document.createElement('button');b.textContent=x.label;b.onclick=()=>{cam=x.id;go()};c.appendChild(b)});
var d=document.getElementById('d');for(let n=1;n<=12;n++){var b=document.createElement('button');b.textContent=n;b.onclick=()=>{duck=n;fetch('api/control',{method:'POST',body:JSON.stringify({action:'select',duck:n})});go()};d.appendChild(b)}
var k=document.getElementById('k');['reset','walk','ramps','stop','push'].forEach(a=>{var b=document.createElement('button');b.textContent=a;b.onclick=()=>fetch('api/control',{method:'POST',body:JSON.stringify({action:a,duck:duck})});k.appendChild(b)});go()});
function go(){document.getElementById('v').src='stream/'+cam+'.mjpg?duck='+duck+'&k='+Date.now()}
setInterval(()=>fetch('api/info').then(r=>r.json()).then(j=>{document.getElementById('i').textContent=JSON.stringify(j.stats)}),2000);
</script></body></html>"""


def make_handler(src, st):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def duck_arg(self, qs):
            try:
                return int(np.clip(int(qs.get("duck", [src.selected + 1])[0]) - 1, 0, Y.NDUCK - 1))
            except Exception:  # noqa: BLE001
                return src.selected

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self.send(200, PREVIEW, "text/html")
            if u.path == "/api/info":
                inf = src.info()
                STACK.update(inf.get("stack") or {})
                inf["cams"] = [dict(id=c, label=l) for c, l in CAMS]
                inf["renderer"] = f"OVRTX {st.r.ovrtx.__version__} ({args.mode}, {args.updates} update/frame, " \
                                  f"{args.width}x{args.height})"
                inf["credit"] = credit_line().strip("\u00b7 ").strip()
                inf["stats"] = dict(sim_time=inf.get("sim_time"), render_ms=round(st.ms, 1),
                                    ms_per_view=st.ms_per_view, frames=st.frames, mode=inf.get("mode"))
                return self.send(200, inf)
            if u.path.startswith("/snapshot/") and u.path.endswith(".jpg"):
                cam = u.path[len("/snapshot/"):-4]
                if cam not in dict(CAMS):
                    return self.send(404, {"error": "unknown cam"})
                jpg = st.snapshot(cam, self.duck_arg(qs))
                return self.send(200, jpg, "image/jpeg") if jpg else self.send(503, {"error": "warming up"})
            if u.path.startswith("/stream/") and u.path.endswith(".mjpg"):
                cam = u.path[len("/stream/"):-5]
                if cam not in dict(CAMS):
                    return self.send(404, {"error": "unknown cam"})
                duck = self.duck_arg(qs)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    for _ in range(int(args.fps * 600)):
                        k = st.request(cam, duck, 3.0)
                        jpg = st.latest(k)
                        if jpg is not None and jpg is not last:
                            last = jpg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                            self.wfile.write(jpg + b"\r\n")
                        time.sleep(1.0 / args.fps)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except Exception:  # noqa: BLE001
                data = {}
            if urlparse(self.path).path != "/api/control":
                return self.send(404, {"error": "not found"})
            try:
                return self.send(200, src.control(data))
            except Exception as e:  # noqa: BLE001
                return self.send(503, {"ok": False, "message": "Simulation offline: " + type(e).__name__})

    return H


def main():
    src = TraceSource(args.trace) if args.source == "trace" else ShmSource(args.shm, args.sim_url)
    rend = LiveRenderer(src)
    st = Streamer(rend)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(src, st))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SC.log(f"serving on http://{args.host}:{args.port} (source {args.source})")
    st.loop()


if __name__ == "__main__":
    main()
