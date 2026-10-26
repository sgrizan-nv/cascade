"""Bad-connection video modes and the "render one camera only" operator switch for a Cascade Lab app directory.

    python3 lowdata_lab_patch.py <app_dir>
- lab_server.py: /api/cam/<cell>/<cam>.jpg?q=low|min re-encodes the JPEG smaller (PIL); settings["only_cam"][cell]
  makes every snapshot and MJPEG request of that cell serve one camera (OVRTX cells then render only that view);
  /api/status reports it; POST /api/admin/only_cam {cam} sets it (admin, "" = all cameras).
- static/app.html (served, additive): quality chip on the stage + "Video quality" in the menu (Auto / Data saver /
  Bad connection). Low modes poll small snapshots instead of the MJPEG stream and pause the thumbnails. Admin menu:
  "Render one camera only".
- make_lab_app.py: the same app.html subs at the end.
Idempotent (marker LOWDATA_ONECAM). ASCII only.
"""
import sys
from pathlib import Path

app = Path(sys.argv[1])

SERVER_SUBS = [
    ('SCENARIOS = json.loads(SCENARIOS_FILE.read_text())\n',
     '''# LOWDATA_ONECAM: smaller JPEGs for bad connections; one camera per cell when the operator asks for it
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


SCENARIOS = json.loads(SCENARIOS_FILE.read_text())
'''),
    ('''            if not cell or cam not in cell.cams:
                return self.send(404, {"error": "camera"})
            try:
                return self.send(200, cell.snapshot(cam, (query.get("duck") or [1])[0]), "image/jpeg")''',
     '''            if not cell or cam not in cell.cams:
                return self.send(404, {"error": "camera"})
            cam = only_cam(cell) or cam
            try:
                q = (query.get("q") or [""])[0]
                data = cell.snapshot(cam, (query.get("duck") or [1])[0])
                return self.send(200, shrink_jpeg(data, q, (cell.id, cam, q)), "image/jpeg")'''),
    ('            return self.proxy_stream(cell, parts[1], (query.get("duck") or [1])[0])',
     '            return self.proxy_stream(cell, only_cam(cell) or parts[1], (query.get("duck") or [1])[0])'),
    ('"cameras": list(cell.cams), "now": time.time()}',
     '"cameras": list(cell.cams), "now": time.time(), "only_cam": only_cam(cell)}'),
    ('"queue": len(cell.queue), "cameras": list(CAMERAS), "now": time.time()})',
     '"queue": len(cell.queue), "cameras": list(CAMERAS), "now": time.time(), "only_cam": only_cam(cell)})'),
    ('        if path == "/api/admin/exclusive":\n',
     '''        if path == "/api/admin/only_cam":
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
'''),
]

HTML_SUBS = [
    ('<div class="tr"><button id="fullBtn" aria-label="Fullscreen">',
     '<div class="tr"><button class="qualchip" id="qualChip" aria-label="Video quality">AUTO</button><button id="fullBtn" aria-label="Fullscreen">'),
    ('  <button class="row" id="themeRow">&#x25D0;<div>Light theme<small>Remembered on this device</small></div><span class="switch" id="themeSwitch"></span></button>\n',
     '  <button class="row" id="themeRow">&#x25D0;<div>Light theme<small>Remembered on this device</small></div><span class="switch" id="themeSwitch"></span></button>\n'
     '  <div class="row" style="cursor:default">&#x25B6;<div>Video quality<small>Data saver and Bad connection send small pictures instead of live video and pause the thumbnails</small></div></div>'
     '<div class="seg" id="qualSeg"><button data-q="auto">Auto</button><button data-q="low">Data saver</button><button data-q="min">Bad connection</button></div>\n'),
    ('    <button class="row" id="clearRow">&#x2715;<div>Cancel all waiting requests<small>Requests from other people in the queue</small></div></button>\n',
     '    <button class="row" id="clearRow">&#x2715;<div>Cancel all waiting requests<small>Requests from other people in the queue</small></div></button>\n'
     '    <div class="row" style="cursor:default">&#x1F3A5;<div>Render one camera only<small>For everyone, in this cell: the other cameras stop streaming (and stop rendering in the duck cells)</small></div></div>'
     '<div class="seg" id="oneCamSeg"></div>\n'),
    ('.orderbox .ob-ex button { font-size: .75rem; }',
     '.orderbox .ob-ex button { font-size: .75rem; }\n'
     '/* LOWDATA_ONECAM */\n'
     '.stage .tr { display: flex; gap: 8px; }\n'
     '.stage .tr .qualchip { width: auto; padding: 0 10px; font-size: .7rem; font-weight: 800; letter-spacing: .06em; color: #fff; }\n'
     '.seg { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: -2px 0 10px; }\n'
     '.seg button { padding: 10px 4px; border-radius: 11px; background: var(--panel); border: 1px solid var(--line); font-size: .76rem;\n'
     '  font-weight: 700; }\n'
     '.seg button.on { background: rgba(118,185,0,.16); border-color: var(--g); }\n'
     '#oneCamSeg { grid-template-columns: repeat(auto-fill, minmax(86px, 1fr)); }\n'
     'body.lowdata .thumbs, body.onecam .thumbs, body.onecam #dotsCam, body.onecam #prevCam, body.onecam #nextCam { display: none; }'),
    ('let mainBusy = false;\nasync function loadMain(force) {\n  if (isSim()) return simStream(force);\n',
     '/* LOWDATA_ONECAM: video quality and the one-camera switch */\n'
     'const QUALS = { auto: "Auto", low: "Data saver", min: "Bad connection" }, QCHIP = { auto: "AUTO", low: "LOW", min: "MIN" };\n'
     'let quality = QUALS[localStorage.cascadeLabQuality] ? localStorage.cascadeLabQuality : "auto", ONLY = null;\n'
     'const lowData = () => quality !== "auto";\n'
     'function setQuality(q, announce) { quality = q; localStorage.cascadeLabQuality = q;\n'
     '  document.querySelectorAll("#qualSeg button").forEach(b => b.classList.toggle("on", b.dataset.q === q));\n'
     '  $("#qualChip").textContent = QCHIP[q]; document.body.classList.toggle("lowdata", lowData());\n'
     '  const im = $("#camImg"); im.onerror = im.onload = null; simUrl = "";\n'
     '  if (announce) toast("Video: " + QUALS[q]); loadMain(true); }\n'
     'function applyOnly(c) { c = c && CAMS.includes(c) ? c : null;\n'
     '  if (c !== ONLY) { ONLY = c; document.body.classList.toggle("onecam", !!c); if (c && cam !== c) setCam(c, false); }\n'
     '  renderOneCam(); }\n'
     'function renderOneCam() { const s = $("#oneCamSeg"); if (!s) return;\n'
     '  const opts = [["", "All"]].concat(CAMS.map(c => [c, camLabel(c)])), sig = JSON.stringify(opts) + (ONLY || "");\n'
     '  if (s.dataset.sig === sig) return; s.dataset.sig = sig; s.innerHTML = "";\n'
     '  opts.forEach(([v, l]) => { const b = document.createElement("button"); b.textContent = l; b.classList.toggle("on", (ONLY || "") === v);\n'
     '    b.onclick = async () => { buzz(10); try { const r = await api("/api/admin/only_cam", { method: "POST", body: JSON.stringify({ cam: v }) });\n'
     '      toast(r.message); applyOnly(r.only_cam); } catch (err) { toast(err.message); } }; s.appendChild(b); }); }\n'
     'let mainBusy = false;\nasync function loadMain(force) {\n  if (isSim() && !lowData()) return simStream(force);\n'),
    ('    const r = await fetch(`/api/cam/${SCN}/${want}.jpg?t=${Date.now()}`, { headers: { "X-Client": clientId } });',
     '    const r = await fetch(`/api/cam/${SCN}/${want}.jpg?t=${Date.now()}${lowData() ? "&q=" + quality : ""}${camQ()}`, { headers: { "X-Client": clientId } });'),
    ('async function loadThumbs() {\n  for (const t of thumbs.querySelectorAll(".thumb")) {',
     'async function loadThumbs() { if (lowData() || ONLY) return;\n  for (const t of thumbs.querySelectorAll(".thumb")) {'),
    ('(function camLoop() { if (!document.hidden) loadMain(false); setTimeout(camLoop, 280); })();',
     '(function camLoop() { if (!document.hidden) loadMain(false); setTimeout(camLoop, quality === "min" ? 1000 : quality === "low" ? 450 : 280); })();'),
    ('(function thumbLoop() { if (!document.hidden) loadThumbs(); setTimeout(thumbLoop, 2500); })();',
     '(function thumbLoop() { if (!document.hidden) loadThumbs(); setTimeout(thumbLoop, 2500); })();\n'
     '$("#qualChip").onclick = e => { e.stopPropagation(); const order = ["auto", "low", "min"];\n'
     '  setQuality(order[(order.indexOf(quality) + 1) % order.length], true); };\n'
     'document.querySelectorAll("#qualSeg button").forEach(b => b.onclick = () => setQuality(b.dataset.q, true));\n'
     'setQuality(quality, false);'),
    ('function setCam(c, manual) {\n',
     'function setCam(c, manual) {\n'
     '  if (ONLY && c !== ONLY) { if (manual) toast("Only the " + camLabel(ONLY) + " camera renders right now"); c = ONLY; }\n'),
    ('  applyRole(status.role, status.exclusive);\n  const cur = status.current',
     '  applyRole(status.role, status.exclusive); applyOnly(status.only_cam);\n  const cur = status.current'),
    ('buildCams(isSim() ? SPEC.cams : BASE_CAMS); updDuckSel();',
     'buildCams(isSim() ? SPEC.cams : BASE_CAMS); ONLY = null; document.body.classList.remove("onecam"); renderOneCam(); updDuckSel();'),
]


def apply(text, subs, what):
    for old, new in subs:
        if text.count(old) != 1:
            raise SystemExit(f"{what}: expected 1 x {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


ls = app / "lab_server.py"
s = ls.read_text()
if "LOWDATA_ONECAM" not in s:
    ls.write_text(apply(s, SERVER_SUBS, "lab_server.py"))
    print("lab_server.py patched")

html = app / "static" / "app.html"
h = html.read_text(encoding="utf-8")
if "LOWDATA_ONECAM" not in h:
    h2 = apply(h, HTML_SUBS, "app.html")
    assert sum(ord(c) > 127 for c in h2) == sum(ord(c) > 127 for c in h), "non-ASCII added to app.html"
    html.write_text(h2, encoding="utf-8")
    print("app.html patched")

mk = app / "make_lab_app.py"
m = mk.read_text()
if "LOWDATA_ONECAM" not in m:
    tail = 'Path(__file__).with_name("static").joinpath("app.html").write_text(s, encoding="utf-8")'
    assert m.count(tail) == 1, "make_lab_app tail"
    blk = "# ---- LOWDATA_ONECAM: bad-connection video modes and the one-camera operator switch\n"
    blk += "".join(f"sub({old!r}, {new!r})\n" for old, new in HTML_SUBS)
    mk.write_text(m.replace(tail, blk + tail))
    print("make_lab_app.py patched")

for f in (ls, mk):
    f.read_text(encoding="utf-8").encode("ascii")
print("ascii ok")
