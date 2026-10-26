"""Add the Microducks agent order box to a Cascade Lab app directory.

    python3 orderbox_patch.py <app_dir>
Patches lab_server.py (typed order -> /api/control with text), the served static/app.html (additive, no regeneration,
so other people's edits stay) and appends the same block to make_lab_app.py for future regenerations.
Idempotent (marker ORDERBOX). ASCII only.
"""
import sys
from pathlib import Path

app = Path(sys.argv[1])

HTML_ANCHOR = '<div class="quick" id="quick"></div>'
HTML_NEW = HTML_ANCHOR + """
      <div class="card orderbox" id="orderBox">
        <div class="ob-title">Type an order <span>&middot; a local LLM agent (Qwen3.8-27B) turns it into a tool call</span></div>
        <form id="obForm" autocomplete="off"><input id="obText" maxlength="200" placeholder="e.g. Regroup in the square" aria-label="Order for the ducks">
          <button class="btn" id="obSend" type="submit">Send</button></form>
        <div class="ob-out" id="obOut"></div>
        <div class="chips ob-ex" id="obEx"></div>
      </div>"""
CSS_ANCHOR = "</style>"
CSS_NEW = """.orderbox { display: none; margin-top: 12px; padding: 12px 14px; }
body.is-sim .orderbox { display: block; }
.orderbox .ob-title { font-weight: 800; font-size: .9rem; margin-bottom: 8px; }
.orderbox .ob-title span { font-weight: 500; opacity: .7; font-size: .78rem; }
.orderbox form { display: flex; gap: 8px; }
.orderbox input { flex: 1; min-width: 0; font: inherit; color: inherit; background: rgba(255,255,255,.06);
  border: 1px solid var(--line, rgba(255,255,255,.14)); border-radius: 12px; padding: 10px 12px; }
.orderbox .ob-out { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem; margin-top: 8px; opacity: .9;
  white-space: pre-wrap; word-break: break-word; min-height: 1.2em; }
.orderbox .ob-ex { margin-top: 8px; }
.orderbox .ob-ex button { font-size: .75rem; }
</style>"""
JS_ANCHOR = "/* status + feed */"
JS_NEW = """/* ORDERBOX */
const OB_EX = ["Regroup in the square", "Two racers to the west ramp start line", "Ready... go!",
  "Follow duck 5 around the cones", "Left group take the west ramp, right group the north ramp", "Scatter!", "Stop!",
  "Stress test: give duck 3 a light push"];
(function () { const ex = $("#obEx"); if (!ex) return;
  OB_EX.forEach(t => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t;
    b.onclick = () => { $("#obText").value = t; $("#obText").focus(); }; ex.appendChild(b); });
  $("#obForm").onsubmit = async e => { e.preventDefault(); const t = $("#obText").value.trim(); if (!t) return;
    const btn = $("#obSend"); btn.disabled = true; $("#obOut").textContent = "Asking the agent..."; buzz(10);
    try { const r = await api("/api/sim", { method: "POST", body: JSON.stringify({ action: "order", text: t, duck: DUCK }) });
      $("#obOut").textContent = r.tool_call ? ("Tool call: " + r.tool_call + "\\n" + (r.navigator || "")) : (r.message || "No tool call");
      if (r.tool_call) $("#obText").value = ""; refreshStatus(); }
    catch (err) { $("#obOut").textContent = err.message || "Simulation offline"; }
    finally { btn.disabled = false; } }; })();

/* status + feed */"""
SUBS = [(HTML_ANCHOR, HTML_NEW), (CSS_ANCHOR, CSS_NEW), (JS_ANCHOR, JS_NEW)]


def apply(text, what):
    for old, new in SUBS:
        if text.count(old) != 1:
            raise SystemExit(f"{what}: expected 1 x {old[:40]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


# 1) lab_server.py
ls = app / "lab_server.py"
s = ls.read_text()
if "ORDERBOX" not in s:
    old = '''    def sim_control(self, action, duck=None):
        body = json.dumps({"action": action, "duck": duck}).encode()
        req = Request(self.spec["stream"] + "/api/control", data=body, method="POST",
                      headers={"Content-Type": "application/json"})
        with OPENER.open(req, timeout=8) as r:
            return json.load(r)'''
    new = '''    def sim_control(self, action, duck=None, text=None):  # ORDERBOX: typed orders go to the Microducks agent
        msg = {"action": action, "duck": duck}
        if text:
            msg["text"] = " ".join(str(text).split())[:200]
        req = Request(self.spec["stream"] + "/api/control", data=json.dumps(msg).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
        with OPENER.open(req, timeout=15 if action == "order" else 8) as r:
            return json.load(r)'''
    assert s.count(old) == 1, "sim_control"
    s = s.replace(old, new)
    old = '''            try:
                r = cell.sim_control(action, data.get("duck"))
                cell.sim_cache = (0.0, {})
                log(cell.id + " SIM " + action + " " + str(data.get("duck")))
                return self.send(200, r)
            except Exception as e:'''
    new = '''            try:
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
            except Exception as e:'''
    assert s.count(old) == 1, "sim handler"
    s = s.replace(old, new)
    s = s.replace("from urllib.request import", "from urllib.error import HTTPError\nfrom urllib.request import", 1)
    ls.write_text(s)
    print("lab_server.py patched")

# 2) served app.html
html = app / "static" / "app.html"
h = html.read_text(encoding="utf-8")
if "ORDERBOX" not in h:
    html.write_text(apply(h, "app.html"), encoding="utf-8")
    print("app.html patched")

# 3) make_lab_app.py: same subs at the end, for future regenerations
mk = app / "make_lab_app.py"
m = mk.read_text()
if "ORDERBOX" not in m:
    tail = 'Path(__file__).with_name("static").joinpath("app.html").write_text(s, encoding="utf-8")'
    assert m.count(tail) == 1, "make_lab_app tail"
    blk = "# ---- ORDERBOX: Microducks agent order box (typed order -> Qwen3.8-27B tool call -> live ducks)\n"
    blk += "".join(f"sub({old!r}, {new!r})\n" for old, new in SUBS)
    mk.write_text(m.replace(tail, blk + tail))
    print("make_lab_app.py patched")

for f in (ls, mk, html):
    f.read_text(encoding="utf-8").encode("ascii")
print("ascii ok")
