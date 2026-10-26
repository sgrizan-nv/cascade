"""Add the "Microducks - Keynote" tab to a Cascade Lab app directory.

    python3 keynote_tab_patch.py <app_dir>
- scenarios.json: the keynote sim cell (stream http://127.0.0.1:18210, keynote_live_ovrtx.py + keynote_live_sim.py).
- static/app.html (served, additive): per-scenario order examples (SPEC.orders), one-tap cards that send a typed order
  to the agent (q.order), and the agent feed under the order box (SPEC.agent_feed: orders, perception tool calls,
  action tool calls, Nemotron's replies and camera checks, polled from /api/sim/info).
- lab_server.py: typed orders may take up to 60 s (Nemotron can look through a camera first).
- make_lab_app.py: the same app.html subs at the end, for future regenerations.
Idempotent (marker KEYNOTE_TAB). ASCII only.
"""
import json
import sys
from pathlib import Path

app = Path(sys.argv[1])

KEYNOTE = {
    "id": "keynote",
    "kind": "sim",
    "title": "Microducks \u00b7 Keynote",
    "icon": "\U0001f3ac",
    "accent": "#76b900",
    "gpu": 0,
    "stream": "http://127.0.0.1:18210",
    "subtitle": "Nemotron steers 12 ducks from 4 cameras",
    "robot": "12 Microducks",
    "reset_message": "Ducks back on their keynote start spots.",
    "reset_sub": "Put the 12 ducks back on their keynote start spots",
    "cams": ["overview", "ramp", "orbit", "chase", "duck", "map", "eye"],
    "cam_labels": {"overview": "Overview", "ramp": "Ramp", "orbit": "Orbit", "chase": "Follow duck",
                   "duck": "Duck camera", "map": "Perceived map", "eye": "Nemotron's camera"},
    "objects": [],
    "dests": [],
    "quick": [
        {"em": "\u2b55", "title": "Gather", "sub": "on the centre circle", "order": "Everyone, gather in the centre circle.", "text": ""},
        {"em": "\U0001f986", "title": "Line up five", "sub": "at the north ramp", "order": "Five of you, line up.", "text": ""},
        {"em": "\U0001f440", "title": "Look", "sub": "is the north ramp clear?", "order": "Look through the north-east camera. Is the north ramp clear?", "text": ""},
        {"em": "\u26f0\ufe0f", "title": "Cross the ramp", "sub": "in single file", "order": "Now walk over the north ramp in single file.", "text": ""},
        {"em": "\u270b", "title": "Stop!", "sub": "everyone freezes", "order": "Stop!", "text": ""},
        {"em": "\U0001f501", "title": "Regroup", "sub": "back to the circle", "order": "Regroup in the centre circle.", "text": ""},
        {"em": "\U0001f6b6", "title": "Everyone walk", "sub": "wander the yard", "action": "walk", "text": ""},
        {"em": "\U0001f4a5", "title": "Push the duck", "sub": "shove the selected duck", "action": "push", "text": ""},
        {"em": "\U0001f504", "title": "Reset", "sub": "keynote start spots", "action": "reset", "text": ""},
    ],
    "orders": ["Everyone, gather in the centre circle.", "Five of you, line up.",
               "Look through the north-east camera. Is the north ramp clear?",
               "Now walk over the north ramp in single file.", "Stop!", "Regroup in the centre circle.", "Scatter!",
               "How many ducks are standing?", "Give duck 3 a light push", "Go back to wandering"],
    "order_hint": "\u00b7 NVIDIA Nemotron 3.5 Bolt reads the cameras, then picks one tool",
    "agent_feed": True,
    "examples": ["Tap a duck number to ride its head camera"],
    "about": ("Live version of the keynote demo: 12 Microducks in NVIDIA Isaac Sim 6.2 + Newton on the Holodeck set, "
              "rendered live with Omniverse RTX (OVRTX). The navigator never reads the simulator: 4 fixed RGB-D cameras "
              "(Isaac Lab cameras, Newton Warp renderer) mapped the yard in the first 1.5 s and track every duck from "
              "depth and segmentation at 10 Hz. Typed orders go to NVIDIA Nemotron 3.5 Bolt, which can read the "
              "perceived state or look through a camera, then picks one tool; after each order the cameras check the "
              "result and Nemotron reports it below. All 12 ducks run the same ONNX walking policy at 50 Hz. "
              "Runs at about 0.8x real time."),
}

HTML_SUBS = [
    ('<div class="ob-title">Type an order <span>&middot; NVIDIA Nemotron 3.5 Bolt turns it into a tool call</span></div>',
     '<div class="ob-title">Type an order <span id="obBrain">&middot; NVIDIA Nemotron 3.5 Bolt turns it into a tool call</span></div>'),
    ('<div class="ob-out" id="obOut"></div>',
     '<div class="ob-out" id="obOut"></div>\n        <div class="ob-feed" id="obFeed"></div>'),
    ('.orderbox .ob-ex button { font-size: .75rem; }',
     '.orderbox .ob-ex button { font-size: .75rem; }\n'
     '/* KEYNOTE_TAB: agent feed */\n'
     '.orderbox .ob-feed { margin-top: 10px; display: grid; gap: 6px; font-size: .78rem; }\n'
     '.orderbox .ob-feed:empty { display: none; }\n'
     '.orderbox .ob-feed > div { background: rgba(255,255,255,.05); border-radius: 10px; padding: 7px 10px; line-height: 1.35;\n'
     '  word-break: break-word; }\n'
     '.orderbox .ob-feed b { font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; opacity: .9; }\n'
     '.orderbox .ob-feed span { opacity: .5; font-size: .68rem; margin-left: 4px; }\n'
     '.orderbox .ob-feed .k-tool b, .orderbox .ob-feed .k-check b { color: var(--accent); }\n'
     '.orderbox .ob-feed .busy { opacity: .75; font-style: italic; }'),
    ('  b.onclick = () => q.action ? simAction(q.action) : send(q.text); $("#quick").appendChild(b); }); }',
     '  b.onclick = () => q.order ? obOrder(q.order) : q.action ? simAction(q.action) : send(q.text); $("#quick").appendChild(b); }); }'),
    ('  renderQuick(); renderChips(); renderSugg();',
     '  renderQuick(); renderOb(); renderChips(); renderSugg();'),
    ('$("#resetSub").textContent = isSim() ? "Scatter all 12 ducks across the yard again" :',
     '$("#resetSub").textContent = isSim() ? (SPEC.reset_sub || "Scatter all 12 ducks across the yard again") :'),
    ('(function () { const ex = $("#obEx"); if (!ex) return;\n'
     '  OB_EX.forEach(t => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t;\n'
     '    b.onclick = () => { $("#obText").value = t; $("#obText").focus(); }; ex.appendChild(b); });\n',
     '/* KEYNOTE_TAB: per-scenario order examples, one-tap orders, agent feed */\n'
     'const OBK = { order: "Order", perception: "Perception tool", tool: "Action tool", reply: "Nemotron", check: "Camera check" };\n'
     'let obFeedSig = "";\n'
     'function renderOb() { const ex = $("#obEx"); if (!ex) return; ex.innerHTML = "";\n'
     '  ((SPEC && SPEC.orders) || OB_EX).forEach(t => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t;\n'
     '    b.onclick = () => { $("#obText").value = t; $("#obText").focus(); }; ex.appendChild(b); });\n'
     '  const hint = $("#obBrain"); if (hint) hint.textContent = (SPEC && SPEC.order_hint) || "\\u00b7 NVIDIA Nemotron 3.5 Bolt turns it into a tool call";\n'
     '  obFeedSig = ""; $("#obFeed").innerHTML = ""; $("#obOut").textContent = ""; obFeedTick(); }\n'
     'function obOrder(t) { $("#obText").value = t; $("#obForm").requestSubmit(); }\n'
     'async function obFeedTick() { if (!SPEC || !SPEC.agent_feed || document.hidden) return; const want = SCN;\n'
     '  try { const r = await api("/api/sim/info"); if (want !== SCN) return; const ag = r.agent || {}; const fd = ag.feed || [];\n'
     '    const sig = fd.map(x => x.id).join(",") + (ag.busy ? "b" : ""); if (sig === obFeedSig) return; obFeedSig = sig;\n'
     '    $("#obFeed").innerHTML = (ag.busy ? \'<div class="busy">Nemotron is working on an order...</div>\' : "") +\n'
     '      fd.slice(-6).reverse().map(x => `<div class="k-${esc(x.kind)}"><b>${esc(OBK[x.kind] || x.kind)}</b><span>${esc(x.wall)}</span><br>${esc(x.text)}</div>`).join("");\n'
     '  } catch {} }\n'
     'setInterval(obFeedTick, 1500);\n'
     '(function () { if (!$("#obEx")) return;\n'),
]


def apply(text, what):
    for old, new in HTML_SUBS:
        if text.count(old) != 1:
            raise SystemExit(f"{what}: expected 1 x {old[:50]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


# 1) scenarios.json
sc = app / "scenarios.json"
items = json.loads(sc.read_text())
if not any(s.get("id") == "keynote" for s in items):
    pos = next((k + 1 for k, s in enumerate(items) if s.get("id") == "microducks"), len(items))
    items.insert(pos, KEYNOTE)
    sc.write_text(json.dumps(items, indent=1) + "\n")
    print("scenarios.json: keynote tab added")

# 2) lab_server.py
ls = app / "lab_server.py"
s = ls.read_text()
if "KEYNOTE_TAB" not in s:
    old = '        with OPENER.open(req, timeout=15 if action == "order" else 8) as r:'
    assert s.count(old) == 1, "sim_control timeout"
    s = s.replace(old, '        with OPENER.open(req, timeout=60 if action == "order" else 8) as r:  # KEYNOTE_TAB: look() can take a while')
    ls.write_text(s)
    print("lab_server.py patched")

# 3) served app.html
html = app / "static" / "app.html"
h = html.read_text(encoding="utf-8")
if "KEYNOTE_TAB" not in h:
    h2 = apply(h, "app.html")
    assert sum(ord(c) > 127 for c in h2) == sum(ord(c) > 127 for c in h), "non-ASCII added to app.html"
    html.write_text(h2, encoding="utf-8")
    print("app.html patched")

# 4) make_lab_app.py
mk = app / "make_lab_app.py"
m = mk.read_text()
if "KEYNOTE_TAB" not in m:
    tail = 'Path(__file__).with_name("static").joinpath("app.html").write_text(s, encoding="utf-8")'
    assert m.count(tail) == 1, "make_lab_app tail"
    blk = "# ---- KEYNOTE_TAB: per-scenario order examples, one-tap orders, agent feed\n"
    blk += "".join(f"sub({old!r}, {new!r})\n" for old, new in HTML_SUBS)
    mk.write_text(m.replace(tail, blk + tail))
    print("make_lab_app.py patched")

for f in (sc, ls, mk):
    f.read_text(encoding="utf-8").encode("ascii")
print("ascii ok")
