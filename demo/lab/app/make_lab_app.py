"""Derive the Cascade Lab app from the BREV mobile app (kept byte-for-byte otherwise)."""
from pathlib import Path

src = Path(__file__).with_name("static") / "app.html"
s = src.read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} x {old[:60]!r}, found {n}")
    s = s.replace(old, new)


sub('<meta name="apple-mobile-web-app-title" content="Cascade">',
    '<meta name="apple-mobile-web-app-title" content="Cascade Lab">')
sub("<title>Cascade \u00b7 Kitchen robot</title>", "<title>Cascade Lab \u00b7 Experimental robot cells</title>")
sub('<div class="brand"><b>CASCADE</b><small>Kitchen robot \u00b7 Isaac Sim</small></div>',
    '<div class="brand"><b>CASCADE <em class="labtag">LAB</em></b><small id="brandSub">Experimental cells \u00b7 Isaac Sim</small></div>')
sub("  </header>\n",
    "  </header>\n  <nav class=\"scn\" id=\"scn\" aria-label=\"Choose a robot cell\"></nav>\n")
sub("<h3>Cascade kitchen</h3>", "<h3 id=\"sheetTitle\">Cascade Lab</h3>")
sub("A robot arm in a simulated kitchen (NVIDIA Isaac Sim).",
    "Experimental robot cells (NVIDIA Isaac Sim): switch between the workshop and the recycling plant at the top.")
sub("Reset the kitchen<small>Put every object back where it started</small>",
    "<span id=\"resetLabel\">Reset this cell</span><small id=\"resetSub\">Put every object back where it started</small>")

css = """
.labtag { font-style: normal; font-size: .62rem; letter-spacing: .12em; padding: 2px 6px; margin-left: 4px;
  border-radius: 6px; background: var(--accent, #f5a623); color: #111; vertical-align: 3px; font-weight: 900; }
.scn { display: flex; gap: 8px; padding: 2px 14px 10px; overflow-x: auto; scrollbar-width: none; }
.scn::-webkit-scrollbar { display: none; }
.scn button { flex: 1 0 auto; min-width: 140px; display: flex; align-items: center; gap: 10px; text-align: left;
  border: 1px solid var(--line, rgba(255,255,255,.12)); background: var(--card, rgba(255,255,255,.04));
  color: inherit; border-radius: 16px; padding: 10px 12px; cursor: pointer; transition: transform .15s, border-color .2s; }
.scn button:active { transform: scale(.97); }
.scn button .ic { font-size: 1.5rem; line-height: 1; }
.scn button b { display: block; font-size: .95rem; }
.scn button small { display: block; font-size: .7rem; opacity: .7; }
.scn button .hd { width: 9px; height: 9px; border-radius: 50%; margin-left: auto; background: #888; flex: none; }
.scn button .hd.healthy { background: #2ec27e; box-shadow: 0 0 8px #2ec27e; }
.scn button .hd.starting { background: #f5a623; }
.scn button.on { border-color: var(--accent, #f5a623); box-shadow: 0 0 0 1px var(--accent, #f5a623) inset; }
"""
sub("</style>", css + "</style>")

old_consts_start = s.index("const CAMS = [")
old_consts_end = s.index("const clientId")
s = s[:old_consts_start] + """const CAMS = ["kitchen", "worktop", "side"];
let SCNS = [], SCN = localStorage.cascadeLabScn || "workshop", SPEC = null;
let OBJECTS = [], DESTS = [], QUICK = [], EXAMPLES = ["What can you see?"];
const withS = p => p.startsWith("/api/") && !p.startsWith("/api/cam/") && !p.startsWith("/api/job/") ?
  p + (p.includes("?") ? "&" : "?") + "s=" + encodeURIComponent(SCN) : p;
const camLabel = c => (SPEC && SPEC.cam_labels && SPEC.cam_labels[c]) || c;
""" + s[old_consts_end:]

sub('const api = (path, opts = {}) => fetch(path, {', 'const api = (path, opts = {}) => fetch(withS(path), {')
sub("fetch(`/api/cam/${want}.jpg?t=${Date.now()}`", "fetch(`/api/cam/${SCN}/${want}.jpg?t=${Date.now()}`")
sub("fetch(`/api/cam/${t.dataset.cam}.jpg?t=${Date.now()}`", "fetch(`/api/cam/${SCN}/${t.dataset.cam}.jpg?t=${Date.now()}`")
sub("b.innerHTML = `<img alt=\"\"><span>${c}</span>`;", "b.innerHTML = `<img alt=\"\"><span data-cl=\"${c}\">${c}</span>`;")
sub('$("#camName").textContent = c;', '$("#camName").textContent = camLabel(c);')

# Quick actions, chips and examples are rebuilt for each scenario.
sub("""QUICK.forEach(q => { const b = document.createElement("button"); b.className = "qa";
  b.innerHTML = `<div class="em">${q.em}</div><b>${esc(q.title)}</b><small>${esc(q.sub)}</small>`;
  b.onclick = () => send(q.text); $("#quick").appendChild(b); });""",
    """function renderQuick() { $("#quick").innerHTML = ""; QUICK.forEach(q => { const b = document.createElement("button"); b.className = "qa";
  b.innerHTML = `<div class="em">${q.em}</div><b>${esc(q.title)}</b><small>${esc(q.sub)}</small>`;
  b.onclick = () => send(q.text); $("#quick").appendChild(b); }); }""")
sub("""OBJECTS.forEach(o => chip($("#objChips"), o, (it, b) => { selObj = it;
  $("#objChips").querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b)); updatePreview(); }));
DESTS.forEach(d => chip($("#dstChips"), d, (it, b) => { selDst = it;
  $("#dstChips").querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b)); updatePreview(); }));
function sentence() { return selObj && selDst ? selDst.tpl(selObj.noun) : null; }""",
    """function renderChips() { $("#objChips").innerHTML = ""; $("#dstChips").innerHTML = ""; selObj = selDst = null;
OBJECTS.forEach(o => chip($("#objChips"), o, (it, b) => { selObj = it;
  $("#objChips").querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b)); updatePreview(); }));
DESTS.forEach(d => chip($("#dstChips"), d, (it, b) => { selDst = it;
  $("#dstChips").querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b)); updatePreview(); }));
updatePreview(); }
function sentence() { return selObj && selDst ? selDst.tpl.replace("{o}", selObj.noun) : null; }""")
sub("""EXAMPLES.slice(0, 5).forEach(x => { const b = document.createElement("button"); b.textContent = x;
  b.onclick = () => { ta.value = x; ta.dispatchEvent(new Event("input")); ta.focus(); }; $("#sugg").appendChild(b); });""",
    """function renderSugg() { $("#sugg").innerHTML = ""; EXAMPLES.slice(0, 5).forEach(x => { const b = document.createElement("button"); b.textContent = x;
  b.onclick = () => { ta.value = x; ta.dispatchEvent(new Event("input")); ta.focus(); }; $("#sugg").appendChild(b); }); }
function applyScenario(id) {
  SPEC = SCNS.find(x => x.id === id) || SCNS[0]; if (!SPEC) return; SCN = SPEC.id; localStorage.cascadeLabScn = SCN;
  OBJECTS = SPEC.objects; DESTS = SPEC.dests; QUICK = SPEC.quick; EXAMPLES = SPEC.examples; exI = 0;
  document.documentElement.style.setProperty("--accent", SPEC.accent || "#f5a623");
  $("#brandSub").textContent = `${SPEC.title} \\u00b7 ${SPEC.subtitle}`;
  $("#sheetTitle").textContent = `Cascade Lab \\u00b7 ${SPEC.title}`;
  $("#resetLabel").textContent = SPEC.id === "recycling" ? "Drop new waste" : `Reset the ${SPEC.title.toLowerCase()}`;
  $("#scn").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.id === SCN));
  document.querySelectorAll("[data-cl]").forEach(e => { e.textContent = camLabel(e.dataset.cl); });
  $("#camName").textContent = camLabel(cam);
  renderQuick(); renderChips(); renderSugg(); items = []; lastSig = ""; renderFeed();
  loadMain(true); loadThumbs(); refreshStatus(); refreshFeed();
}
function renderScnBar() { const bar = $("#scn"); bar.innerHTML = "";
  SCNS.forEach(x => { const b = document.createElement("button"); b.dataset.id = x.id;
    b.innerHTML = `<span class="ic">${x.icon}</span><span><b>${esc(x.title)}</b><small>${esc(x.subtitle)}</small></span><i class="hd ${esc(x.health)}"></i>`;
    b.onclick = () => { if (x.id !== SCN) { buzz(10); applyScenario(x.id); } }; bar.appendChild(b); }); }
async function loadScenarios() { try { const r = await api("/api/scenarios"); SCNS = r.items;
  renderScnBar(); applyScenario(SCNS.some(x => x.id === SCN) ? SCN : r.default); } catch (e) { setTimeout(loadScenarios, 3000); } }
setInterval(async () => { try { const r = await api("/api/scenarios"); r.items.forEach(x => {
  const d = $(`#scn button[data-id="${x.id}"] .hd`); if (d) d.className = "hd " + x.health; }); } catch {} }, 10000);""")

sub("rotateGhost(); setInterval(rotateGhost, 3600);", "setInterval(rotateGhost, 3600);")
s = s.replace("</script>\n</body>", "loadScenarios(); rotateGhost();\n</script>\n</body>", 1)
if "loadScenarios(); rotateGhost();" not in s:
    raise SystemExit("could not append bootstrap")
sub('SCN = localStorage.cascadeLabScn || "workshop"', 'SCN = localStorage.cascadeLabScn || ""')
# ---- Microducks (kind "sim"): stream cameras, duck selector, one-tap controls. See MICRODUCKS_INTEGRATION.md.
sub('const CAMS = ["kitchen", "worktop", "side"];\nlet SCNS',
    'let CAMS = ["kitchen", "worktop", "side"]; const BASE_CAMS = CAMS.slice();\n'
    'let DUCK = +(localStorage.cascadeLabDuck || 1) || 1, simUrl = "", simAt = 0, simRetryAt = 0;\n'
    'const DUCK_COLORS = ["#f23f3f", "#f99926", "#f9e033", "#72d84c", "#26b28c", "#3fccf2", "#3f72f2", "#8c59e5",'
    ' "#e559cc", "#f9f9f9", "#333338", "#a5724c"];\n'
    'const isSim = () => !!(SPEC && SPEC.kind === "sim");\n'
    'const camQ = () => isSim() ? "&duck=" + DUCK : "";\n'
    'let SCNS')
sub("""CAMS.forEach(c => {
  const b = document.createElement("button"); b.className = "thumb"; b.dataset.cam = c;
  b.innerHTML = `<img alt=""><span data-cl="${c}">${c}</span>`; b.onclick = () => setCam(c, true); thumbs.appendChild(b);
  dotsCam.appendChild(document.createElement("i"));
});""", """function buildCams(list) { CAMS = list.slice(); thumbs.innerHTML = ""; dotsCam.innerHTML = "";
CAMS.forEach(c => {
  const b = document.createElement("button"); b.className = "thumb"; b.dataset.cam = c;
  b.innerHTML = `<img alt=""><span data-cl="${c}">${c}</span>`; b.onclick = () => setCam(c, true); thumbs.appendChild(b);
  dotsCam.appendChild(document.createElement("i"));
});
if (!CAMS.includes(cam)) cam = CAMS[0];
thumbs.querySelectorAll(".thumb").forEach(t => t.classList.toggle("on", t.dataset.cam === cam));
dotsCam.querySelectorAll("i").forEach((d, i) => d.classList.toggle("on", CAMS[i] === cam)); }
buildCams(CAMS);
function updDuckSel() { const show = isSim() && (cam === "duck" || cam === "chase");
  $("#duckSel").classList.toggle("show", show);
  $("#duckSel").querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.d === DUCK)); }
function buildDucks() { const d = $("#duckSel"); d.innerHTML = '<span class="dl">Pick a duck</span>';
  DUCK_COLORS.forEach((c, i) => { const b = document.createElement("button"); b.textContent = i + 1; b.dataset.d = i + 1;
    b.style.background = c; b.style.color = (i === 10 || i === 7 || i === 6) ? "#fff" : "#111";
    b.onclick = () => pickDuck(i + 1); d.appendChild(b); }); updDuckSel(); }
function pickDuck(n) { DUCK = n; localStorage.cascadeLabDuck = n; buzz(8);
  api("/api/sim", { method: "POST", body: JSON.stringify({ action: "select", duck: n }) }).catch(() => {});
  if (cam !== "duck" && cam !== "chase") setCam("duck", true); else { updDuckSel(); loadMain(true); loadThumbs(); } }
function simStream(force) {
  const u = `/api/stream/${SCN}/${cam}.mjpg?duck=${DUCK}`;
  if (!force && (Date.now() < simRetryAt || (simUrl === u && Date.now() - simAt < 240000))) return;
  simUrl = u; simAt = Date.now(); const img = $("#camImg");
  if (lastUrl) { URL.revokeObjectURL(lastUrl); lastUrl = null; }
  img.onerror = () => { if (!isSim()) return; $("#stage").classList.add("isoff"); simUrl = ""; simRetryAt = Date.now() + 2500; };
  img.onload = () => { $("#stage").classList.remove("isoff"); };
  img.src = u + "&k=" + Date.now(); }
async function simAction(a) { buzz(10);
  try { const r = await api("/api/sim", { method: "POST", body: JSON.stringify({ action: a, duck: DUCK }) });
    toast(r.message || "OK"); if (a === "push" && cam !== "duck" && cam !== "chase") setCam("chase", true); refreshStatus(); }
  catch (err) { toast(err.message || "Simulation offline"); } }""")
sub("  loadMain(true);\n}\nconst step", "  updDuckSel(); loadMain(true);\n}\nconst step")
sub("async function loadMain(force) {\n  if (mainBusy && !force) return;",
    "async function loadMain(force) {\n  if (isSim()) return simStream(force);\n  if (mainBusy && !force) return;")
sub("fetch(`/api/cam/${SCN}/${t.dataset.cam}.jpg?t=${Date.now()}`",
    "fetch(`/api/cam/${SCN}/${t.dataset.cam}.jpg?t=${Date.now()}${camQ()}`")
sub("b.onclick = () => send(q.text); $(\"#quick\").appendChild(b); }); }",
    "b.onclick = () => q.action ? simAction(q.action) : send(q.text); $(\"#quick\").appendChild(b); }); }")
sub("  OBJECTS = SPEC.objects; DESTS = SPEC.dests;",
    "  document.body.classList.toggle(\"is-sim\", isSim()); buildCams(isSim() ? SPEC.cams : BASE_CAMS); updDuckSel();\n"
    "  if (!isSim()) { const im = $(\"#camImg\"); im.onerror = im.onload = null; simUrl = \"\"; }\n"
    "  $(\"#resetSub\").textContent = isSim() ? \"Scatter all 12 ducks across the yard again\" :"
    " \"Put every object back where it started\";\n"
    "  $(\"#simAbout\").textContent = SPEC.about || \"\"; $(\"#oneTapSub\").textContent = isSim() ?"
    " \"\\u00b7 the ducks react live\" : \"\\u00b7 the robot does it live\";\n"
    "  OBJECTS = SPEC.objects; DESTS = SPEC.dests;")
sub('$("#nowTitle").textContent = locked ? "Live demo in progress" : "Ready for you";',
    '$("#nowTitle").textContent = status.sim_title || (locked ? "Live demo in progress" : "Ready for you");')
sub('$("#nowSub").textContent = status.queue ?', '$("#nowSub").textContent = status.sim_sub ? status.sim_sub : status.queue ?')
sub('<div class="thumbs" id="thumbs"></div>\n', '<div class="thumbs" id="thumbs"></div>\n      <div class="ducksel" id="duckSel"></div>\n')
sub('<div class="section-title">One tap <span>\u00b7 the robot does it live</span></div>',
    '<div class="card simabout" id="simAbout"></div>\n'
    '      <div class="section-title">One tap <span id="oneTapSub">\u00b7 the robot does it live</span></div>')
sub('<div class="section-title">Build a task', '<div class="section-title bt">Build a task')
s = s.replace("loadScenarios(); rotateGhost();\n</script>", "buildDucks(); loadScenarios(); rotateGhost();\n</script>", 1)
sub("</style>", """
.ducksel { display: none; flex-wrap: wrap; gap: 7px; align-items: center; padding: 10px 2px 2px; }
.ducksel.show { display: flex; }
.ducksel .dl { font-size: .75rem; opacity: .7; margin-right: 4px; width: 100%; }
.ducksel button { width: 40px; height: 40px; border-radius: 50%; border: 3px solid rgba(0,0,0,.35); font-weight: 900;
  font-size: .95rem; cursor: pointer; transition: transform .12s; }
.ducksel button.on { border-color: #fff; box-shadow: 0 0 0 3px var(--accent, #ffcc33); transform: scale(1.08); }
.simabout { display: none; font-size: .8rem; line-height: 1.4; opacity: .8; margin-top: 12px; padding: 12px 14px; }
body.is-sim .simabout { display: block; }
body.is-sim .builder, body.is-sim .bt, body.is-sim .side { display: none !important; }
body.is-sim .layout { grid-template-columns: minmax(0, 1fr) !important; max-width: 1180px; margin: 0 auto; }
@media (min-width: 960px) { body.is-sim .quick { grid-template-columns: repeat(5, 1fr); } }
</style>""")
# Recycling v3: a quick action with "big": true spans the whole row (e.g. "Sort the waste").
sub('b.className = "qa";\n  b.innerHTML = `<div class="em">${q.em}</div>', 'b.className = q.big ? "qa big" : "qa";\n  b.innerHTML = `<div class="em">${q.em}</div>')
sub("</style>", """.qa.big { grid-column: 1 / -1; border: 2px solid var(--accent, #2ec27e); }
.qa.big b { font-size: 1.15rem; }
</style>""")
# ---- ORDERBOX: Microducks agent order box (typed order -> Nemotron tool call -> live ducks)
sub('<div class="quick" id="quick"></div>', '<div class="quick" id="quick"></div>\n      <div class="card orderbox" id="orderBox">\n        <div class="ob-title">Type an order <span>&middot; NVIDIA Nemotron 3.5 Bolt turns it into a tool call</span></div>\n        <form id="obForm" autocomplete="off"><input id="obText" maxlength="200" placeholder="e.g. Regroup in the square" aria-label="Order for the ducks">\n          <button class="btn" id="obSend" type="submit">Send</button></form>\n        <div class="ob-out" id="obOut"></div>\n        <div class="chips ob-ex" id="obEx"></div>\n      </div>')
sub('</style>', '.orderbox { display: none; margin-top: 12px; padding: 12px 14px; }\nbody.is-sim .orderbox { display: block; }\n.orderbox .ob-title { font-weight: 800; font-size: .9rem; margin-bottom: 8px; }\n.orderbox .ob-title span { font-weight: 500; opacity: .7; font-size: .78rem; }\n.orderbox form { display: flex; gap: 8px; }\n.orderbox input { flex: 1; min-width: 0; font: inherit; color: inherit; background: rgba(255,255,255,.06);\n  border: 1px solid var(--line, rgba(255,255,255,.14)); border-radius: 12px; padding: 10px 12px; }\n.orderbox .ob-out { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem; margin-top: 8px; opacity: .9;\n  white-space: pre-wrap; word-break: break-word; min-height: 1.2em; }\n.orderbox .ob-ex { margin-top: 8px; }\n.orderbox .ob-ex button { font-size: .75rem; }\n</style>')
sub('/* status + feed */', '/* ORDERBOX */\nconst OB_EX = ["Regroup in the square", "Two racers to the west ramp start line", "Ready... go!",\n  "Follow duck 5 around the cones", "Left group take the west ramp, right group the north ramp", "Scatter!", "Stop!",\n  "Stress test: give duck 3 a light push", "Everyone, gather in the centre circle",\n  "Walk around the circle, clockwise", "Form a triangle", "Make a square"];\n(function () { const ex = $("#obEx"); if (!ex) return;\n  OB_EX.forEach(t => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t;\n    b.onclick = () => { $("#obText").value = t; $("#obText").focus(); }; ex.appendChild(b); });\n  $("#obForm").onsubmit = async e => { e.preventDefault(); const t = $("#obText").value.trim(); if (!t) return;\n    const btn = $("#obSend"); btn.disabled = true; $("#obOut").textContent = "Asking the agent..."; buzz(10);\n    try { const r = await api("/api/sim", { method: "POST", body: JSON.stringify({ action: "order", text: t, duck: DUCK }) });\n      $("#obOut").textContent = r.tool_call ? ("Tool call: " + r.tool_call + "\\n" + (r.navigator || "")) : (r.message || "No tool call");\n      if (r.tool_call) $("#obText").value = ""; refreshStatus(); }\n    catch (err) { $("#obOut").textContent = err.message || "Simulation offline"; }\n    finally { btn.disabled = false; } }; })();\n\n/* status + feed */')
# ---- KEYNOTE_TAB: per-scenario order examples, one-tap orders, agent feed
sub('<div class="ob-title">Type an order <span>&middot; NVIDIA Nemotron 3.5 Bolt turns it into a tool call</span></div>', '<div class="ob-title">Type an order <span id="obBrain">&middot; NVIDIA Nemotron 3.5 Bolt turns it into a tool call</span></div>')
sub('<div class="ob-out" id="obOut"></div>', '<div class="ob-out" id="obOut"></div>\n        <div class="ob-feed" id="obFeed"></div>')
sub('.orderbox .ob-ex button { font-size: .75rem; }', '.orderbox .ob-ex button { font-size: .75rem; }\n/* KEYNOTE_TAB: agent feed */\n.orderbox .ob-feed { margin-top: 10px; display: grid; gap: 6px; font-size: .78rem; }\n.orderbox .ob-feed:empty { display: none; }\n.orderbox .ob-feed > div { background: rgba(255,255,255,.05); border-radius: 10px; padding: 7px 10px; line-height: 1.35;\n  word-break: break-word; }\n.orderbox .ob-feed b { font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; opacity: .9; }\n.orderbox .ob-feed span { opacity: .5; font-size: .68rem; margin-left: 4px; }\n.orderbox .ob-feed .k-tool b, .orderbox .ob-feed .k-check b { color: var(--accent); }\n.orderbox .ob-feed .busy { opacity: .75; font-style: italic; }')
sub('  b.onclick = () => q.action ? simAction(q.action) : send(q.text); $("#quick").appendChild(b); }); }', '  b.onclick = () => q.order ? obOrder(q.order) : q.action ? simAction(q.action) : send(q.text); $("#quick").appendChild(b); }); }')
sub('  renderQuick(); renderChips(); renderSugg();', '  renderQuick(); renderOb(); renderChips(); renderSugg();')
sub('$("#resetSub").textContent = isSim() ? "Scatter all 12 ducks across the yard again" :', '$("#resetSub").textContent = isSim() ? (SPEC.reset_sub || "Scatter all 12 ducks across the yard again") :')
sub('(function () { const ex = $("#obEx"); if (!ex) return;\n  OB_EX.forEach(t => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t;\n    b.onclick = () => { $("#obText").value = t; $("#obText").focus(); }; ex.appendChild(b); });\n', '/* KEYNOTE_TAB: per-scenario order examples, one-tap orders, agent feed */\nconst OBK = { order: "Order", perception: "Perception tool", tool: "Action tool", reply: "Nemotron", check: "Camera check" };\nlet obFeedSig = "";\nfunction renderOb() { const ex = $("#obEx"); if (!ex) return; ex.innerHTML = "";\n  ((SPEC && SPEC.orders) || OB_EX).forEach(t => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t;\n    b.onclick = () => { $("#obText").value = t; $("#obText").focus(); }; ex.appendChild(b); });\n  const hint = $("#obBrain"); if (hint) hint.textContent = (SPEC && SPEC.order_hint) || "\\u00b7 NVIDIA Nemotron 3.5 Bolt turns it into a tool call";\n  obFeedSig = ""; $("#obFeed").innerHTML = ""; $("#obOut").textContent = ""; obFeedTick(); }\nfunction obOrder(t) { $("#obText").value = t; $("#obForm").requestSubmit(); }\nasync function obFeedTick() { if (!SPEC || !SPEC.agent_feed || document.hidden) return; const want = SCN;\n  try { const r = await api("/api/sim/info"); if (want !== SCN) return; const ag = r.agent || {}; const fd = ag.feed || [];\n    const sig = fd.map(x => x.id).join(",") + (ag.busy ? "b" : ""); if (sig === obFeedSig) return; obFeedSig = sig;\n    $("#obFeed").innerHTML = (ag.busy ? \'<div class="busy">Nemotron is working on an order...</div>\' : "") +\n      fd.slice(-6).reverse().map(x => `<div class="k-${esc(x.kind)}"><b>${esc(OBK[x.kind] || x.kind)}</b><span>${esc(x.wall)}</span><br>${esc(x.text)}</div>`).join("");\n  } catch {} }\nsetInterval(obFeedTick, 1500);\n(function () { if (!$("#obEx")) return;\n')
# ---- LOWDATA_ONECAM: bad-connection video modes and the one-camera operator switch
sub('<div class="tr"><button id="fullBtn" aria-label="Fullscreen">', '<div class="tr"><button class="qualchip" id="qualChip" aria-label="Video quality">AUTO</button><button id="fullBtn" aria-label="Fullscreen">')
sub('  <button class="row" id="themeRow">&#x25D0;<div>Light theme<small>Remembered on this device</small></div><span class="switch" id="themeSwitch"></span></button>\n', '  <button class="row" id="themeRow">&#x25D0;<div>Light theme<small>Remembered on this device</small></div><span class="switch" id="themeSwitch"></span></button>\n  <div class="row" style="cursor:default">&#x25B6;<div>Video quality<small>Data saver and Bad connection send small pictures instead of live video and pause the thumbnails</small></div></div><div class="seg" id="qualSeg"><button data-q="auto">Auto</button><button data-q="low">Data saver</button><button data-q="min">Bad connection</button></div>\n')
sub('    <button class="row" id="clearRow">&#x2715;<div>Cancel all waiting requests<small>Requests from other people in the queue</small></div></button>\n', '    <button class="row" id="clearRow">&#x2715;<div>Cancel all waiting requests<small>Requests from other people in the queue</small></div></button>\n    <div class="row" style="cursor:default">&#x1F3A5;<div>Render one camera only<small>For everyone, in this cell: the other cameras stop streaming (and stop rendering in the duck cells)</small></div></div><div class="seg" id="oneCamSeg"></div>\n')
sub('.orderbox .ob-ex button { font-size: .75rem; }', '.orderbox .ob-ex button { font-size: .75rem; }\n/* LOWDATA_ONECAM */\n.stage .tr { display: flex; gap: 8px; }\n.stage .tr .qualchip { width: auto; padding: 0 10px; font-size: .7rem; font-weight: 800; letter-spacing: .06em; color: #fff; }\n.seg { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: -2px 0 10px; }\n.seg button { padding: 10px 4px; border-radius: 11px; background: var(--panel); border: 1px solid var(--line); font-size: .76rem;\n  font-weight: 700; }\n.seg button.on { background: rgba(118,185,0,.16); border-color: var(--g); }\n#oneCamSeg { grid-template-columns: repeat(auto-fill, minmax(86px, 1fr)); }\nbody.lowdata .thumbs, body.onecam .thumbs, body.onecam #dotsCam, body.onecam #prevCam, body.onecam #nextCam { display: none; }')
sub('let mainBusy = false;\nasync function loadMain(force) {\n  if (isSim()) return simStream(force);\n', '/* LOWDATA_ONECAM: video quality and the one-camera switch */\nconst QUALS = { auto: "Auto", low: "Data saver", min: "Bad connection" }, QCHIP = { auto: "AUTO", low: "LOW", min: "MIN" };\nlet quality = QUALS[localStorage.cascadeLabQuality] ? localStorage.cascadeLabQuality : "auto", ONLY = null;\nconst lowData = () => quality !== "auto";\nfunction setQuality(q, announce) { quality = q; localStorage.cascadeLabQuality = q;\n  document.querySelectorAll("#qualSeg button").forEach(b => b.classList.toggle("on", b.dataset.q === q));\n  $("#qualChip").textContent = QCHIP[q]; document.body.classList.toggle("lowdata", lowData());\n  const im = $("#camImg"); im.onerror = im.onload = null; simUrl = "";\n  if (announce) toast("Video: " + QUALS[q]); loadMain(true); }\nfunction applyOnly(c) { c = c && CAMS.includes(c) ? c : null;\n  if (c !== ONLY) { ONLY = c; document.body.classList.toggle("onecam", !!c); if (c && cam !== c) setCam(c, false); }\n  renderOneCam(); }\nfunction renderOneCam() { const s = $("#oneCamSeg"); if (!s) return;\n  const opts = [["", "All"]].concat(CAMS.map(c => [c, camLabel(c)])), sig = JSON.stringify(opts) + (ONLY || "");\n  if (s.dataset.sig === sig) return; s.dataset.sig = sig; s.innerHTML = "";\n  opts.forEach(([v, l]) => { const b = document.createElement("button"); b.textContent = l; b.classList.toggle("on", (ONLY || "") === v);\n    b.onclick = async () => { buzz(10); try { const r = await api("/api/admin/only_cam", { method: "POST", body: JSON.stringify({ cam: v }) });\n      toast(r.message); applyOnly(r.only_cam); } catch (err) { toast(err.message); } }; s.appendChild(b); }); }\nlet mainBusy = false;\nasync function loadMain(force) {\n  if (isSim() && !lowData()) return simStream(force);\n')
sub('    const r = await fetch(`/api/cam/${SCN}/${want}.jpg?t=${Date.now()}`, { headers: { "X-Client": clientId } });', '    const r = await fetch(`/api/cam/${SCN}/${want}.jpg?t=${Date.now()}${lowData() ? "&q=" + quality : ""}${camQ()}`, { headers: { "X-Client": clientId } });')
sub('async function loadThumbs() {\n  for (const t of thumbs.querySelectorAll(".thumb")) {', 'async function loadThumbs() { if (lowData() || ONLY) return;\n  for (const t of thumbs.querySelectorAll(".thumb")) {')
sub('(function camLoop() { if (!document.hidden) loadMain(false); setTimeout(camLoop, 280); })();', '(function camLoop() { if (!document.hidden) loadMain(false); setTimeout(camLoop, quality === "min" ? 1000 : quality === "low" ? 450 : 280); })();')
sub('(function thumbLoop() { if (!document.hidden) loadThumbs(); setTimeout(thumbLoop, 2500); })();', '(function thumbLoop() { if (!document.hidden) loadThumbs(); setTimeout(thumbLoop, 2500); })();\n$("#qualChip").onclick = e => { e.stopPropagation(); const order = ["auto", "low", "min"];\n  setQuality(order[(order.indexOf(quality) + 1) % order.length], true); };\ndocument.querySelectorAll("#qualSeg button").forEach(b => b.onclick = () => setQuality(b.dataset.q, true));\nsetQuality(quality, false);')
sub('function setCam(c, manual) {\n', 'function setCam(c, manual) {\n  if (ONLY && c !== ONLY) { if (manual) toast("Only the " + camLabel(ONLY) + " camera renders right now"); c = ONLY; }\n')
sub('  applyRole(status.role, status.exclusive);\n  const cur = status.current', '  applyRole(status.role, status.exclusive); applyOnly(status.only_cam);\n  const cur = status.current')
sub('buildCams(isSim() ? SPEC.cams : BASE_CAMS); updDuckSel();', 'buildCams(isSim() ? SPEC.cams : BASE_CAMS); ONLY = null; document.body.classList.remove("onecam"); renderOneCam(); updDuckSel();')
# ---- LOWDATA_ONECAM_2: no overlapping thumbnail passes
sub('async function loadThumbs() { if (lowData() || ONLY) return;\n  for (const t of thumbs.querySelectorAll(".thumb")) {', '/* LOWDATA_ONECAM_2 */\nasync function loadThumbs() { if (thumbsBusy || lowData() || ONLY) return; thumbsBusy = true;\n  try { await thumbsPass(); } finally { thumbsBusy = false; } }\nasync function thumbsPass() {\n  for (const t of thumbs.querySelectorAll(".thumb")) { if (lowData() || ONLY) return;')
sub('let mainBusy = false;\nasync function loadMain(force) {', 'let mainBusy = false, thumbsBusy = false;\nasync function loadMain(force) {')
# ---- LOWDATA_ONECAM_3: scrollable menu sheet
sub('body.lowdata .thumbs, body.onecam .thumbs, body.onecam #dotsCam, body.onecam #prevCam, body.onecam #nextCam { display: none; }', 'body.lowdata .thumbs, body.onecam .thumbs, body.onecam #dotsCam, body.onecam #prevCam, body.onecam #nextCam { display: none; }\n/* LOWDATA_ONECAM_3 */\n.sheet { max-height: 88vh; max-height: 88dvh; overflow-y: auto; overscroll-behavior: contain; -webkit-overflow-scrolling: touch; }')
Path(__file__).with_name("static").joinpath("app.html").write_text(s, encoding="utf-8")
print("lab app written", len(s))
