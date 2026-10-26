"""Agent layer: typed order -> NVIDIA Nemotron (OpenAI-compatible tools) -> perception tools / one action -> navigator.

The operator only decides WHEN an order is typed and its exact text. Nemotron may first call the perception tools
(perceived_state: positions and headings from the cameras; look: one yard-camera frame described by Nemotron's vision),
then picks exactly one action tool. After the navigator finishes, the camera verification goes back to Nemotron, which
answers the operator. Every request and reply is logged.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request

import numpy as np

BASE = os.environ.get("NEMOTRON_URL", "http://127.0.0.1:18310/v1").rstrip("/")
MODEL = os.environ.get("NEMOTRON_MODEL", "")
MODEL_NAME = "NVIDIA Nemotron 3.5 Bolt"
SYSTEM = ("You are NVIDIA Nemotron, the agent that commands 12 small walking robot ducks (numbered 1-12) in a yard. "
          "You never see the simulator: you only know what the yard cameras perceive. Four fixed RGB-D cameras "
          "(ne, nw, sw, se) mapped the yard before the ducks moved (walls, cones, planters and three ramps: north, "
          "west, south) and track every duck's position and heading from depth and segmentation. "
          "Tools: perceived_state() returns that perceived data; look(camera) shows you one camera image. "
          "For each typed order, call at most one perception tool if it helps, then exactly one action tool. If the "
          "operator only asks a question, call the perception tool and answer in one or two short sentences without an "
          "action tool. "
          "Every duck walks with the same neural network policy; you only decide what the group does. Use push_duck "
          "only when the operator explicitly asks for a push or a stress test. "
          "Going onto or up a ramp and staying there is hold_on_ramp; walking round in a circle or a parade is "
          "walk_circle; turning or spinning on the spot is spin. Orders can come in any language; always answer in "
          "English.")

ACTIONS = [
    {"type": "function", "function": {"name": "gather_center", "description": "All ducks walk to the centre circle and stand on it.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "line_up", "description": "Ducks line up in single file in the queue corridor in front of the north ramp and wait.",
                                      "parameters": {"type": "object", "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 5}}}}},
    {"type": "function", "function": {"name": "cross_ramp", "description": "The ducks that are lined up walk over a ramp in single file.",
                                      "parameters": {"type": "object", "properties": {
                                          "ramp": {"type": "string", "enum": ["north", "west", "south"]},
                                          "formation": {"type": "string", "enum": ["line", "group"]},
                                          "count": {"type": "integer", "minimum": 1, "maximum": 5, "description": "how many ducks; omit it to send the ducks already lined up"}},
                                          "required": ["ramp", "formation"]}}},
    {"type": "function", "function": {"name": "stop_all", "description": "Every duck stops and stands still.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "regroup", "description": "Free ducks walk to a meeting place and stand there in a tidy group.",
                                      "parameters": {"type": "object", "properties": {
                                          "place": {"type": "string", "enum": ["circle"]}}, "required": ["place"]}}},
    {"type": "function", "function": {"name": "scatter", "description": "Ducks spread out across the yard, each to its own free spot.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "resume_wander", "description": "Ducks go back to wandering around the yard.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "hold_on_ramp", "description": "Ducks walk up a ramp along its slope and stand still on its flat top until the next order (north ramp holds 3 ducks, west and south 2 each; 'all' fills every ramp, 7 ducks).",
                                      "parameters": {"type": "object", "properties": {
                                          "ramp": {"type": "string", "enum": ["north", "west", "south", "all"]},
                                          "count": {"type": "integer", "minimum": 1, "maximum": 7, "description": "how many ducks; omit it to fill the ramp"}},
                                          "required": ["ramp"]}}},
    {"type": "function", "function": {"name": "walk_circle", "description": "All 12 ducks walk round the centre circle together like a parade, then stop on it.",
                                      "parameters": {"type": "object", "properties": {
                                          "direction": {"type": "string", "enum": ["clockwise", "counterclockwise"]},
                                          "laps": {"type": "integer", "minimum": 1, "maximum": 3}}}}},
    {"type": "function", "function": {"name": "spin", "description": "Every duck turns round on the spot (spins in place), then stands still.",
                                      "parameters": {"type": "object", "properties": {
                                          "direction": {"type": "string", "enum": ["clockwise", "counterclockwise"]},
                                          "turns": {"type": "integer", "minimum": 1, "maximum": 2}}}}},
    {"type": "function", "function": {"name": "push_duck", "description": "Operator stress test: give one duck a sideways shove.",
                                      "parameters": {"type": "object", "properties": {
                                          "duck_id": {"type": "integer", "minimum": 1, "maximum": 12},
                                          "strength": {"type": "string", "enum": ["light", "strong"]}},
                                          "required": ["duck_id"]}}},
]
PERCEPTION = [
    {"type": "function", "function": {"name": "perceived_state",
                                      "description": "What the yard cameras perceive right now: every duck's position (m), heading (deg) and whether it walks or stands, plus the mapped ramps, cones and walls and the last camera check.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "look",
                                      "description": "Show you one image from a fixed yard camera so you can describe the scene.",
                                      "parameters": {"type": "object", "properties": {
                                          "camera": {"type": "string", "enum": ["ne", "nw", "sw", "se"]},
                                          "question": {"type": "string", "description": "what to look for"}},
                                          "required": ["camera"]}}},
]
TOOLS = PERCEPTION + ACTIONS
ACTION_NAMES = {t["function"]["name"] for t in ACTIONS}

HISTORY: list = []
SENSORS = {}            # set by the simulator: perceived_state() -> dict, rgb(camera) -> HxWx3 uint8


def model_id():
    global MODEL
    if MODEL:
        return MODEL
    try:
        with urllib.request.urlopen(BASE + "/models", timeout=5) as r:
            MODEL = json.load(r)["data"][0]["id"]
    except Exception:  # noqa: BLE001
        MODEL = "nemotron-3.5-bolt"
    return MODEL


def chat(msgs, tools=None, max_tokens=320, timeout=25):
    body = {"model": model_id(), "messages": msgs, "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, time.time() - t0


def call_str(call):
    a = ", ".join(f'{k}={json.dumps(v)}' for k, v in call["arguments"].items())
    return f'{call["name"]}({a})'


def _parse(msg):
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0]["function"]
    args = fn.get("arguments") or "{}"
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except Exception:  # noqa: BLE001
        args = {}
    return {"name": fn["name"], "arguments": args or {}, "id": calls[0].get("id") or "call_0"}


def jpeg_b64(img, q=88):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def look(camera, question=""):
    """One camera frame to Nemotron's vision (one image per prompt). Returns (text, record)."""
    img = SENSORS["rgb"](camera)
    q = question or "Describe what you see in this yard camera image."
    prompt = (f"This is the RGB image of the fixed yard camera '{camera}' (simulated RGB-D camera). {q} Answer in at "
              "most two short sentences, concrete: how many robot ducks, where they are, what they do.")
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64(img)}}]}]
    resp, dt = chat(msgs, max_tokens=160)
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    rec = dict(camera=camera, question=q, prompt=prompt, latency_s=round(dt, 2), image_hw=list(img.shape[:2]), answer=text)
    if SENSORS.get("look_dir"):
        os.makedirs(SENSORS["look_dir"], exist_ok=True)
        k = len([f for f in os.listdir(SENSORS["look_dir"]) if f.endswith(".jpg")])
        path = os.path.join(SENSORS["look_dir"], f"look_{k:02d}_{camera}.jpg")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(jpeg_b64(img)))
        rec["image"] = path
        with open(path.replace(".jpg", ".json"), "w") as fh:
            json.dump(rec, fh, indent=1)
    return text, rec


def run_perception_tool(call):
    if call["name"] == "perceived_state":
        st = SENSORS["perceived_state"]()
        return json.dumps(st, separators=(",", ":")), dict(name="perceived_state", chars=len(json.dumps(st)))
    cam = call["arguments"].get("camera", "ne")
    text, rec = look(cam, call["arguments"].get("question", ""))
    return text, dict(name="look", **rec)


def decide(text):
    """Nemotron turns one typed order into (perception tool calls) + one action. Returns the record."""
    rec = {"wall": time.strftime("%H:%M:%S"), "order": text, "model": MODEL_NAME, "model_id": model_id(), "steps": []}
    msgs = [{"role": "system", "content": SYSTEM}] + HISTORY + [{"role": "user", "content": text}]
    t0 = time.time()
    for step in range(4):
        if time.time() - t0 > 40:
            rec["steps"].append(dict(error="TimeBudget: no action after 40 s"))
            break
        try:
            resp, dt = chat(msgs, tools=TOOLS)
        except Exception as e:  # noqa: BLE001
            rec["steps"].append(dict(error=f"{type(e).__name__}: {e}"))
            continue
        msg = resp["choices"][0]["message"]
        call = _parse(msg)
        rec["steps"].append(dict(latency_s=round(dt, 2), raw=msg, usage=resp.get("usage")))
        if call is None:
            rec["reply"] = msg.get("content")
            break
        tc = {"id": call["id"], "type": "function",
              "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
        if call["name"] in ACTION_NAMES:
            rec.update(tool_call={"name": call["name"], "arguments": call["arguments"]}, tool_call_str=call_str(call),
                       _tc=tc, _msgs=msgs + [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": [tc]}])
            break
        result, prec = run_perception_tool(call)
        rec["steps"][-1].update(perception_call=call_str(call), perception=prec)
        rec.setdefault("perception_calls", []).append(call_str(call))
        msgs = msgs + [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": [tc]},
                       {"role": "tool", "tool_call_id": call["id"], "content": result}]
    rec["latency_s"] = round(time.time() - t0, 2)
    rec["_turn"] = msgs[len(HISTORY) + 1:]
    return rec


def remember(rec, note):
    """Keep the turn (order, perception calls, action call, navigator note) in the history."""
    turn = rec.pop("_msgs", None)
    tc = rec.pop("_tc", None)
    rec.pop("_turn", None)
    if turn is None:
        return
    HISTORY.extend(turn[1 + len(HISTORY):])
    HISTORY.append({"role": "tool", "tool_call_id": tc["id"], "content": note})
    # perception payloads are large: keep only the latest one verbatim
    for m in HISTORY[:-6]:
        if m.get("role") == "tool" and len(m.get("content", "")) > 400:
            m["content"] = m["content"][:400] + "..."


def _int(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def to_events(call, t, D):
    name, a = call["name"], dict(call.get("arguments") or {})
    for k in ("count", "laps", "turns", "duck_id"):
        if k in a:
            a[k] = _int(a[k], None)
    pre = [(t, "mode", "idle")]
    D.check_after = None
    if name == "hold_on_ramp":
        D.busy, D.lined = set(), []
        ramp = str(a.get("ramp") or "north").lower()
        ramp = ramp if ramp in ("north", "west", "south", "all") else "north"
        ev, note = D.agent_ramp_hold(t, ramp, a.get("count") or 0)
        return pre + ev, note
    if name == "walk_circle":
        D.busy, D.lined = set(), []
        ev, note = D.agent_circle_walk(t, str(a.get("direction") or "counterclockwise").lower(), a.get("laps") or 2)
        return pre + ev, note
    if name == "spin":
        D.busy, D.lined = set(), []
        ev, note = D.agent_spin(t, str(a.get("direction") or "counterclockwise").lower(), a.get("turns") or 1)
        return pre + ev, note
    if name == "gather_center":
        D.busy, D.lined = set(), []
        ev, note = D.agent_gather(t)
        return pre + ev, note
    if name == "line_up":
        ev, note = D.agent_line_up(t, int(a.get("count") or 5))
        return pre + ev, note
    if name == "cross_ramp":
        n = min(max(int(a.get("count") or (len(D.lined) or 5)), 1), 5)
        ev, note = D.agent_cross(t, str(a.get("ramp") or "north"), str(a.get("formation") or "line"), n)
        return pre + ev, note
    if name == "stop_all":
        ev, note = D.agent_stop(t)
        return ev, note
    if name == "regroup":
        D.busy, D.lined = set(), []
        ev, note = D.agent_gather(t)
        return pre + ev, note.replace("to the nearest", "regroup: nearest")
    if name == "scatter":
        D.busy, D.lined = set(), []
        ev, note = D.agent_scatter(t)
        return pre + ev, note
    if name == "resume_wander":
        return D.agent_wander(t), "release holds, wander mode"
    if name == "push_duck":
        i = min(max(int(a.get("duck_id") or 3), 1), 12) - 1
        mag = 1.6 if a.get("strength", "strong") == "strong" else 1.0
        return [(t, "push", i, mag)], f"operator stress test: {mag} m/s sideways shove on duck {i + 1}"
    return [], f"unknown tool {name}: nothing executed"


def execute(rec, t, director):
    ev, note = to_events(rec["tool_call"], t, director)
    remember(rec, note)
    rec.update(t_sim=round(t, 2), navigator=note,
               events=[[round(e[0], 2), e[1]] + [x if not isinstance(x, np.ndarray) else x.tolist() for x in e[2:3]]
                       for e in ev])
    return ev


def answer(check, order):
    """Camera verification -> Nemotron's one-line answer to the operator."""
    payload = json.dumps(check, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    msgs = [{"role": "system", "content": SYSTEM}] + HISTORY + [
        {"role": "user", "content": f"[camera check after '{order}'] {payload}\nTell the operator the result in one "
                                    "short sentence that gives the count as n/N; start with 'Done:' only if ok equals n, "
                                    "otherwise start with 'Not yet:'."}]
    try:
        resp, dt = chat(msgs, max_tokens=60, timeout=20)
        text = (resp["choices"][0]["message"].get("content") or "").strip().split("\n")[0]
    except Exception as e:  # noqa: BLE001
        text, dt = f"(no answer: {type(e).__name__})", 0.0
    HISTORY.append({"role": "user", "content": f"[camera check after '{order}'] ok {check['ok']}/{check['n']}"})
    HISTORY.append({"role": "assistant", "content": text})
    return dict(order=order, check=check, answer=text, latency_s=round(dt, 2), wall=time.strftime("%H:%M:%S"))


def handle(text, t, director, log):
    rec = decide(text)
    rec["t_sim"] = round(t, 2)
    if rec.get("tool_call") is None:
        rec.pop("_turn", None)
        log.append(rec)
        return []
    ev = execute(rec, t, director)
    log.append(rec)
    return ev
