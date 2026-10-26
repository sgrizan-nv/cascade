"""Agent layer: typed order -> NVIDIA Nemotron 3.5 Bolt (vLLM, OpenAI-compatible tools) -> tool call -> navigator events.

The operator only decides WHEN an order is typed and its exact text. Nemotron picks the tool and its arguments; this module
maps the tool call onto the navigator (director.py) with the same steering and the same ONNX policy. Every order is
logged (text, raw model response, parsed tool call, resulting events, timestamps).
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np

URL = "http://127.0.0.1:18310/v1/chat/completions"
MODEL = "nemotron-3.5-bolt"
SYSTEM = ("You control 12 small walking robot ducks (numbered 1-12) in a schoolyard simulation. The yard has a centre "
          "circle, a painted square with four boxes (A-D) in the south-west, cones, and three ramps (north, west, "
          "south). Every duck walks with the same neural network policy; you only decide what the group does. For each "
          "order, call exactly one tool. Ducks already busy (racing, in a line, following a leader) are left out of "
          "group moves such as regroup, scatter or gather. Use push_duck only when the operator explicitly asks for a "
          "push or a stress test. Head colours: 1 red, 2 orange, 3 yellow, 4 green, 5 teal, 6 cyan, 7 blue, 8 "
          "purple, 9 pink, 10 white, 11 black, 12 brown.")
TOOLS = [
    {"type": "function", "function": {"name": "gather_center", "description": "All ducks walk to the centre circle and stand on it.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "cross_ramp", "description": "Ducks walk over a ramp, either as a single-file line or as a loose group.",
                                      "parameters": {"type": "object", "properties": {
                                          "ramp": {"type": "string", "enum": ["north", "west", "south"]},
                                          "formation": {"type": "string", "enum": ["line", "group"]},
                                          "count": {"type": "integer", "minimum": 1, "maximum": 12, "description": "how many ducks; omit it to send the ducks that are already lined up"}},
                                          "required": ["ramp", "formation"]}}},
    {"type": "function", "function": {"name": "line_up", "description": "Ducks line up in a row and wait.",
                                      "parameters": {"type": "object", "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 12}}}}},
    {"type": "function", "function": {"name": "stop_all", "description": "Every duck stops and stands still.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "resume_wander", "description": "Ducks go back to wandering around the yard.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "regroup", "description": "Free ducks walk to a meeting place and stand there in a tidy group.",
                                      "parameters": {"type": "object", "properties": {
                                          "place": {"type": "string", "enum": ["square", "circle"]}}, "required": ["place"]}}},
    {"type": "function", "function": {"name": "scatter", "description": "Free ducks spread out across the yard, each to its own spot.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "race_start_line", "description": "Racers walk to the two lane starts in front of a ramp and wait for the start signal.",
                                      "parameters": {"type": "object", "properties": {
                                          "ramp": {"type": "string", "enum": ["north", "west"]},
                                          "racers": {"type": "integer", "minimum": 1, "maximum": 2}}, "required": ["ramp"]}}},
    {"type": "function", "function": {"name": "race_go", "description": "Start signal: the racers waiting at the start line run their lanes over the ramp.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "follow_leader", "description": "One duck leads and four others line up behind it and follow it on a route.",
                                      "parameters": {"type": "object", "properties": {
                                          "leader_id": {"type": "integer", "minimum": 1, "maximum": 12},
                                          "route": {"type": "string", "enum": ["around_cones", "over_north_ramp"]}}, "required": ["leader_id"]}}},
    {"type": "function", "function": {"name": "split_cross", "description": "Split the free ducks into a left (west) half and a right (east) half; each half crosses its own ramp.",
                                      "parameters": {"type": "object", "properties": {
                                          "left_ramp": {"type": "string", "enum": ["west", "north"]},
                                          "right_ramp": {"type": "string", "enum": ["west", "north"]}},
                                          "required": ["left_ramp", "right_ramp"]}}},
    {"type": "function", "function": {"name": "circle_walk", "description": "Ducks standing on the centre circle walk around it together.",
                                      "parameters": {"type": "object", "properties": {
                                          "direction": {"type": "string", "enum": ["clockwise", "counterclockwise"]},
                                          "laps": {"type": "number", "minimum": 0.5, "maximum": 2}}, "required": ["direction"]}}},
    {"type": "function", "function": {"name": "form_shape", "description": "Free ducks walk to evenly spaced places on a shape in the middle of the yard.",
                                      "parameters": {"type": "object", "properties": {
                                          "shape": {"type": "string", "enum": ["triangle", "line", "square"]}}, "required": ["shape"]}}},
    {"type": "function", "function": {"name": "send_duck", "description": "Send one duck, by number, to a named place; it walks there alone and waits there (group moves leave it out) until it is sent somewhere else. The others stay where they are.",
                                      "parameters": {"type": "object", "properties": {
                                          "duck_id": {"type": "integer", "minimum": 1, "maximum": 12},
                                          "place": {"type": "string", "enum": ["north_ramp_top", "west_ramp_top", "south_ramp_top", "circle"],
                                                    "description": "circle = back to its own spot on the centre circle"}},
                                          "required": ["duck_id", "place"]}}},
    {"type": "function", "function": {"name": "push_duck", "description": "Operator stress test: give one duck a sideways shove.",
                                      "parameters": {"type": "object", "properties": {
                                          "duck_id": {"type": "integer", "minimum": 1, "maximum": 12},
                                          "strength": {"type": "string", "enum": ["light", "strong"]}},
                                          "required": ["duck_id"]}}},
]


HISTORY: list = []


def ask(text, timeout=60):
    msgs = [{"role": "system", "content": SYSTEM}] + HISTORY + [{"role": "user", "content": text}]
    body = {"model": MODEL, "messages": msgs,
            "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, time.time() - t0


def parse(resp):
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, msg.get("content")
    fn = calls[0]["function"]
    args = fn.get("arguments") or "{}"
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except Exception:  # noqa: BLE001
        return None, f"unparseable arguments: {args}"
    return {"name": fn["name"], "arguments": args}, None


def call_str(call):
    a = ", ".join(f'{k}={json.dumps(v)}' for k, v in call["arguments"].items())
    return f'{call["name"]}({a})'


def to_events(call, t, director):
    """Map a tool call onto navigator events (absolute sim times). Returns (events, note)."""
    D = director
    name, a = call["name"], call["arguments"]

    def idle(res):
        ev, note = res
        return ([(t, "mode", "idle")] if D.mode == "stop" or name != "resume_wander" else []) + ev, note
    if name == "gather_center":
        D.busy, D.lined, D.racers = set(getattr(D, "parked", set())), [], []   # lines, racers, followers join in; parked ducks wait
        return idle((D.agent_gather(t), "all ducks to slots on a 0.65 m ring around the centre circle"))
    if name == "cross_ramp":
        ramp, form = a.get("ramp", "north"), a.get("formation", "group")
        n = int(a.get("count") or (5 if form == "line" else 12))
        return idle(D.agent_cross(t, ramp, form, n))
    if name == "line_up":
        n = int(a.get("count") or 5)
        return idle((D.agent_line_up(t, n), f"{n} ducks to queue slots 42 cm apart"))
    if name == "stop_all":
        return D.agent_stop(t)
    if name == "resume_wander":
        return D.agent_wander(t), "release holds, wander mode"
    if name == "regroup":
        if a.get("place") == "circle":  # the circle is everyone's meeting place: lines and racers join in too
            D.busy, D.lined, D.racers = set(getattr(D, "parked", set())), [], []
        return idle(D.agent_regroup(t, a.get("place", "square")))
    if name == "scatter":
        return idle(D.agent_scatter(t))
    if name == "race_start_line":
        return idle(D.agent_race_start(t, a.get("ramp", "west"), int(a.get("racers") or 2)))
    if name == "race_go":
        return idle(D.agent_race_go(t))
    if name == "follow_leader":
        return idle(D.agent_follow(t, int(a.get("leader_id", 4)) - 1, a.get("route", "around_cones")))
    if name == "split_cross":
        return idle(D.agent_split(t, a.get("left_ramp", "west"), a.get("right_ramp", "north")))
    if name == "circle_walk":
        return idle(D.agent_circle(t, a.get("direction", "clockwise"), a.get("laps") or 1))
    if name == "form_shape":
        return idle(D.agent_shape(t, a.get("shape", "triangle")))
    if name == "send_duck":
        return idle(D.agent_send(t, int(a.get("duck_id", 7)) - 1, a.get("place", "circle")))
    if name == "push_duck":
        i = int(a.get("duck_id", 3)) - 1
        mag = 1.6 if a.get("strength", "strong") == "strong" else 1.0
        return [(t, "push", i, mag)], f"operator stress test: {mag} m/s sideways shove on duck {i + 1}"
    return [], f"unknown tool {name}: nothing executed"


def decide(text):
    """Ask Nemotron (at most 2 attempts). Returns a record with the raw message and the parsed call (or None)."""
    rec = {"wall": time.strftime("%H:%M:%S"), "order": text}
    for attempt in (1, 2):
        try:
            resp, dt = ask(text)
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"{type(e).__name__}: {e}", attempt=attempt)
            continue
        call, err = parse(resp)
        rec.update(attempt=attempt, latency_s=round(dt, 2), raw=resp["choices"][0]["message"])
        if call is not None:
            rec.update(tool_call=call, tool_call_str=call_str(call))
            return rec
        rec.update(parse_error=err)
    return rec


def remember(rec, note, keep=None):
    msg = rec["raw"]
    tc = (msg.get("tool_calls") or [{}])[0]
    HISTORY.extend([{"role": "user", "content": rec["order"]},
                    {"role": "assistant", "content": msg.get("content") or "", "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": tc.get("id", "call_0"), "content": note}])
    if keep:
        del HISTORY[:-keep]


def execute(rec, t, director):
    """Map a decided tool call onto navigator events; updates rec in place."""
    ev, note = to_events(rec["tool_call"], t, director)
    remember(rec, note, keep=None)
    rec.update(t_sim=round(t, 2), navigator=note,
               events=[[round(e[0], 2), e[1]] + [x if not isinstance(x, np.ndarray) else x.tolist() for x in e[2:3]]
                       for e in ev])
    return ev


def handle(text, t, director, log):
    rec = decide(text)
    rec["t_sim"] = round(t, 2)
    if rec.get("tool_call") is None:
        log.append(rec)
        return []
    ev = execute(rec, t, director)
    log.append(rec)
    return ev
