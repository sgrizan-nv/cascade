#!/usr/bin/env python3
"""Call one CASCADE MCP tool inside a lab cell, without the LLM in the loop.

Runs inside the container (docker exec -i ... python - < mcp_direct.py) with the same server
command, arguments, working directory and environment that OpenClaw uses. Tests motion skills
that the visitor instructions do not expose yet. Prints one line: DIRECT <json result>.

    LAB_TOOL=pick_and_place LAB_ARGS='{"object": "red collar", "destination": "steel peg"}'
"""
import json
import os
import select
import subprocess
import time

cfg = json.load(open("/data/private/openclaw/openclaw.json"))
srv = cfg["mcp"]["servers"]["cascade"]
env = dict(os.environ)
env.update(srv.get("env", {}))
proc = subprocess.Popen([srv["command"], *srv["args"]], cwd=srv["cwd"], env=env, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)


def send(msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def recv(msg_id, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 1)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("DIRECT " + json.dumps({"error": "MCP server closed its stdout"}))
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("id") == msg_id:
            return msg
    raise SystemExit("DIRECT " + json.dumps({"error": f"no reply to request {msg_id} in {timeout} s"}))


send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                 "clientInfo": {"name": "cascade-lab-direct", "version": "1"}}})
recv(1, 240)
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
listed = recv(2, 240)["result"]["tools"]
names = [t["name"] for t in listed]
tool = os.environ["LAB_TOOL"]
if tool == "__list__":
    raise SystemExit("DIRECT " + json.dumps({"tools": names}))
if tool.startswith("__schema__:"):
    wanted = tool.split(":", 1)[1].split(",")
    raise SystemExit("DIRECT " + json.dumps({t["name"]: {"description": t.get("description", "")[:900],
                                                         "input": t.get("inputSchema")}
                                             for t in listed if t["name"] in wanted}))
calls = (json.loads(os.environ["LAB_CALLS"]) if tool == "__seq__"
         else [[tool, json.loads(os.environ.get("LAB_ARGS", "{}"))]])
missing = [name for name, _ in calls if name not in names]
if missing:
    raise SystemExit("DIRECT " + json.dumps({"error": f"tools not offered: {missing}", "tools": names}))
results = []
for index, (name, arguments) in enumerate(calls):
    t0 = time.time()
    send({"jsonrpc": "2.0", "id": 3 + index, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    reply = recv(3 + index, int(os.environ.get("LAB_TIMEOUT", "900")))
    texts = [c.get("text", "") for c in (reply.get("result") or {}).get("content", []) if c.get("type") == "text"]
    try:
        result = json.loads(texts[0])
    except (IndexError, ValueError):
        result = {"text": "\n".join(texts)[:4000], "error": reply.get("error")}
    results.append({"tool": name, "seconds": round(time.time() - t0, 1), "result": result})
print("DIRECT " + json.dumps(results[0] if len(results) == 1 else {"tool": tool, "steps": results,
                                                                     "result": results[-1]["result"]}))
proc.stdin.close()
try:
    proc.wait(timeout=60)
except subprocess.TimeoutExpired:
    proc.terminate()
