#!/usr/bin/env python3
"""Summary of soak JSONL files: per order, right tool share, camera check share, failures; resets; lights."""
import collections
import json
import sys

rows = []
for p in sys.argv[1:]:
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
orders = [r for r in rows if r.get("step") == "order"]
by = collections.defaultdict(list)
for r in orders:
    by[r["text"]].append(r)
print(f"{len(orders)} orders, {sum(1 for r in rows if r.get('step') == 'soft_reset')} soft resets, "
      f"{sum(1 for r in rows if r.get('step') == 'hard_reset')} hard resets, from {rows[0]['wall'] if rows else '-'} to "
      f"{rows[-1]['wall'] if rows else '-'}")
bad_http = [r for r in rows if r.get("code") not in (200, None)]
print(f"HTTP errors: {len(bad_http)}", [(r['wall'][11:19], r.get('step'), r.get('code'), (r.get('message') or '')[:60]) for r in bad_http][:8])
for text, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    right = sum(1 for r in rs if r.get("right_tool"))
    chk = [r for r in rs if r.get("check_n")]
    full = sum(1 for r in chk if r["check_ok"] == r["check_n"])
    frac = sum(r["check_ok"] / r["check_n"] for r in chk) / len(chk) if chk else None
    secs = sorted(r.get("check_s") or 0 for r in chk)
    tools = collections.Counter((r.get("tool") or "none").split("(")[0] for r in rs)
    print(f"  {text[:58]:58s} n={len(rs):2d} right={right}/{len(rs)} check_full={full}/{len(chk)} "
          f"mean_ok={'-' if frac is None else f'{100 * frac:.0f}%'} check_s_med={secs[len(secs) // 2] if secs else '-'} {dict(tools)}")
lows = [r for r in rows if r.get("up") is not None and r["up"] < 12]
print(f"steps with a duck down: {len(lows)}", [(r['wall'][11:19], r.get('up')) for r in lows][:10])
dark = [r for r in rows if r.get("frame_bytes") == 0]
print(f"steps with no camera frame: {len(dark)}", [r['wall'][11:19] for r in dark][:10])
red = [r for r in rows if r.get("lights") and False in r["lights"].values()]
print(f"steps with a red light: {len(red)}", [(r['wall'][11:19], r['lights']) for r in red][:6])
rtf = [r["rtf"] for r in rows if r.get("rtf")]
if rtf:
    print(f"sim speed: min {min(rtf)} median {sorted(rtf)[len(rtf) // 2]}")
for r in rows:
    if r.get("step") == "hard_reset":
        print("hard reset", r["wall"][11:19], r.get("secs"), (r.get("reset") or {}).get("ok"), (r.get("reset") or {}).get("message"))
