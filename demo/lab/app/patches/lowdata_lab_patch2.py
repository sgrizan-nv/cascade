"""Follow-up to lowdata_lab_patch.py: one thumbnail pass at a time, and a running pass stops as soon as a low-data
mode or the one-camera switch turns on (slow OVRTX snapshots made 2.5 s passes pile up).

    python3 lowdata_lab_patch2.py <app_dir>
Idempotent (marker LOWDATA_ONECAM_2). ASCII only.
"""
import sys
from pathlib import Path

app = Path(sys.argv[1])
SUBS = [
    ('async function loadThumbs() { if (lowData() || ONLY) return;\n  for (const t of thumbs.querySelectorAll(".thumb")) {',
     '/* LOWDATA_ONECAM_2 */\n'
     'async function loadThumbs() { if (thumbsBusy || lowData() || ONLY) return; thumbsBusy = true;\n'
     '  try { await thumbsPass(); } finally { thumbsBusy = false; } }\n'
     'async function thumbsPass() {\n'
     '  for (const t of thumbs.querySelectorAll(".thumb")) { if (lowData() || ONLY) return;'),
    ('let mainBusy = false;\nasync function loadMain(force) {',
     'let mainBusy = false, thumbsBusy = false;\nasync function loadMain(force) {'),
]


def apply(text):
    for old, new in SUBS:
        if text.count(old) != 1:
            raise SystemExit(f"expected 1 x {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


html = app / "static" / "app.html"
h = html.read_text(encoding="utf-8")
if "LOWDATA_ONECAM_2" not in h:
    assert "LOWDATA_ONECAM" in h, "run lowdata_lab_patch.py first"
    html.write_text(apply(h), encoding="utf-8")
    print("app.html patched")

mk = app / "make_lab_app.py"
m = mk.read_text()
if "LOWDATA_ONECAM_2" not in m:
    tail = 'Path(__file__).with_name("static").joinpath("app.html").write_text(s, encoding="utf-8")'
    assert m.count(tail) == 1, "make_lab_app tail"
    blk = "# ---- LOWDATA_ONECAM_2: no overlapping thumbnail passes\n"
    blk += "".join(f"sub({old!r}, {new!r})\n" for old, new in SUBS)
    mk.write_text(m.replace(tail, blk + tail))
    print("make_lab_app.py patched")
mk.read_text().encode("ascii")
print("ascii ok")
