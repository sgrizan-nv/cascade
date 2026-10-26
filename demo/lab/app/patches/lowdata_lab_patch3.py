"""Follow-up to lowdata_lab_patch.py: the menu sheet scrolls when it is taller than the phone screen (the new video
quality and one-camera rows pushed the operator rows off the bottom).

    python3 lowdata_lab_patch3.py <app_dir>
Idempotent (marker LOWDATA_ONECAM_3). ASCII only.
"""
import sys
from pathlib import Path

app = Path(sys.argv[1])
SUBS = [
    ('body.lowdata .thumbs, body.onecam .thumbs, body.onecam #dotsCam, body.onecam #prevCam, body.onecam #nextCam { display: none; }',
     'body.lowdata .thumbs, body.onecam .thumbs, body.onecam #dotsCam, body.onecam #prevCam, body.onecam #nextCam { display: none; }\n'
     '/* LOWDATA_ONECAM_3 */\n'
     '.sheet { max-height: 88vh; max-height: 88dvh; overflow-y: auto; overscroll-behavior: contain; -webkit-overflow-scrolling: touch; }'),
]


def apply(text):
    for old, new in SUBS:
        if text.count(old) != 1:
            raise SystemExit(f"expected 1 x {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


html = app / "static" / "app.html"
h = html.read_text(encoding="utf-8")
if "LOWDATA_ONECAM_3" not in h:
    html.write_text(apply(h), encoding="utf-8")
    print("app.html patched")

mk = app / "make_lab_app.py"
m = mk.read_text()
if "LOWDATA_ONECAM_3" not in m:
    tail = 'Path(__file__).with_name("static").joinpath("app.html").write_text(s, encoding="utf-8")'
    assert m.count(tail) == 1, "make_lab_app tail"
    blk = "# ---- LOWDATA_ONECAM_3: scrollable menu sheet\n"
    blk += "".join(f"sub({old!r}, {new!r})\n" for old, new in SUBS)
    mk.write_text(m.replace(tail, blk + tail))
    print("make_lab_app.py patched")
mk.read_text().encode("ascii")
print("ascii ok")
