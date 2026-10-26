#!/usr/bin/env python3
"""Before/after contact sheet for one Cascade Lab experiment run (English labels).

    python3 make_sheet.py RUN_DIR OUT.jpg "Title" [cam]
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = Path.home() / "Library/Fonts"


def font(name, size):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


def label(draw, xy, text, f):
    x, y = xy
    draw.text((x, y), text, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))


def main():
    run, out, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    cam = sys.argv[4] if len(sys.argv) > 4 else "worktop"
    summary = json.loads((run / "summary.json").read_text())
    before = Image.open(run / f"before_{cam}.jpg").convert("RGB")
    after = Image.open(run / f"after_{cam}.jpg").convert("RGB")
    w, h = before.size
    bar = 74
    sheet = Image.new("RGB", (2 * w + 12, h + bar), (18, 20, 24))
    sheet.paste(before, (0, bar))
    sheet.paste(after, (w + 12, bar))
    d = ImageDraw.Draw(sheet)
    verdict = "PASS" if summary["pass"] else "FAIL"
    d.text((18, 14), title, font=font("NVIDIASans_Bd.ttf", 34), fill=(255, 255, 255))
    vf = font("NVIDIASans_Bd.ttf", 34)
    vw = d.textlength(verdict, font=vf)
    d.text((2 * w + 12 - vw - 18, 14), verdict, font=vf,
           fill=(118, 185, 0) if summary["pass"] else (230, 70, 60))
    f = font("NVIDIASans_Md.ttf", 28)
    label(d, (18, bar + 12), "BEFORE", f)
    label(d, (w + 30, bar + 12), "AFTER", f)
    small = font("NVIDIASans_Rg.ttf", 22)
    stack = os.environ.get("CASCADE_LAB_SHEET_STACK", "Isaac Sim 6.2 + Newton, live bridge camera")
    label(d, (18, bar + h - 40), stack, small)
    label(d, (w + 30, bar + h - 40), summary["rule"], small)
    sheet.save(out, quality=90)


if __name__ == "__main__":
    main()
