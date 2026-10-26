#!/usr/bin/env python3
"""Download the two Poly Haven CC0 HDRIs listed in textures/SOURCES.json and check their md5.

The procedural maps come from make_lab_textures.py and make_recycling_v2_textures.py (byte-identical to
the sha256 in procedural_textures.json / recycling_v2_textures.json). Missing files fall back to plain
materials in demo/lab_scenes.py.

    python demo/lab/fetch_lab_hdris.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

TEX = Path(__file__).resolve().parent / "textures"


def main():
    sources = json.loads((TEX / "SOURCES.json").read_text())["hdri"]
    bad = 0
    for name, meta in sources.items():
        dst = TEX / name
        if not dst.exists():
            req = urllib.request.Request(meta["download"], headers={"User-Agent": "cascade-lab/1.0 (CC0 HDRI fetch)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                dst.write_bytes(r.read())
        md5 = hashlib.md5(dst.read_bytes()).hexdigest()
        ok = md5 == meta["md5"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name} {dst.stat().st_size} B md5 {md5}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
