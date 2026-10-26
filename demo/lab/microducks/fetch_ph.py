"""Fetch Poly Haven (CC0) texture maps: diffuse, roughness, OpenGL normal at 2k jpg."""
import json, sys, urllib.request
UA = {"User-Agent": "microducks-ovrtx/1.0 (CC0 texture fetch)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)
from pathlib import Path
out = Path(sys.argv[1])
for aid in sys.argv[2:]:
    d = out / aid
    d.mkdir(parents=True, exist_ok=True)
    files = json.load(get(f"https://api.polyhaven.com/files/{aid}"))
    for key, name in (("Diffuse", "diff"), ("Rough", "rough"), ("nor_gl", "nor_gl")):
        try:
            url = files[key]["2k"]["jpg"]["url"]
        except KeyError:
            print(aid, "no", key); continue
        dst = d / f"{aid}_{name}_2k.jpg"
        if not dst.exists():
            dst.write_bytes(get(url).read())
        print(aid, key, dst.stat().st_size)
