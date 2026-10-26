"""Procedural maps for the recycling cell v2 (conveyor belt, bin labels, belt sign).

Run: python3 demo/lab/make_recycling_v2_textures.py  (needs numpy + Pillow).
Original artwork, CC0. Text is NVIDIA Sans Bold if installed, else Liberation/DejaVu Sans Bold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "textures"
FONTS = [Path.home() / "Library/Fonts/NVIDIASans_Bd.ttf",
         Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]


def font(size):
    for f in FONTS:
        if f.is_file():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


def noise(shape, scale, rng):
    """Smooth value noise (bilinear upsampled random grid)."""
    h, w = shape
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    grid = rng.random((gh, gw)).astype(np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, np.float32) / 255.


def belt(rng):
    """1024 (along the belt, 1.0 m) x 512 (across the belt, full width) rubber belt."""
    h, w = 512, 1024
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = .085 + .018 * noise((h, w), 64, rng) + .012 * noise((h, w), 12, rng)
    # fabric impression: fine diagonal weave
    weave = .006 * np.sin(x * .9 + y * .9) * np.sin(x * .9 - y * .9)
    # longitudinal wear streaks (items ride in the middle lanes)
    lanes = noise((1, w), 1, rng)[0] * 0 + noise((h, 1), 6, rng)[:, :1]
    wear = .03 * np.clip(lanes - .45, 0, 1) * (1 + .5 * noise((h, w), 200, rng))
    # scuffs: short pale arcs
    scuff = np.zeros((h, w), np.float32)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(70):
        cx, cy, r = rng.uniform(0, w), rng.uniform(40, h - 40), rng.uniform(8, 40)
        a0 = rng.uniform(0, 360)
        d.arc([cx - r, cy - r * .4, cx + r, cy + r * .4], a0, a0 + rng.uniform(30, 120),
              fill=int(rng.uniform(30, 90)), width=int(rng.uniform(1, 3)))
    scuff = np.asarray(img.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.
    # dust near the edges
    edge = np.clip(1 - np.minimum(y, h - 1 - y) / 40., 0, 1) ** 2
    dust = .05 * edge * noise((h, w), 30, rng)
    # skived splice: diagonal joint with a row of staples-free vulcanised seam
    seam = np.zeros((h, w), np.float32)
    sx = 700 + (y - h / 2) * .35
    dist = np.abs(x - sx)
    seam += np.clip(1 - dist / 2.5, 0, 1) * -.045      # dark groove
    seam += np.clip(1 - np.abs(dist - 7) / 3., 0, 1) * .02   # vulcanised lip
    # edge bands (belt side lips, slightly smoother & darker)
    band = ((y < 14) | (y > h - 15)).astype(np.float32)
    v = base + weave + wear + .10 * scuff + dust + seam - .02 * band
    rgb = np.stack([v * 1.0, v * 1.02, v * 1.05], -1)
    albedo = np.clip(rgb ** (1 / 2.2), 0, 1)
    rough = np.clip(.5 + .25 * (noise((h, w), 20, rng) - .5) - .25 * wear / .03 * .3 + .2 * scuff
                    - .15 * band, 0, 1)
    hgt = (weave * 40 + seam * 8 + .3 * noise((h, w), 6, rng))
    gy, gx = np.gradient(hgt)
    nrm = np.stack([-gx * 2, -gy * 2, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    return (Image.fromarray((albedo * 255).astype(np.uint8)),
            Image.fromarray((rough * 255).astype(np.uint8)),
            Image.fromarray(((nrm * .5 + .5) * 255).astype(np.uint8)))


def label(text, sub, bg, fg, rng):
    w, h = 1024, 512
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([14, 14, w - 15, h - 15], outline=fg, width=10)
    size = 300
    while size > 60:
        f = font(size)
        box = d.textbbox((0, 0), text, font=f)
        if box[2] - box[0] < w - 120:
            break
        size -= 10
    box = d.textbbox((0, 0), text, font=f)
    d.text(((w - (box[2] - box[0])) / 2 - box[0], 150 - (box[3] - box[1]) / 2 - box[1]), text, font=f, fill=fg)
    f2 = font(70)
    box = d.textbbox((0, 0), sub, font=f2)
    d.text(((w - (box[2] - box[0])) / 2 - box[0], 390 - (box[3] - box[1]) / 2 - box[1]), sub, font=f2, fill=fg)
    arr = np.asarray(img, np.float32) / 255.
    grime = 1 - .06 * noise((h, w), 40, rng)[..., None]
    return Image.fromarray((np.clip(arr * grime, 0, 1) * 255).astype(np.uint8))


def sign(rng):
    w, h = 1600, 400
    img = Image.new("RGB", (w, h), (24, 92, 52))
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, w - 13, h - 13], outline=(235, 238, 232), width=8)
    f, f2 = font(150), font(62)
    d.text((70, 50), "WASTE SORTING", font=f, fill=(240, 242, 236))
    d.text((74, 250), "PLASTIC = BLUE BIN     METAL = YELLOW BIN", font=f2, fill=(250, 210, 40))
    return img


def main():
    rng = np.random.default_rng(20261017)
    OUT.mkdir(exist_ok=True)
    a, r, n = belt(rng)
    files = {"sortbelt_albedo.jpg": a, "sortbelt_rough.png": r, "sortbelt_normal.png": n,
             "binlabel_plastic.jpg": label("PLASTIC", "BOTTLES - CUPS - PET", (22, 86, 170), (245, 246, 240), rng),
             "binlabel_metal.jpg": label("METAL", "CANS - TINS - ALU", (232, 176, 8), (22, 22, 24), rng),
             "sign_sorting_albedo.jpg": sign(rng)}
    manifest = {}
    for name, im in files.items():
        path = OUT / name
        im.save(path, quality=92) if name.endswith(".jpg") else im.save(path)
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "recycling_v2_textures.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
