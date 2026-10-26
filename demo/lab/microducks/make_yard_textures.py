"""Procedural textures for the Microducks schoolyard (numpy + Pillow, ASCII only)."""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "textures"
FONT_DIR = sys.argv[2] if len(sys.argv) > 2 else "fonts"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def font(size, bold=True):
    for name in (["NVIDIASans_Bd.ttf"] if bold else ["NVIDIASans_Rg.ttf"]) + ["DejaVuSans-Bold.ttf"]:
        for d in (FONT_DIR, "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def noise(h, w, scales=((8, 0.5), (32, 0.3), (128, 0.2))):
    acc = np.zeros((h, w), np.float32)
    for s, a in scales:
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += a * (np.asarray(img, np.float32) / 255.0)
    return acc


# ---------------- yard floor: 5 x 5 m, 2048 px (2.44 mm / px) ----------------
N = 2048
YARD = 5.0
px = N / YARD


def P(x, y):
    """World (x, y) in metres (origin at yard centre, +y north) to pixel."""
    return (x + YARD / 2) * px, (YARD / 2 - y) * px


base = 0.30 + 0.10 * noise(N, N) + 0.06 * rng.random((N, N)).astype(np.float32)
asphalt = np.stack([base * 0.95, base * 0.97, base * 1.02], -1)
speck = rng.random((N, N)) > 0.985
asphalt[speck] *= 1.6
img = Image.fromarray(np.clip(asphalt * 255, 0, 255).astype(np.uint8))
d = ImageDraw.Draw(img, "RGBA")

# painted court: outer line and centre circle (yellow), plus a big four-square (white)
yellow = (240, 200, 60, 230)
white = (240, 240, 235, 235)
lw = int(0.025 * px)
x0, y0 = P(-2.2, 2.2)
x1, y1 = P(2.2, -2.2)
d.rectangle([x0, y0, x1, y1], outline=yellow, width=lw)
cx, cy = P(0.6, -0.2)
r = 0.55 * px
d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=yellow, width=lw)
d.line([P(0.6, 2.2), P(0.6, -2.2)], fill=yellow, width=lw)

# four-square, south-west
fx0, fy0 = P(-2.0, -0.6)
fx1, fy1 = P(-0.8, -1.8)
d.rectangle([fx0, fy0, fx1, fy1], outline=white, width=lw)
mx, my = (fx0 + fx1) / 2, (fy0 + fy1) / 2
d.line([(mx, fy0), (mx, fy1)], fill=white, width=lw)
d.line([(fx0, my), (fx1, my)], fill=white, width=lw)
f = font(int(0.28 * px))
for i, (qx, qy) in enumerate([(fx0, fy0), (mx, fy0), (fx0, my), (mx, my)]):
    d.text((qx + 0.18 * px, qy + 0.12 * px), "ABCD"[i], font=f, fill=(240, 240, 235, 140))

# hopscotch in coloured chalk, east side running north-south
chalk = [(255, 120, 150, 210), (120, 200, 255, 210), (255, 230, 120, 210), (150, 240, 150, 210)]
sq = 0.28
layout = [[0.0], [0.0], [-0.5, 0.5], [0.0], [-0.5, 0.5], [0.0], [-0.5, 0.5], [0.0]]
hx, hy = 1.85, -1.95
n = 1
fn = font(int(0.17 * px))
for row, cols in enumerate(layout):
    for c in cols:
        x = hx + c * sq
        y = hy + row * sq
        a, b = P(x - sq / 2, y + sq)
        e, g = P(x + sq / 2, y)
        col = chalk[(n - 1) % len(chalk)]
        d.rectangle([a, b, e, g], outline=col, width=int(0.012 * px))
        tw = d.textlength(str(n), font=fn)
        d.text(((a + e) / 2 - tw / 2, (b + g) / 2 - 0.1 * px), str(n), font=fn, fill=col)
        n += 1
# half circle "home"
a, b = P(hx - sq / 2, hy + 8 * sq + sq * 0.9)
e, g = P(hx + sq / 2, hy + 8 * sq - sq * 0.1)
d.pieslice([a, b, e, g], 180, 360, outline=chalk[1], width=int(0.012 * px))

# chalk drawings: a sun and a duck-ish smiley, north-west
sx, sy = P(-1.7, 1.6)
rr = 0.22 * px
d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=(255, 220, 80, 220), width=int(0.014 * px))
for k in range(12):
    ang = k * math.pi / 6
    d.line([(sx + 1.3 * rr * math.cos(ang), sy + 1.3 * rr * math.sin(ang)),
            (sx + 1.8 * rr * math.cos(ang), sy + 1.8 * rr * math.sin(ang))],
           fill=(255, 220, 80, 220), width=int(0.012 * px))
fx, fy = P(-0.9, 1.75)
for k in range(6):
    ang = k * math.pi / 3
    pr = 0.07 * px
    ox, oy = fx + 0.09 * px * math.cos(ang), fy + 0.09 * px * math.sin(ang)
    d.ellipse([ox - pr, oy - pr, ox + pr, oy + pr], outline=(255, 140, 200, 220), width=int(0.01 * px))
d.ellipse([fx - 0.05 * px, fy - 0.05 * px, fx + 0.05 * px, fy + 0.05 * px], fill=(255, 220, 80, 200))
d.text(P(-2.2, 1.2), "MICRODUCK SCHOOL", font=font(int(0.13 * px)), fill=(255, 255, 255, 120))
# ramp arrows painted on the ground
for (ax, ay) in [(-1.75, 0.75), (1.0, 1.55)]:
    a1 = P(ax, ay)
    d.polygon([P(ax - 0.12, ay + 0.06), P(ax + 0.05, ay + 0.06), P(ax + 0.05, ay + 0.13),
               P(ax + 0.18, ay), P(ax + 0.05, ay - 0.13), P(ax + 0.05, ay - 0.06),
               P(ax - 0.12, ay - 0.06)], fill=(255, 255, 255, 150))
img = img.filter(ImageFilter.GaussianBlur(0.6))
img.save(os.path.join(OUT, "yard_floor.png"))

# ---------------- grass outside ----------------
G = 1024
g = noise(G, G, ((4, 0.4), (16, 0.35), (64, 0.25)))
grass = np.stack([0.22 + 0.12 * g, 0.42 + 0.18 * g, 0.16 + 0.08 * g], -1)
grass *= (0.85 + 0.3 * rng.random((G, G, 1)))
Image.fromarray(np.clip(grass * 255, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "grass.png"))

# ---------------- low brick wall ----------------
W, H = 1024, 256
br = np.zeros((H, W, 3), np.float32)
bh, bw = 32, 96
for row in range(H // bh):
    off = (bw // 2) * (row % 2)
    for col in range(-1, W // bw + 1):
        c = np.array([0.62, 0.28, 0.20]) * (0.8 + 0.4 * rng.random())
        xa, xb = col * bw + off + 3, (col + 1) * bw + off - 3
        ya, yb = row * bh + 3, (row + 1) * bh - 3
        br[max(0, ya):yb, max(0, xa):max(0, min(W, xb))] = c
mortar = br.sum(-1) == 0
br[mortar] = (0.78, 0.76, 0.72)
br *= (0.9 + 0.2 * noise(H, W, ((4, 0.5), (16, 0.5))))[..., None]
Image.fromarray(np.clip(br * 255, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "brick.png"))

# ---------------- school facade ----------------
FW, FH = 2048, 512
fac = Image.new("RGB", (FW, FH), (236, 214, 170))
fd = ImageDraw.Draw(fac)
for k in range(FH // 16):
    fd.line([(0, k * 16), (FW, k * 16)], fill=(222, 198, 156), width=2)
fd.rectangle([0, FH - 70, FW, FH], fill=(150, 92, 70))
for i in range(9):
    wx = 60 + i * 225
    if i == 4:
        fd.rectangle([wx - 10, 200, wx + 170, FH - 70], fill=(70, 120, 170))
        fd.rectangle([wx + 75, 210, wx + 85, FH - 70], fill=(40, 60, 90))
        continue
    for wy in (70, 260):
        fd.rectangle([wx, wy, wx + 150, wy + 130], fill=(250, 250, 245))
        fd.rectangle([wx + 8, wy + 8, wx + 142, wy + 122], fill=(120, 170, 210))
        fd.line([(wx + 75, wy + 8), (wx + 75, wy + 122)], fill=(250, 250, 245), width=6)
        fd.line([(wx + 8, wy + 65), (wx + 142, wy + 65)], fill=(250, 250, 245), width=6)
fd.rectangle([700, 8, 1350, 62], fill=(30, 90, 60))
sf = font(44)
t = "MICRODUCK PRIMARY SCHOOL"
fd.text((1025 - fd.textlength(t, font=sf) / 2, 10), t, font=sf, fill=(255, 245, 200))
fac.save(os.path.join(OUT, "facade.png"))

# ---------------- mural on the west wall (above bricks, on a panel) ----------------
MW, MH = 1024, 256
mur = Image.new("RGB", (MW, MH), (60, 140, 200))
md = ImageDraw.Draw(mur)
md.ellipse([40, 30, 160, 150], fill=(255, 220, 70))
for k in range(5):
    x = 220 + k * 160
    md.ellipse([x, 120, x + 120, 220], fill=(255, 210, 40))
    md.ellipse([x + 70, 80, x + 130, 140], fill=(255, 210, 40))
    md.polygon([(x + 128, 105), (x + 158, 112), (x + 128, 120)], fill=(255, 130, 30))
    md.ellipse([x + 100, 95, x + 110, 105], fill=(20, 20, 20))
md.rectangle([0, 220, MW, MH], fill=(80, 170, 90))
mur.save(os.path.join(OUT, "mural.png"))

# ---------------- wood for benches/ramps, rubber for ramps ----------------
wood = noise(256, 1024, ((2, 0.3), (8, 0.4), (64, 0.3)))
wood_rgb = np.stack([0.55 + 0.2 * wood, 0.36 + 0.14 * wood, 0.20 + 0.08 * wood], -1)
Image.fromarray(np.clip(wood_rgb * 255, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "wood.png"))
rub = 0.85 + 0.15 * noise(512, 512, ((4, 0.5), (32, 0.5)))
rub_rgb = np.stack([0.20 * rub + 0.55, 0.32 * rub + 0.08, 0.30 * rub + 0.06], -1)
Image.fromarray(np.clip(rub_rgb * 255, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "ramp_rubber.png"))
print("textures written to", OUT)
