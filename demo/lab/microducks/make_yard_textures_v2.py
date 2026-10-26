"""v2 schoolyard textures: Poly Haven asphalt_02 (CC0) tiled over the 5 x 5 m yard with painted markings baked in,
plus the school sign. numpy + Pillow only.

    python make_yard_textures_v2.py <assets_ph dir> <out dir> <fonts dir>
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PH, OUT, FONTS = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
N = 4096
YARD = 5.0
px = N / YARD


def font(size, bold=True):
    for n in (["NVIDIASans_Bd.ttf"] if bold else ["NVIDIASans_Rg.ttf"]) + ["DejaVuSans-Bold.ttf"]:
        for d in (FONTS, "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def P(x, y):
    return (x + YARD / 2) * px, (YARD / 2 - y) * px


def tiled(path, tiles=2, mode="RGB"):
    im = Image.open(path).convert(mode).resize((N // tiles, N // tiles), Image.LANCZOS)
    out = Image.new(mode, (N, N))
    for i in range(tiles):
        for j in range(tiles):
            out.paste(im, (i * N // tiles, j * N // tiles))
    return out


def noise(h, w, scales):
    acc = np.zeros((h, w), np.float32)
    for s, a in scales:
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        acc += a * np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                              np.float32) / 255.0
    return acc


A = os.path.join(PH, "asphalt_02")
diff = tiled(os.path.join(A, "asphalt_02_diff_4k.jpg"))
rough = tiled(os.path.join(A, "asphalt_02_rough_4k.jpg"), mode="L")
nor = tiled(os.path.join(A, "asphalt_02_nor_gl_4k.jpg"))

paint = Image.new("RGBA", (N, N), (0, 0, 0, 0))
d = ImageDraw.Draw(paint)
yellow, white = (236, 196, 52, 255), (236, 236, 228, 255)
lw = int(0.022 * px)
d.rectangle([*P(-2.2, 2.2), *P(2.2, -2.2)], outline=yellow, width=lw)
cx, cy = P(0.6, -0.2)
r = 0.55 * px
d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=yellow, width=lw)
d.line([P(0.6, 2.2), P(0.6, -2.2)], fill=yellow, width=lw)
fx0, fy0 = P(-2.0, -0.6)
fx1, fy1 = P(-0.8, -1.8)
d.rectangle([fx0, fy0, fx1, fy1], outline=white, width=lw)
mx, my = (fx0 + fx1) / 2, (fy0 + fy1) / 2
d.line([(mx, fy0), (mx, fy1)], fill=white, width=lw)
d.line([(fx0, my), (fx1, my)], fill=white, width=lw)
f = font(int(0.26 * px))
for i, (qx, qy) in enumerate([(fx0, fy0), (mx, fy0), (fx0, my), (mx, my)]):
    d.text((qx + 0.19 * px, qy + 0.13 * px), "ABCD"[i], font=f, fill=(236, 236, 228, 170))
chalk = [(250, 130, 160, 215), (130, 200, 250, 215), (250, 228, 130, 215), (160, 236, 160, 215)]
sq = 0.28
layout = [[0.0], [0.0], [-0.5, 0.5], [0.0], [-0.5, 0.5], [0.0], [-0.5, 0.5], [0.0]]
hx, hy, n = 1.85, -1.95, 1
fn = font(int(0.16 * px))
for row, cols in enumerate(layout):
    for c in cols:
        x, y = hx + c * sq, hy + row * sq
        a, b = P(x - sq / 2, y + sq)
        e, g = P(x + sq / 2, y)
        col = chalk[(n - 1) % 4]
        d.rectangle([a, b, e, g], outline=col, width=int(0.011 * px))
        tw = d.textlength(str(n), font=fn)
        d.text(((a + e) / 2 - tw / 2, (b + g) / 2 - 0.1 * px), str(n), font=fn, fill=col)
        n += 1
sx, sy = P(-1.7, 1.6)
rr = 0.22 * px
d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=(250, 216, 90, 215), width=int(0.013 * px))
for k in range(12):
    ang = k * math.pi / 6
    d.line([(sx + 1.3 * rr * math.cos(ang), sy + 1.3 * rr * math.sin(ang)),
            (sx + 1.8 * rr * math.cos(ang), sy + 1.8 * rr * math.sin(ang))], fill=(250, 216, 90, 215),
           width=int(0.011 * px))
fxx, fyy = P(-0.9, 1.75)
for k in range(6):
    ang = k * math.pi / 3
    pr = 0.07 * px
    ox, oy = fxx + 0.09 * px * math.cos(ang), fyy + 0.09 * px * math.sin(ang)
    d.ellipse([ox - pr, oy - pr, ox + pr, oy + pr], outline=(250, 140, 200, 215), width=int(0.01 * px))
for (ax, ay) in [(-2.15, 0.75), (0.15, 1.45)]:
    d.polygon([P(ax - 0.1, ay + 0.05), P(ax + 0.04, ay + 0.05), P(ax + 0.04, ay + 0.11), P(ax + 0.15, ay),
               P(ax + 0.04, ay - 0.11), P(ax + 0.04, ay - 0.05), P(ax - 0.1, ay - 0.05)], fill=(236, 236, 228, 160))
# wear: paint thins out where the asphalt is coarse, plus large-scale scuffs
pa = np.asarray(paint, np.float32)
wear = 0.55 + 0.45 * noise(N, N, ((8, 0.4), (64, 0.35), (256, 0.25)))
grain = np.asarray(rough, np.float32) / 255.0
alpha = pa[..., 3] / 255.0 * np.clip(wear * (1.15 - 0.5 * grain), 0, 1)
alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                   np.float32) / 255.0
base = np.asarray(diff, np.float32)
base = base * 0.92  # slightly darker, sun-bleached look comes from the light
out = base * (1 - alpha[..., None]) + pa[..., :3] * alpha[..., None] * (0.75 + 0.25 * grain[..., None])
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "yard_v2_diff.jpg"), quality=92)
rr_ = np.asarray(rough, np.float32) * (1 - 0.35 * alpha)
Image.fromarray(np.clip(rr_, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "yard_v2_rough.jpg"), quality=92)
nor.save(os.path.join(OUT, "yard_v2_nor.jpg"), quality=92)

# school sign board
W, H = 2400, 300
sign = Image.new("RGB", (W, H), (22, 74, 52))
sd = ImageDraw.Draw(sign)
sd.rectangle([10, 10, W - 10, H - 10], outline=(230, 214, 160), width=8)
t = "MICRODUCK PRIMARY SCHOOL"
size = 170
ft = font(size)
while sd.textlength(t, font=ft) > W * 0.88 and size > 40:
    size -= 4
    ft = font(size)
tw = sd.textlength(t, font=ft)
sd.text(((W - tw) / 2, (H - size * 1.2) / 2), t, font=ft, fill=(250, 240, 205))
sign.save(os.path.join(OUT, "sign_v2.png"))
print("v2 textures ->", OUT)
