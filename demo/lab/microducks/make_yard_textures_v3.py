"""v3 'photographic' textures: darker worn asphalt with dirt and oil stains and softer, worn paint; grimy desaturated
plaster; muted grass. Inputs: v2 outputs + Poly Haven maps. numpy + Pillow only.

    python make_yard_textures_v3.py <assets_ph dir> <textures dir>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

PH, TEX = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(23)


def lin(a):
    a = a / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255


def noise(h, w, scales):
    acc = np.zeros((h, w), np.float32)
    for s, a in scales:
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        acc += a * np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                              np.float32) / 255.0
    return acc


def desat(L, k):
    y = (L @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    return y + (L - y) * k


# --- yard: start from v2 (asphalt + paint), darken slightly, add dirt / oil stains, soften paint
v2 = np.asarray(Image.open(os.path.join(TEX, "yard_v2_diff.jpg")).convert("RGB"), np.float32)
N = v2.shape[0]
L = lin(v2)
paint = (L.max(-1) > 0.25)[..., None]  # bright pixels = markings
L = np.where(paint, L * 0.72, L * 0.86)  # paint ~0.45-0.6, asphalt ~0.08
dirt = noise(N, N, ((16, 0.3), (96, 0.4), (512, 0.3)))
L *= (0.78 + 0.32 * dirt)[..., None]
for _ in range(26):  # oil / water stains
    cx, cy = rng.integers(0, N, 2)
    r = rng.integers(N // 80, N // 25)
    yy, xx = np.ogrid[:N, :N]
    m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r * r)))
    L *= (1 - 0.35 * m * rng.uniform(0.4, 1.0))[..., None]
L = desat(L, 0.85)
Image.fromarray(srgb(L).astype(np.uint8)).save(os.path.join(TEX, "yard_v3_diff.jpg"), quality=92)

# --- plaster: neutral, slightly darker, grime toward the bottom + vertical streaks (texture tiles in metres)
P = np.asarray(Image.open(os.path.join(PH, "painted_plaster_wall", "painted_plaster_wall_diff_2k.jpg")).convert("RGB"),
               np.float32)
Lp = desat(lin(P), 0.45) * np.array([0.92, 0.88, 0.80])
h, w = Lp.shape[:2]
streak = noise(h, w, ((1, 0.0), (64, 1.0)))
streak = np.asarray(Image.fromarray((streak * 255).astype(np.uint8)).resize((w // 8, h)).resize((w, h), Image.BICUBIC),
                    np.float32) / 255.0
Lp *= (0.82 + 0.25 * streak)[..., None]
Image.fromarray(srgb(Lp).astype(np.uint8)).save(os.path.join(TEX, "plaster_v3_diff.jpg"), quality=92)

# --- grass: muted, darker
G = np.asarray(Image.open(os.path.join(TEX, "grass.png")).convert("RGB"), np.float32)
Lg = desat(lin(G), 0.55) * np.array([0.95, 0.62, 0.85])
Image.fromarray(srgb(Lg).astype(np.uint8)).save(os.path.join(TEX, "grass_v3.png"))


def report(name, arr):
    print(f"{name:12s} linear albedo mean {np.round(arr.reshape(-1, 3).mean(0), 3)}")


report("yard_v3", L[::8, ::8])
report("plaster_v3", Lp[::4, ::4])
report("grass_v3", Lg[::4, ::4])
