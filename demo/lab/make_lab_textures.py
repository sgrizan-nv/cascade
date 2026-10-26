#!/usr/bin/env python3
"""Generate the procedural PBR textures used by demo/lab_scenes.py.

Maintainer tool (numpy + Pillow); the scenes only read the files it writes into
demo/lab/textures/. Everything here is original procedural work (CC0-1.0).
Conventions shared with lab_scenes.py:
  *_albedo   sRGB. "Neutral" albedos have a linear mean of NEUTRAL (0.8) and are
             tinted per material through UsdUVTexture.scale.
  *_rough    raw, single channel, a variation map centred on 0.5; each material
             remaps it with scale/bias around its own roughness.
  *_normal   raw, tangent space, OpenGL convention (+Y = +v).
Run:  python3 demo/lab/make_lab_textures.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "textures"
NEUTRAL = 0.8
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_NARROW = "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf"


# ------------------------------------------------------------------ helpers
def noise(h, w, beta=2.0, seed=0, ax=1.0, ay=1.0, kmin=0.0):
    """Periodic 1/f^beta noise, zero mean, unit std. ax/ay stretch frequencies."""
    rng = np.random.default_rng(seed)
    spec = np.fft.fft2(rng.standard_normal((h, w)))
    ky = np.fft.fftfreq(h)[:, None] * h * ay
    kx = np.fft.fftfreq(w)[None, :] * w * ax
    k = np.sqrt(kx * kx + ky * ky)
    amp = 1.0 / np.maximum(k, 1.0) ** beta
    amp[k < kmin] = 0
    amp[0, 0] = 0
    field = np.real(np.fft.ifft2(spec * amp))
    return (field - field.mean()) / (field.std() + 1e-9)


def blur(a, radius):
    """Periodic gaussian blur via FFT (radius = sigma in px)."""
    h, w = a.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    g = np.exp(-2 * (math.pi * radius) ** 2 * (kx * kx + ky * ky))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= .0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - .055)


def to_lin(srgb):
    s = np.asarray(srgb, float)
    return np.where(s <= .04045, s / 12.92, ((s + .055) / 1.055) ** 2.4)


def normal_from_height(hgt, strength):
    """Tangent-space normal from a periodic height field (strength = slope per unit height per px)."""
    dx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * .5
    dy = (np.roll(hgt, 1, 0) - np.roll(hgt, -1, 0)) * .5  # image rows grow downwards, v grows up
    n = np.stack([-dx * strength, -dy * strength, np.ones_like(hgt)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def save_rgb(name, lin, *, neutral=False, quality=93):
    lin = np.asarray(lin, float)
    if neutral:
        lin = lin * (NEUTRAL / max(lin.mean(), 1e-6))
    img = Image.fromarray((to_srgb(lin) * 255 + .5).astype(np.uint8), "RGB")
    path = OUT / name
    if name.endswith(".jpg"):
        img.save(path, quality=quality, subsampling=0)
    else:
        img.save(path, optimize=True)
    return path


def save_rgba(name, lin, alpha):
    rgb = (to_srgb(lin) * 255 + .5).astype(np.uint8)
    a = (np.clip(alpha, 0, 1) * 255 + .5).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, a]), "RGBA").save(OUT / name, optimize=True)


def save_rough(name, var):
    """var: variation field; stored as 0.5 + var, clipped."""
    Image.fromarray((np.clip(.5 + var, 0, 1) * 255 + .5).astype(np.uint8), "L").save(OUT / name, optimize=True)


def save_gray(name, v):
    Image.fromarray((np.clip(v, 0, 1) * 255 + .5).astype(np.uint8), "L").save(OUT / name, optimize=True)


def save_normal(name, n):
    Image.fromarray(((n * .5 + .5) * 255 + .5).astype(np.uint8), "RGB").save(OUT / name, optimize=True)


def stroke_mask(h, w, strokes, width, seed=0, soft=1.0):
    """Anti-aliased polyline mask drawn with periodic wrap (strokes in px)."""
    img = Image.new("L", (w * 3, h * 3), 0)
    d = ImageDraw.Draw(img)
    for pts, val in strokes:
        for ox in (0, w, 2 * w):
            for oy in (0, h, 2 * h):
                d.line([(x + ox, y + oy) for x, y in pts], fill=int(255 * val), width=width)
    a = np.asarray(img.crop((w, h, 2 * w, 2 * h)), float) / 255
    return blur(a, soft) if soft else a


def random_walk(rng, x, y, n, step, turn):
    pts, ang = [(x, y)], rng.uniform(0, 2 * math.pi)
    for _ in range(n):
        ang += rng.normal(0, turn)
        x, y = x + step * math.cos(ang), y + step * math.sin(ang)
        pts.append((x, y))
    return pts


def blobs(h, w, rng, count, rmin, rmax):
    """Soft irregular blobs (stains), periodic."""
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w))
    for _ in range(count):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(rmin, rmax)
        dx = (xx - cx + w / 2) % w - w / 2
        dy = (yy - cy + h / 2) % h - h / 2
        out = np.maximum(out, np.exp(-(dx * dx + dy * dy) / (2 * r * r)) * rng.uniform(.5, 1))
    return out


# ------------------------------------------------------------------ surfaces
def concrete_floor(size=2048, tile_m=4.0):
    """Sealed (epoxy) concrete slab: mottling, aggregate, stains, scuffs, saw-cut joints."""
    rng = np.random.default_rng(11)
    big, mid = noise(size, size, 2.6, 1), noise(size, size, 1.7, 2)
    fine = noise(size, size, .6, 3)
    agg = noise(size, size, .2, 4)
    speck_dark = (agg > 2.6).astype(float)
    speck_light = (agg < -2.7).astype(float)
    base = np.array([.205, .200, .190])
    shade = 1 + .11 * big + .05 * mid + .025 * fine - .22 * speck_dark + .16 * speck_light
    stains = blobs(size, size, rng, 9, 25, 120) * (noise(size, size, 1.5, 5) > -.3)
    stains = blur(stains, 6)
    shade *= 1 - .32 * stains
    # Pallet-jack and trolley scuffs: long faint dark arcs.
    strokes = []
    for _ in range(26):
        pts = random_walk(rng, rng.uniform(0, size), rng.uniform(0, size), 60, 14, .03)
        strokes.append((pts, rng.uniform(.25, .8)))
    scuff = stroke_mask(size, size, strokes, 9, soft=3)
    shade *= 1 - .07 * scuff
    # Hairline shrinkage cracks.
    cracks = []
    for _ in range(5):
        cracks.append((random_walk(rng, rng.uniform(0, size), rng.uniform(0, size), 70, 5, .28), .8))
    crack = stroke_mask(size, size, cracks, 1, soft=.6)
    shade *= 1 - .22 * crack
    # Saw-cut control joints every 2 m (two per tile in each direction).
    yy, xx = np.mgrid[0:size, 0:size]
    px_per_m = size / tile_m
    jx = np.minimum((xx % (2 * px_per_m)), 2 * px_per_m - (xx % (2 * px_per_m)))
    jy = np.minimum((yy % (2 * px_per_m)), 2 * px_per_m - (yy % (2 * px_per_m)))
    joint = np.clip(1.6 - np.minimum(jx, jy), 0, 1)
    joint = blur(joint, .7)
    shade *= 1 - .55 * joint
    alb = base[None, None, :] * shade[..., None]
    # Very slight warm/cool drift.
    alb[..., 2] *= 1 + .03 * noise(size, size, 2.8, 6)
    save_rgb("concrete_albedo.jpg", alb)
    rough = .10 * big + .05 * fine + .05 * scuff - .30 * stains + .35 * joint + .10 * speck_dark
    save_rough("concrete_rough.png", blur(rough, .6)[::2, ::2])
    hgt = .45 * blur(agg.clip(1.4, None) - 1.4, 1.2) + .15 * fine - 3.0 * joint - .35 * crack
    save_normal("concrete_normal.png", normal_from_height(hgt[::2, ::2], 1.4))


def floor_paint(w=1024, h=128):
    """Worn line paint (neutral, alpha cut-out). u runs along the line, v across it."""
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[0:h, 0:w]
    edge = noise(h, w, 1.4, 22, ay=0.0001)  # varies along u only
    ragged = 7 + 3 * edge + 1.5 * noise(h, w, .8, 23)
    across = np.minimum(yy, h - 1 - yy)
    alpha = np.clip((across - ragged + 1) / 2, 0, 1)
    wear = noise(h, w, 1.6, 24) + .6 * noise(h, w, .7, 25)
    chips = (wear > 1.65).astype(float)
    alpha *= 1 - chips
    dirt = 1 - .16 * np.clip(noise(h, w, 1.8, 26), 0, None) - .08 * np.clip(noise(h, w, .5, 27), 0, None)
    tread = stroke_mask(h, w, [(random_walk(rng, rng.uniform(0, w), rng.uniform(0, h), 80, 9, .02), .7)
                               for _ in range(10)], 5, soft=2)
    dirt *= 1 - .25 * tread
    alb = np.ones((h, w, 3)) * dirt[..., None]
    alb = alb * (NEUTRAL / alb.mean())
    save_rgba("floorpaint_albedo.png", alb, alpha)
    save_rough("floorpaint_rough.png", .25 * tread + .12 * noise(h, w, 1.0, 28) + .2 * (1 - alpha))


def block_wall(w=2048, h=1024, tile=(1.6, .8)):
    """Painted CMU block wall in running bond (blocks 400 x 200 mm incl. 10 mm joints)."""
    rng = np.random.default_rng(31)
    ppm = w / tile[0]
    yy, xx = np.mgrid[0:h, 0:w]
    row = (yy // (0.2 * ppm)).astype(int)
    xo = xx + (row % 2) * 0.2 * ppm
    bx = xo % (0.4 * ppm)
    by = yy % (0.2 * ppm)
    jw = 0.010 * ppm
    dist = np.minimum.reduce([bx, 0.4 * ppm - bx, by, 0.2 * ppm - by])
    joint = np.clip((jw / 2 + 1 - dist) / 2, 0, 1)
    block_id = row * 7 + (xo // (0.4 * ppm)).astype(int)
    tint = 1 + .035 * np.sin(block_id * 12.9898) * np.cos(block_id * 4.1414)
    pores = noise(h, w, .3, 32)
    pore = np.clip(pores - 1.9, 0, None)
    mottle = noise(h, w, 2.2, 33)
    shade = tint * (1 + .03 * mottle) - .08 * pore - .035 * joint
    # Faint vertical grime streaks, coherent across the tile.
    streak = noise(h, w, 1.5, 34, ax=1, ay=12)
    shade *= 1 - .03 * np.clip(streak, 0, None)
    alb = np.ones((h, w, 3)) * shade[..., None]
    save_rgb("blockwall_albedo.jpg", alb, neutral=True)
    save_rough("blockwall_rough.png", (.10 * joint + .04 * mottle + .1 * pore)[::2, ::2])
    hgt = -1.4 * blur(joint, 1.8) - .6 * pore + .12 * noise(h, w, .8, 35)
    save_normal("blockwall_normal.png", normal_from_height(hgt[::2, ::2], 1.6))


def brushed_steel(size=2048):
    """Brushed stainless sheet; brushing along u. Fine and low contrast so the bench stays readable."""
    rng = np.random.default_rng(41)
    streak = noise(size, size, .5, 42, ax=80, ay=1, kmin=60)
    streak2 = noise(size, size, .3, 43, ax=300, ay=1, kmin=200)
    smudge = blur(blobs(size, size, rng, 14, 25, 80) * (noise(size, size, 1.2, 44) > .3), 6)
    strokes = []
    for _ in range(70):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        ang, ln = rng.uniform(0, math.pi), rng.uniform(20, 140)
        strokes.append(([(x, y), (x + ln * math.cos(ang), y + ln * math.sin(ang))], rng.uniform(.2, .7)))
    scratch = stroke_mask(size, size, strokes, 1, soft=.5)
    big = noise(size, size, 2.8, 45)
    shade = 1 + .007 * streak + .006 * streak2 + .012 * big + .025 * scratch - .02 * smudge
    alb = np.array([.56, .565, .57])[None, None, :] * shade[..., None]
    save_rgb("steel_albedo.jpg", alb[::2, ::2])
    save_rough("steel_rough.png", .06 * streak + .05 * streak2 - .14 * scratch + .12 * smudge + .04 * big)
    hgt = .35 * streak2 + .2 * streak + .8 * scratch
    save_normal("steel_normal.png", normal_from_height(hgt[::2, ::2], .3))


def painted_metal(size=1024):
    """Powder-coated / enamel steel, neutral, with light scratches, dents and dirt."""
    rng = np.random.default_rng(51)
    peel = noise(size, size, .9, 52)
    mottle = noise(size, size, 2.3, 53)
    strokes = []
    for _ in range(60):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        ang, ln = rng.uniform(0, math.pi), rng.uniform(10, 70)
        strokes.append(([(x, y), (x + ln * math.cos(ang), y + ln * math.sin(ang))], rng.uniform(.4, 1)))
    scratch = stroke_mask(size, size, strokes, 1, soft=.5)
    chips = blur((noise(size, size, 1.1, 54) > 2.5).astype(float), .8)
    dirt = np.clip(noise(size, size, 2.0, 55), 0, None)
    shade = 1 + .025 * mottle - .08 * dirt - .35 * chips + .10 * scratch
    alb = np.ones((size, size, 3)) * shade[..., None]
    save_rgb("paint_albedo.jpg", alb, neutral=True)
    save_rough("paint_rough.png", .06 * mottle + .10 * dirt - .18 * scratch + .2 * chips + .03 * peel)
    hgt = .25 * peel + 1.2 * scratch - .8 * chips + .3 * noise(size, size, 2.4, 56)
    save_normal("paint_normal.png", normal_from_height(hgt, .5))


def pegboard(size=1024, holes=20):
    """Tempered hardboard pegboard, 25.4 mm pitch, 6.4 mm holes; tile = 20 holes (0.508 m)."""
    pitch = size / holes
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx % pitch) - pitch / 2 + .5
    dy = (yy % pitch) - pitch / 2 + .5
    r = np.sqrt(dx * dx + dy * dy)
    rad = pitch * 6.4 / 25.4 / 2
    hole = np.clip(rad - r + .5, 0, 1)
    rim = np.clip(1 - np.abs(r - rad - 1.2) / 1.5, 0, 1)
    fiber = noise(size, size, .9, 61, ax=1, ay=3)
    mottle = noise(size, size, 2.2, 62)
    base = np.array([.22, .14, .075])
    shade = (1 + .05 * fiber + .07 * mottle) * (1 - .25 * rim)
    alb = base[None, None, :] * shade[..., None]
    alb = alb * (1 - hole[..., None]) + np.array([.008, .007, .006]) * hole[..., None]
    save_rgb("pegboard_albedo.jpg", alb)
    save_rough("pegboard_rough.png", .05 * fiber + .06 * mottle + .4 * hole)
    hgt = -3 * blur(hole, 1.0) + .15 * fiber
    save_normal("pegboard_normal.png", normal_from_height(hgt, 1.2))


def cardboard(size=1024):
    """Kraft corrugated board, flutes along u (8 mm pitch at a 0.6 m tile)."""
    rng = np.random.default_rng(71)
    yy, xx = np.mgrid[0:size, 0:size]
    pitch = size / (600 / 8)
    flute = np.sin(2 * math.pi * yy / pitch)
    fiber = noise(size, size, .8, 72)
    mottle = noise(size, size, 2.4, 73)
    stain = blur(blobs(size, size, rng, 5, 30, 90) * (noise(size, size, 1.3, 74) > .2), 4)
    ring = np.clip(blur(stain, 2) - stain * .9, 0, None)
    scuff = stroke_mask(size, size, [(random_walk(rng, rng.uniform(0, size), rng.uniform(0, size), 25, 8, .3),
                                      rng.uniform(.3, .8)) for _ in range(18)], 3, soft=1.5)
    base = np.array([.33, .205, .105])
    shade = 1 + .04 * fiber + .07 * mottle + .012 * flute - .18 * stain - .25 * ring + .12 * scuff
    alb = base[None, None, :] * shade[..., None]
    img = Image.fromarray((to_srgb(alb) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    ink = (52, 40, 32)
    f = ImageFont.truetype(FONT_BOLD, 34)
    x0, y0 = int(size * .58), int(size * .18)
    for k in range(2):  # "this side up" arrows
        ax0 = x0 + k * 46
        d.polygon([(ax0, y0 + 60), (ax0 + 18, y0), (ax0 + 36, y0 + 60)], fill=ink)
        d.rectangle([ax0 + 12, y0 + 58, ax0 + 24, y0 + 100], fill=ink)
    d.text((x0 - 10, y0 + 112), "THIS SIDE UP", font=f, fill=ink)
    d.rectangle([int(size * .12), int(size * .66), int(size * .44), int(size * .80)], outline=ink, width=4)
    d.text((int(size * .14), int(size * .68)), "LOT 2611  SORT B", font=ImageFont.truetype(FONT_REG, 30), fill=ink)
    printed = to_lin(np.asarray(img, float) / 255)
    printed = printed * (1 + .05 * fiber[..., None])
    save_rgb("cardboard_albedo.jpg", printed)
    save_rough("cardboard_rough.png", .05 * fiber + .1 * stain)
    hgt = .25 * flute + .3 * fiber - .4 * scuff
    save_normal("cardboard_normal.png", normal_from_height(hgt, .6))


def bale(size=1024, kind="paper"):
    """Compressed bale face: strata of crushed board (or cans), with three wire straps."""
    rng = np.random.default_rng(81 if kind == "paper" else 91)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    alb = np.zeros((h, w, 3))
    hgt = np.zeros((h, w))
    rough = np.zeros((h, w))
    metal = np.zeros((h, w))
    if kind == "paper":
        palette = [(.33, .21, .11), (.28, .18, .10), (.38, .26, .14), (.22, .15, .09), (.55, .53, .48),
                   (.42, .40, .36), (.14, .18, .26), (.34, .14, .10), (.30, .29, .27)]
        weights = np.array([7, 6, 5, 4, .35, .3, .08, .06, .5])
    else:
        palette = [(.62, .63, .65), (.50, .51, .53), (.42, .08, .07), (.08, .12, .30), (.10, .24, .12),
                   (.50, .40, .16), (.72, .72, .70), (.06, .06, .07), (.40, .41, .43)]
        weights = np.array([9, 8, 1.6, 1.2, .7, .6, 3, .8, 5])
    weights = weights / weights.sum()
    y = 0.0
    warp = noise(h, w, 1.8, 82) * 14 + noise(h, w, .8, 83) * 3
    alb[:] = np.array([.10, .066, .04]) if kind == "paper" else np.array([.16, .16, .17])
    hgt[:] = -1.2
    while y < h:
        th = rng.uniform(9, 34) if kind == "paper" else rng.uniform(5, 16)
        slope = rng.uniform(-.012, .012)
        yw = (yy + warp + slope * ((xx - w / 2 + w) % w - w / 2)) % h
        band = (yw >= y) & (yw < y + th)
        # Each stratum is broken into pieces along u.
        bounds, x = [], rng.uniform(0, 60)
        while x < w + 200:
            bounds.append(x)
            x += rng.uniform(60, 260)
        piece = np.searchsorted(np.array(bounds), np.arange(w), side="right") % max(1, len(bounds) - 1)
        n = len(bounds)
        cols = np.array([palette[i] for i in rng.choice(len(palette), size=n, p=weights)])
        cols = cols * rng.uniform(.65, 1.2, (n, 1))
        is_metal = (rng.random(n) < .75) if kind != "paper" else np.zeros(n, bool)
        heights = rng.uniform(-.4, .4, n)
        roughs = np.where(is_metal, rng.uniform(-.15, .05, n), rng.uniform(-.05, .1, n))
        ys, xs = np.nonzero(band)
        pid = piece[xs]
        # Lighter towards the middle of each sheet edge: crushed edges catch light.
        mid = 1 - .25 * np.abs((yw[ys, xs] - y) / th - .5) * 2
        alb[ys, xs] = cols[pid] * mid[:, None]
        hgt[ys, xs] = heights[pid] + .5 * mid
        metal[ys, xs] = is_metal[pid].astype(float)
        rough[ys, xs] = roughs[pid]
        gap = (yw >= y + th - 3.0) & (yw < y + th)
        alb[gap] *= .35
        hgt[gap] = -2.0
        y += th
    if kind == "paper":
        # A few printed fragments (white board with faded ink) caught in the strata.
        for _ in range(12):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            pw, ph = rng.uniform(25, 90), rng.uniform(5, 14)
            m = ((np.abs((xx - cx + w / 2) % w - w / 2) < pw / 2) & (np.abs((yy - cy + h / 2) % h - h / 2) < ph / 2))
            ink = np.array(rng.choice([(.46, .45, .42), (.40, .38, .33), (.16, .19, .27), (.30, .14, .10)]))
            alb[m] = ink * rng.uniform(.7, 1.0)
    fine = noise(h, w, .7, 84)
    alb *= (1 + .08 * fine)[..., None]
    # Three galvanised wire straps along v.
    strap = np.zeros((h, w))
    for s in range(3):
        cx = (s + .5) * w / 3 + rng.uniform(-8, 8)
        strap = np.maximum(strap, np.clip(2.5 - np.abs(xx - cx - 3 * np.sin(yy / 40)), 0, 1))
    alb = alb * (1 - strap[..., None]) + np.array([.45, .46, .47]) * strap[..., None]
    metal = np.maximum(metal, strap)
    hgt = blur(hgt, 1.2) + .2 * fine + 2.0 * blur(strap, 1)
    rough = rough + .05 * fine - .25 * strap
    prefix = "bale" if kind == "paper" else "canbale"
    save_rgb(f"{prefix}_albedo.jpg", alb)
    save_normal(f"{prefix}_normal.png", normal_from_height(hgt, 1.0))
    save_rough(f"{prefix}_rough.png", rough)
    if kind != "paper":
        save_gray(f"{prefix}_metal.png", metal)


def rubber_belt(w=1024, h=512):
    """Dark PVC/rubber conveyor belt; u along the belt. Fine grit, wear lanes, dust."""
    grit = noise(h, w, .2, 101)
    lanes = noise(h, w, 1.6, 102, ax=40, ay=1)
    dust = np.clip(noise(h, w, 2.1, 103), 0, None)
    shade = 1 + .06 * grit + .10 * lanes + .9 * dust
    alb = np.array([.030, .031, .033])[None, None, :] * shade[..., None]
    save_rgb("belt_albedo.jpg", alb)
    save_rough("belt_rough.png", .05 * grit - .12 * lanes + .15 * dust)
    save_normal("belt_normal.png", normal_from_height(.6 * grit + .2 * lanes, .5))


def hazard(w=1000, h=200):
    """Yellow/black 45 deg stripes, worn. Tile 0.5 m (u, along the band) x 0.1 m (v); period 100 mm."""
    yy, xx = np.mgrid[0:h, 0:w]
    period = w / 5
    phase = ((xx + yy) % period) / period
    stripe = blur((phase < .5).astype(float), .7)
    yellow, black = np.array([.60, .40, .018]), np.array([.018, .018, .02])
    alb = yellow[None, None] * stripe[..., None] + black[None, None] * (1 - stripe[..., None])
    wear = noise(h, w, 1.5, 111) + .5 * noise(h, w, .6, 112)
    chips = blur((wear > 1.7).astype(float), .8)
    dirt = np.clip(noise(h, w, 2.2, 113), 0, None)
    metal = np.array([.25, .25, .26])
    alb = alb * (1 - chips[..., None]) + metal * chips[..., None]
    alb *= (1 - .18 * dirt)[..., None]
    save_rgb("hazard_albedo.jpg", alb)
    save_rough("hazard_rough.png", .1 * dirt - .1 * chips)


def wood(w=1024, h=512):
    """Rough-sawn pallet pine; grain along u."""
    rng = np.random.default_rng(121)
    yy, xx = np.mgrid[0:h, 0:w]
    warp = noise(h, w, 2.0, 122, ax=8, ay=1) * 9
    rings = np.sin((yy + warp) * 2 * math.pi / 11)
    fiber = noise(h, w, .7, 123, ax=30, ay=1)
    knots = np.zeros((h, w))
    for _ in range(3):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(8, 16)
        dx = (xx - cx + w / 2) % w - w / 2
        dy = (yy - cy + h / 2) % h - h / 2
        knots = np.maximum(knots, np.exp(-(dx * dx / 3 + dy * dy) / (2 * r * r)))
    dirt = np.clip(noise(h, w, 2.2, 124), 0, None)
    base = np.array([.36, .25, .14])
    shade = 1 + .08 * rings + .07 * fiber - .45 * knots - .2 * dirt
    alb = base[None, None] * shade[..., None]
    save_rgb("wood_albedo.jpg", alb)
    save_normal("wood_normal.png", normal_from_height(.6 * fiber + .3 * rings - .6 * knots, .7))


def galvanized(size=1024, cells=220):
    """Hot-dip galvanised spangle: Voronoi crystals with brightness/roughness per cell."""
    rng = np.random.default_rng(131)
    lo = 256
    pts = rng.uniform(0, lo, (cells, 2))
    yy, xx = np.mgrid[0:lo, 0:lo]
    best = np.full((lo, lo), 1e9)
    idx = np.zeros((lo, lo), int)
    for i, (px, py) in enumerate(pts):
        dx = np.abs(xx - px)
        dy = np.abs(yy - py)
        dx = np.minimum(dx, lo - dx)
        dy = np.minimum(dy, lo - dy)
        d = dx * dx + dy * dy
        m = d < best
        best[m], idx[m] = d[m], i
    val = rng.uniform(-1, 1, cells)[idx]
    rv = rng.uniform(-1, 1, cells)[idx]
    val = np.asarray(Image.fromarray(((val + 1) * 127.5).astype(np.uint8)).resize((size, size), Image.NEAREST),
                     float) / 127.5 - 1
    rv = np.asarray(Image.fromarray(((rv + 1) * 127.5).astype(np.uint8)).resize((size, size), Image.NEAREST),
                    float) / 127.5 - 1
    fine = noise(size, size, .9, 132)
    dirt = np.clip(noise(size, size, 2.0, 133), 0, None)
    shade = 1 + .05 * val + .02 * fine - .10 * dirt
    alb = np.array([.60, .615, .625])[None, None] * shade[..., None]
    save_rgb("galv_albedo.jpg", alb)
    save_rough("galv_rough.png", .10 * rv + .04 * fine + .15 * dirt)


def machined(w=256, h=1024):
    """Turning / facing marks: stripes vary along v only (lathe UVs make them concentric)."""
    yy = np.arange(h)[:, None] * np.ones((1, w))
    marks = np.sin(2 * math.pi * yy / 3.0) * .6 + np.sin(2 * math.pi * yy / 7.3) * .4
    chatter = noise(h, w, 1.2, 141, ax=1, ay=6)
    smudge = np.clip(noise(h, w, 2.0, 142), 0, None)
    save_rough("machined_rough.png", .08 * marks + .06 * chatter + .14 * smudge)
    save_normal("machined_normal.png", normal_from_height(.6 * marks + .2 * chatter, .6))


def metal_smudge(size=1024):
    """Generic handled-metal roughness (fingerprints, polish) + micro-scratch normal."""
    rng = np.random.default_rng(151)
    smudge = blur(blobs(size, size, rng, 30, 12, 50) * (noise(size, size, 1.0, 152) > -.2), 3)
    big = noise(size, size, 2.4, 153)
    strokes = []
    for _ in range(220):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        ang, ln = rng.uniform(0, math.pi), rng.uniform(8, 60)
        strokes.append(([(x, y), (x + ln * math.cos(ang), y + ln * math.sin(ang))], rng.uniform(.3, 1)))
    scratch = stroke_mask(size, size, strokes, 1, soft=.4)
    save_rough("metal_rough.png", .18 * smudge + .08 * big - .15 * scratch)
    save_normal("metal_normal.png", normal_from_height(1.2 * scratch + .1 * big, .5))
    # Tarnish / patina variation for brass (multiplies a constant colour through scale).
    tarn = 1 - .12 * np.clip(big, 0, None) - .10 * smudge + .05 * scratch
    save_rgb("metal_albedo.jpg", np.ones((size, size, 3)) * tarn[..., None], neutral=True)


def rubber_mat(size=512):
    """Diamond-plate rubber mat for the toolbox top and bench end."""
    yy, xx = np.mgrid[0:size, 0:size]
    p = size / 16
    a = ((xx + yy) % p) / p
    b = ((xx - yy) % p) / p
    ridge = np.clip(1 - 4 * np.abs(a - .5), 0, 1) * np.clip(1 - 4 * np.abs(b - .5), 0, 1)
    dust = np.clip(noise(size, size, 2.0, 161), 0, None)
    alb = np.array([.03, .03, .032])[None, None] * (1 + .5 * dust + .2 * ridge)[..., None]
    save_rgb("mat_albedo.jpg", alb)
    save_normal("mat_normal.png", normal_from_height(2.0 * blur(ridge, 1), 1.0))


# ------------------------------------------------------------------ signs and labels
def _font(path, px):
    return ImageFont.truetype(path, px)


def _fit(d, text, font, width):
    """Shrink a font until the text fits the given width."""
    size = font.size
    while size > 8 and d.textlength(text, font=font) > width:
        size -= 2
        font = ImageFont.truetype(font.path, size)
    return font


def _centered(d, box, text, font, fill):
    font = _fit(d, text, font, (box[2] - box[0]) * .92)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    x0, y0, x1, y1 = box
    d.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t), text, font=font, fill=fill)


def _weather(img, seed, amount=.10):
    """Light grime so printed panels do not look like vector art."""
    a = to_lin(np.asarray(img.convert("RGB"), float) / 255)
    h, w = a.shape[:2]
    g = np.clip(noise(h, w, 2.0, seed), 0, None) + .4 * np.clip(noise(h, w, .8, seed + 1), 0, None)
    a *= (1 - amount * g)[..., None]
    return a


def signs():
    yellow, black, white = (236, 178, 0), (22, 22, 24), (238, 238, 232)
    # Caution: robot cell.
    img = Image.new("RGB", (1024, 720), yellow)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1023, 719], outline=black, width=26)
    d.rectangle([26, 26, 997, 210], fill=black)
    _centered(d, (26, 26, 997, 210), "CAUTION", _font(FONT_BOLD, 150), yellow)
    tri = [(150, 610), (290, 330), (430, 610)]
    d.polygon(tri, fill=black)
    d.polygon([(185, 585), (290, 375), (395, 585)], fill=yellow)
    d.rectangle([278, 430, 302, 530], fill=black)
    d.ellipse([276, 545, 304, 573], fill=black)
    for i, line in enumerate(("ROBOT WORK CELL", "KEEP CLEAR WHILE", "ARM IS MOVING")):
        font = _fit(d, line, _font(FONT_BOLD, 66 if i == 0 else 56), 500)
        d.text((462, 300 + i * 105), line, font=font, fill=black)
    save_rgb("sign_caution_albedo.jpg", _weather(img, 201))
    # Fire extinguisher.
    img = Image.new("RGB", (512, 768), (178, 24, 22))
    d = ImageDraw.Draw(img)
    d.rectangle([14, 14, 497, 753], outline=white, width=10)
    d.rounded_rectangle([196, 150, 316, 520], radius=40, fill=white)
    d.rectangle([228, 96, 284, 150], fill=white)
    d.line([(284, 116), (360, 140), (380, 230)], fill=white, width=16)
    _centered(d, (20, 560, 492, 640), "FIRE", _font(FONT_BOLD, 84), white)
    _centered(d, (20, 640, 492, 730), "EXTINGUISHER", _font(FONT_BOLD, 62), white)
    save_rgb("sign_fire_albedo.jpg", _weather(img, 203))
    # Mandatory eye protection.
    img = Image.new("RGB", (768, 1024), white)
    d = ImageDraw.Draw(img)
    d.ellipse([109, 60, 659, 610], fill=(16, 82, 160))
    d.rounded_rectangle([210, 270, 360, 380], radius=40, outline=white, width=24)
    d.rounded_rectangle([408, 270, 558, 380], radius=40, outline=white, width=24)
    d.line([(360, 300), (408, 300)], fill=white, width=20)
    d.line([(210, 300), (160, 285)], fill=white, width=20)
    d.line([(558, 300), (608, 285)], fill=white, width=20)
    for i, line in enumerate(("EYE PROTECTION", "MUST BE WORN")):
        _centered(d, (20, 680 + i * 120, 748, 790 + i * 120), line, _font(FONT_BOLD, 80), black)
    save_rgb("sign_ppe_albedo.jpg", _weather(img, 205))
    # Line identification banner (recycling).
    img = Image.new("RGB", (2048, 512), (40, 62, 52))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 2047, 511], outline=white, width=16)
    _centered(d, (0, 40, 2048, 330), "SORTING LINE 2", _font(FONT_BOLD, 230), white)
    _centered(d, (0, 330, 2048, 480), "MATERIALS RECOVERY  -  AUTHORIZED STAFF ONLY", _font(FONT_BOLD, 92),
              (200, 214, 204))
    save_rgb("sign_line_albedo.jpg", _weather(img, 207))
    # Workshop banner.
    img = Image.new("RGB", (2048, 512), (34, 40, 48))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 2047, 511], outline=(236, 178, 0), width=16)
    _centered(d, (0, 40, 2048, 330), "ASSEMBLY CELL 04", _font(FONT_BOLD, 230), white)
    _centered(d, (0, 330, 2048, 480), "TORQUE TOOLS  -  CALIBRATED FIXTURES", _font(FONT_BOLD, 92), (196, 202, 210))
    save_rgb("sign_cell_albedo.jpg", _weather(img, 209))
    # Bin placards.
    for name, color in (("plastic", (22, 86, 168)), ("metal", (206, 150, 0)), ("glass", (30, 120, 60)),
                        ("paper", (120, 84, 46))):
        img = Image.new("RGB", (768, 384), white)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 767, 150], fill=color)
        _centered(d, (0, 0, 768, 150), name.upper(), _font(FONT_BOLD, 120), white)
        _centered(d, (0, 170, 768, 270), "SORTED " + name.upper() + " ONLY", _font(FONT_BOLD, 54), black)
        _centered(d, (0, 280, 768, 360), "NO FOOD WASTE", _font(FONT_REG, 46), (80, 80, 80))
        d.rectangle([0, 0, 767, 383], outline=(60, 60, 60), width=8)
        save_rgb(f"placard_{name}_albedo.jpg", _weather(img, 211 + len(name), .06))


def can_label(w=1024, h=512):
    """Fictional 'FIZZ' cola wrap: printed on aluminium (u around the can, v up the body)."""
    red = (186, 18, 26)
    img = Image.new("RGB", (w, h), red)
    d = ImageDraw.Draw(img)
    for k in range(w):
        y = int(h * .30 + 26 * math.sin(2 * math.pi * k / w * 2))
        d.line([(k, y), (k, y + 14)], fill=(245, 240, 232))
        d.line([(k, y + 26), (k, y + 32)], fill=(245, 240, 232))
    big = _font(FONT_BOLD, 210)
    d.text((40, int(h * .38)), "FIZZ", font=big, fill=(250, 248, 242))
    d.text((70, int(h * .80)), "CLASSIC COLA", font=_font(FONT_BOLD, 44), fill=(250, 240, 230))
    d.text((560, int(h * .08)), "330 ml", font=_font(FONT_BOLD, 40), fill=(250, 240, 230))
    # Back panel: nutrition table and barcode.
    x0, y0 = 640, int(h * .42)
    d.rectangle([x0, y0, x0 + 230, y0 + 170], fill=(250, 248, 242))
    for i in range(6):
        d.line([(x0 + 8, y0 + 22 + i * 24), (x0 + 222, y0 + 22 + i * 24)], fill=(120, 120, 120), width=2)
    d.text((x0 + 10, y0 + 2), "NUTRITION", font=_font(FONT_BOLD, 18), fill=(20, 20, 20))
    bx, by = 900, int(h * .46)
    d.rectangle([bx, by, bx + 100, by + 120], fill=(250, 250, 250))
    rng = np.random.default_rng(7)
    x = bx + 6
    while x < bx + 94:
        wd = int(rng.integers(1, 4))
        d.rectangle([x, by + 8, x + wd - 1, by + 100], fill=(15, 15, 15))
        x += wd + int(rng.integers(1, 3))
    d.text((x0, int(h * .80)), "RECYCLE ME", font=_font(FONT_BOLD, 30), fill=(250, 240, 230))
    a = to_lin(np.asarray(img, float) / 255)
    a *= (1 + .03 * noise(h, w, 1.5, 171))[..., None]
    save_rgb("can_label_albedo.jpg", a)


def bottle_label(w=1024, h=256):
    img = Image.new("RGB", (w, h), (246, 248, 250))
    d = ImageDraw.Draw(img)
    pts = [(0, h)]
    for k in range(0, w + 1, 8):
        pts.append((k, int(h * .62 - 40 * abs(math.sin(k / w * 2 * math.pi * 3)) - 10 * math.sin(k / 37))))
    pts.append((w, h))
    d.polygon(pts, fill=(120, 168, 214))
    d.rectangle([0, int(h * .86), w, h], fill=(26, 84, 160))
    d.text((40, 30), "SPRING", font=_font(FONT_BOLD, 76), fill=(26, 84, 160))
    d.text((340, 30), "WATER", font=_font(FONT_REG, 76), fill=(26, 84, 160))
    d.text((640, 50), "natural mineral  500 ml", font=_font(FONT_REG, 34), fill=(60, 100, 150))
    save_rgb("bottle_label_albedo.jpg", to_lin(np.asarray(img, float) / 255))


def paper(w=1024, h=512, folds=46):
    """Crumpled office paper on an equirect (lat-long) map; creases evaluated in 3D."""
    rng = np.random.default_rng(181)
    v = (np.arange(h) + .5) / h
    u = (np.arange(w) + .5) / w
    lat = (v - .5) * math.pi
    lon = u * 2 * math.pi
    LON, LAT = np.meshgrid(lon, lat[::-1])
    P = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    hgt = np.zeros((h, w))
    for _ in range(folds):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        dist = P @ n - rng.uniform(-.7, .7)
        sharp = rng.uniform(.015, .05)
        hgt += rng.choice((-1, 1)) * rng.uniform(.3, 1) * np.exp(-np.abs(dist) / sharp)
    hgt += .15 * noise(h, w, 1.2, 182)
    # Gradient on the sphere: longitude spacing shrinks with cos(lat).
    dlon = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * .5 / np.maximum(np.cos(LAT), .08)
    dlat = (np.roll(hgt, 1, 0) - np.roll(hgt, -1, 0)) * .5
    dlat[0], dlat[-1] = 0, 0
    s = .9
    n = np.stack([-dlon * s, -dlat * s, np.ones_like(hgt)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    save_normal("paper_normal.png", n)
    # Albedo: white paper, faint printed lines, crease grime.
    lines = ((np.floor((P[..., 2] * 9 + P[..., 0] * 2.3) * 6) % 2) == 0) & (np.abs(P[..., 1] + .2 * P[..., 0]) < .55)
    crease = np.clip(np.abs(hgt) - .6, 0, None)
    shade = 1 - .07 * lines - .08 * crease + .02 * noise(h, w, .8, 183)
    alb = np.array([.80, .80, .78])[None, None] * shade[..., None]
    save_rgb("paper_albedo.jpg", alb)


def screen_ui(w=1024, h=640):
    """CNC control screen (emissive)."""
    img = Image.new("RGB", (w, h), (10, 20, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 60], fill=(18, 60, 96))
    d.text((20, 10), "SPINDLE  READY", font=_font(FONT_BOLD, 40), fill=(200, 230, 255))
    for i, (lab, val) in enumerate((("X", "-124.350"), ("Y", "  38.020"), ("Z", "  12.500"), ("S", "8000 RPM"))):
        d.text((40, 100 + i * 110), lab, font=_font(FONT_BOLD, 72), fill=(120, 200, 255))
        d.text((140, 100 + i * 110), val, font=_font(FONT_BOLD, 72), fill=(230, 240, 250))
    d.rectangle([620, 100, 990, 560], outline=(60, 120, 170), width=4)
    for k in range(20):
        d.line([(640, 520 - k * 20), (970, 520 - (k * 17) % 400)], fill=(40, 90, 130), width=2)
    d.rectangle([620, 580, 800, 625], fill=(30, 140, 70))
    d.text((640, 586), "CYCLE OK", font=_font(FONT_BOLD, 30), fill=(230, 255, 230))
    save_rgb("screen_albedo.jpg", to_lin(np.asarray(img, float) / 255))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [concrete_floor, floor_paint, block_wall, brushed_steel, painted_metal, pegboard, cardboard,
             lambda: bale(kind="paper"), lambda: bale(kind="cans"), rubber_belt, hazard, wood, galvanized,
             machined, metal_smudge, rubber_mat, signs, can_label, bottle_label, paper, screen_ui]
    for step in steps:
        step()
        print("done", getattr(step, "__name__", "bale"), flush=True)
    files = {}
    for p in sorted(OUT.iterdir()):
        if p.suffix in (".png", ".jpg"):
            files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (OUT / "procedural_textures.json").write_text(json.dumps(
        {"generator": "demo/lab/make_lab_textures.py", "license": "CC0-1.0",
         "note": "Original procedural textures (numpy + Pillow); fonts: Liberation Sans (SIL OFL 1.1) "
                 "rasterised into sign and label images.", "files": files}, indent=1) + "\n")


if __name__ == "__main__":
    main()
