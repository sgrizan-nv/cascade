"""Procedural clear-day sky (latlong RGBE .hdr): bright pale horizon -> light blue zenith, neutral ground below.
No sun disc (the DistantLight provides the sun); only a soft broad brightening around the sun azimuth."""
import math
import sys

import numpy as np

out = sys.argv[1]
W, H = 2048, 1024
v = (np.arange(H) + 0.5) / H            # 0 = top (zenith), 1 = bottom (nadir)
u = (np.arange(W) + 0.5) / W
elev = (0.5 - v) * math.pi              # +pi/2 zenith .. -pi/2 nadir
E = np.repeat(elev[:, None], W, 1)
horizon = np.array([0.98, 1.02, 1.05])
zenith = np.array([0.42, 0.64, 0.98])
ground = np.array([0.30, 0.29, 0.27])
k = np.clip(E / (math.pi / 2), 0, 1) ** 0.55
sky = horizon[None, None] * (1 - k[..., None]) + zenith[None, None] * k[..., None]
haze = np.exp(-np.clip(E, 0, None) / 0.12)[..., None] * 0.18      # thin bright band at the horizon
img = np.where((E >= 0)[..., None], sky + haze, ground[None, None] * (0.6 + 0.4 * np.exp(E / 0.2))[..., None])
az = np.repeat(((u - 0.5) * 2 * math.pi)[None, :], H, 0)
glow = np.exp(-((np.angle(np.exp(1j * (az - math.radians(-54.0))))) ** 2) / 0.9) * np.exp(-np.abs(E - 0.5) / 0.6)
img = img * (1 + 0.25 * glow[..., None])
img = img.astype(np.float32) * 0.55


def write_hdr(path, rgb):
    """Radiance RGBE with new-style RLE scanlines (literal chunks only), as most loaders expect for 8 <= W < 32768."""
    h, w, _ = rgb.shape
    mx = rgb.max(-1)
    e = np.floor(np.log2(np.maximum(mx, 1e-32))) + 1
    m = rgb / np.ldexp(1.0, e.astype(int))[..., None] * 256.0
    rgbe = np.zeros((h, w, 4), np.uint8)
    rgbe[..., :3] = np.clip(m, 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.clip(e + 128, 0, 255).astype(np.uint8)
    rgbe[mx < 1e-32] = 0
    out = bytearray(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode())
    head = bytes([2, 2, w >> 8, w & 255])
    for y in range(h):
        out += head
        for c in range(4):
            row = rgbe[y, :, c].tobytes()
            for i in range(0, w, 128):
                chunk = row[i:i + 128]
                out.append(len(chunk))
                out += chunk
    with open(path, "wb") as f:
        f.write(out)


write_hdr(out, img)
print(out, "upper-hemisphere mean", float(img[: H // 2].mean()), "zenith", img[0, 0], "horizon", img[H // 2 - 2, 0])
