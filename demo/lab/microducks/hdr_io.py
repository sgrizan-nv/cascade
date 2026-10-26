"""Minimal Radiance RGBE reader/writer (new-style RLE)."""
import numpy as np


def read_hdr(path):
    data = open(path, "rb").read()
    i = data.index(b"\n\n") + 2
    j = data.index(b"\n", i)
    dims = data[i:j].split()
    h, w = int(dims[1]), int(dims[3])
    p = j + 1
    out = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        assert data[p] == 2 and data[p + 1] == 2, "only new RLE supported"
        p += 4
        for c in range(4):
            x = 0
            row = out[y, :, c]
            while x < w:
                n = data[p]
                p += 1
                if n > 128:
                    n -= 128
                    row[x:x + n] = data[p]
                    p += 1
                else:
                    row[x:x + n] = np.frombuffer(data, np.uint8, n, p)
                    p += n
                x += n
    m = out[..., :3].astype(np.float32)
    e = out[..., 3].astype(np.int32)
    f = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    return (m + 0.5) * f[..., None] * (e[..., None] > 0)


def write_hdr(path, rgb):
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
    open(path, "wb").write(out)
