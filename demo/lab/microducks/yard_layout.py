"""Schoolyard layout shared by the Isaac Lab + Newton simulation and the OVRTX stage (metres, z up, yaw in rad)."""
import math

import numpy as np

NDUCK = 12
YARD = 2.5            # half size of the paved floor
WALL_T = 0.06         # wall thickness
WALL_H = 0.14         # wall height
WALL_IN = YARD - WALL_T  # inner face of the walls

# (name, centre x, centre y, yaw, up length, top length, down length, height, width)
RAMPS = [
    ("ramp_west", -1.15, 0.75, 0.0, 0.70, 0.40, 0.70, 0.050, 0.70),
    ("ramp_north", 1.10, 1.45, 0.0, 0.60, 0.40, 0.60, 0.055, 0.80),
    ("ramp_south", -0.20, -1.55, math.pi / 2, 0.55, 0.30, 0.55, 0.045, 0.60),
]
CONES = [(0.15, 0.45), (1.25, -0.95), (-1.55, -0.2), (-1.0, 2.05)]
PLANTERS = [(-1.9, -2.15, 0.0), (2.05, 2.1, math.pi / 2), (0.95, -2.2, 0.0)]
PLANTER_SIZE = (0.44, 0.16, 0.08)
BALLS = [(0.6, -0.2), (-0.6, 0.15), (1.6, 0.6)]
BALL_R = 0.035
CONE_R, CONE_H = 0.036, 0.14

DUCK_COLORS = [(0.95, 0.25, 0.25), (0.98, 0.60, 0.15), (0.98, 0.88, 0.20), (0.45, 0.85, 0.30),
               (0.15, 0.70, 0.55), (0.25, 0.80, 0.95), (0.25, 0.45, 0.95), (0.55, 0.35, 0.90),
               (0.90, 0.35, 0.80), (0.98, 0.98, 0.98), (0.20, 0.20, 0.22), (0.65, 0.45, 0.30)]


def quat_xyzw_from_yaw_pitch(yaw, pitch=0.0):
    """xyzw quaternion for R = Rz(yaw) * Ry(pitch)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    w = cy * cp
    x = -sy * sp
    y = cy * sp
    z = sy * cp
    return (x, y, z, w)


def ramp_boxes(r):
    """Ramp as oriented boxes: list of (name, centre xyz, half sizes xyz, quat xyzw, role)."""
    name, cx, cy, yaw, lu, lt, ld, h, w = r
    total = lu + lt + ld
    cyw, syw = math.cos(yaw), math.sin(yaw)

    def place(lx, lz):
        return (cx + cyw * lx, cy + syw * lx, lz)

    out = []
    x0 = -total / 2
    t = h + 0.03
    for k, (xa, xb, za, zb) in enumerate([(x0, x0 + lu, 0.0, h), (x0 + lu + lt, x0 + total, h, 0.0)]):
        L = xb - xa
        th = math.atan2(zb - za, L)
        ln = math.hypot(L, zb - za)
        mx, mz = (xa + xb) / 2, (za + zb) / 2
        nx, nz = -math.sin(th), math.cos(th)
        out.append((f"{name}_slope{k}", place(mx - nx * t / 2, mz - nz * t / 2), (ln / 2, w / 2, t / 2),
                    quat_xyzw_from_yaw_pitch(yaw, -th), "slope"))
    out.append((f"{name}_top", place(x0 + lu + lt / 2, h / 2), (lt / 2, w / 2, h / 2),
                quat_xyzw_from_yaw_pitch(yaw), "top"))
    return out


def ramp_endpoints(r, margin=0.35):
    name, cx, cy, yaw, lu, lt, ld, h, w = r
    total = lu + lt + ld
    d = np.array([math.cos(yaw), math.sin(yaw)])
    c = np.array([cx, cy])
    return c - d * (total / 2 + margin), c + d * (total / 2 + margin)


def wall_boxes():
    s = YARD - WALL_T / 2
    return [("wall_n", (0.0, s, WALL_H / 2), (YARD, WALL_T / 2, WALL_H / 2)),
            ("wall_s", (0.0, -s, WALL_H / 2), (YARD, WALL_T / 2, WALL_H / 2)),
            ("wall_e", (s, 0.0, WALL_H / 2), (WALL_T / 2, YARD, WALL_H / 2)),
            ("wall_w", (-s, 0.0, WALL_H / 2), (WALL_T / 2, YARD, WALL_H / 2))]
