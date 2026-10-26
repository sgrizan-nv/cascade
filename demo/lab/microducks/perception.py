"""Perception for the 12 Microducks: fixed RGB-D + segmentation cameras -> 3D map -> duck poses.

Nothing here reads simulator state. Inputs are the simulated camera images (depth in metres, semantic and instance
segmentation ids from the renderer) plus the camera calibration (intrinsics from the lens model, extrinsics from where
the cameras are mounted). Outputs: an occupancy map of the yard (walls, cones, planters, ramps) built once before the
ducks move, and a pose (x, y, heading) per duck at the perception rate.

Torch on the GPU for the per-pixel work, numpy for the map analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------- camera rig (world frame, metres)
W_PX, H_PX, HFOV_DEG = 848, 480, 87.0      # RealSense D435-like depth stream: 848 x 480, 87 deg x 58 deg


@dataclass
class CamSpec:
    name: str
    eye: tuple
    target: tuple
    w: int = W_PX
    h: int = H_PX
    hfov: float = HFOV_DEG

    @property
    def K(self):
        fx = (self.w / 2) / math.tan(math.radians(self.hfov) / 2)
        return np.array([[fx, 0.0, self.w / 2], [0.0, fx, self.h / 2], [0.0, 0.0, 1.0]])

    def axes(self):
        """forward, left, up unit vectors (world convention of Isaac Lab cameras: +X forward, +Z up)."""
        f = np.asarray(self.target, float) - np.asarray(self.eye, float)
        f /= np.linalg.norm(f)
        l = np.cross([0.0, 0.0, 1.0], f)
        l /= np.linalg.norm(l)
        u = np.cross(f, l)
        return f, l, u

    def quat_world_xyzw(self):
        f, l, u = self.axes()
        return quat_from_matrix(np.stack([f, l, u], axis=1))

    def R_ros(self):
        """camera (x right, y down, z forward) -> world rotation."""
        f, l, u = self.axes()
        return np.stack([-l, -u, f], axis=1)

    @property
    def vfov(self):
        return math.degrees(2 * math.atan(math.tan(math.radians(self.hfov) / 2) * self.h / self.w))


RIG = [CamSpec("ne", (2.85, 2.85, 2.4), (0.35, 0.35, 0.0)),
       CamSpec("nw", (-2.85, 2.85, 2.4), (-0.35, 0.35, 0.0)),
       CamSpec("sw", (-2.85, -2.85, 2.4), (-0.35, -0.35, 0.0)),
       CamSpec("se", (2.85, -2.85, 2.4), (0.35, -0.35, 0.0))]

CLASSES = ["background", "unlabelled", "floor", "wall", "ramp", "cone", "planter",
           "duck", "duck_head", "duck_leg_left", "duck_leg_right"]
CID = {c: i for i, c in enumerate(CLASSES)}
DUCK_CLASSES = ("duck", "duck_head", "duck_leg_left", "duck_leg_right")
STATIC_OBSTACLE = ("wall", "cone", "planter")
PARTS = {"duck": 0, "duck_head": 1, "duck_leg_left": 2, "duck_leg_right": 3}


def quat_from_matrix(Rm):
    """xyzw quaternion of a rotation matrix."""
    m = np.asarray(Rm, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return tuple(q / np.linalg.norm(q))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- label tables (from the renderer info)
def class_table(sem_info, n_ids=4096):
    """semantic id -> our class index, from the renderer's idToLabels ({id: {"class": "duck"}})."""
    lut = np.full(n_ids, CID["unlabelled"], np.int64)
    for k, v in sem_info.items():
        k = int(k)
        name = v.get("class", "") if isinstance(v, dict) else str(v)
        name = {"BACKGROUND": "background", "UNLABELLED": "unlabelled"}.get(name, name)
        if 0 <= k < n_ids and name in CID:
            lut[k] = CID[name]
    return lut


def instance_table(inst_info, n_ids=4096):
    """instance id -> (duck index, part index) from the instance prim paths (".../Duck07/..." = track id 7)."""
    duck = np.full(n_ids, -1, np.int64)
    part = np.full(n_ids, -1, np.int64)
    labels, sems = inst_info.get("idToLabels", {}), inst_info.get("idToSemantics", {})
    for k, path in labels.items():
        k = int(k)
        path = str(path)
        if "/Duck" not in path or not (0 <= k < n_ids):
            continue
        tok = path.split("/Duck")[1][:2]
        if not tok.isdigit():
            continue
        cls = sems.get(k, sems.get(str(k), {}))
        cls = cls.get("class", "duck") if isinstance(cls, dict) else "duck"
        duck[k] = int(tok)
        part[k] = PARTS.get(cls, 0)
    return duck, part


# ----------------------------------------------------------------------------- per-pixel back-projection (torch)
class Backprojector:
    def __init__(self, spec: CamSpec, device, noise_k=0.0, seed=0):
        import torch
        self.spec, self.device, self.noise_k = spec, device, noise_k
        K = spec.K
        v, u = torch.meshgrid(torch.arange(spec.h, device=device, dtype=torch.float32),
                              torch.arange(spec.w, device=device, dtype=torch.float32), indexing="ij")
        self.rx = (u + 0.5 - K[0, 2]) / K[0, 0]
        self.ry = (v + 0.5 - K[1, 2]) / K[1, 1]
        self.R = torch.tensor(spec.R_ros(), dtype=torch.float32, device=device)
        self.t = torch.tensor(spec.eye, dtype=torch.float32, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def noisy(self, depth):
        """Stereo-like depth noise: sigma = k * z^2, 1 % dropouts, 1 mm quantisation."""
        import torch
        if self.noise_k <= 0:
            return depth
        z = depth
        n = torch.randn(z.shape, generator=self.gen, device=z.device) * (self.noise_k * z * z)
        drop = torch.rand(z.shape, generator=self.gen, device=z.device) < 0.01
        out = torch.round((z + n) * 1000.0) / 1000.0
        return torch.where(drop | (z <= 0), torch.zeros_like(z), out)

    def points(self, depth):
        """(H, W) forward depth -> (H, W, 3) world points (rows with depth <= 0 are invalid)."""
        import torch
        z = depth
        pc = torch.stack([self.rx * z, self.ry * z, z], dim=-1)
        return pc @ self.R.T + self.t


# ----------------------------------------------------------------------------- mapping (before the ducks move)
GRID_RES = 0.02
GRID_HALF = 2.70
GRID_N = int(round(2 * GRID_HALF / GRID_RES))


def cell_of(xy):
    return np.floor((np.asarray(xy) + GRID_HALF) / GRID_RES).astype(int)


def cell_centre(ix, iy):
    return np.stack([(ix + 0.5) * GRID_RES - GRID_HALF, (iy + 0.5) * GRID_RES - GRID_HALF], -1)


class Mapper:
    """Accumulates static points (everything that is not a duck) into a 2 cm grid: class votes + heights."""

    def __init__(self, device):
        import torch
        self.device = device
        n = GRID_N * GRID_N
        self.votes = torch.zeros(n * len(CLASSES), dtype=torch.float32, device=device)
        self.zmax = torch.full((n,), -1.0, dtype=torch.float32, device=device)
        self.zsum = torch.zeros(n, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(n, dtype=torch.float32, device=device)
        self.frames = 0
        self.points_seen = 0

    def add(self, pts, cls):
        """pts (N, 3) world, cls (N,) class index; only static classes are kept."""
        import torch
        keep = (cls >= CID["floor"]) & (cls < CID["duck"])
        pts, cls = pts[keep], cls[keep]
        ij = torch.floor((pts[:, :2] + GRID_HALF) / GRID_RES).long()
        ok = (ij >= 0).all(1) & (ij < GRID_N).all(1)
        ij, pts, cls = ij[ok], pts[ok], cls[ok]
        lin = ij[:, 0] * GRID_N + ij[:, 1]
        self.votes.index_add_(0, lin * len(CLASSES) + cls, torch.ones_like(lin, dtype=torch.float32))
        self.zmax.scatter_reduce_(0, lin, pts[:, 2], reduce="amax")
        self.zsum.index_add_(0, lin, pts[:, 2])
        self.cnt.index_add_(0, lin, torch.ones_like(pts[:, 2]))
        self.frames += 1
        self.points_seen += int(lin.numel())

    def finalize(self):
        votes = self.votes.view(GRID_N, GRID_N, len(CLASSES)).cpu().numpy()
        cnt = self.cnt.view(GRID_N, GRID_N).cpu().numpy()
        zmean = self.zsum.view(GRID_N, GRID_N).cpu().numpy() / np.maximum(cnt, 1)
        return PerceivedMap.from_votes(votes, cnt, zmean, self.frames, self.points_seen)


def _components(mask):
    """4-connected components of a boolean grid -> list of (ix array, iy array)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        out = []
        for k in range(1, n + 1):
            ix, iy = np.nonzero(lab == k)
            out.append((ix, iy))
        return out
    except ImportError:
        pass
    seen = np.zeros_like(mask, bool)
    out = []
    for sx, sy in zip(*np.nonzero(mask)):
        if seen[sx, sy]:
            continue
        stack, xs, ys = [(sx, sy)], [], []
        seen[sx, sy] = True
        while stack:
            x, y = stack.pop()
            xs.append(x)
            ys.append(y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = x + dx, y + dy
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        out.append((np.array(xs), np.array(ys)))
    return out


@dataclass
class PerceivedMap:
    """What the cameras mapped. Occupancy codes: 0 unknown, 1 floor, 2 obstacle (wall / cone / planter), 3 ramp."""
    occ: np.ndarray
    height: np.ndarray
    ramps: list = field(default_factory=list)
    cones: list = field(default_factory=list)
    planters: list = field(default_factory=list)
    walls: dict = field(default_factory=dict)
    coverage: float = 0.0
    frames: int = 0
    points: int = 0

    @staticmethod
    def from_votes(votes, cnt, zmean, frames, points):
        cls = votes.argmax(-1)
        seen = votes.sum(-1) > 0
        occ = np.zeros(cls.shape, np.uint8)
        occ[seen & (cls == CID["floor"])] = 1
        occ[seen & np.isin(cls, [CID[c] for c in STATIC_OBSTACLE])] = 2
        occ[seen & (cls == CID["ramp"])] = 3
        m = PerceivedMap(occ=occ, height=np.where(seen, zmean, np.nan).astype(np.float32), frames=frames, points=points)
        m._wall_votes = votes[..., CID["wall"]]
        m._analyse(cls, seen)
        return m

    # -- objects from the grid
    def _analyse(self, cls, seen):
        H = self.height
        # walls: inner faces per side from the wall cells near each border
        # (only cells with many wall votes: far-camera depth noise scatters a few wall points inwards)
        wall = seen & (cls == CID["wall"]) & (self._wall_votes >= 6)
        ix, iy = np.nonzero(wall)
        c = cell_centre(ix, iy)
        walls = {}
        for side, k, sgn in (("xmax", 0, 1), ("xmin", 0, -1), ("ymax", 1, 1), ("ymin", 1, -1)):
            sel = (c[:, k] * sgn > 1.5) & (np.abs(c[:, 1 - k]) < 1.8)     # this wall only, away from the corners
            if sel.sum() > 20:
                walls[side] = float(sgn * np.percentile(c[sel, k] * sgn, 10.0) - sgn * GRID_RES / 2)
        self.walls = walls
        inner = self.inner_box(margin=0.0)
        # interior coverage: cells inside the walls that some camera observed
        gx = cell_centre(np.arange(GRID_N), np.zeros(GRID_N, int))[:, 0]
        insx = (gx > inner[0]) & (gx < inner[1])
        insy = (gx > inner[2]) & (gx < inner[3])
        ins = insx[:, None] & insy[None, :]
        self.coverage = float(seen[ins].mean()) if ins.any() else 0.0
        # ramps
        self.ramps = []
        for ix, iy in _components(seen & (cls == CID["ramp"])):
            if len(ix) * GRID_RES ** 2 < 0.08:
                continue
            p = cell_centre(ix, iy)
            mu = p.mean(0)
            ev, evec = np.linalg.eigh(np.cov((p - mu).T))
            ax = evec[:, 1]
            yaw = math.atan2(ax[1], ax[0])
            if abs(wrap(yaw)) > math.pi / 2:
                yaw = wrap(yaw + math.pi)
            ax = np.array([math.cos(yaw), math.sin(yaw)])
            nr = np.array([-ax[1], ax[0]])
            s, r = (p - mu) @ ax, (p - mu) @ nr
            s0, s1 = np.percentile(s, [0.5, 99.5])
            r0, r1 = np.percentile(r, [0.5, 99.5])
            L, Wd = s1 - s0 + GRID_RES, r1 - r0 + GRID_RES
            cen = mu + ax * (s0 + s1) / 2 + nr * (r0 + r1) / 2
            h = H[ix, iy]
            hmax = float(np.nanpercentile(h, 98))
            # height profile along the axis -> up slope / top / down slope lengths
            bins = np.linspace(s0, s1, 41)
            prof = np.array([np.nanmedian(h[(s >= a) & (s < b)]) if np.any((s >= a) & (s < b)) else np.nan
                             for a, b in zip(bins[:-1], bins[1:])])
            top = np.nonzero(prof > 0.85 * hmax)[0]
            if len(top):
                lu = (bins[top[0]] - s0)
                ld = (s1 - bins[top[-1] + 1])
                lt = L - lu - ld
            else:
                lu = ld = L * 0.4
                lt = L * 0.2
            ang = math.degrees(math.atan2(cen[1], cen[0]))
            name = min((("north", 90), ("west", 180), ("south", -90), ("east", 0)),
                       key=lambda kv: abs(wrap(math.radians(ang - kv[1]))))[0]
            self.ramps.append(dict(name=name, c=[float(cen[0]), float(cen[1])], yaw=float(yaw), L=float(L),
                                   W=float(Wd), H=hmax, lu=float(lu), lt=float(lt), ld=float(ld), cells=int(len(ix))))
        # cones and planters: blobs
        self.cones, self.planters = [], []
        for name, out, rmin in (("cone", self.cones, 0.0015), ("planter", self.planters, 0.01)):
            for ix, iy in _components(seen & (cls == CID[name])):
                if len(ix) * GRID_RES ** 2 < rmin:
                    continue
                p = cell_centre(ix, iy)
                mu = p.mean(0)
                ext = np.ptp(p, axis=0) + GRID_RES
                out.append(dict(c=[float(mu[0]), float(mu[1])], size=[float(ext[0]), float(ext[1])],
                                h=float(np.nanpercentile(H[ix, iy], 98)), cells=int(len(ix))))

    def inner_box(self, margin=0.0):
        w = self.walls
        return (w.get("xmin", -2.44) + margin, w.get("xmax", 2.44) - margin,
                w.get("ymin", -2.44) + margin, w.get("ymax", 2.44) - margin)

    def ramp(self, name):
        return next((r for r in self.ramps if r["name"] == name), None)

    def obstacles(self):
        """(x, y, r) circles for the reactive avoidance: cones r 0.09, planters from their size."""
        out = [(c["c"][0], c["c"][1], 0.09) for c in self.cones]
        out += [(p["c"][0], p["c"][1], max(0.2, 0.5 * max(p["size"]))) for p in self.planters]
        return out

    def summary(self):
        return dict(walls_inner={k: round(v, 3) for k, v in self.walls.items()},
                    ramps=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in self.ramps],
                    cones=[dict(c=[round(v, 3) for v in c["c"]], h=round(c["h"], 3)) for c in self.cones],
                    planters=[dict(c=[round(v, 3) for v in p["c"]], size=[round(v, 3) for v in p["size"]])
                              for p in self.planters],
                    coverage=round(self.coverage, 4), frames=self.frames, points=self.points,
                    grid=dict(res_m=GRID_RES, n=GRID_N, origin=[-GRID_HALF, -GRID_HALF]))

    def save(self, path):
        np.savez_compressed(path, occ=self.occ, height=self.height, res=GRID_RES, origin=np.array([-GRID_HALF, -GRID_HALF]))


# ----------------------------------------------------------------------------- localisation (every perception tick)
NDUCK = 12
SURF_BIAS = 0.025       # a camera sees the near surface: shift its centroid this far along the viewing direction


class Localizer:
    """Per duck: position from the segmented depth points of every camera, heading from the left / right legs (and the
    walking direction), fused with the navigator's own velocity command between measurements."""

    def __init__(self, device, min_px=25):
        self.device = device
        self.min_px = min_px
        self.pos = np.full((NDUCK, 2), np.nan)
        self.yaw = np.zeros(NDUCK)
        self.yaw_ok = np.zeros(NDUCK, bool)
        self.ztop = np.zeros(NDUCK)
        self.npx = np.zeros(NDUCK, int)
        self.ncam = np.zeros(NDUCK, int)
        self.last_seen = np.full(NDUCK, -1e9)
        self.hist = [[] for _ in range(NDUCK)]          # (t, x, y) for the motion heading
        self.meas_legs = np.full(NDUCK, np.nan)
        self.meas_head = np.full(NDUCK, np.nan)
        self.meas_motion = np.full(NDUCK, np.nan)
        self.t = None

    def update(self, t, cams, cmd):
        """cams: list of dicts {pts (H,W,3) torch, duck (H,W) long, part (H,W) long, eye (3,)}; cmd (12, 3) vx, vy, wz
        actually sent by the navigator (smoothed)."""
        import torch
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        # predict with the navigator's own command (dead reckoning between measurements)
        if dt > 0:
            for i in range(NDUCK):
                if np.isfinite(self.pos[i, 0]):
                    c, s = math.cos(self.yaw[i]), math.sin(self.yaw[i])
                    self.pos[i] += dt * np.array([c * cmd[i, 0] - s * cmd[i, 1], s * cmd[i, 0] + c * cmd[i, 1]])
                self.yaw[i] = wrap(self.yaw[i] + dt * cmd[i, 2])
        nseg = NDUCK * 4
        acc_xy = torch.zeros((len(cams), nseg, 2), device=self.device)
        acc_n = torch.zeros((len(cams), nseg), device=self.device)
        acc_ztop = torch.zeros(NDUCK, device=self.device)
        eyes = []
        for k, c in enumerate(cams):
            m = c["duck"] >= 0
            if not bool(m.any()):
                eyes.append(np.asarray(c["eye"], float))
                continue
            p = c["pts"][m]
            seg = c["duck"][m] * 4 + c["part"][m]
            acc_xy[k].index_add_(0, seg, p[:, :2])
            acc_n[k].index_add_(0, seg, torch.ones_like(p[:, 0]))
            acc_ztop.scatter_reduce_(0, c["duck"][m], p[:, 2], reduce="amax")
            eyes.append(np.asarray(c["eye"], float))
        xy = acc_xy.cpu().numpy()
        n = acc_n.cpu().numpy()
        ztop = acc_ztop.cpu().numpy()
        for i in range(NDUCK):
            sl = slice(4 * i, 4 * i + 4)
            n_cam = n[:, sl].sum(1)                        # px per camera for this duck
            tot = n_cam.sum()
            self.npx[i] = int(tot)
            self.ncam[i] = int((n_cam >= self.min_px).sum())
            if tot < self.min_px:
                continue
            est = np.zeros(2)
            wsum = 0.0
            part_xy = np.zeros((4, 2))
            part_n = np.zeros(4)
            for k in range(len(cams)):
                if n_cam[k] < self.min_px:
                    continue
                cxy = xy[k, sl].sum(0) / n_cam[k]
                v = cxy - eyes[k][:2]
                v /= np.linalg.norm(v) + 1e-9
                est += n_cam[k] * (cxy + SURF_BIAS * v)
                wsum += n_cam[k]
                part_xy += xy[k, sl]
                part_n += n[k, sl]
            if wsum <= 0:
                continue
            self.pos[i] = est / wsum
            self.ztop[i] = float(ztop[i])
            self.last_seen[i] = t
            # heading candidates
            if part_n[2] > 8 and part_n[3] > 8:
                lat = part_xy[2] / part_n[2] - part_xy[3] / part_n[3]
                if np.linalg.norm(lat) > 0.03:
                    self.meas_legs[i] = wrap(math.atan2(lat[1], lat[0]) - math.pi / 2)
                else:
                    self.meas_legs[i] = np.nan
            else:
                self.meas_legs[i] = np.nan
            if part_n[1] > 8 and part_n[0] > 8:
                fw = part_xy[1] / part_n[1] - part_xy[0] / part_n[0]
                self.meas_head[i] = math.atan2(fw[1], fw[0]) if np.linalg.norm(fw) > 0.008 else np.nan
            else:
                self.meas_head[i] = np.nan
            h = self.hist[i]
            h.append((t, *self.pos[i]))
            while h and t - h[0][0] > 0.6:
                h.pop(0)
            self.meas_motion[i] = np.nan
            if len(h) >= 4 and cmd[i, 0] > 0.08:
                d = np.array(h[-1][1:]) - np.array(h[0][1:])
                if np.linalg.norm(d) / max(h[-1][0] - h[0][0], 1e-3) > 0.07:
                    self.meas_motion[i] = math.atan2(d[1], d[0])
            # fuse: legs are the main measurement, the walking direction helps while the duck walks straight
            for meas, gain in ((self.meas_legs[i], 0.35), (self.meas_motion[i], 0.25), (self.meas_head[i], 0.15)):
                if np.isfinite(meas):
                    if not self.yaw_ok[i]:
                        self.yaw[i], self.yaw_ok[i] = meas, True
                    else:
                        self.yaw[i] = wrap(self.yaw[i] + gain * wrap(meas - self.yaw[i]))
        return self.state(t)

    def state(self, t):
        return dict(t=t, pos=self.pos.copy(), yaw=self.yaw.copy(), ztop=self.ztop.copy(), npx=self.npx.copy(),
                    ncam=self.ncam.copy(), stale=(t - self.last_seen) > 0.25, yaw_ok=self.yaw_ok.copy(),
                    meas_legs=self.meas_legs.copy(), meas_head=self.meas_head.copy(), meas_motion=self.meas_motion.copy())


# ----------------------------------------------------------------------------- images for artefacts
SEM_COLORS = {"background": (20, 20, 20), "unlabelled": (0, 0, 0), "floor": (70, 74, 80), "wall": (150, 150, 160),
              "ramp": (226, 160, 60), "cone": (240, 90, 40), "planter": (120, 170, 80), "duck": (245, 245, 245),
              "duck_head": (118, 185, 0), "duck_leg_left": (28, 140, 252), "duck_leg_right": (252, 60, 120)}
DUCK_RGB = [(0.95, 0.25, 0.25), (0.98, 0.60, 0.15), (0.98, 0.88, 0.20), (0.45, 0.85, 0.30), (0.15, 0.70, 0.55),
            (0.25, 0.80, 0.95), (0.25, 0.45, 0.95), (0.55, 0.35, 0.90), (0.90, 0.35, 0.80), (0.98, 0.98, 0.98),
            (0.20, 0.20, 0.22), (0.65, 0.45, 0.30)]


def sem_image(cls_img):
    lut = np.array([SEM_COLORS[c] for c in CLASSES], np.uint8)
    return lut[np.clip(cls_img, 0, len(CLASSES) - 1)]


def inst_image(duck_img, cls_img):
    out = sem_image(cls_img) // 3
    lut = (np.array(DUCK_RGB) * 255).astype(np.uint8)
    m = duck_img >= 0
    out[m] = lut[duck_img[m] % 12]
    return out


def depth_image(depth, near=0.8, far=7.5):
    """near = white, far = dark (turbo-free grey ramp with a slight cool tint), invalid = black."""
    d = np.asarray(depth, np.float32)
    v = np.clip((far - d) / (far - near), 0, 1) ** 0.8
    img = np.stack([v * 235 + 10, v * 240 + 10, v * 245 + 10], -1)
    img[d <= 0] = 0
    return img.astype(np.uint8)


def map_image(m: PerceivedMap, scale=3, ducks=None, yaws=None, paths=None, slots=None):
    """Top view of the perceived map (+ optional duck poses, paths and slots), y up."""
    occ = m.occ
    img = np.zeros(occ.shape + (3,), np.uint8)
    img[occ == 0] = (24, 24, 24)
    img[occ == 1] = (58, 62, 68)
    img[occ == 2] = (210, 210, 220)
    hh = np.nan_to_num(m.height, nan=0.0)
    rv = np.clip(hh / 0.06, 0, 1)
    rm = occ == 3
    img[rm] = (np.stack([140 + 100 * rv, 95 + 75 * rv, 30 + 40 * rv], -1)[rm]).astype(np.uint8)
    img = np.flipud(np.transpose(img, (1, 0, 2)))           # x right, y up
    img = np.kron(img, np.ones((scale, scale, 1), np.uint8))
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    N = occ.shape[0] * scale

    def px(x, y):
        return ((x + GRID_HALF) / GRID_RES * scale, N - (y + GRID_HALF) / GRID_RES * scale)

    for r in m.ramps:
        c, yaw, L, Wd = np.array(r["c"]), r["yaw"], r["L"], r["W"]
        ax, nr = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        poly = [px(*(c + sa * ax * L / 2 + sb * nr * Wd / 2)) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        d.polygon(poly, outline=(255, 200, 90, 255))
    if paths:
        for pts in paths.values():
            q = [px(*p) for p in pts]
            if len(q) > 1:
                d.line(q, fill=(28, 140, 252, 255), width=3)
            for p in q[1:]:
                d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(255, 255, 255, 255), outline=(28, 140, 252, 255))
    if slots:
        for p in slots:
            q = px(*p)
            d.rectangle([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], outline=(118, 185, 0, 255), width=2)
    if ducks is not None:
        for i, p in enumerate(ducks):
            if not np.all(np.isfinite(p)):
                continue
            q = px(*p)
            col = tuple(int(255 * v) for v in DUCK_RGB[i]) + (255,)
            d.ellipse([q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9], fill=col, outline=(0, 0, 0, 255), width=2)
            if yaws is not None:
                a = yaws[i]
                d.line([q, (q[0] + 22 * math.cos(a), q[1] - 22 * math.sin(a))], fill=(255, 255, 255, 255), width=3)
    return np.asarray(im)
