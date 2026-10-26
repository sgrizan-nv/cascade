"""Navigator for the 12 ducks that only knows what the cameras perceived (numpy only).

Same steering, routing and choreography as isaac/director.py (the take-proven waypoint navigator), but every input
comes from perception.py:
- the yard (walls, cones, planters, ramps) is the PerceivedMap built from the camera depth before the ducks move;
- duck positions and headings are the Localizer estimates, refreshed at the perception rate;
- queue slots, the path over the north ramp and the exit spots are derived from the detected north ramp;
- "goal reached" and the order verification use perceived positions.
The only fixed coordinates are named meeting places an operator would mark on the map (the centre circle).
yard_layout.py is never imported here: it describes the simulated world, which the navigator must not know.
"""
import math

import numpy as np

NDUCK = 12
WZ_MAX = 0.5


class NavDirector:
    RING_C, RING_R = (1.0, -0.1), 0.65    # named place "centre circle" (map frame), 12 slots 34 cm apart

    def __init__(self, seed, hero=2):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hero = hero
        self.mode = "idle"
        self.nav = [dict(kind="pause", until=1e9, hold=True) for _ in range(NDUCK)]
        self.noavoid = set()
        self.vmax = self.rng.uniform(0.25, 0.4, NDUCK)
        self.target = np.zeros((NDUCK, 3))
        self.cmd = np.zeros((NDUCK, 3))
        self.fallen_since = np.full(NDUCK, -1.0)
        self.map = None
        self.RAMPS = []
        self.obstacles = []
        self.box = (-2.44, 2.44, -2.44, 2.44)
        self.POS = None
        self.YAW = None
        self.busy, self.lined, self.racers = set(), [], []
        self.goals = {}
        self.goal_kind = None
        self.goal_t = 0.0
        self.paths_log = []
        self.on_ramp_seen = set()
        self.headcam_slow = np.ones(NDUCK)
        self.goal_until = 0.0
        self.check_after = None
        self.progress = [None] * NDUCK      # (t, position, nav) of the last real move, for the stuck detector
        self.replans = 0
        self.ready = False

    def soft_reset(self):
        """Forget every order: no goals, holds, queues or speed overrides; the map stays."""
        self.mode = "idle"
        self.busy, self.lined, self.racers = set(), [], []
        self.noavoid = set()
        self.goals, self.goal_kind, self.goal_until, self.check_after = {}, None, 0.0, None
        self.vmax = np.random.default_rng(self.seed).uniform(0.25, 0.4, NDUCK)
        self.on_ramp_seen = set()
        self.nav = [dict(kind="pause", until=1e9, hold=True) for _ in range(NDUCK)]
        self.target[:] = 0
        self.cmd[:] = 0
        self.fallen_since[:] = -1.0
        self.progress = [None] * NDUCK

    # ------------------------------------------------------------------ world model from the cameras
    def set_map(self, m):
        self.map = m
        self.RAMPS = [(r["name"], r["c"][0], r["c"][1], r["yaw"], r["lu"], r["lt"], r["ld"], r["H"], r["W"])
                      for r in m.ramps]
        self.obstacles = m.obstacles()
        self.box = m.inner_box()
        n = m.ramp("north")
        if n is None:
            raise RuntimeError("the cameras did not find the north ramp")
        ax = np.array([math.cos(n["yaw"]), math.sin(n["yaw"])])
        if ax[0] < 0:
            ax = -ax
        c = np.array(n["c"])
        west_end, east_end = c - ax * n["L"] / 2, c + ax * n["L"] / 2
        xmax = self.box[1]
        corr_x = west_end[0] - 0.40                         # queue corridor 40 cm before the low west end
        self.STAGE = [(corr_x, west_end[1] - 0.45 - 0.42 * k) for k in range(5)]   # front slot first, 42 cm apart
        self.PATH = [(corr_x, west_end[1]), tuple(west_end), tuple(east_end),
                     (min(east_end[0] + 0.22, xmax - 0.32), east_end[1] - 0.07)]
        ex = xmax - 0.49
        self.EXIT = [(ex, y) for y in (-0.55, -0.15, 0.25, 0.65, 1.05)]
        self.ramp_north = n
        self.ring_clearance = self._ring_clearance()
        self.ready = True

    def _ring_clearance(self):
        """Smallest distance from a centre-circle slot to a perceived obstacle or ramp cell (m)."""
        from perception import GRID_HALF, GRID_RES
        occ = self.map.occ
        ix, iy = np.nonzero((occ == 2) | (occ == 3))
        pts = np.stack([(ix + 0.5) * GRID_RES - GRID_HALF, (iy + 0.5) * GRID_RES - GRID_HALF], -1)
        best = 9.0
        for s in self._ring_slots():
            best = min(best, float(np.min(np.linalg.norm(pts - s, axis=1))) if len(pts) else 9.0)
        return best

    def _ring_slots(self):
        c = np.array(self.RING_C)
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        return [c + self.RING_R * np.array([math.cos(a), math.sin(a)]) for a in ang]

    def _pos(self):
        return np.asarray(self.POS)[:, :2]

    def _clip(self, x, m=0.3):
        return np.array([np.clip(x[0], self.box[0] + m, self.box[1] - m), np.clip(x[1], self.box[2] + m, self.box[3] - m)])

    def keepout(self, p, extra=0.0):
        x, y = p
        b = self.box
        if x < b[0] + 0.25 + extra or x > b[1] - 0.25 - extra or y < b[2] + 0.25 + extra or y > b[3] - 0.25 - extra:
            return True
        for ox, oy, r in self.obstacles:
            if math.hypot(x - ox, y - oy) < r + 0.15 + extra:
                return True
        for name, cx, cy, yaw, lu, lt, ld, h, w in self.RAMPS:
            dx, dy = x - cx, y - cy
            lx = math.cos(yaw) * dx + math.sin(yaw) * dy
            ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if abs(lx) < (lu + lt + ld) / 2 + 0.1 + extra and abs(ly) < w / 2 + 0.1 + extra:
                return True
        return False

    def free_spot(self, others):
        b = self.box
        for _ in range(1000):
            p = np.array([self.rng.uniform(b[0] + 0.3, b[1] - 0.3), self.rng.uniform(b[2] + 0.3, b[3] - 0.3)])
            if not self.keepout(p) and all(np.hypot(*(p - o)) > 0.32 for o in others):
                return p
        return np.zeros(2)

    def _assign(self, ducks, slots):
        """Greedy nearest assignment of ducks to slots (longest trips first), no slot used twice."""
        P = self._pos()
        free = list(range(len(slots)))
        out = {}
        order = sorted(ducks, key=lambda d: -min(np.hypot(*(P[d] - np.array(slots[s]))) for s in free))
        for d in order:
            s = min(free, key=lambda s: np.hypot(*(P[d] - np.array(slots[s]))))
            out[d] = s
            free.remove(s)
        return out

    @staticmethod
    def _path_len(start, pts):
        q = [np.asarray(start, float)] + [np.asarray(x, float) for x in pts]
        return float(sum(np.linalg.norm(q[j + 1] - q[j]) for j in range(len(q) - 1)))

    def _free_ducks(self):
        return [i for i in range(NDUCK) if i not in self.busy]

    # ------------------------------------------------------------------ routing round the perceived ramps
    @staticmethod
    def _seg_hits(a, b, hl, hw):
        t0, t1 = 0.0, 1.0
        d = b - a
        for k, lim in ((0, hl), (1, hw)):
            for sg in (-1, 1):
                num, den = lim - sg * a[k], sg * d[k]
                if abs(den) < 1e-9:
                    if num < 0:
                        return False
                elif den > 0:
                    t1 = min(t1, num / den)
                else:
                    t0 = max(t0, num / den)
        return t0 < t1 - 1e-6

    def _ramp_boxes(self):
        out = []
        for name, cx, cy, yaw, lu, lt, ld, h, w in self.RAMPS:
            c, s_ = math.cos(yaw), math.sin(yaw)
            out.append((np.array([cx, cy]), np.array([[c, s_], [-s_, c]]), (lu + lt + ld) / 2, w / 2))
        return out

    def _route(self, p, q, infl=0.30, margin=0.15):
        """Shortest path from p to q clear of every perceived ramp footprint (visibility graph over the corners of the
        inflated footprints, Dijkstra). A duck that starts on or next to a ramp first gets clear of it."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        boxes = self._ramp_boxes()

        def inside(x, k):
            c, R, hl, hw = boxes[k]
            v = R @ (x - c)
            return abs(v[0]) < hl + margin and abs(v[1]) < hw + margin

        pre = []
        for k, (c, R, hl, hw) in enumerate(boxes):
            if inside(p, k) and not inside(q, k):
                v = R @ (p - c)
                if abs(v[1]) < hw:
                    # off the ramp along its slope, at the end that faces the destination, unless that end is
                    # against a wall (the clipped point would stay next to the ramp and the duck gets stuck there)
                    toward = math.copysign(1.0, (R @ (q - c))[0])
                    p2 = None
                    for sgn in (toward, -toward):
                        cand = R.T @ np.array([sgn * (hl + infl), v[1]]) + c
                        if np.linalg.norm(self._clip(cand) - cand) < 0.05:
                            p2 = cand
                            break
                    if p2 is None:
                        p2 = self._clip(R.T @ np.array([toward * (hl + infl), v[1]]) + c)
                else:
                    p2 = self._clip(R.T @ np.array([v[0], math.copysign(hw + infl, v[1])]) + c)
                pre.append(p2)
                p = p2
        skip_q = {k for k in range(len(boxes)) if inside(q, k)}

        def clear(a, b, end_q=False):
            for k, (c, R, hl, hw) in enumerate(boxes):
                if end_q and k in skip_q:
                    continue
                if self._seg_hits(R @ (a - c), R @ (b - c), hl + margin - 0.01, hw + margin - 0.01):
                    return False
            return True

        if clear(p, q, True):
            return pre + [q]
        nodes = [p, q]
        for c, R, hl, hw in boxes:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x = self._clip(R.T @ np.array([sx * (hl + infl), sy * (hw + infl)]) + c)
                    if not any(inside(x, k) for k in range(len(boxes))):
                        nodes.append(x)
        n = len(nodes)
        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        done = [False] * n
        for _ in range(n):
            u = min((d, i) for i, d in enumerate(dist) if not done[i])[1]
            if dist[u] == math.inf or u == 1:
                break
            done[u] = True
            for v in range(n):
                if done[v] or v == u:
                    continue
                if not clear(nodes[u], nodes[v], v == 1):
                    continue
                d = dist[u] + float(np.linalg.norm(nodes[v] - nodes[u]))
                if d < dist[v]:
                    dist[v], prev[v] = d, u
        if dist[1] == math.inf:
            return pre + [q]
        path, u = [], 1
        while u != 0:
            path.append(nodes[u])
            u = prev[u]
        return pre + path[::-1]

    # ------------------------------------------------------------------ tools (called by the agent)
    def _goals(self, t, kind, goals, tol):
        self.goal_kind, self.goal_t = kind, t
        self.goals = {int(d): (np.asarray(g, float), tol) for d, g in goals.items()}

    def _ring_assign(self, ducks, slots):
        """Ducks keep their angular order round the circle centre (no crossing paths); best rotation by distance."""
        P = self._pos()
        c = np.array(self.RING_C)
        da = sorted(ducks, key=lambda d: math.atan2(P[d][1] - c[1], P[d][0] - c[0]))
        sa = sorted(range(len(slots)), key=lambda s: math.atan2(slots[s][1] - c[1], slots[s][0] - c[0]))
        best = None
        for k in range(len(sa)):
            pick = [sa[(j * len(sa) // max(len(da), 1) + k) % len(sa)] for j in range(len(da))]
            cost = sum(float(np.linalg.norm(P[d] - slots[s])) for d, s in zip(da, pick))
            if best is None or cost < best[0]:
                best = (cost, dict(zip(da, pick)))
        return best[1]

    def agent_gather(self, t):
        slots = self._ring_slots()
        free = [d for d in range(NDUCK) if d not in self.busy]
        slot_of = self._ring_assign(free, slots)
        ev = [(t, "avoid", free)]
        P = self._pos()
        rank = {d: k for k, d in enumerate(sorted(free, key=lambda d: np.linalg.norm(P[d] - slots[slot_of[d]])))}
        longest = 0.0
        for duck, sl in slot_of.items():     # precise: stop on the slot (7 cm), so the circle reads as a circle
            path = self._route(P[duck], slots[sl])
            ev.append((t + 0.3 * rank[duck], "navstay", duck, path, {"precise": True}))
            longest = max(longest, 0.3 * rank[duck] + self._path_len(P[duck], path) / 0.22)
        self.home = {d: np.array(slots[sl]) for d, sl in slot_of.items()}
        self._goals(t, "circle", {d: slots[sl] for d, sl in slot_of.items()}, 0.12)
        self.check_after = float(np.clip(10.0 + longest, 22.0, 60.0))
        return ev, f"{len(free)} ducks to the nearest free slot on the 0.65 m centre circle (perceived positions)"

    def agent_line_up(self, t, n):
        n = max(1, min(n, len(self.STAGE)))
        P = self._pos()
        corr = np.mean(np.array(self.STAGE[:n]), axis=0)
        free = [d for d in self._free_ducks()]
        chosen = sorted(free, key=lambda d: np.linalg.norm(P[d] - corr))[:n]
        a = self._assign(chosen, self.STAGE[:n])
        self.lined = sorted(chosen, key=lambda d: a[d])         # front of the line first
        self.busy |= set(self.lined)
        ev = [(t, "navstay", d, self._route(P[d], self.STAGE[a[d]]), {"precise": True, "face": math.pi / 2})
              for d in self.lined]
        self._goals(t, "line_up", {d: self.STAGE[a[d]] for d in self.lined}, 0.10)
        return ev, (f"{n} ducks nearest the queue corridor to slots 42 cm apart, 40 cm before the low end of the "
                    f"detected north ramp")

    def agent_cross(self, t, ramp, form, n):
        ev = []
        if ramp != "north":
            note0 = f"only the north ramp has a mapped single-file route; using it instead of {ramp}; "
        else:
            note0 = ""
        if not self.lined or len(self.lined) < min(n, len(self.STAGE)):
            e2, _ = self.agent_line_up(t, n)
            ev += e2
            t = t + 9.0
            note = f"queue {len(self.lined)} ducks first (wait 9 s), then one path over the north ramp"
        else:
            note = f"queue of {len(self.lined)} ducks: one path over the detected north ramp, starts 1.0 s apart"
        for qi, d in enumerate(self.lined):
            self.vmax[d] = 0.36
            tq = t + 1.0 * qi
            ev.append((tq, "noavoid", [d]))
            ev.append((tq, "navstay", d, [np.array(p) for p in self.PATH] + [np.array(self.EXIT[qi])]))
        ev.append((t + 21.0, "avoid", list(self.lined)))
        self.on_ramp_seen = set()
        self._goals(t, "cross_ramp", {d: self.EXIT[qi] for qi, d in enumerate(self.lined)}, 0.25)
        return ev, note0 + note + ", 0.36 m/s command, exits at the east wall"

    def agent_stop(self, t):
        P = self._pos()
        c = np.array(self.ramp_north["c"])
        order = sorted(range(NDUCK), key=lambda d: np.hypot(*(P[d] - c)))
        self._goals(t, "stop", {d: P[d] for d in range(NDUCK)}, 0.25)
        return [(t + 0.2 * k, "hold", d) for k, d in enumerate(order)], "every duck holds still, 0.2 s apart, from the ramp outwards"

    def agent_scatter(self, t):
        ducks = self._free_ducks()
        spots = []
        rng = np.random.default_rng(int(t * 10))
        b = self.box
        for _ in range(4000):
            if len(spots) == len(ducks):
                break
            p = np.array([rng.uniform(b[0] + 0.45, b[1] - 0.45), rng.uniform(b[2] + 0.45, b[3] - 0.45)])
            if not self.keepout(p, 0.1) and all(np.hypot(*(p - o)) > 0.75 for o in spots):
                spots.append(p)
        a = self._assign(ducks[:len(spots)], spots)
        P = self._pos()
        routes = {d: self._route(P[d], spots[s]) for d, s in a.items()}
        ev = [(t, "avoid", ducks)] + [(t, "navstay", d, routes[d]) for d in a]
        self._goals(t, "scatter", {d: spots[s] for d, s in a.items()}, 0.25)
        self.check_after = float(np.clip(10.0 + max([self._path_len(P[d], r) / 0.22 for d, r in routes.items()] or [0]),
                                         16.0, 60.0))
        return ev, f"{len(ducks)} ducks to spots at least 75 cm apart, away from the mapped ramps and cones"

    # ------------------------------------------------------------------ booth choreography
    RAMP_LANES = {"north": 3, "west": 2, "south": 2}

    def _ramp_slots(self, name):
        """Standing spots on the flat top of a perceived ramp, one per lane, with the two ways in along the slope.
        Returns [(slot, [(approach, entry, facing yaw), ...])], outer lanes first."""
        r = self.map.ramp(name) if self.map is not None else None
        if r is None:
            return []
        ax = np.array([math.cos(r["yaw"]), math.sin(r["yaw"])])
        nr = np.array([-ax[1], ax[0]])
        c = np.array(r["c"])
        k = self.RAMP_LANES.get(name, 2)
        half = max(0.0, r["W"] / 2 - 0.14)
        lanes = sorted(([0.0] if k == 1 else list(np.linspace(-half, half, k))), key=lambda v: -abs(v))
        top = (r["lu"] - r["ld"]) / 2
        out = []
        for ly in lanes:
            slot = c + ax * top + nr * ly
            ways = []
            for e in (-1.0, 1.0):
                b = c + ax * e * (r["L"] / 2 - 0.04) + nr * ly
                for off in (0.30, 0.20, 0.14):          # straight run-up onto the slope; shorter near walls/cones
                    a = c + ax * e * (r["L"] / 2 + off) + nr * ly
                    inside = (self.box[0] + 0.2 < a[0] < self.box[1] - 0.2 and
                              self.box[2] + 0.2 < a[1] < self.box[3] - 0.2)
                    if inside and not any(math.hypot(a[0] - ox, a[1] - oy) < rr + 0.12 for ox, oy, rr in self.obstacles):
                        ways.append((a, b, math.atan2(-e * ax[1], -e * ax[0])))
                        break
            if ways:
                out.append((slot, ways))
        return out

    def agent_ramp_hold(self, t, ramp, n):
        names = [r["name"] for r in self.map.ramps if r["name"] in self.RAMP_LANES] if ramp == "all" else [ramp]
        spots = [(nm, s, w) for nm in names for s, w in self._ramp_slots(nm)]
        if not spots:
            spots = [("north", s, w) for s, w in self._ramp_slots("north")]
            names = ["north"]
        if ramp == "all":
            by = [[x for x in spots if x[0] == nm] for nm in names]
            spots = [lst[k] for k in range(max(map(len, by), default=0)) for lst in by if k < len(lst)]
        n = len(spots) if n <= 0 else max(1, min(n, len(spots)))
        spots = spots[:n]
        P = self._pos()
        pts = [s for _, s, _ in spots]
        free = self._free_ducks()
        chosen = sorted(free, key=lambda d: min(np.linalg.norm(P[d] - q) for q in pts))[:len(spots)]
        a = self._assign(chosen, pts)
        ev = [(t, "avoid", chosen)]
        per_ramp = {}
        longest = 0.0
        for d in sorted(chosen, key=lambda d: a[d]):
            nm, slot, ways = spots[a[d]]
            routes = [(self._route(P[d], w[0]) + [w[1], slot], w) for w in ways]

            def length(pts, start=P[d]):
                q = [np.asarray(start)] + [np.asarray(x) for x in pts]
                return float(sum(np.linalg.norm(q[j + 1] - q[j]) for j in range(len(q) - 1)))
            path, way = min(routes, key=lambda pw: length(pw[0]))
            k = per_ramp.get(nm, 0)
            per_ramp[nm] = k + 1
            longest = max(longest, 1.6 * k + length(path) / 0.16)
            ev.append((t + 1.6 * k, "navstay", d, path, {"precise": True, "face": way[2], "pmin": 0.6, "calm": 0.6}))
        self._goals(t, "ramp_hold", {d: spots[a[d]][1] for d in chosen}, 0.2)
        self.check_after = min(60.0, 12.0 + longest)
        where = "every ramp" if ramp == "all" else f"the {names[0]} ramp"
        return ev, (f"{len(chosen)} ducks nearest {where} walk up its slope in parallel lanes and stand still on the "
                    f"flat top (perceived ramp geometry); the others wait where they are")

    def agent_circle_walk(self, t, direction, laps):
        laps = int(np.clip(laps, 1, 3))
        slots = self._ring_slots()
        ducks = list(range(NDUCK))
        slot_of = self._ring_assign(ducks, slots)
        c = np.array(self.RING_C)
        w = (-1.0 if direction == "clockwise" else 1.0) * 0.22 / self.RING_R
        t0 = t + 7.0
        t_end = t0 + laps * 2 * math.pi / abs(w)
        ev = [(t, "avoid", ducks)]
        P = self._pos()
        for d, s in slot_of.items():
            th0 = math.atan2(slots[s][1] - c[1], slots[s][0] - c[0])
            # routed approach (off a ramp top, round the ramps), then the moving point on the circle
            pts = self._route(P[d], slots[s])[:-1]
            ev.append((t, "orbit", d, dict(c=c, R=self.RING_R, th0=th0, w=w, t0=t0, t_end=t_end, pts=pts)))
        self._goals(t, "circle_walk", {d: c for d in ducks}, 0.3)
        self.goal_until = t_end
        self.check_after = (t_end - t) + 5.0
        return ev, (f"12 ducks onto the 0.65 m centre circle, then {laps} lap(s) {direction} at 0.22 m/s "
                    f"(about {t_end - t:.0f} s), then they stop on the circle")

    def agent_spin(self, t, direction, turns):
        turns = float(np.clip(turns, 1, 2))
        wz = (-1.0 if direction == "clockwise" else 1.0) * WZ_MAX
        t_end = t + turns * 2 * math.pi / WZ_MAX
        ev = [(t, "spin", d, dict(wz=wz, t_end=t_end)) for d in range(NDUCK)]
        self._goals(t, "spin", {d: self._pos()[d].copy() for d in range(NDUCK)}, 0.3)
        self.goal_until = t_end
        self.check_after = (t_end - t) + 3.0
        return ev, f"every duck turns {turns:g} time(s) {direction} on the spot at {WZ_MAX} rad/s, then stands still"

    def agent_wander(self, t):
        self.busy, self.lined = set(), []
        ev = [(t, "mode", "wander"), (t, "avoid", list(range(NDUCK)))]
        ev += [(t, "pause_until", i, 0.0) for i in range(NDUCK)]
        self.goals, self.goal_kind = {}, "wander"
        return ev

    # ------------------------------------------------------------------ verification from perception
    def verify(self, t, perc):
        """Per goal duck: perceived distance to its goal, standing still, upright (from the camera height)."""
        if not self.goals:
            return None
        P = self._pos()
        speed = perc.get("speed", np.zeros(NDUCK))
        ztop = perc.get("ztop", np.full(NDUCK, 0.25))
        rows, ok_n = [], 0
        r = self.ramp_north
        for d, (g, tol) in sorted(self.goals.items()):
            dist = float(np.linalg.norm(P[d] - g))
            still = bool(speed[d] < 0.04)
            up = bool(ztop[d] > 0.15)
            ok = dist < tol and still and up
            if self.goal_kind == "stop":           # stopped = the cameras see it standing still, upright
                ok = still and up
            if self.goal_kind == "circle_walk":    # on the circle (radius error), stopped after the laps
                dist = abs(float(np.linalg.norm(P[d] - g)) - self.RING_R)
                ok = dist < tol and still and up and t >= self.goal_until
            if self.goal_kind == "spin":           # turned for the whole time, upright, back to standing
                ok = up and still and t >= self.goal_until
            if self.goal_kind == "cross_ramp":     # seen on the ramp, now past its far end (or at its exit spot)
                lx = math.cos(r["yaw"]) * (P[d][0] - r["c"][0]) + math.sin(r["yaw"]) * (P[d][1] - r["c"][1])
                ok = d in self.on_ramp_seen and up and (lx > r["L"] / 2 - 0.05 or dist < tol)
            ok = bool(ok)
            ok_n += int(ok)
            rows.append(dict(duck=int(d) + 1, goal=[round(float(g[0]), 2), round(float(g[1]), 2)], dist_m=round(dist, 3),
                             still=still, upright=up, ok=bool(ok),
                             **({"crossed_ramp": d in self.on_ramp_seen} if self.goal_kind == "cross_ramp" else {})))
        return dict(t=round(float(t), 2), kind=self.goal_kind, ok=int(ok_n), n=len(rows), ducks=rows)

    # ------------------------------------------------------------------ events and steering (as isaac/director.py)
    def apply(self, e, t):
        op = e[1]
        if op in ("nav", "navstay"):
            opts = e[4] if len(e) > 4 else {}
            pts = [np.array(p, float) for p in e[3]]
            self.nav[e[2]] = dict(kind="path", pts=pts, until=t + 60.0, stay=op == "navstay",
                                  precise=opts.get("precise", False), face=opts.get("face"),
                                  pmin=opts.get("pmin", 0.3), calm=opts.get("calm", 0.0))
            start = self._pos()[e[2]] if self.POS is not None else pts[0]
            self.paths_log.append(dict(t=round(t, 2), duck=int(e[2]) + 1,
                                       pts=[[round(float(start[0]), 3), round(float(start[1]), 3)]] +
                                           [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts]))
        elif op == "hold":
            self.nav[e[2]] = dict(kind="pause", until=t + 1e9, hold=True)
        elif op == "noavoid":
            self.noavoid |= set(e[2])
        elif op == "avoid":
            self.noavoid -= set(e[2])
        elif op == "mode":
            self.mode = e[2]
            if self.mode in ("wander",):
                for i in range(NDUCK):
                    if self.nav[i].get("kind") == "pause":
                        self.nav[i] = dict(kind="pause", until=0)
        elif op in ("orbit", "spin"):
            self.nav[e[2]] = dict(kind=op, until=1e12, **e[3])
        elif op == "push":
            return e[2], e[3]
        elif op == "pause_until":
            self.nav[e[2]] = dict(kind="pause", until=e[3])
        return None

    def new_plan(self, i, pos, t):
        if self.mode != "wander":
            self.nav[i] = dict(kind="pause", until=t + 1e9, hold=True)
            return
        r = self.rng.random()
        if r < 0.5:
            self.nav[i] = dict(kind="pause", until=t + self.rng.uniform(1.5, 3.5))
        else:
            self.nav[i] = dict(kind="path", pts=self._route(pos, self.free_spot([])), until=t + 25.0)

    def update_falls(self, t, upz):
        """Upright from each duck's own IMU (on-board proprioception, the same signal the policy reads)."""
        upright = upz > math.cos(math.radians(55))
        for i in range(NDUCK):
            if not upright[i]:
                if self.fallen_since[i] < 0:
                    self.fallen_since[i] = t
            elif self.fallen_since[i] >= 0 and upz[i] > 0.9:
                self.fallen_since[i] = -1.0
                if not self.nav[i].get("hold") and self.nav[i].get("kind") not in ("path", "orbit", "spin"):
                    self.nav[i] = dict(kind="pause", until=t + 1.0)
        return upright

    def steer(self, t, pos, yaw, upright):
        """pos, yaw: perceived. Only map-derived geometry below."""
        b = self.box
        for i in range(NDUCK):
            if self.mode == "stop" or not upright[i] or not np.all(np.isfinite(pos[i])):
                self.target[i] = 0
                continue
            p = pos[i]
            if self.goal_kind == "cross_ramp" and i in self.goals and self.ramp_north is not None:
                r = self.ramp_north
                c_, s_ = math.cos(r["yaw"]), math.sin(r["yaw"])
                lx = c_ * (p[0] - r["c"][0]) + s_ * (p[1] - r["c"][1])
                ly = -s_ * (p[0] - r["c"][0]) + c_ * (p[1] - r["c"][1])
                if abs(lx) < r["L"] / 2 - 0.15 and abs(ly) < r["W"] / 2:
                    self.on_ramp_seen.add(i)
            nv = self.nav[i]
            if t > nv.get("until", 0):
                self.new_plan(i, p, t)
                nv = self.nav[i]
            if nv["kind"] == "pause":
                self.target[i] = 0
                continue
            if nv["kind"] == "face":
                err = (nv["yaw"] - yaw[i] + math.pi) % (2 * math.pi) - math.pi
                self.target[i] = [0.0, 0.0, float(np.clip(1.4 * err, -WZ_MAX, WZ_MAX))] if abs(err) > 0.12 else 0
                continue
            if nv["kind"] in ("orbit", "spin") and t >= nv["t_end"]:
                self.nav[i] = dict(kind="pause", until=t + 1e9, hold=True)
                self.target[i] = 0
                continue
            if nv["kind"] == "spin":
                self.target[i] = [0.0, 0.0, float(nv["wz"])]
                continue
            orbit = nv["kind"] == "orbit"
            if orbit:
                while nv.get("pts") and np.hypot(*(np.asarray(nv["pts"][0]) - p)) < 0.22:
                    nv["pts"].pop(0)
                th = nv["th0"] + nv["w"] * max(0.0, t - nv["t0"])
                if t >= nv["t0"]:
                    th += math.copysign(0.45, nv["w"])
                goal = nv["c"] + nv["R"] * np.array([math.cos(th), math.sin(th)])
                gap = float(np.hypot(*(goal - p)))
                if nv.get("pts"):                    # still on its way to the circle
                    goal = np.asarray(nv["pts"][0], float)
                    gap = 1.0
                elif t < nv["t0"] and gap < 0.08:    # on its start spot: wait for the others
                    self.target[i] = 0
                    continue
            else:
                goal = nv["pts"][0]
            self._unstick(i, t, p, goal, nv, orbit)
            last = orbit or len(nv["pts"]) == 1
            if not orbit and np.hypot(*(goal - p)) < (0.07 if (nv.get("precise") and last) else 0.22):
                nv["pts"].pop(0)
                if not nv["pts"] and nv.get("stay"):
                    if nv.get("face") is not None:
                        self.nav[i] = dict(kind="face", yaw=float(nv["face"]), until=t + 1e9, hold=True)
                    else:
                        self.nav[i] = dict(kind="pause", until=t + 1e9, hold=True)
                    self.target[i] = 0
                    continue
                if not nv["pts"]:
                    self.new_plan(i, p, t)
                    self.target[i] = 0
                    continue
                goal = nv["pts"][0]
            dvec = goal - p
            dvec = dvec / (np.linalg.norm(dvec) + 1e-9)
            rep = np.zeros(2)
            for name, cx, cy, ryaw, lu, lt, ld, rh, rw in self.RAMPS:   # ramp side edges, for every duck
                c_, s_ = math.cos(ryaw), math.sin(ryaw)
                lx = c_ * (p[0] - cx) + s_ * (p[1] - cy)
                ly = -s_ * (p[0] - cx) + c_ * (p[1] - cy)
                if abs(lx) < (lu + lt + ld) / 2 - 0.05:
                    dist = abs(ly) - rw / 2
                    reach = 0.22 if dist > 0 else 0.10
                    if abs(dist) < reach:
                        sgn = math.copysign(1.0, ly) * (1.0 if dist > 0 else -1.0)
                        rep += sgn * 1.6 * (reach - abs(dist)) / reach * np.array([-s_, c_])
            calm = (not orbit and nv.get("calm") and nv["pts"] and
                    np.hypot(*(np.asarray(nv["pts"][-1]) - p)) < nv["calm"])
            if i not in self.noavoid:
                for ox, oy, r in self.obstacles:
                    v = p - np.array([ox, oy])
                    dist = np.linalg.norm(v)
                    if dist < r + 0.3:
                        rep += v / (dist + 1e-9) * 1.5 * (r + 0.3 - dist) / 0.3
                for j in range(NDUCK):
                    if j != i and not calm and np.all(np.isfinite(pos[j])):
                        v = p - pos[j]
                        dist = np.linalg.norm(v)
                        if dist < 0.32:
                            rep += v / (dist + 1e-9) * 1.2 * (0.32 - dist) / 0.32
                for k, lo, hi in ((0, b[0], b[1]), (1, b[2], b[3])):
                    for s, wall in ((-1, lo), (1, hi)):
                        dw = s * (wall - p[k])
                        if dw < 0.45:
                            e = np.zeros(2)
                            e[k] = -s
                            rep += e * 2.0 * (0.45 - dw) / 0.45
            dvec = dvec + rep
            des = math.atan2(dvec[1], dvec[0])
            err = (des - yaw[i] + math.pi) % (2 * math.pi) - math.pi
            wz = float(np.clip(1.6 * err, -WZ_MAX, WZ_MAX))
            vx = self.vmax[i] * max(0.0, math.cos(err)) ** 2
            if orbit:
                vx *= float(np.clip(gap / 0.30, 0.0 if t >= nv["t0"] else 0.3, 1.0))
            elif nv.get("precise") and len(nv["pts"]) == 1:
                vx *= float(np.clip(np.hypot(*(nv["pts"][0] - p)) / 0.35, nv.get("pmin", 0.3), 1.0))
            if abs(err) > 1.2:
                vx = 0.0
            vx *= float(self.headcam_slow[i])
            self.target[i] = [vx, 0.0, wz]

    def _unstick(self, i, t, p, goal, nv, orbit):
        """A duck that should walk but has not moved 8 cm in 6 s is caught between repulsions (cone, ramp edge,
        other ducks): side step 30 cm and route again from there."""
        pr = self.progress[i]
        if pr is None or pr[2] is not nv or np.hypot(*(p - pr[1])) > 0.08:
            self.progress[i] = (t, p.copy(), nv)
            return
        far = np.hypot(*(np.asarray(goal) - p)) > 0.3
        if t - pr[0] < 6.0 or not far:
            return
        self.progress[i] = (t, p.copy(), nv)
        self.replans += 1
        d = np.asarray(goal, float) - p
        d = d / (np.linalg.norm(d) + 1e-9)
        side = np.array([-d[1], d[0]]) * (1.0 if self.rng.random() < 0.5 else -1.0)
        step = self._clip(p + 0.30 * side)
        if self.keepout(step):
            step = self._clip(p - 0.30 * side)
        if orbit:
            nv["pts"] = [step] + list(nv.get("pts") or [])
        else:
            nv["pts"] = [step] + self._route(step, nv["pts"][-1])

    def smooth(self):
        self.cmd += 0.2 * (self.target - self.cmd)
        return self.cmd
