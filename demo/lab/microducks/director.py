"""Scripted velocity commands for the 12 ducks (simulator-agnostic, numpy only).

The policy only receives (vx, vy, wz); everything here is a scripted operator: wander / waypoint paths with obstacle,
wall and duck avoidance, ramp crossings, holds, and timed pushes (applied by the simulator).
"""
import math

import numpy as np

from yard_layout import CONES, NDUCK, PLANTERS, RAMPS, WALL_IN, ramp_endpoints

WZ_MAX = 0.5


def crossing(name, reverse=False, margin=0.4):
    r = next(r for r in RAMPS if r[0] == name)
    a, b = ramp_endpoints(r, margin)
    return [b, a] if reverse else [a, b]


class Director:
    def __init__(self, seed, hero):
        self.rng = np.random.default_rng(seed)
        self.hero = hero
        self.A, self.B = (hero + 4) % NDUCK, (hero + 7) % NDUCK
        self.mode = "wander"
        self.nav = [dict(kind="pause", until=0.5) for _ in range(NDUCK)]
        self.noavoid = set()
        self.vmax = self.rng.uniform(0.25, 0.4, NDUCK)
        self.target = np.zeros((NDUCK, 3))
        self.cmd = np.zeros((NDUCK, 3))
        self.obstacles = [(x, y, 0.09) for x, y in CONES] + [(x, y, 0.2) for x, y, _ in PLANTERS]
        self.fallen_since = np.full(NDUCK, -1.0)

    # -- placement helpers
    def keepout(self, p, extra=0.0):
        x, y = p
        if abs(x) > WALL_IN - 0.25 - extra or abs(y) > WALL_IN - 0.25 - extra:
            return True
        for ox, oy, r in self.obstacles:
            if math.hypot(x - ox, y - oy) < r + 0.15 + extra:
                return True
        for r in RAMPS:
            name, cx, cy, yaw, lu, lt, ld, h, w = r
            dx, dy = x - cx, y - cy
            lx = math.cos(yaw) * dx + math.sin(yaw) * dy
            ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if abs(lx) < (lu + lt + ld) / 2 + 0.1 + extra and abs(ly) < w / 2 + 0.1 + extra:
                return True
        return False

    def free_spot(self, others):
        for _ in range(1000):
            p = self.rng.uniform(-WALL_IN + 0.3, WALL_IN - 0.3, 2)
            if not self.keepout(p) and all(np.hypot(*(p - o)) > 0.32 for o in others):
                return p
        return np.zeros(2)

    def schedule(self, kind="a"):
        """Initial placements {duck: (xy, yaw)} and timed events [(t, op, ...)]."""
        h, A, B = self.hero, self.A, self.B
        intro = 0
        G = [4, 5, 8, 11]
        place = {h: (crossing("ramp_north")[0] + np.array([-0.35, 0.0]), 0.0),
                 A: (np.array([-1.5, -0.05]), 0.0), B: (np.array([0.75, -0.05]), math.pi),
                 intro: (np.array([-0.45, 2.02]), 0.0)}
        others = [p for p, _ in place.values()]
        for i in range(NDUCK):
            if i not in place:
                p = self.free_spot(others)
                others.append(p)
                place[i] = (p, float(self.rng.uniform(-math.pi, math.pi)))
        ev = [(0.0, "nav", h, crossing("ramp_north") + [np.array([1.9, 0.35])] + crossing("ramp_south", reverse=True)),
              (0.0, "hold", A), (0.0, "hold", B),
              (9.0, "noavoid", [A, B]), (9.0, "nav", A, [np.array([1.6, -0.05])]),
              (9.0, "nav", B, [np.array([-1.7, -0.05])]), (17.0, "avoid", [A, B]),
              (24.0, "push", h, 1.0), (31.0, "push", A, 1.0), (31.5, "push", B, 1.0),
              (0.0, "noavoid", [intro]), (0.0, "nav", intro, [np.array([1.3, 2.02]), np.array([1.9, 0.6])]),
              (7.0, "avoid", [intro]),
              (33.0, "nav", G[0], [crossing("ramp_west")[0] + np.array([0, 0.18]), crossing("ramp_west")[1] + np.array([0, 0.18])]),
              (33.0, "nav", G[1], [crossing("ramp_west")[0] + np.array([0, -0.18]), crossing("ramp_west")[1] + np.array([0, -0.18])]),
              (33.0, "nav", G[2], [crossing("ramp_west")[0] + np.array([-0.3, 0.0]), crossing("ramp_west")[1]]),
              (33.0, "nav", G[3], [crossing("ramp_west")[0] + np.array([-0.55, 0.1]), crossing("ramp_west")[1] + np.array([0, 0.1])]),
              (38.0, "mode", "ramps"), (38.0, "nav", h, crossing("ramp_west") + crossing("ramp_west", reverse=True))]
        if kind == "b":
            ev = [e for e in ev if e[1] != "push"] + [(22.0, "push", h, 1.1), (30.0, "push", A, 1.1),
                                                     (30.5, "push", B, 1.1)]
        others = [i for i in range(NDUCK) if i not in (h, A, B, intro)]
        if kind == "gather":  # everyone (but the hero) walks to a ring in the centre circle at 18 s, released at 26 s
            ev = [e for e in ev if not (e[1] == "mode" or (e[1] == "nav" and e[0] >= 33.0))]
            ring = [np.array([0.6 + 0.42 * math.cos(2 * math.pi * k / len(others)),
                              -0.2 + 0.42 * math.sin(2 * math.pi * k / len(others))]) for k in range(len(others))]
            ev += [(18.0, "nav", i, [ring[k]]) for k, i in enumerate(others)]
            ev += [(26.5, "hold", i) for i in others] + [(29.5, "mode", "wander")]
            ev += [(29.5, "nav", i, [self.free_spot([])]) for i in others]
        elif kind == "lineup":  # line up along y = -1.05 at 18 s, hold, then all cross the ramps
            ev = [e for e in ev if not (e[1] == "mode" or (e[1] == "nav" and e[0] >= 33.0))]
            xs = np.linspace(-1.45, 1.45, len(others))
            ev += [(18.0, "nav", i, [np.array([xs[k], -1.05])]) for k, i in enumerate(others)]
            ev += [(27.0, "hold", i) for i in others] + [(30.0, "mode", "ramps")]
        elif kind == "stop":  # everyone freezes at 19 s for 4 s, then the ramps
            ev = [e for e in ev if e[1] != "mode"] + [(19.0, "mode", "stop"), (23.0, "mode", "wander"),
                                                     (36.0, "mode", "ramps")]
        elif kind == "parade":  # two groups cross the west and the north ramps one after another
            ev = [e for e in ev if not (e[1] == "nav" and e[0] >= 33.0)]
            g1, g2 = others[:4], others[4:]
            for k, i in enumerate(g1):
                off = np.array([0.0, -0.21 + 0.14 * k])
                ev.append((16.0 + 0.6 * k, "nav", i, [crossing("ramp_west")[0] + off, crossing("ramp_west")[1] + off]))
            for k, i in enumerate(g2):
                off = np.array([0.0, -0.24 + 0.16 * k])
                ev.append((30.0 + 0.6 * k, "nav", i, [crossing("ramp_north")[0] + off, crossing("ramp_north")[1] + off]))
        if kind == "story":
            ev, place = self.story(place)
        if kind == "race":
            ev, place = self.race(place)
        if kind == "conga":
            ev, place = self.conga(place)
        if kind.startswith("agentic"):
            ev, place = self.agentic(place, kind)
        return place, sorted(ev, key=lambda e: e[0])

    # ------------------------------------------------------------------ story (Asier 18:24)
    QUEUE = [4, 5, 2, 8, 11]                 # the line, front to back; hero (index 2) is third
    LOOSE = [0, 6, 9, 10, 3, 7, 1]           # 6 and 9 collide head-on later; 0 is the intro duck
    T_GATHER, T_RAMPS, T_GO = 4.0, 19.0, 28.0
    STAGE = [(-0.10, 1.00), (-0.10, 0.58), (-0.10, 0.16), (-0.10, -0.26), (-0.10, -0.68)]
    PATH = [(-0.10, 1.45), (0.3, 1.45), (1.9, 1.45), (2.12, 1.38), (1.95, 0.45)]

    RACERS = {2: [(-0.30, 1.30), (2.25, 1.30)], 8: [(-0.30, 1.62), (2.25, 1.62)],
              4: [(0.40, 0.62), (-2.10, 0.62)], 5: [(0.40, 0.90), (-2.10, 0.90)]}

    # ------------------------------------------------------------------ agentic: orders are typed, Qwen picks the tool
    EXIT = [(1.95, -0.55), (1.95, -0.15), (1.95, 0.25), (1.95, 0.65), (1.95, 1.05)]   # front duck walks furthest
    ORDERS = [(4.0, "Everyone, gather in the centre circle."), (18.0, "Five of you, line up."),
              (27.0, "Now walk over the north ramp in single file."), (43.0, "Stress test: give duck 3 a strong push."),
              (47.5, "Everyone, stop!"), (51.5, "Okay, back to wandering around.")]

    # operator scripts per storyline: only WHEN an order is typed and its exact text
    SCRIPTS = {
        "agentic": ORDERS,
        "agentic_regroup": [(4.0, "Everybody, go for a walk around the yard."), (15.0, "Stop!"),
                            (19.0, "Regroup in the square."), (35.0, "Stress test: give duck 9 a light push."),
                            (39.0, "Scatter!"), (50.0, "Okay, everyone to the centre circle.")],
        "agentic_race": [(4.0, "Two racers to the west ramp start line."),
                         (8.0, "Everyone else, wait in the square."), (19.0, "Ready... go!"),
                         (33.0, "Stress test: push duck 6, strong."), (38.0, "Back to wandering, everyone.")],
        "agentic_conga": [(4.0, "Follow duck 5 around the cones."), (39.0, "Stress test: give duck 12 a light push."),
                          (44.0, "Okay, everyone regroup in the square.")],
        "agentic_carousel": [(4.0, "Everyone, gather in the centre circle."),
                             (17.0, "Walk around the circle, clockwise."),
                             (34.0, "Stress test: give duck 7 a light push."),
                             (38.0, "Now the other way round!"), (55.0, "Stop!")],
        "agentic_shapes": [(4.0, "Form a triangle in the middle of the yard."),
                           (19.0, "Now one long line across the yard."),
                           (34.0, "Make a square."), (48.0, "Stress test: give duck 4 a light push."),
                           (52.0, "Scatter!")],
        "agentic_parade": [(4.0, "Follow duck 9 over the north ramp."),
                           (20.0, "Everyone else, make a triangle."),
                           (42.0, "Stress test: give duck 9 a light push."),
                           (46.0, "Okay, everyone back to the centre circle.")],
        "agentic_medley": [(4.0, "Everyone, gather in the centre circle."),
                           (16.0, "Walk half a lap around the circle, counterclockwise."),
                           (27.0, "Now make a square."), (40.0, "Stress test: give duck 11 a strong push."),
                           (44.0, "Scatter!"), (54.0, "Stop!")],
        "agentic_race_north": [(4.0, "Two racers to the north ramp start line."),
                               (8.0, "Everyone else, one long line across the yard."), (21.0, "Ready... go!"),
                               (34.0, "Stress test: give duck 10 a strong push."), (39.0, "Back to wandering, everyone.")],
        "agentic_split_bs": [(3.0, "Everyone, gather in the centre circle."),
                             (16.0, "Left group take the west ramp, right group the north ramp.")],
        "agentic_race_gold": [(3.0, "Two racers to the north ramp start line."),
                              (7.0, "Everyone else, wait in the square."), (22.0, "Ready... go!")],
        "agentic_carousel_dark": [(3.0, "Everyone, gather in the centre circle."),
                                  (16.0, "Walk around the circle, clockwise."), (31.0, "Now the other way round!")],
        "agentic_shapes_dark": [(3.0, "Form a triangle in the middle of the yard."),
                                (18.0, "Now one long line across the yard."), (33.0, "Make a square.")],
        "agentic_slalom": [(3.0, "Green duck, lead four others around the cones.")],
        "agentic_dusk": [(1.0, "Everyone, wander around the yard."), (8.0, "Freeze!"),
                         (12.0, "Regroup in a tidy square."), (34.0, "Now scatter!")],
        "agentic_one": [(3.0, "Cyan duck, go to the top of the north ramp."),
                        (19.0, "Now everyone else, gather in the centre circle."),
                        (34.0, "Cyan duck, come back and join them.")],
        "agentic_keynote": [(3.0, "Everyone, gather in the centre circle."), (14.0, "Five of you, line up."),
                            (26.0, "Now walk over the north ramp in single file."), (43.0, "Stop!"),
                            (46.0, "Regroup in the centre circle.")],
        "agentic_split": [(4.0, "Everyone, gather in the centre circle."),
                          (17.0, "Left group take the west ramp, right group the north ramp."),
                          (56.0, "Stress test: push duck 2, strong."), (60.0, "Stop!")],
    }

    def agentic(self, place, kind="agentic"):
        place = {i: (np.array(p), float(self.rng.uniform(-math.pi, math.pi)) if i else 0.0) for i, p in self.START.items()}
        self.lined, self.racers, self.busy = [], [], set()
        ev = [(0.0, "noavoid", [0]), (0.0, "nav", 0, [np.array([0.2, 2.02])]), (3.5, "avoid", [0])]
        ev += [(0.0, "hold", i) for i in range(1, 12)]
        ev += [(t, "order", text) for t, text in self.SCRIPTS[kind]]
        return ev, place

    # live positions, set by the simulator before each order (navigator plans from where the ducks really are)
    POS = None

    def _pos(self):
        if self.POS is not None:
            return np.asarray(self.POS)[:, :2]
        return np.array([self.START[i] for i in range(12)], float)

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

    def _free_ducks(self):
        return [i for i in range(12) if i not in self.busy]

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

    def _ramp_boxes(self, infl):
        out = []
        for name, cx, cy, yaw, lu, lt, ld, h, w in RAMPS:
            c, s_ = math.cos(yaw), math.sin(yaw)
            out.append((np.array([cx, cy]), np.array([[c, s_], [-s_, c]]), (lu + lt + ld) / 2, w / 2))
        return out

    def _route(self, p, q, infl=0.30, margin=0.15):
        """Shortest path from p to q that keeps clear of every ramp footprint (visibility graph over the corners of the
        inflated footprints, Dijkstra). A duck that starts on or next to a ramp first gets clear of it."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        boxes = self._ramp_boxes(infl)

        def inside(x, k):
            c, R, hl, hw = boxes[k]
            v = R @ (x - c)
            return abs(v[0]) < hl + margin and abs(v[1]) < hw + margin

        pre = []
        for k, (c, R, hl, hw) in enumerate(boxes):
            if inside(p, k) and not inside(q, k):
                v = R @ (p - c)
                esc = np.array([math.copysign(hl + infl, v[0]), v[1]]) if abs(v[1]) < hw else \
                    np.array([v[0], math.copysign(hw + infl, v[1])])
                p2 = np.clip(R.T @ esc + c, -WALL_IN + 0.3, WALL_IN - 0.3)
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
                    x = np.clip(R.T @ np.array([sx * (hl + infl), sy * (hw + infl)]) + c, -WALL_IN + 0.3, WALL_IN - 0.3)
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

    SQUARE = [(x, y) for y in (-0.72, -1.05, -1.38, -1.71) for x in (-1.85, -1.45, -1.05)]  # 55 cm clear of ramp_south

    def agent_regroup(self, t, place):
        if place == "circle":
            return self.agent_gather(t), "all ducks to slots on a 0.65 m ring around the centre circle"
        ducks = self._free_ducks()
        a = self._assign(ducks, self.SQUARE)
        P = self._pos()
        ev = [(t, "avoid", ducks)] + [(t, "navstay", d, self._route(P[d], self.SQUARE[s])) for d, s in a.items()]
        return ev, f"{len(ducks)} ducks to a 3 x 4 grid of slots in the square (40 x 33 cm), nearest free slot each"

    def agent_scatter(self, t):
        ducks = self._free_ducks()
        spots = []
        rng = np.random.default_rng(int(t * 10))
        for _ in range(4000):
            if len(spots) == len(ducks):
                break
            p = rng.uniform(-WALL_IN + 0.45, WALL_IN - 0.45, 2)
            if not self.keepout(p, 0.1) and all(np.hypot(*(p - o)) > 0.75 for o in spots):
                spots.append(p)
        a = self._assign(ducks[:len(spots)], spots)
        P = self._pos()
        ev = [(t, "avoid", ducks)] + [(t, "navstay", d, self._route(P[d], spots[s])) for d, s in a.items()]
        return ev, f"{len(ducks)} ducks to spots at least 75 cm apart, away from ramps and cones"

    LANES = {"north": [((-0.20, 1.30), (2.0, 1.30)), ((-0.20, 1.60), (2.0, 1.60))],
             "west": [((-0.02, 0.62), (-2.10, 0.62)), ((-0.02, 0.92), (-2.10, 0.92))]}

    def agent_race_start(self, t, ramp, n):
        lanes = self.LANES.get(ramp, self.LANES["west"])
        n = max(1, min(n, len(lanes)))
        P = self._pos()
        cand = sorted(self._free_ducks(), key=lambda d: np.hypot(*(P[d] - np.array(lanes[0][0]))))
        cand = [d for d in cand if d != 0] or cand
        self.racers = [(d, lanes[k]) for k, d in enumerate(cand[:n])]
        self.busy |= {d for d, _ in self.racers}
        ev = [(t, "avoid", [d]) for d, _ in self.racers]
        for d, ln in self.racers:
            ax = np.array(ln[1]) - np.array(ln[0])
            ax /= np.linalg.norm(ax)
            pre = np.array(ln[0]) - ax * 0.45
            ev.append((t, "navstay", d, self._route(P[d], pre) + [np.array(ln[0]) - ax * 0.12]))
        gap = abs(lanes[1][0][1] - lanes[0][0][1]) * 100
        return ev, f"{n} racers to lane starts on ramp_{ramp} side, lanes {gap:.0f} cm apart; they wait for the start"

    def agent_race_go(self, t):
        ev = []
        note = ""
        if not self.racers:
            ev, _ = self.agent_race_start(t, "west", 2)
            t += 8.0
            note = "no racers staged: staged 2 and started after 8 s; "
        for d, ln in self.racers:
            self.vmax[d] = 0.40
            ev += [(t, "noavoid", [d]), (t, "navstay", d, [np.array(ln[1])])]
        ev.append((t + 12.0, "avoid", [d for d, _ in self.racers]))
        return ev, note + f"{len(self.racers)} lanes released on the same tick, 0.40 m/s command, no dodging inside a lane"

    def agent_follow(self, t, leader, route):
        if route == "over_north_ramp":
            return self.agent_parade(t, leader)
        ducks = [leader] + [d for d in sorted(self._free_ducks(), key=lambda d: np.hypot(
            *(self._pos()[d] - np.array(self.CONGA_STAGE[1])))) if d not in (leader, 0)][:4]
        self.busy |= set(ducks)
        self.lined = ducks
        ev = []
        for qi, d in enumerate(ducks):
            ev += [(t, "avoid", [d]), (t, "navstay", d, self._route(self._pos()[d], self.CONGA_STAGE[qi]))]
        go = t + 10.0
        path = list(self.CONGA_PATH)
        ends = {0: path + [(1.35, -0.3), (1.8, -0.3)], 1: path + [(1.35, -0.3)], 2: path, 3: path[:-1],
                4: path[:-2] + [(-0.1, -0.5)]}
        for qi, d in enumerate(ducks):
            self.vmax[d] = 0.36
            ev += [(go + 1.2 * qi, "noavoid", [d]), (go + 1.2 * qi, "navstay", d, [np.array(p) for p in ends[qi]])]
        ev.append((go + 30.0, "avoid", ducks))
        # bystanders standing on or next to the route step aside first
        poly = [np.array(p) for p in [self.CONGA_STAGE[0]] + path]
        P = self._pos()
        moved = 0
        taken = []
        for d in self._free_ducks():
            if d in ducks:
                continue
            best = None
            for a_, b_ in zip(poly[:-1], poly[1:]):
                ab = b_ - a_
                u = np.clip((P[d] - a_) @ ab / (ab @ ab + 1e-9), 0, 1)
                q = a_ + u * ab
                dd = np.linalg.norm(P[d] - q)
                if best is None or dd < best[0]:
                    best = (dd, q, ab)
            if best[0] < 0.5:
                def seg_d(x):
                    return min(np.linalg.norm(x - (a_ + np.clip((x - a_) @ (b_ - a_) / ((b_ - a_) @ (b_ - a_)), 0, 1)
                                                    * (b_ - a_))) for a_, b_ in zip(poly[:-1], poly[1:]))
                others = [P[o] for o in range(12) if o not in ducks and o != d]
                for r in (0.7, 0.9, 1.1, 1.35, 1.6):
                    cands = [P[d] + r * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
                    ok = [c for c in cands if not self.keepout(c, 0.05) and seg_d(c) > 0.5
                          and all(np.linalg.norm(c - o) > 0.4 for o in taken + others)]
                    if ok:
                        tgt = max(ok, key=seg_d)
                        ev += [(t, "avoid", [d]), (t, "navstay", d, self._route(P[d], tgt))]
                        taken.append(tgt)
                        moved += 1
                        break
        return ev, (f"duck {leader + 1} leads, 4 followers line up 38 cm apart, then follow one path around the cone "
                    f"with 1.2 s staggered starts; {moved} bystanders step off the route")

    SPLIT_EXITS = {"north": [(2.05, y) for y in (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)],
                   "west": [(-2.05, y) for y in (-0.9, 1.75, -0.5, 1.35, -0.1, 0.3)]}

    def _ramp_ends(self, ramp, P, g):
        r = next(r for r in RAMPS if r[0] == f"ramp_{ramp}")
        name, cx, cy, yaw, lu, lt, ld, h, w = r
        ax = np.array([math.cos(yaw), math.sin(yaw)])
        c = np.array([cx, cy])
        a, b = c - ax * ((lu + lt + ld) / 2 + 0.2), c + ax * ((lu + lt + ld) / 2 + 0.2)
        if ramp == "west":  # the far ends of both ramps are close to a wall: always walk away from the centre
            a, b = b, a
        return a, b

    def _clear_route(self, t, poly, keep):
        """Bystanders within 50 cm of a route step aside first. Returns (events, how many moved)."""
        poly = [np.asarray(p, float) for p in poly]
        P = self._pos()

        def seg_d(x):
            return min(np.linalg.norm(x - (a_ + np.clip((x - a_) @ (b_ - a_) / ((b_ - a_) @ (b_ - a_) + 1e-9), 0, 1)
                                            * (b_ - a_))) for a_, b_ in zip(poly[:-1], poly[1:]))
        ev, taken, moved = [], [], 0
        for d in self._free_ducks():
            if d in keep or seg_d(P[d]) >= 0.5:
                continue
            others = [P[o] for o in range(12) if o not in keep and o != d]
            for r in (0.7, 0.9, 1.1, 1.35, 1.6):
                cands = [P[d] + r * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
                ok = [c for c in cands if not self.keepout(c, 0.05) and seg_d(c) > 0.5
                      and all(np.linalg.norm(c - o) > 0.4 for o in taken + others)]
                if ok:
                    tgt = max(ok, key=seg_d)
                    ev += [(t, "avoid", [d]), (t, "navstay", d, self._route(P[d], tgt))]
                    taken.append(tgt)
                    moved += 1
                    break
        return ev, moved

    def agent_parade(self, t, leader):
        """Leader + 4 followers queue in the corridor, then cross ramp_north together as one line (as in take08)."""
        P = self._pos()
        ducks = [leader] + [d for d in sorted(self._free_ducks(), key=lambda d: np.hypot(*(P[d] - np.array(self.STAGE[2]))))
                            if d not in (leader, 0)][:4]
        self.busy |= set(ducks)
        self.lined = ducks
        ev = []
        for qi, d in enumerate(ducks):
            ev += [(t, "avoid", [d]), (t, "navstay", d, self._route(P[d], self.STAGE[qi]))]
        go = t + 11.0
        for qi, d in enumerate(ducks):  # staggered: the front duck slows for the corner, so equal starts bunch up
            self.vmax[d] = 0.36
            tq = go + 1.3 * qi
            ev += [(tq, "noavoid", [d]), (tq, "navstay", d, [np.array(p) for p in self.PATH[:-1]] + [np.array(self.EXIT[qi])])]
        ev.append((go + 22.0, "avoid", ducks))
        ev2, moved = self._clear_route(t, list(self.STAGE) + list(self.PATH[:-1]) + [self.EXIT[0]], set(ducks))
        return ev + ev2, (f"duck {leader + 1} leads, 4 followers queue 42 cm apart in the corridor, then the line crosses "
                          f"ramp_north at 0.36 m/s, starts 1.3 s apart; {moved} bystanders step off the route")

    def agent_split(self, t, left, right):
        P = self._pos()
        ducks = sorted(self._free_ducks(), key=lambda d: P[d][0])
        half = len(ducks) // 2
        if left == right:
            right = "north" if left == "west" else "west"
        ev = [(t, "avoid", ducks)]
        for g, ramp, delay in ((ducks[half:], right, 0.0), (ducks[:half], left, 10.0)):
            a, b = self._ramp_ends(ramp, P, g)
            axis = (b - a) / np.linalg.norm(b - a)
            side = np.array([-axis[1], axis[0]])
            g = sorted(g, key=lambda d: np.hypot(*(P[d] - a)))
            exits = self.SPLIT_EXITS.get(ramp, self.SPLIT_EXITS["north"])
            lane = 0.12 if ramp == "west" else 0.15
            for k, d in enumerate(g):
                off = side * (lane if k % 2 else -lane)
                tk = t + delay + 2.2 * (k // 2)
                pre = self._route(P[d], a + off)      # straight to the low end of the lane: no U-turn before the ramp
                pts = pre + [b + off, np.array(exits[k % len(exits)])]
                self.vmax[d] = 0.32
                ev.append((tk, "navstay", d, pts))
                # on the ramp a lane goes straight: no dodging from 0.5 m before the ramp until it is off the far end
                L_pre = sum(np.linalg.norm(np.asarray(pts[j + 1]) - np.asarray(pts[j]))
                            for j in range(len(pre))) + np.linalg.norm(np.asarray(pre[0]) - P[d])
                L_ramp = np.linalg.norm(b - a)
                ev.append((tk + L_pre / 0.30, "noavoid", [d]))
                ev.append((tk + (L_pre + L_ramp) / 0.28 + 2.0, "avoid", [d]))
        return ev, (f"{half} ducks (west half) over ramp_{left}, {len(ducks) - half} (east half) over ramp_{right}; "
                    "east group first, west group 10 s later; pairs 24-30 cm apart, 2.2 s apart, no dodging on the ramp, own exit spots")

    RING_C, RING_R = (1.0, -0.1), 0.65   # 12 slots 34 cm apart, 45 cm east of the queue corridor, clear of cones and ramps

    def _ring(self):
        c = np.array(self.RING_C)
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        slots = [c + self.RING_R * np.array([math.cos(a), math.sin(a)]) for a in ang]
        west = sorted(sorted(range(12), key=lambda k: slots[k][0])[:5], key=lambda k: -slots[k][1])
        slot_of = {duck: west[qi] for qi, duck in enumerate(self.QUEUE)}
        rest = [k for k in range(12) if k not in west]
        for k, duck in enumerate(self.LOOSE):
            slot_of[duck] = rest[k]
        return slots, slot_of

    def agent_gather(self, t):
        slots, slot_of = self._ring()
        if self.POS is not None and hasattr(self, "_gathered"):  # regroup: nearest free slot, no paths across the ring
            free = [d for d in range(12) if d not in getattr(self, "busy", set())]
            slot_of = self._assign(free, slots)
        busy = getattr(self, "busy", set())
        ev = [(t, "avoid", [d for d in range(12) if d not in busy])]
        P = self._pos()
        movers = [d for d in slot_of if d not in busy]
        rank = {d: k for k, d in enumerate(sorted(movers, key=lambda d: np.linalg.norm(P[d] - slots[slot_of[d]])))}
        for duck, sl in slot_of.items():
            if duck in busy:
                continue
            if duck == 0 and not hasattr(self, "_gathered") and self.POS is None:
                pts = [np.array([-0.10, 1.75]), np.array([-0.10, 1.15]), slots[sl]]
            else:
                pts = self._route(P[duck], slots[sl])
            ev.append((t + 0.3 * rank[duck], "navstay", duck, pts))  # staggered starts, nearest first
        self._gathered = True
        self.home = {d: np.array(slots[sl]) for d, sl in slot_of.items()}
        return ev

    RAMP_TOPS = {"north_ramp_top": ([(-0.05, 1.45), (0.3, 1.45)], (1.10, 1.45), -math.pi / 2),
                 "west_ramp_top": ([(0.15, 0.75), (-0.25, 0.75)], (-1.15, 0.75), 0.0),
                 "south_ramp_top": ([(-0.20, -0.45), (-0.20, -0.85)], (-0.20, -1.55), math.pi / 2)}

    def agent_send(self, t, d, place):
        """One duck walks alone to a named place and stands there; everyone else keeps still."""
        P = self._pos()
        self.vmax[d] = 0.30
        if place in self.RAMP_TOPS:
            (app, foot), top, face = self.RAMP_TOPS[place]
            pts = self._route(P[d], np.array(app)) + [np.array(foot), np.array(top)]
            self.busy.add(d)
            self.parked = getattr(self, "parked", set()) | {d}
            return ([(t, "avoid", [d]), (t, "navstay", d, pts, {"precise": True, "face": face})],
                    f"duck {d + 1} alone: routed to the foot of {place.replace('_top', '')}, then up to the top, faces the yard")
        home = getattr(self, "home", {}).get(d)
        if home is None:
            home = np.array(self.RING_C)
        self.busy.discard(d)
        self.parked = getattr(self, "parked", set()) - {d}
        face = math.atan2(self.RING_C[1] - home[1], self.RING_C[0] - home[0])
        pts = self._route(P[d], home)
        for (app, foot), top, _ in self.RAMP_TOPS.values():
            if np.linalg.norm(P[d] - np.array(top)) < 0.45:   # on a ramp top: walk off whichever end is shorter
                top, foot, app = np.array(top), np.array(foot), np.array(app)
                ends = []
                for f, a_ in ((foot, app), (2 * top - foot, 2 * top - app)):
                    a_ = np.clip(a_, -2.12, 2.12)
                    rest = self._route(a_, home)
                    L = np.linalg.norm(f - P[d]) + np.linalg.norm(a_ - f) + sum(
                        np.linalg.norm(np.asarray(q) - np.asarray(p_)) for p_, q in zip([a_] + rest[:-1], rest))
                    ends.append((L, [f, a_] + rest))
                pts = min(ends, key=lambda e: e[0])[1]
        return ([(t, "avoid", [d]), (t, "navstay", d, pts, {"precise": True, "face": face})],
                f"duck {d + 1} alone: routed back to its slot on the centre circle")

    def agent_line_up(self, t, n):
        n = max(1, min(n, len(self.QUEUE)))
        self.lined = self.QUEUE[:n]
        P = self._pos()
        return [(t, "navstay", d, self._route(P[d], self.STAGE[qi]), {"precise": True, "face": math.pi / 2})
                for qi, d in enumerate(self.lined)]

    def agent_cross(self, t, ramp, form, n):
        if ramp == "north" and form == "line":
            ev = []
            if not self.lined or len(self.lined) < min(n, len(self.QUEUE)):
                ev += self.agent_line_up(t, n)
                t = t + 9.0
                note = f"queue {len(self.lined)} ducks: slots 42 cm apart, wait 9 s, then all go together on one path"
            else:
                note = f"queue of {len(self.lined)} ducks: one path over ramp_north, starts 1.0 s apart, 0.36 m/s command"
            for qi, d in enumerate(self.lined):  # staggered starts: equal starts bunch up at the corner
                self.vmax[d] = 0.36
                tq = t + 1.0 * qi
                ev.append((tq, "noavoid", [d]))
                ev.append((tq, "navstay", d, [np.array(p) for p in self.PATH[:-1]] + [np.array(self.EXIT[qi])]))
            ev.append((t + 21.0, "avoid", list(self.lined)))
            return ev, note
        ducks = (self.lined or self.QUEUE)[:n] if n < 12 else list(range(12))
        ev = []
        for k, d in enumerate(ducks):
            ev.append((t + (1.5 * k if form == "line" else 0.0), "nav", d, crossing(f"ramp_{ramp}")))
        return ev, f"{len(ducks)} ducks cross ramp_{ramp} ({form}), waypoints at both ends"

    def agent_circle(self, t, direction, laps):
        """Ducks on the centre-circle ring walk round it together: same speed, same start, no dodging in the ring."""
        c, r = np.array(self.RING_C), self.RING_R
        P = self._pos()
        ducks = [d for d in self._free_ducks() if abs(np.linalg.norm(P[d] - c) - r) < 0.25]
        sgn = -1.0 if direction == "clockwise" else 1.0
        laps = max(0.5, min(float(laps or 1), 2.0))
        n_pts = int(round(24 * laps))
        ev = [(t, "noavoid", ducks)]
        for d in ducks:
            a0 = math.atan2(*(P[d] - c)[::-1])
            pts = [c + r * np.array([math.cos(a0 + sgn * 2 * math.pi * k / 24), math.sin(a0 + sgn * 2 * math.pi * k / 24)])
                   for k in range(1, n_pts + 1)]
            self.vmax[d] = 0.25
            ev.append((t, "navstay", d, pts))
        T_lap = 2 * math.pi * r * laps / 0.22
        ev.append((t + T_lap + 3.0, "avoid", ducks))
        return ev, (f"{len(ducks)} ducks on the {r:.2f} m ring walk {direction} together, {laps:g} lap, 0.25 m/s command, "
                    "no dodging inside the ring")

    def shape_slots(self, shape, n=12):
        """n slots evenly spaced on a shape, clear of cones and ramps (the centre of the yard is busy). With fewer
        than 12 ducks the shape shrinks and moves off the corridor at x = -0.10 used by lines."""
        if shape == "line":
            x0, x1 = (-1.85, 2.05) if n >= 9 else (0.3, min(2.1, 0.3 + 0.36 * (n - 1)))
            return [np.array([x, -0.55]) for x in np.linspace(x0, x1, n)]
        small = n < 12
        if shape == "square":  # turned 45 degrees so it fits between the cones
            c, r = (np.array([0.85, -0.3]), 0.6) if small else (np.array([0.72, -0.18]), 0.82)
            corners = [c + r * np.array(v) for v in ((1, 0), (0, 1), (-1, 0), (0, -1))]
        else:
            c, R = (np.array([0.95, -0.28]), 0.75) if small else (np.array([0.75, -0.15]), 0.95)
            corners = [c + R * np.array([math.cos(a), math.sin(a)]) for a in
                       (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)]
        segs = [(corners[k], corners[(k + 1) % len(corners)]) for k in range(len(corners))]
        L = [np.linalg.norm(b - a) for a, b in segs]
        tot = sum(L)
        pts = []
        for m in range(n):
            u = tot * m / n
            for (a, b), l in zip(segs, L):
                if u <= l + 1e-9:
                    pts.append(a + (b - a) * u / l)
                    break
                u -= l
        return pts

    def agent_shape(self, t, shape):
        ducks = self._free_ducks()
        slots = self.shape_slots(shape, len(ducks))
        a = self._assign(ducks, slots)
        P = self._pos()
        ev = [(t, "avoid", ducks)] + [(t, "navstay", d, self._route(P[d], slots[s])) for d, s in a.items()]
        sp = np.linalg.norm(slots[1] - slots[0])
        return ev, f"{len(ducks)} ducks to {len(slots)} slots on a {shape}, {sp * 100:.0f} cm apart, nearest free slot each"

    def agent_stop(self, t):
        """Everyone holds still, 0.2 s apart, starting nearest the north ramp (a ripple outwards)."""
        P = self._pos()
        order = sorted(range(12), key=lambda d: np.hypot(*(P[d] - np.array([1.1, 1.45]))))
        return [(t + 0.2 * k, "hold", d) for k, d in enumerate(order)], "every duck holds still, 0.2 s apart, from the ramp outwards"

    def agent_wander(self, t):
        self.busy, self.lined, self.racers = set(), [], []
        ev = [(t, "mode", "wander"), (t, "avoid", list(range(12)))]
        ev += [(t, "pause_until", i, 0.0) for i in range(12)]
        return ev

    CONGA = [4, 8, 2, 5, 11]                 # front to back; hero (index 2) is third
    CONGA_STAGE = [(0.60, -1.5), (0.98, -1.5), (1.36, -1.5), (1.74, -1.5), (2.12, -1.5)]
    CONGA_PATH = [(0.45, -1.15), (0.45, -0.45), (0.15, -0.1), (-0.15, 0.05), (-0.8, 0.15), (-1.35, 0.15), (-1.75, 0.12),
                  (-1.95, -0.15), (-1.9, -0.5), (-1.6, -0.72), (-1.2, -0.75), (-0.6, -0.55), (0.3, -0.45), (0.9, -0.3)]
    CONGA_ROW = {0: -2.0, 1: -1.5, 3: -0.5, 6: 0.1, 7: 0.7, 9: 1.3, 10: 1.9}

    def conga(self, place):
        place = {}
        for qi, duck in enumerate(self.CONGA):
            place[duck] = (np.array(self.CONGA_STAGE[qi]), math.pi)
        for i, x in self.CONGA_ROW.items():
            place[i] = (np.array([x, 1.95]), -math.pi / 2)
        place[0] = (np.array([-0.45, 2.02]), 0.0)
        ev = [(0.0, "noavoid", [0]), (0.0, "nav", 0, [np.array([0.2, 2.02])]), (3.5, "navstay", 0, [np.array([-2.0, 1.95])])]
        for qi, duck in enumerate(self.CONGA):
            self.vmax[duck] = 0.36
            ev.append((0.0, "hold", duck))
            ev.append((6.0 + 1.2 * qi, "noavoid", [duck]))
            path = list(self.CONGA_PATH)
            path = {0: path + [(1.35, -0.3), (1.8, -0.3)], 1: path + [(1.35, -0.3)], 2: path,
                    3: path[:-1], 4: path[:-2] + [(-0.1, -0.5)]}[qi]
            ev.append((6.0 + 1.2 * qi, "navstay", duck, [np.array(p) for p in path]))
        for i in self.CONGA_ROW:
            if i:
                ev.append((0.0, "hold", i))
        ev += [(40.0, "avoid", list(self.CONGA)), (41.0, "push", 2, 1.6), (43.5, "push", 11, 1.4)]
        return ev, place

    RACE_START = {0: (-0.45, 2.02), 1: (1.5, -1.6), 2: (-0.6, 1.30), 3: (0.6, -1.4), 4: (1.0, 0.62), 5: (1.0, 0.90),
                  6: (-1.75, -0.55), 7: (1.7, 0.1), 8: (-0.6, 1.62), 9: (0.4, -0.1), 10: (1.9, -0.4), 11: (0.35, -0.75)}
    ROW = {0: -1.9, 1: -1.4, 6: -0.9, 3: 0.35, 9: 0.8, 11: 1.2, 7: 1.6, 10: 2.0}

    def race(self, place):
        place = {i: (np.array(p), float(self.rng.uniform(-math.pi, math.pi)) if i else 0.0)
                 for i, p in self.RACE_START.items()}
        for i, (start, finish) in self.RACERS.items():
            place[i] = (np.array(place[i][0]), 0.0 if finish[0] > start[0] else math.pi)
        ev = [(0.0, "noavoid", [0]), (0.0, "nav", 0, [np.array([0.2, 2.02])]), (3.5, "avoid", [0])]
        for i, x in self.ROW.items():
            pre = [np.array([-0.10, 1.75]), np.array([-0.10, 1.15])] if i == 0 else []
            ev.append((4.0, "navstay", i, pre + [np.array([x, -0.55])]))
        for i, (start, finish) in self.RACERS.items():
            ev.append((4.0, "navstay", i, [np.array(start)]))
            self.vmax[i] = 0.40
            ev.append((17.0, "noavoid", [i]))
            ev.append((17.0, "navstay", i, [np.array(finish)]))
        ev += [(31.0, "avoid", list(self.RACERS)), (33.0, "push", 2, 1.6), (38.5, "push", 5, 1.4)]
        ev += [(31.0, "navstay", 6, [np.array([-1.2, 0.08])]), (31.0, "navstay", 9, [np.array([1.3, 0.08])]),
               (36.0, "noavoid", [6, 9]), (36.0, "nav", 6, [np.array([1.6, 0.08])]),
               (36.0, "nav", 9, [np.array([-1.5, 0.08])]), (46.0, "avoid", [6, 9])]
        return ev, place

    START = {0: (-0.45, 2.02), 1: (1.5, -1.6), 2: (-1.0, -0.7), 3: (0.6, -1.4), 4: (-1.7, -1.0), 5: (-0.7, 0.15),
             6: (-1.75, -0.55), 7: (1.6, 0.7), 8: (1.1, 0.35), 9: (0.75, 0.75), 10: (1.9, -0.4), 11: (0.35, -0.75)}

    def story(self, place):
        c = np.array([0.6, -0.2])
        place = {i: (np.array(p), float(self.rng.uniform(-math.pi, math.pi)) if i else 0.0) for i, p in self.START.items()}
        ev = []
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False) + 0.2
        slots = [c + 0.62 * np.array([math.cos(a), math.sin(a)]) for a in ang]
        # queue ducks take the west-side slots (nearest the corridor), in line order from north to south
        west = sorted(range(12), key=lambda k: slots[k][0])[:len(self.QUEUE)]
        west = sorted(west, key=lambda k: -slots[k][1])
        slot_of = {duck: west[qi] for qi, duck in enumerate(self.QUEUE)}
        rest = [k for k in range(12) if k not in west]
        for k, duck in enumerate(self.LOOSE):
            slot_of[duck] = rest[k]
        ev.append((0.0, "noavoid", [0]))
        ev.append((0.0, "nav", 0, [np.array([0.2, 2.02])]))
        ev.append((self.T_GATHER - 0.5, "avoid", [0]))
        for duck, sl in slot_of.items():
            pre = [np.array([-0.10, 1.75]), np.array([-0.10, 1.15])] if duck == 0 else []
            ev.append((self.T_GATHER, "navstay", duck, pre + [slots[sl]]))
        # order 2: the line forms in the corridor (others head for the other ramps)
        for qi, duck in enumerate(self.QUEUE):
            ev.append((self.T_RAMPS, "navstay", duck, [np.array(self.STAGE[qi])]))
        ev.append((self.T_RAMPS + 0.5, "navstay", 6, [np.array([-1.35, -0.55])]))
        ev.append((self.T_RAMPS + 0.5, "navstay", 9, [np.array([1.05, -0.55])]))
        # go: the whole line walks off together on one path, same speed, no dodging inside the line
        for qi, duck in enumerate(self.QUEUE):
            self.vmax[duck] = 0.38
            ev.append((self.T_GO, "noavoid", [duck]))
            ev.append((self.T_GO, "navstay", duck, [np.array(p) for p in self.PATH[(0 if qi else 0):]]))
        # falls with a visible cause: a head-on collision in the open and two shoves after the ramp
        ev.append((44.0, "noavoid", [6, 9]))
        ev.append((44.0, "nav", 6, [np.array([1.4, -0.55])]))
        ev.append((44.0, "nav", 9, [np.array([-1.7, -0.55])]))
        ev.append((53.0, "avoid", [6, 9]))
        ev.append((46.0, "avoid", list(self.QUEUE)))
        ev.append((47.0, "push", 2, 1.6))
        ev.append((49.5, "push", 4, 1.4))
        return ev, place

    def apply(self, e, t):
        """Non-physical events; returns a push request (duck, magnitude) or None."""
        op = e[1]
        if op in ("nav", "navstay"):
            opts = e[4] if len(e) > 4 else {}
            self.nav[e[2]] = dict(kind="path", pts=[np.array(p) for p in e[3]], until=t + 60.0, stay=op == "navstay",
                                  precise=opts.get("precise", False), face=opts.get("face"))
        elif op == "hold":
            self.nav[e[2]] = dict(kind="pause", until=t + 1e9, hold=True)
        elif op == "noavoid":
            self.noavoid |= set(e[2])
        elif op == "avoid":
            self.noavoid -= set(e[2])
        elif op == "mode":
            self.mode = e[2]
            if self.mode in ("wander", "ramps"):
                for i in range(NDUCK):
                    if i != self.hero and self.nav[i].get("kind") == "pause":
                        self.nav[i] = dict(kind="pause", until=0)
        elif op == "push":
            return e[2], e[3]
        elif op == "pause_until":
            self.nav[e[2]] = dict(kind="pause", until=e[3])
        return None

    def new_plan(self, i, pos, t):
        if self.mode == "idle":
            self.nav[i] = dict(kind="pause", until=t + 1e9, hold=True)
            return
        r = self.rng.random()
        if self.mode == "ramps" or (self.mode == "wander" and r < 0.4):
            ramp = RAMPS[self.rng.integers(len(RAMPS))]
            a, b = ramp_endpoints(ramp)
            if np.hypot(*(pos - b)) < np.hypot(*(pos - a)):
                a, b = b, a
            self.nav[i] = dict(kind="path", pts=self._route(pos, a, ) + [b], until=t + 40.0)
        elif self.mode == "wander" and r < 0.5:
            self.nav[i] = dict(kind="pause", until=t + self.rng.uniform(1.5, 3.5))
        elif self.mode == "wander" and r < 0.56:
            self.nav[i] = dict(kind="spin", wz=float(self.rng.choice([-WZ_MAX, WZ_MAX])),
                               until=t + self.rng.uniform(2.0, 4.0))
        else:
            self.nav[i] = dict(kind="path", pts=self._route(pos, self.free_spot([])), until=t + 25.0)

    def update_falls(self, t, upz):
        upright = upz > math.cos(math.radians(55))
        for i in range(NDUCK):
            if not upright[i]:
                if self.fallen_since[i] < 0:
                    self.fallen_since[i] = t
            elif self.fallen_since[i] >= 0 and upz[i] > 0.9:
                self.fallen_since[i] = -1.0
                if not self.nav[i].get("hold") and self.nav[i].get("kind") != "path":
                    self.nav[i] = dict(kind="pause", until=t + 1.0)
        return upright

    def steer(self, t, pos, yaw, upright):
        for i in range(NDUCK):
            if self.mode == "stop" or not upright[i]:
                self.target[i] = 0
                continue
            nv = self.nav[i]
            if t > nv.get("until", 0):
                self.new_plan(i, pos[i], t)
                nv = self.nav[i]
            if nv["kind"] == "pause":
                self.target[i] = 0
                continue
            if nv["kind"] == "spin":
                self.target[i] = [0.0, 0.0, nv["wz"]]
                continue
            if nv["kind"] == "face":  # turn on the spot to a heading, then stand
                err = (nv["yaw"] - yaw[i] + math.pi) % (2 * math.pi) - math.pi
                self.target[i] = [0.0, 0.0, float(np.clip(1.4 * err, -WZ_MAX, WZ_MAX))] if abs(err) > 0.12 else 0
                continue
            p = pos[i]
            goal = nv["pts"][0]
            last = len(nv["pts"]) == 1
            if np.hypot(*(goal - p)) < (0.07 if (nv.get("precise") and last) else 0.22):
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
            for name, cx, cy, ryaw, lu, lt, ld, rh, rw in RAMPS:  # ramp side edges (up to 5 cm high), for every duck
                c_, s_ = math.cos(ryaw), math.sin(ryaw)
                lx = c_ * (p[0] - cx) + s_ * (p[1] - cy)
                ly = -s_ * (p[0] - cx) + c_ * (p[1] - cy)
                if abs(lx) < (lu + lt + ld) / 2 - 0.05:
                    dist = abs(ly) - rw / 2
                    reach = 0.22 if dist > 0 else 0.10     # beside the ramp: 22 cm; on top: only the last 10 cm
                    if abs(dist) < reach:
                        sgn = math.copysign(1.0, ly) * (1.0 if dist > 0 else -1.0)
                        rep += sgn * 1.6 * (reach - abs(dist)) / reach * np.array([-s_, c_])
            if i not in self.noavoid:
                for ox, oy, r in self.obstacles:
                    v = p - np.array([ox, oy])
                    dist = np.linalg.norm(v)
                    if dist < r + 0.3:
                        rep += v / (dist + 1e-9) * 1.5 * (r + 0.3 - dist) / 0.3
                for j in range(NDUCK):
                    if j != i:
                        v = p - pos[j]
                        dist = np.linalg.norm(v)
                        if dist < 0.32:
                            rep += v / (dist + 1e-9) * 1.2 * (0.32 - dist) / 0.32
                for k in range(2):
                    for s in (-1, 1):
                        dw = WALL_IN - s * p[k]
                        if dw < 0.45:
                            e = np.zeros(2)
                            e[k] = -s
                            rep += e * 2.0 * (0.45 - dw) / 0.45
            dvec = dvec + rep
            des = math.atan2(dvec[1], dvec[0])
            err = (des - yaw[i] + math.pi) % (2 * math.pi) - math.pi
            wz = float(np.clip(1.6 * err, -WZ_MAX, WZ_MAX))
            vx = self.vmax[i] * max(0.0, math.cos(err)) ** 2
            if nv.get("precise") and len(nv["pts"]) == 1:  # slow final approach so the duck stops on its slot
                vx *= float(np.clip(np.hypot(*(nv["pts"][0] - p)) / 0.35, 0.3, 1.0))
            if abs(err) > 1.2:
                vx = 0.0
            self.target[i] = [vx, 0.0, wz]

    def smooth(self):
        self.cmd += 0.2 * (self.target - self.cmd)
        return self.cmd
