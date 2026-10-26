"""Keynote demo, live: 12 Microducks steered only from what 4 RGB-D cameras perceive, ordered by NVIDIA Nemotron.

    python keynote_live_sim.py --kit --servo pd [--ctrl-port 18211] [--shm /dev/shm/microducks_keynote_pose.bin]

Same world, policy, perception (perception.py) and navigator (director_nemo.py) as the keynote take
(isaac_nemo/isaac_ducks_sim_nemo.py), run forever in real time:
- 50 Hz: ONNX walking policy for the 12 ducks, one sim step with 4 Newton substeps.
- 10 Hz: the 4 fixed cameras render (Isaac Lab cameras, Newton Warp renderer), perception turns depth + segmentation
  into duck poses, the navigator steers from those poses on the map the cameras built in the first 1.5 s.
- Typed orders (POST /api/control {action: "order", text}) go to Nemotron in a worker thread: perception tools, then one
  action tool, executed by the navigator in the main loop. The camera check of each order goes back to Nemotron, whose
  answer lands in the agent feed (/api/info -> agent.feed).
Body poses go to shared memory for keynote_live_ovrtx.py; /sensor/map.jpg and /sensor/eye.jpg are the perceived map and
the camera Nemotron looks through.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import queue
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=102)
parser.add_argument("--servo", choices=("pd",), default="pd")
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--ls-iters", type=int, default=50)
parser.add_argument("--njmax", type=int, default=2400)
parser.add_argument("--nconmax", type=int, default=600)
parser.add_argument("--kit", action="store_true")
parser.add_argument("--asset", default=os.path.join(HERE, "assets", "microduck_allcollisions_isaac.usda"))
parser.add_argument("--policy", default=os.path.join(HERE, "policy_np.npz"))
parser.add_argument("--perc-hz", type=float, default=10.0)
parser.add_argument("--map-secs", type=float, default=1.5)
parser.add_argument("--depth-noise", type=float, default=0.002)
parser.add_argument("--shm", default="/dev/shm/microducks_keynote_pose.bin")
parser.add_argument("--ctrl-port", type=int, default=18211)
parser.add_argument("--fonts", default=os.path.join(os.path.dirname(HERE), "fonts"))
parser.add_argument("--start", choices=("wander", "idle"), default="wander", help="after the map: wander or stand")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this sim time (0 = forever, for tests)")
add_launcher_args(parser)
args = parser.parse_args()
if args.kit:
    args.require_kit = True

import numpy as np  # noqa: E402

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
          "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HOME = {"left_hip_yaw": 0.0, "right_hip_yaw": 0.0, "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
        "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579, "left_knee": -0.0049, "right_knee": 0.0049,
        "left_ankle": 0.4530, "right_ankle": -0.4530, "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0,
        "head_roll": 0.0}
CTRL_DT, NSUB = 0.02, 4
START = {0: (-0.45, 2.02), 1: (1.5, -1.6), 2: (-1.0, -0.7), 3: (0.6, -1.4), 4: (-1.7, -1.0), 5: (-0.7, 0.15),
         6: (-1.75, -0.55), 7: (1.6, 0.7), 8: (1.1, 0.35), 9: (0.75, 0.75), 10: (1.9, -0.4), 11: (0.35, -0.75)}
BODY_ORDER = ['trunk_base', 'bearing_roll', 'hip_l_2', 'upper_leg_right', 'leg_2', 'ankle_right', 'neck',
              'neck_pitch_1', 'yaw_roll_motion', 'jaw_soft', 'yaw2roll', 'hip_l', 'upper_leg_left', 'leg', 'ankle_left']
HDR = 16
CHECK_WAIT = {"gather_center": 22.0, "regroup": 22.0, "line_up": 14.0, "cross_ramp": 26.0, "stop_all": 4.5,
              "scatter": 16.0, "hold_on_ramp": 24.0, "walk_circle": 50.0, "spin": 16.0}
GOAL_WORDS = {"circle": "gathering on the centre circle", "line_up": "lining up at the north ramp",
              "cross_ramp": "crossing the north ramp", "stop": "standing still", "scatter": "spreading out",
              "ramp_hold": "standing on a ramp", "circle_walk": "walking round the centre circle",
              "spin": "spinning on the spot"}


def numpy_policy(path):
    d = np.load(path)
    graph = json.loads(str(d["graph"]))
    meta = json.loads(str(d["meta"]))

    def run(x):
        x = np.asarray(x, np.float64)
        for s in graph:
            P = [d[i].astype(np.float64) for i in s["ins"] if i != "x"]
            if s["op"] == "Sub":
                x = x - P[0]
            elif s["op"] == "Div":
                x = x / P[0]
            elif s["op"] == "Gemm":
                W = P[0].T if s["transB"] else P[0]
                x = x @ W + (P[1] if len(P) > 1 else 0.0)
            elif s["op"] == "Elu":
                x = np.where(x > 0, x, s["alpha"] * (np.exp(np.minimum(x, 0)) - 1))
            else:
                raise RuntimeError(s["op"])
        return x

    return run, meta


def main():
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import instantiate
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    import agent_nemo as AG
    import director_nemo as DN
    import perception as P
    import yard_layout as Y          # builds the simulated world and the respawn supervisor; the navigator never sees it

    act_cfg = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.55, damping=0.053, effort_limit_sim=0.96,
                                  armature=0.0018, friction=0.0048)
    sim_cfg = sim_utils.SimulationCfg(dt=CTRL_DT, render_interval=1, device=args.device or "cuda:0",
                                      physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(
                                          njmax=args.njmax, nconmax=args.nconmax, iterations=args.iters,
                                          ls_iterations=args.ls_iters, cone="pyramidal", impratio=1.0,
                                          integrator="implicitfast", use_mujoco_contacts=False), num_substeps=NSUB))
    if hasattr(sim_cfg, "use_newton_actuators"):
        sim_cfg.use_newton_actuators = False

    director = DN.NavDirector(args.seed)
    rng = np.random.default_rng(args.seed)
    place = {i: (np.array(p), float(rng.uniform(-math.pi, math.pi)) if i else 0.0) for i, p in START.items()}

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0, replicate_physics=True, lazy_sensor_update=True)
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)))
    scene_cfg.light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    col = sim_utils.CollisionPropertiesCfg()
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8)
    gray = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.62, 0.64))
    wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.55, 0.36))
    k = 0
    for r in Y.RAMPS:
        for name, c, hs, q, role in Y.ramp_boxes(r):
            setattr(scene_cfg, f"yard_{k:02d}_{name}", AssetBaseCfg(
                prim_path=f"/World/Yard/{name}", spawn=sim_utils.CuboidCfg(
                    size=tuple(2 * v for v in hs), collision_props=col, physics_material=mat, visual_material=wood,
                    semantic_tags=[("class", "ramp")]),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(c), rot=tuple(q))))
            k += 1
    for name, c, hs in Y.wall_boxes():
        setattr(scene_cfg, f"yard_{k:02d}_{name}", AssetBaseCfg(
            prim_path=f"/World/Yard/{name}", spawn=sim_utils.CuboidCfg(
                size=tuple(2 * v for v in hs), collision_props=col, physics_material=mat, visual_material=gray,
                semantic_tags=[("class", "wall")]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(c))))
        k += 1
    for i, (x, y) in enumerate(Y.CONES):
        setattr(scene_cfg, f"yard_{k:02d}_cone{i}", AssetBaseCfg(
            prim_path=f"/World/Yard/cone{i}", spawn=sim_utils.CylinderCfg(
                radius=Y.CONE_R, height=Y.CONE_H, collision_props=col, physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.15)),
                semantic_tags=[("class", "cone")]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, Y.CONE_H / 2))))
        k += 1
    for i, (x, y, yaw) in enumerate(Y.PLANTERS):
        setattr(scene_cfg, f"yard_{k:02d}_planter{i}", AssetBaseCfg(
            prim_path=f"/World/Yard/planter{i}", spawn=sim_utils.CuboidCfg(
                size=Y.PLANTER_SIZE, collision_props=col, physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.55, 0.3)),
                semantic_tags=[("class", "planter")]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, Y.PLANTER_SIZE[2] / 2),
                                                    rot=Y.quat_xyzw_from_yaw_pitch(yaw))))
        k += 1
    art_props = None
    try:
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg
        art_props = NewtonArticulationCfg(self_collision_enabled=False)
    except Exception:  # noqa: BLE001
        pass
    for i in range(Y.NDUCK):
        p, yaw = place[i]
        spawn = sim_utils.UsdFileCfg(usd_path=args.asset, activate_contact_sensors=False,
                                     semantic_tags=[("class", "duck")])
        if art_props is not None:
            spawn.articulation_props = art_props
        setattr(scene_cfg, f"duck{i:02d}", ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Duck{i:02d}", spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(float(p[0]), float(p[1]), 0.125),
                                                       rot=Y.quat_xyzw_from_yaw_pitch(float(yaw)),
                                                       joint_pos=dict(HOME), joint_vel={".*": 0.0}),
            soft_joint_pos_limit_factor=0.9, actuators={"servos": act_cfg}))
    rcfg = NewtonWarpRendererCfg(colorize_semantic_segmentation=False, colorize_instance_segmentation=False,
                                 enable_shadows=True, depth_clipping_behavior="zero")
    for cs in P.RIG:
        fx = cs.K[0, 0]
        setattr(scene_cfg, f"perc_{cs.name}", CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/PercCam_{cs.name}", update_period=0.0, width=cs.w, height=cs.h,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation", "instance_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=fx * 20.955 / cs.w, horizontal_aperture=20.955,
                                             clipping_range=(0.05, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(cs.eye), rot=cs.quat_world_xyzw(), convention="world"),
            renderer_cfg=rcfg))

    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        from isaaclab.sim.utils.semantics import add_labels
        try:
            from isaaclab.sim.utils.stage import get_current_stage
            stage = get_current_stage()
        except Exception:  # noqa: BLE001
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        nlab = 0
        for i in range(Y.NDUCK):
            base = f"/World/envs/env_0/Duck{i:02d}/Geometry/trunk_base"
            for sub, lab in (("neck", "duck_head"), ("yaw2roll", "duck_leg_left"), ("bearing_roll", "duck_leg_right")):
                prim = stage.GetPrimAtPath(f"{base}/{sub}")
                if prim.IsValid():
                    add_labels(prim, [lab], instance_name="class")
                    nlab += 1
        gp = stage.GetPrimAtPath("/World/ground")
        if gp.IsValid():
            add_labels(gp, ["floor"], instance_name="class")
        print(f"[keynote] part labels {nlab}/36, ground labelled {gp.IsValid()}", flush=True)
        sim.reset()
        stack = dict(servo="pd (derived from BAM)", kit=bool(args.kit), stepping="1 x 20 ms step / tick, 4 Newton substeps")
        try:
            import importlib.metadata as md
            for pkg in ("isaacsim", "isaaclab", "isaaclab_newton", "newton", "warp-lang"):
                try:
                    stack[pkg + "_dist"] = md.version(pkg)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        ducks = [scene[f"duck{i:02d}"] for i in range(Y.NDUCK)]
        cams = {cs.name: scene[f"perc_{cs.name}"] for cs in P.RIG}
        jids = ducks[0].find_joints(JOINTS, preserve_order=True)[0]
        body_names = list(ducks[0].body_names)
        order = [body_names.index(n) for n in BODY_ORDER]
        policy, meta = numpy_policy(args.policy)
        stack["physics"] = "newton"
        stack["policy"] = json.loads(meta.get("label", '"?"'))
        stack["perception"] = f"{len(P.RIG)} fixed RGB-D cameras, Isaac Lab camera + Newton Warp renderer, {args.perc_hz:g} Hz"
        stack["agent"] = AG.MODEL_NAME
        dev = ducks[0].data.joint_pos.torch.device
        jids_t = torch.as_tensor(jids, device=dev)
        state = dict(last=np.zeros((Y.NDUCK, 14)), t=0.0, selected=2, respawns=0)

        def gather():
            g = torch.stack([d.data.projected_gravity_b.torch[0] for d in ducks])
            w = torch.stack([d.data.root_ang_vel_b.torch[0] for d in ducks])
            jp = torch.stack([d.data.joint_pos.torch[0] - d.data.default_joint_pos.torch[0] for d in ducks])[:, jids_t]
            jv = torch.stack([d.data.joint_vel.torch[0] for d in ducks])[:, jids_t]
            return [x.cpu().numpy() for x in (g, w, jp, jv)]

        def body_poses():
            return torch.stack([d.data.body_link_pose_w.torch[0] for d in ducks]).cpu().numpy()

        def push(i, mag):
            d = ducks[i]
            vel = d.data.root_vel_w.torch.clone()
            yaw = float(d.data.heading_w.torch.reshape(-1)[0])
            ang = yaw + math.pi / 2
            vel[0, 0] += mag * math.cos(ang)
            vel[0, 1] += mag * math.sin(ang)
            vel[0, 3] += -math.sin(ang) * 4.0
            vel[0, 4] += math.cos(ang) * 4.0
            d.write_root_velocity_to_sim_index(root_velocity=vel)

        def respawn(i, p, yaw):
            d = ducks[i]
            pose = d.data.default_root_pose.torch.clone()
            q = Y.quat_xyzw_from_yaw_pitch(float(yaw))
            pose[0, :3] = torch.tensor([float(p[0]), float(p[1]), 0.125], device=dev)
            pose[0, 3:7] = torch.tensor(q, device=dev)
            d.write_root_pose_to_sim_index(root_pose=pose)
            d.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(d.data.root_vel_w.torch))
            d.write_joint_position_to_sim_index(position=d.data.default_joint_pos.torch.clone())
            d.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(d.data.joint_vel.torch))
            state["last"][i] = 0
            director.cmd[i] = 0
            director.target[i] = 0
            director.fallen_since[i] = -1.0
            director.nav[i] = dict(kind="pause", until=state["t"] + 1.0)

        # ------------------------------------------------------------------ perception (as the keynote take)
        def T(x):
            return x.torch if hasattr(x, "torch") else x

        bp = {cs.name: P.Backprojector(cs, dev, args.depth_noise, seed=args.seed * 31 + j) for j, cs in enumerate(P.RIG)}
        luts = {}
        mapper = P.Mapper(dev)
        loc = P.Localizer(dev)
        tim = dict(render=collections.deque(maxlen=100), perceive=collections.deque(maxlen=100),
                   step=collections.deque(maxlen=100))

        def build_luts(name, cam):
            info_ = cam.data.info
            sem_info = info_.get("semantic_segmentation", {}).get("idToLabels", {})
            inst_info = info_.get("instance_segmentation", {})
            cls = torch.as_tensor(P.class_table(sem_info), device=dev)
            duck, part = P.instance_table(inst_info)
            luts[name] = (cls, torch.as_tensor(duck, device=dev), torch.as_tensor(part, device=dev))

        def perceive(t):
            t0 = time.perf_counter()
            frames = []
            for cs in P.RIG:
                cam = cams[cs.name]
                out = cam.data.output
                if cs.name not in luts:
                    build_luts(cs.name, cam)
                frames.append((cs, out))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_cam = []
            for cs, out in frames:
                depth = T(out["distance_to_image_plane"])[0, ..., 0].float()
                sem = T(out["semantic_segmentation"])[0, ..., 0].long()
                inst = T(out["instance_segmentation"])[0, ..., 0].long()
                cls_l, duck_l, part_l = luts[cs.name]
                semc = cls_l[sem.clamp(0, cls_l.numel() - 1)]
                instc = inst.clamp(0, duck_l.numel() - 1)
                dmeas = bp[cs.name].noisy(depth)
                pts = bp[cs.name].points(dmeas)
                valid = dmeas > 0
                duck = torch.where(valid, duck_l[instc], torch.full_like(instc, -1))
                part = part_l[instc]
                if t < args.map_secs:
                    mapper.add(pts[valid], semc[valid])
                per_cam.append(dict(pts=pts, duck=duck, part=part, eye=cs.eye))
            stt = loc.update(t, per_cam, director.cmd)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            tim["render"].append(1000 * (t1 - t0))
            tim["perceive"].append(1000 * (t2 - t1))
            spd = np.zeros(Y.NDUCK)
            for i in range(Y.NDUCK):
                h = loc.hist[i]
                if len(h) >= 2 and h[-1][0] - h[0][0] > 0.2:
                    spd[i] = math.hypot(h[-1][1] - h[0][1], h[-1][2] - h[0][2]) / (h[-1][0] - h[0][0])
            stt["speed"] = spd
            return stt, frames

        checks = collections.deque(maxlen=20)

        def perceived_summary(perc, t):
            ducks_s = []
            for i in range(Y.NDUCK):
                p = perc["pos"][i]
                ducks_s.append(dict(id=i + 1, x=round(float(p[0]), 2), y=round(float(p[1]), 2),
                                    heading_deg=int(round(math.degrees(perc["yaw"][i]))),
                                    state="walking" if perc["speed"][i] > 0.05 else "standing",
                                    upright=bool(perc["ztop"][i] > 0.15), cameras=int(perc["ncam"][i]),
                                    **({"in_line": director.lined.index(i) + 1} if i in director.lined else {})))
            m = director.map
            out = dict(t_s=round(t, 1), source="4 fixed RGB-D cameras, depth + segmentation", ducks=ducks_s)
            if m is not None:
                out["map"] = dict(ramps=[dict(name=r["name"], centre=[round(v, 2) for v in r["c"]],
                                              length_m=round(r["L"], 2), width_m=round(r["W"], 2),
                                              height_cm=round(100 * r["H"], 1)) for r in m.ramps],
                                  cones=len(m.cones), planters=len(m.planters),
                                  walls_inner={k_: round(v, 2) for k_, v in m.walls.items()})
            if checks:
                c = checks[-1]
                out["last_check"] = dict(order=c["order"], ok=c["check"]["ok"], n=c["check"]["n"])
            return out

        # ------------------------------------------------------------------ shared state with the HTTP / agent threads
        tmp = args.shm + ".tmp"
        mm = np.memmap(tmp, dtype=np.float64, mode="w+", shape=(HDR + Y.NDUCK * 15 * 7,))
        os.replace(tmp, args.shm)
        cmds = queue.Queue()
        info = {}
        pending = []
        # the order lock lives in a dict so a soft reset (or a stuck order) can hand out a fresh one
        OL = dict(lock=threading.Lock(), since=0.0)
        ag_lock = threading.Lock()
        state["epoch"] = 0
        state["loop_errors"] = 0
        logdir = os.path.join(HERE, "logs")
        os.makedirs(logdir, exist_ok=True)
        feed = collections.deque(maxlen=30)
        feed_id = [0]
        shared = dict(summary=None, rgb={}, rgb_req={}, eye="ne", eye_until=0.0, eye_jpg=None,
                      map_until=0.0, map_snap=None, map_jpg=(0.0, None))

        def feed_add(kind, text):
            feed_id[0] += 1
            feed.append(dict(id=feed_id[0], wall=time.strftime("%H:%M:%S"), kind=kind, text=str(text)[:400]))

        def jlog(name, rec):
            try:
                with open(os.path.join(logdir, name), "a") as fh:
                    fh.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            except OSError:
                pass

        def trim_history(keep=24):
            H_ = AG.HISTORY
            if len(H_) <= keep:
                return
            cut = next((j for j in range(len(H_) - keep, len(H_)) if H_[j].get("role") == "user"), None)
            if cut:
                del H_[:cut]

        def sensor_rgb(name):
            """One RGB frame of a yard camera, copied by the main loop at its next perception tick."""
            ev = threading.Event()
            shared["rgb_req"][name] = ev
            ev.wait(2.0)
            shared["eye"] = name
            shared["eye_until"] = max(shared["eye_until"], time.time() + 20.0)
            img = shared["rgb"].get(name)
            if img is None:
                cs = next(c for c in P.RIG if c.name == name)
                img = np.zeros((cs.h, cs.w, 3), np.uint8)
            return img

        AG.SENSORS["perceived_state"] = lambda: shared["summary"] or {"error": "the cameras are still mapping the yard"}
        AG.SENSORS["rgb"] = sensor_rgb
        AG.SENSORS["look_dir"] = ""

        def answer_async(chk, text):
            epoch = state["epoch"]

            def run():
                if not ag_lock.acquire(timeout=30):
                    return
                try:
                    try:
                        rec = AG.answer(chk, text)
                    except Exception as ex:  # noqa: BLE001
                        rec = dict(order=text, check=chk, answer=f"(answer failed: {type(ex).__name__})")
                    trim_history()
                finally:
                    ag_lock.release()
                if epoch != state["epoch"]:
                    return
                checks.append(rec)
                feed_add("check", f"{rec['answer']}  [camera check {chk['ok']}/{chk['n']}]")
                jlog("keynote_live_checks.jsonl", rec)
                print(f"[keynote] check {chk['kind']} {chk['ok']}/{chk['n']} -> {rec['answer']!r}", flush=True)
            threading.Thread(target=run, daemon=True).start()

        def perception_notes(rec):
            out = []
            for st in rec.get("steps", []):
                pc = st.get("perception")
                if not pc:
                    continue
                if pc.get("name") == "look":
                    out.append(f"look({pc.get('camera')}) -> {pc.get('answer', '')}")
                else:
                    out.append("perceived_state() -> 12 duck poses from the 4 cameras, mapped ramps, cones and walls")
            return out

        # ------------------------------------------------------------------ sensor images for the live views
        font = small = None
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(os.path.join(args.fonts, "NVIDIASans_Bd.ttf"), 19)
            small = ImageFont.truetype(os.path.join(args.fonts, "NVIDIASans_Bd.ttf"), 15)
        except Exception:  # noqa: BLE001
            pass

        def label(d, text, y):
            d.text((48, y), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))

        def jpeg(im, q=82):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            return buf.getvalue()

        def map_jpg():
            from PIL import Image, ImageDraw
            snap = shared["map_snap"]
            if snap is None or director.map is None:
                return None
            pos, yaw, paths, slots, t = snap
            sc = 2
            img = P.map_image(director.map, sc, pos, yaw, paths, slots)
            N = img.shape[0]
            canvas = Image.new("RGB", (960, 540), (14, 15, 17))
            canvas.paste(Image.fromarray(img[..., :3]), ((960 - N) // 2 - 150, (540 - N) // 2))
            d = ImageDraw.Draw(canvas)
            ox = (960 - N) // 2 - 150
            for i, p in enumerate(pos):
                if np.all(np.isfinite(p)):
                    x = ox + (p[0] + P.GRID_HALF) / P.GRID_RES * sc
                    y = N - (p[1] + P.GRID_HALF) / P.GRID_RES * sc + (540 - N) // 2
                    d.text((x + 9, y - 22), str(i + 1), font=small, fill=(255, 255, 255), stroke_width=2,
                           stroke_fill=(0, 0, 0))
            x0 = ox + N + 24
            lines = [("Perceived map", (255, 255, 255)), ("built by the 4 RGB-D cameras", (190, 190, 190)),
                     ("", None), ("walls, cones, planters", (210, 210, 220)), ("ramps", (255, 200, 90)),
                     ("planned paths", (80, 160, 255)), ("goal slots", (118, 185, 0)),
                     ("", None), ("Duck dots: positions and", (190, 190, 190)),
                     ("headings from depth +", (190, 190, 190)), ("segmentation, 10 Hz", (190, 190, 190))]
            yy = 60
            for txt, col in lines:
                if txt:
                    d.text((x0, yy), txt, font=font if yy == 60 else small, fill=col)
                yy += 30 if yy == 60 else 24
            label(d, f"Perceived map  \u00b7  SIMULATION  \u00b7  t = {t:.0f} s", 540 - 37)
            return jpeg(canvas)

        def eye_jpg():
            from PIL import Image, ImageDraw
            hit = shared["eye_jpg"]
            if hit is None:
                return None
            name, arr = hit
            im = Image.fromarray(arr).resize((960, 540), Image.BILINEAR)
            label(ImageDraw.Draw(im), f"Camera {name.upper()}  \u00b7  RGB-D sensor Nemotron looks through  \u00b7  SIMULATION",
                  540 - 37)
            return jpeg(im)

        # ------------------------------------------------------------------ HTTP control (127.0.0.1 only)
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, obj, ctype="application/json"):
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path.startswith("/api/info"):
                    out = dict(info)
                    out["agent"] = dict(model=AG.MODEL_NAME, busy=OL["lock"].locked(), feed=list(feed)[-10:])
                    return self.send(200, out)
                if self.path.startswith("/api/health"):
                    return self.send(200, dict(ok=True, seq=seq_box[0], sim_time=round(state["t"], 2),
                                               wall=time.time(), map_ready=bool(director.ready),
                                               up=int(info.get("up", 0)), rtf=info.get("rtf"), epoch=state["epoch"],
                                               order_busy=OL["lock"].locked(),
                                               order_busy_s=round(time.time() - OL["since"], 1) if OL["lock"].locked() else 0,
                                               loop_errors=state["loop_errors"], respawns=state["respawns"]))
                if self.path.startswith("/sensor/map.jpg"):
                    shared["map_until"] = time.time() + 3.0
                    t_, jpg = shared["map_jpg"]
                    if jpg is None or time.time() - t_ > 0.2:
                        jpg = map_jpg()
                        shared["map_jpg"] = (time.time(), jpg)
                    return self.send(200, jpg, "image/jpeg") if jpg else self.send(503, {"error": "mapping"})
                if self.path.startswith("/sensor/eye.jpg"):
                    shared["eye_until"] = max(shared["eye_until"], time.time() + 3.0)
                    jpg = eye_jpg()
                    return self.send(200, jpg, "image/jpeg") if jpg else self.send(503, {"error": "warming up"})
                return self.send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    data = json.loads(self.rfile.read(n) or b"{}")
                except Exception:  # noqa: BLE001
                    data = {}
                if str(data.get("action", "")) == "order":
                    return self.order(data)
                box = queue.Queue()
                cmds.put((data, box))
                try:
                    return self.send(200, box.get(timeout=5))
                except queue.Empty:
                    return self.send(503, {"ok": False, "message": "simulation busy"})

            def order(self, data):
                text = " ".join(str(data.get("text") or "").split())[:200]
                if data.get("duck") is not None:
                    state["selected"] = int(np.clip(int(data["duck"]) - 1, 0, Y.NDUCK - 1))
                if not text:
                    return self.send(400, {"ok": False, "message": "Type an order for the ducks first."})
                if not director.ready:
                    return self.send(503, {"ok": False, "message": "The cameras are still mapping the yard."})
                lk = OL["lock"]
                if not lk.acquire(blocking=False):
                    if time.time() - OL["since"] < 60:
                        return self.send(429, {"ok": False, "message": "Nemotron is busy with another order. Try again."})
                    OL["lock"] = lk = threading.Lock()      # the previous order is stuck: let it go
                    lk.acquire()
                OL["since"] = time.time()
                epoch = state["epoch"]
                try:
                    feed_add("order", text)
                    if not ag_lock.acquire(timeout=20):
                        return self.send(429, {"ok": False, "message": "Nemotron is still answering. Try again."})
                    try:
                        rec = AG.decide(text)
                    finally:
                        ag_lock.release()
                    if epoch != state["epoch"]:
                        return self.send(200, {"ok": False, "message": "The demo was reset while Nemotron was "
                                                                       "thinking. Send the order again."})
                    rec["t_sim"] = round(state["t"], 2)
                    notes = perception_notes(rec)
                    for s in notes:
                        feed_add("perception", s)
                    pre = "".join(f"Perception: {s}\n" for s in notes)
                    if rec.get("tool_call") is None:
                        rec.pop("_turn", None)
                        reply = (rec.get("reply") or "").strip()
                        if reply:
                            with ag_lock:
                                AG.HISTORY.extend([{"role": "user", "content": text},
                                                   {"role": "assistant", "content": reply}])
                                trim_history()
                            feed_add("reply", reply)
                        else:
                            errs = [s.get("error") for s in rec.get("steps", []) if s.get("error")]
                            reply = ("Nemotron did not answer (" + errs[-1][:80] + ")") if errs else \
                                "Nemotron did not pick an action for that. Try 'Everyone, gather in the centre circle.'"
                            feed_add("reply", reply)
                        jlog("keynote_live_orders.jsonl", rec)
                        return self.send(200, {"ok": True, "message": pre + "Nemotron: " + reply, "reply": reply,
                                               "latency_s": rec.get("latency_s")})
                    box = queue.Queue()
                    cmds.put(({"action": "order_exec", "rec": rec, "text": text}, box))
                    try:
                        res = box.get(timeout=5)
                    except queue.Empty:
                        return self.send(503, {"ok": False, "message": "simulation busy"})
                    with ag_lock:
                        AG.remember(rec, res["note"])
                        trim_history()
                    rec.update(navigator=res["note"], live=True)
                    feed_add("tool", f"{rec['tool_call_str']}  ->  {res['note']}")
                    jlog("keynote_live_orders.jsonl", rec)
                    print(f"[keynote] order {text!r} -> {rec.get('perception_calls')} {rec['tool_call_str']} "
                          f"({rec.get('latency_s')} s)", flush=True)
                    return self.send(200, {"ok": True, "tool_call": rec["tool_call_str"],
                                           "navigator": pre + "Navigator: " + res["note"],
                                           "message": rec["tool_call_str"], "latency_s": rec.get("latency_s"),
                                           "perception_calls": rec.get("perception_calls", [])})
                except Exception as ex:  # noqa: BLE001
                    feed_add("reply", f"(order failed: {type(ex).__name__})")
                    return self.send(500, {"ok": False, "message": f"Order failed: {type(ex).__name__}: {ex}"[:200]})
                finally:
                    lk.release()

        seq_box = [0]
        srv = ThreadingHTTPServer(("127.0.0.1", args.ctrl_port), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[keynote] live: poses -> {args.shm}, controls on 127.0.0.1:{args.ctrl_port}", flush=True)

        # ------------------------------------------------------------------ main loop
        perc_every = max(1, int(round(1.0 / (args.perc_hz * CTRL_DT))))
        pending_check = None
        perc = None
        seq = 0
        f = 0
        nxt = time.perf_counter()
        t_start = time.time()
        prof = np.zeros(7)

        def mode_words():
            if not director.ready:
                return "mapping the yard"
            if director.mode == "wander":
                return "wandering"
            if director.mode == "stop":
                return "standing still"
            return GOAL_WORDS.get(director.goal_kind, "waiting for an order")

        def fire_check(t):
            nonlocal pending_check
            if pending_check is None or perc is None:
                return
            chk = director.verify(t, perc)
            if chk is not None:
                answer_async(chk, pending_check["order"])
            pending_check = None

        while True:
            st0 = time.perf_counter()
            t = state["t"]
            while not cmds.empty():
                data, box = cmds.get()
                try:
                    a = str(data.get("action", ""))
                    if data.get("duck") is not None:
                        state["selected"] = int(np.clip(int(data["duck"]) - 1, 0, Y.NDUCK - 1))
                except Exception:  # noqa: BLE001
                    a = ""
                msg, extra = "ok", {}
                if a == "order_exec":
                    try:
                        rec = data["rec"]
                        name = rec["tool_call"]["name"]
                        fire_check(t)
                        if name != "push_duck":
                            pending.clear()
                        ev, note = AG.to_events(rec["tool_call"], t, director)
                        pending.extend(ev)
                        pending.sort(key=lambda e: e[0])
                        if name not in ("resume_wander", "push_duck"):
                            wait = director.check_after or CHECK_WAIT.get(name, 16.0)
                            pending_check = dict(order=data["text"], deadline=t + wait)
                    except Exception as ex:  # noqa: BLE001
                        state["loop_errors"] += 1
                        print(f"[keynote] order_exec failed: {type(ex).__name__}: {ex}", flush=True)
                        note = f"the navigator could not run that ({type(ex).__name__}); nothing executed"
                    box.put(dict(ok=True, note=note))
                    continue
                if not director.ready and a not in ("select",):
                    box.put(dict(ok=False, message="The cameras are still mapping the yard. One moment."))
                    continue
                if a in ("reset", "scatter", "walk", "wander", "stop"):
                    pending.clear()
                    pending_check = None
                if a in ("reset", "scatter"):
                    # soft reset: every order forgotten (navigator, Nemotron's memory, the feed), ducks on their
                    # keynote start spots; the perceived map stays
                    state["epoch"] += 1
                    director.soft_reset()
                    AG.HISTORY[:] = []
                    checks.clear()
                    feed.clear()
                    OL["lock"], OL["since"] = threading.Lock(), 0.0
                    for i, p in START.items():
                        respawn(i, np.array(p), float(rng.uniform(-math.pi, math.pi)))
                    feed_add("reply", "Reset: ducks back on their keynote start spots. Nemotron starts fresh.")
                    print(f"[keynote] soft reset epoch {state['epoch']}", flush=True)
                    msg = "Ducks back on their keynote start spots. Type an order for Nemotron."
                elif a in ("walk", "wander", "ramps"):
                    for e in director.agent_wander(t):
                        director.apply(e, t)
                    msg = "Everyone wanders around the yard."
                elif a == "stop":
                    director.busy, director.lined = set(), []
                    ev, _ = director.agent_stop(t)
                    pending.extend(ev)
                    pending.sort(key=lambda e: e[0])
                    msg = "Everyone stands still."
                elif a == "push":
                    push(state["selected"], 1.0)
                    msg = f"Duck {state['selected'] + 1} got a shove."
                elif a == "select":
                    msg = f"Duck {state['selected'] + 1} selected."
                box.put(dict(ok=True, message=msg, mode=mode_words(), selected=state["selected"] + 1, **extra))

            pq = [time.perf_counter()]
            g, w, jp, jv = gather()
            upz = -g[:, 2]
            upright = director.update_falls(t, upz)
            pq.append(time.perf_counter())
            if f % perc_every == 0:
                for c in cams.values():
                    c.update(CTRL_DT * perc_every)
                perc, frames = perceive(t)
                if t >= args.map_secs and not director.ready:
                    m = mapper.finalize()
                    director.set_map(m)
                    print(f"[keynote] map ready t={t:.2f}: {json.dumps(m.summary())[:600]}", flush=True)
                    feed_add("reply", "The 4 cameras mapped the yard: walls, cones, planters and 3 ramps.")
                    if args.start == "wander":
                        for e in director.agent_wander(t):
                            director.apply(e, t)
                director.POS, director.YAW = perc["pos"], perc["yaw"]
                if director.ready:
                    try:
                        director.steer(t, perc["pos"], perc["yaw"], upright)
                        if pending_check is not None:
                            chk = director.verify(t, perc)
                            late = t > pending_check["deadline"]
                            if late and chk is not None and chk["ok"] < chk["n"] and not pending_check.get("extended"):
                                pending_check["extended"] = True      # one grace period for a straggler
                                pending_check["deadline"] = t + 15.0
                                late = False
                            if chk is not None and (chk["ok"] == chk["n"] or late):
                                fire_check(t)
                        shared["summary"] = perceived_summary(perc, t)
                    except Exception as ex:  # noqa: BLE001
                        state["loop_errors"] += 1
                        director.target[:] = 0
                        if state["loop_errors"] < 50:
                            import traceback
                            print(f"[keynote] steer failed: {type(ex).__name__}: {ex}\n{traceback.format_exc()[-800:]}",
                                  flush=True)
                # camera copies only when somebody looks (Nemotron's look tool or a live view)
                for name, ev in list(shared["rgb_req"].items()):
                    out = None
                    for cs, o in frames:
                        if cs.name == name:
                            out = o
                    if out is not None:
                        shared["rgb"][name] = T(out["rgb"])[0].cpu().numpy()[..., :3].copy()
                    shared["rgb_req"].pop(name, None)
                    ev.set()
                if time.time() < shared["eye_until"]:
                    for cs, o in frames:
                        if cs.name == shared["eye"]:
                            shared["eye_jpg"] = (cs.name, T(o["rgb"])[0].cpu().numpy()[..., :3].copy())
                if time.time() < shared["map_until"] and director.ready:
                    paths = {}
                    for i in range(Y.NDUCK):
                        nv = director.nav[i]
                        if nv.get("kind") == "path" and nv.get("pts") and np.all(np.isfinite(perc["pos"][i])):
                            paths[i] = [tuple(perc["pos"][i])] + [tuple(p) for p in nv["pts"]]
                    slots = [list(map(float, gg)) for gg, _ in director.goals.values()]
                    shared["map_snap"] = (perc["pos"].copy(), perc["yaw"].copy(), paths, slots, t)
                if len(director.paths_log) > 200:
                    del director.paths_log[:-100]
            while pending and pending[0][0] <= t + 1e-9 and director.ready:
                try:
                    req = director.apply(pending.pop(0), t)
                    if req is not None:
                        push(*req)
                except Exception as ex:  # noqa: BLE001
                    state["loop_errors"] += 1
                    print(f"[keynote] event failed: {type(ex).__name__}: {ex}", flush=True)
            pq.append(time.perf_counter())
            obs = np.zeros((Y.NDUCK, 61))
            obs[:, 0:3], obs[:, 3:6], obs[:, 6:20], obs[:, 20:34] = w, g, jp, jv
            obs[:, 34:48] = state["last"]
            obs[:, 48:51] = director.smooth()
            act = np.clip(policy(obs.astype(np.float32)), -3.0, 3.0)
            state["last"] = act
            act_t = torch.as_tensor(act, dtype=torch.float32, device=dev)
            for i, d in enumerate(ducks):
                tgt = d.data.default_joint_pos.torch.clone()
                tgt[0, jids_t] += act_t[i].to(tgt.dtype)
                d.set_joint_position_target_index(target=tgt)
            pq.append(time.perf_counter())
            scene.write_data_to_sim()
            pq.append(time.perf_counter())
            sim.step(render=False)
            pq.append(time.perf_counter())
            for d in ducks:
                d.update(CTRL_DT)
            pq.append(time.perf_counter())
            state["t"] = t + CTRL_DT
            f += 1
            poses = body_poses()
            pq.append(time.perf_counter())
            prof[:] = 0.98 * prof + 0.02 * np.diff(pq) * 1000
            # respawn supervisor (simulation housekeeping, not navigation): out of the yard or down for 12 s
            for i in range(Y.NDUCK):
                rp = poses[i, 0, :3]
                out_ = abs(rp[0]) > Y.YARD or abs(rp[1]) > Y.YARD or not np.isfinite(rp).all() or rp[2] < -0.1
                down = director.fallen_since[i] >= 0 and state["t"] - director.fallen_since[i] > 12.0
                if out_ or down:
                    others = [poses[j, 0, :2] for j in range(Y.NDUCK) if j != i]
                    respawn(i, director.free_spot(others) if director.ready else np.array(START[i]),
                            float(rng.uniform(-math.pi, math.pi)))
                    state["respawns"] += 1
            seq += 1
            seq_box[0] = seq
            mm[0] = -seq
            mm[1] = state["t"]
            mm[HDR:] = poses[:, order].astype(np.float64).reshape(-1)
            mm[0] = seq
            tim["step"].append(time.perf_counter() - st0)
            if seq % 10 == 0:
                step_s = float(np.mean(tim["step"]))
                pos = perc["pos"] if perc is not None else np.full((Y.NDUCK, 2), np.nan)
                info.update(dict(
                    up=int(np.sum(upz > math.cos(math.radians(55)))), epoch=state["epoch"],
                    sim_time=round(state["t"], 1), mode=mode_words(), selected=state["selected"] + 1,
                    respawns=state["respawns"], stack=stack, step_ms=round(1000 * step_s, 1),
                    rtf=round(CTRL_DT / max(step_s, 1e-6), 2), map_ready=bool(director.ready),
                    perception=dict(render_ms=round(float(np.mean(tim["render"])), 1) if tim["render"] else None,
                                    perceive_ms=round(float(np.mean(tim["perceive"])), 1) if tim["perceive"] else None,
                                    cameras=[cs.name for cs in P.RIG], positions="perceived (cameras), not simulator"),
                    ducks=[dict(id=i + 1, color="#%02x%02x%02x" % tuple(int(255 * v) for v in Y.DUCK_COLORS[i]),
                                x=round(float(pos[i][0]), 2) if np.isfinite(pos[i][0]) else None,
                                y=round(float(pos[i][1]), 2) if np.isfinite(pos[i][1]) else None,
                                upright=bool(upz[i] > math.cos(math.radians(55))),
                                cmd=[round(float(v), 2) for v in director.cmd[i]]) for i in range(Y.NDUCK)]))
            if seq % 250 == 0 and os.environ.get("KEYNOTE_PROF"):
                print("[keynote] ms gather %.1f perc %.1f policy+targets %.1f write %.1f step %.1f update %.1f poses %.1f"
                      % tuple(prof), flush=True)
            if seq % 1500 == 0:
                print(f"[keynote] t={state['t']:7.1f}s wall {time.time() - t_start:7.1f}s up={int(np.sum(upz > 0.57))}/12 "
                      f"step {info.get('step_ms')} ms rtf {info.get('rtf')} cams render "
                      f"{info['perception']['render_ms']} ms perceive {info['perception']['perceive_ms']} ms "
                      f"mode {mode_words()} respawns {state['respawns']}", flush=True)
            if args.seconds and state["t"] >= args.seconds:
                print("[keynote] test done", json.dumps({k_: info.get(k_) for k_ in ("step_ms", "rtf", "perception")}),
                      flush=True)
                os._exit(0)
            nxt += CTRL_DT
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            elif dt < -0.5:
                nxt = time.perf_counter()


if __name__ == "__main__":
    main()
