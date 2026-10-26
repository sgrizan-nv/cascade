"""12 Microducks in one Isaac Lab + Newton (MJWarp) world: the schoolyard, one ONNX walking/recovery policy, scripted
velocity commands. Records a trace (all body poses per control tick) for the OVRTX re-render.

    python isaac_ducks_sim.py --kit --seconds 48 --seed 3 --out takes/isaac_s3   [--servo bam|pd]

Policy contract (from the ONNX metadata): obs 61 = base_ang_vel_b 3, projected_gravity_b 3, joint_pos - default 14,
joint_vel 14, last raw action 14, command 13 (vx, vy, wz, head 4 = 0, body 6 = 0); action 14 -> target = default +
1.0 * action, 50 Hz (dt 5 ms, decimation 4). Joint order: the policy's, mapped by name.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=48.0)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--hero", type=int, default=3)
parser.add_argument("--schedule", default="a")
parser.add_argument("--out", default="")
parser.add_argument("--servo", choices=("auto", "bam", "pd"), default="auto")
parser.add_argument("--balls", action="store_true")
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--ls-iters", type=int, default=50)
parser.add_argument("--njmax", type=int, default=2400)
parser.add_argument("--nconmax", type=int, default=600)
parser.add_argument("--fast", action="store_true", help="one sim.step per 50 Hz tick, Newton does 4 substeps")
parser.add_argument("--live", action="store_true", help="run forever in real time, publish poses to shared memory")
parser.add_argument("--shm", default="/dev/shm/microducks_pose.bin")
parser.add_argument("--ctrl-port", type=int, default=18201)
parser.add_argument("--kit", action="store_true", help="launch Isaac Sim Kit alongside Newton (require_kit)")
parser.add_argument("--asset", default=os.path.join(HERE, "assets", "microduck_allcollisions_isaac.usda"))
parser.add_argument("--policy", default=os.path.join(HERE, "policy_np.npz"))
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
CTRL_DT, DECIM = 0.02, 4


def numpy_policy(path):
    """The exported MLP (Sub, Div, Gemm, Elu) from policy_np.npz (weights extracted from the ONNX, verified 3e-5)."""
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
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import instantiate
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    import director as D
    import yard_layout as Y

    servo = args.servo
    bam_cfg = None
    if servo in ("auto", "bam"):
        try:
            from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
            motor = BamMotorCfg(model="m6", kt=0.36601349688984386, resistance=2.8113923539223227,
                                error_gain=0.0028773775022263564, max_pwm=1.0, max_current=1.75,
                                friction_base=0.004771183165566, friction_viscous=0.005359668274599504,
                                friction_stribeck=0.004676345799486616, dtheta_stribeck=2.890372094130307,
                                alpha=8.683259907618984, load_friction_motor=0.2667860954283698,
                                load_friction_external=8.515871897059342e-06,
                                load_friction_motor_stribeck=1.0722918395099123e-05,
                                load_friction_external_stribeck=0.08077928978935671,
                                load_friction_motor_quad=0.009972471242139415,
                                load_friction_external_quad=0.004902565732332559)
            bam_cfg = BamActuatorCfg(joint_names_expr=[".*"], motor=motor, kp_fw=200.0, vin=7.4, vin_min=6.0,
                                     min_delay=3, max_delay=6)
            servo = "bam"
        except Exception as e:  # noqa: BLE001
            if servo == "bam":
                raise
            print("[ducks] BAM actuator not available in this Isaac Lab:", type(e).__name__, e, flush=True)
            servo = "pd"
    if servo == "pd":
        # first-order servo derived from the BAM XL330 m6 fit at kp_fw 200: kp 0.55 Nm/rad, kt^2/R + viscous ~ 0.053
        act_cfg = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.55, damping=0.053, effort_limit_sim=0.96,
                                      armature=0.0018, friction=0.0048)
    else:
        act_cfg = bam_cfg

    global DECIM
    fast = args.fast or args.live
    nsub = 1
    if fast:
        nsub, DECIM = DECIM, 1
    sim_kwargs = dict(dt=CTRL_DT / DECIM, render_interval=DECIM, device=args.device or "cuda:0",
                      physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=args.njmax, nconmax=args.nconmax, iterations=args.iters,
                                                                   ls_iterations=args.ls_iters, cone="pyramidal", impratio=1.0,
                                                                   integrator="implicitfast",
                                                                   use_mujoco_contacts=False),
                                        num_substeps=nsub))
    sim_cfg = sim_utils.SimulationCfg(**sim_kwargs)
    if hasattr(sim_cfg, "use_newton_actuators"):
        sim_cfg.use_newton_actuators = servo == "bam"

    director = D.Director(args.seed, args.hero - 1)
    place, events = director.schedule(args.schedule)

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0, replicate_physics=True)
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)))
    scene_cfg.light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    col = sim_utils.CollisionPropertiesCfg()
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8)
    k = 0
    for r in Y.RAMPS:
        for name, c, hs, q, role in Y.ramp_boxes(r):
            setattr(scene_cfg, f"yard_{k:02d}_{name}", AssetBaseCfg(
                prim_path=f"/World/Yard/{name}", spawn=sim_utils.CuboidCfg(size=tuple(2 * v for v in hs),
                                                                            collision_props=col,
                                                                            physics_material=mat),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(c), rot=tuple(q))))
            k += 1
    for name, c, hs in Y.wall_boxes():
        setattr(scene_cfg, f"yard_{k:02d}_{name}", AssetBaseCfg(
            prim_path=f"/World/Yard/{name}", spawn=sim_utils.CuboidCfg(size=tuple(2 * v for v in hs),
                                                                        collision_props=col, physics_material=mat),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(c))))
        k += 1
    for i, (x, y) in enumerate(Y.CONES):
        setattr(scene_cfg, f"yard_{k:02d}_cone{i}", AssetBaseCfg(
            prim_path=f"/World/Yard/cone{i}", spawn=sim_utils.CylinderCfg(radius=Y.CONE_R, height=Y.CONE_H,
                                                                           collision_props=col, physics_material=mat),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, Y.CONE_H / 2))))
        k += 1
    for i, (x, y, yaw) in enumerate(Y.PLANTERS):
        setattr(scene_cfg, f"yard_{k:02d}_planter{i}", AssetBaseCfg(
            prim_path=f"/World/Yard/planter{i}", spawn=sim_utils.CuboidCfg(size=Y.PLANTER_SIZE, collision_props=col,
                                                                            physics_material=mat),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, Y.PLANTER_SIZE[2] / 2),
                                                    rot=Y.quat_xyzw_from_yaw_pitch(yaw))))
        k += 1
    for i, (x, y) in enumerate(Y.BALLS if args.balls else []):
        setattr(scene_cfg, f"ball{i}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Ball{i}", spawn=sim_utils.SphereCfg(
                radius=Y.BALL_R, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03), collision_props=col,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, Y.BALL_R + 0.001))))
    art_props = None
    try:
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg
        art_props = NewtonArticulationCfg(self_collision_enabled=False)
    except Exception:  # noqa: BLE001
        pass
    for i in range(Y.NDUCK):
        p, yaw = place[i]
        spawn = sim_utils.UsdFileCfg(usd_path=args.asset, activate_contact_sensors=False)
        if art_props is not None:
            spawn.articulation_props = art_props
        setattr(scene_cfg, f"duck{i:02d}", ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Duck{i:02d}", spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(float(p[0]), float(p[1]), 0.125),
                                                       rot=Y.quat_xyzw_from_yaw_pitch(float(yaw)),
                                                       joint_pos=dict(HOME), joint_vel={".*": 0.0}),
            soft_joint_pos_limit_factor=0.9, actuators={"servos": act_cfg}))

    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)
        if hasattr(sim.cfg, "use_newton_actuators"):
            print("[ducks] use_newton_actuators before:", sim.cfg.use_newton_actuators, flush=True)
            sim.cfg.use_newton_actuators = servo == "bam"
        scene = instantiate(scene_cfg)
        sim.reset()
        stack = dict(servo=servo, kit=bool(args.kit), stepping=("newton 4 substeps / tick" if fast else "4 x 5 ms steps / tick"))
        try:
            import isaacsim
            stack["isaacsim"] = getattr(isaacsim, "__version__", None)
        except Exception:  # noqa: BLE001
            stack["isaacsim"] = None
        try:
            import importlib.metadata as md
            for pkg in ("isaacsim", "isaaclab", "isaaclab_newton", "newton", "warp-lang"):
                try:
                    stack[pkg + "_dist"] = md.version(pkg)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            stack["kit_app"] = app.get_build_version()
            for attr in ("get_app_name", "get_app_version"):
                try:
                    stack["kit_" + attr[4:]] = getattr(app, attr)()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            stack["kit_app"] = f"not running ({type(e).__name__})"
        for env in ("ISAAC_PATH", "ISAACSIM_PATH", "CARB_APP_PATH", "EXP_PATH"):
            if os.environ.get(env):
                stack["env_" + env] = os.environ[env]
                for cand in (os.environ[env], os.path.dirname(os.environ[env])):
                    vf = os.path.join(cand, "VERSION")
                    if os.path.exists(vf) and "isaacsim_VERSION_file" not in stack:
                        stack["isaacsim_VERSION_file"] = open(vf).read().strip()
        print("[ducks] stack", json.dumps(stack), flush=True)

        ducks = [scene[f"duck{i:02d}"] for i in range(Y.NDUCK)]
        balls = [scene[f"ball{i}"] for i in range(len(Y.BALLS))] if args.balls else []
        jids = ducks[0].find_joints(JOINTS, preserve_order=True)[0]
        body_names = list(ducks[0].body_names)
        print("[ducks] joints(sim order)", list(ducks[0].joint_names), "bodies", body_names, flush=True)
        policy, meta = numpy_policy(args.policy)
        stack["physics"] = "newton"
        stack["policy"] = json.loads(meta.get("label", '"?"'))
        dev = ducks[0].data.joint_pos.torch.device
        jids_t = torch.as_tensor(jids, device=dev)
        state = dict(last=np.zeros((Y.NDUCK, 14)), t=0.0, respawns=0, selected=args.hero - 1, rtf=0.0)
        up_prev = np.ones(Y.NDUCK, bool)
        fall_t = np.full(Y.NDUCK, -1.0)
        log = dict(stack=stack, seed=args.seed, hero=args.hero, schedule=args.schedule, pushes=[], falls=[],
                   getups=[], respawns=[], policy=stack["policy"])

        def gather():
            g = torch.stack([d.data.projected_gravity_b.torch[0] for d in ducks])
            w = torch.stack([d.data.root_ang_vel_b.torch[0] for d in ducks])
            jp = torch.stack([d.data.joint_pos.torch[0] - d.data.default_joint_pos.torch[0] for d in ducks])[:, jids_t]
            jv = torch.stack([d.data.joint_vel.torch[0] for d in ducks])[:, jids_t]
            rp = torch.stack([d.data.root_pos_w.torch[0] for d in ducks])
            hd = torch.stack([d.data.heading_w.torch[0].reshape(()) for d in ducks])
            return [x.cpu().numpy() for x in (g, w, jp, jv, rp, hd)]

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
            log["pushes"].append(dict(t=round(state["t"], 2), duck=i + 1))

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
            director.fallen_since[i] = -1.0
            director.nav[i] = dict(kind="pause", until=state["t"] + 1.0)
            up_prev[i] = True

        prof = np.zeros(5)

        def control_step():
            st0 = time.perf_counter()
            g, w, jp, jv, rp, hd = gather()
            director.POS = np.array(rp[:, :2])
            pa = time.perf_counter()
            obs = np.zeros((Y.NDUCK, 61))
            obs[:, 0:3], obs[:, 3:6], obs[:, 6:20], obs[:, 20:34] = w, g, jp, jv
            obs[:, 34:48] = state["last"]
            upz = -g[:, 2]
            t = state["t"]
            upright = director.update_falls(t, upz)
            if int(round(t / CTRL_DT)) % 5 == 0:
                director.steer(t, rp[:, :2], hd, upright)
            obs[:, 48:51] = director.smooth()
            act = np.clip(policy(obs.astype(np.float32)), -3.0, 3.0)
            pb = time.perf_counter()
            state["last"] = act
            act_t = torch.as_tensor(act, dtype=torch.float32, device=dev)
            for i, d in enumerate(ducks):
                tgt = d.data.default_joint_pos.torch.clone()
                tgt[0, jids_t] += act_t[i].to(tgt.dtype)
                d.set_joint_position_target_index(target=tgt)
            pc = time.perf_counter()
            tw = ts = tu = 0.0
            for _ in range(DECIM):
                q0 = time.perf_counter()
                scene.write_data_to_sim()
                q1 = time.perf_counter()
                sim.step(render=False)
                q2 = time.perf_counter()
                scene.update(CTRL_DT / DECIM)
                q3 = time.perf_counter()
                tw += q1 - q0
                ts += q2 - q1
                tu += q3 - q2
            prof[:] = 0.98 * prof + 0.02 * np.array([pa - st0, pb - pa, pc - pb, tw + tu, ts]) * 1000
            if int(round(t / CTRL_DT)) % 250 == 0:
                print("[ducks] ms gather %.1f policy %.1f targets %.1f write+update %.1f step %.1f" % tuple(prof), flush=True)
            state["t"] = t + CTRL_DT
            for i in range(Y.NDUCK):
                up_i = upz[i] > math.cos(math.radians(55))
                if up_prev[i] and not up_i:
                    fall_t[i] = state["t"]
                    log["falls"].append(dict(t=round(state["t"], 2), duck=i + 1))
                    up_prev[i] = False
                elif not up_prev[i] and upz[i] > 0.9:
                    log["getups"].append(dict(t=round(state["t"], 2), duck=i + 1, after=round(state["t"] - fall_t[i], 2)))
                    up_prev[i] = True
            return rp, upz, (time.perf_counter() - st0)

        if args.live:
            run_live(locals())
            return

        events_l = list(events)
        agent_log = []
        n = int(args.seconds / CTRL_DT)
        T_pose = np.zeros((n, Y.NDUCK, len(body_names), 7), np.float32)
        T_cmd = np.zeros((n, Y.NDUCK, 3), np.float32)
        T_up = np.zeros((n, Y.NDUCK), np.float32)
        ei = 0
        t0 = time.time()
        for f in range(n):
            while ei < len(events_l) and events_l[ei][0] <= state["t"] + 1e-9:
                if events_l[ei][1] == "order":
                    import agent
                    new = agent.handle(events_l[ei][2], state["t"], director, agent_log)
                    print(f"[ducks] order t={state['t']:.2f} {events_l[ei][2]!r} -> {agent_log[-1].get('tool_call_str')}", flush=True)
                    ei += 1
                    events_l = events_l[:ei] + sorted(events_l[ei:] + list(new), key=lambda e: e[0])
                    continue
                req = director.apply(events_l[ei], state["t"])
                ei += 1
                if req is not None:
                    push(*req)
            rp, upz, _ = control_step()
            T_pose[f] = body_poses()
            T_cmd[f] = director.cmd
            T_up[f] = upz
            if f % 250 == 0:
                h = args.hero - 1
                print(f"[ducks] t={state['t']:5.1f}s wall {time.time() - t0:5.1f}s up={int(np.sum(upz > 0.57))}/12 "
                      f"hero=({rp[h][0]:.2f},{rp[h][1]:.2f})", flush=True)
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "trace.npz"), body_pose_xyzw=T_pose, cmd=T_cmd, upz=T_up,
                            body_names=np.array(body_names), ctrl_dt=CTRL_DT)
        log["wall_s"] = round(time.time() - t0, 1)
        log["summary"] = dict(falls=len(log["falls"]), getups=len(log["getups"]), pushes=len(log["pushes"]))
        path = np.sum(np.linalg.norm(np.diff(T_pose[::25, :, 0, :2], axis=0), axis=-1), axis=0)
        log["path_m"] = [round(float(v), 2) for v in path]
        with open(os.path.join(args.out, "events.json"), "w") as fh:
            json.dump(log, fh, indent=1)
        if agent_log:
            with open(os.path.join(args.out, "agent_log.jsonl"), "w") as fh:
                for r in agent_log:
                    fh.write(json.dumps(r) + "\n")
        print("[ducks] done", json.dumps({k: log[k] for k in ("summary", "path_m", "wall_s")}), flush=True)


def run_live(L):
    """Real-time loop: poses to shared memory, controls over HTTP (127.0.0.1:ctrl-port)."""
    import queue
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import yard_layout as Y
    state, director, log, stack = L["state"], L["director"], L["log"], L["stack"]
    control_step, body_poses, push, respawn = L["control_step"], L["body_poses"], L["push"], L["respawn"]
    body_names = L["body_names"]
    order = [body_names.index(n) for n in ['trunk_base', 'bearing_roll', 'hip_l_2', 'upper_leg_right', 'leg_2',
                                           'ankle_right', 'neck', 'neck_pitch_1', 'yaw_roll_motion', 'jaw_soft',
                                           'yaw2roll', 'hip_l', 'upper_leg_left', 'leg', 'ankle_left']]
    HDR = 16
    tmp = args.shm + ".tmp"
    mm = np.memmap(tmp, dtype=np.float64, mode="w+", shape=(HDR + Y.NDUCK * 15 * 7,))
    os.replace(tmp, args.shm)
    cmds = queue.Queue()
    info = {}
    pending = []                       # timed navigator events from agent tool calls
    order_lock = threading.Lock()
    order_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "live_orders.jsonl")
    director.lined, director.racers, director.busy = [], [], set()
    director.mode = "wander"
    for i in range(Y.NDUCK):
        director.nav[i] = dict(kind="pause", until=1.0)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/info"):
                return self.send(200, dict(info))
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except Exception:  # noqa: BLE001
                data = {}
            if str(data.get("action", "")) == "order":
                text = " ".join(str(data.get("text") or "").split())[:200]
                if not text:
                    return self.send(400, {"ok": False, "message": "Type an order for the ducks first."})
                if not order_lock.acquire(blocking=False):
                    return self.send(429, {"ok": False, "message": "The agent is busy with another order. Try again."})
                try:
                    import agent
                    rec = agent.decide(text)
                    if rec.get("tool_call") is None:
                        with open(order_log, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        return self.send(200, {"ok": False, "message": "The agent did not pick a tool for that. "
                                                                      "Try e.g. 'Regroup in the square'."})
                    box = queue.Queue()
                    cmds.put(({"action": "order_exec", "rec": rec}, box))
                    try:
                        return self.send(200, box.get(timeout=5))
                    except queue.Empty:
                        return self.send(503, {"ok": False, "message": "simulation busy"})
                finally:
                    order_lock.release()
            box = queue.Queue()
            cmds.put((data, box))
            try:
                return self.send(200, box.get(timeout=5))
            except queue.Empty:
                return self.send(503, {"ok": False, "message": "simulation busy"})

    srv = ThreadingHTTPServer(("127.0.0.1", args.ctrl_port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[ducks] live: poses -> {args.shm}, controls on 127.0.0.1:{args.ctrl_port}", flush=True)
    seq = 0
    nxt = time.perf_counter()
    rtf_win = []
    while True:
        while not cmds.empty():
            data, box = cmds.get()
            a = str(data.get("action", ""))
            if data.get("duck") is not None:
                state["selected"] = int(np.clip(int(data["duck"]) - 1, 0, Y.NDUCK - 1))
            msg = "ok"
            extra = {}
            if a in ("reset", "scatter", "walk", "wander", "ramps", "stop"):
                pending.clear()
                director.lined, director.racers, director.busy = [], [], set()
            if a == "order_exec":
                import agent
                rec = data["rec"]
                ev = agent.execute(rec, state["t"], director)
                del agent.HISTORY[:-12]
                pending.extend(ev)
                pending.sort(key=lambda e: e[0])
                rec["live"] = True
                try:
                    with open(order_log, "a") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
                except OSError:
                    pass
                msg = f"{rec['tool_call_str']} -> {rec['navigator']}"
                extra = dict(tool_call=rec["tool_call_str"], navigator=rec["navigator"], latency_s=rec.get("latency_s"))
            elif a in ("reset", "scatter"):
                spots = []
                for i in range(Y.NDUCK):
                    p = director.free_spot(spots)
                    spots.append(p)
                    respawn(i, p, director.rng.uniform(-math.pi, math.pi))
                director.mode = "wander"
                msg = "Ducks scattered across the schoolyard."
            elif a in ("walk", "wander"):
                director.mode = "wander"
                for i in range(Y.NDUCK):
                    director.nav[i] = dict(kind="pause", until=0)
                msg = "Everyone walks."
            elif a == "ramps":
                director.mode = "ramps"
                for i in range(Y.NDUCK):
                    director.nav[i] = dict(kind="pause", until=0)
                msg = "Everyone heads for the ramps."
            elif a == "stop":
                director.mode = "stop"
                msg = "Everyone stands still."
            elif a == "push":
                push(state["selected"], 1.0)
                msg = f"Duck {state['selected'] + 1} got a shove."
            elif a == "select":
                msg = f"Duck {state['selected'] + 1} selected."
            box.put(dict(ok=True, message=msg, mode=director.mode, selected=state["selected"] + 1, **extra))
        while pending and pending[0][0] <= state["t"] + 1e-9:
            req = director.apply(pending.pop(0), state["t"])
            if req is not None:
                push(*req)
        rp, upz, wall = control_step()
        t = state["t"]
        for i in range(Y.NDUCK):
            out = abs(rp[i][0]) > Y.YARD or abs(rp[i][1]) > Y.YARD or not np.isfinite(rp[i]).all() or rp[i][2] < -0.1
            down = director.fallen_since[i] >= 0 and t - director.fallen_since[i] > 12.0
            if out or down:
                others = [rp[j][:2] for j in range(Y.NDUCK) if j != i]
                respawn(i, director.free_spot(others), director.rng.uniform(-math.pi, math.pi))
                state["respawns"] += 1
                log["respawns"].append(dict(t=round(t, 2), duck=i + 1))
        poses = body_poses()[:, order].astype(np.float64)
        seq += 1
        mm[0] = -seq
        mm[1] = t
        mm[HDR:] = poses.reshape(-1)
        mm[0] = seq
        rtf_win.append(wall)
        if len(rtf_win) > 100:
            rtf_win.pop(0)
        if seq % 10 == 0:
            info.update(dict(sim_time=round(t, 1), mode=director.mode, selected=state["selected"] + 1,
                             respawns=state["respawns"], stack=stack,
                             step_ms=round(1000 * float(np.mean(rtf_win)), 1),
                             rtf=round(CTRL_DT / max(float(np.mean(rtf_win)), 1e-6), 2),
                             ducks=[dict(id=i + 1, color="#%02x%02x%02x" % tuple(int(255 * v) for v in Y.DUCK_COLORS[i]),
                                         x=round(float(rp[i][0]), 2), y=round(float(rp[i][1]), 2),
                                         upright=bool(upz[i] > math.cos(math.radians(55))),
                                         cmd=[round(float(v), 2) for v in director.cmd[i]]) for i in range(Y.NDUCK)]))
        nxt += CTRL_DT
        dt = nxt - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        elif dt < -0.5:
            nxt = time.perf_counter()


if __name__ == "__main__":
    main()
