# OAK-D workstation with CASCADE

This workstation uses a Seeed B601-RS, a fixed OAK-D-W-97
(serial `19443010D1BCB22F00`), and a Sonix camera/holder on arm link 4.
The OAK supplies manipulation RGB-D; the Sonix holder and cables are included
in the robot geometry. Astra runs through CASCADE's existing OpenAI-compatible
provider and BrainDetector/SAM perception pipeline.

## Current capabilities

The standard interactive CLI executes home moves and grasp attempts on the
physical arm. An attended pickup has been demonstrated. Grasp reliability is
not established: the latest recorded attempt missed the box, and its planned
jaw orientation is under investigation. The selected backend is the existing
analytic OBB planner (`grasp.backend: obb`), not a learned grasp model.

Live-camera grasp verification uses fresh, memory-disabled localization before
and after the attempt, with physics truth preferred when available. It has
correctly refuted a physical miss. Automatic confirmation of a successful
physical pickup with this verification path, and a complete pick-and-place
cycle, remain unvalidated. Missing, stale, inferred-depth or incomparable
observations return `unverified`; the measured lift criterion is 15 mm.

Recorded evidence remains under `runs/`: `rebot-oak-run-w11wjefc` contains
the demonstrated pickup and video review; `rebot-oak-run-t6mykgib` contains
the missed grasp and fresh verification observations. These are local run
artifacts, not prerequisites for installation.

## Installation and configuration

From the checkout, `bash scripts/rebot_oak.sh setup` creates the workstation's
`.venv-rebot` environment with the package extras and pinned hardware packages.
The general OAK adapter is available through the optional `camera-oak` extra.
Torch must match the host, and SAM weights must be available at
`models/sam2.1_l.pt`. The Seeed SDK lives at the arm profile's `sdk_path`.

- `configs/cameras/oak_d.yaml` selects the camera serial, native stereo settings,
  paired sensor/hand-eye calibration, drift monitor and detector.
- `configs/arms/rebot_rs_oak.yaml` extends `rebot_rs` with the installed model
  and Sonix/cable geometry. It inherits gains, gravity compensation, home/park
  poses, joint speeds, contact monitoring and settling policy.
- `configs/llm/inferencehub_astra.yaml` selects the existing Astra provider.
  The launcher reads the key from `~/.config/cascade/inferencehub_api_key`;
  credentials are not stored in profiles or recordings.

The launcher calls `cascade.apps.demo.main` with
`--arm rebot_rs_oak --camera oak_d --llm inferencehub_astra --interactive`.
It owns one ephemeral instance of the existing Warp occupancy bridge on
`cuda:0`, retained until CASCADE finishes shutdown. It supplies no separate
controller, trajectory or agent loop.

## Operation

Run these checks with control applications closed:

```bash
bash scripts/rebot_oak.sh status
bash scripts/rebot_oak.sh check
```

`status` prepares/verifies the local model and resolves configuration and
calibration without opening hardware. `check` opens the OAK and reads encoders
without enabling motors. It checks arm/depth alignment, builds masked maps and
assesses the standard home route through CASCADE's existing planner. It also
reports persistent hold markers without clearing them. CAN must already be up
at 1 Mbps; no other controller may own the bus during this check.

In the operator's workstation terminal:

```bash
bash scripts/rebot_oak.sh run
```

**`run` enables the arm.** Start with the arm supported/resting and the workspace
clear. At `task>`, enter tasks such as `go to initial pose`. The runtime checks
fresh geometry again; a saved preflight pass does not authorize later motion.

If startup reports a retained hold, support the arm and use CASCADE's existing
operator confirmation. Run the commands separately, within the confirmation's
120-second window:

```bash
.venv-rebot/bin/python scripts/arm_hold.py confirm --arm can0
bash scripts/rebot_oak.sh run
```

Do not join them with `&&`: the confirmation command can return a nonzero
status while reporting a held arm even after confirmation succeeds. Keep
support through reconnect. Once the CLI reports that the arm is connected and
holding, clear your hands and enter exactly `go` at its readiness prompt.
Enter task text only after `task>` appears.

Normal exit can request a park move. If parking is refused, CASCADE retains
torque and records the hold for the next startup. Follow its support/power-off
instructions; do not remove hold markers manually.

## Camera calibration

The active pair is in `configs/calib/oak_sensor_corrected_20261031/`:

- `sensor.json`: DepthAI sensor calibration, loaded into device RAM on open.
  The adapter never writes EEPROM. A requested missing/invalid file refuses
  startup rather than falling back to another calibration.
- `oak_d_rebot_rs.handeye.json`: accepted camera-to-base transform, bound to
  this serial and mounting through the existing calibration loader.

The six-pose fit has 3.5 mm surface RMS. Independent resting and wrist-down
captures, excluded from fitting, measured 2.5–7.2 mm surface displacement and
at most 1.85 degrees, within the runtime limits of 10 mm / 3 degrees.
The existing passive drift monitor is enabled with `auto_apply: false`.

Moving the camera/base, changing encoder zeros or changing sensor calibration
requires revalidation. Use `scripts/capture_rebot_oak_calibration.py` for
supported, disabled-arm samples and `scripts/fit_rebot_oak_calibration.py` for
the existing markerless solver and record validator. Their `--help` lists
arguments. Keep independent validation captures outside the fitter's pose set.
Datasets reject mixed sensor-calibration fingerprints. The retained calibration
dataset is `runs/oak_sensor_recalibration_20261031/`.

## Local adaptations and limits

These pending extensions support this rig; the checkout is not unmodified upstream.

| Extension | Requirement served |
|---|---|
| OAK `CameraBase` adapter and factory branch | Undistorted BGR, aligned metric depth, capture timestamps, serial binding and native DepthAI configuration; UVC alone cannot provide these. |
| Passive encoder reader and capture/preflight tools | Joint/depth observations without calling the actuator's enabling `connect()`. |
| Link-frame attachment checks | The Sonix follows link 4, so the parent's TCP-mounted D455F model and orientation rule do not describe it. Shared attachment data feeds masking and collision checks. |
| Driver error propagation and startup preload | Report failures suppressed by SDK group sends and preload measured positions before enable. Transmission success is not proof of physical arrival. |
| Passive lifecycle reads | Masking and drift checks must not reconnect an arm during teardown. |
| Capture-time mask state | Select the retained command in effect when an image was captured; missing history refuses integration. This is a target-based pose approximation, not independent feedback. Idle images use measured state. |
| Fresh grasp verification | Compare measured observations rather than a cached belief frozen during motion. |
| Existing OBB upper-section correction | Prevent a noisy high patch from treating a flat box as tapered and lowering the grasp toward the table. This patch does not change yaw selection. |

The standard 30 mm obstacle margin remains in force. Sonix boxes and the
upper-wrist motor cable envelope are checked alongside housing/fingers.
Coverage of flexible cables throughout the full workspace and the error of
command-based masking outside the exercised poses are not established.

## Diagnostic recordings

Each run gets its own `runs/rebot-oak-run-*` directory. Keep these recordings
when investigating a failure:

- `trace.jsonl`, task summaries and keyframes: standard CASCADE tool execution
  and postcondition results.
- `grasp_evidence/`: localized RGB-D/masks, point clouds, candidate grasps,
  IK targets, selection results and source/configuration receipts.
- `grasp_evidence/verification/`: fresh before/after observations linked from
  the postcondition trace.
- `safety_evidence/`: retained depth and mask inputs linked from clearance
  reports. Each snapshot is the latest committed input, not complete TSDF history.
- `extrinsics_evidence/`: the depth, measured joints and transforms from a
  drift check that invalidated fusion; these are not replacement calibrations.

Evidence capture adds no robot commands. Map snapshots are written after the
skill returns, outside the motor loop. Missing snapshots and write failures
are reported rather than changing motion or calibration decisions.
