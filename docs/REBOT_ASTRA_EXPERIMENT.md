# B601-RS and Astra in MuJoCo

This optional experiment runs a fixed red-cube scene through CASCADE's existing
runtime, skills, safety harness and OpenAI-compatible client. The reusable
integration is the `rebot_rs_mujoco` arm profile, `rebot_mujoco_scene` camera
profile and `inferencehub_astra` LLM profile. They can also be selected through
the normal `cascade.apps.demo` CLI.

The experiment helpers in `scripts/rebot_astra/` add a plate, request capture,
physical outcome checks and Linux desktop recording. They are not robot-driver
requirements or a new manipulation controller. Both default task prompts describe
the desired outcome without selecting a tool; the ordinary system instructions
and tool descriptions still apply.

## Install and run

From the repository root:

```bash
uv sync --extra dev --extra kinematics --extra llm --extra sim
.venv/bin/python scripts/fetch_robot_assets.py rebot_rs

# Render and save the starting scene; no credential or API call needed.
.venv/bin/python scripts/run_rebot_mujoco_astra.py --inspect-only

# Live Astra: pick up the cube and keep holding it.
.venv/bin/python scripts/run_rebot_mujoco_astra.py

# Live Astra: put the cube on the blue plate and leave the gripper clear.
.venv/bin/python scripts/run_rebot_mujoco_astra.py --plate
```

The experiment reads `~/.config/cascade/inferencehub_api_key` (override with
`--key-file`). The normal CLI uses the profile's `NV_API_KEY` environment
variable. Credentials are not included in recorded model inputs.

For an API-only smoke check using a static mock camera and restricted read-only
tool dispatch, run `scripts/verify_inferencehub_astra.py`. That check makes live
API calls but is not evidence of physical manipulation.

## Desktop viewer and recording

These helpers target Linux with EGL rendering and an X11/Xwayland desktop:

```bash
uv pip install --python .venv/bin/python python-xlib imageio-ffmpeg
.venv/bin/python scripts/run_rebot_mujoco_astra.py --plate --view --record --keep-open
```

The physics process renders the agent's RGB-D camera using EGL. A separate
native MuJoCo viewer process displays measured simulation state, refreshed at
about 20 Hz. This avoids mixing its GLFW context lifetime with the camera's EGL
context. The viewer is read-only with respect to the actual experiment: GUI
controls cannot command the robot. The recorder captures only that viewer's
window. Camera frames sent to Astra use the overhead viewpoint and differ from
the desktop view.

`--keep-open` leaves the completed scene visible until the window is closed.
Without it, the recorder, viewer, runtime and API client are closed before exit.
Viewer cleanup waits for the native render thread before GLFW termination.

## Custom tasks and verification

A custom `--task` must select its verifier explicitly:

```bash
.venv/bin/python scripts/run_rebot_mujoco_astra.py --plate \
  --task "Put the red cube on the blue plate." --verify plate

.venv/bin/python scripts/run_rebot_mujoco_astra.py \
  --task "Describe the objects you can see." --verify none
```

`--verify lift` requires a sustained cube-height increase above 3 cm after an
additional half-second of native physics. `--verify plate` requires `--plate`
and checks footprint containment, surface height, support contact, absence of
robot contact, an open gripper and low cube velocity. `--verify none` uses only
the model's task report and records `independent_success: null`; it makes no
independent physical-success claim. Cleanup failure produces a nonzero exit
status even if the task succeeded.

The plate geometry and audit deliberately target this single fixture. They do
not implement a general destination verifier.

## Evidence and limitations

Each run has a fresh directory under `runs/rebot-mujoco-astra-*`, with isolated
belief, grasp, envelope and episodic stores. Existing run directories are kept.
Outputs include:

- `initial.png`, `final.png`, `settled.png`: camera snapshots (inspection-only
  runs save the initial snapshot).
- `model-inputs/NN/input.json` and adjacent JPEGs: exact adapter-level messages,
  tool specifications and image bytes; this is not a raw HTTP capture.
- `model-inputs/NN/response.json` or `error.json`: returned model response or
  error type/status. Request numbering includes failed attempts.
- `result.json` and the runtime's trace/keyframes: model calls, outcomes,
  selected verifier and limitations.
- `teardown.json` and `model-inputs/teardown.json`: independent software cleanup
  receipts. A task result alone does not establish clean teardown.
- `mujoco-screen.mp4`: optional desktop recording.

The demo explicitly uses analytic OBB grasps and color-threshold detection.
Occupancy mapping is disabled. Native contact simulation does not substitute for
collision-avoidance planning. Decomposition, the fast planner and milestone VLM
checks are disabled so the recorded planner calls come from Astra; ordinary
skill verification and motion safety gates still run. These settings are local
to the experiment and do not change the global defaults.

This is simulation evidence for one scene, not evidence of real-arm readiness
or repeatability across arbitrary object placements. The Menagerie asset is
pinned by the normal asset fetcher; recording dependencies are optional and are
not imported by the core robot backend.
