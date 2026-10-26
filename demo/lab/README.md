# Cascade Lab

Experimental fork of the Cascade booth demo: one password-gated web page with four live tabs, each one a different
robot world driven by an agent. Everything here is opt-in; the kitchen demo stays the default and is untouched.
Deviations in the shared source are listed in [`LAB_DEVIATIONS.md`](../../LAB_DEVIATIONS.md).

| Tab | World | Agent | Code |
|---|---|---|---|
| Microducks | 12 Microducks in a closed schoolyard with ramps, Isaac Sim 6.2 + Newton (MJWarp), OVRTX cameras | NVIDIA Nemotron 3.5 Bolt picks a tool, the navigator plans the paths | `microducks/isaac_ducks_sim.py`, `ovrtx_live.py`, `agent.py`, `director.py` |
| Microducks · Keynote | Same 12 ducks on the keynote Holodeck set; the navigator only sees 4 RGB-D cameras | Nemotron reads the perceived state or looks through a camera, then picks one tool | `microducks/keynote_live_sim.py`, `keynote_live_ovrtx.py`, `agent_nemo.py`, `director_nemo.py`, `perception.py` |
| Workshop | Arm cell with nuts, gears, a collar, a peg and a parts tray (Isaac Sim 6.2 + Newton container) | Cascade agent on OpenClaw | `demo/lab_scenes.py`, `workshop_config.json`, `workshop-instructions.md` |
| Recycling | Conveyor belt with a can and a cup, metal and plastic bins | Cascade agent on OpenClaw | `recycling_config.json`, `recycling-instructions.md`, `app/belt_driver.py` |

All 12 ducks run the same ONNX walking and recovery policy at 50 Hz, trained in Isaac Lab with BAM actuator
models. Physics runs in Isaac Sim; the cameras visitors see are rendered with OVRTX from the pose stream in
`/dev/shm`. In the Keynote tab the 4 cameras the navigator uses are Isaac Lab cameras on the Newton Warp renderer.

## Layout

- `app/`: the web app. `lab_server.py` (port 8095) does the password gate, scenario switcher, camera and MJPEG
  proxies, one job queue per cell, operator settings (exclusive mode, one camera per cell, video quality) and the
  Keynote Soft / Hard reset. `scenarios.json` defines the tabs. `static/app.html` is the page as served;
  `make_lab_app.py` and `patches/` are how it was built from `static/app.brev.html` (the served file carries a few
  later hand edits). `lab_tests.py` runs the Workshop and Recycling experiments against measured poses.
- `microducks/`: both duck sims plus the shared scene code (`yard_layout.py`, `ovrtx_scene.py`, `kitfilm.py`,
  `usdkit.py`), the systemd units as deployed (`systemd/`) and the Keynote operations tools (`ops/`: watchdog,
  hard reset, soak and fault tests).
- `lab_scenes.py` (one level up), `*_config.json`, `*-instructions.md`: the Workshop and Recycling cells.

## Services and ports (as deployed on a 2-GPU host)

| Service | GPU | Port | What |
|---|---|---|---|
| `cascade-lab-app` | none | 8095 | `app/lab_server.py` |
| `microducks-isaac-sim` | 1 | 18201 | schoolyard physics and order box |
| `microducks-ovrtx` | 1 | 18200 | schoolyard cameras |
| `microducks-keynote-sim` | 0 | 18211 | Keynote physics, perception and agent |
| `microducks-keynote-ovrtx` | 1 | 18210 | Keynote cameras (Holodeck look) |
| `microducks-nemotron-vllm` | 0 | 18310 | Nemotron 3.5 Bolt, OpenAI-compatible |
| `keynote-watchdog` | none | | restarts whatever stays broken |
| `cascade-lab-workshop`, `cascade-lab-recycling` | 0 / 1 | | Cascade containers, image `cascade-lab:6.2-x86_64` (Isaac Sim 6.2 source build), `PAAI_LAB_ENGINE=newton` |

The sims use the Isaac Sim 6.2 pip venv (Isaac Lab, Newton MJWarp) and the cameras an `ovrtx` 0.5.1 venv; the
paths in `systemd/` are the ones on the demo host.

## Running the app

```bash
export CASCADE_LAB_PASSWORD=...        # visitors
export CASCADE_LAB_ADMIN_PASSWORD=...  # operator
CASCADE_NGROK=0 python3 demo/lab/app/lab_server.py
```

Unset passwords get a random value for that run, printed in the log. Other knobs: `CASCADE_LAB_PORT`,
`CASCADE_LAB_SCENARIOS`, `CASCADE_LAB_PUBLIC_URLS` (named ngrok endpoints), `KEYNOTE_STATE` and
`KEYNOTE_HARD_RESET` (where `ops/` writes its state and the reset script lives). A tab whose sim or container is
down turns grey; the other tabs keep working.

## Assets (not in git)

- Workshop and Recycling textures: `python demo/lab/make_lab_textures.py && python demo/lab/make_recycling_v2_textures.py`
  rebuilds the 63 procedural maps byte for byte (sha256 in the JSON manifests), and `python demo/lab/fetch_lab_hdris.py`
  downloads the two Poly Haven CC0 HDRIs and checks their md5.
- `microducks/assets/`: `microduck_allcollisions.usd` and `microduck_allcollisions_isaac.usda` (Isaac Lab MicroDuck
  USD; the 3D model license is in [`assets/microduck/NOTICE.md`](../../assets/microduck/NOTICE.md)),
  `policy_np.npz` (the policy MLP weights extracted from its ONNX), `sky_daylift_v2.hdr` (procedural day sky, base generator `make_sky_day.py`),
  `holodeck/holodeck_grid_light.png` (Keynote Holodeck grid) and `assets_ph/` (Poly Haven CC0 planter, shrub and tree
  models plus the maps `fetch_ph.py` downloads).
- `microducks/textures/`: schoolyard textures from `make_yard_textures.py`, `make_yard_textures_v2.py` and
  `make_yard_textures_v3.py`.
- `fonts/`: NVIDIA Sans Bold (`NVIDIASans_Bd.ttf`) for the in-frame labels; not redistributable.
