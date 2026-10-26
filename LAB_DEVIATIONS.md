# Cascade Lab (experimental fork) deviations

Base: johnnynunez/cascade main 42eaddd4 as bundled on the PAAI Brev (obb grasp
backend, same as the production deployment). Changes, all opt-in by env/config:

- demo/lab_scenes.py: procedural workshop and recycling rooms and prop visuals.
- demo/isaac_scene.py: `lab_scene` configs use lab rooms, lab prop names and
  `lab:<kind>` visuals; painted steel tray style; `box` convex proxy.
- demo/scene_identity.py: lab configs bind their own source bytes instead of
  the Cocina Asier audit (`experimental: true` in the identity).
- deploy/runtime/runtime.py: PAAI_LAB_BIND (bridge/camera bind inside a bridged
  container), PAAI_LAB_SCENE_CONFIG, PAAI_LAB_INSTRUCTIONS, PAAI_LAB_ROBOT_NAME.
- demo/kitchen/visitor_instructions.py: instruction file from PAAI_LAB_INSTRUCTIONS.
- src/cascade/skills/runtime.py: tray/bin synonyms map to the open-box destination
  ("bin" no longer means the green square).
- scripts/isaac_bridge.py: optional `drop` block in lab configs; after a VERIFIED reset the
  listed props are lifted above the chute with a seeded jitter and dropped by PhysX.
- demo/lab/: scene configs and agent instructions.
- demo/lab/textures/ (look-dev v2): procedural CC0 PBR maps written by demo/lab/make_lab_textures.py and
  two Poly Haven CC0 HDRIs (provenance in textures/SOURCES.json), read by demo/lab_scenes.py at
  authoring time; missing files fall back to plain materials (CASCADE_LAB_TEXTURES=0 forces it).

Not production. The kitchen scene, unchanged, remains the default.

## Isaac Sim 6.2 + Newton (2026-10-17, Isaac Sim 6.2 agent)

Rebased on johnnynunez/cascade main `ec2d95c` (branch `lab-6.2-newton` in `/tmp/cascade_lab_src62`); the 4
upstream commits after `42eaddd4` are merged into `configs/demo.yaml`, `src/cascade/agent/aspire.py`,
`apps/demo.py`, `memory/programs.py`, `skills/library.py`, `skills/runtime.py`. Opt-in, default unchanged:

- deploy/runtime/runtime.py: `PAAI_LAB_ENGINE=newton|physx` (default physx) picks the bridge `--engine`, the
  engine the health check requires from the bridge ping, and the smoke contact channel
  (`newton_mjwarp_contact_force` on Newton). `ISAACSIM_PATH` is passed to the bridge, which is how upstream
  `scripts/isaac_runtime.py` admits a 6.2 source release.
- scripts/isaac_bridge.py: lab `drop` on Newton goes through upstream `_newton_teleport` (identity yaw).
- scripts/isaac_bridge.py: on Newton the stage/timeline get `1/dt` time codes per second (env
  `CASCADE_NEWTON_ONE_STEP_PER_UPDATE=0` disables). The Newton experience steps with fixed time codes (24/s on
  this stage), so each `app.update()` advanced 5 physics steps and every render landed on an intermediate step:
  all four cameras failed with "render frame has no unique matching state history" (the open 6.2 camera-cadence
  item in docs/NEWTON_ENGINE.md, candidate fix "one physics step per update"). Measured cost: 0.25x real time vs
  0.39x for the 6.1 PhysX cell on the same shared GPU.
- assets/newton/ (rebot_gripper_hulls.usda) added to the bundle: the Newton bridge needs the vendor gripper hulls.
- Image `cascade-lab:6.2-x86_64` = the lab Dockerfile on `isaac-sim-src:6.2.0-74a9356b`, built from
  isaac-sim/IsaacSim tag v6.2.0 with the official recipe. Compose: `PAAI_LAB_ENGINE=newton`, `ISAACSIM_PATH=/isaac-sim`.
