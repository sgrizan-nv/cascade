# Cascade Lab (experimental fork) deviations

Base: johnnynunez/cascade main 42eaddd4 as bundled on the PAAI Brev (obb grasp
backend, same as the production deployment). Changes, all opt-in by env/config:

- demo/lab_scenes.py: procedural workshop and recycling rooms and prop visuals.
- demo/isaac_scene.py: `lab_scene` configs use lab rooms, lab prop names and
  `lab:<kind>` visuals; painted steel tray style; `box` convex proxy.
- demo/scene_identity.py: lab configs bind their own source bytes instead of
  the Cocina Asier audit (`experimental: true` in the identity).
- deploy/runtime/runtime.py: PAAI_LAB_BIND (bridge/camera bind inside a bridged
  container), PAAI_LAB_SCENE_CONFIG, PAAI_LAB_INSTRUCTIONS, PAAI_LAB_ROBOT_NAME. With
  PAAI_LAB_BIND set, visitor URLs may be plain http on the loopback PAAI_HTTP_BIND (the
  cells are not on Tailscale; the lab app is the only public front door).
- demo/kitchen/visitor_instructions.py: instruction file from PAAI_LAB_INSTRUCTIONS.
- src/cascade/sim/placement.py: with CASCADE_SCENE_CONFIG naming a lab config, the post-place
  verdict comes from live PhysX rigid-body pose + speed over >= 0.6 s of sim time against the
  config's tray cavity / green square (the kitchen auditor is bound to kitchen props).
- src/cascade/skills/runtime.py: tray/bin synonyms map to the open-box destination
  ("bin" no longer means the green square).
- scripts/isaac_bridge.py: optional `drop` block in lab configs; after a VERIFIED reset the
  listed props are lifted above the chute with a seeded jitter and dropped by PhysX.
- demo/lab/: scene configs and agent instructions.
- demo/lab/textures/ (look-dev v2): procedural CC0 PBR maps written by demo/lab/make_lab_textures.py and
  two Poly Haven CC0 HDRIs (provenance in textures/SOURCES.json), read by demo/lab_scenes.py at
  authoring time; missing files fall back to plain materials (CASCADE_LAB_TEXTURES=0 forces it).

Not production. The kitchen scene, unchanged, remains the default.
