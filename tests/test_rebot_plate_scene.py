"""The demo's independent plate verdict requires physical support and release."""
from pathlib import Path

import pytest

from cascade.config import load_demo_config
from cascade.control.mujoco_arm import MujocoArm
from scripts.rebot_astra.scene import PLATE_TOP, PLATE_XY, audit, prepare


def test_explicit_detector_labels():
    from cascade.apps.demo import _make_detector
    cfg = load_demo_config(camera='rebot_mujoco_scene', arm='rebot_rs_mujoco', llm='mock')
    cfg.camera.detector._data['extra_labels'] = ['blue plate', 'blue plate']
    detector = _make_detector(cfg)
    assert detector._extra_labels == ['blue plate']


def test_plate_verdict_rejects_initial_and_accepts_supported_released_cube(tmp_path):
    mj = pytest.importorskip('mujoco')
    cfg = load_demo_config(camera='rebot_mujoco_scene', arm='rebot_rs_mujoco', llm='mock')
    if not Path(cfg.arm.mjcf).exists():
        pytest.skip('fetch_robot_assets.py rebot_rs is required')
    prepare(cfg, tmp_path)
    arm = MujocoArm(cfg.arm)
    arm.connect()
    try:
        world = arm.world
        assert not audit(world, arm, cfg.arm.gripper)['passed']
        m, d = world.model, world.data
        body = m.body('red_cube').id
        adr = int(m.jnt_qposadr[m.body_jntadr[body]])
        half = m.geom('red_cube').size[2]
        d.qpos[adr:adr+3] = [*PLATE_XY, PLATE_TOP + half + .01]
        d.qpos[adr+3:adr+7] = [1,0,0,0]
        d.qvel[:] = 0
        for _ in range(1000):
            mj.mj_step(m,d)
        verdict = audit(world, arm, cfg.arm.gripper)
        assert verdict['passed'], verdict
        # A visually aligned cube in midair has no physical plate support.
        d.qpos[adr+2] += .1
        mj.mj_forward(m,d)
        verdict = audit(world, arm, cfg.arm.gripper)
        assert not verdict['passed']
        assert not verdict['checks']['plate_support_contact']
        assert not verdict['checks']['bottom_at_plate_surface']
    finally:
        arm.disconnect()
