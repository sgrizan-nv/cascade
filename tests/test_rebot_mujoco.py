"""Coordinate and coupled-jaw contracts for the B601-RS MuJoCo adapter."""
import numpy as np
import pytest

from cascade.config import Cfg, load_profile
from cascade.control.mujoco_arm import MujocoArm


@pytest.mark.parametrize('signs', [[1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, float('nan')]])
def test_invalid_sign_map_refused(signs):
    cfg = load_profile('arms', 'rebot_rs_mujoco').as_dict()
    cfg['mj_joint_signs'] = signs
    with pytest.raises(ValueError, match='mj_joint_signs'):
        MujocoArm(Cfg(cfg))


def test_mirrored_coordinates_and_coupled_actuators(tmp_path):
    pytest.importorskip('mujoco')
    path = tmp_path / 'arm.xml'
    path.write_text('''<mujoco><compiler angle="radian"/><option gravity="0 0 0" integrator="implicitfast"/>
      <worldbody><body><joint name="arm" range="-2 0"/>
        <geom type="sphere" size=".02" mass="1"/>
        <body pos=".1 0 0"><joint name="left" type="slide" axis="1 0 0" range="0 .05"/>
          <geom type="sphere" size=".01" mass=".1"/></body>
        <body pos="-.1 0 0"><joint name="right" type="slide" axis="-1 0 0" range="0 .05"/>
          <geom type="sphere" size=".01" mass=".1"/></body>
      </body></worldbody>
      <actuator><position name="arm" joint="arm" kp="100" kv="10"/>
      <position name="left" joint="left" kp="100" kv="10"/>
      <position name="right" joint="right" kp="100" kv="10"/></actuator></mujoco>''')
    arm = MujocoArm(Cfg(dict(mjcf=str(path), n_joints=1, mj_joints=['arm'],
        mj_joint_signs=[-1], home_q=[.5], mj_gripper_joint='left',
        mj_gripper_followers=[dict(joint='right', actuator='right')],
        gripper=dict(open_pos=0, closed_pos=.05))))
    arm.connect()
    try:
        assert arm.get_state().q == pytest.approx([.5])
        assert arm._data.qpos[0] == pytest.approx(-.5)
        arm._data.qvel[0] = -.2
        assert arm.get_state().dq == pytest.approx([.2])
        arm._data.qvel[:] = 0
        arm.send_joint_target(np.array([.7]))
        assert arm._data.ctrl[0] == pytest.approx(-.7)
        assert arm.wait_settled(np.array([.7]), .01, 2)
        arm.set_gripper(.04)
        assert arm._data.ctrl[1:] == pytest.approx([.04, .04])
        assert arm._data.qpos[1:] == pytest.approx([.04, .04], abs=.001)
        before = arm._data.qpos.copy()
        arm.stop()
        assert arm._data.ctrl == pytest.approx(before)
        ctrl = arm._data.ctrl.copy()
        arm.set_gripper(0)
        arm.send_joint_target(np.array([1.]))
        assert arm._data.ctrl == pytest.approx(ctrl)
    finally:
        arm.disconnect()


def test_rebot_native_fk_agrees_with_signed_urdf():
    mj = pytest.importorskip('mujoco')
    pytest.importorskip('pinocchio')
    from pathlib import Path
    from cascade.control.kinematics import Kinematics
    cfg = load_profile('arms', 'rebot_rs_mujoco')
    if not Path(cfg.mjcf).exists():
        pytest.skip('fetch_robot_assets.py rebot_rs is required')
    kin = Kinematics(cfg.model, cfg.ee_frame, joint_signs=cfg.joint_signs)
    arm = MujocoArm(cfg, kin)
    arm.connect()
    try:
        rng = np.random.default_rng(23)
        for _ in range(30):
            q = np.asarray(cfg.home_q) + rng.uniform(-.1, .1, 6)
            arm._data.qpos[arm._qadr] = -q
            mj.mj_forward(arm._model, arm._data)
            body = arm._data.body('gripper_end')
            T = kin.fk(q)
            np.testing.assert_allclose(body.xpos, T[:3, 3], atol=3e-6)
            np.testing.assert_allclose(body.xmat.reshape(3, 3), T[:3, :3], atol=3e-6)
    finally:
        arm.disconnect()


def test_rebot_open_jaws_separate_in_tcp_frame():
    mj = pytest.importorskip('mujoco')
    from pathlib import Path
    cfg = load_profile('arms', 'rebot_rs_mujoco')
    if not Path(cfg.mjcf).exists():
        pytest.skip('fetch_robot_assets.py rebot_rs is required')
    arm = MujocoArm(cfg)
    arm.connect()
    try:
        gaps = []
        for value in (cfg.gripper.closed_pos, cfg.gripper.open_pos):
            arm._data.qpos[arm._grip_qadr] = value
            for adr, _ in arm._follower_indices:
                arm._data.qpos[adr] = value
            mj.mj_forward(arm._model, arm._data)
            left = arm._data.body('gripper_left').xpos
            right = arm._data.body('gripper_right').xpos
            gaps.append(np.linalg.norm(left - right))
        assert gaps[1] - gaps[0] == pytest.approx(.1, abs=1e-6)
    finally:
        arm.disconnect()
