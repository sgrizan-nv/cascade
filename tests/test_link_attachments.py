import numpy as np
import pytest

from cascade.config import Cfg
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.link_attachments import build_link_attachments


def test_oak_profile_replaces_reference_camera_with_actual_link_geometry():
    from cascade.config import load_demo_config
    from cascade.safety.tool_volume import build_tool_volume
    cfg = load_demo_config(arm="rebot_rs_oak", camera="mock", llm="mock", _ignore_robot_environment=True)
    volume = build_tool_volume(cfg, cfg.arm)
    assert volume.names == ("housing", "fingers", "jaws")
    assert cfg.relative_motion.wrist_camera is None
    assert cfg.arm.check_link_attachments is True
    assert [a["link"] for a in cfg.arm.body_mask.attachments] == ["link4", "link4", "gripper_end"]


def test_reported_wrist_cable_is_covered_and_has_collision_geometry():
    """2026-10-31 trace voxels: cable was mapped as an obstacle to its holder."""
    from pathlib import Path
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.perception.robot_mask import build_mesh_body_mask
    from cascade.types import RobotState
    pytest.importorskip("pinocchio")
    cfg = load_demo_config(arm="rebot_rs_oak", camera="mock", llm="mock", _ignore_robot_environment=True)
    if not Path(cfg.arm.model).is_file():
        pytest.skip("workstation verified mesh copy is not installed")
    kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame, n_controlled=6, joint_signs=cfg.arm.joint_signs)
    q = np.deg2rad([-.2, 68.5, 68.8, -.2, -.2, .2])
    voxels = np.array([[.35, -.03, .50], [.36, -.02, .50]])
    cable = next(p for p in cfg.arm.body_mask.attachments if p['name'] == 'wrist_motor_cable')
    frame = kin.frame_poses(q, [cable['link']])[0]
    local = (voxels - frame[:3, 3]) @ frame[:3, :3]
    assert np.all(np.abs(local - cable['xyz']) < np.array(cable['box']) / 2)
    volume = build_link_attachments(cfg.arm, kin)
    assert 'wrist_motor_cable' in [name for name, points in volume.points(kin, q)]
    # Look toward the recorded voxels from a clear fixed-camera view.
    from test_occupancy_mesh_body_mask import look_at
    T = look_at([.7, .4, .5], voxels.mean(axis=0))
    K = np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
    pc = (voxels-T[:3,3]) @ T[:3,:3]
    uv = pc @ K.T
    uv = np.rint(uv[:,:2]/uv[:,2:]).astype(int)
    state = RobotState(q=q, gripper_valid=False)
    mask = build_mesh_body_mask(cfg.arm, kin, lambda: state, name="rebot_rs_oak")(
        np.zeros((480,640)), K, T)
    assert all(mask[y,x] for x,y in uv)


def test_oak_part_boxes_contain_verified_meshes_over_full_gripper_stroke():
    from pathlib import Path
    from types import SimpleNamespace
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.perception.link_mask import LinkGeometry, KinematicsLinkPoses
    from cascade.safety.tool_volume import build_tool_volume
    pytest.importorskip("pinocchio")
    cfg = load_demo_config(arm="rebot_rs_oak", camera="mock", llm="mock", _ignore_robot_environment=True)
    a = cfg.arm
    if not Path(a.model).is_file():
        pytest.skip("workstation verified mesh copy is not installed")
    kin = Kinematics(a.model, a.ee_frame, n_controlled=6, joint_signs=a.joint_signs)
    geo = LinkGeometry.from_urdf(a.model, cell_m=.02)
    names = ["gripper_end", "gripper_left", "gripper_right"]
    poses = KinematicsLinkPoses(kin, names)
    q = np.zeros(6)
    inv = np.linalg.inv(kin.fk(q))
    volume = build_tool_volume(cfg, a)
    for placed in poses(SimpleNamespace(q=q, gripper_valid=False)):
        for name in names:
            assert geo.sources[name] == "mesh"
            part = "housing" if name == "gripper_end" else "fingers"
            centre, axes, half = volume.boxes[volume.names.index(part)]
            T = inv @ placed[name]
            points = np.concatenate(geo.pieces[name]) @ T[:3, :3].T + T[:3, 3]
            assert np.all(np.abs((points-centre) @ axes) <= half + 1e-8)


class Kin:
    n = 1
    joint_limits = (np.array([-3.]), np.array([3.]))

    def fk(self, q):
        t = np.eye(4)
        t[:3, 3] = [.2, 0, .3]
        return t

    def frame_poses(self, q, frames):
        if any(f != "housing" for f in frames):
            raise ValueError("unknown link")
        t = self.fk(q)
        t[0, 3] += float(q[0])
        return [t.copy() for _ in frames]

    def link_positions(self, q):
        return np.array([[0, 0, .1], [.2, 0, .3]])


def rig(occupancy=None, *, xyz=(0, 0, .1)):
    cfg = Cfg({"check_link_attachments": True, "body_mask": {
        "mode": "mesh", "attachments": [{"name": "camera", "link": "housing",
                                            "box": [.02, .02, .02], "xyz": list(xyz)}]}})
    kin = Kin()
    volume = build_link_attachments(cfg, kin)
    return SafetyHarness(SafetyLimits(workspace_min=[-1,-1,0], workspace_max=[1,1,1]),
                         kin, occupancy, link_attachments=volume), volume


class Map:
    required = True

    def clearance(self, points):
        return np.linalg.norm(points - [.3, 0, .4], axis=1)


def test_attachment_follows_its_link_not_stationary_tcp():
    harness, volume = rig()
    a = volume.points(harness.kin, np.array([0.]))[0][1]
    b = volume.points(harness.kin, np.array([.1]))[0][1]
    np.testing.assert_allclose(b-a, np.tile([.1,0,0], (len(a),1)))


def test_attachment_collision_refuses_static_and_streamed_checks():
    harness, _ = rig(Map())
    assert "attachment camera" in harness.vet_pose(np.array([.1]))
    assert "attachment camera" in harness.vet_step(np.array([.099]),np.array([.1]),.02)
    harness.heartbeat()
    from cascade.types import SafetyViolation
    with pytest.raises(SafetyViolation, match="attachment camera"):
        harness.approve(np.array([.099]),np.array([.1]),.02)


def test_camera_gets_no_target_exemption():
    harness, _ = rig(Map())
    assert "attachment camera" in harness.vet_pose(
        np.array([.1]), exempt_xy=np.array([.3,0]), exempt_radius_m=1.)


def test_attachment_refusal_uses_existing_report_and_map_coordinates():
    class RecordedMap(Map):
        def clearance(self, points):
            return np.full(len(points), .015)
        def clearance_record(self):
            return {"refreshes": 7}
        def nearby_occupied(self, point, radius_m, limit):
            return np.array([np.r_[point + [.015, 0, 0], .015]])
    h, _ = rig(RecordedMap())
    h.base_pose = np.eye(4)
    h.base_pose[0, 3] = 1.
    reason = h.vet_pose(np.array([.1]))
    reports = h.drain_clearance_reports()
    assert len(reports) == 1
    r = reports[0]
    assert r["reason"] == reason and r["gate"] == "link attachment camera"
    assert r["phase"] == "vet_pose" and r["exemption"] is None
    assert r["sample"][0] > 1. and r["clearance_m"] == .015
    assert r["voxels"] and "snapshot_error" not in r
    assert str(round(r["sample"][0], 4)) in reason


def test_attachment_map_exception_is_reported_and_preserved():
    from cascade.types import SafetyViolation
    error = SafetyViolation("body mask unavailable")
    class FailedMap(Map):
        def clearance(self, points):
            raise error
    h, _ = rig(FailedMap())
    with pytest.raises(SafetyViolation) as raised:
        h.vet_pose(np.array([.1]))
    assert raised.value is error
    reports = h.drain_clearance_reports()
    assert len(reports) == 1 and reports[0]["raised"] == "SafetyViolation"


def test_attachment_cannot_hit_table_when_tcp_is_clear():
    harness, _ = rig(xyz=(0,0,-.3))
    assert "table clearance" in harness.vet_pose(np.array([0.]))


def test_missing_map_fails_closed_for_required_map():
    class Missing(Map):
        def clearance(self, points):
            return None
    harness, _ = rig(Missing())
    assert "clearance unavailable" in harness.vet_pose(np.array([0.]))
