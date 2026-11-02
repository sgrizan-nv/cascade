"""A cached tabletop belief cannot refute a physical pickup (w11wjefc)."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.agent.effects import CONFIRMED, REFUTED, UNVERIFIED, PostconditionChecker
from cascade.config import Cfg
from cascade.skills.runtime import SkillRuntime
from cascade.types import Frame, SkillError


def observation(z, **kwargs):
    return dict(position=[.246, .002, z], observed_t=time.monotonic(),
                camera='oak_d', frame_id=1, **kwargs)


def checker(reader):
    return PostconditionChecker(holding_pose=reader, belief_pose=lambda _: [.24, 0, .0471],
                                gripper_frac=lambda: .3412365)


@pytest.mark.parametrize('after_z,status', [(.0971, CONFIRMED), (.0471, REFUTED)])
def test_fresh_measurement_distinguishes_lift_from_miss(after_z, status):
    heights = iter([.0471, after_z])
    c = checker(lambda label, floor: observation(next(heights)))
    before = c.snapshot('small cardboard box', 'grasp_object')
    pc = c.verify('grasp_object', {'label': 'small cardboard box'}, {'ok': True}, before)
    assert pc.status == status and pc.channel == 'perception'
    assert pc.measured['rise_m'] == pytest.approx(after_z - .0471)
    assert pc.measured['after_observation']['observed_t'] >= before['observation']['observed_t']


@pytest.mark.parametrize('problem', ['stale', 'occluded', 'nonfinite', 'camera_changed', 'future'])
def test_unusable_post_lift_observation_never_falls_back_to_tabletop_belief(problem):
    c = checker(lambda label, floor: observation(.0471))
    before = c.snapshot('small cardboard box', 'grasp_object')

    def unavailable(label, floor):
        if problem == 'occluded':
            raise SkillError('target hidden behind the gripper')
        value = observation(.0471)
        if problem == 'stale':
            value['observed_t'] = before['observation']['observed_t']
        if problem == 'future':
            value['observed_t'] += 100
        if problem == 'nonfinite':
            value['position'][2] = float('nan')
        if problem == 'camera_changed':
            value['camera'] = 'different_camera'
        return value

    c._holding_pose_reader = unavailable
    pc = c.verify('grasp_object', {'label': 'small cardboard box'}, {'ok': True}, before)
    assert pc.status == UNVERIFIED
    assert 'did not rise' not in pc.evidence
    assert 'rise_m' not in pc.measured
    assert 'after_observation' in pc.measured


def test_missing_before_observation_cannot_confirm_rise_from_table_height():
    c = checker(lambda label, floor: observation(.1))
    pc = c.verify('grasp_object', {'label': 'box'}, {'ok': True}, {})
    assert pc.status == UNVERIFIED


def test_physics_remains_preferred_and_air_grasp_remains_refuted():
    z = [.05]
    c = checker(lambda *_: pytest.fail('physics must not trigger camera inference'))
    c._object_pose = lambda _: [.2, 0, z[0]]
    before = c.snapshot('box', 'grasp_object')
    z[0] += .05
    pc = c.verify('grasp_object', {'label': 'box'}, {'ok': True}, before)
    assert pc.status == CONFIRMED and pc.channel == 'physics'
    c._gripper_frac = lambda: .01
    pc = c.verify('grasp_object', {'label': 'box'}, {'ok': True}, before)
    assert pc.status == REFUTED and pc.channel == 'gripper'


def runtime_double(monkeypatch):
    monkeypatch.delenv('CASCADE_GRASP_EVIDENCE_DIR', raising=False)
    r = SkillRuntime.__new__(SkillRuntime)
    r.cfg = Cfg({'camera': {'name': 'oak_d'}})
    r.observe_fresh = lambda: None
    r._localized_camera = lambda: SimpleNamespace(name='oak_d')
    r._to_world = lambda p: p + np.array([.1, 0, .02])
    fix = SimpleNamespace(position=np.array([.2, 0, .07]), points=np.ones((20, 3)),
                          detection=SimpleNamespace(label='cardboard box', mask=np.ones((2, 2), bool)))
    frames = []

    def localize(label, *, memory):
        assert memory is False
        f = Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth_m=np.ones((2, 2)),
                  K=np.eye(3), depth_source='sensor', T_base_cam=np.eye(4), frame_id=42)
        frames.append(f)
        return f, fix

    r._localize = localize
    return r, frames, fix


def test_runtime_reuses_memory_free_localization_and_retains_evidence(monkeypatch, tmp_path):
    r, frames, fix = runtime_double(monkeypatch)
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    value = r._observe_holding_pose('small cardboard box', time.monotonic())
    assert value['position'] == pytest.approx([.3, 0, .09])  # shared world frame
    assert value['frame_id'] == 42
    paths = list((tmp_path / 'verification').glob('*.json'))
    assert len(paths) == 1 and str(paths[0]) == value['receipt']
    evidence = json.loads(paths[0].read_text())
    assert evidence['logging_ok']
    arrays = np.load(next((tmp_path / 'verification').glob('*.npz')), allow_pickle=False)
    assert any(k.endswith('depth_m') for k in arrays.files)
    assert any(k.endswith('target_mask') for k in arrays.files)


@pytest.mark.parametrize('invalid', ['old_capture', 'plane_depth'])
def test_runtime_rejects_old_or_inferred_geometry(monkeypatch, invalid):
    r, _, fix = runtime_double(monkeypatch)
    frame = Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth_m=np.ones((2, 2)), K=np.eye(3),
                  depth_source='sensor')
    if invalid == 'old_capture':
        frame.observed_t = time.monotonic() - 10
    else:
        frame.depth_source = 'plane'
    r._localize = lambda *args, **kwargs: (frame, fix)
    with pytest.raises(SkillError):
        r._observe_holding_pose('box', time.monotonic() - 1)
