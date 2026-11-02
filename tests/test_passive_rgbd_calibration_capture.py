"""A stationary calibration image must be bracketed by measured encoder poses."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cascade.types import Frame

spec = importlib.util.spec_from_file_location(
    "capture_rebot_oak",
    Path(__file__).resolve().parents[1] / "scripts/capture_rebot_oak_calibration.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fit_spec = importlib.util.spec_from_file_location(
    "fit_rebot_oak",
    Path(__file__).resolve().parents[1] / "scripts/fit_rebot_oak_calibration.py",
)
fit_module = importlib.util.module_from_spec(fit_spec)
fit_spec.loader.exec_module(fit_module)


def inputs():
    before = {
        "model_sha256": "a",
        "q_rad": [0.0] * 6,
        "motors": [{"read_completed": 1.0}],
    }
    after = {
        "model_sha256": "a",
        "q_rad": [0.001] * 6,
        "motors": [{"read_started": 2.0}],
    }
    frame = Frame(
        np.zeros((2, 3, 3), np.uint8),
        np.ones((2, 3), np.float32),
        np.eye(3),
        t=1.6,
        observed_t=1.5,
        depth_source="sensor",
    )
    return before, after, [frame]


def test_bracketed_stationary_sample():
    np.testing.assert_allclose(module.validate_static_interval(*inputs()), 0.001)


@pytest.mark.parametrize(
    "fault",
    ["moving", "nan", "model", "old", "future", "receipt", "repeat", "no_depth", "K"],
)
def test_refuses_unusable_sample(fault):
    before, after, frames = inputs()
    if fault == "moving":
        after["q_rad"][1] = 0.01
    elif fault == "nan":
        after["q_rad"][1] = float("nan")
    elif fault == "model":
        after["model_sha256"] = "different"
    elif fault == "old":
        frames[0].observed_t = 0.9
    elif fault == "future":
        frames[0].observed_t = 3.0
    elif fault == "receipt":
        frames[0].t = 3.0
    elif fault == "repeat":
        frames.append(copy.deepcopy(frames[0]))
    elif fault == "no_depth":
        frames[0].depth_source = "mono"
    else:
        second = copy.deepcopy(frames[0])
        second.observed_t = 1.7
        second.t = 1.8
        second.K[0, 0] += 1
        frames.append(second)
    with pytest.raises(ValueError):
        module.validate_static_interval(before, after, frames)


@pytest.fixture
def saved_sample(tmp_path):
    folder = tmp_path / "pose01"
    folder.mkdir()
    pose = {"q_rad": [0.0] * 6, "model_sha256": "model", "joint_signs": [-1] * 6}
    meta = {
        "schema": "cascade.passive-rgbd-calibration/1",
        "motion_commands_sent": 0,
        "camera": {"serial": "camera"},
        "before": pose,
        "after": copy.deepcopy(pose),
        "K": np.eye(3).tolist(),
    }
    (folder / "sample.json").write_text(json.dumps(meta))
    np.savez(
        folder / "sample.npz",
        q=np.zeros(6),
        K=np.eye(3),
        depth_m=np.ones((3, 4), dtype=np.float32),
        T_gripper2base=np.eye(4),
    )
    return tmp_path, folder, meta


def test_fitter_loads_measured_sample(saved_sample):
    directory, _, _ = saved_sample
    samples, files = fit_module.load_samples(
        directory, model_sha256="model", serial="camera"
    )
    assert len(samples) == len(files) == 1
    assert samples[0].label == "pose01"


@pytest.mark.parametrize("first_hash", [None, "old-calibration"])
def test_fitter_refuses_depth_from_different_sensor_calibrations(saved_sample, first_hash):
    import shutil

    directory, folder, meta = saved_sample
    if first_hash is not None:
        meta["camera"]["sensor_calibration_sha256"] = first_hash
    (folder / "sample.json").write_text(json.dumps(meta))
    other = directory / "pose02"
    shutil.copytree(folder, other)
    meta["camera"]["sensor_calibration_sha256"] = "new-calibration"
    (other / "sample.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sensor calibration changed"):
        fit_module.load_samples(directory, model_sha256="model", serial="camera")


@pytest.mark.parametrize(
    "fault", ["serial", "model", "signs", "moving", "stored_pose", "intrinsics"]
)
def test_fitter_refuses_mixed_or_inconsistent_capture(saved_sample, fault):
    directory, folder, meta = saved_sample
    if fault == "serial":
        meta["camera"]["serial"] = "other"
    elif fault == "model":
        meta["before"]["model_sha256"] = "other"
    elif fault == "signs":
        meta["before"]["joint_signs"] = [1] * 6
    elif fault == "moving":
        meta["after"]["q_rad"][0] = 0.02
    elif fault == "stored_pose":
        meta["before"]["q_rad"][0] = 0.02
    else:
        meta["K"][0][0] = 2.0
    (folder / "sample.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        fit_module.load_samples(directory, model_sha256="model", serial="camera")
