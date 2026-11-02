"""OAK capture contract without the SDK or any attached device."""

from datetime import timedelta
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.perception.camera_base import CameraError, make_camera
from cascade.perception.oak_camera import (
    OakCamera, _decode_pair, _select_device, _sensor_calibration,
)


def test_sensor_calibration_override_uses_ram_and_records_content(tmp_path):
    path = tmp_path / "sensor.json"
    path.write_text('{"cameraData": [1]}')
    calls = []
    candidate = SimpleNamespace(eepromToJson=lambda: {"cameraData": [1]})
    sdk = SimpleNamespace(CalibrationHandler=lambda p: (
        calls.append(("load", p)), candidate)[1])
    device = SimpleNamespace(
        setCalibration=lambda c: calls.append(("ram", c)),
        readCalibration=lambda: pytest.fail("silently fell back to stored calibration"),
        flashCalibration=lambda c: pytest.fail("wrote EEPROM"),
    )
    cfg = Cfg({"serial": "camera", "sensor_calibration_json": str(path)})
    calibration, digest = _sensor_calibration(device, sdk, cfg)
    assert calibration is candidate and len(digest) == 64
    assert calls == [("load", path), ("ram", candidate)]
    candidate.eepromToJson = lambda: {"cameraData": [2]}
    assert _sensor_calibration(device, sdk, cfg)[1] != digest


def test_sensor_calibration_override_requires_serial_and_never_falls_back(tmp_path):
    touched = []
    device = SimpleNamespace(
        setCalibration=lambda c: touched.append("ram"),
        readCalibration=lambda: touched.append("read"),
    )
    cfg = Cfg({"sensor_calibration_json": str(tmp_path / "missing.json")})
    with pytest.raises(CameraError, match="pinned serial"):
        _sensor_calibration(device, SimpleNamespace(), cfg)
    cfg._data["serial"] = "camera"

    def missing(path):
        raise FileNotFoundError(path)

    with pytest.raises(FileNotFoundError):
        _sensor_calibration(device, SimpleNamespace(CalibrationHandler=missing), cfg)
    assert touched == []


def test_default_sensor_calibration_is_read_only():
    stored = SimpleNamespace(eepromToJson=lambda: {"cameraData": [1]})
    device = SimpleNamespace(
        readCalibration=lambda: stored,
        setCalibration=lambda c: pytest.fail("changed default camera calibration"),
    )
    assert _sensor_calibration(device, SimpleNamespace(), Cfg({}))[0] is stored


class Packet:
    def __init__(self, data, stamp=10.0, sequence=1):
        self.data, self.stamp, self.sequence = data, stamp, sequence
        self.K = np.array([[200.0, 0.0, 1.0], [0.0, 200.0, 1.0], [0.0, 0.0, 1.0]])

    def getCvFrame(self):
        return self.data

    getFrame = getCvFrame

    def getTransformation(self):
        return SimpleNamespace(getIntrinsicMatrix=lambda: self.K)

    def getTimestamp(self):
        return timedelta(seconds=self.stamp)

    def getSequenceNum(self):
        return self.sequence


def pair():
    return Packet(np.full((2, 3, 3), [1, 2, 3], dtype=np.uint8)), Packet(
        np.array([[0, 200, 1000], [2500, 3500, 65535]], dtype=np.uint16)
    )


def decode(rgb, depth):
    return _decode_pair(
        rgb,
        depth,
        now_sdk=10.1,
        receipt=20.0,
        min_depth=0.25,
        max_depth=3.0,
        max_skew=0.02,
        max_age=0.5,
        serial="unit-test",
    )


def test_bgr_metric_depth_invalid_zero_and_capture_age():
    frame = decode(*pair())
    np.testing.assert_array_equal(frame.rgb[0, 0], [1, 2, 3])
    np.testing.assert_allclose(frame.depth_m, [[0, 0, 1], [2.5, 0, 0]])
    assert frame.depth_m.dtype == np.float32
    assert frame.t == 20.0
    assert frame.observed_t == pytest.approx(19.9)
    assert frame.depth_source == "sensor"
    assert frame.frame_id == 0  # CameraBase owns IDs.


@pytest.mark.parametrize(
    "fault",
    ["shape", "dtype", "K_mismatch", "K_nan", "focal", "stale", "future", "skew"],
)
def test_bad_pairs_are_refused(fault):
    rgb, depth = pair()
    if fault == "shape":
        depth.data = depth.data[:1]
    elif fault == "dtype":
        depth.data = depth.data.astype(float)
    elif fault == "K_mismatch":
        depth.K[0, 2] += 1
    elif fault == "K_nan":
        rgb.K[0, 0] = np.nan
    elif fault == "focal":
        rgb.K[0, 0] = 0
    elif fault == "stale":
        rgb.stamp = depth.stamp = 9
    elif fault == "future":
        rgb.stamp = depth.stamp = 11
    else:
        depth.stamp -= 0.03
    with pytest.raises(CameraError):
        decode(rgb, depth)


def test_device_identity_never_falls_back_or_chooses_arbitrarily():
    a, b = SimpleNamespace(deviceId="a"), SimpleNamespace(deviceId="b")
    assert _select_device([a, b], "b") is b
    for infos, serial in [([], ""), ([a, b], ""), ([a], "b")]:
        with pytest.raises(CameraError):
            _select_device(infos, serial)


def test_factory_construction_does_not_import_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "depthai", None)
    cam = make_camera(Cfg({"type": "oak"}))
    assert isinstance(cam, OakCamera) and cam.has_depth
    with pytest.raises(CameraError, match="camera-oak"):
        cam.open()


@pytest.mark.parametrize("value", [True, False, 0, -1, 28.5, "28"])
def test_invalid_speckle_range_refused_before_device_access(value, monkeypatch):
    monkeypatch.setitem(sys.modules, "depthai", None)
    with pytest.raises(ValueError, match="speckle_range"):
        OakCamera(Cfg({"speckle_range": value}))


def test_partial_open_closes_device(monkeypatch):
    closed = []
    dev = SimpleNamespace(
        getCameraSensorNames=lambda: {}, close=lambda: closed.append(True)
    )
    fake = SimpleNamespace(
        Device=SimpleNamespace(
            getAllAvailableDevices=lambda: [SimpleNamespace(deviceId="a")]
        )
    )

    class Device:
        getAllAvailableDevices = fake.Device.getAllAvailableDevices

        def __new__(cls, info):
            return dev

    fake.Device = Device
    fake.CameraBoardSocket = SimpleNamespace(CAM_A=1, CAM_B=2, CAM_C=3)
    monkeypatch.setitem(sys.modules, "depthai", fake)
    cam = OakCamera(Cfg({}))
    with pytest.raises(CameraError, match="stereo"):
        cam.open()
    assert closed == [True] and cam._device is None


def test_close_releases_device_even_if_pipeline_stop_fails():
    cam = OakCamera(Cfg({}))
    closed = []

    def fail():
        raise RuntimeError("disconnected")

    cam._pipeline = SimpleNamespace(stop=fail)
    cam._device = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError):
        cam.close()
    cam.close()
    assert closed == [True]


def test_duplicate_capture_cannot_refresh_frame_age():
    cam = OakCamera(Cfg({}))
    rgb, depth = pair()
    cam._queue = SimpleNamespace(tryGet=lambda: {"rgb": rgb, "depth": depth})
    cam._dai = SimpleNamespace(
        Clock=SimpleNamespace(now=lambda: timedelta(seconds=10.1))
    )
    cam._grab()
    with pytest.raises(CameraError, match="duplicate"):
        cam._grab()


def test_profile_missing_calibration_cannot_fall_back_to_identity(tmp_path):
    from cascade.config import load_demo_config
    from cascade.perception.grounding import Extrinsics

    cfg = load_demo_config(
        camera="oak_d", arm="mock", llm="mock", _ignore_robot_environment=True
    )
    assert cfg.camera.fuse_beliefs is True
    assert cfg.camera.extrinsics.hand_eye_json.endswith("oak_d_rebot_rs.handeye.json")
    ext = cfg.camera.extrinsics.as_dict()
    ext["hand_eye_json"] = str(tmp_path / "absent.json")
    assert not Extrinsics.from_config(
        Cfg(ext), camera_serial=cfg.camera.serial
    ).calibrated


def test_local_profile_loads_gated_serial_bound_calibration():
    from cascade.config import load_demo_config
    from cascade.perception.grounding import Extrinsics
    from cascade.apps.demo import _camera_fusion

    cfg = load_demo_config(
        camera="oak_d", arm="rebot_rs", llm="mock", _ignore_robot_environment=True
    )
    ext = Extrinsics.from_config(cfg.camera.extrinsics, camera_serial=cfg.camera.serial)
    assert ext.calibrated, ext.calibration_error
    assert _camera_fusion(cfg.camera, ext)[0] is True
    wrong = Extrinsics.from_config(cfg.camera.extrinsics, camera_serial="other-camera")
    assert not wrong.calibrated
    assert _camera_fusion(cfg.camera, wrong) == (False, False)
