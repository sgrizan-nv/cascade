"""OAK stereo RGB-D via DepthAI v3, with factory-calibrated pixel geometry.

The RGB output is undistorted, depth is explicitly reprojected to that output,
and paired timestamps are checked before yielding a CASCADE Frame. The SDK is
optional and imported only on open. No assumed camera-to-robot transform.
"""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..types import Frame
from .camera_base import CameraBase, CameraError


def _sensor_calibration(device, dai, cfg):
    """Load SDK calibration into RAM only; return its reproducible identity."""
    path = cfg.get("sensor_calibration_json")
    if path:
        if not cfg.get("serial"):
            raise CameraError("OAK: a sensor calibration file requires a pinned serial")
        calibration = dai.CalibrationHandler(Path(path))
        device.setCalibration(calibration)  # SDK runtime override; never flash EEPROM.
    else:
        calibration = device.readCalibration()
    encoded = json.dumps(
        calibration.eepromToJson(), sort_keys=True, separators=(",", ":")
    ).encode()
    return calibration, hashlib.sha256(encoded).hexdigest()


def _select_device(infos, serial):
    matches = [i for i in infos if not serial or i.deviceId == serial]
    if len(matches) != 1:
        raise CameraError(
            f"OAK: expected one accessible device for serial {serial!r}; found {len(matches)}. Check USB permissions and serial."
        )
    return matches[0]


def _decode_pair(
    rgb_msg,
    depth_msg,
    *,
    now_sdk,
    receipt,
    min_depth,
    max_depth,
    max_skew,
    max_age,
    serial,
):
    rgb = rgb_msg.getCvFrame()
    raw = depth_msg.getFrame()
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CameraError("OAK: invalid BGR image")
    if raw.dtype != np.uint16 or raw.shape != rgb.shape[:2]:
        raise CameraError("OAK: invalid or unaligned millimetre depth")
    K = np.asarray(rgb_msg.getTransformation().getIntrinsicMatrix(), dtype=float)
    depth_K = np.asarray(
        depth_msg.getTransformation().getIntrinsicMatrix(), dtype=float
    )
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1])
        or depth_K.shape != (3, 3)
        or not np.allclose(K, depth_K, atol=1e-3, rtol=1e-5)
    ):
        raise CameraError("OAK: missing or inconsistent aligned intrinsics")
    stamps = [m.getTimestamp().total_seconds() for m in (rgb_msg, depth_msg)]
    ages = [now_sdk - t for t in stamps]
    skew = abs(stamps[0] - stamps[1])
    if (
        not np.isfinite(stamps + ages).all()
        or min(ages) < -0.005
        or max(ages) > max_age
        or skew > max_skew
    ):
        raise CameraError("OAK: stale, future or unsynchronized RGB/depth pair")
    depth = raw.astype(np.float32) * np.float32(0.001)
    depth[(depth < min_depth) | (depth > max_depth)] = 0
    return Frame(
        rgb=rgb,
        depth_m=depth,
        K=K,
        t=receipt,
        depth_source="sensor",
        observed_t=receipt - max(0.0, max(ages)),
        capture={
            "source": "depthai",
            "serial": serial,
            "rgb_sequence": rgb_msg.getSequenceNum(),
            "depth_sequence": depth_msg.getSequenceNum(),
            "rgb_sdk_timestamp_s": stamps[0],
            "depth_sdk_timestamp_s": stamps[1],
            "pair_skew_s": skew,
            "oldest_age_s": max(ages),
            "rgb_undistorted": True,
            "depth_aligned_to_rgb": True,
        },
    )


class OakCamera(CameraBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._device = self._pipeline = self._queue = self._dai = None
        self._last_sequences = None
        self._K = None
        self.serial = str(cfg.get("serial", ""))
        self.device_info = {}
        self.speckle_range = cfg.get("speckle_range")
        if self.speckle_range is not None and (
            isinstance(self.speckle_range, bool)
            or not isinstance(self.speckle_range, int)
            or self.speckle_range <= 0
        ):
            raise ValueError("OAK: speckle_range must be a positive integer or omitted")
        self.min_depth = float(cfg.get("min_depth_m", 0.25))
        self.max_depth = float(cfg.get("max_depth_m", 3.0))
        self.max_age = float(cfg.get("max_frame_age_s", 0.5))
        self.max_skew = float(cfg.get("max_pair_skew_s", 0.02))
        if not np.isfinite(
            [self.min_depth, self.max_depth, self.max_age, self.max_skew]
        ).all() or not (
            0 < self.min_depth < self.max_depth
            and 0 < self.max_skew <= self.max_age <= 2
        ):
            raise ValueError("OAK: invalid depth range or timing bounds")

    @property
    def has_depth(self):
        return True

    @property
    def K(self):
        return self._K

    @property
    def D(self):
        return np.zeros(5)  # RGB output was explicitly undistorted by the SDK.

    def open(self):
        if self._device is not None:
            raise CameraError("OAK: camera already open")
        try:
            import depthai as dai
        except ImportError as exc:
            raise CameraError("OAK requires the camera-oak extra (DepthAI v3)") from exc
        self._dai = dai
        try:
            info = _select_device(dai.Device.getAllAvailableDevices(), self.serial)
            self._device = device = dai.Device(info)
            self.serial = info.deviceId
            sensors = device.getCameraSensorNames()
            sockets = [
                dai.CameraBoardSocket.CAM_A,
                dai.CameraBoardSocket.CAM_B,
                dai.CameraBoardSocket.CAM_C,
            ]
            if any(s not in sensors for s in sockets):
                raise CameraError(
                    "OAK backend needs RGB on CAM_A and stereo on CAM_B/C"
                )
            calibration, calibration_hash = _sensor_calibration(device, dai, self.cfg)
            eeprom = calibration.getEepromData()
            self.device_info = {
                "serial": self.serial,
                "product": eeprom.productName,
                "board": eeprom.boardName,
                "usb_speed": str(device.getUsbSpeed()),
                "sdk": dai.__version__,
                "sensors": {str(k): v for k, v in sensors.items()},
                "sensor_calibration_sha256": calibration_hash,
                "sensor_calibration_json": self.cfg.get("sensor_calibration_json"),
            }
            self._pipeline = p = dai.Pipeline(device)
            fps = float(self.cfg.get("fps", 10))
            size = (int(self.cfg.get("width", 1280)), int(self.cfg.get("height", 800)))
            rgb, left, right = [p.create(dai.node.Camera).build(s) for s in sockets]
            sync_mode = self.cfg.get("frame_sync", "auto")
            if sync_mode == "rgb_output":
                # Measured on OAK-D-W-97: AUTO left ~58 ms RGB/stereo skew.
                # Explicit internal FSYNC reduces it to ~20 us after startup.
                if [sensors[s] for s in sockets] != ["OV9782", "OV9282", "OV9282"]:
                    raise CameraError(
                        "OAK rgb_output FSYNC is validated only for OV9782 + OV9282 stereo"
                    )
                rgb.initialControl.setFrameSyncMode(
                    dai.CameraControl.FrameSyncMode.OUTPUT
                )
                for mono in (left, right):
                    mono.initialControl.setFrameSyncMode(
                        dai.CameraControl.FrameSyncMode.INPUT
                    )
            elif sync_mode != "auto":
                raise CameraError(f"OAK: unknown frame_sync mode {sync_mode!r}")
            color = rgb.requestOutput(
                size, type=dai.ImgFrame.Type.BGR888p, fps=fps, enableUndistortion=True
            )
            stereo = p.create(dai.node.StereoDepth)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.initialConfig.setDepthUnit(dai.LengthUnit.MILLIMETER)
            if self.speckle_range is not None:
                stereo.initialConfig.postProcessing.speckleFilter.enable = True
                stereo.initialConfig.postProcessing.speckleFilter.speckleRange = self.speckle_range
            sc = stereo.initialConfig
            self.device_info["stereo"] = {
                "confidence_threshold": sc.costMatching.confidenceThreshold,
                "lr_threshold": sc.algorithmControl.leftRightCheckThreshold,
                "subpixel_bits": sc.algorithmControl.subpixelFractionalBits,
                "median": str(sc.postProcessing.median),
                "speckle_enabled": sc.postProcessing.speckleFilter.enable,
                "speckle_range": sc.postProcessing.speckleFilter.speckleRange,
                "temporal_enabled": sc.postProcessing.temporalFilter.enable,
                "spatial_enabled": sc.postProcessing.spatialFilter.enable,
            }
            left.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=fps).link(
                stereo.left
            )
            right.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=fps).link(
                stereo.right
            )
            align = p.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            color.link(align.inputAlignTo)
            sync = p.create(dai.node.Sync)
            sync.setSyncThreshold(timedelta(seconds=self.max_skew))
            sync.setSyncAttempts(-1)
            color.link(sync.inputs["rgb"])
            align.outputAligned.link(sync.inputs["depth"])
            self._queue = sync.out.createOutputQueue(maxSize=2, blocking=False)
            p.start()
        except BaseException:
            self.close()
            raise

    def close(self):
        pipeline, device = self._pipeline, self._device
        self._pipeline = self._device = self._queue = None
        self._last_sequences = self._K = None
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            if device is not None:
                device.close()

    def _grab(self):
        if self._queue is None:
            raise CameraError("OAK camera is not open")
        try:
            deadline = time.monotonic() + 1.0
            group = None
            while time.monotonic() < deadline:
                group = self._queue.tryGet()
                if group is not None:
                    break
                time.sleep(0.005)
            if group is None:
                raise CameraError("OAK: timed out waiting for synchronized RGB-D")
            rgb, depth = group["rgb"], group["depth"]
            sequences = (rgb.getSequenceNum(), depth.getSequenceNum())
            if self._last_sequences is not None and any(
                a <= b for a, b in zip(sequences, self._last_sequences)
            ):
                raise CameraError("OAK: duplicate or reversed capture sequence")
            frame = _decode_pair(
                rgb,
                depth,
                now_sdk=self._dai.Clock.now().total_seconds(),
                receipt=time.monotonic(),
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                max_skew=self.max_skew,
                max_age=self.max_age,
                serial=self.serial,
            )
            self._last_sequences = sequences
            self._K = frame.K
            frame.capture["stereo"] = dict(self.device_info.get("stereo", {}))
            frame.capture["sensor_calibration_sha256"] = self.device_info.get(
                "sensor_calibration_sha256"
            )
            return frame
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(f"OAK capture failed: {exc}") from exc
