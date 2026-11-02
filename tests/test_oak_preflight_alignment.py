"""A clear route in a misregistered map is not readiness evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.calibration.markerless import OffsetCheck
from cascade.perception.drift_monitor import DriftMonitorConfig
from scripts import preflight_rebot_oak as preflight


def test_preflight_reports_hold_without_clearing_or_connecting(monkeypatch):
    marker = {"reason": "failed park"}
    monkeypatch.setattr(preflight.arm_hold, "hold_key", lambda *a: "can0")
    monkeypatch.setattr(preflight.arm_hold, "read", lambda key: marker)
    monkeypatch.setattr(preflight.arm_hold, "refusal", lambda m, key: f"held on {key}: {m['reason']}")
    monkeypatch.setattr(preflight.arm_hold, "clear", lambda *a, **k: pytest.fail("passive check cleared hold"))
    assert preflight.startup_hold_blocker(SimpleNamespace(type="rebot_rs")) == "held on can0: failed park"
    monkeypatch.setattr(preflight.arm_hold, "read", lambda key: None)
    assert preflight.startup_hold_blocker(SimpleNamespace(type="rebot_rs")) is None


@pytest.mark.parametrize("conclusive,offset,angle,expected", [
    (True, .004, 1.4, True),
    (True, .0164, 1.9, False),
    (True, .004, 3.1, False),
    (False, float("nan"), float("nan"), False),
])
def test_preflight_uses_runtime_alignment_limits(monkeypatch, conclusive, offset, angle, expected):
    seen = []
    q = np.arange(6) * .01
    frame = SimpleNamespace(depth_m=np.ones((3, 4)), K=np.eye(3))
    kin = SimpleNamespace(fk=lambda q: np.eye(4))
    surface, transform = object(), np.eye(4)

    def measure(sample, supplied_surface, supplied_transform):
        seen.append(sample)
        assert supplied_surface is surface and supplied_transform is transform
        return OffsetCheck(conclusive, "test", offset_m=offset, offset_deg=angle,
                           n_inliers_after=500)

    monkeypatch.setattr(preflight, "measure_offset", measure)
    result = preflight.check_alignment(frame, q, kin, surface, transform,
                                      DriftMonitorConfig(enabled=True))
    assert result["passed"] is expected
    assert seen[0].q == tuple(q)
    assert seen[0].depth_m is frame.depth_m
    assert result["max_offset_mm"] == 10.0
    assert result["max_rot_deg"] == 3.0
    if not conclusive:
        assert result["offset_mm"] is None
