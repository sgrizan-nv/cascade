"""A streamed native renderer must be freed before its owning thread exits."""
import threading
from types import SimpleNamespace

import pytest

from cascade.perception.mujoco_camera import MujocoCamera
from cascade.perception.stream import CameraStream


def test_renderer_freed_on_capture_thread_before_world_close():
    camera = object.__new__(MujocoCamera)
    camera._lock = threading.RLock()
    calls = []
    stream = CameraStream(camera, name='test')

    def pump():
        camera._render_owner = threading.current_thread()
        camera._renderer = SimpleNamespace(close=lambda: calls.append(threading.current_thread()))

    stream._pump = pump
    camera.open = lambda: None
    camera.close = lambda: calls.append('world closed')
    stream.open()
    owner = stream._thread
    stream.close()
    assert calls == [owner, 'world closed']
    assert camera._renderer is None
    assert camera._render_owner is None


def test_capture_cleanup_failure_is_reported_without_cross_thread_retry():
    def fail():
        raise RuntimeError('native release failed')

    camera = SimpleNamespace(open=lambda: None, release_capture_resources=fail,
                             close=lambda: pytest.fail('unsafe cross-thread retry'))
    stream = CameraStream(camera, name='test')
    stream._pump = lambda: None
    stream.open()
    with pytest.raises(RuntimeError, match='capture resource cleanup failed'):
        stream.close()
