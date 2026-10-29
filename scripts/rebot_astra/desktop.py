"""Read-only native viewer in its own process, isolated from RGB-D EGL contexts.

The parent remains the sole physics owner. Only measured state is copied to
the viewer; GUI controls never command the experiment's arm.
"""
import multiprocessing
import threading

FIELDS = ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')


def _show(scene, shared, lock, ready, stop, errors):
    import os
    os.environ['MUJOCO_GL'] = 'glfw'
    os.environ['PYGLFW_LIBRARY_VARIANT'] = 'x11'
    viewer_threads = []
    try:
        import glfw
        import mujoco
        import mujoco.viewer
        import numpy as np

        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        before = set(threading.enumerate())
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer_threads = [t for t in threading.enumerate() if t not in before]
            with viewer.lock():
                viewer.cam.lookat[:] = [0.25, 0.02, 0.20]
                viewer.cam.distance = 1.15
                viewer.cam.azimuth, viewer.cam.elevation = 135, -25
            ready.set()
            while not stop.is_set() and viewer.is_running():
                with lock:
                    values = np.frombuffer(shared).copy()
                with viewer.lock():
                    offset = 0
                    for field in FIELDS:
                        array = getattr(data, field)
                        array[:] = values[offset:offset + array.size].reshape(array.shape)
                        offset += array.size
                    data.time = values[-1]
                    mujoco.mj_forward(model, data)
                viewer.sync()
                stop.wait(.05)
    except Exception as exc:
        errors.put(f'{type(exc).__name__}: {exc}')
    finally:
        # Handle.close() only signals the passive viewer. Its render thread
        # must finish before Python's atexit glfw.terminate destroys windows.
        for thread in viewer_threads:
            thread.join(timeout=10)
            if thread.is_alive():
                errors.put('MuJoCo viewer render thread did not stop')
        stop.set()
        ready.set()


class DesktopView:
    def __init__(self, world):
        self.world = world
        ctx = multiprocessing.get_context('spawn')
        self.shared = ctx.RawArray('d', sum(getattr(world.data, f).size for f in FIELDS) + 1)
        self.lock = ctx.Lock()
        self.ready, self.stop_event = ctx.Event(), ctx.Event()
        self.errors = ctx.Queue()
        self.process = ctx.Process(target=_show, args=(world.path, self.shared, self.lock,
                                   self.ready, self.stop_event, self.errors))
        self.thread = None
        self.sync()
        self.process.start()
        if not self.ready.wait(20) or self.stop_event.is_set():
            self.close()
            raise RuntimeError('MuJoCo desktop viewer failed to start')
        self.thread = threading.Thread(target=self._pump, name='demo-viewer-state', daemon=True)
        self.thread.start()

    @property
    def pid(self):
        return self.process.pid

    def is_running(self):
        return self.process.is_alive() and not self.stop_event.is_set()

    def sync(self):
        import numpy as np
        with self.world.lock:
            values = np.concatenate([getattr(self.world.data, f).reshape(-1) for f in FIELDS]
                                    + [np.array([self.world.data.time])])
        # A failed viewer must not block control or cleanup.
        if self.lock.acquire(timeout=.2):
            try:
                np.frombuffer(self.shared)[:] = values
            finally:
                self.lock.release()

    def _pump(self):
        while not self.stop_event.wait(.05) and self.process.is_alive():
            self.sync()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.process.join(timeout=15)
        forced = self.process.is_alive()
        if forced:
            self.process.terminate()
            self.process.join(timeout=5)
        from queue import Empty
        try:
            error = self.errors.get_nowait()
        except Empty:
            error = None
        self.errors.close()
        if forced or self.process.exitcode != 0 or error:
            raise RuntimeError(error or f'Viewer failed to exit cleanly ({self.process.exitcode})')
