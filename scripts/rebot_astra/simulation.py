"""Native MuJoCo diagnostics used only by the optional experiment runner.

Private engine stepping is confined here. This is deliberately not an ArmBase
extension: advancing a held scene is a simulation diagnostic. The desktop
viewer is an isolated read-only observer of the same measured state.
"""


class NativeSimulation:
    def __init__(self, arm):
        self.arm = arm
        self.world = arm.world
        if self.world is None:
            raise ValueError('The experiment requires the native MuJoCo C engine')
        self.viewer = None

    def configure_viewer(self):
        from .desktop import DesktopView
        self.viewer = DesktopView(self.world)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def snapshot(self, runtime, path):
        import cv2

        frame = runtime.observe(require_fresh=True)
        if not cv2.imwrite(str(path), frame.rgb):
            raise OSError(f'Could not save camera frame: {path}')
        with self.world.lock:
            self.world.mj.mj_forward(self.world.model, self.world.data)
            return {'sim_time_s': float(self.world.data.time),
                    'cube_xyz': self.world.data.body('red_cube').xpos.tolist(),
                    'q': self.arm.get_state().q.tolist()}

    def hold(self, seconds):
        with self.world.lock:
            self.arm._engine.step(int(seconds / self.world.model.opt.timestep))
