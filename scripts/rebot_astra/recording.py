"""Record the visible MuJoCo X11 window to H.264, without capturing the desktop."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time


class WindowRecording:
    def __init__(self, path: Path, fps: int = 20, target_pid: int | None = None):
        self.path, self.fps = path, fps
        self.target_pid = target_pid if target_pid is not None else os.getpid()
        self.frames = 0
        self.error = None
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='mujoco-window-recording')

    def start(self):
        self.thread.start()
        if not self.ready.wait(15):
            self.stop()
            raise RuntimeError('Timed out opening the MuJoCo window recorder')
        if self.error:
            self.stop()
            raise RuntimeError(self.error)
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Video encoder did not finish')
        if self.error:
            raise RuntimeError(self.error)
        if self.frames == 0:
            raise RuntimeError('No MuJoCo window frames captured')

    def _run(self):
        connection = writer = None
        try:
            import cv2
            import imageio_ffmpeg
            import numpy as np
            from Xlib import X, display, protocol

            connection = display.Display()
            root = connection.screen().root
            pid_atom = connection.intern_atom('_NET_WM_PID')
            window = None
            deadline = time.monotonic() + 10
            def descendants(parent):
                for child in parent.query_tree().children:
                    yield child
                    yield from descendants(child)
            while window is None and time.monotonic() < deadline and not self.stop_event.is_set():
                for child in descendants(root):
                    title = child.get_wm_name() or ''
                    pid = child.get_full_property(pid_atom, X.AnyPropertyType)
                    if 'mujoco' in title.lower() and pid is not None and int(pid.value[0]) == self.target_pid:
                        window = child
                        break
                if window is None:
                    self.stop_event.wait(.1)
            if window is None:
                raise RuntimeError(f'No visible MuJoCo window belonging to PID {self.target_pid}')
            window.configure(x=60, y=70, width=1440, height=960, stack_mode=X.Above)
            root.send_event(protocol.event.ClientMessage(
                window=window, client_type=connection.intern_atom('_NET_ACTIVE_WINDOW'),
                data=(32, [1, X.CurrentTime, 0, 0, 0])),
                event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
            connection.sync()
            self.stop_event.wait(.5)
            geometry = window.get_geometry()
            width, height = geometry.width // 2 * 2, geometry.height // 2 * 2
            writer = imageio_ffmpeg.write_frames(
                str(self.path), (width, height), fps=self.fps, codec='libx264',
                pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=8,
                macro_block_size=2, output_params=['-movflags', '+faststart'])
            writer.send(None)
            self.ready.set()
            start = time.monotonic()
            while not self.stop_event.is_set():
                geometry = window.get_geometry()
                shot = window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xffffffff)
                # Xwayland's 24-bit TrueColor drawable is stored as 32-bit BGRX.
                pixels = np.frombuffer(shot.data, np.uint8).reshape(geometry.height, geometry.width, 4)
                rgb = pixels[:, :, :3][:, :, ::-1].copy()
                if rgb.shape[:2] != (height, width):
                    rgb = cv2.resize(rgb, (width, height))
                if self.frames == 0:
                    cv2.imwrite(str(self.path.with_suffix('.png')), rgb[:, :, ::-1])
                writer.send(rgb)
                self.frames += 1
                self.stop_event.wait(max(0, start + self.frames / self.fps - time.monotonic()))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.ready.set()
        finally:
            try:
                if writer is not None:
                    writer.close()
            except Exception as exc:
                self.error = self.error or f'{type(exc).__name__}: {exc}'
            finally:
                try:
                    if connection is not None:
                        connection.close()
                except Exception as exc:
                    self.error = self.error or f'{type(exc).__name__}: {exc}'
                self.ready.set()
