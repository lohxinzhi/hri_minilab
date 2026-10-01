"""Simulation-time-throttled RGB capture from the robot's front camera."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Optional

import mujoco
import numpy as np


FRONT_CAMERA_NAME = "dog_front_camera"


@dataclass(frozen=True)
class FrameMetadata:
    """Capture times for the latest RGB frame.

    ``simulation_time`` is MuJoCo's ``data.time`` in seconds. ``wall_time`` is
    the local monotonic clock in seconds and is suitable for elapsed-time
    measurements within this process (it is not a calendar timestamp).
    """

    simulation_time: float
    wall_time: float


class FrontCamera:
    """Render RGB frames from the required ``dog_front_camera`` MuJoCo camera.

    The camera must already exist in ``model``. This class never falls back to
    a tracking/external camera. Rendering is throttled against simulation time;
    call :meth:`update` after physics steps and it renders at most once per
    ``1 / render_hz`` simulated seconds.

    Frames are H x W x 3 NumPy arrays in RGB channel order with ``uint8`` dtype.
    :meth:`get_latest_frame` returns a safe copy by default. Use
    ``copy=False`` for read-only-by-convention access to the stored latest
    image without an additional copy.

    The caller owns the MuJoCo model and data. Create and update the renderer
    on the same thread, then call :meth:`close` before releasing the model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_name: str = FRONT_CAMERA_NAME,
        width: int = 640,
        height: int = 480,
        render_hz: float = 15.0,
    ) -> None:
        if camera_name != FRONT_CAMERA_NAME:
            raise ValueError(
                f"FrontCamera requires {FRONT_CAMERA_NAME!r}; got {camera_name!r}"
            )
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if not np.isfinite(render_hz) or not 0 < render_hz <= 20:
            raise ValueError("render_hz must be finite and in the range (0, 20]")

        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, FRONT_CAMERA_NAME
        )
        if camera_id < 0:
            raise ValueError(
                f"MuJoCo model does not contain required camera "
                f"{FRONT_CAMERA_NAME!r}"
            )

        self.camera_name = FRONT_CAMERA_NAME
        self.width = int(width)
        self.height = int(height)
        self.channels = 3
        self.dtype = np.dtype(np.uint8)
        self.render_hz = float(render_hz)
        self._interval = 1.0 / self.render_hz
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
        self._renderer: Optional[mujoco.Renderer] = mujoco.Renderer(
            model, width=self.width, height=self.height
        )
        self._latest_frame: Optional[np.ndarray] = None
        self.latest_metadata: Optional[FrameMetadata] = None
        self.frames_captured = 0
        self._next_capture_time: Optional[float] = None
        self._last_simulation_time: Optional[float] = None

    def update(self, data: mujoco.MjData) -> bool:
        """Capture when due; return whether a new frame was rendered."""
        if self._renderer is None:
            raise RuntimeError("FrontCamera is closed")

        simulation_time = float(data.time)
        if (
            self._last_simulation_time is not None
            and simulation_time < self._last_simulation_time
        ):
            # MuJoCo reset: capture the new episode immediately.
            self._next_capture_time = None
        self._last_simulation_time = simulation_time
        if (
            self._next_capture_time is not None
            and simulation_time + 1e-12 < self._next_capture_time
        ):
            return False

        self._renderer.update_scene(data, camera=FRONT_CAMERA_NAME)
        frame = np.asarray(self._renderer.render())
        if frame.shape != (self.height, self.width, self.channels):
            raise RuntimeError(
                f"MuJoCo returned unexpected RGB frame shape {frame.shape}; "
                f"expected {(self.height, self.width, self.channels)}"
            )
        if frame.dtype != self.dtype:
            raise RuntimeError(
                f"MuJoCo returned unexpected frame dtype {frame.dtype}; "
                f"expected {self.dtype}"
            )

        # Renderer.render() produces an image array; retain that result and
        # copy only when a caller requests a safe mutable image.
        self._latest_frame = frame
        self.latest_metadata = FrameMetadata(
            simulation_time=simulation_time,
            wall_time=monotonic(),
        )
        self.frames_captured += 1
        self._next_capture_time = simulation_time + self._interval
        return True

    def get_latest_frame(self, *, copy: bool = True) -> Optional[np.ndarray]:
        """Return the latest RGB image, copied by default for caller safety."""
        if self._latest_frame is None:
            return None
        return self._latest_frame.copy() if copy else self._latest_frame

    def close(self) -> None:
        """Release MuJoCo renderer resources; safe to call more than once."""
        renderer, self._renderer = self._renderer, None
        self._latest_frame = None
        self.latest_metadata = None
        if renderer is not None:
            renderer.close()

    def __enter__(self) -> "FrontCamera":
        if self._renderer is None:
            raise RuntimeError("FrontCamera is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
