"""Timed velocity commands polled by the robot control loop."""

import time

import numpy as np

_velocity = np.zeros(3, dtype=np.float32)
_deadline = 0.0


def timed_vel_cmd(
    vx: float,
    vy: float,
    wz: float,
    duration: float = 1.0,
    new_command: bool = False,
) -> np.ndarray:
    """Return the active [vx, vy, wz] command, or zero after it expires.

    Set new_command=True to start or replace a command. Otherwise the inputs
    are ignored and the current command is polled without extending its timer.
    Duration is measured in monotonic wall-clock seconds. A zero duration
    cancels movement immediately. Call from a single robot control thread.
    """
    global _velocity, _deadline

    now = time.monotonic()
    if new_command:
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")
        velocity = np.asarray([vx, vy, wz], dtype=np.float32)
        if not np.isfinite(velocity).all():
            raise ValueError("velocity components must be finite")
        _velocity = velocity
        _deadline = now + duration

    if now >= _deadline:
        _velocity = np.zeros(3, dtype=np.float32)
    return _velocity.copy()
