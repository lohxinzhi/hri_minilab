"""Timed velocity and closed-loop heading commands for the robot control loop."""

import time

import numpy as np

_velocity = np.zeros(3, dtype=np.float32)
_deadline = 0.0
_turn_angle = None
_turn_remaining = 0.0
_turn_heading = 0.0
_turn_target_heading = 0.0
_turn_log_time = None
_yaw_integral = 0.0
_yaw_time = None


def move(
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
    global _velocity, _deadline, _turn_angle

    now = time.monotonic()
    if new_command:
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")
        velocity = np.asarray([vx, vy, wz], dtype=np.float32)
        if not np.isfinite(velocity).all():
            raise ValueError("velocity components must be finite")
        _velocity = velocity
        _deadline = now + duration
        _turn_angle = None

    if now >= _deadline:
        _velocity = np.zeros(3, dtype=np.float32)
    return _velocity.copy()


def is_move_active() -> bool:
    """Return whether the timed move is still running, including zero-speed waits."""
    return time.monotonic() < _deadline


def get_turn_target_heading() -> float | None:
    """Return the captured heading target, or None when no turn is active."""
    return _turn_target_heading if _turn_angle is not None else None


def turn(
    yaw_deg: float,
    current_yaw_deg: float,
    *,
    new_command: bool = False,
    yaw_rate_deg_s: float = 0.0,
    kp: float = 2.0,
    ki: float = 0.5,
    kd: float = 0.2,
    max_wz: float = 1.0,
    tolerance_deg: float = 1.0,
    dt: float | None = None,
) -> np.ndarray:
    """Return [0, 0, wz] for a relative turn using bounded PID control.

    Headings and measured yaw rate use degrees and degrees/second; wz uses
    radians/second. Positive yaw_deg turns counterclockwise; negative turns
    clockwise. A new command captures the current heading once. Poll with
    the same angle and latest heading without shifting the target. Set
    new_command=True to repeat an angle or replace the active turn. The first
    call, or a changed angle, also starts a turn. Measured heading changes are
    unwrapped, allowing turns larger than 180 degrees and crossing +/-180.
    dt is simulation seconds per update, or monotonic elapsed time if omitted.
    Call from one control thread; there is one active heading controller.
    """
    global _turn_angle, _turn_remaining, _turn_heading, _yaw_integral, _yaw_time
    global _turn_target_heading, _turn_log_time
    global _velocity, _deadline
    values = (
        yaw_deg,
        current_yaw_deg,
        yaw_rate_deg_s,
        kp,
        ki,
        kd,
        max_wz,
        tolerance_deg,
    )
    if not np.isfinite(values).all():
        raise ValueError("yaw controller inputs must be finite")
    if kp < 0 or ki < 0 or kd < 0 or max_wz <= 0 or tolerance_deg < 0:
        raise ValueError("gains/tolerance must be non-negative and max_wz positive")
    if dt is not None and (not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be finite and positive")
    now = time.monotonic()
    if new_command or _turn_angle != yaw_deg:
        _turn_angle, _turn_remaining, _turn_heading = yaw_deg, yaw_deg, current_yaw_deg
        _turn_target_heading = current_yaw_deg + yaw_deg
        _turn_log_time = None
        _yaw_integral, _yaw_time = 0.0, now
        _velocity, _deadline = np.zeros(3, dtype=np.float32), now
    else:
        change = (current_yaw_deg - _turn_heading + 180.0) % 360.0 - 180.0
        _turn_remaining -= change
        _turn_heading = current_yaw_deg
    elapsed = dt if dt is not None else min(max(now - _yaw_time, 0.0), 0.1)
    _yaw_time = now
    error_deg = _turn_remaining
    # Use continuous headings so target - current also works across +/-180.
    # Log new commands immediately, then at most 10 Hz during control updates.
    if _turn_log_time is None or now - _turn_log_time >= 0.1:
        current_heading = _turn_target_heading - error_deg
        print(
            f"[TURN] target = {_turn_target_heading:.2f} "
            f"current = {current_heading:.2f} error = {error_deg:.2f}"
        )
        _turn_log_time = now
    if abs(error_deg) <= tolerance_deg:
        _yaw_integral = 0.0
        return np.zeros(3, dtype=np.float32)
    error = np.deg2rad(error_deg)
    pd = kp * error - kd * np.deg2rad(yaw_rate_deg_s)
    candidate = _yaw_integral + error * elapsed
    if ki > 0:
        candidate = float(np.clip(candidate, -max_wz / ki, max_wz / ki))
    else:
        candidate = 0.0
    # Do not wind up the integrator while turning at the speed limit.
    raw = pd + ki * candidate
    if abs(raw) <= max_wz or raw * error <= 0:
        _yaw_integral = candidate
    wz = pd + ki * _yaw_integral
    return np.array([0.0, 0.0, np.clip(wz, -max_wz, max_wz)], dtype=np.float32)
