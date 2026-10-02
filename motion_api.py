"""Timed velocity and closed-loop heading commands for the robot control loop."""

import time
from dataclasses import dataclass

import numpy as np

GOTO_TIMEOUT_SECONDS = 60.0  # Shared wall-clock limit for search and approach.
APPROACH_DISTANCE_METERS = 0.8

_velocity = np.zeros(3, dtype=np.float32)
_deadline = 0.0
_turn_angle = None
_turn_remaining = 0.0
_turn_heading = 0.0
_turn_target_heading = 0.0
_turn_log_time = None
_yaw_integral = 0.0
_yaw_time = None


@dataclass
class _ObjectMission:
    object_type: str
    object_color: str
    deadline: float
    timeout: float
    phase: str | None = None
    status: str = "RUNNING"
    reason: str | None = None


_object_mission: _ObjectMission | None = None


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


def _normalized_bbox(bbox, frame_size):
    """Clip a pixel box to the frame and normalize by (width, height)."""
    size = np.asarray(frame_size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("frame_size must contain a finite positive width and height")
    if bbox is None:
        return None
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bbox must contain four finite pixel coordinates")
    box = np.clip(box / np.tile(size, 2), 0, 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox must have positive width and height inside the frame")
    return box


def _object_distance(robot_position, object_position):
    robot = np.asarray(robot_position, dtype=float)
    target = np.asarray(object_position, dtype=float)
    if (
        robot.shape != (2,)
        or target.shape != (2,)
        or not np.isfinite([robot, target]).all()
    ):
        raise ValueError(
            "robot_position and object_position must be finite 2D coordinates"
        )
    return float(np.linalg.norm(target - robot))


def approach(
    object_type: str,
    object_color: str,
    bbox,
    *,
    robot_position,
    object_position,
    frame_size=(640, 480),
    new_command: bool = False,
) -> np.ndarray:
    """Steer toward a visible target using its pixel box and (width, height).

    Horizontal error is box-center x / frame width minus 0.5. Positive
    image error requires a clockwise (negative wz) turn. Forward speed is
    0.5 m/s when abs(error) < 0.25; yaw rate is -2 * error rad/s.
    Stop when the world-frame 2D distance is strictly below 0.8 m. A missing box stops
    movement; goto_object handles the transition back to search.
    """
    box = _normalized_bbox(bbox, frame_size)
    distance = _object_distance(robot_position, object_position)
    if new_command:
        print(f"[APPROACH] object={object_type} color={object_color}")
    if box is None or distance < APPROACH_DISTANCE_METERS:
        return move(0, 0, 0, duration=0, new_command=True)
    error = (box[0] + box[2]) / 2 - 0.5
    vx = 0.5 if abs(error) < 0.25 else 0.0
    return move(vx, 0, -2.0 * error, new_command=True)


def search(
    current_yaw_deg: float,
    *,
    new_command: bool = False,
    yaw_rate_deg_s: float = 0.0,
    dt: float | None = None,
) -> np.ndarray:
    """Poll a single 360-degree counterclockwise search using turn().

    Set new_command=True on search entry, then poll with updated heading.
    A zero command signals that the search revolution is complete.
    """
    command = turn(
        360,
        current_yaw_deg,
        new_command=new_command,
        yaw_rate_deg_s=yaw_rate_deg_s,
        dt=dt,
    )
    if new_command:
        print("[SEARCH]")
    return command


def get_goto_status() -> str | None:
    """Return RUNNING, SUCCESS, FAIL, or None before the first object mission."""
    return _object_mission.status if _object_mission is not None else None


def get_goto_reason() -> str | None:
    """Return the most recent mission's failure reason, when available."""
    return _object_mission.reason if _object_mission is not None else None


def cancel_goto_object() -> np.ndarray:
    """Cancel a running mission and stop its motion from the control thread."""
    if _object_mission is not None and _object_mission.status == "RUNNING":
        return _finish_object_mission("FAIL", "cancelled")
    return move(0, 0, 0, duration=0, new_command=True)


def _finish_object_mission(status, reason=None):
    _object_mission.status = status
    _object_mission.reason = reason
    command = move(0, 0, 0, duration=0, new_command=True)
    message = f"[MISSION] status={status}"
    if reason is not None:
        message += f" reason={reason}"
    print(message)
    return command


def goto_object(
    object_type: str,
    object_color: str,
    bbox=None,
    *,
    current_yaw_deg: float,
    robot_position,
    object_position,
    frame_size=(640, 480),
    new_command: bool = False,
    yaw_rate_deg_s: float = 0.0,
    dt: float | None = None,
) -> np.ndarray:
    """Poll a nonblocking search/approach mission with a configurable timeout.

    object_type is the target COCO class and object_color its requested color.
    The caller selects a matching vision detection and supplies its current
    pixel bbox, or None when no matching object is visible. Supply the current
    yaw in degrees and camera frame_size=(width, height) on each control update.
    robot_position and object_position are world-frame (x, y) in meters; the
    caller obtains the target position from the scene YAML. Completion depends
    only on their distance being strictly below APPROACH_DISTANCE_METERS.
    Physics and vision must continue updating between calls on the control
    thread; this function does not render cameras or block the simulation.

    The first call or a changed target starts a mission. Set new_command=True
    once to explicitly restart the same target. Terminal calls return zeros
    without repeating mission logs; get_goto_status() reports the outcome.
    GOTO_TIMEOUT_SECONDS sets the deadline when a mission starts; it is shared
    across all approach/search transitions.
    """
    global _object_mission
    if not isinstance(object_type, str) or not object_type.strip():
        raise ValueError("object_type must be a nonempty COCO class name")
    if not isinstance(object_color, str) or not object_color.strip():
        raise ValueError("object_color must be nonempty text")
    if not np.isfinite([current_yaw_deg, yaw_rate_deg_s]).all():
        raise ValueError("heading and yaw rate must be finite")
    if dt is not None and (not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be finite and positive")
    box = _normalized_bbox(bbox, frame_size)
    distance = _object_distance(robot_position, object_position)
    now = time.monotonic()
    if (
        new_command
        or _object_mission is None
        or (_object_mission.object_type, _object_mission.object_color)
        != (object_type, object_color)
    ):
        if not np.isfinite(GOTO_TIMEOUT_SECONDS) or GOTO_TIMEOUT_SECONDS <= 0:
            raise ValueError("GOTO_TIMEOUT_SECONDS must be finite and positive")
        if _object_mission is not None and _object_mission.status == "RUNNING":
            _finish_object_mission("FAIL", "replaced by a new mission")
        _object_mission = _ObjectMission(
            object_type, object_color, now + GOTO_TIMEOUT_SECONDS, GOTO_TIMEOUT_SECONDS
        )
    if _object_mission.status != "RUNNING":
        return np.zeros(3, dtype=np.float32)
    if now >= _object_mission.deadline:
        return _finish_object_mission(
            "FAIL", f"{_object_mission.timeout:g}-second timeout"
        )
    phase = (
        "approach"
        if box is not None or distance < APPROACH_DISTANCE_METERS
        else "search"
    )
    entering = _object_mission.phase != phase
    _object_mission.phase = phase
    if phase == "approach":
        command = approach(
            object_type,
            object_color,
            bbox,
            robot_position=robot_position,
            object_position=object_position,
            frame_size=frame_size,
            new_command=entering,
        )
        if distance < APPROACH_DISTANCE_METERS:
            return _finish_object_mission("SUCCESS")
        return command
    command = search(
        current_yaw_deg,
        new_command=entering,
        yaw_rate_deg_s=yaw_rate_deg_s,
        dt=dt,
    )
    if not np.any(command):
        return _finish_object_mission("FAIL", "object not found after one revolution")
    return command
