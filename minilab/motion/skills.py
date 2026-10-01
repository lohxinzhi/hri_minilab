"""Reusable velocity and closed-loop yaw skills for DogPolicyRunner."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_to_pi(angle):
    """Wrap radians to the half-open interval [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_wxyz_to_yaw(quaternion):
    """Return yaw from MuJoCo's free-joint quaternion order [w, x, y, z]."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must contain four finite values in wxyz order")
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class YawAccumulator:
    """Accumulate actual wrapped yaw increments without losing full turns."""

    def __init__(self, initial_yaw):
        if not math.isfinite(float(initial_yaw)):
            raise ValueError("initial_yaw must be finite")
        self.previous_yaw = float(initial_yaw)
        self.accumulated = 0.0

    def update(self, yaw_now):
        yaw_now = float(yaw_now)
        if not math.isfinite(yaw_now):
            raise ValueError("yaw must be finite")
        self.accumulated += wrap_to_pi(yaw_now - self.previous_yaw)
        self.previous_yaw = yaw_now
        return self.accumulated


@dataclass(frozen=True)
class TurnConfig:
    proportional_gain: float = 5.0  # normalized policy yaw reference per radian
    max_yaw_command: float = 1.0
    tolerance_deg: float = 5.0
    settle_margin_deg: float = 0.5
    yaw_rate_threshold: float = 0.10  # rad/s from MuJoCo free-joint qvel[5]
    stable_policy_updates: int = 3
    timeout_sim_s: float = 18.0  # base safety bound; scales with requested angle
    neutral_settle_sim_s: float = 0.10


@dataclass(frozen=True)
class TurnResult:
    target_deg: float
    accumulated_deg: float
    final_error_deg: float
    elapsed_sim_s: float
    success: bool
    status: str


class MotionSkills:
    """Execute policy command references through an object exposing step().

    vx/vy/wz are policy command references. They are not promises about
    measured chassis velocity; the pretrained policy determines actual motion.
    """

    def __init__(self, runner, *, turn_config=TurnConfig()):
        self.runner = runner
        self.turn_config = turn_config

    def _advance_for(self, duration, command):
        start = self.runner.sim_time
        while self.runner.sim_time - start + 1e-12 < duration:
            self.runner.step(command)
        return self.runner.sim_time - start

    def _neutralize(self, settle_sim_s=0.0):
        self.runner.stop()
        timestep = float(self.runner.scene.model.opt.timestep)
        steps = max(0, int(math.ceil(max(0.0, settle_sim_s) / timestep)) - 1)
        for _ in range(steps):
            self.runner.step((0.0, 0.0, 0.0))

    def move(self, vx, vy, wz, duration):
        values = np.asarray([vx, vy, wz, duration], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("move inputs must be finite numeric values")
        if float(duration) < 0.0:
            raise ValueError("duration must be non-negative")
        command = self.runner.validate_command(values[:3])
        start = self.runner.sim_time
        try:
            if duration > 0:
                self._advance_for(float(duration), command)
        finally:
            self._neutralize(self.turn_config.neutral_settle_sim_s)
        elapsed = self.runner.sim_time - start
        print(
            f"[MOVE] vx={vx:.3f} vy={vy:.3f} wz={wz:.3f} "
            f"duration={duration:.3f} elapsed_sim={elapsed:.3f}"
        )
        return elapsed

    def turn(self, angle_deg):
        angle_deg = float(angle_deg)
        if not math.isfinite(angle_deg):
            raise ValueError("turn angle must be finite")
        cfg = self.turn_config
        if cfg.proportional_gain <= 0 or cfg.max_yaw_command <= 0:
            raise ValueError("turn gain and command limit must be positive")
        if cfg.max_yaw_command > 1.0:
            raise ValueError("max_yaw_command must not exceed the runner's +/-1 limit")
        if cfg.tolerance_deg <= 0 or cfg.timeout_sim_s <= 0:
            raise ValueError("turn tolerance and timeout must be positive")
        if not 0 <= cfg.settle_margin_deg < cfg.tolerance_deg:
            raise ValueError("settle_margin_deg must be in [0, tolerance_deg)")

        target = math.radians(angle_deg)
        # A half-turn needs roughly twice the policy motion time of a quarter
        # turn. Keep the configured bound for <=90 degrees and scale it
        # linearly for larger requests so ±180° remain feasible and bounded.
        timeout_limit = cfg.timeout_sim_s * max(1.0, abs(target) / (math.pi / 2.0))
        tracker = YawAccumulator(self.runner.yaw)
        start = self.runner.sim_time
        stable_count = 0
        status = "timeout"
        error = target
        try:
            if abs(target) <= math.radians(cfg.tolerance_deg):
                status = "success"
            else:
                while self.runner.sim_time - start < timeout_limit:
                    error = target - tracker.accumulated
                    raw_wz = cfg.proportional_gain * error
                    command = float(np.clip(
                        raw_wz, -cfg.max_yaw_command, cfg.max_yaw_command
                    ))
                    self.runner.step((0.0, 0.0, command))
                    tracker.update(self.runner.yaw)
                    error = target - tracker.accumulated
                    settle_tolerance = math.radians(
                        cfg.tolerance_deg - cfg.settle_margin_deg
                    )
                    at_target = abs(error) <= settle_tolerance
                    settled = abs(self.runner.yaw_rate) <= cfg.yaw_rate_threshold
                    # Count stability only at policy boundaries (50 Hz).
                    if self.runner.step_count % self.runner.decimation == 0:
                        stable_count = stable_count + 1 if at_target and settled else 0
                    if stable_count >= cfg.stable_policy_updates:
                        status = "success"
                        break
        finally:
            self._neutralize(cfg.neutral_settle_sim_s)
            # Include any real orientation change during the neutral settling
            # interval in the final measured error.
            tracker.update(self.runner.yaw)

        elapsed = self.runner.sim_time - start
        final_error = target - tracker.accumulated
        if status == "success" and (
            abs(final_error) > math.radians(cfg.tolerance_deg)
            or abs(self.runner.yaw_rate) > cfg.yaw_rate_threshold
        ):
            status = "not_settled"
        result = TurnResult(
            angle_deg,
            math.degrees(tracker.accumulated),
            math.degrees(final_error),
            elapsed,
            status == "success",
            status,
        )
        print(
            f"[TURN] target={result.target_deg:.1f} "
            f"accumulated={result.accumulated_deg:.2f} "
            f"final_error={result.final_error_deg:.2f} "
            f"elapsed={result.elapsed_sim_s:.2f} status={result.status}"
        )
        return result
