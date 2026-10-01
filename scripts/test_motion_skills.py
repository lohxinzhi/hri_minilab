#!/usr/bin/env python3
"""Run one selected Task 2 move/turn skill headlessly or with a live viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minilab.platform.dog_policy_runner import DogPolicyRunner  # noqa: E402
from minilab.motion.skills import MotionSkills  # noqa: E402


MOTIONS = {
    "forward": ("move", (0.5, 0.0, 0.0)),
    "backward": ("move", (-0.5, 0.0, 0.0)),
    "strafe-left": ("move", (0.0, 0.5, 0.0)),
    "strafe-right": ("move", (0.0, -0.5, 0.0)),
    "turn+90": ("turn", 90.0),
    "turn-90": ("turn", -90.0),
    "turn+180": ("turn", 180.0),
    "turn-180": ("turn", -180.0),
}


class UnsafeVisualState(RuntimeError):
    """A diagnostic safety limit was crossed during a visual trial."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", choices=("headless", "native", "browser"),
                        default="headless")
    parser.add_argument("--motion", choices=tuple(MOTIONS), required=True,
                        help="run exactly one motion from a reset start pose")
    args = parser.parse_args()

    # Diagnostic abort thresholds; these do not affect controller results:
    # trunk z < 0.15 m, |roll| or |pitch| > 45 degrees, or non-finite state.
    minimum_height = math.inf
    visual_fault = None
    with DogPolicyRunner(viewer_mode=args.viewer) as runner:
        skills = MotionSkills(runner)
        runner.reset()
        initial_height = float(runner.position[2])
        initial_yaw = runner.yaw
        turn_result = None

        def observe(state, command):
            nonlocal minimum_height, visual_fault
            data = runner.scene.data
            roll, pitch = runner.roll_pitch
            finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
            if finite:
                minimum_height = min(minimum_height, float(state["position"][2]))
            unsafe = (
                not finite
                or float(state["position"][2]) < 0.15
                or abs(roll) > math.radians(45.0)
                or abs(pitch) > math.radians(45.0)
            )
            if unsafe and visual_fault is None:
                if not finite:
                    visual_fault = "non-finite MuJoCo state"
                elif float(state["position"][2]) < 0.15:
                    visual_fault = f"base height {state['position'][2]:.3f} m < 0.15 m"
                else:
                    visual_fault = (
                        "roll/pitch exceeded 45 deg "
                        f"(roll={math.degrees(roll):.1f}, pitch={math.degrees(pitch):.1f})"
                    )
                raise UnsafeVisualState(visual_fault)

        runner.step_observer = observe
        try:
            # Neutral warm-up is measured in simulator time, never sleep().
            for _ in range(round(0.5 / runner.scene.model.opt.timestep)):
                runner.step((0.0, 0.0, 0.0))
            initial_height = float(runner.position[2])
            initial_yaw = runner.yaw
            minimum_height = initial_height
            print(
                f"[VISUAL] motion={args.motion} viewer={args.viewer} "
                f"initial_height={initial_height:.3f} "
                f"initial_yaw={math.degrees(initial_yaw):.2f}deg"
            )

            kind, spec = MOTIONS[args.motion]
            if kind == "move":
                vx, vy, wz = spec
                skills.move(vx, vy, wz, duration=1.0)
            else:
                turn_result = skills.turn(spec)

            # Observe neutral recovery for two additional simulated seconds.
            for _ in range(round(2.0 / runner.scene.model.opt.timestep)):
                runner.step((0.0, 0.0, 0.0))
        except UnsafeVisualState as exc:
            print(f"[SAFETY] stopped={exc}")
        finally:
            runner.step_observer = None

        final_height = float(runner.position[2])
        final_yaw = runner.yaw
        roll, pitch = runner.roll_pitch
        print(
            f"[VISUAL] motion={args.motion} min_height={minimum_height:.3f} "
            f"final_height={final_height:.3f} "
            f"initial_yaw={math.degrees(initial_yaw):.2f}deg "
            f"final_yaw={math.degrees(final_yaw):.2f}deg "
            f"roll={math.degrees(roll):.2f}deg pitch={math.degrees(pitch):.2f}deg"
        )
        if turn_result is not None:
            print(
                f"[VISUAL-TURN] requested={turn_result.target_deg:.1f}deg "
                f"measured={turn_result.accumulated_deg:.2f}deg "
                f"final_error={turn_result.final_error_deg:.2f}deg "
                f"status={turn_result.status}"
            )


if __name__ == "__main__":
    main()
