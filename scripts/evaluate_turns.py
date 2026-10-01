#!/usr/bin/env python3
"""Matched open-loop and closed-loop yaw trials on the flat motion map.

The open-loop duration is fixed from the upstream command scale (1.0 command
unit maps to a 0.25 rad/s policy yaw reference): t = |angle| / 0.25. This is a
transparent nominal baseline, not per-angle empirical tuning. The closed-loop
safety bound is 18 simulated seconds through 90 degrees and scales linearly
with angle beyond 90 degrees.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minilab.platform.dog_policy_runner import DogPolicyRunner  # noqa: E402
from minilab.motion.skills import MotionSkills, TurnConfig, YawAccumulator  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluation/results/turn_comparison.csv")
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with DogPolicyRunner() as runner:
        skills = MotionSkills(runner, turn_config=TurnConfig(proportional_gain=5.0))
        for angle in (45, -45, 90, -90, 180, -180):
            for trial in range(1, args.trials + 1):
                runner.reset()
                tracker = YawAccumulator(runner.yaw)
                start_time = runner.sim_time
                open_duration = abs(np.radians(angle)) / 0.25
                open_command = 1.0 if angle > 0 else -1.0
                while runner.sim_time - start_time < open_duration:
                    runner.step((0.0, 0.0, open_command))
                    tracker.update(runner.yaw)
                measured = float(np.degrees(tracker.accumulated))
                runner.stop()
                tracker.update(runner.yaw)
                settle_steps = int(np.ceil(0.10 / runner.scene.model.opt.timestep))
                for _ in range(settle_steps):
                    runner.step((0.0, 0.0, 0.0))
                    tracker.update(runner.yaw)
                error = float(angle - measured)
                rows.append({
                    "method": "open_loop",
                    "requested_angle_deg": angle,
                    "trial": trial,
                    "measured_turn_deg": measured,
                    "final_error_deg": error,
                    "simulated_time_s": runner.sim_time - start_time,
                    "success": abs(error) <= skills.turn_config.tolerance_deg,
                })

                runner.reset()
                result = skills.turn(angle)
                rows.append({
                    "method": "closed_loop",
                    "requested_angle_deg": angle,
                    "trial": trial,
                    "measured_turn_deg": result.accumulated_deg,
                    "final_error_deg": result.final_error_deg,
                    "simulated_time_s": result.elapsed_sim_s,
                    "success": result.success,
                })

    fields = ["method", "requested_angle_deg", "trial", "measured_turn_deg",
              "final_error_deg", "simulated_time_s", "success"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[EVAL] csv={args.output} trials_per_angle={args.trials}")
    for row in rows:
        print("[EVAL-RESULT] " + " ".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
