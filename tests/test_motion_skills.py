import math
import unittest

import numpy as np
from minilab.motion.skills import (
    MotionSkills,
    YawAccumulator,
    quaternion_wxyz_to_yaw,
    wrap_to_pi,
)


class FakeRunner:
    def __init__(self, fail=False):
        self.scene = type("Scene", (), {"model": type("Model", (), {"opt": type("Opt", (), {"timestep": 0.01})()})()})()
        self.sim_time = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.step_count = 0
        self.decimation = 4
        self.commands = []
        self.fail = fail

    def validate_command(self, command):
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.all(np.isfinite(command)) or np.any(abs(command) > 1):
            raise ValueError("invalid")
        return command

    def step(self, command):
        self.commands.append(tuple(command))
        if self.fail and any(command):
            raise RuntimeError("test failure")
        self.sim_time += 0.01
        self.step_count += 1
        return None

    def stop(self):
        pass


class MotionSkillTests(unittest.TestCase):
    def test_wrap_range(self):
        for angle in (0.0, math.pi / 2, -math.pi / 2, math.pi, 9 * math.pi):
            wrapped = wrap_to_pi(angle)
            self.assertGreaterEqual(wrapped, -math.pi)
            self.assertLess(wrapped, math.pi)

    def test_quaternion_yaw(self):
        for yaw, expected in (
            (0, 0), (math.pi / 2, math.pi / 2),
            (-math.pi / 2, -math.pi / 2), (math.pi, math.pi),
        ):
            quaternion = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            self.assertAlmostEqual(quaternion_wxyz_to_yaw(quaternion), expected)

    def test_yaw_accumulator_positive_negative_and_full_turn(self):
        positive = YawAccumulator(math.radians(170))
        positive.update(math.radians(-170))
        self.assertAlmostEqual(math.degrees(positive.accumulated), 20)
        negative = YawAccumulator(math.radians(-170))
        negative.update(math.radians(170))
        self.assertAlmostEqual(math.degrees(negative.accumulated), -20)
        full_turn = YawAccumulator(0.0)
        for yaw in np.linspace(0, 2 * math.pi, 17)[1:]:
            full_turn.update(wrap_to_pi(yaw))
        self.assertAlmostEqual(math.degrees(full_turn.accumulated), 360)

    def test_zero_degree_turn_neutralizes(self):
        runner = FakeRunner()
        result = MotionSkills(runner).turn(0)
        self.assertTrue(result.success)
        self.assertEqual(runner.commands[-1], (0.0, 0.0, 0.0))

    def test_move_rejects_negative_duration(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MotionSkills(FakeRunner()).move(0, 0, 0, -1)

    def test_move_always_neutralizes_after_exception(self):
        runner = FakeRunner(fail=True)
        with self.assertRaises(RuntimeError):
            MotionSkills(runner).move(0.5, 0, 0, 0.1)
        self.assertEqual(runner.commands[-1], (0.0, 0.0, 0.0))

    def test_move_neutralizes_after_normal_termination(self):
        runner = FakeRunner()
        MotionSkills(runner).move(0.2, 0, 0, 0.03)
        self.assertEqual(runner.commands[-1], (0.0, 0.0, 0.0))

    def test_turn_neutralizes_after_exception(self):
        runner = FakeRunner(fail=True)
        with self.assertRaises(RuntimeError):
            MotionSkills(runner).turn(45)
        self.assertEqual(runner.commands[-1], (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
