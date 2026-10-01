"""Check timed velocity commands without real-time sleeps."""

import unittest
from unittest.mock import patch

import numpy as np

import motion_api


class TimedVelocityTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(motion_api.time, "monotonic", return_value=10.0)
        self.now = self.clock.start()
        motion_api.move(0, 0, 0, duration=0, new_command=True)

    def tearDown(self):
        motion_api.move(0, 0, 0, duration=0, new_command=True)
        self.clock.stop()

    def test_default_duration_and_polling_do_not_extend_timer(self):
        command = motion_api.move(1, -0.5, 0.2, new_command=True)
        self.assertEqual(command.dtype, np.float32)
        self.now.return_value = 10.9
        np.testing.assert_array_equal(motion_api.move(9, 9, 9), command)
        self.now.return_value = 11.0
        np.testing.assert_array_equal(motion_api.move(9, 9, 9), [0, 0, 0])

    def test_replacement_uses_new_velocity_and_duration(self):
        motion_api.move(1, 0, 0, new_command=True)
        self.now.return_value = 10.5
        motion_api.move(0, 1, -1, duration=2, new_command=True)
        self.now.return_value = 12.0
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [0, 1, -1])
        self.now.return_value = 12.5
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [0, 0, 0])

    def test_zero_duration_stops_immediately(self):
        motion_api.move(1, 0, 0, new_command=True)
        np.testing.assert_array_equal(
            motion_api.move(0, 0, 0, duration=0, new_command=True), [0, 0, 0]
        )

    def test_returned_array_cannot_modify_active_command(self):
        command = motion_api.move(1, 0, 0, new_command=True)
        command[:] = 9
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [1, 0, 0])

    def test_invalid_commands_do_not_replace_active_command(self):
        motion_api.move(1, 0, 0, new_command=True)
        for duration in (-1, np.nan, np.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                motion_api.move(0, 0, 0, duration, new_command=True)
        with self.assertRaises(ValueError):
            motion_api.move(np.nan, 0, 0, new_command=True)
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [1, 0, 0])


class YawCommandTests(unittest.TestCase):
    def setUp(self):
        motion_api.move(0, 0, 0, duration=0, new_command=True)

    def test_turn_direction_and_speed_limit(self):
        command = motion_api.turn(90, 0)
        np.testing.assert_array_equal(command, [0, 0, 1])
        self.assertEqual(command.dtype, np.float32)
        self.assertLess(motion_api.turn(-90, 90)[2], 0)

    def test_turn_log_reports_fixed_target_and_continuous_heading(self):
        with (
            patch.object(motion_api.time, "monotonic", return_value=10.0) as clock,
            patch("builtins.print") as log,
        ):
            motion_api.turn(10, 179, new_command=True)
            log.assert_called_with(
                "[TURN] target = 189.00 current = 179.00 error = 10.00"
            )
            clock.return_value = 10.05
            motion_api.turn(10, -179)
            self.assertEqual(log.call_count, 1)
            clock.return_value = 10.2
            motion_api.turn(10, -178)
            log.assert_called_with(
                "[TURN] target = 189.00 current = 182.00 error = 7.00"
            )

    def test_relative_turn_crosses_heading_boundary(self):
        motion_api.turn(10, 179, new_command=True)
        self.assertGreater(motion_api.turn(10, -179)[2], 0)
        motion_api.turn(-10, -179, new_command=True)
        self.assertLess(motion_api.turn(-10, 179)[2], 0)

    def test_relative_target_is_captured_once_and_can_be_repeated(self):
        self.assertGreater(motion_api.turn(10, 60, new_command=True)[2], 0)
        np.testing.assert_array_equal(motion_api.turn(10, 70), [0, 0, 0])
        self.assertGreater(motion_api.turn(10, 70, new_command=True)[2], 0)
        np.testing.assert_array_equal(motion_api.turn(10, 80), [0, 0, 0])

    def test_large_turn_keeps_requested_direction(self):
        self.assertGreater(motion_api.turn(270, 60, new_command=True)[2], 0)
        motion_api.turn(270, 160)
        motion_api.turn(270, -100)
        np.testing.assert_array_equal(motion_api.turn(270, -30), [0, 0, 0])

    def test_turn_cancels_previous_move(self):
        motion_api.move(1, 0, 0, new_command=True)
        motion_api.turn(10, 60, new_command=True)
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [0, 0, 0])

    def test_damping_and_heading_tolerance(self):
        undamped = motion_api.turn(20, 0)[2]
        damped = motion_api.turn(20, 0, yaw_rate_deg_s=30)[2]
        self.assertLess(damped, undamped)
        np.testing.assert_array_equal(motion_api.turn(20, 19.5), [0, 0, 0])

    def test_closed_loop_converges_for_ideal_yaw_dynamics(self):
        yaw, rate = 0.0, 0.0
        for _ in range(500):
            command = motion_api.turn(90, yaw, yaw_rate_deg_s=rate, dt=0.02)
            rate = np.rad2deg(command[2])
            yaw += rate * 0.02
        self.assertLessEqual(abs(90 - yaw), 1.0)

    def test_integral_corrects_a_velocity_deadband(self):
        yaw, rate = 0.0, 0.0
        for _ in range(1500):
            command = motion_api.turn(60, yaw, yaw_rate_deg_s=rate, dt=0.02)
            rate = np.rad2deg(command[2]) if abs(command[2]) > 0.25 else 0.0
            yaw += rate * 0.02
        self.assertLessEqual(abs(60 - yaw), 1.0)

    def test_invalid_inputs(self):
        for kwargs in (
            {"kp": -1},
            {"ki": -1},
            {"kd": -1},
            {"max_wz": 0},
            {"tolerance_deg": -1},
            {"yaw_rate_deg_s": np.nan},
            {"dt": 0},
            {"dt": np.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                motion_api.turn(0, 0, **kwargs)
        with self.assertRaises(ValueError):
            motion_api.turn(np.inf, 0)


if __name__ == "__main__":
    unittest.main()
