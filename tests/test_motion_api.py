"""Check timed velocity commands without real-time sleeps."""

import unittest
from unittest.mock import patch

import numpy as np

import motion_api


class TimedVelocityTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(motion_api.time, "monotonic", return_value=10.0)
        self.now = self.clock.start()
        motion_api.timed_vel_cmd(0, 0, 0, duration=0, new_command=True)

    def tearDown(self):
        motion_api.timed_vel_cmd(0, 0, 0, duration=0, new_command=True)
        self.clock.stop()

    def test_default_duration_and_polling_do_not_extend_timer(self):
        command = motion_api.timed_vel_cmd(1, -0.5, 0.2, new_command=True)
        self.assertEqual(command.dtype, np.float32)
        self.now.return_value = 10.9
        np.testing.assert_array_equal(motion_api.timed_vel_cmd(9, 9, 9), command)
        self.now.return_value = 11.0
        np.testing.assert_array_equal(motion_api.timed_vel_cmd(9, 9, 9), [0, 0, 0])

    def test_replacement_uses_new_velocity_and_duration(self):
        motion_api.timed_vel_cmd(1, 0, 0, new_command=True)
        self.now.return_value = 10.5
        motion_api.timed_vel_cmd(0, 1, -1, duration=2, new_command=True)
        self.now.return_value = 12.0
        np.testing.assert_array_equal(motion_api.timed_vel_cmd(0, 0, 0), [0, 1, -1])
        self.now.return_value = 12.5
        np.testing.assert_array_equal(motion_api.timed_vel_cmd(0, 0, 0), [0, 0, 0])

    def test_zero_duration_stops_immediately(self):
        motion_api.timed_vel_cmd(1, 0, 0, new_command=True)
        np.testing.assert_array_equal(
            motion_api.timed_vel_cmd(0, 0, 0, duration=0, new_command=True), [0, 0, 0]
        )

    def test_returned_array_cannot_modify_active_command(self):
        command = motion_api.timed_vel_cmd(1, 0, 0, new_command=True)
        command[:] = 9
        np.testing.assert_array_equal(motion_api.timed_vel_cmd(0, 0, 0), [1, 0, 0])

    def test_invalid_commands_do_not_replace_active_command(self):
        motion_api.timed_vel_cmd(1, 0, 0, new_command=True)
        for duration in (-1, np.nan, np.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                motion_api.timed_vel_cmd(0, 0, 0, duration, new_command=True)
        with self.assertRaises(ValueError):
            motion_api.timed_vel_cmd(np.nan, 0, 0, new_command=True)
        np.testing.assert_array_equal(motion_api.timed_vel_cmd(0, 0, 0), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
