"""Validate sequential motion execution without sleeping or calling the API."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

import motion_api
import play


def movement(vx=1, duration=1):
    return {
        "action": "move",
        "velocity": {"vx": vx, "vy": 0, "wz": 0},
        "duration": duration,
    }


class ActionSequenceTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(motion_api.time, "monotonic", return_value=10.0)
        self.now = clock.start()
        self.addCleanup(clock.stop)
        self.sequence = play.RobotActionSequence()
        self.addCleanup(self.sequence.cancel)
        self.output = StringIO()
        stdout = redirect_stdout(self.output)
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def test_move_then_relative_turn_then_move(self):
        self.sequence.replace(
            [movement(duration=2), {"action": "turn", "angle": 90}, movement(vx=-0.5)]
        )
        np.testing.assert_array_equal(self.sequence.update(30, dt=0.02), [1, 0, 0])
        self.now.return_value = 11.9
        np.testing.assert_array_equal(self.sequence.update(35, dt=0.02), [1, 0, 0])
        self.now.return_value = 12
        self.assertGreater(self.sequence.update(35, dt=0.02)[2], 0)
        self.assertEqual(play.get_turn_target_heading(), 125)
        np.testing.assert_array_equal(
            self.sequence.update(122.5, dt=0.02), [-0.5, 0, 0]
        )
        self.assertIsNone(play.get_turn_target_heading())
        self.now.return_value = 13
        np.testing.assert_array_equal(self.sequence.update(125, dt=0.02), [0, 0, 0])
        self.assertIsNone(self.sequence.active)

    def test_active_history_index_advances_and_clears(self):
        self.sequence.replace(
            [
                {"action": "chat", "reply": "Go"},
                movement(),
                {"action": "turn", "angle": 90},
            ],
            history_start=4,
        )
        self.sequence.update(0, dt=0.02)
        self.assertEqual(self.sequence.active_index, 5)
        self.now.return_value = 11
        self.sequence.update(0, dt=0.02)
        self.assertEqual(self.sequence.active_index, 6)
        self.sequence.update(88, dt=0.02)
        self.assertIsNone(self.sequence.active_index)
        self.sequence.replace([movement()], history_start=7)
        self.sequence.update(88)
        self.assertEqual(self.sequence.active_index, 7)
        self.sequence.cancel()
        self.assertIsNone(self.sequence.active_index)

    def test_zero_velocity_move_waits_for_duration(self):
        self.sequence.replace([movement(vx=0, duration=2), movement()])
        np.testing.assert_array_equal(self.sequence.update(0), [0, 0, 0])
        self.now.return_value = 11
        np.testing.assert_array_equal(self.sequence.update(0), [0, 0, 0])
        self.now.return_value = 12
        np.testing.assert_array_equal(self.sequence.update(0), [1, 0, 0])

    def test_stop_discards_remainder_and_chat_does_not_move(self):
        self.sequence.replace(
            [{"action": "chat", "reply": "Hello"}, {"action": "stop"}, movement()]
        )
        np.testing.assert_array_equal(self.sequence.update(0), [0, 0, 0])
        self.assertIn("Hello", self.output.getvalue())
        self.assertFalse(self.sequence.pending)

    def test_new_plan_overrides_turn_and_invalid_plan_preserves_move(self):
        self.sequence.replace([{"action": "turn", "angle": 90}])
        self.assertGreater(self.sequence.update(0)[2], 0)
        self.sequence.replace([movement(vx=-1, duration=2)])
        np.testing.assert_array_equal(self.sequence.update(0), [-1, 0, 0])
        with self.assertRaises(ValueError):
            self.sequence.replace([movement(), {"action": "approach"}])
        np.testing.assert_array_equal(self.sequence.update(0), [-1, 0, 0])
        self.assertIsNone(play.get_turn_target_heading())

    def test_large_turn_and_wrap_complete_without_starting_next_action_early(self):
        self.sequence.replace([{"action": "turn", "angle": 270}, movement()])
        self.assertGreater(self.sequence.update(179)[2], 0)
        self.assertGreater(self.sequence.update(-91)[2], 0)
        self.assertGreater(self.sequence.update(-1)[2], 0)
        np.testing.assert_array_equal(self.sequence.update(87), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
