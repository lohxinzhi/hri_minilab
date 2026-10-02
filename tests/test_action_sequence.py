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

    def lifecycle_logs(self):
        return [
            line
            for line in self.output.getvalue().splitlines()
            if line.startswith(("[EXEC]", "[DONE]"))
        ]

    def test_execution_logs_once_per_action_and_done_after_final_motion(self):
        self.sequence.update(0)
        self.assertEqual(self.lifecycle_logs(), [])
        self.sequence.replace(
            [
                {"action": "chat", "reply": "Go\nnow"},
                movement(duration=2),
                {"action": "turn", "angle": 90},
            ]
        )
        self.sequence.update(0, dt=0.02)
        self.sequence.update(0, dt=0.02)
        self.assertEqual(len(self.lifecycle_logs()), 2)
        self.assertNotIn("[DONE]", self.lifecycle_logs())
        self.now.return_value = 12
        self.sequence.update(0, dt=0.02)
        self.sequence.update(30, dt=0.02)
        self.assertNotIn("[DONE]", self.lifecycle_logs())
        self.sequence.update(88, dt=0.02)
        self.sequence.update(90, dt=0.02)
        self.assertEqual(
            self.lifecycle_logs(),
            [
                '[EXEC] action=chat reply="Go\\nnow"',
                '[EXEC] action=move velocity={"vx": 1, "vy": 0, "wz": 0} duration=2',
                "[EXEC] action=turn angle=90",
                "[DONE]",
            ],
        )

    def test_cancel_and_replacement_do_not_report_completed_plan(self):
        self.sequence.replace([movement(duration=2)])
        self.sequence.update(0)
        self.sequence.cancel()
        self.sequence.update(0)
        self.assertNotIn("[DONE]", self.lifecycle_logs())
        self.sequence.replace([{"action": "turn", "angle": 90}])
        self.sequence.update(0)
        self.sequence.replace([{"action": "stop"}])
        self.sequence.update(0)
        self.sequence.update(0)
        self.assertEqual(self.lifecycle_logs().count("[DONE]"), 1)
        self.assertEqual(self.lifecycle_logs()[-2:], ["[EXEC] action=stop", "[DONE]"])

    def test_immediate_plan_completes_once_and_stop_remainder_is_not_completed(self):
        self.sequence.replace([movement(duration=0), {"action": "turn", "angle": 0}])
        self.sequence.update(0)
        self.sequence.update(0)
        self.assertEqual(self.lifecycle_logs().count("[DONE]"), 1)
        self.output.truncate(0)
        self.output.seek(0)
        self.sequence.replace([{"action": "stop"}, movement()])
        self.sequence.update(0)
        self.sequence.update(0)
        self.assertEqual(self.lifecycle_logs(), ["[EXEC] action=stop"])

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
