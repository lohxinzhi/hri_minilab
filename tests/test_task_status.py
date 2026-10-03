"""Link task execution outcomes to the originating chat prompt and reply."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

import motion_api
import play
from dialogue_manager import DialogueManager


class TaskStatusTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(motion_api.time, "monotonic", return_value=10.0)
        self.now = clock.start()
        self.addCleanup(clock.stop)
        mission = patch.object(motion_api, "_object_mission", None)
        mission.start()
        self.addCleanup(mission.stop)
        output = redirect_stdout(StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.manager = DialogueManager(client=Mock())
        self.sequence = play.RobotActionSequence(
            on_status=self.manager.update_task_status
        )
        self.addCleanup(self.sequence.cancel)

    def plan(self, actions):
        request_id = self.manager.submit_prompt("request")
        self.manager._reply(request_id, "plan", "planned", actions=actions)
        offset = self.manager.action_history_start(actions)
        self.assertEqual(self.manager.action_request_id(actions), request_id)
        self.assertIsNone(self.manager.action_request_id(actions))
        self.sequence.replace(actions, offset, request_id=request_id)
        return request_id

    def statuses(self, request_id):
        return [
            item["status"]
            for item in self.manager.chat_snapshot()
            if item["id"] == request_id and item.get("kind") != "task_result"
        ]

    def test_complete_only_after_all_moves_and_turns_finish(self):
        request = self.plan(
            [
                {
                    "action": "move",
                    "velocity": {"vx": 0.5, "vy": 0, "wz": 0},
                    "duration": 2,
                },
                {"action": "turn", "angle": 90},
            ]
        )
        self.assertEqual(self.statuses(request), ["planned", "planned"])
        self.sequence.update(0, dt=0.02)
        self.assertEqual(self.statuses(request), ["executing", "executing"])
        self.now.return_value = 12
        self.sequence.update(0, dt=0.02)
        self.assertEqual(self.statuses(request), ["executing", "executing"])
        self.sequence.update(88, dt=0.02)
        self.assertEqual(self.statuses(request), ["completed", "completed"])
        self.sequence.update(88, dt=0.02)
        self.sequence.cancel()
        self.assertEqual(self.statuses(request), ["completed", "completed"])

    def test_goto_failure_updates_chat_and_discards_following_actions(self):
        request = self.plan(
            [
                {"action": "goto", "object_type": "chair", "object_color": "red"},
                {"action": "turn", "angle": 90},
            ]
        )
        self.sequence.update(0)
        self.now.return_value = 70
        self.sequence.update(0)
        self.assertEqual(self.statuses(request), ["failed", "failed"])
        self.assertFalse(self.sequence.pending)
        self.sequence.cancel()
        self.assertEqual(self.statuses(request), ["failed", "failed"])

    def test_replacement_updates_only_the_interrupted_prompt(self):
        old = self.plan([{"action": "turn", "angle": 90}])
        self.sequence.update(0)
        new = self.plan([{"action": "stop"}])
        queued = self.manager.submit_prompt("still generating")
        self.assertEqual(self.statuses(old), ["cancelled", "cancelled"])
        self.assertEqual(self.statuses(new), ["planned", "planned"])
        self.sequence.update(0)
        self.assertEqual(self.statuses(new), ["completed", "completed"])
        self.assertEqual(self.statuses(queued), ["queued"])
        self.assertEqual(self.statuses(old), ["cancelled", "cancelled"])

    def test_cancel_before_execution_and_chat_only_completion(self):
        cancelled = self.plan([{"action": "turn", "angle": 90}])
        self.sequence.cancel()
        self.assertEqual(self.statuses(cancelled), ["cancelled", "cancelled"])
        replied = self.plan([{"action": "chat", "reply": "Hello"}])
        self.sequence.update(0)
        self.assertEqual(self.statuses(replied), ["completed", "completed"])

    def test_status_updates_preserve_chat_content_and_export_plan(self):
        request = self.plan([{"action": "stop"}])
        before = self.manager.export_snapshot()
        self.sequence.update(0)
        after = self.manager.export_snapshot()
        self.assertEqual(before, after[:2])
        self.assertEqual(after[-1]["text"], "Successfully stopped.")
        self.assertEqual(after[-1]["action"], [])
        self.assertIsNone(after[-1]["estimated_api_cost_usd"])
        self.assertEqual(self.statuses(request), ["completed", "completed"])
        self.manager.update_task_status(request, "executing")
        self.assertEqual(self.statuses(request), ["completed", "completed"])
        with self.assertRaises(ValueError):
            self.manager.update_task_status(request, "unknown")

    def test_chat_reply_preserves_move_deadline_pending_actions_and_status(self):
        task = self.plan(
            [
                {
                    "action": "move",
                    "velocity": {"vx": 0.5, "vy": 0, "wz": 0},
                    "duration": 2,
                },
                {"action": "turn", "angle": 90},
            ]
        )
        self.sequence.update(0)
        index = self.sequence.active_index
        self.now.return_value = 11
        reply = self.plan([{"action": "chat", "reply": "Still walking."}])
        self.assertEqual(self.statuses(task), ["executing", "executing"])
        self.assertEqual(self.statuses(reply), ["completed", "completed"])
        self.assertEqual(self.sequence.active, "move")
        self.assertEqual(self.sequence.active_index, index)
        self.assertEqual(self.sequence.request_id, task)
        np.testing.assert_array_equal(self.sequence.update(0), [0.5, 0, 0])
        self.now.return_value = 12
        self.assertGreater(self.sequence.update(0)[2], 0)
        self.sequence.update(88)
        self.assertEqual(self.statuses(task), ["completed", "completed"])

    def test_chat_reply_preserves_turn_target(self):
        task = self.plan([{"action": "turn", "angle": 90}])
        self.sequence.update(30)
        reply = self.plan([{"action": "chat", "reply": "Turning left."}])
        self.assertEqual(motion_api.get_turn_target_heading(), 120)
        self.assertEqual(self.sequence.active, "turn")
        self.assertGreater(self.sequence.update(60)[2], 0)
        self.sequence.update(118)
        self.assertEqual(self.statuses(task), ["completed", "completed"])
        self.assertEqual(self.statuses(reply), ["completed", "completed"])

    def test_chat_reply_preserves_goto_mission_and_timeout(self):
        task = self.plan(
            [{"action": "goto", "object_type": "chair", "object_color": "red"}]
        )
        self.sequence.update(0)
        self.now.return_value = 50
        reply = self.plan([{"action": "chat", "reply": "Searching for the chair."}])
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.assertEqual(self.sequence.active, "goto")
        self.assertEqual(self.statuses(task), ["executing", "executing"])
        self.now.return_value = 70
        self.sequence.update(0)
        self.assertEqual(self.statuses(task), ["failed", "failed"])
        self.assertEqual(self.statuses(reply), ["completed", "completed"])

    def results(self, request_id):
        return [
            item["content"]
            for item in self.manager.chat_snapshot()
            if item["id"] == request_id and item.get("kind") == "task_result"
        ]

    def test_right_turn_completion_adds_one_natural_language_reply(self):
        task = self.plan([{"action": "turn", "angle": -84}])
        self.sequence.update(0)
        self.assertEqual(self.results(task), [])
        self.sequence.update(-83)
        self.assertEqual(
            self.results(task), ["Successfully turned 84 degrees to the right."]
        )
        self.sequence.update(-84)
        self.sequence.cancel()
        self.assertEqual(len(self.results(task)), 1)

    def test_failed_search_reports_target_and_reason(self):
        task = self.plan(
            [{"action": "goto", "object_type": "car", "object_color": "red"}]
        )
        for heading in (0, 90, 180, -90, 0):
            self.sequence.update(heading)
        self.assertEqual(
            self.results(task), ["Failed to find the red car after a full search."]
        )

    def test_timeout_reply_includes_configured_duration(self):
        with patch.object(motion_api, "GOTO_TIMEOUT_SECONDS", 5):
            task = self.plan(
                [{"action": "goto", "object_type": "car", "object_color": "red"}]
            )
            self.sequence.update(0)
            self.now.return_value = 15
            self.sequence.update(0)
        self.assertEqual(
            self.results(task),
            ["Failed to find or approach the red car: timed out after 5 seconds."],
        )

    def test_missing_map_position_is_explained(self):
        task = self.plan(
            [{"action": "goto", "object_type": "person", "object_color": "any"}]
        )
        self.sequence.update(0)
        self.assertEqual(
            self.results(task),
            ["Failed to approach the person: its position is unavailable in this map."],
        )

    def test_multi_action_success_waits_for_full_plan_and_skips_chat(self):
        task = self.plan(
            [
                {"action": "chat", "reply": "Turning now."},
                {"action": "turn", "angle": 84},
                {"action": "stop"},
            ]
        )
        self.sequence.update(0)
        self.assertEqual(self.results(task), [])
        self.sequence.update(83)
        self.assertEqual(
            self.results(task),
            ["Successfully turned 84 degrees to the left.\nSuccessfully stopped."],
        )

    def test_result_export_does_not_duplicate_api_cost_or_actions(self):
        actions = [{"action": "stop"}]
        request = self.manager.submit_prompt("stop")
        self.manager._reply(
            request, "Stop.", "planned", actions=actions, estimated_cost=0.123
        )
        self.sequence.replace(actions, request_id=request)
        self.sequence.update(0)
        rows = self.manager.export_snapshot()
        self.assertEqual(
            [row["estimated_api_cost_usd"] for row in rows], [None, 0.123, None]
        )
        self.assertEqual([row["action"] for row in rows], [None, actions, []])
        self.assertEqual(self.manager.actions_snapshot(), actions)


if __name__ == "__main__":
    unittest.main()
