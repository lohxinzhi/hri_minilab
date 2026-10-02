"""Validate dialogue-to-vision-to-motion object missions without cloud calls."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import dialogue_manager
import motion_api
import play
from dialogue_manager import DialogueManager, validate_actions


def goto(color="red", label="chair"):
    return {"action": "goto", "object_type": label, "object_color": color}


def detection(color="red", label="chair", bbox=None, confidence=0.8):
    return {
        "label": label,
        "color": color,
        "bbox": bbox or [40, 10, 60, 30],
        "confidence": confidence,
    }


class GotoSchemaTests(unittest.TestCase):
    def test_schema_and_validation_accept_only_canonical_coco_targets(self):
        variants = dialogue_manager.ACTION_SCHEMA["properties"]["actions"]["items"][
            "anyOf"
        ]
        schema = next(
            item
            for item in variants
            if item["properties"]["action"]["enum"] == ["goto"]
        )
        self.assertEqual(len(schema["properties"]["object_type"]["enum"]), 80)
        self.assertEqual(
            set(schema["required"]), {"action", "object_type", "object_color"}
        )
        self.assertFalse(schema["additionalProperties"])
        for label in dialogue_manager.COCO_CLASSES:
            self.assertEqual(
                validate_actions([goto("any", label)]), [goto("any", label)]
            )
        for action in (
            goto(label="door"),
            goto(label="bike"),
            goto(label="Toyota Supra"),
            goto(color="rainbow"),
            goto(color="grey"),
            goto(color=None),
            {"action": "goto", "object_type": "chair"},
            {**goto(), "bbox": [1, 2, 3, 4]},
        ):
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_actions([{"action": "stop"}, action])

    def test_mapped_goto_is_generated_summarized_queued_and_exported(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        refusal=None,
                        content=json.dumps({"actions": [goto("blue", "car")]}),
                    ),
                )
            ]
        )
        manager = DialogueManager(client=client)
        manager.submit_prompt("Approach the blue Toyota Supra")
        with redirect_stdout(StringIO()) as output:
            manager.start()
            try:
                plan = manager._plans.get(timeout=2)
            finally:
                manager.close()
                manager._thread.join(1)
        self.assertEqual(plan, [goto("blue", "car")])
        self.assertIn("[CMD] actions=", output.getvalue())
        self.assertEqual(manager.actions_snapshot(), plan)
        self.assertEqual(manager.export_snapshot()[-1]["action"], plan)
        self.assertEqual(
            manager.chat_snapshot()[-1]["content"], "Search for and approach blue car."
        )
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertIn("bike -> bicycle", prompt)
        self.assertIn("Reject unmappable objects", prompt)
        self.assertNotIn("those actions are not implemented", prompt)


class GotoExecutionTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(motion_api.time, "monotonic", return_value=10.0)
        self.now = clock.start()
        self.addCleanup(clock.stop)
        mission = patch.object(motion_api, "_object_mission", None)
        mission.start()
        self.addCleanup(mission.stop)
        self.sequence = play.RobotActionSequence()
        self.addCleanup(self.sequence.cancel)
        self.output = StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def update(self, detections=(), heading=0):
        return self.sequence.update(
            heading, dt=0.02, detections=detections, frame_size=(100, 100)
        )

    def test_live_class_and_color_matching_then_following_action(self):
        self.sequence.replace(
            [goto(), {"action": "turn", "angle": 90}], history_start=4
        )
        self.assertTrue(self.sequence.needs_vision)
        command = self.update([detection("blue"), detection(label="car")])
        self.assertGreater(command[2], 0)
        self.assertEqual(self.sequence.active_index, 4)
        self.assertIn("[SEARCH]", self.output.getvalue())
        np.testing.assert_array_equal(self.update([detection()]), [0.5, 0, 0])
        self.assertIn("[APPROACH]", self.output.getvalue())
        self.assertGreater(self.update([detection(bbox=[40, 0, 60, 70])])[2], 0)
        self.assertEqual(self.sequence.active, "turn")
        self.assertEqual(self.sequence.active_index, 5)
        self.assertFalse(self.sequence.needs_vision)
        self.update(heading=88)
        self.assertEqual(self.output.getvalue().count("[DONE]"), 1)
        self.assertEqual(self.output.getvalue().count("[EXEC] action=goto"), 1)

    def test_any_color_and_highest_confidence_matching_box(self):
        self.sequence.replace([goto("any")])
        with patch.object(
            play, "goto_object", wraps=motion_api.goto_object
        ) as controller:
            self.update([detection("blue", confidence=0.95), detection(confidence=0.7)])
        self.assertEqual(controller.call_args.args, ("chair", "any", [40, 10, 60, 30]))
        self.assertTrue(controller.call_args.kwargs["new_command"])
        self.update([])
        self.assertEqual(self.sequence.active, "goto")
        self.assertIn("[SEARCH]", self.output.getvalue())

    def test_failed_mission_aborts_remaining_actions_without_done(self):
        self.sequence.replace([goto(), {"action": "turn", "angle": 90}])
        self.update([])
        self.now.return_value = 70
        np.testing.assert_array_equal(self.update([]), [0, 0, 0])
        self.assertFalse(self.sequence.pending)
        self.assertIsNone(self.sequence.active)
        self.assertIsNone(self.sequence.active_index)
        self.assertNotIn("[DONE]", self.output.getvalue())
        self.assertNotIn("[EXEC] action=turn", self.output.getvalue())

    def test_cancel_and_replacement_stop_missions_and_restart_same_target(self):
        self.sequence.replace([goto()])
        self.update([detection()])
        self.sequence.cancel()
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [0, 0, 0])
        self.sequence.replace([goto()])
        np.testing.assert_array_equal(self.update([detection()]), [0.5, 0, 0])
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.sequence.replace([{"action": "stop"}])
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.assertEqual(self.output.getvalue().count("reason=cancelled"), 2)


class GotoLoopTests(unittest.TestCase):
    def test_submitted_prompt_runs_goto_with_headless_vision(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        refusal=None,
                        content=json.dumps({"actions": [goto()]}),
                    ),
                )
            ]
        )
        manager = DialogueManager(client=client)
        manager.submit_prompt("Go to the red chair")
        with (
            patch.object(play, "DialogueManager", return_value=manager),
            patch.object(play, "VisionModule") as constructor,
            patch.object(
                play, "goto_object", wraps=motion_api.goto_object
            ) as controller,
            patch.object(play, "plot_heading"),
            patch.object(motion_api, "_object_mission", None),
            patch.object(
                play.sys,
                "argv",
                [
                    "play.py",
                    "--headless",
                    "--no-policy",
                    "--map",
                    "coco_scene",
                    "--duration",
                    "0.2",
                ],
            ),
            redirect_stdout(StringIO()) as output,
        ):
            vision = constructor.return_value
            vision.detections = [detection(bbox=[40, 0, 60, 70])]
            vision.frame_size = (100, 100)
            play.main()
            vision.render_fpv.assert_called()
            controller.assert_called()
            self.assertEqual(
                controller.call_args.args, ("chair", "red", [40, 0, 60, 70])
            )
            self.assertIn("[MISSION] status=SUCCESS", output.getvalue())
            self.assertIn("[DONE]", output.getvalue())
            self.assertEqual(
                [message["status"] for message in manager.chat_snapshot()],
                ["completed", "completed"],
            )
        manager._thread.join(1)


if __name__ == "__main__":
    unittest.main()
