"""Exercise visual planning, camera input, history, and independent execution."""

import base64
import json
import threading
import unittest
from concurrent.futures import Future
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import dialogue_manager
import motion_api
import play
import vision_module


def completion(content="A red car is visible.", finish_reason="stop", refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
    )


def describe(mode="describe", **kwargs):
    return {"action": "describe", "mode": mode, **kwargs}


class VisualPlanningTests(unittest.TestCase):
    def test_schema_and_validation_distinguish_description_from_question(self):
        variants = dialogue_manager.ACTION_SCHEMA["properties"]["actions"]["items"][
            "anyOf"
        ]
        visual_schemas = {
            variant["properties"]["mode"]["enum"][0]: variant
            for variant in variants
            if variant["properties"]["action"]["enum"] == ["describe"]
        }
        self.assertEqual(set(visual_schemas), {"describe", "vqa"})
        self.assertEqual(
            set(visual_schemas["describe"]["required"]), {"action", "mode"}
        )
        self.assertEqual(
            set(visual_schemas["vqa"]["required"]), {"action", "mode", "question"}
        )
        self.assertFalse(visual_schemas["vqa"]["additionalProperties"])
        for action in [describe(), describe("vqa", question="What colour is the car?")]:
            self.assertEqual(dialogue_manager.validate_actions([action]), [action])
        for action in [
            describe("invalid"),
            describe("vqa"),
            describe("vqa", question=" "),
            describe("vqa", question=1),
            describe(question="extra"),
            describe(mode=[]),
        ]:
            with (
                self.subTest(action=action),
                self.assertRaises((ValueError, TypeError)),
            ):
                dialogue_manager.validate_actions([action])

    def test_visual_answer_is_in_chat_export_and_next_planner_request(self):
        client = Mock()
        action = describe("vqa", question="What colour is the car?")
        client.chat.completions.create.return_value = completion(
            json.dumps({"actions": [action]})
        )
        manager = dialogue_manager.DialogueManager(client=client)
        request_id = manager.submit_prompt("What colour is the car?")
        actions = manager.user_cmd("What colour is the car?")
        manager._reply(
            request_id,
            manager._summarize(actions),
            "planned",
            actions=actions,
            estimated_cost=0.01,
        )
        manager.update_task_status(request_id, "executing")
        manager.add_visual_response(request_id, "The car is red.", estimated_cost=0.02)
        manager.update_task_status(request_id, "completed")
        self.assertEqual(manager.chat_snapshot()[-1]["content"], "The car is red.")
        self.assertEqual(manager.chat_snapshot()[-1]["status"], "completed")
        exported = manager.export_snapshot()[-1]
        self.assertEqual(exported["action"], [])
        self.assertEqual(exported["estimated_api_cost_usd"], 0.02)
        manager.user_cmd("Tell me more about it")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn({"role": "assistant", "content": "The car is red."}, messages)
        self.assertEqual(manager._summarize([describe()]), "")

    def test_visual_plans_show_only_vlm_answer_without_empty_chat_bubble(self):
        for action in [describe(), describe("vqa", question="What is ahead?")]:
            with self.subTest(action=action):
                manager = dialogue_manager.DialogueManager(client=Mock())
                request_id = manager.submit_prompt("Look ahead")
                manager._reply(
                    request_id,
                    manager._summarize([action]),
                    "planned",
                    actions=[action],
                )
                self.assertEqual(len(manager.chat_snapshot()), 1)
                self.assertEqual(manager.chat_snapshot()[0]["status"], "planned")
                self.assertEqual(manager.actions_snapshot(), [action])
                manager.update_task_status(request_id, "executing")
                manager.add_visual_response(request_id, "A red car is visible.")
                manager.update_task_status(request_id, "completed")
                replies = [
                    message
                    for message in manager.chat_snapshot()
                    if message["role"] == "assistant"
                ]
                self.assertEqual(
                    [message["content"] for message in replies],
                    ["A red car is visible."],
                )
                self.assertEqual(replies[0]["status"], "completed")
        self.assertEqual(
            dialogue_manager.DialogueManager._summarize(
                [
                    describe(),
                    {"action": "chat", "reply": "Hello"},
                    describe("vqa", question="What is ahead?"),
                ]
            ),
            "Hello",
        )

    def test_saving_visual_history_does_not_wait_for_planner_network_call(self):
        entered, release = threading.Event(), threading.Event()
        client = Mock()

        def blocked(**kwargs):
            entered.set()
            release.wait(2)
            return completion(json.dumps({"actions": [describe()]}))

        client.chat.completions.create.side_effect = blocked
        manager = dialogue_manager.DialogueManager(client=client)
        worker = threading.Thread(target=manager.user_cmd, args=("Describe",))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            manager.add_visual_response(1, "A blue chair is visible.")
            self.assertIn(
                {"role": "assistant", "content": "A blue chair is visible."},
                manager.history,
            )
        finally:
            release.set()
            worker.join(2)


class VisualModelTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.chat.completions.create.return_value = completion()
        self.vision = vision_module.VisionModule(vlm_client=self.client)
        self.addCleanup(self.vision.close)
        self.frame = np.zeros((12, 16, 3), dtype=np.uint8)
        self.frame[:] = (20, 40, 200)

    def image_sent(self):
        kwargs = self.client.chat.completions.create.call_args.kwargs
        parts = kwargs["messages"][1]["content"]
        payload = parts[1]["image_url"]["url"].split(",", 1)[1]
        pixels = cv2.imdecode(
            np.frombuffer(base64.b64decode(payload), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        return kwargs, parts[0]["text"], pixels

    def test_general_description_uses_prompt_and_correct_colour_image(self):
        answer, cost = self.vision.describe(self.frame)
        self.assertEqual(answer, "A red car is visible.")
        self.assertIsNotNone(cost)
        kwargs, text, pixels = self.image_sent()
        self.assertEqual(kwargs["model"], "gpt-6-luna")
        self.assertEqual(
            kwargs["messages"][0]["content"], vision_module.SCENE_DESCRIPTION_PROMPT
        )
        self.assertNotIn("Question:", text)
        np.testing.assert_array_equal(pixels, self.frame)

    def test_configured_vlm_model_is_used_for_description_and_vqa(self):
        vision = vision_module.VisionModule(vlm_model="gpt-4o", vlm_client=self.client)
        self.addCleanup(vision.close)
        for mode, question in (("describe", None), ("vqa", "What colour is the car?")):
            with self.subTest(mode=mode):
                vision.describe(self.frame, mode, question)
                self.assertEqual(
                    self.client.chat.completions.create.call_args.kwargs["model"],
                    "gpt-4o",
                )

    def test_vqa_uses_separate_prompt_and_quoted_question(self):
        question = 'What colour is the "car"?'
        self.vision.describe(self.frame, "vqa", question)
        kwargs, text, _ = self.image_sent()
        self.assertEqual(
            kwargs["messages"][0]["content"], vision_module.VISUAL_VQA_PROMPT
        )
        self.assertIn(json.dumps(question), text)
        self.assertEqual(len(kwargs["messages"]), 2)

    def test_incomplete_refused_empty_answers_and_invalid_inputs_fail(self):
        for response in [
            completion(""),
            completion(finish_reason="length"),
            completion(refusal="No"),
        ]:
            self.client.chat.completions.create.return_value = response
            with self.assertRaises(ValueError):
                self.vision.describe(self.frame)
        for args in [(self.frame, "vqa"), (self.frame, "unknown"), (np.zeros((1, 1)),)]:
            with self.assertRaises(ValueError):
                self.vision.describe(*args)

    def test_worker_copies_frame_and_keeps_cloud_request_off_caller_thread(self):
        entered, release = threading.Event(), threading.Event()
        caller_thread = threading.get_ident()
        thread_ids = []

        def blocked(**kwargs):
            thread_ids.append(threading.get_ident())
            entered.set()
            release.wait(2)
            return completion()

        self.client.chat.completions.create.side_effect = blocked
        callback = Mock()
        expected = self.frame.copy()
        future = self.vision.submit_visual(self.frame, describe(), callback)
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(future.done())
            self.frame[:] = 0
        finally:
            release.set()
        self.assertEqual(future.result(timeout=2), "A red car is visible.")
        self.assertNotEqual(thread_ids[0], caller_thread)
        callback.assert_called_once()
        np.testing.assert_array_equal(self.image_sent()[2], expected)

    def test_failed_request_does_not_kill_worker(self):
        self.client.chat.completions.create.side_effect = [
            ValueError("bad image"),
            completion(),
        ]
        first = self.vision.submit_visual(self.frame, describe(), Mock())
        with self.assertRaisesRegex(ValueError, "bad image"):
            first.result(timeout=2)
        second = self.vision.submit_visual(self.frame, describe(), Mock())
        self.assertEqual(second.result(timeout=2), "A red car is visible.")


class VisualExecutionTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(motion_api.time, "monotonic", return_value=10)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.status = Mock()
        self.future = Future()
        self.start = Mock(return_value=self.future)
        self.sequence = play.RobotActionSequence(
            on_status=self.status, on_describe=self.start
        )
        self.addCleanup(self.sequence.cancel)
        self.output = StringIO()
        stdout = redirect_stdout(self.output)
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def test_description_leaves_motion_and_pending_turn_running(self):
        motion = {
            "action": "move",
            "velocity": {"vx": 1, "vy": 0, "wz": 0},
            "duration": 10,
        }
        self.sequence.replace([motion, {"action": "turn", "angle": 84}], request_id=1)
        self.sequence.update(0)
        self.status.reset_mock()
        self.sequence.replace([describe()], request_id=2)
        self.assertEqual(self.sequence.active, "move")
        self.assertEqual(self.sequence.request_id, 1)
        self.assertEqual(len(self.sequence.pending), 1)
        np.testing.assert_array_equal(self.sequence.update(0), [1, 0, 0])
        self.assertNotIn("[DONE]", self.output.getvalue())
        self.future.set_result("A car.")
        np.testing.assert_array_equal(self.sequence.update(0), [1, 0, 0])
        self.status.assert_any_call(2, "completed", reason=None)
        self.assertEqual(self.sequence.active, "move")
        self.assertEqual(self.output.getvalue().count("[DONE]"), 1)

    def test_visual_failure_reports_reason_without_cancelling_turn(self):
        self.sequence.replace([{"action": "turn", "angle": 90}], request_id=1)
        self.sequence.update(0)
        self.sequence.replace(
            [describe("vqa", question="What is ahead?")], request_id=2
        )
        self.future.set_exception(ValueError("VLM timeout"))
        self.sequence.update(0)
        self.status.assert_any_call(2, "failed", reason="VLM timeout")
        self.assertEqual(self.sequence.active, "turn")
        self.assertEqual(self.sequence.request_id, 1)

    def test_visual_question_leaves_goto_running(self):
        target = {"action": "goto", "object_type": "car", "object_color": "blue"}
        with patch.object(
            self.sequence, "_update_goto", return_value=np.array([0.3, 0, 0.1])
        ):
            self.sequence.replace([target], request_id=1)
            self.sequence.update(0)
            self.sequence.replace(
                [describe("vqa", question="What is ahead?")], request_id=2
            )
            self.future.set_result("A blue car.")
            np.testing.assert_array_equal(self.sequence.update(0), [0.3, 0, 0.1])
        self.assertEqual(self.sequence.active, "goto")
        self.assertEqual(self.sequence.goto_target, target)
        self.assertEqual(self.sequence.request_id, 1)
        self.assertNotIn(unittest.mock.call(1, "cancelled"), self.status.call_args_list)

    def test_worker_reply_reaches_chat_and_history_with_completed_status(self):
        planner, vlm = Mock(), Mock()
        action = describe("vqa", question="What colour is the car?")
        planner.chat.completions.create.return_value = completion(
            json.dumps({"actions": [action]})
        )
        vlm.chat.completions.create.return_value = completion("The car is red.")
        manager = dialogue_manager.DialogueManager(client=planner)
        request_id = manager.submit_prompt("What colour is the car?")
        actions = manager.user_cmd("What colour is the car?")
        manager._reply(
            request_id, manager._summarize(actions), "planned", actions=actions
        )
        vision = vision_module.VisionModule(vlm_client=vlm)
        self.addCleanup(vision.close)
        futures = []

        def on_describe(action, prompt_id):
            future = vision.submit_visual(
                np.zeros((12, 16, 3), dtype=np.uint8),
                action,
                lambda answer, cost: manager.add_visual_response(
                    prompt_id, answer, cost
                ),
            )
            futures.append(future)
            return future

        self.sequence.on_status = manager.update_task_status
        self.sequence.on_describe = on_describe
        self.sequence.replace(actions, request_id=request_id)
        futures[0].result(timeout=2)
        self.sequence.update(0)
        self.assertEqual(manager.chat_snapshot()[-1]["content"], "The car is red.")
        self.assertTrue(
            all(message["status"] == "completed" for message in manager.chat_snapshot())
        )
        self.assertEqual(
            manager.history[-1], {"role": "assistant", "content": "The car is red."}
        )

    def test_mixed_plan_waits_for_visual_answer_before_next_motion(self):
        self.sequence.replace(
            [describe(), {"action": "turn", "angle": 90}], request_id=3
        )
        self.sequence.update(0)
        self.assertEqual(self.sequence.active, "describe")
        self.sequence.update(0)
        self.assertEqual(self.sequence.active, "describe")
        self.future.set_result("A chair.")
        self.sequence.update(0)
        self.assertEqual(self.sequence.active, "turn")
        self.sequence.update(90)
        self.status.assert_any_call(3, "completed")

    def test_missing_visual_handler_fails_without_stopping_motion(self):
        self.sequence.replace([{"action": "turn", "angle": 90}], request_id=1)
        self.sequence.update(0)
        self.sequence.on_describe = None
        self.sequence.replace([describe()], request_id=2)
        self.status.assert_any_call(
            2, "failed", reason="visual description is unavailable"
        )
        self.assertEqual(self.sequence.active, "turn")


if __name__ == "__main__":
    unittest.main()
