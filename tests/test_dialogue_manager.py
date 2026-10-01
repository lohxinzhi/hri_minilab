"""Exercise dialogue history, response validation and worker lifecycle offline."""

import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import OpenAIError

import dialogue_manager
from dialogue_manager import DialogueManager, validate_actions


def completion(actions=None, *, content=None, refusal=None, finish_reason="stop"):
    if content is None:
        content = json.dumps({"actions": actions or [{"action": "stop"}]})
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ]
    )


class DialogueTests(unittest.TestCase):
    def make_manager(self):
        client = Mock()
        client.chat.completions.create.return_value = completion()
        return DialogueManager(client=client), client

    def test_user_prompt_and_full_conversation_history(self):
        manager, client = self.make_manager()
        self.assertEqual(manager.user_cmd("move forward"), [{"action": "stop"}])
        manager.user_cmd("now turn left")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(messages[1]["content"], "move forward")
        self.assertEqual(messages[-1]["content"], "now turn left")
        first = client.chat.completions.create.call_args_list[0].kwargs
        self.assertEqual(len(first["messages"]), 2)
        self.assertEqual(first["model"], "gpt-6-luna")
        self.assertTrue(first["response_format"]["json_schema"]["strict"])

    def test_model_environment_and_lazy_sdk_configuration(self):
        with (
            patch.dict(os.environ, {"OPENAI_MODEL": "custom-model"}),
            patch.object(dialogue_manager, "OpenAI") as factory,
        ):
            manager = DialogueManager()
            factory.assert_not_called()
            factory.return_value.chat.completions.create.return_value = completion()
            manager.user_cmd("stop")
            factory.assert_called_once_with(timeout=30.0, max_retries=0)
            self.assertEqual(manager.model, "custom-model")

    def test_rejects_invalid_response_without_assistant_history(self):
        for response in [
            completion(content="not JSON"),
            completion(content='{"actions": []}'),
            completion(content='{"actions": [{"action": "approach"}]}'),
            completion(refusal="refused"),
            completion(finish_reason="length"),
        ]:
            with self.subTest(response=response):
                manager, client = self.make_manager()
                client.chat.completions.create.return_value = response
                with self.assertRaises(ValueError):
                    manager.user_cmd("request")
                self.assertEqual(
                    [m["role"] for m in manager.history], ["system", "user"]
                )

    def test_invalid_action_numbers_fields_and_whole_plan(self):
        invalid = [
            [],
            [
                {
                    "action": "move",
                    "velocity": {"vx": 1, "vy": 0, "wz": 0},
                    "duration": -1,
                }
            ],
            [{"action": "turn", "angle": float("nan")}],
            [{"action": "turn", "angle": True}],
            [{"action": "stop", "extra": 0}],
            [{"action": "chat", "reply": ""}],
            [{"action": "stop"}, {"action": "jump"}],
        ]
        for plan in invalid:
            with self.subTest(plan=plan), self.assertRaises((ValueError, TypeError)):
                validate_actions(plan)

    def test_worker_api_does_not_block_polling_or_close(self):
        manager, client = self.make_manager()
        entered, release = threading.Event(), threading.Event()

        def slow_api(**kwargs):
            self.assertNotEqual(threading.current_thread(), threading.main_thread())
            entered.set()
            release.wait(2)
            return completion()

        client.chat.completions.create.side_effect = slow_api
        manager.submit_prompt("turn left")
        manager.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(manager.chat_snapshot()[0]["status"], "thinking")
            self.assertIsNone(manager.poll_actions())
            started = time.monotonic()
            manager.close()
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
            manager._thread.join(1)
        self.assertFalse(manager._thread.is_alive())
        self.assertIsNone(manager.poll_actions())

    def test_api_failure_recovers_for_next_prompt(self):
        manager, client = self.make_manager()
        client.chat.completions.create.side_effect = [
            OpenAIError("offline"),
            completion(),
        ]
        manager.submit_prompt("bad request")
        manager.submit_prompt("stop")
        manager.start()
        actions = manager._plans.get(timeout=1)
        manager.close()
        self.assertFalse(manager._thread.is_alive())
        self.assertIn("offline", manager.poll_error())
        self.assertEqual(actions, [{"action": "stop"}])
        self.assertIsNone(manager.poll_actions())
        self.assertEqual(
            [m["role"] for m in manager.history],
            ["system", "user", "user", "assistant"],
        )

    def test_chat_queue_order_and_idle_worker_shutdown(self):
        manager, client = self.make_manager()
        manager.submit_prompt("first")
        manager.submit_prompt("second")
        manager.start()
        self.assertEqual(manager._plans.get(timeout=1), [{"action": "stop"}])
        self.assertEqual(manager._plans.get(timeout=1), [{"action": "stop"}])
        manager.close()
        self.assertFalse(manager._thread.is_alive())
        prompts = [
            call.kwargs["messages"][-1]["content"]
            for call in client.chat.completions.create.call_args_list
        ]
        self.assertEqual(prompts, ["first", "second"])
        messages = manager.chat_snapshot()
        self.assertEqual(
            [m["role"] for m in messages], ["user", "user", "assistant", "assistant"]
        )
        self.assertTrue(all(m["status"] == "planned" for m in messages))
        messages[0]["content"] = "changed"
        self.assertEqual(manager.chat_snapshot()[0]["content"], "first")
        with self.assertRaises(RuntimeError):
            manager.submit_prompt("after close")

    def test_blank_and_oversized_chat_prompts_are_rejected(self):
        manager, _ = self.make_manager()
        for prompt in (None, "", " ", "x" * 8001):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                manager.submit_prompt(prompt)
        self.assertEqual(manager.chat_snapshot(), [])


if __name__ == "__main__":
    unittest.main()
