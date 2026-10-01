"""Exercise dashboard HTTP, video, chat and captured logs without a real API."""

import json
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from browser_gui import BrowserGUI, BrowserState, ConsoleTee
from dialogue_manager import DialogueManager


class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.state = BrowserState()
        self.dialogue = DialogueManager()
        self.gui = BrowserGUI(self.state, self.dialogue, port=0).start(
            open_browser=False
        )
        self.addCleanup(self.gui.close)
        self.addCleanup(self.dialogue.close)

    def request(self, path, payload=None):
        request = Request(
            self.gui.url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        return urlopen(request, timeout=2)

    def test_page_chat_history_and_console_logs(self):
        with self.request("/") as response:
            html = response.read().decode()
        self.assertIn("/stream/follow", html)
        self.assertIn("/stream/fpv", html)
        self.assertIn("Console logs", html)
        with self.request("/api/chat", {"prompt": "turn left"}) as response:
            self.assertEqual(response.status, 202)
        sink = StringIO()
        tee = ConsoleTee(sink, self.state)
        print("[TURN] test output", file=tee)
        with self.request("/api/state") as response:
            snapshot = json.load(response)
        self.assertEqual(snapshot["messages"][0]["content"], "turn left")
        self.assertEqual(snapshot["messages"][0]["status"], "queued")
        self.assertEqual(snapshot["model"], self.dialogue.model)
        self.assertEqual(snapshot["actions"], [])
        self.state.set_active_action(3)
        with self.request("/api/state") as response:
            self.assertEqual(json.load(response)["active_action"], 3)
        self.state.set_active_action(None)
        self.assertEqual(
            "".join(item["text"] for item in snapshot["logs"]), sink.getvalue()
        )

    def test_camera_mjpeg_contains_encoded_frame(self):
        frame = np.full((16, 24, 3), [10, 20, 200], dtype=np.uint8)
        self.state.publish_frame("fpv", frame)
        with self.request("/stream/fpv") as response:
            self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
            self.assertEqual(response.readline(), b"--frame\r\n")
            self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
            size = int(response.readline().split(b":")[1])
            response.readline()
            jpg = response.read(size)
        decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, frame.shape)
        np.testing.assert_allclose(decoded[0, 0], frame[0, 0], atol=3)

    def test_generated_action_properties_exclude_chat_replies(self):
        actions = [
            {"action": "chat", "reply": "Ready to go"},
            {
                "action": "move",
                "velocity": {"vx": 0.5, "vy": -0.2, "wz": 0.1},
                "duration": 2.5,
            },
            {"action": "turn", "angle": -45},
            {"action": "stop"},
        ]
        self.dialogue.model = "test-cloud-model"
        self.dialogue.client = Mock()
        self.dialogue.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        refusal=None, content=json.dumps({"actions": actions})
                    ),
                )
            ]
        )
        self.dialogue.start()
        with self.request("/api/chat", {"prompt": "move then turn"}):
            pass
        plan = self.dialogue._plans.get(timeout=1)
        self.assertEqual(plan, actions)
        self.assertEqual(self.dialogue.action_history_start(plan), 0)
        with self.request("/api/state") as response:
            snapshot = json.load(response)
        self.assertEqual(snapshot["model"], "test-cloud-model")
        self.assertEqual(snapshot["actions"], [{"action": "chat"}, *actions[1:]])
        self.assertNotIn("Ready to go", json.dumps(snapshot["actions"]))
        # Chat-only responses show the action type without duplicating the reply.
        self.dialogue.client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps({"actions": actions[:1]})
        with self.request("/api/chat", {"prompt": "say hello"}):
            pass
        plan = self.dialogue._plans.get(timeout=1)
        self.assertEqual(self.dialogue.action_history_start(plan), len(actions))
        with self.request("/api/state") as response:
            self.assertEqual(
                json.load(response)["actions"],
                [{"action": "chat"}, *actions[1:], {"action": "chat"}],
            )

    def test_invalid_prompt_and_stop(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/chat", {"prompt": ""})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.dialogue.chat_snapshot(), [])
        with self.request("/api/stop", {}) as response:
            self.assertEqual(response.status, 200)
        self.assertTrue(self.state.stop_requested.is_set())

    def test_logs_are_bounded(self):
        for i in range(2500):
            self.state.append_log(str(i))
        logs = self.state.snapshot()["logs"]
        self.assertEqual(len(logs), 2000)
        self.assertEqual(logs[-1]["text"], "2499")


if __name__ == "__main__":
    unittest.main()
