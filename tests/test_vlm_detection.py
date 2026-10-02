"""Validate VLM pixel boxes, worker scheduling, and approach compatibility."""

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

import play
import vision_module


def completion(payload):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(payload), refusal=None),
            )
        ],
        usage=None,
    )


def target(color="red", label="chair"):
    return {"action": "goto", "object_type": label, "object_color": color}


class VlmDetectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((48, 64, 3), (0, 0, 255), dtype=np.uint8)
        self.client = Mock()
        self.client.chat.completions.create.return_value = completion(
            {"found": True, "bbox": [5, 10, 30, 40]}
        )
        self.vision = vision_module.VisionModule(
            detection_mode="vlm", vlm_model="gpt-4o", vlm_client=self.client
        )
        self.addCleanup(self.vision.close)

    def test_detection_prompt_model_pixels_and_original_coordinates(self):
        with patch.object(vision_module, "YOLO") as yolo:
            detections, _ = self.vision.get_vlm_bbox(self.frame, "chair", "red")
        yolo.assert_not_called()
        self.assertEqual(
            detections,
            [
                {
                    "bbox": [5, 10, 30, 40],
                    "label": "chair",
                    "color": "red",
                    "confidence": None,
                }
            ],
        )
        kwargs = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(
            kwargs["messages"][0]["content"],
            vision_module.DETECTION_PROMPT.replace("{object_name}", "red chair"),
        )
        parts = kwargs["messages"][1]["content"]
        self.assertIn("64 x 48", parts[0]["text"])
        data = base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])
        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(pixels, self.frame)

    def test_any_color_and_not_found(self):
        self.client.chat.completions.create.return_value = completion(
            {"found": False, "bbox": []}
        )
        self.assertEqual(self.vision.get_vlm_bbox(self.frame, "chair")[0], [])
        prompt = self.client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertIn("Locate the chair", prompt)
        self.assertNotIn("any chair", prompt)

    def test_malformed_and_unsafe_boxes_are_rejected(self):
        payloads = [
            [],
            {"found": "true", "bbox": [1, 2, 3, 4]},
            {"found": True},
            {"found": False, "bbox": [1, 2, 3, 4]},
        ]
        for bbox in (
            [],
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [True, 2, 3, 4],
            ["1", 2, 3, 4],
            [1, 2, float("nan"), 4],
            [1, 2, float("inf"), 4],
            [-1, 2, 3, 4],
            [1, 2, 65, 4],
            [1, 2, 3, 49],
            [3, 2, 1, 4],
            [1, 4, 3, 4],
        ):
            payloads.append({"found": True, "bbox": bbox})
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.client.chat.completions.create.return_value = completion(payload)
                self.vision.get_vlm_bbox(self.frame, "chair")
        self.client.chat.completions.create.return_value.choices[
            0
        ].message.content = "not JSON"
        with self.assertRaises(ValueError):
            self.vision.get_vlm_bbox(self.frame, "chair")

    def test_worker_is_nonblocking_and_copies_detection_frame(self):
        entered, release = threading.Event(), threading.Event()
        thread_ids = []
        caller = threading.get_ident()
        expected = self.frame.copy()

        def request(**_kwargs):
            thread_ids.append(threading.get_ident())
            entered.set()
            release.wait(2)
            return completion({"found": True, "bbox": [5, 10, 30, 40]})

        self.client.chat.completions.create.side_effect = request
        self.vision.set_detection_target(target())
        try:
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertTrue(entered.wait(1))
            future = self.vision._detection_future[1]
            self.assertFalse(future.done())
            self.assertTrue(self.vision.detection_waiting)
            self.frame[:] = 0
            for _ in range(3):
                self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertEqual(self.client.chat.completions.create.call_count, 1)
        finally:
            release.set()
        self.assertEqual(future.result(timeout=2)[0]["color"], "red")
        self.assertEqual(
            self.vision._poll_vlm_detections(self.frame)[0]["label"], "chair"
        )
        self.assertFalse(self.vision.detection_waiting)
        self.assertNotEqual(thread_ids[0], caller)
        kwargs = self.client.chat.completions.create.call_args.kwargs
        payload = kwargs["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[
            1
        ]
        pixels = cv2.imdecode(
            np.frombuffer(base64.b64decode(payload), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        np.testing.assert_array_equal(pixels, expected)

    def test_refresh_reuses_last_valid_box_and_expires_only_after_age_limit(self):
        first, second = Future(), Future()
        result = [
            {
                "label": "chair",
                "color": "red",
                "bbox": [5, 10, 30, 40],
                "confidence": None,
            }
        ]
        with (
            patch.object(
                self.vision, "submit_visual", side_effect=[first, second]
            ) as submit,
            patch.object(vision_module.time, "monotonic", return_value=10) as clock,
        ):
            self.vision.set_detection_target(target())
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            first.set_result(result)
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), result)
            self.assertFalse(self.vision.detection_waiting)
            clock.return_value = 10.3
            self.assertTrue(self.vision.detection_waiting)
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), result)
            clock.return_value = 12
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), result)
            self.assertEqual(submit.call_count, 2)
            clock.return_value = 10 + vision_module.VLM_DETECTION_MAX_AGE_SECONDS
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertEqual(submit.call_count, 2)
            second.set_result([])
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertFalse(self.vision.detection_waiting)

    def test_changed_cancelled_and_restarted_targets_discard_old_results(self):
        for replacement in (target("blue"), target(), None):
            with self.subTest(replacement=replacement):
                old = Future()
                pending = Future()
                self.vision.set_detection_target(target())
                with patch.object(
                    self.vision, "submit_visual", side_effect=[old, pending]
                ) as submit:
                    self.vision._poll_vlm_detections(self.frame)
                    self.vision.set_detection_target(replacement)
                    old.set_result([{"bbox": [1, 2, 3, 4]}])
                    self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
                    self.assertEqual(submit.call_count, 1 if replacement is None else 2)
                # Finish any pending request before the next subtest.
                pending.set_result([])
                self.vision.set_detection_target(None)
                self.vision._poll_vlm_detections(self.frame)

    def test_first_not_found_is_a_received_result_and_new_target_resets_it(self):
        future = Future()
        self.vision.set_detection_target(target())
        self.assertFalse(self.vision.detection_received)
        with patch.object(self.vision, "submit_visual", return_value=future):
            self.vision._poll_vlm_detections(self.frame)
            self.assertFalse(self.vision.detection_received)
            future.set_result([])
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertTrue(self.vision.detection_received)
        self.vision.set_detection_target(target("blue"))
        self.assertFalse(self.vision.detection_received)

    def test_api_failure_holds_motion_and_throttles_retry(self):
        failed, retry = Future(), Future()
        with (
            patch.object(
                self.vision, "submit_visual", side_effect=[failed, retry]
            ) as submit,
            patch.object(vision_module.time, "monotonic", return_value=10) as clock,
            redirect_stdout(StringIO()) as output,
        ):
            self.vision.set_detection_target(target())
            self.vision._poll_vlm_detections(self.frame)
            failed.set_exception(ValueError("bad response"))
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertTrue(self.vision.detection_waiting)
            self.vision._poll_vlm_detections(self.frame)
            self.assertEqual(submit.call_count, 1)
            clock.return_value = 11
            self.vision._poll_vlm_detections(self.frame)
            self.assertEqual(submit.call_count, 2)
            self.assertIn("VLM error: bad response", output.getvalue())

    def test_vlm_overlay_is_purple_above_yolo_with_object_label_and_no_score(self):
        self.vision.set_detection_target(target())
        future = Future()
        future.set_result(self.vision.get_vlm_bbox(self.frame, "chair", "red")[0])
        self.vision._detection_future = self.vision._detection_generation, future
        yolo_boxes = [{"bbox": [1, 2, 20, 30], "label": "car", "confidence": 0.9}]
        with (
            patch.object(self.vision, "render_frame", return_value=self.frame),
            patch.object(self.vision, "get_bbox", return_value=yolo_boxes) as yolo,
            patch.object(vision_module.cv2, "putText") as text,
            patch.object(
                vision_module.cv2, "rectangle", wraps=cv2.rectangle
            ) as rectangle,
            redirect_stdout(StringIO()) as output,
        ):
            self.vision.render_fpv(Mock(), Mock())
        yolo.assert_called_once_with(self.frame, include_labels=True)
        self.assertEqual(
            [call.args[1] for call in text.call_args_list],
            ["car 0.90 red", "VLM: chair"],
        )
        self.assertEqual(
            [call.args[3] for call in rectangle.call_args_list],
            [(0, 0, 255), self.vision.VLM_BOX_COLOR],
        )
        self.assertEqual(text.call_args.args[5], self.vision.VLM_BOX_COLOR)
        np.testing.assert_array_equal(self.frame[10, 5], self.vision.VLM_BOX_COLOR)
        self.assertIn("class=car color=red conf=0.90", output.getvalue())
        self.assertEqual(self.vision.detections[0]["bbox"], [1, 2, 20, 30])
        self.assertEqual(self.vision.approach_detections[0]["bbox"], [5, 10, 30, 40])
        self.assertIsNone(self.vision.approach_detections[0]["confidence"])

    def test_cancelled_target_removes_vlm_overlay_but_keeps_yolo(self):
        self.vision.set_detection_target(target())
        self.vision.approach_detections = [
            {
                "bbox": [5, 10, 30, 40],
                "label": "chair",
                "color": "red",
                "confidence": None,
            }
        ]
        self.vision.set_detection_target(None)
        yolo_boxes = [{"bbox": [1, 2, 20, 30], "label": "car", "confidence": 0.9}]
        with (
            patch.object(self.vision, "render_frame", return_value=self.frame),
            patch.object(self.vision, "get_bbox", return_value=yolo_boxes),
            patch.object(vision_module.cv2, "putText") as text,
            patch.object(self.vision, "submit_visual") as submit,
            redirect_stdout(StringIO()),
        ):
            self.vision.render_fpv(Mock(), Mock())
        self.assertEqual(text.call_count, 1)
        self.assertEqual(text.call_args.args[1], "car 0.90 red")
        self.assertEqual(self.vision.approach_detections, [])
        submit.assert_not_called()

    def test_vlm_idle_does_not_request_detection(self):
        with patch.object(self.vision, "submit_visual") as submit:
            self.assertEqual(self.vision._poll_vlm_detections(self.frame), [])
            self.assertFalse(self.vision.detection_waiting)
            submit.assert_not_called()

    def test_mode_validation_and_yolo_default(self):
        self.assertEqual(vision_module.VisionModule().detection_mode, "yolo")
        with self.assertRaises(ValueError):
            vision_module.VisionModule(detection_mode="invalid")

    def test_vlm_box_reaches_approach_controller_without_confidence(self):
        positions = {
            "chair": {"coco_class": "chair", "expected_color": "red", "x": 5, "y": 0}
        }
        sequence = play.RobotActionSequence(object_positions=positions)
        self.addCleanup(sequence.cancel)
        detections, _ = self.vision.get_vlm_bbox(self.frame, "chair", "red")
        with (
            patch.object(play, "goto_object", return_value=np.zeros(3)) as controller,
            patch.object(play, "get_goto_status", return_value="RUNNING"),
            redirect_stdout(StringIO()),
        ):
            sequence.replace([target()])
            sequence.update(
                0, detections=detections, frame_size=(64, 48), robot_position=(0, 0)
            )
        self.assertEqual(controller.call_args.args, ("chair", "red", [5, 10, 30, 40]))


if __name__ == "__main__":
    unittest.main()
