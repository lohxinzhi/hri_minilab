"""Test the detection interface without downloading model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vision_module


class GetBboxTests(unittest.TestCase):
    def setUp(self):
        self.vision = vision_module.VisionModule()
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def test_coordinates_and_model_reuse(self):
        coordinates = [[1.5, 2.0, 30.0, 40.0], [4.0, 5.0, 10.0, 20.0]]
        xyxy = Mock()
        xyxy.cpu.return_value.tolist.return_value = coordinates
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy))
        with patch.object(vision_module, "YOLO") as constructor:
            model = constructor.return_value
            model.predict.return_value = [result]
            self.assertEqual(self.vision.get_bbox(self.frame), coordinates)
            self.assertEqual(self.vision.get_bbox(self.frame), coordinates)
            constructor.assert_called_once_with("yolov8n.pt")
            model.predict.assert_called_with(source=self.frame, verbose=False)

    def test_custom_model(self):
        with patch.object(vision_module, "YOLO") as constructor:
            constructor.return_value.predict.return_value = [
                SimpleNamespace(boxes=None)
            ]
            vision = vision_module.VisionModule(model="custom_weights.pt")
            constructor.assert_not_called()
            self.assertEqual(vision.get_bbox(self.frame), [])
            constructor.assert_called_once_with("custom_weights.pt")

    def test_labeled_boxes_preserve_class_and_confidence(self):
        boxes = Mock()
        boxes.xyxy.cpu.return_value.tolist.return_value = [[1, 2, 30, 40]]
        boxes.cls.cpu.return_value.tolist.return_value = [56]
        boxes.conf.cpu.return_value.tolist.return_value = [0.9]
        with patch.object(vision_module, "YOLO") as constructor:
            constructor.return_value.predict.return_value = [
                SimpleNamespace(boxes=boxes, names={56: "chair"})
            ]
            self.assertEqual(
                self.vision.get_bbox(self.frame, include_labels=True),
                [{"bbox": [1, 2, 30, 40], "label": "chair", "confidence": 0.9}],
            )

    def test_no_detections(self):
        xyxy = Mock()
        xyxy.cpu.return_value.tolist.return_value = []
        for boxes in (None, SimpleNamespace(xyxy=xyxy)):
            with (
                self.subTest(boxes=boxes),
                patch.object(vision_module, "YOLO") as constructor,
            ):
                constructor.return_value.predict.return_value = [
                    SimpleNamespace(boxes=boxes)
                ]
                self.assertEqual(vision_module.VisionModule().get_bbox(self.frame), [])

    def test_invalid_frames_are_rejected_before_loading_model(self):
        cases = [
            (None, TypeError),
            (np.zeros((48, 64), dtype=np.uint8), ValueError),
            (np.zeros((48, 64, 4), dtype=np.uint8), ValueError),
            (np.zeros((0, 64, 3), dtype=np.uint8), ValueError),
            (np.zeros((48, 64, 3), dtype=np.float32), ValueError),
        ]
        with patch.object(vision_module, "YOLO") as constructor:
            for frame, error in cases:
                with self.subTest(frame=frame), self.assertRaises(error):
                    self.vision.get_bbox(frame)
            constructor.assert_not_called()


class FpvStreamTests(unittest.TestCase):
    def test_all_boxes_are_labeled_with_selected_and_other_class_colors(self):
        labels = ["chair", "bench", "car", "bicycle", "bottle", "cup"]
        detections = [
            {"bbox": [5, 10, 30, 40], "label": label, "confidence": 0.8}
            for label in labels
        ]
        with (
            patch.object(
                vision_module.VisionModule, "get_bbox", return_value=detections
            ) as detect,
            patch.object(vision_module.mujoco, "Renderer") as constructor,
            patch.object(vision_module.cv2, "namedWindow"),
            patch.object(vision_module.cv2, "imshow"),
            patch.object(vision_module.cv2, "waitKey", return_value=-1),
            patch.object(vision_module.cv2, "getWindowProperty", return_value=1),
            patch.object(vision_module.cv2, "destroyWindow"),
            patch.object(vision_module.cv2, "rectangle") as rectangle,
            patch.object(vision_module.cv2, "putText") as text,
            vision_module.VisionModule() as vision,
        ):
            constructor.return_value.render.return_value = np.zeros(
                (48, 64, 3), dtype=np.uint8
            )
            self.assertTrue(vision.stream_fpv(Mock(), Mock()))
            self.assertTrue(detect.call_args.kwargs["include_labels"])
            self.assertEqual(rectangle.call_count, 6)
            self.assertEqual(
                [call.args[3] for call in rectangle.call_args_list],
                [(0, 255, 0)] * 4 + [(64, 64, 64)] * 2,
            )
            self.assertEqual(
                [call.args[1] for call in text.call_args_list],
                [label + " 0.80" for label in labels],
            )
            self.assertEqual(
                [call.args[5] for call in text.call_args_list],
                [(0, 255, 0)] * 4 + [(64, 64, 64)] * 2,
            )

    def test_camera_renderer_reuse_and_rgb_conversion(self):
        rgb = np.array([[[255, 10, 20]]], dtype=np.uint8)
        model, data = Mock(), Mock()
        with (
            patch.object(vision_module.VisionModule, "get_bbox", return_value=[]),
            patch.object(vision_module.mujoco, "Renderer") as constructor,
            patch.object(vision_module.cv2, "namedWindow") as create_window,
            patch.object(vision_module.cv2, "imshow") as show,
            patch.object(vision_module.cv2, "waitKey", return_value=-1),
            patch.object(vision_module.cv2, "getWindowProperty", return_value=1),
            patch.object(vision_module.cv2, "destroyWindow") as destroy,
            vision_module.VisionModule() as vision,
        ):
            renderer = constructor.return_value
            renderer.render.return_value = rgb
            self.assertTrue(vision.stream_fpv(model, data))
            self.assertTrue(vision.stream_fpv(model, data))
            constructor.assert_called_once_with(model, height=480, width=640)
            create_window.assert_called_once()
            renderer.update_scene.assert_called_with(data, camera="dog_front_camera")
            np.testing.assert_array_equal(
                show.call_args.args[1], np.array([[[20, 10, 255]]], dtype=np.uint8)
            )
        renderer.close.assert_called_once()
        destroy.assert_called_once()

    def test_exit_keys_and_window_close_release_resources(self):
        for key, visible in ((ord("q"), 1), (27, 1), (-1, 0)):
            with (
                self.subTest(key=key, visible=visible),
                patch.object(vision_module.VisionModule, "get_bbox", return_value=[]),
                patch.object(vision_module.mujoco, "Renderer") as constructor,
                patch.object(vision_module.cv2, "namedWindow"),
                patch.object(vision_module.cv2, "imshow"),
                patch.object(vision_module.cv2, "waitKey", return_value=key),
                patch.object(
                    vision_module.cv2, "getWindowProperty", return_value=visible
                ),
                patch.object(vision_module.cv2, "destroyWindow") as destroy,
            ):
                constructor.return_value.render.return_value = np.zeros(
                    (2, 2, 3), dtype=np.uint8
                )
                vision = vision_module.VisionModule()
                self.assertFalse(vision.stream_fpv(Mock(), Mock()))
                vision.close()
                constructor.return_value.close.assert_called_once()
                destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
