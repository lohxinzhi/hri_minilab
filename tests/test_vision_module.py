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


if __name__ == "__main__":
    unittest.main()
