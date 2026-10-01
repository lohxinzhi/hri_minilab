"""Weight-free unit tests for project-owned perception helpers."""

from types import SimpleNamespace
import unittest

import numpy as np

from minilab.perception.color import HSVColorClassifier, clip_bbox_xyxy
from minilab.perception.detector import detections_from_result


class PerceptionHelperTests(unittest.TestCase):
    def setUp(self):
        self.classifier = HSVColorClassifier()

    def test_rgb_red_is_classified_red(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (235, 25, 25)  # RGB red, intentionally not BGR
        result = self.classifier.classify(frame, (5, 5, 95, 75))
        self.assertEqual(result.color, "red")
        self.assertGreater(result.red_fraction, 0.9)

    def test_rgb_green_is_classified_green(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (20, 220, 35)
        result = self.classifier.classify(frame, (5, 5, 95, 75))
        self.assertEqual(result.color, "green")
        self.assertGreater(result.green_fraction, 0.9)

    def test_low_saturation_gray_is_unknown(self):
        frame = np.full((80, 100, 3), 128, dtype=np.uint8)
        result = self.classifier.classify(frame, (5, 5, 95, 75))
        self.assertEqual(result.color, "unknown")
        self.assertEqual(result.eligible_fraction, 0.0)

    def test_bbox_clips_at_image_boundaries(self):
        self.assertEqual(clip_bbox_xyxy((-4.2, 10.1, 105.0, 90.0), 100, 80),
                         (0, 10, 100, 80))

    def test_ultralytics_like_result_converts_to_plain_detection(self):
        box = SimpleNamespace(
            cls=np.array([56.0]), conf=np.array([0.875]),
            xyxy=np.array([[1.0, 2.0, 11.0, 22.0]]),
        )
        result = SimpleNamespace(boxes=[box], names={56: "chair"})
        detections = detections_from_result(result)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.class_name, "chair")
        self.assertEqual(detection.class_id, 56)
        self.assertEqual(detection.bbox_xyxy, (1.0, 2.0, 11.0, 22.0))
        self.assertEqual(detection.bbox_center, (6.0, 12.0))
        self.assertEqual(detection.bbox_width, 10.0)
        self.assertEqual(detection.bbox_height, 20.0)
        self.assertEqual(detection.bbox_area, 200.0)


if __name__ == "__main__":
    unittest.main()
