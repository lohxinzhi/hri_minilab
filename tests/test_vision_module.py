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
        conf = Mock()
        conf.cpu.return_value.tolist.return_value = [0.9, 0.8]
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy, conf=conf))
        with patch.object(vision_module, "YOLO") as constructor:
            model = constructor.return_value
            model.predict.return_value = [result]
            self.assertEqual(self.vision.get_bbox(self.frame), coordinates)
            self.assertEqual(self.vision.get_bbox(self.frame), coordinates)
            constructor.assert_called_once_with("yolov8n.pt")
            model.predict.assert_called_with(source=self.frame, conf=0.5, verbose=False)

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
        conf = Mock()
        conf.cpu.return_value.tolist.return_value = []
        for boxes in (None, SimpleNamespace(xyxy=xyxy, conf=conf)):
            with (
                self.subTest(boxes=boxes),
                patch.object(vision_module, "YOLO") as constructor,
            ):
                constructor.return_value.predict.return_value = [
                    SimpleNamespace(boxes=boxes)
                ]
                self.assertEqual(vision_module.VisionModule().get_bbox(self.frame), [])

    def test_strict_configurable_confidence_for_all_classes_and_box_formats(self):
        boxes = Mock()
        coordinates = [[1, 2, 30, 40]] * 4
        boxes.xyxy.cpu.return_value.tolist.return_value = coordinates
        boxes.cls.cpu.return_value.tolist.return_value = [56, 39, 41, 0]
        boxes.conf.cpu.return_value.tolist.return_value = [0.49, 0.5, 0.65, 0.9]
        result = SimpleNamespace(
            boxes=boxes, names={56: "chair", 39: "bottle", 41: "cup", 0: "person"}
        )
        with patch.object(vision_module, "YOLO") as constructor:
            constructor.return_value.predict.return_value = [result]
            for threshold, expected in ((0.5, ["cup", "person"]), (0.65, ["person"])):
                with self.subTest(threshold=threshold):
                    vision = vision_module.VisionModule(confidence_threshold=threshold)
                    labeled = vision.get_bbox(self.frame, include_labels=True)
                    self.assertEqual([item["label"] for item in labeled], expected)
                    self.assertEqual(
                        vision.get_bbox(self.frame), coordinates[: len(expected)]
                    )
                    self.assertEqual(
                        constructor.return_value.predict.call_args.kwargs["conf"],
                        threshold,
                    )

    def test_invalid_confidence_thresholds(self):
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                vision_module.VisionModule(confidence_threshold=threshold)

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


class ObjectColorTests(unittest.TestCase):
    def test_dark_blue_background_is_excluded_from_all_hsv_medians(self):
        for name, object_hsv in (
            ("red", (0, 255, 255)),
            ("yellow", (30, 255, 255)),
            ("blue", (120, 255, 255)),
            ("white", (0, 0, 255)),
            ("black", (0, 0, 0)),
        ):
            with self.subTest(name=name):
                hsv = np.full((10, 10, 3), (115, 200, 80), dtype=np.uint8)
                hsv[4:6, 4:6] = object_hsv
                frame = vision_module.cv2.cvtColor(hsv, vision_module.cv2.COLOR_HSV2BGR)
                actual_name, bgr = vision_module.VisionModule._object_color(
                    frame, [0, 0, 10, 10]
                )
                self.assertEqual(actual_name, name)
                np.testing.assert_allclose(bgr, frame[4, 4], atol=1)

    def test_only_dark_blue_pixels_return_unknown(self):
        hsv = np.full((10, 10, 3), (115, 200, 80), dtype=np.uint8)
        frame = vision_module.cv2.cvtColor(hsv, vision_module.cv2.COLOR_HSV2BGR)
        self.assertEqual(
            vision_module.VisionModule._object_color(frame, [0, 0, 10, 10]),
            ("unknown", (128, 128, 128)),
        )

    def test_dark_colors_outside_blue_hue_range_are_retained(self):
        for name, object_hsv in (
            ("red", (0, 255, 80)),
            ("green", (60, 255, 80)),
            ("purple", (140, 255, 80)),
            ("gray", (115, 20, 80)),
        ):
            with self.subTest(name=name):
                hsv = np.full((10, 10, 3), object_hsv, dtype=np.uint8)
                frame = vision_module.cv2.cvtColor(hsv, vision_module.cv2.COLOR_HSV2BGR)
                self.assertEqual(
                    vision_module.VisionModule._object_color(frame, [0, 0, 10, 10])[0],
                    name,
                )

    def test_chromatic_and_neutral_colors(self):
        cases = [
            ("red", (0, 255, 255)),
            ("orange", (15, 255, 255)),
            ("brown", (15, 255, 120)),
            ("yellow", (30, 255, 255)),
            ("green", (60, 255, 255)),
            ("cyan", (90, 255, 255)),
            ("blue", (120, 255, 255)),
            ("purple", (140, 255, 255)),
            ("pink", (160, 255, 255)),
            ("white", (0, 0, 255)),
            ("gray", (0, 0, 128)),
            ("black", (0, 0, 0)),
        ]
        for name, hsv in cases:
            with self.subTest(name=name):
                frame = vision_module.cv2.cvtColor(
                    np.full((10, 10, 3), hsv, dtype=np.uint8),
                    vision_module.cv2.COLOR_HSV2BGR,
                )
                actual_name, bgr = vision_module.VisionModule._object_color(
                    frame, [0, 0, 10, 10]
                )
                self.assertEqual(actual_name, name)
                np.testing.assert_allclose(bgr, frame[0, 0], atol=1)

    def test_median_resists_outliers_and_clips_boxes(self):
        frame = np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8)
        frame[:2] = (0, 255, 0)
        self.assertEqual(
            vision_module.VisionModule._object_color(frame, [-5, -5, 15, 15]),
            ("blue", (255, 0, 0)),
        )
        for bbox in ([20, 20, 30, 30], [-20, -20, -10, -10], [5, 5, 5, 8]):
            with self.subTest(bbox=bbox):
                self.assertEqual(
                    vision_module.VisionModule._object_color(frame, bbox),
                    ("unknown", (128, 128, 128)),
                )

    def test_red_hue_wraparound(self):
        hsv = np.full((10, 10, 3), (1, 255, 255), dtype=np.uint8)
        hsv[:5, :, 0] = 179
        frame = vision_module.cv2.cvtColor(hsv, vision_module.cv2.COLOR_HSV2BGR)
        self.assertEqual(
            vision_module.VisionModule._object_color(frame, [0, 0, 10, 10]),
            ("red", (0, 0, 255)),
        )


class FpvStreamTests(unittest.TestCase):
    def test_all_classes_use_object_color_boxes_and_white_labels(self):
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
            constructor.return_value.render.return_value[:, :] = (255, 0, 0)
            self.assertTrue(vision.stream_fpv(Mock(), Mock()))
            self.assertTrue(detect.call_args.kwargs["include_labels"])
            self.assertEqual(rectangle.call_count, 6)
            self.assertEqual(
                [call.args[3] for call in rectangle.call_args_list],
                [(0, 0, 255)] * 6,
            )
            self.assertEqual(
                [call.args[1] for call in text.call_args_list],
                [label + " 0.80 red" for label in labels],
            )
            self.assertEqual(
                [call.args[5] for call in text.call_args_list],
                [(255, 255, 255)] * 6,
            )

    def test_overlapping_annotations_do_not_change_color_estimates(self):
        frame = np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8)
        detections = [{"bbox": [5, 10, 30, 40], "label": "cup", "confidence": 0.9}] * 2
        with (
            patch.object(
                vision_module.VisionModule, "render_frame", return_value=frame
            ),
            patch.object(
                vision_module.VisionModule, "get_bbox", return_value=detections
            ),
            patch.object(
                vision_module.cv2, "rectangle", side_effect=lambda *args: frame.fill(0)
            ),
            patch.object(vision_module.cv2, "putText") as text,
        ):
            vision_module.VisionModule().render_fpv(Mock(), Mock())
            self.assertEqual(
                [call.args[1] for call in text.call_args_list],
                ["cup 0.90 blue"] * 2,
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
