"""Check COCO map registration and robot scene compatibility."""

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from runtime_control import RuntimeControl, RuntimeScene

import play


class PlayMapTests(unittest.TestCase):
    def test_invalid_vision_rates_are_rejected_before_scene_creation(self):
        for rate in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(rate=rate),
                patch.object(play.sys, "argv", ["play.py", "--vision-hz", rate]),
                patch.object(play, "RuntimeScene") as scene,
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                play.main()
            self.assertEqual(error.exception.code, 2)
            scene.assert_not_called()

    def make_config(self, map_name):
        args = SimpleNamespace(map=map_name, gui=False, gui_port=8765)
        return play.build_runtime_config(args, [40.0], [1.0])

    def test_coco_startup_config(self):
        config = self.make_config("coco_scene")
        self.assertEqual(config["runtime_ui"]["default_map"], "coco_scene")
        self.assertEqual(config["runtime_ui"]["maps"]["coco_scene"], "COCO Objects")
        self.assertEqual(config["simulation"]["initial_position"], [0.0, 0.0, 0.42])

    def test_original_map_spawn(self):
        config = self.make_config("rc26_track")
        self.assertEqual(config["runtime_ui"]["default_map"], "rc26_track")
        self.assertEqual(config["simulation"]["initial_position"], [3.7, -9.0, 0.45])

    def test_composed_robot_map(self):
        with RuntimeScene(
            robot_xml=play.DEFAULT_ROBOT_XML,
            map_specs=play.MAP_SPECS,
            runtime_config=self.make_config("coco_scene"),
            robot_body_name="trunk",
            robot_cameras=play.ROBOT_CAMERAS,
            expected_dimensions=(19, 18, 12),
        ) as scene:
            spawn = scene.runtime.maps.activate(scene.model, scene.data, "coco_scene")
            self.assertEqual(spawn["position"], [0.0, 0.0, 0.42])
            geom = scene.model.geom("map_coco_scene_0")
            self.assertGreater(geom.contype[0], 0)
            self.assertTrue(np.isfinite(scene.data.qpos).all())

    def test_gui_initializes_fpv_before_runtime_updates(self):
        make_config = play.make_runtime_config
        update_runtime = RuntimeControl.runtime_control

        def config_without_browser(**kwargs):
            kwargs["gui"] = False
            return make_config(**kwargs)

        with (
            patch.object(play, "VisionModule") as constructor,
            patch.object(
                play, "make_runtime_config", side_effect=config_without_browser
            ),
            patch.object(
                play.sys,
                "argv",
                [
                    "play.py",
                    "--gui",
                    "--map",
                    "coco_scene",
                    "--no-policy",
                    "--duration",
                    "0.02",
                    "--vision-hz",
                    "10",
                ],
            ),
            redirect_stdout(StringIO()),
        ):
            vision = constructor.return_value
            vision.stream_fpv.return_value = True

            def update_after_fpv(runtime, model, data):
                self.assertGreater(vision.stream_fpv.call_count, 0)
                return update_runtime(runtime, model, data)

            with patch.object(
                RuntimeControl,
                "runtime_control",
                autospec=True,
                side_effect=update_after_fpv,
            ) as update:
                play.main()
                self.assertGreater(update.call_count, 0)


if __name__ == "__main__":
    unittest.main()
