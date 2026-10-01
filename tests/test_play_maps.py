"""Check COCO map registration and robot scene compatibility."""

import unittest
from types import SimpleNamespace

import numpy as np
from runtime_control import RuntimeScene

import play


class PlayMapTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
