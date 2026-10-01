"""Validate the standalone COCO map and its object instances."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "map" / "coco_scene.xml"


class MapSceneTests(unittest.TestCase):
    def test_scene_compiles_and_steps(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.geom_xpos).all())
        self.assertEqual(model.njnt, 0)
        self.assertGreaterEqual(model.camera("overview").id, 0)

    def test_object_classes_and_chair_colors(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        names = {"chair_red", "chair_blue", "bench", "bottle", "cup"}
        object_names = {model.body(i).name for i in range(1, model.nbody)}
        self.assertEqual(object_names, names)
        red_material = model.geom("red_seat").matid[0]
        blue_material = model.geom("blue_seat").matid[0]
        red = model.mat_rgba[red_material]
        blue = model.mat_rgba[blue_material]
        self.assertGreater(red[0], red[2])
        self.assertGreater(blue[2], blue[0])


if __name__ == "__main__":
    unittest.main()
