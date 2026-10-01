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
        red_material = model.geom("chair_red_visual").matid[0]
        blue_material = model.geom("chair_blue_visual").matid[0]
        red = model.mat_rgba[red_material]
        blue = model.mat_rgba[blue_material]
        self.assertGreater(red[0], red[2])
        self.assertGreater(blue[2], blue[0])

    def test_all_objects_use_local_meshes(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.assertEqual(model.nmesh, 4)
        for name in ("chair_red", "chair_blue", "bench", "bottle", "cup"):
            with self.subTest(name=name):
                geom = model.geom(name + "_visual")
                self.assertEqual(geom.type[0], mujoco.mjtGeom.mjGEOM_MESH)
                self.assertEqual(geom.contype[0], 0)
        for name in ("chair", "bench", "bottle", "cup"):
            self.assertTrue((SCENE.parent / "meshes" / (name + ".obj")).is_file())
        for name in ("bench", "bottle"):
            mesh_id = model.mesh(name + "_mesh").id
            self.assertGreater(model.mesh_texcoordnum[mesh_id], 0)


if __name__ == "__main__":
    unittest.main()
