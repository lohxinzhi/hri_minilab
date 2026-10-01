"""Validate the standalone COCO map and its object instances."""

import unittest
import xml.etree.ElementTree as ET
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
        names = {"chair_red", "chair_blue", "bench", "car", "bicycle", "supra_blue"}
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
        self.assertGreaterEqual(model.nmesh, 5)
        for name in (
            "chair_red",
            "chair_blue",
            "bench",
            "car",
            "bicycle",
            "supra_blue",
        ):
            with self.subTest(name=name):
                geom = model.geom(name + "_visual")
                self.assertEqual(geom.type[0], mujoco.mjtGeom.mjGEOM_MESH)
                self.assertEqual(geom.contype[0], 0)
        assets = ET.parse(SCENE).getroot().find("asset")
        for asset in assets:
            file = asset.get("file")
            if file:
                self.assertTrue((SCENE.parent / "meshes" / file).is_file())
                self.assertNotEqual(Path(file).suffix, ".glb")
        for name in ("bench", "car", "bicycle"):
            mesh_id = model.mesh(name + "_mesh").id
            self.assertGreater(model.mesh_texcoordnum[mesh_id], 0)

    def test_replacement_vehicle_materials_and_collision_bounds(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        supra_material = model.geom("supra_blue_visual").matid[0]
        color = model.mat_rgba[supra_material]
        self.assertGreater(color[2], color[0])
        for body_name in ("car", "supra_blue"):
            body = model.body(body_name)
            geom_ids = range(body.geomadr[0], body.geomadr[0] + body.geomnum[0])
            visuals = [i for i in geom_ids if model.geom_contype[i] == 0]
            collisions = [i for i in geom_ids if model.geom_contype[i] != 0]
            self.assertGreater(len(visuals), 1)
            self.assertEqual(len(collisions), 1)
            # Collision proxy spans the model's documented dimensions.
            self.assertGreater(model.geom_size[collisions[0], 0] * 2, 4.0)

    def test_objects_are_evenly_spaced_on_six_meter_circle(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        positions = np.array([model.body(i).pos[:2] for i in range(1, model.nbody)])
        np.testing.assert_allclose(np.linalg.norm(positions, axis=1), 6.0)
        angles = np.sort(
            np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2 * np.pi)
        )
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        np.testing.assert_allclose(gaps, np.deg2rad(60))

    def test_robot_spawn_area_is_clear(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # A conservative horizontal bounding sphere for each collision proxy
        # must leave at least half a meter around the robot's spawn.
        for i in range(model.ngeom):
            if model.geom_bodyid[i] and model.geom_contype[i]:
                center_distance = np.linalg.norm(data.geom_xpos[i, :2])
                radius = np.linalg.norm(model.geom_size[i, :2])
                self.assertGreater(center_distance - radius, 0.5)


if __name__ == "__main__":
    unittest.main()
