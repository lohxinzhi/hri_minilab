"""Validate the standalone COCO map and its object instances."""

import json
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
        names = {"chair_red", "chair_blue", "bench", "sedan", "bicycle", "suv"}
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
            "sedan",
            "bicycle",
            "suv",
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
        for name in ("bench", "sedan", "bicycle"):
            mesh_id = model.mesh(name + "_mesh").id
            self.assertGreater(model.mesh_texcoordnum[mesh_id], 0)

    def test_vehicle_source_materials_and_collision_bounds(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        metadata = json.loads((SCENE.parent / "mesh_transforms.json").read_text())
        for body_name, source_name, uid in (
            ("sedan", "oldsmobile", "78f76d386a4341b0b71745bdc50fd5ab"),
            ("suv", "land_cruiser", "91b5815c64eb43b0a88f6fdb9df774e4"),
        ):
            record = metadata[source_name]
            self.assertEqual(record["objaverse_uid"], uid)
            body = model.body(body_name)
            geom_ids = range(body.geomadr[0], body.geomadr[0] + body.geomnum[0])
            visuals = [i for i in geom_ids if model.geom_contype[i] == 0]
            collisions = [i for i in geom_ids if model.geom_contype[i] != 0]
            self.assertEqual(len(visuals), len(record["parts"]))
            self.assertEqual(len(collisions), 1)
            np.testing.assert_allclose(
                model.geom_size[collisions[0]] * 2, record["dimensions_m"]
            )
            for geom_id, part in zip(visuals, record["parts"]):
                material_id = model.geom_matid[geom_id]
                np.testing.assert_allclose(model.mat_rgba[material_id], part["rgba"])
                if part["texture"]:
                    self.assertGreaterEqual(model.mat_texid[material_id, 1], 0)

    def test_bench_has_visible_and_collidable_backrest(self):
        vertices = np.array(
            [
                [float(value) for value in line.split()[1:4]]
                for line in (SCENE.parent / "meshes" / "bench.obj")
                .read_text()
                .splitlines()
                if line.startswith("v ")
            ]
        )
        self.assertAlmostEqual(np.ptp(vertices[:, 0]), 2.0, places=5)
        self.assertAlmostEqual(vertices[:, 2].min(), 0.0, places=5)
        # The tall backrest sits behind the seat in the bench's local frame.
        back = vertices[(vertices[:, 2] > 0.8) & (vertices[:, 1] < -0.2)]
        self.assertGreater(len(back), 0)
        self.assertGreater(np.ptp(back[:, 0]), 1.5)
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        seat = model.geom("bench_seat_collision")
        backrest = model.geom("bench_back_collision")
        self.assertEqual(backrest.bodyid[0], model.body("bench").id)
        self.assertNotEqual(backrest.contype[0], 0)
        self.assertLess(backrest.pos[1], seat.pos[1])
        self.assertGreater(backrest.pos[2], seat.pos[2])

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
