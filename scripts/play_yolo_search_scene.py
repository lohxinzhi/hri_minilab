#!/usr/bin/env python3
"""Run the pinned quadruped player on the project-owned YOLO search map.

The launcher reuses the upstream policy/keyboard/render loop via runpy. It
overrides only the runtime_control map registry and UI map/spawn config; the
upstream checkout itself is not edited. A settled dog_front_camera RGB frame
is saved to outputs/scene_test/ on each run.
"""

from __future__ import annotations

import math
import runpy
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "quadruped_mujoco"
PLAYER_PATH = UPSTREAM_ROOT / "eg" / "play.py"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT / "src"))

from minilab.platform.camera import FRONT_CAMERA_NAME, FrontCamera  # noqa: E402
import runtime_control  # noqa: E402


def _scene_description():
    config_path = PROJECT_ROOT / "scenes/configs/yolo_search_scene.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    xml_path = PROJECT_ROOT / config["map_xml"]
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Scene MJCF has no worldbody: {xml_path}")

    objects = []
    for item in config["objects"]:
        body = worldbody.find(f"./body[@name='{item['id']}']")
        if body is None:
            raise ValueError(f"Scene object body missing: {item['id']}")
        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        euler = np.fromstring(body.get("euler", "0 0 0"), sep=" ")
        if body_pos.size != 3 or euler.size != 3:
            raise ValueError(f"Invalid pose on scene body {item['id']}")

        vertices = []
        mesh_path = PROJECT_ROOT / item["mesh"]
        with mesh_path.open("r", encoding="utf-8") as mesh_file:
            for line in mesh_file:
                if line.startswith("v "):
                    vertices.append([float(value) for value in line.split()[1:4]])
        bounds = np.asarray(vertices, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 3 or len(bounds) == 0:
            raise ValueError(f"No OBJ vertices found in {mesh_path}")
        local_min, local_max = bounds.min(axis=0), bounds.max(axis=0)
        if not math.isclose(float(body_pos[2] + local_min[2]), 0.0, abs_tol=1e-7):
            raise ValueError(f"{item['id']} mesh does not rest on the scene floor")
        z_mid = (local_min[2] + local_max[2]) / 2.0
        yaw = float(euler[2])
        c, s = math.cos(yaw), math.sin(yaw)
        local_xy = (local_min[:2] + local_max[:2]) / 2.0
        center = body_pos.copy()
        center[:2] += np.array(
            [c * local_xy[0] - s * local_xy[1], s * local_xy[0] + c * local_xy[1]]
        )
        center[2] += z_mid
        corners = np.array(
            [[x, y] for x in (local_min[0], local_max[0])
                    for y in (local_min[1], local_max[1])],
            dtype=np.float64,
        )
        rotated = np.column_stack(
            (c * corners[:, 0] - s * corners[:, 1],
             s * corners[:, 0] + c * corners[:, 1])
        ) + body_pos[:2]
        objects.append({
            **item,
            "body_pos": body_pos,
            "yaw": yaw,
            "center": center,
            "world_xy_bounds": np.column_stack((rotated.min(axis=0), rotated.max(axis=0))),
        })

    if len(objects) != 3:
        raise ValueError(f"Expected exactly three configured objects, got {len(objects)}")
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            overlap = np.minimum(
                first["world_xy_bounds"][:, 1], second["world_xy_bounds"][:, 1]
            ) - np.maximum(
                first["world_xy_bounds"][:, 0], second["world_xy_bounds"][:, 0]
            )
            if np.all(overlap > 0):
                raise ValueError(
                    f"Object footprints intersect: {first['id']} / {second['id']}"
                )
    return config, xml_path, objects


def _rotation_tilt(quaternion_wxyz):
    _, x, y, _ = quaternion_wxyz
    cos_tilt = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return math.acos(cos_tilt)


def main() -> int:
    if not PLAYER_PATH.is_file():
        raise FileNotFoundError(f"Pinned upstream player not found: {PLAYER_PATH}")
    config, scene_xml, objects = _scene_description()
    map_name = str(config["scene"])
    map_label = str(config["map_label"])
    spawn = config["robot_spawn"]

    source_specs = runtime_control.bundled_map_specs
    source_runtime_config = runtime_control.make_runtime_config

    def project_map_specs(names=None):
        return {map_name: runtime_control.MapSpec(scene_xml)}

    def project_runtime_config(**kwargs):
        kwargs["maps"] = {map_name: map_label}
        kwargs["map_spawns"] = {
            map_name: {
                "position": list(spawn["position"]),
                "quaternion": list(spawn["quaternion_wxyz"]),
            }
        }
        kwargs["initial_position"] = list(spawn["position"])
        kwargs["initial_quaternion"] = list(spawn["quaternion_wxyz"])
        kwargs["command"] = (0.0, 0.0, 0.0, 0.25)
        return source_runtime_config(**kwargs)

    original_step = mujoco.mj_step
    original_argv = sys.argv[:]
    cameras: list[FrontCamera] = []
    image_for_save = None
    state_for_check = None
    scene_logged = False
    capture_after_time = 0.5

    def capture_after_step(model, data, *args, **kwargs):
        nonlocal image_for_save, state_for_check, scene_logged
        original_step(model, data, *args, **kwargs)
        if not scene_logged:
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, FRONT_CAMERA_NAME
            )
            if camera_id < 0:
                raise RuntimeError(f"Composed model lacks {FRONT_CAMERA_NAME}")
            mesh_names = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
                for i in range(model.nmesh)
            }
            for required in (f"{map_name}_chair", f"{map_name}_couch"):
                if required not in mesh_names:
                    raise RuntimeError(f"Composed model lacks mesh {required!r}")
            mesh_geom_count = sum(
                name is not None
                and name.startswith(f"map_{map_name}_")
                and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                for i in range(model.ngeom)
                for name in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)]
            )
            if mesh_geom_count != 3:
                raise RuntimeError(f"Expected 3 visible mesh geoms, found {mesh_geom_count}")
            expected_materials = {
                "chair_green": np.array([0.02, 0.82, 0.04, 1.0]),
                "chair_red": np.array([0.92, 0.025, 0.02, 1.0]),
                "couch_brown": np.array([0.48, 0.29, 0.16, 1.0]),
            }
            map_mesh_materials = set()
            for geom_id in range(model.ngeom):
                geom_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                if (
                    geom_name is not None
                    and geom_name.startswith(f"map_{map_name}_")
                    and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
                ):
                    map_mesh_materials.add(int(model.geom_matid[geom_id]))
            for suffix, rgba in expected_materials.items():
                material_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{map_name}_{suffix}"
                )
                if material_id < 0 or not np.allclose(model.mat_rgba[material_id], rgba):
                    raise RuntimeError(f"Compiled scene material mismatch: {suffix}")
                if material_id not in map_mesh_materials:
                    raise RuntimeError(f"No visible mesh uses material {suffix}")
            cameras.append(FrontCamera(
                model,
                camera_name=FRONT_CAMERA_NAME,
                width=640,
                height=480,
                render_hz=15.0,
            ))
            scene_logged = True
            print(f"[SCENE] loaded={map_name} objects={len(objects)}")
            print("[SCENE] mesh_materials=green,red,warm_brown footprints=separated")
            for obj in objects:
                center = " ".join(f"{v:.4f}" for v in obj["center"])
                print(
                    f"[OBJECT] id={obj['id']} class={obj['class']} "
                    f"position=[{center}] yaw_deg={math.degrees(obj['yaw']):.1f}"
                )

        camera = cameras[0]
        camera.update(data)
        if data.time >= capture_after_time and image_for_save is None:
            frame = camera.get_latest_frame(copy=True)
            if frame is not None:
                if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                    raise RuntimeError(
                        f"Unexpected front-camera frame {frame.shape}/{frame.dtype}"
                    )
                image_for_save = frame

        if data.qpos.size >= 7:
            state_for_check = data.qpos[:7].copy()

    args = sys.argv[1:]
    if "--duration" not in args:
        args += ["--duration", "3"]
    sys.argv = [str(PLAYER_PATH), *args]
    started = time.monotonic()
    try:
        runtime_control.bundled_map_specs = project_map_specs
        runtime_control.make_runtime_config = project_runtime_config
        mujoco.mj_step = capture_after_step
        runpy.run_path(str(PLAYER_PATH), run_name="__main__")
    finally:
        runtime_control.bundled_map_specs = source_specs
        runtime_control.make_runtime_config = source_runtime_config
        mujoco.mj_step = original_step
        sys.argv = original_argv
        for camera in cameras:
            camera.close()

    elapsed = time.monotonic() - started
    if not scene_logged or not cameras:
        raise RuntimeError("The upstream simulation did not advance")
    if state_for_check is None or not np.all(np.isfinite(state_for_check)):
        raise RuntimeError("Robot final state is missing or non-finite")
    tilt = _rotation_tilt(state_for_check[3:7])
    base_z = float(state_for_check[2])
    if not 0.20 <= base_z <= 0.80 or tilt >= 0.6:
        raise RuntimeError(
            f"Headless stability check failed: base_z={base_z:.3f} m, "
            f"tilt={math.degrees(tilt):.1f} deg"
        )
    if image_for_save is None:
        raise RuntimeError(
            f"No settled {FRONT_CAMERA_NAME} frame was captured; "
            "increase --duration"
        )

    output_path = PROJECT_ROOT / "outputs/scene_test/front_camera_scene.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_for_save, mode="RGB").save(output_path)
    print(
        f"[CAMERA] name={FRONT_CAMERA_NAME} shape={image_for_save.shape} "
        f"dtype={image_for_save.dtype}"
    )
    print(f"[CAMERA] saved={output_path}")
    print(
        f"[SCENE] robot_base_z={base_z:.3f}m "
        f"body_tilt={math.degrees(tilt):.1f}deg elapsed_wall={elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
