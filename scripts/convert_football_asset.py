#!/usr/bin/env python3
"""Convert the inflated Football glTF node to a textured MuJoCo OBJ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads/football_1k.gltf/football_1k.gltf"
DEFAULT_OUTPUT = PROJECT_ROOT / "scenes/assets/objects"
TARGET_DIAMETER_M = 0.5
NODE_NAME = "football_inflated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Football glTF not found: {args.source}")

    scene = trimesh.load(args.source, force="scene", process=False)
    if NODE_NAME not in scene.graph.nodes_geometry:
        raise ValueError(f"Expected glTF node {NODE_NAME!r}; found {scene.graph.nodes_geometry}")
    node_transform, geometry_name = scene.graph[NODE_NAME]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(node_transform)

    # glTF is Y-up; this +90 degree X rotation maps it to MuJoCo's Z-up.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    source_z_up_dimensions = mesh.extents.copy()
    scale = TARGET_DIAMETER_M / float(source_z_up_dimensions.max())
    mesh.apply_scale(scale)
    lower, upper = mesh.bounds
    mesh.apply_translation(
        [-0.5 * (lower[0] + upper[0]), -0.5 * (lower[1] + upper[1]), -lower[2]]
    )

    obj_text, sidecars = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=False,
        include_texture=True,
        return_texture=True,
    )
    material = sidecars.pop("material.mtl").decode("utf-8")
    texture_name = next(
        (name for name in sidecars if name.lower().endswith(".png")), None
    )
    if texture_name is None:
        raise RuntimeError("trimesh did not export the football diffuse texture")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "football.obj").write_text(
        obj_text.replace("mtllib material.mtl", "mtllib football.mtl"),
        encoding="utf-8",
    )
    (args.output_dir / "football.mtl").write_text(
        material.replace(texture_name, "football_diffuse.png"), encoding="utf-8"
    )
    (args.output_dir / "football_diffuse.png").write_bytes(sidecars[texture_name])

    print(f"source={args.source} node={NODE_NAME} geometry={geometry_name}")
    print(f"trimesh={trimesh.__version__} rotation=+90deg_x scale={scale:.8f}")
    print(f"selected_variant_z_up_dimensions={source_z_up_dimensions.tolist()}")
    print(f"bounds_min={mesh.bounds[0].tolist()}")
    print(f"bounds_max={mesh.bounds[1].tolist()}")
    print(f"dimensions={mesh.extents.tolist()}")
    print(f"vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    print("outputs=football.obj, football.mtl, football_diffuse.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
