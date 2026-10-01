# COCO object map

Load `coco_scene.xml` directly with `mujoco.MjModel.from_xml_path()`.
The scene uses meters, Z up, and static collidable objects on a ground plane.
The area around the origin is open for a robot spawn. The `overview` camera
provides a view of the five objects.

| Body | COCO class | YOLO class index | Color |
| --- | --- | --- | --- |
| `chair_red` | chair | 56 | red |
| `chair_blue` | chair | 56 | blue |
| `bench` | bench | 13 | textured wood and supports |
| `bottle` | bottle | 39 | textured historical bottle with chain |
| `cup` | cup | 41 | cream |

Class indices follow the Ultralytics COCO dataset:
https://docs.ultralytics.com/datasets/detect/coco/

The objects use downloaded Sketchfab meshes from the Objaverse mirror.
OBJ files and extracted textures are in `meshes/`, referenced by relative
paths. The chair mesh is reused with red and blue materials. The cup model
includes its saucer and spoon. Invisible primitive collision proxies keep
simulation contacts inexpensive; they approximate the visible meshes.

See `ASSET_SOURCES.md` for authors, licenses, and download links, and
`mesh_transforms.json` for the conversion scale and mesh bounds. Meshes
were converted from GLB, rotated from Y up to Z up, uniformly scaled to
meters, and placed with their lowest vertex at ground level. The chairs
also rotate -90 degrees around Z in the scene.

Launch the robot from `hri_minilab` with:

```bash
python play.py --map coco_scene
```

Add `--gui` for the browser panel. The map is also available as
**COCO Objects** in the panel's map selector. The robot spawns at
`(0, 0, 0.42)` facing positive X.

Recognition by a pretrained detector has not been verified.
