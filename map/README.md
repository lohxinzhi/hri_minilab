# COCO object map

![COCO object map preview](preview.png)

Load `coco_scene.xml` directly with `mujoco.MjModel.from_xml_path()`.
The scene uses meters, Z up, and static collidable objects on a ground plane.
Six object centers sit at 60-degree intervals on a circle of radius 6 m.
The area around the origin is open for the robot spawn. The `overview` camera
shows the six objects.

| Body | COCO class | YOLO class index | Appearance |
| --- | --- | --- | --- |
| `chair_red` | chair | 56 | red |
| `chair_blue` | chair | 56 | blue |
| `bench` | bench | 13 | green wooden slats, backrest, and metal armrests |
| `bicycle` | bicycle | 1 | yellow frame with source material colors |
| `sedan` | car | 2 | red 1971 Oldsmobile Cutlass Supreme sedan |
| `suv` | car | 2 | cream-white Toyota Land Cruiser |

Class indices follow the [Ultralytics COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/).

The objects use meshes from Sketchfab and Objaverse. Vehicle assets live in
`meshes/oldsmobile/` and `meshes/land_cruiser/`. The sedan has red body paint; the SUV retains its source appearance. The chair mesh is shared by the
red and blue instances. Invisible primitive collision proxies approximate the
visible objects, including the bench seat and backrest.

The bench is 2 m wide. The sedan is 5.3 m long and the SUV is 4.6 m long,
with each mesh uniformly scaled to preserve its proportions. Vehicle and bench
long axes are tangent to the circle. Meshes have baked node transforms,
Z-up axes, horizontal centering, and ground-level alignment.

See `ASSET_SOURCES.md` for authors, licenses, and download links, and
`mesh_transforms.json` for conversion scales, bounds, and material parts.
Only converted OBJ meshes and PNG textures are needed at runtime.
`coco_scene_positions.yaml` records object centers and navigation color labels.
Navigation color labels are `green` for the bench, `red` for the sedan,
and `gray` for the Toyota Land Cruiser.

Launch the robot from `hri_minilab` with:

```bash
python play.py --map coco_scene
```

Add `--gui` for the browser panel. The map is available as **COCO Objects**
in the panel's map selector. The robot spawns at `(0, 0, 0.42)` facing positive X.

Recognition by a pretrained detector has not been verified.
