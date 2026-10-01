# COCO object map

Load `coco_scene.xml` directly with `mujoco.MjModel.from_xml_path()`.
The scene uses meters, Z up, and static collidable objects on a ground plane.
The six object centers are evenly spaced at 60-degree intervals around a
circle of radius 6 m. The blue Supra lies between the bench and Lamborghini. The area around the origin is open for a robot spawn. The `overview` camera
provides a view of the six objects.

| Body | COCO class | YOLO class index | Color |
| --- | --- | --- | --- |
| `chair_red` | chair | 56 | red |
| `chair_blue` | chair | 56 | blue |
| `bench` | bench | 13 | textured wood and supports |
| `bicycle` | bicycle | 1 | yellow frame with original material colors |
| `car` | car | 2 | Lamborghini Centenario with original textures |
| `supra_blue` | car | 2 | Toyota Supra with blue body paint |

Class indices follow the Ultralytics COCO dataset:
https://docs.ultralytics.com/datasets/detect/coco/

The objects use downloaded meshes from Sketchfab/Objaverse.
Car assets are grouped in `meshes/lamborghini_centenario/` and
`meshes/toyota_supra/`. OBJ files and extracted textures are referenced by relative
paths. The chair mesh is reused with red and blue materials. Invisible primitive collision proxies keep
simulation contacts inexpensive; they approximate the visible meshes.

See `ASSET_SOURCES.md` for authors, licenses, and download links, and
`mesh_transforms.json` for the conversion scale and mesh bounds. Meshes
were converted from GLB, rotated from Y up to Z up, scaled to
meters (the replacement cars are uniformly scaled), and placed with their lowest vertex at ground level. Vehicle and bench long axes are tangent to the circle.

Launch the robot from `hri_minilab` with:

```bash
python play.py --map coco_scene
```

Add `--gui` for the browser panel. The map is also available as
**COCO Objects** in the panel's map selector. The robot spawns at
`(0, 0, 0.42)` facing positive X.

The source GLBs were removed after conversion; they are not needed at runtime.
MuJoCo loads only the converted OBJ parts and extracted PNG textures.
The Supra is simplified to reduce rendering cost, retaining its material groups.

Recognition by a pretrained detector has not been verified.
