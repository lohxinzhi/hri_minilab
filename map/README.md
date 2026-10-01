# COCO object map

Load `coco_scene.xml` directly with `mujoco.MjModel.from_xml_path()`.
The scene uses meters, Z up, and static collidable objects on a ground plane.
The area around the origin is open for a robot spawn. The `overview` camera
provides a view of the five objects.

| Body | COCO class | YOLO class index | Color |
| --- | --- | --- | --- |
| `chair_red` | chair | 56 | red |
| `chair_blue` | chair | 56 | blue |
| `bench` | bench | 13 | brown with metal legs |
| `bottle` | bottle | 39 | green |
| `cup` | cup | 41 | cream |

Class indices follow the Ultralytics COCO dataset:
https://docs.ultralytics.com/datasets/detect/coco/

All object geometry is defined with MJCF primitives inside the scene file;
no external meshes or textures are required. Keep any future map assets in
this directory and reference them with relative paths.

These are simplified object shapes. Recognition by a pretrained detector
has not been verified. This file does not automatically register the map
with the map selector in `play.py`.
