# Mesh sources and attribution

The scene contains chair, bench, bicycle, Oldsmobile sedan, and Toyota Land Cruiser
meshes from Sketchfab and the public
[Objaverse dataset](https://huggingface.co/datasets/allenai/objaverse).
Sketchfab API author and license metadata were checked on 2026-10-01 for the
chair, 2026-10-02 for the bicycle, and 2026-10-03 for the bench and vehicles.
Attribution follows the API license metadata.

| Local mesh | Original model / author | License |
| --- | --- | --- |
| `meshes/chair.obj` | [Country style chair](https://sketchfab.com/3d-models/d601c2907f114376bf9826272d686e81) by gerardusnl0 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/bench.obj` | [uvbancexpo](https://sketchfab.com/3d-models/9ef388cd9cb947f48957a497e8b8184a) by Giriga | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/bicycle.obj` | [Yellow bicycle](https://sketchfab.com/3d-models/yellow-bicycle-4cc719f387cb4cb4aea6e12a27e06e2e) by Tatyana Volkova | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/oldsmobile/oldsmobile_*.obj` | [Oldsmobile Cutlass Supreme Sedan '71](https://sketchfab.com/3d-models/78f76d386a4341b0b71745bdc50fd5ab) by Barbo | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/land_cruiser/land_cruiser_*.obj` | [Toyota Land Cruiser](https://sketchfab.com/3d-models/91b5815c64eb43b0a88f6fdb9df774e4) by Renafox | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

## Source files

- [Chair GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-023/d601c2907f114376bf9826272d686e81.glb)
- [Bench GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-034/9ef388cd9cb947f48957a497e8b8184a.glb)
- [Bicycle GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-137/4cc719f387cb4cb4aea6e12a27e06e2e.glb)
- [Oldsmobile sedan GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-080/78f76d386a4341b0b71745bdc50fd5ab.glb)
- [Toyota Land Cruiser GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-038/91b5815c64eb43b0a88f6fdb9df774e4.glb)

MuJoCo uses the converted OBJ meshes and PNG textures; source GLBs are not
required at runtime.

## Mesh preparation

Meshes have baked node transforms, triangulated geometry, Z-up axes,
horizontal centering, and ground-level alignment. Conversion scales,
dimensions, and per-material face counts are recorded in `mesh_transforms.json`.
Animations and PBR channels other than base color are omitted.

The chair mesh is shared by the red and blue chairs, with colors set in MJCF.
The 2 m wide bench includes wooden seat and backrest slats and metal armrests.
Its base-color texture is stored in `meshes/bench_texture.png`, with wood
recolored bright green and gray metal retained using the built-in imagegen tool.
Bicycle material colors are baked into `meshes/bicycle_texture.png` as a
solid-color atlas; its geometry is merged and scaled along each axis to the
recorded dimensions.

Vehicle geometry is grouped by source material and uniformly scaled to
5.3 m in length for the sedan and 4.6 m for the SUV. UV coordinates and material color factors are preserved. The sedan
body-paint texture is recolored red using the built-in imagegen tool;
other vehicle textures retain their source colors. Extracted
PNG textures are stored beside the OBJ parts. The sedan contains 70,256
triangles across eight material groups; the SUV contains 4,396 triangles
across two groups. Source presentation ground planes are omitted.
