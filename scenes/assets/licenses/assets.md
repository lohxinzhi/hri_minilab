# Scene asset provenance

The two visible meshes in `scenes/assets/objects/` were converted from the
following Poly Haven assets. Poly Haven marks its assets CC0 and explicitly
permits redistribution; this record is for provenance and reproducibility,
not because CC0 requires attribution.

| Project asset | Source asset | Creator | Source page | License | Downloaded format |
|---|---|---|---|---|---|
| `objects/chair.obj` (instanced twice) | Plastic Monobloc Chair 01 | Kuutti Siitonen | <https://polyhaven.com/a/plastic_monobloc_chair_01> | CC0 | 1K glTF 2.0 package |
| `objects/couch.obj` | Sofa 01 | Kirill Sannikov | <https://polyhaven.com/a/Sofa_01> | CC0 | 1K glTF 2.0 package |
| `objects/football.obj` | Football | Amal Kumar | <https://polyhaven.com/a/football> | CC0 | 1K glTF 2.0 package |

License statement: <https://polyhaven.com/license>. The source glTF packages
also contain external binary buffers and 1K image textures. The chair and
couch OBJ exports contain geometry only because their controlled colors come
from MJCF materials. The football OBJ retains UV coordinates and uses its
diffuse panel texture, converted to PNG for MuJoCo. Unused normal/roughness
maps and source binary buffers are not included.

## Conversion record

- Tool: `trimesh` 5.1.0 in the `quadruped_mujoco` Conda environment.
- Input discovery: one `.gltf` file in each named 1K asset directory; both
  glTF scenes contain one mesh and one identity-transform node.
- Scene transforms: loaded with `trimesh.load(..., force="scene")` and
  combined using `Scene.to_geometry()`.
- Axis conversion: one +90 degree rotation about X, mapping glTF Y-up
  `(x, y, z)` to MuJoCo Z-up `(x, -z, y)`.
- Placement normalization: recentered each mesh footprint at local X/Y zero
  and shifted its minimum Z to zero. No scale adjustment was applied.
- Export: triangulated OBJ geometry without textures, colors, or generated
  normals. MJCF requests smooth normals when MuJoCo loads each mesh.
- The source backrests are toward glTF negative Z. After the axis conversion,
  the nominal object front points along local negative Y. A +90 degree MJCF
  yaw presents that front toward the robot at the scene origin (world -X).

| Project mesh | Vertices | Triangular faces | Final bounds (m), min → max | Final size (m), X × Y × Z |
|---|---:|---:|---|---|
| `objects/chair.obj` | 3,271 | 3,356 | `[-0.320931, -0.313796, 0]` → `[0.320931, 0.313796, 0.879841]` | `0.641861 × 0.627592 × 0.879841` |
| `objects/couch.obj` | 2,708 | 4,101 | `[-0.785734, -0.329007, 0]` → `[0.785734, 0.329007, 0.796515]` | `1.571468 × 0.658015 × 0.796515` |

## Football conversion

- Source: official [Football](https://polyhaven.com/a/football) page; creator
  Amal Kumar; CC0. The page offers 1K glTF and lists the complete asset as
  approximately 0.5 m wide. Poly Haven permits redistribution of CC0 assets;
  this record is for provenance.
- The downloaded glTF contains two meshes/nodes: `football_deflated` and
  `football_inflated`. Each has an explicit translation; the inflated node is
  translated by `[0.15, 0.111, 0]` m in glTF coordinates. The active scene uses
  only the inflated mesh so it contains one football. Its node transform is
  applied before axis conversion. The asset-page 0.5 m width spans the two
  side-by-side variants; the selected inflated variant is about 0.222 m across
  before scaling.
- Conversion uses `trimesh` 5.1.0. The inflated mesh is rotated +90 degrees
  about X from glTF Y-up to MuJoCo Z-up, uniformly scaled by 2.25342263 to a
  0.5 m diameter, centered in local X/Y, and shifted so its bottom is Z=0.
- The OBJ preserves UV coordinates and the source football diffuse/panel
  texture. The 1K JPEG diffuse image is decoded and exported as PNG because
  MuJoCo accepts PNG/KTX 2D texture files. The source normal and packed ARM
  maps are not used. `football.mtl` is retained as an OBJ companion; MJCF
  binds the same diffuse PNG through a MuJoCo texture/material.
- Final inflated mesh: 1,187 vertices and 1,920 faces; bounds
  `[-0.248234, -0.248538, 0]` → `[0.248234, 0.248538, 0.5]` m; dimensions
  `0.496468 × 0.497075 × 0.500000` m.
