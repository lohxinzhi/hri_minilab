# Scene asset provenance

The two visible meshes in `scenes/assets/objects/` were converted from the
following Poly Haven assets. Poly Haven marks its assets CC0 and explicitly
permits redistribution; this record is for provenance and reproducibility,
not because CC0 requires attribution.

| Project asset | Source asset | Creator | Source page | License | Downloaded format |
|---|---|---|---|---|---|
| `objects/chair.obj` (instanced twice) | Plastic Monobloc Chair 01 | Kuutti Siitonen | <https://polyhaven.com/a/plastic_monobloc_chair_01> | CC0 | 1K glTF 2.0 package |
| `objects/couch.obj` | Sofa 01 | Kirill Sannikov | <https://polyhaven.com/a/Sofa_01> | CC0 | 1K glTF 2.0 package |

License statement: <https://polyhaven.com/license>. The source glTF packages
also contained external binary buffers and 1K image textures. The OBJ exports
contain geometry only; source textures and material maps are not needed by
the scene because MJCF materials provide the controlled object colors.

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
