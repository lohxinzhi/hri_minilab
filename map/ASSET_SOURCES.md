# Mesh sources and attribution

The scene uses chair, bench, bicycle, Lamborghini Centenario and Toyota Supra
meshes from Sketchfab and the public
[Objaverse dataset](https://huggingface.co/datasets/allenai/objaverse).
Author and license metadata were checked with the public Sketchfab API on
2026-10-01 for the chair and bench, and 2026-10-02 for the bicycle and cars.
Some source descriptions say CC0 while the API lists CC BY; the attribution
below follows the API license recorded on those dates.

| Local mesh | Original model / author | License |
| --- | --- | --- |
| `meshes/chair.obj` | [Country style chair](https://sketchfab.com/3d-models/d601c2907f114376bf9826272d686e81) by gerardusnl0 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/bench.obj` | [CC0 - Bench](https://sketchfab.com/3d-models/096d4f915819409f9739326118bd5aa7) by plaggy | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/bicycle.obj` | [Yellow bicycle](https://sketchfab.com/3d-models/yellow-bicycle-4cc719f387cb4cb4aea6e12a27e06e2e) by Tatyana Volkova | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/lamborghini_centenario/lamborghini_centenario_*.obj` | [lamborghini centenario](https://sketchfab.com/3d-models/lamborghini-centenario-b8d20339dc654831b6d70768be1c5ad1) by amogusstrikesback2 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/toyota_supra/toyota_supra_*.obj` | [Toyota Supra](https://sketchfab.com/3d-models/toyota-supra-b7616ec43ecf4ffd8ed810d94f15eea6) by Jiaxing | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Source files

- [Chair GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-023/d601c2907f114376bf9826272d686e81.glb)
- [Bench GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-045/096d4f915819409f9739326118bd5aa7.glb)
- [Bicycle GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-137/4cc719f387cb4cb4aea6e12a27e06e2e.glb)
- Lamborghini Centenario and Toyota Supra GLBs supplied by the user from the
  Sketchfab models linked above.

MuJoCo uses the converted OBJ meshes and PNG textures. The source GLBs are
not required at runtime.

## Mesh preparation

All meshes have baked scene/node transforms, triangulated geometry, axes rotated
from Y up to Z up, horizontal centering, and ground-level alignment. Mesh scale,
dimensions and per-part face counts are recorded in `mesh_transforms.json`.
Original animations and other PBR material channels are omitted.

The chair mesh is shared by the red and blue chairs, with colors set in MJCF.
The bench uses `meshes/bench_texture.png`, extracted from the source base-color
texture. Bicycle material colors are baked into `meshes/bicycle_texture.png`
as a solid-color texture atlas; its geometry is merged and scaled along each
axis to the recorded dimensions.

Vehicle geometry is split by source material and uniformly scaled to lengths
of 4.924 m for the Lamborghini and 4.38 m for the Supra. Vehicle textures are
stored beside their OBJ parts in their respective mesh folders and resized to
at most 1024 pixels per axis. The Supra is simplified from approximately
1.44 million to 252,446 triangles, with nearest source-vertex UVs used on
simplified parts. Its body paint is set to blue while preserving other
materials. The Lamborghini's stray light-emitter faces outside the assembled
body are removed.
