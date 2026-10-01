# Mesh sources and attribution

Downloaded from the public [Objaverse dataset](https://huggingface.co/datasets/allenai/objaverse).
License and author metadata were checked with the public Sketchfab API on
2026-10-01. Some descriptions say CC0 while the current API lists CC BY;
the attribution below follows the current API license.

| Local mesh | Original model / author | License |
| --- | --- | --- |
| `meshes/chair.obj` | [Country style chair](https://sketchfab.com/3d-models/d601c2907f114376bf9826272d686e81) by gerardusnl0 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/bench.obj` | [CC0 - Bench](https://sketchfab.com/3d-models/096d4f915819409f9739326118bd5aa7) by plaggy | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/cup.obj` | [Coffee Cup for Blender](https://sketchfab.com/3d-models/a9e6ba591b6c4934b1531ccbc09e634a) by kananav | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `meshes/bottle.obj` | [Bottle with a chain](https://sketchfab.com/3d-models/8cf8bec1b7584783958ffe58ec580ea5) by Virtual Museums of Małopolska | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

Original GLB download links:

- [Chair](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-023/d601c2907f114376bf9826272d686e81.glb)
- [Bench](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-045/096d4f915819409f9739326118bd5aa7.glb)
- [Cup](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-158/a9e6ba591b6c4934b1531ccbc09e634a.glb)
- [Bottle](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-004/8cf8bec1b7584783958ffe58ec580ea5.glb)

Changes: converted GLB scene geometry to triangulated OBJ, applied scene
node transforms, rotated axes, scaled, and recentered. Chair colors and cup
materials are overridden in MJCF. Bench and bottle base-color textures are
extracted from the original files; other PBR material channels are omitted.
The original GLB files are not needed at runtime.

## Current scene vehicle replacements

- `meshes/car.obj` and `meshes/car_texture.png`: **sedan** from
  [Kenney Car Kit](https://kenney.nl/assets/car-kit), by Kenney,
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
  [Original ZIP](https://kenney.nl/media/pages/assets/car-kit/1a312ec241-1775131960/kenney_car-kit.zip).
  Source: `Models/GLB format/sedan.glb` and `Textures/colormap.png`.
- `meshes/bicycle.obj` and `meshes/bicycle_texture.png`:
  [Yellow bicycle](https://sketchfab.com/3d-models/yellow-bicycle-4cc719f387cb4cb4aea6e12a27e06e2e),
  by Tatyana Volkova, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  [Original GLB from Objaverse](https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-137/4cc719f387cb4cb4aea6e12a27e06e2e.glb).
  Author and license verified using the Sketchfab API on 2026-10-02.

Changes: baked node transforms, triangulated and merged geometry, rotated
Y up to Z up, centered horizontally, grounded, and scaled along each axis
to the dimensions in `mesh_transforms.json`. The sedan retains its original
color texture. Bicycle material colors are baked into a solid-color texture
atlas. Original animations and other PBR channels are omitted.
The cup and bottle assets listed above are retained but no longer used.

## Blue SUV

`meshes/suv.obj` and `meshes/suv_texture.png`: **suv** from
[Kenney Car Kit](https://kenney.nl/assets/car-kit), by Kenney,
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Uses `Models/GLB format/suv.glb` and its `Textures/colormap.png` from the
same ZIP linked above. Node transforms are baked, geometry triangulated
and merged, axes rotated, and dimensions scaled to 4.6 × 1.95 × 1.8 m.
The original green body-paint palette is recolored blue; trim is preserved.

## Current Sketchfab vehicle replacements

- `meshes/lamborghini_centenario/lamborghini_centenario_*.obj/png` (converted from `lamborghini_centenario.glb`):
  [lamborghini centenario](https://sketchfab.com/3d-models/lamborghini-centenario-b8d20339dc654831b6d70768be1c5ad1)
  by amogusstrikesback2, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- `meshes/toyota_supra/toyota_supra_*.obj/png` (converted from `toyota_supra.glb`):
  [Toyota Supra](https://sketchfab.com/3d-models/toyota-supra-b7616ec43ecf4ffd8ed810d94f15eea6)
  by Jiaxing, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source GLBs supplied by the user. Author and license verified with the
Sketchfab API on 2026-10-02. These replace the Kenney sedan and SUV above;
the old converted assets are retained but no longer referenced by the scene.

Changes: baked scene transforms, rotated Y up to Z up, centered and grounded,
uniformly scaled to lengths 4.924 m and 4.38 m, and split geometry by source
material. Textures are extracted and resized to at most 1024 pixels per axis;
other PBR channels are omitted. The Supra is simplified from approximately
1.44 million to 252,446 triangles; nearest source-vertex UVs are used on
simplified parts. Its body paint is overridden blue, preserving other materials.
The Lamborghini's stray light-emitter faces outside the assembled body are
removed. See `mesh_transforms.json` for dimensions and per-part face counts.
Original GLBs are not required at runtime and were deleted after conversion.
