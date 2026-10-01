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
