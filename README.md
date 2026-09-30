# NUS EE5112 MiniLab 1.3

Assignment: **Command Your Robot Dog: An LLM + YOLO Powered Quadruped.**

**Status: Work in progress — repository scaffolding only.** The simulator has not been imported or tested, and no MiniLab functionality has been implemented.

Chosen platform: the official [quadruped_mujoco](https://github.com/aoqianz/quadruped_mujoco), to be fetched locally at a pinned commit under `third_party/quadruped_mujoco/`.

## Directory layout

- `third_party/`: tracked provenance and a future Git-ignored local simulator checkout; upstream code is not redistributed.
- `minilab/`: future platform integration, motion, perception, dialogue, and evaluation code.
- `scenes/`: custom scene configuration and object assets.
- `prompts/`: versioned LLM prompts and schemas.
- `evaluation/`: test cases, evaluation-only ground truth, and selected results. Ground-truth positions must never be used for steering.
- `report_assets/`: selected figures and screenshots.
- `reproducibility/`: future environment specifications and installed-version records.
- `patches/`: explicit patches if upstream modifications later become unavoidable.
- `scripts/`: future setup, launch, verification, and submission-packaging scripts.
- `docs/`: setup checklist and future verified platform documentation.
- `tests/`: future tests of the MiniLab implementation.
- `outputs/`: ignored temporary experimental output and generated media.
