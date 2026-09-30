# Setup checklist

```text
Machine inspected        [x]
Miniconda installed      [x]
Repository structured    [x]
Upstream source pinned   [x]
Conda environment ready  [x]
Dependencies installed   [x]
Headless simulation      [x]
Native viewer            [x]
Keyboard controls        [x]
Browser GUI              [x]
Three terrain maps       [x]
Three cameras            [x]
Architecture inspected   [ ]
```

Repository scaffolding has been approved. The `quadruped_mujoco` Conda environment was created and verified after activation: Python 3.11.16 at `/home/briansyc/miniconda3/envs/quadruped_mujoco/bin/python`; `conda env list` confirmed it was active. The upstream editable package and required assignment dependencies have been installed and verified. Visual and manual tests remain pending until explicitly confirmed by the user.

## Required Canvas submission videos

The final Canvas ZIP must contain all three demonstration videos, with their actual video extensions:

- `Video_Task2`
- `Video_Task3`
- `Video_Task4`

Generated videos may remain untracked by Git. The future packaging procedure must explicitly include these final videos even when ignored by Git, and fail if any is missing. Verify all three are present in the extracted ZIP. A Git-only archive is insufficient when the videos are untracked.

Selected evaluation CSVs, plots, screenshots, prompts, scene configurations, and required project-owned source assets must also be retained in the submission. No packaging procedure or demonstration video has been created yet.

Empty scaffold directories currently exist only on disk: Git does not track empty directories. No placeholder modules or files have been added.

## Platform installation and headless verification

Verified on 2026-09-30 using Python 3.11.16 in `quadruped_mujoco`.

- Simulator commit: `dd40180f1121a66373d261e64a9a09eb69b1b2a7` (see `third_party/README.md`).
- Editable install: `python -m pip install -e third_party/quadruped_mujoco` from the project root.
- Additional missing packages installed: `onnxruntime pyyaml ultralytics openai`. No upgrade flag was used.
- All eight required imports passed; `python -m pip check` reported no broken requirements.

| Import | Distribution | Installed version |
| --- | --- | --- |
| `mujoco` | mujoco | 3.14.0 |
| `numpy` | numpy | 2.4.6 |
| `onnxruntime` | onnxruntime | 1.30.0 |
| `yaml` | PyYAML | 6.0.3 |
| `PIL` | Pillow | 12.3.0 |
| `ultralytics` | ultralytics | 8.4.168 |
| `openai` | openai | 3.22.1 |
| `runtime_control` | mujoco-runtime-control (editable) | 0.6.0 |

From the upstream directory, `python eg/play.py --headless --duration 1` exited with code 0. MuJoCo loaded the robot, joint remapping verification matched, the pretrained ONNX policy reported input/output shapes `['batch', 276]` and `['batch', 12]`, and the run ended with `[INFO] simulation finished`. No fatal exception occurred.

A supplementary in-memory run of the unchanged entry point confirmed 197 physics steps, 50 policy inferences, and 0.985 seconds of simulated time. Upstream's duration option limits wall-clock runtime; it does not guarantee exactly one simulated second.

At the time of the headless test, optional `evdev` was absent. It was subsequently installed (version 2.0.0). After the user changed input-group membership and logged out/in, read-only checks confirmed active `input` membership and successful access to both detected keyboards.

The user explicitly confirmed the native viewer launches, the robot is visible and stable, W/S forward/backward, A/D strafe, Q/E turning, R/F body-height control, and T reset all work. Native viewer and keyboard controls are therefore marked complete. The user also confirmed the browser server starts, the page opens at http://127.0.0.1:8765, the live simulation is visible and stable, movement/E-Stop/Reset Robot work, and Map/Camera selectors are visible. Browser GUI verification is complete. The user has also manually verified all three listed terrain maps and all three robot-mounted camera views; results are recorded below. Architecture inspection remains pending.

## Verified terrain-map and onboard-camera tests

Source inspected at pinned commit `dd40180f1121a66373d261e64a9a09eb69b1b2a7`. Selection method: browser Map dropdown.

| Test | Result | Notes |
| --- | --- | --- |
| Race Track | PASS | baseline terrain |
| Stairs | PASS | stepped terrain |
| Perlin Rough Terrain | PASS | uneven heightfield |
| Front view | PASS | robot-mounted first-person camera |
| Rear overhead | PASS | robot-mounted rear/overhead camera |
| Top-down | PASS | robot-mounted overhead camera |

For all three maps, the user confirmed correct map loading and spawn, stable robot, forward movement, turning, and reset. No unreported measurements are included.

Camera definitions: `src/runtime_control/integration.py` (`make_standard_robot_cameras`, `standard_camera_options`); `eg/play.py` uses prefix `dog`; `compose_scene` in `map_manager.py` attaches them beneath `trunk`.

- [x] Camera 1 view works: `dog_front_camera` — Front view (first person)
- [x] Camera 2 view works: `dog_rear_overhead_camera` — Rear overhead follow
- [x] Camera 3 view works: `dog_top_camera` — Top-down view
- [x] Front camera identified: user confirmed `dog_front_camera` is the first-person camera for future Task 4 perception
- [x] All three camera views move with the robot, as confirmed by the user

Terrain and camera milestone boxes are complete based on explicit user confirmation. No camera wrapper or programmatic frame capture has been created.
