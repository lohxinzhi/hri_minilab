# NUS EE5112 MiniLab 1.3 — Project handover

This is the authoritative project-status and technical handover for the next teammate and any fresh Codex session. It was prepared from the current source tree, Git state, existing evidence files, and a fresh Conda/test verification on 2026-10-01 (Asia/Singapore). Motion and parser work is committed on the handover branch but remains explicitly incomplete as described below.

## 1. Project identity

- Course: NUS EE5112 Human Robot Interaction
- MiniLab: 1.3
- Title: **Command Your Robot Dog: An LLM + YOLO Powered Quadruped**
- Local repository: `~/hri_minilab` (`/home/briansyc/hri_minilab`)

Intended behavior:

```text
natural-language command
        ↓
structured target command
        ↓
onboard dog_front_camera
        ↓
YOLO + colour grounding
        ↓
visual search
        ↓
target centering
        ↓
approach
        ↓
safe stop
```

Current implementation state:

- **Structured target command:** deterministic parser and strict provider-neutral LLM adapter are committed WIP; no behavior manager consumes them yet.
- **Front camera:** implemented and committed.
- **YOLO and chair-colour grounding:** implemented and committed; smoke-tested on saved and live front-camera images.
- **Visual search, bbox centering, object approach, end-to-end safe stop:** not implemented.
- **Velocity/turn skills:** implemented and numerically tested in a clearly labelled WIP commit; manual native-viewer visual stability validation is still pending.
- **Live LLM service, dialogue loop, Task 3/Task 4 integration:** not implemented.

## 2. Platform and environment

Freshly verified in this handover:

| Item | Verified value |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS |
| Conda environment | `quadruped_mujoco` |
| Python | 3.11.16 |
| Python executable | `/home/briansyc/miniconda3/envs/quadruped_mujoco/bin/python` |
| MuJoCo | 3.14.0 |
| Ultralytics | 8.4.168 |
| trimesh | 5.1.0 |
| ONNX Runtime | 1.30.0 |
| NumPy | 2.4.6 |
| PyYAML | 6.0.3 |
| Pillow | 12.3.0 |
| OpenAI Python package | 3.22.1 (installed; no API service connected) |

Always start project work with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quadruped_mujoco
cd ~/hri_minilab
```

Do not use system `python3` for this project unless deliberately intended; it does not have the project dependencies.

No environment lockfile or reproducibility specification is currently present under `reproducibility/`; the versions above are a fresh local inventory, not a fully resolved environment export.

The simulator is the official `aoqianz/quadruped_mujoco`, locally checked out at `third_party/quadruped_mujoco/`, pinned to `dd40180f1121a66373d261e64a9a09eb69b1b2a7`. Current read-only Git verification reports that checkout clean at that SHA. See `third_party/README.md` for provenance and redistribution caveat. Do not casually modify the upstream checkout. Project-owned functionality belongs outside `third_party/`.

## 3. Verified upstream architecture

The source was inspected at the pinned commit; detailed source locations and observations are in `docs/platform_architecture.md`. Key configuration comes from `third_party/quadruped_mujoco/eg/dog.yaml` and `eg/play.py`:

- MuJoCo timestep: 0.005 s, nominal 200 Hz physics.
- Control decimation: 4 physics steps per policy update, nominal 50 Hz policy.
- Single observation: 46 float values; six-frame history gives 276 policy inputs.
- ONNX model: `eg/model_3400.onnx`; output: 12 actions.
- Action scale: 0.25 around learned default joint angles; raw action is clipped before target construction.
- IsaacGym order is FL, FR, RL, RR; MuJoCo order is FL, FR, RR, RL. The rear 3-joint groups are permuted when observations enter and actions leave the policy.
- Observation blocks are command references, scaled body angular velocity, projected gravity, joint position offsets, joint velocities, last action, and normalized body-height command. `docs/platform_architecture.md` records exact slots/scales.
- The upstream example runs policy inference at each decimation boundary, then uses runtime-control PD torque helpers, torque limits, configurable motor-command delay, and MuJoCo actuator controls before each `mujoco.mj_step()`.

`minilab/platform/dog_policy_runner.py` mirrors the minimum needed policy loop while using public `runtime_control` interfaces. It asserts the pinned configuration and preserves the learned ONNX/PD chain; it does not directly command joints for high-level motion. It is committed as WIP on `handover/minilab-1.3`, not a declaration of visual approval.

## 4. Platform verification already completed

These platform checks were manually confirmed in earlier sessions and recorded in `docs/setup_checklist.md`; this documentation-preparation session did not reopen a GUI:

- headless MuJoCo simulation and ONNX policy load;
- native MuJoCo viewer, stable robot, and keyboard locomotion (W/S, A/D, Q/E, R/F, T reset);
- browser GUI at `http://127.0.0.1:8765`, movement, E-Stop, reset, map and camera selectors;
- Race Track, Stairs, and Perlin Rough Terrain map load/spawn/stability/movement/turn/reset;
- all three robot-mounted camera views.

Camera names and browser labels from `runtime_control.integration`:

| Camera name | Browser label | Purpose |
| --- | --- | --- |
| `dog_front_camera` | Front view (first person) | Required perception source |
| `dog_rear_overhead_camera` | Rear overhead follow | Robot-mounted rear/overhead view |
| `dog_top_camera` | Top-down view | Robot-mounted overhead view |

The external tracking view is not an onboard camera and is not the Task 4 image source.

## 5. Front-camera pipeline

Committed in `6a41a1b` (`Implement Task 2 front camera pipeline`). Project files: `minilab/platform/camera.py` and `scripts/test_front_camera.py`.

`FrontCamera` explicitly requires `dog_front_camera`, rejects any other camera name, verifies that camera exists in the MuJoCo model, and has no tracking-camera fallback. It uses MuJoCo `Renderer.update_scene(..., camera="dog_front_camera")` and `Renderer.render()`, returning RGB `uint8` with shape `(height, width, 3)`. The smoke script used 640×480 and configurable render rate; configured/tested rate is 15 Hz simulation time, with the recorded measurement approximately 14.29 Hz. The camera API throttles captures against simulation time and the returned frame is copied by default for safety.

The frame callback added to `scripts/play_yolo_search_scene.py` also passes only the `FrontCamera` frame from `dog_front_camera` into the live perception smoke test.

## 6. Final perception scene

The active scene is `scenes/configs/yolo_search_scene.xml`, with launch metadata in `scenes/configs/yolo_search_scene.yaml`:

| Instance | COCO class | Appearance | Evaluation position (world m) |
| --- | --- | --- | --- |
| `chair_green_01` | `chair` | saturated green | `[2.5, 1.2, 0.4399206]` |
| `chair_red_01` | `chair` | saturated red | `[3.5, -1.2, 0.4399206]` |
| `ball_01` | `sports ball` | natural textured football | `[4.0, 0.1, 0.25]` |

The chairs share the same `scenes/assets/objects/chair.obj`; material colour is the controlled difference. The football's OBJ UV coordinates and diffuse panel texture are retained in `football.obj`, `football.mtl`, and `football_diffuse.png`. Asset provenance, creators, CC0 notes, conversion steps, dimensions, and source URLs are in `scenes/assets/licenses/assets.md`. An inactive `couch.obj` and its provenance remain in the repository; it is not an active scene object.

`evaluation/ground_truth/yolo_search_scene.yaml` is explicitly marked `usage: evaluation_only`. It is only for later evaluation-distance calculation. Perception, navigation, and steering code must never import/read these coordinates or use them to locate targets.

## 7. YOLO asset validation

`docs/yolo_smoke_test.md` contains the committed smoke-test record for pretrained COCO YOLO11n (Ultralytics 8.4.168, default confidence threshold 0.25):

| View | Green chair confidence | Red chair confidence | Sports-ball confidence |
| --- | ---: | ---: | ---: |
| Center | 0.8529 | 0.8710 | 0.7003 |
| Offset | 0.8935 | 0.8186 | 0.8081 |
| Closer (~2.9 m) | outside frame | 0.8312 | 0.8575 |

These are smoke-test observations, not a complete evaluation dataset or a guarantee of detection in other conditions. The rejected couch experiment produced `chair` instead of `couch`; the plant experiment produced `cup` instead of `potted plant`. Those experiments were explicitly stopped, and neither object is active. The accepted football was classified as `sports ball` in all three tested viewpoints.

## 8. Reusable perception pipeline

Committed in `cd3c120` (`Implement reusable YOLO and colour perception`). Files: `minilab/perception/detector.py`, `color.py`, `pipeline.py`, `__init__.py`; tests/scripts: `tests/test_perception.py`, `scripts/test_perception_saved.py`, `scripts/test_perception_live.py`; behavior notes: `docs/perception.md`.

The project `Detection` record contains class ID/name, confidence, `bbox_xyxy`, and derived bbox center, width, height, and area. The pipeline retains all detector results above confidence 0.25 and adds HSV colour only for `chair` detections. It expects RGB `uint8` `(H,W,3)` frames. HSV uses OpenCV `COLOR_RGB2HSV`; red handles hue wrap with ranges 0–12 and 168–179, green uses 35–90, and saturation/value below 70/45 are excluded. It analyzes the centered inner 70% × 72% bbox region to limit background contamination. Weak/ambiguous evidence returns `unknown`.

The accepted live headless smoke test used the project scene launcher and only `dog_front_camera`: 28 frames processed; green chair 28/28, red chair 28/28, sports ball 28/28. Measured mean YOLO inference was approximately 0.118 s including the first cold call and 0.023 s for warmed calls. This smoke test did not move the robot and is not the final Task 4 evaluation.

## 9. Motion skills — implemented, numerically tested, visual validation pending

**Status: IMPLEMENTED / NUMERICALLY TESTED / MANUAL VISUAL STABILITY VALIDATION PENDING.** The following files are committed on `handover/minilab-1.3` in `f06f158dfe36278ad4647f0bc58c2de17fb612d5` (`WIP: add motion skills pending visual validation`):

- `minilab/platform/dog_policy_runner.py`
- `minilab/motion/skills.py`
- `scripts/test_motion_skills.py`
- `scripts/evaluate_turns.py`
- `scenes/configs/motion_test_map.xml`
- `tests/test_motion_skills.py`
- `evaluation/results/motion_smoke.csv`
- `evaluation/results/turn_comparison.csv`

`DogPolicyRunner.step(command)` executes one physics step and refreshes the policy every four steps. It keeps the ONNX policy, 46-D observation, six-frame history, 12 outputs, scale/remap, PD, motor delay, and `mj_step()` path. `move(vx, vy, wz, duration)` uses MuJoCo simulation time and sends neutral command after the requested interval. `turn(angle_deg)` reads actual yaw from MuJoCo `qpos[3:7]`, accumulates wrapped increments (`delta = wrap_to_pi(yaw_now - yaw_previous)`), and uses proportional yaw command feedback; it does not estimate angle from elapsed time.

Current `TurnConfig` values in source:

- proportional gain: 5.0 normalized yaw command per radian of error;
- yaw command limit: ±1.0 (runner command range is ±1 for vx/vy/wz);
- requested tolerance: 5 degrees; entry settling band: 4.5 degrees;
- yaw-rate threshold: 0.10 rad/s;
- stable samples: 3 consecutive policy updates (50 Hz boundaries);
- safety timeout: 18 simulated seconds through ±90°, scaled linearly to 36 seconds at ±180°;
- neutral settling: 0.10 simulated seconds.

Current CSV statistics (three trials per requested angle; numerical simulation evidence only):

| Requested | Open-loop mean absolute error | Closed-loop mean absolute error | Closed-loop successes |
| ---: | ---: | ---: | ---: |
| +45° | 22.633° | 4.576° | 3/3 |
| −45° | 31.313° | 4.547° | 3/3 |
| +90° | 44.959° | 4.521° | 3/3 |
| −90° | 56.516° | 4.534° | 3/3 |
| +180° | 88.425° | 4.492° | 3/3 |
| −180° | 113.753° | 4.548° | 3/3 |

Overall mean absolute error is 59.599872° for open loop (0/18 within tolerance) and 4.536379° for closed loop (18/18 numerical successes). The four move trials show final base-height changes from −0.167894 m to −0.077791 m. This unexplained low posture is why numerical results are not enough to approve motion. The next teammate must visually run forward, backward, strafe-left, strafe-right, turn+90, turn−90, turn+180, and turn−180 before calling motion complete. `scripts/test_motion_skills.py` has diagnostic aborts at base height <0.15 m, roll/pitch >45°, or non-finite state; those are safety checks, not proof of stability.

Exact CLI choices in the script are `--viewer {headless,native,browser}` and `--motion {forward,backward,strafe-left,strafe-right,turn+90,turn-90,turn+180,turn-180}`. Use a separate fresh invocation for each motion.

## 10. Natural-language command parser — implemented, unit-tested, provider/integration pending

**Status: IMPLEMENTED / UNIT TESTED / LIVE LLM PROVIDER NOT CONNECTED / NOT CONNECTED TO ROBOT BEHAVIOR.** Files `minilab/dialogue/parser.py`, `minilab/dialogue/llm_parser.py`, `minilab/dialogue/__init__.py`, `scripts/test_command_parser.py`, and `tests/test_dialogue_parser.py` are committed on `handover/minilab-1.3` in `d3e6373874f0dec9cfbe322203679738fa67e933` (`WIP: add validated target command parser`).

The deterministic parser recognizes actions `find` and `goto_object`. Current phrases include `find`, `locate`, `look for`, `go to`, `go over to`, `approach`, `move toward(s)`, and `head to`. `football`, `soccer ball`, and `ball` normalize to `sports ball`. Supported targets are chair (red/green only) and sports ball (no colour). No `seat` alias is implemented. An uncoloured chair is `clarification_required / ambiguous_target`; unsupported object/color is rejected; empty input is malformed; unrecognized text is unintelligible.

`TargetCommand` has exactly `status`, `action`, `object_class`, `color`, and `reason`. A central validator rejects missing/extra fields, enforces enum and class/colour combinations, and turns accepted but uncoloured chair commands into clarification. The current parser handles one target command at a time; multi-step parsing is not implemented. The provider-neutral `LLMTargetParser` accepts an injected text-generation callable, provides a strict JSON schema with `additionalProperties: false`, parses JSON locally, retries malformed/schema-invalid output once, then returns a safe rejected result. It never passes free-form model text to robot code. No live external LLM provider/API has been connected. The assignment's future comparison of at least two LLM providers on at least 20 utterances remains outstanding. The complete suite was rerun after both WIP commits and passed 25/25.

## 11. Git state: committed vs uncommitted

Current branch is `handover/minilab-1.3`; implementation HEAD before the handover documentation commit is `d3e6373874f0dec9cfbe322203679738fa67e933`. The motion WIP commit is `f06f158dfe36278ad4647f0bc58c2de17fb612d5`; the parser WIP commit is `d3e6373874f0dec9cfbe322203679738fa67e933`. The preceding perception milestone is `cd3c120` (`Implement reusable YOLO and colour perception`). The root has an `origin` URL configured, but its local `origin/main` tracking ref is absent; do not infer that the remote branch itself was deleted.

| Component | Git state | Verification state |
| --- | --- | --- |
| Platform setup/provenance and architecture docs | Committed (`59fcf4c`, `dbe63fe`, related setup commits) | Headless test and manual viewer/browser/maps/camera checks recorded as passed |
| Front camera | Committed (`6a41a1b`) | User inspected valid RGB frame; 640×480 smoke test recorded |
| Custom object scene/assets | Committed (`b1dfc3c`, `df6b075`) | User visually approved scene; YOLO11n smoke test recorded |
| Reusable perception | Committed (`cd3c120`) | Saved/live smoke results accepted; tests pass |
| Motion skills and CSVs | Committed WIP on handover branch (`f06f158`) | Numerical unit/simulation tests pass; **manual visual validation pending** |
| Command parser | Committed WIP on handover branch (`d3e6373`) | Deterministic/mock LLM unit tests pass; live provider and behavior integration absent |
| Handover documentation | Added to this handover branch in the documentation checkpoint | Documentation only; no new functionality |

The four handover files in this table are documentation-only additions to the branch; the implementation files remain in their separately labelled WIP commits.

## 12. How to run the project today

Start every new terminal with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quadruped_mujoco
cd ~/hri_minilab
```

Do not use system `python3` for project commands. Commands supported by current scripts/source:

```bash
# Original upstream native MuJoCo viewer
cd third_party/quadruped_mujoco
python eg/play.py

# Original upstream browser GUI (open http://127.0.0.1:8765)
python eg/play.py --gui

# Re-run saved-image reusable perception on existing local v3 frames
cd ~/hri_minilab
python scripts/test_perception_saved.py

# Short headless live perception on dog_front_camera
python scripts/test_perception_live.py --duration 6

# Deterministic command parser; no simulator/model required
python scripts/test_command_parser.py "Go to the red chair"

# Run exactly one motion per native-viewer trial; manual sign-off still required
python scripts/test_motion_skills.py --viewer native --motion forward
python scripts/test_motion_skills.py --viewer native --motion backward
python scripts/test_motion_skills.py --viewer native --motion strafe-left
python scripts/test_motion_skills.py --viewer native --motion strafe-right
python scripts/test_motion_skills.py --viewer native --motion turn+90
python scripts/test_motion_skills.py --viewer native --motion turn-90
python scripts/test_motion_skills.py --viewer native --motion turn+180
python scripts/test_motion_skills.py --viewer native --motion turn-180

# Full headless unit suite
python -m unittest discover -s tests
```

The motion viewer commands are present in the committed WIP script; they have not been manually verified. The current saved-image script expects `outputs/yolo_smoke_test_v3/{center,offset,near}.png`, which are local ignored artifacts and are present in the inspected workspace. The live test may be slower while YOLO warms up.

## 13. Generated/local files

The project `.gitignore` excludes `outputs/`, the specifically named root `yolo11n.pt`, caches/build products, Conda/virtual environment folders, credentials, and generated video extensions. Ultralytics obtains `yolo11n.pt` when needed and uses its normal local/cache behavior; do not commit weights. The Conda environment is outside the project repository. Keep selected report evidence intentionally if it becomes a required submission artifact, while remembering that required final `Video_Task2`, `Video_Task3`, and `Video_Task4` must be explicitly included in the Canvas ZIP even though generated videos are ignored by Git.

## 14. Remaining work in dependency order

1. Manually visually validate the eight motion cases listed above before declaring motion complete.
2. Fix motion only if those observations show instability or incorrect directions; rerun relevant numerical tests and record genuine evidence.
3. After visual review, update motion status accurately; its current commit remains explicitly WIP.
4. Implement target matching using detected class plus chair colour; do not consult ground truth.
5. Implement search rotation when the target is absent from the camera view.
6. Implement visual bbox horizontal centering.
7. Implement approach control and conservative stopping distance/safe stop.
8. Connect validated `TargetCommand` objects to a behavior manager; do not let parser text invoke skills directly.
9. Connect and compare at least two real LLM providers on at least 20 utterances after deterministic behavior/validation is stable; preserve deterministic fallback.
10. Run systematic Task 3/4 evaluation and save genuine evidence.
11. Record the required Task 2/3/4 demonstration videos.
12. Prepare and inspect a reproducible final report and Canvas ZIP with required videos explicitly included.

Search/centering/approach depend on the committed camera/perception APIs. Behavior integration depends on a validated parser and tested motion skills. Live LLM integration should follow, not precede, the validation boundary. Final evaluation/video/report depend on the end-to-end behavior being verified.

## 15. Non-negotiable design constraints

1. Do not casually modify the pinned upstream simulator.
2. Use `dog_front_camera` as the Task 4 perception image source.
3. Never use evaluation ground-truth positions for steering or search.
4. Free-form LLM text must never directly control the robot.
5. Validate all structured LLM output locally and reject additional fields.
6. Preserve the pretrained ONNX locomotion policy.
7. Do not directly command joints to implement high-level motion.
8. Generated weights/output images are not source code.
9. Distinguish user-confirmed/manual evidence, measured data, and assumptions.
10. Do not mark motion complete until the manual visual tests pass.

## 16. Useful project history

This history is included to avoid repeating experiments; some failed experiment frames are not retained as committed evidence:

- Poly Haven's automated file API returned HTTP 403; the approved CC0 assets were downloaded manually from official pages.
- 4K Blender packages were initially downloaded by mistake; the working asset conversion used the correct 1K glTF packages.
- Couch was detected as `chair`, not `couch`; potted plant was detected as `cup`, not `potted plant`. Those failed candidates were removed from the active scene.
- Football passed `sports ball` detection in center, offset, and closer dog-front-camera views.
- Native W/S/A/D/Q/E initially controlled the MuJoCo viewport because evdev could not read the physical keyboard. Adding the user to Linux `input` and logging out/in resolved the permission problem; keyboard controls were subsequently manually verified.

The root `README.md` still contains its original scaffold-only status text. The small handover pointer added there identifies this document as authoritative for the current status.
