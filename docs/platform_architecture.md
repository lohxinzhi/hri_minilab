# Pinned simulator architecture

Inspected against upstream commit `dd40180f1121a66373d261e64a9a09eb69b1b2a7` (`third_party/quadruped_mujoco/`). This describes the official Dog example and reusable `runtime_control` package; it does not describe MiniLab code.

## 1. Command input

**Sources:** `eg/play.py` (`_find_keyboards`, `get_commands`, `key_callback`, `build_runtime_config`, main loop); `src/runtime_control/runtime.py` (`RuntimeControl.update_command`); `src/runtime_control/panel.py` (browser key endpoints); `eg/dog.yaml`.

The policy command is ordered `[vx, vy, wz]`: forward/backward, lateral left/right, and yaw left/right. `get_commands()` converts the native evdev held-key set into signed unit values. W/S set `vx` to +1/−1, A/D set `vy` to +1/−1, and Q/E set `wz` to +1/−1. Releasing the key returns that component to zero.

In browser mode, JavaScript in `panel.py` captures W/S/A/D/Q/E key-down and key-up events and sends them to `/api/key`. `RuntimeControl.update_command()` combines these with any physical pressed keys and the selected command speed values. The panel also exposes speed sliders. Its command values and height are read from the panel state each simulation iteration. The `--gui` mode uses the browser panel and does not create the native MuJoCo viewer.

R/F adjust `height_cmd` by 0.02 m, clamped to 0.20–0.35 m; T requests reset. These are native viewer callback keys. The browser exposes the equivalent command-height slider and Reset Robot action. Height is supplied separately to the observation builder, not included as a fourth motion-command element in `[vx, vy, wz]`.

`eg/dog.yaml` sets `cmd_scale: [2.0, 2.0, 0.25, 0.0]`. `build_single_obs()` multiplies the three command values elementwise by the first three scales before putting them in observation slots 0–2. Thus native unit-key values become ±2, ±2, and ±0.25; browser values are signed selected speeds times those scales. The fourth YAML scale is unused by this 3-element command assignment. Observation clipping is applied afterwards.

## 2. Observation construction

**Source:** `eg/play.py::build_single_obs`; scale and clipping values come from `eg/dog.yaml`.

`build_single_obs()` creates a float32 vector of 46 values and clips every value to `[-clip_obs, +clip_obs]`; the configured `clip_obs` is 100. The blocks are:

| Slots | Size | Meaning and configured transformation |
| --- | ---: | --- |
| 0–2 | 3 | `[vx, vy, wz] * cmd_scale[:3]`; keyboard/browser command, then global observation clipping |
| 3–5 | 3 | Body angular velocity (`qvel[3:6]`) multiplied by `ang_vel_scale = 0.25`; then clipping |
| 6–8 | 3 | World-down gravity vector rotated into body coordinates; components nominally in `[-1, 1]`; then clipping |
| 9–20 | 12 | Isaac-order joint position minus default angle, multiplied by `dof_pos_scale = 1.0`; then clipping |
| 21–32 | 12 | Isaac-order joint velocity multiplied by `dof_vel_scale = 0.05`; then clipping |
| 33–44 | 12 | Previous policy action, already clipped to `[-10, 10]` before storage |
| 45 | 1 | `(height_cmd - 0.25) / 0.1`; with configured 0.20–0.35 m command range this is `[-0.5, 1.0]` |

The source does not impose tighter physical bounds on angular velocity, joint position deviation, or joint velocity; only the configured final observation clip applies. Gravity component bounds follow from the unit gravity vector and rotation.

## 3. Observation history

**Source:** `eg/play.py::ObsHistoryBuffer`; initialization and reset in the main loop.

The example sets `NUM_ONE_STEP_OBS = 46` and `HISTORY_LEN = 6`. `ObsHistoryBuffer.push()` shifts the flat buffer by one frame and places the newest observation at the beginning. `get()` returns a copied batch with shape `(1, 276)`, because `6 × 46 = 276`. The loop pushes one new frame only at a policy/control-decimation boundary. Six initial frames are populated during warm-up. On T or browser Reset Robot, `obs_history.reset()` zeros the history along with the action state.

## 4. ONNX policy inference

**Source:** `eg/play.py` (`DEFAULT_ONNX`, argument parser, session creation and loop); model file `eg/model_3400.onnx`.

The default model path is resolved relative to `eg/play.py` as `eg/model_3400.onnx`; `--onnx` can select another file. Unless `--no-policy` is given, the script constructs `onnxruntime.InferenceSession(policy_path, providers=['CPUExecutionProvider'])`. It reads the model's input/output tensor names from the session. The verified model shapes printed by this pinned example are `['batch', 276] → ['batch', 12]`. The model receives `obs_history.get()` as the named input. `policy.run()` returns the action tensor, with the first batch row used by the loop.

Inference runs inside `if count % control_decimation == 0`, not on every physics step. With decimation 4 and timestep 0.005 s, nominal policy inference is 50 Hz. The script uses ONNX Runtime's CPU provider.

## 5. Policy action processing

**Source:** `eg/play.py` main loop; defaults in `eg/dog.yaml`.

There are 12 actions, one desired joint-position offset per actuator in the learned policy's IsaacGym order. Raw output is clipped elementwise to `[-10, 10]`. The example then computes `target_q_isaac = action_isaac * action_scale + DEFAULT_ANGLES_ISAAC`, with `action_scale = 0.25`. Thus the policy predicts scaled offsets around the learned default pose, rather than absolute joint angles. The default angles are defined in `DEFAULT_ANGLES_ISAAC` in `eg/play.py` and are also documented in `eg/dog.yaml`.

## 6. Joint-order remapping

**Source:** `eg/play.py` constants and the observation/action sections of the main loop.

MuJoCo/URDF order is FL, FR, RR, RL. IsaacGym policy order is FL, FR, RL, RR. In 12-joint index terms, the first six front-leg joints stay in place and the rear-leg groups of three swap:

```text
MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
```

`MUJOCO_TO_ISAAC` is applied to measured joint positions and velocities before the observation enters the policy. `ISAAC_TO_MUJOCO` is applied to policy-produced target angles before PD control writes actuator torques. Both arrays are the same permutation because swapping the two rear-leg groups is its own inverse. The mapping is required at both boundaries so the policy sees the trained leg ordering and MuJoCo receives targets in its actuator ordering. The startup verification also checks that mapped MuJoCo initial positions equal `DEFAULT_ANGLES_ISAAC`.

## 7. PD torque control

**Sources:** `eg/play.py` (gain/limit constants and actuator assignment); `src/runtime_control/control.py::compute_pd_torques`, `scale_torque_limits`, and `MotorCommandDelay`.

`eg/dog.yaml` configures Kp = 40 and Kd = 1.0 for each joint, in MuJoCo order. Nominal limits are 23.7 Nm for hip/thigh joints and 35.55 Nm for calf joints. A configurable motor delay can delay target application. The example scales each nominal per-joint torque limit by the runtime torque-limit slider relative to the 35.55 Nm reference, preserving hip/calf ratios.

`compute_pd_torques()` implements:

```text
tau = motor_strength * (kp * (target - position) - kd * velocity)
tau = clip(tau, -torque_limit, +torque_limit)
```

Motor strength defaults to 1.0 and is adjustable in the panel. The resulting 12 torques are assigned in MuJoCo order by `mj_data.ctrl[:NUM_ACTIONS] = tau`.

## 8. MuJoCo physics loop

**Sources:** `eg/dog.yaml`; `eg/play.py` main loop.

The configured `simulation_dt` is 0.005 s. Each `mujoco.mj_step(mj_model, mj_data)` advances one physics step, so the nominal physics rate is `1 / 0.005 = 200 Hz`. `control_decimation` is 4. Each physics step computes/applies PD torque, but the observation/policy/action target is refreshed only every fourth step. This yields nominal policy/control-target updates at `200 / 4 = 50 Hz`. The loop sleeps for the remainder of each timestep to target real-time pacing; rendering or compute overhead can reduce wall-clock rates. The browser display's FPS is separately configured and does not set physics rate.

## 9. Camera pipeline

**Sources:** `src/runtime_control/integration.py::make_standard_robot_cameras` and `standard_camera_options`; `eg/play.py::ROBOT_CAMERAS`, `CAMERA_OPTIONS`, `build_runtime_config`; `src/runtime_control/map_manager.py::compose_scene`; `src/runtime_control/runtime.py::_render_loop`; `src/runtime_control/panel.py`.

`make_standard_robot_cameras(prefix="dog")` defines `dog_front_camera`, `dog_rear_overhead_camera`, and `dog_top_camera`. `compose_scene(..., robot_body_name="trunk", robot_cameras=ROBOT_CAMERAS)` inserts them as camera children of the robot's `trunk` body, so they move with it. The names map in `standard_camera_options()` to “Front view (first person)”, “Rear overhead follow”, and “Top-down view”. “Third-person follow” is a separate tracking camera mode, not one of these three attached cameras.

The browser dropdown sends a selection to `/api/camera`; `RuntimeControlPanel` stores it, `RuntimeControl.runtime_control()` consumes it, and `_render_loop()` uses the selected camera name in `mujoco.Renderer.update_scene()`. The renderer advances a separate `MjData` from a copied simulation `qpos`, calls `mj_forward()`, renders RGB pixels, then `encode_rgb_jpeg()` converts them for `/api/stream.mjpg`.

For future programmatic RGB capture, the clean MuJoCo API is a `mujoco.Renderer(model, height, width)` owned by the MiniLab integration: copy current simulation state into a dedicated `mujoco.MjData`, call `mujoco.mj_forward(model, data)`, then `renderer.update_scene(data, camera="dog_front_camera")` and `renderer.render()`. `render()` returns an RGB NumPy array before JPEG encoding. Coordinate access with the simulation thread or use a safe state snapshot; do not read mutable `MjData` concurrently. No capture wrapper is implemented here.

## 10. Terrain/map system

**Sources:** `src/runtime_control/resources.py` (`_BUNDLED_MAPS`, `bundled_map_specs`); `eg/play.py::MAP_SPECS`, `build_runtime_config`; `src/runtime_control/map_manager.py::compose_scene`, `MapManager.initialize`, `MapManager.activate`; `src/runtime_control/runtime.py::RuntimeControl.runtime_control`; `src/runtime_control/panel.py` map endpoints.

`resources.py` registers these 11 bundled identifiers and XML resources. `eg/play.py` requests all maps with `bundled_map_specs()` and supplies browser labels and spawn poses:

| Identifier | Browser label | XML resource | Spawn position (m) |
| --- | --- | --- | --- |
| `rc26_track` | 26RC Track | `26rc_track.xml` | `[3.7, -9.0, 0.45]` |
| `race_track` | Race Track | `race_track.xml` | `[0, 0, 0.42]` |
| `stairs` | Stairs | `stairs.xml` | `[0, 0, 0.42]` |
| `cross_stairs` | Cross Stairs | `cross_stairs.xml` | `[0, 0, 0.42]` |
| `cross_slope` | Cross Slope | `cross_slope.xml` | `[0, 0, 0.42]` |
| `google_barkour` | Google Barkour | `google_barkour.xml` | `[1.0, -1.5, 0.42]` |
| `gap_jump` | Gap Jump | `gap_jump.xml` | `[0, 0, 0.42]` |
| `hurdles` | Hurdles | `hurdles.xml` | `[0, 0, 0.42]` |
| `suspended_steps` | Suspended Steps | `suspended_steps.xml` | `[0, 0, 0.42]` |
| `perlin_rough` | Perlin Rough Terrain | `perlin_rough.xml` | `[0, 0, 0.42]` |
| `dynamic_obstacles` | Dynamic Obstacles | `dynamic_obstacles.xml` | `[0, 0, 0.42]` |

Spawn quaternions are `[1, 0, 0, 0]` (MuJoCo `wxyz`) except `rc26_track`, which uses `[-0.7071068, 0, 0, 0.7071068]`. Resource paths are relative to `src/runtime_control/maps/`.

`RuntimeScene.open()` composes the robot and all map terrain resources into one MJCF before loading `MjModel`. `compose_scene()` prefixes map geom names. `MapManager.initialize()` groups the already-compiled geoms by those names. On selection, `MapManager.activate()` restores the selected map's saved geom properties and enables its collision masks, while disabling collisions and hiding inactive maps. It does not recompile `MjModel` when switching.

The browser dropdown submits the internal map identifier to `/api/map`. `RuntimeControl.runtime_control()` consumes the pending selection, calls `MapManager.activate()`, updates the configured initial position/quaternion from that map's `map_spawns` entry, and requests a reset. The host loop then calls `reset_robot()`, applying root XYZ, MuJoCo `wxyz` quaternion, default joint angles, zero velocity, and `mj_forward()`. Dynamic Obstacles additionally has a map randomizer when activated.

## Control pipeline

```text
Native evdev or browser key + speed controls; separate height command
                         ↓
                  vx, vy, wz, height
                         ↓
               46-D observation frame
                         ↓
                six-frame history
                         ↓
       ONNX policy: 276 inputs → 12 actions
                         ↓
       clip, scale, add Isaac-order defaults
                         ↓
             remap targets to MuJoCo order
                         ↓
       delayed, strength-scaled, clipped PD torque
                         ↓
              MuJoCo actuator controls
                         ↓
                 MuJoCo physics step
                         ↓
                  quadruped motion
```
