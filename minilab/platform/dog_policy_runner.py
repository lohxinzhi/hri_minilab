"""Project-owned programmable runner for the pinned quadruped policy.

Mirrors upstream commit dd40180f1121a66373d261e64a9a09eb69b1b2a7,
principally ``eg/play.py``'s ``build_single_obs`` and control loop and
``eg/dog.yaml``. Scene composition and runtime state use the public
``runtime_control.RuntimeScene``, ``RobotAdapter``, PD and delay helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import monotonic, sleep

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "quadruped_mujoco"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

from runtime_control import (  # noqa: E402
    MapSpec,
    RobotAdapter,
    RuntimeScene,
    MotorCommandDelay,
    compute_pd_torques,
    make_runtime_config,
    make_standard_robot_cameras,
    scale_torque_limits,
    setup_tracking_camera,
)


MUJOCO_TO_ISAAC = np.asarray([0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8])
ISAAC_TO_MUJOCO = np.asarray([0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8])
TORQUE_LIMITS_MUJOCO = np.tile([23.7, 23.7, 35.55], 4).astype(np.float64)
TAU_REFERENCE = 35.55
OBS_DIM = 46
HISTORY = 6
ACTION_DIM = 12
COMMAND_LIMITS = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)


def _quat_rotate_inverse_xyzw(q, vector):
    q = np.asarray(q, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    q_vec = q[:3]
    return (
        vector * (2.0 * q[3] ** 2 - 1.0)
        + np.cross(q_vec, vector) * q[3] * 2.0
        + q_vec * np.dot(q_vec, vector) * 2.0
    )


class DogAdapter(RobotAdapter):
    """Dog policy observation/action adapter implementing the public contract."""

    root_body_name = "trunk"
    expected_dimensions = (19, 18, 12)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.default_mujoco = np.asarray(config["default_angles"], dtype=np.float64)
        self.default_isaac = self.default_mujoco[MUJOCO_TO_ISAAC]
        self.kp = np.asarray(config["kps"], dtype=np.float64)
        self.kd = np.asarray(config["kds"], dtype=np.float64)
        self.history = np.zeros((HISTORY, OBS_DIM), dtype=np.float32)
        self.action_isaac = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_action_isaac = np.zeros(ACTION_DIM, dtype=np.float32)
        self.target_mujoco = self.default_mujoco.copy()
        self.motor_delay = None
        self.qpos_adr = np.arange(7, 19, dtype=np.int32)
        self.qvel_adr = np.arange(6, 18, dtype=np.int32)

    def bind(self, model):
        if model.nq != 19 or model.nv != 18 or model.nu != ACTION_DIM:
            raise ValueError(
                f"dog dimensions mismatch: nq/nv/nu={model.nq}/{model.nv}/{model.nu}"
            )
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        actuator_names = [
            "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh",
            "FR_calf", "RR_hip", "RR_thigh", "RR_calf", "RL_hip",
            "RL_thigh", "RL_calf",
        ]
        joint_ids, actuator_ids = [], []
        for joint_name, actuator_name in zip(joint_names, actuator_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if jid < 0 or aid < 0:
                raise ValueError(f"missing dog joint/actuator: {joint_name}/{actuator_name}")
            joint_ids.append(jid)
            actuator_ids.append(aid)
        self.qpos_adr = model.jnt_qposadr[joint_ids].astype(np.int32)
        self.qvel_adr = model.jnt_dofadr[joint_ids].astype(np.int32)
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
        if not np.array_equal(self._actuator_ids, np.arange(ACTION_DIM)):
            raise ValueError("unexpected actuator ordering; refusing silent remap")
        self.motor_delay = MotorCommandDelay(model.opt.timestep)

    def reset(self, model, data, runtime_config):
        self._require_bound(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = runtime_config["simulation"]["initial_position"]
        data.qpos[3:7] = runtime_config["simulation"]["initial_quaternion"]
        data.qpos[self.qpos_adr] = self.default_mujoco
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.history.fill(0.0)
        self.action_isaac.fill(0.0)
        self.last_action_isaac.fill(0.0)
        self.target_mujoco[:] = self.default_mujoco
        self.motor_delay.reset()
        self.last_control = None
        mujoco.mj_forward(model, data)

    def build_observation(self, model, data, command, *, height_cmd=0.25):
        command = np.asarray(command, dtype=np.float32)
        config = self.config
        q_wxyz = data.qpos[3:7]
        q_xyzw = np.asarray([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        omega = data.qvel[3:6].astype(np.float64)
        q_isaac = data.qpos[self.qpos_adr].astype(np.float64)[MUJOCO_TO_ISAAC]
        dq_isaac = data.qvel[self.qvel_adr].astype(np.float64)[MUJOCO_TO_ISAAC]
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[:3] = command * np.asarray(config["cmd_scale"][:3], dtype=np.float32)
        obs[3:6] = omega.astype(np.float32) * float(config["ang_vel_scale"])
        gravity = _quat_rotate_inverse_xyzw(q_xyzw, [0.0, 0.0, -1.0])
        obs[6:9] = gravity.astype(np.float32)
        obs[9:21] = ((q_isaac - self.default_isaac) * float(config["dof_pos_scale"])).astype(np.float32)
        obs[21:33] = (dq_isaac * float(config["dof_vel_scale"])).astype(np.float32)
        obs[33:45] = self.last_action_isaac
        obs[45] = np.float32((float(height_cmd) - 0.25) / 0.1)
        return np.clip(obs, -float(config["clip_obs"]), float(config["clip_obs"]))

    def push_observation(self, observation):
        self.history[1:] = self.history[:-1]
        self.history[0] = observation
        packed = self.history.reshape(-1).copy()
        if packed.shape != (276,):
            raise AssertionError(f"policy history has shape {packed.shape}, expected (276,)")
        return packed

    def compute_control(self, model, data, action, runtime_state):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy action must have shape (12,), got {action.shape}")
        self.last_action_isaac[:] = action.astype(np.float32)
        delayed_isaac = self.motor_delay.apply(action, runtime_state["motor_delay_ms"])
        target_isaac = self.default_isaac + float(self.config["action_scale"]) * delayed_isaac
        target_mujoco = target_isaac[ISAAC_TO_MUJOCO]
        q = data.qpos[self.qpos_adr]
        dq = data.qvel[self.qvel_adr]
        torque_limits = scale_torque_limits(
            TORQUE_LIMITS_MUJOCO,
            runtime_state["torque_limit"],
            reference_limit=TAU_REFERENCE,
        )
        return compute_pd_torques(
            target_mujoco,
            q,
            dq,
            self.kp * (runtime_state["kp"] / 40.0),
            self.kd * runtime_state["kd"],
            motor_strength=runtime_state["motor_strength"],
            torque_limit=torque_limits,
        )


class DogPolicyRunner:
    """One-step-at-a-time normal MuJoCo/ONNX locomotion runner."""

    def __init__(self, *, initial_position=(0.0, 0.0, 0.42),
                 initial_quaternion=(1.0, 0.0, 0.0, 0.0),
                 viewer_mode="headless"):
        if viewer_mode not in {"headless", "native", "browser"}:
            raise ValueError("viewer_mode must be headless, native, or browser")
        self.viewer_mode = viewer_mode
        self.viewer = None
        self._viewer_context = None
        self._wall_deadline = None
        self.step_observer = None
        self.config_path = UPSTREAM_ROOT / "eg" / "dog.yaml"
        self.robot_xml = UPSTREAM_ROOT / "eg" / "dog" / "xml" / "dog_terrain.xml"
        self.policy_path = UPSTREAM_ROOT / "eg" / "model_3400.onnx"
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        if (self.config["simulation_dt"], self.config["control_decimation"],
                self.config["num_one_step_obs"], self.config["num_obs"],
                self.config["num_actions"], self.config["action_scale"]) != (0.005, 4, 46, 276, 12, 0.25):
            raise ValueError("upstream dog.yaml differs from the verified policy configuration")
        if not self.policy_path.is_file():
            raise FileNotFoundError(self.policy_path)
        self.initial_position = tuple(float(v) for v in initial_position)
        self.initial_quaternion = tuple(float(v) for v in initial_quaternion)
        self.adapter = DogAdapter(self.config)
        map_path = PROJECT_ROOT / "scenes" / "configs" / "motion_test_map.xml"
        map_spec = MapSpec(map_path)
        kp = float(np.mean(self.adapter.kp))
        kd = float(np.mean(self.adapter.kd))
        runtime_config = make_runtime_config(
            gui=viewer_mode == "browser",
            title="MiniLab motion runner",
            maps={"motion_test": "Motion Test Flat"},
            map_spawns={"motion_test": {"position": self.initial_position,
                                        "quaternion": self.initial_quaternion}},
            kp=kp,
            kd=kd,
            torque_limit=TAU_REFERENCE,
            initial_position=self.initial_position,
            initial_quaternion=self.initial_quaternion,
            command=(1.0, 1.0, 1.0, float(self.config["height_cmd_default"])),
            height_range=tuple(self.config["height_cmd_range"]),
            cameras={"dog_front_camera": "Front view"},
            render={"fps": 1},
        )
        self.scene = RuntimeScene.for_adapter(
            self.adapter,
            self.robot_xml,
            {"motion_test": map_spec},
            runtime_config,
            robot_cameras=make_standard_robot_cameras(prefix="dog"),
            dimension_context="pinned dog policy runner",
        )
        self.scene.open()
        if not np.isclose(self.scene.model.opt.timestep, float(self.config["simulation_dt"])):
            self.close()
            raise ValueError("composed model timestep does not match upstream dog.yaml")
        self.session = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.decimation = int(self.config["control_decimation"])
        self.step_count = 0
        self.command = np.zeros(3, dtype=np.float64)
        self.height_command = float(self.config["height_cmd_default"])
        self.adapter.reset(self.scene.model, self.scene.data, runtime_config)
        self.scene.runtime.reset_simulation_state()
        if viewer_mode != "headless":
            self._viewer_context = self.scene.viewer(
                browser_only=viewer_mode == "browser"
            )
            self.viewer = self._viewer_context.__enter__()
            if viewer_mode == "native":
                setup_tracking_camera(
                    self.viewer, self.scene.model, "trunk",
                    distance=2.0, azimuth=135.0, elevation=-25.0,
                )
            self._wall_deadline = monotonic()
            if viewer_mode == "browser":
                print("[VIEWER] browser=http://127.0.0.1:8765")

    @property
    def sim_time(self):
        return float(self.scene.data.time)

    @property
    def position(self):
        return self.scene.data.qpos[:3].copy()

    @property
    def yaw(self):
        w, x, y, z = self.scene.data.qpos[3:7]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    @property
    def yaw_rate(self):
        return float(self.scene.data.qvel[5])

    @property
    def roll_pitch(self):
        """Return MuJoCo root roll and pitch in radians from qpos wxyz."""
        w, x, y, z = self.scene.data.qpos[3:7]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        return float(roll), float(np.arcsin(sin_pitch))

    def validate_command(self, command):
        """Validate/return a [vx, vy, wz] policy reference within limits."""
        values = np.asarray(command, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("command must contain exactly three finite values [vx, vy, wz]")
        if np.any(np.abs(values) > COMMAND_LIMITS):
            raise ValueError(f"command values must be within +/-{COMMAND_LIMITS.tolist()}")
        return values

    def step(self, command):
        """Advance exactly one physics step; infer a new action every 4 steps."""
        self.command = self.validate_command(command)
        model, data, runtime = self.scene.model, self.scene.data, self.scene.runtime
        state = runtime.runtime_control(model, data)
        if runtime.consume_reset():
            self.reset()
            state = runtime.runtime_control(model, data)
        if self.step_count % self.decimation == 0:
            obs = self.adapter.build_observation(
                model, data, self.command, height_cmd=self.height_command
            )
            model_input = self.adapter.push_observation(obs)[None, :]
            raw = self.session.run([self.output_name], {self.input_name: model_input})[0]
            action = np.asarray(raw, dtype=np.float64).reshape(-1)
            if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                raise ValueError(f"ONNX returned invalid action shape/value: {action.shape}")
            self.adapter.action_isaac[:] = np.clip(action, -10.0, 10.0)
        self.adapter.apply_control(model, data, self.adapter.action_isaac, state)
        runtime.apply_external_forces(model, data)
        mujoco.mj_step(model, data)
        self.step_count += 1
        state = self.state()
        if self.step_observer is not None:
            self.step_observer(state, self.command.copy())
        if self.viewer_mode == "native":
            if self.viewer.is_running():
                self.viewer.sync()
            elif np.any(self.command):
                raise RuntimeError("native viewer was closed during a commanded motion")
        if self.viewer_mode != "headless":
            # Pace wall-clock display only. MuJoCo time and all skill durations
            # continue to be measured exclusively from data.time.
            timestep = float(model.opt.timestep)
            self._wall_deadline += timestep
            delay = self._wall_deadline - monotonic()
            if delay > 0.0:
                sleep(delay)
            elif delay < -0.1:
                self._wall_deadline = monotonic()
        return state

    def state(self):
        return {"sim_time": self.sim_time, "position": self.position,
                "yaw": self.yaw, "yaw_rate": self.yaw_rate}

    def reset(self):
        self.adapter.reset(self.scene.model, self.scene.data, self.scene.runtime.config)
        self.scene.runtime.reset_simulation_state()
        self.step_count = 0
        self.command[:] = 0.0
        return self.state()

    def stop(self):
        """Advance one normal physics step with a neutral policy reference."""
        return self.step((0.0, 0.0, 0.0))

    def close(self):
        context, self._viewer_context = self._viewer_context, None
        self.viewer = None
        if context is not None:
            context.__exit__(None, None, None)
        scene = getattr(self, "scene", None)
        if scene is not None:
            scene.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
