#!/usr/bin/env python3
"""Smoke-test MiniLab RGB capture while the pinned upstream player runs.

The unchanged upstream eg/play.py runs its pretrained ONNX policy headlessly.
This script observes MuJoCo physics steps in-process and throttles front-camera
capture by simulation time; it does not duplicate the policy/control loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
import time

import mujoco
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "quadruped_mujoco"
sys.path.insert(0, str(PROJECT_ROOT))

from minilab.platform.camera import FRONT_CAMERA_NAME, FrontCamera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=2.0,
        help="wall-clock seconds for the upstream player's --duration",
    )
    parser.add_argument("--render-hz", type=float, default=15.0,
                        help="maximum capture rate in MuJoCo simulation time")
    parser.add_argument("--width", type=int, default=640, help="RGB frame width")
    parser.add_argument("--height", type=int, default=480, help="RGB frame height")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if not 0 < args.render_hz <= 20:
        parser.error("--render-hz must be in the range (0, 20]")
    return args


def main() -> int:
    args = parse_args()
    player = UPSTREAM_ROOT / "eg" / "play.py"
    if not player.is_file():
        raise FileNotFoundError(f"Pinned upstream player not found: {player}")

    original_mj_step = mujoco.mj_step
    original_argv = sys.argv[:]
    camera_holder: list[FrontCamera] = []
    representative_frame = None
    first_capture_time = None
    last_capture_time = None

    def capture_after_physics_step(model, data, *step_args, **step_kwargs):
        nonlocal representative_frame, first_capture_time, last_capture_time
        original_mj_step(model, data, *step_args, **step_kwargs)
        if not camera_holder:
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, FRONT_CAMERA_NAME
            )
            if camera_id < 0:
                raise RuntimeError(
                    f"Upstream composed model lacks required camera "
                    f"{FRONT_CAMERA_NAME!r}"
                )
            camera_holder.append(FrontCamera(
                model,
                camera_name=FRONT_CAMERA_NAME,
                width=args.width,
                height=args.height,
                render_hz=args.render_hz,
            ))
        camera = camera_holder[0]
        if camera.update(data):
            metadata = camera.latest_metadata
            frame = camera.get_latest_frame(copy=False)
            assert metadata is not None and frame is not None
            assert frame.shape == (args.height, args.width, 3)
            assert frame.dtype == np.uint8
            if representative_frame is None:
                representative_frame = camera.get_latest_frame(copy=True)
                cached = camera.get_latest_frame(copy=False)
                cached_before = cached.copy()
                representative_frame[...] = 0
                assert np.array_equal(cached, cached_before), (
                    "mutating the safe frame copy changed the cached image"
                )
                representative_frame = camera.get_latest_frame(copy=True)
                first_capture_time = metadata.simulation_time
            last_capture_time = metadata.simulation_time

    sys.argv = [
        str(player), "--headless", "--duration", str(args.duration)
    ]
    started = time.monotonic()
    try:
        # Run the actual pinned ONNX/PD simulation loop without editing it.
        mujoco.mj_step = capture_after_physics_step
        runpy.run_path(str(player), run_name="__main__")
    finally:
        mujoco.mj_step = original_mj_step
        sys.argv = original_argv
        for camera in camera_holder:
            camera.close()

    wall_elapsed = time.monotonic() - started
    if not camera_holder:
        raise RuntimeError("Upstream simulation advanced no physics steps")
    camera = camera_holder[0]
    frames = camera.frames_captured
    if frames < 2 or representative_frame is None:
        raise RuntimeError(f"Expected multiple rendered frames; got {frames}")
    if first_capture_time is None or last_capture_time is None:
        raise RuntimeError("Rendered frame timestamps were not recorded")
    sim_elapsed = last_capture_time - first_capture_time
    if sim_elapsed <= 0:
        raise RuntimeError("Rendered frames did not span simulation time")
    measured_hz = (frames - 1) / sim_elapsed

    output_path = PROJECT_ROOT / "outputs" / "camera_test" / "front_camera_rgb.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(representative_frame, mode="RGB").save(output_path)
    print(
        f"[CAMERA] name={FRONT_CAMERA_NAME} "
        f"shape={representative_frame.shape} dtype={representative_frame.dtype}"
    )
    print(
        f"[CAMERA] frames={frames} elapsed_sim={sim_elapsed:.3f}s "
        f"measured_hz={measured_hz:.2f} (simulation time); "
        f"run_wall={wall_elapsed:.3f}s"
    )
    print(f"[CAMERA] saved={output_path}")
    print("[CAMERA] source=unchanged upstream ONNX simulation; headless")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
