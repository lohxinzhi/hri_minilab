#!/usr/bin/env python3
"""Headless live YOLO smoke test on dog_front_camera frames."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minilab.perception import PerceptionPipeline, YOLODetector  # noqa: E402
from minilab.perception.pipeline import save_annotated_frame  # noqa: E402
from scripts.play_yolo_search_scene import main as run_scene  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="upstream headless wall-clock duration in seconds")
    args = parser.parse_args()
    pipeline = PerceptionPipeline(YOLODetector("yolo11n.pt", 0.25))
    output_dir = ROOT / "outputs/perception_test"
    counts = Counter()
    inference_seconds = []
    saved = 0

    def on_rgb_frame(frame, simulation_time):
        nonlocal saved
        detections = pipeline.process(frame)
        inference_seconds.append(pipeline.detector.last_inference_seconds)
        classes = {item.class_name for item in detections}
        colors = {item.color for item in detections if item.class_name == "chair"}
        if "chair" in classes and "green" in colors:
            counts["green_chair_frames"] += 1
        if "chair" in classes and "red" in colors:
            counts["red_chair_frames"] += 1
        if "sports ball" in classes:
            counts["sports_ball_frames"] += 1
        counts["frames"] += 1
        print(
            f"[LIVE-FRAME] sim_time={simulation_time:.3f} "
            f"detections={len(detections)} classes={sorted(classes)} "
            f"chair_colors={sorted(color for color in colors if color)} "
            f"inference_s={inference_seconds[-1]:.4f}"
        )
        for item in detections:
            print(
                "[DETECTION] "
                f"class={item.class_name} confidence={item.confidence:.4f} "
                f"color={item.color or 'none'} center=({item.bbox_center[0]:.1f},"
                f"{item.bbox_center[1]:.1f}) bbox={tuple(round(v, 1) for v in item.bbox_xyxy)}"
            )
        if saved < 3:
            path = output_dir / f"live_{saved + 1:02d}_annotated.png"
            save_annotated_frame(frame, detections, path)
            print(f"[ANNOTATED] saved={path}")
            saved += 1

    started = perf_counter()
    run_scene(
        frame_callback=on_rgb_frame,
        upstream_args=["--headless", "--duration", str(args.duration)],
    )
    elapsed_wall = perf_counter() - started
    average_inference = (
        sum(inference_seconds) / len(inference_seconds) if inference_seconds else 0.0
    )
    warmed_inference = (
        sum(inference_seconds[1:]) / len(inference_seconds[1:])
        if len(inference_seconds) > 1 else average_inference
    )
    print(
        f"[PERCEPTION-LIVE] camera=dog_front_camera frames={counts['frames']} "
        f"green_chair_frames={counts['green_chair_frames']} "
        f"red_chair_frames={counts['red_chair_frames']} "
        f"sports_ball_frames={counts['sports_ball_frames']} "
        f"average_inference_s={average_inference:.4f} "
        f"warmed_average_inference_s={warmed_inference:.4f} "
        f"elapsed_wall_s={elapsed_wall:.2f}"
    )


if __name__ == "__main__":
    main()
