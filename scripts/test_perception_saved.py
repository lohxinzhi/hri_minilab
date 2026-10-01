#!/usr/bin/env python3
"""Run the reusable perception pipeline over saved dog_front_camera views."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minilab.perception import PerceptionPipeline, YOLODetector  # noqa: E402
from minilab.perception.pipeline import save_annotated_frame  # noqa: E402


def main():
    source_dir = ROOT / "outputs/yolo_smoke_test_v3"
    output_dir = ROOT / "outputs/perception_test"
    pipeline = PerceptionPipeline(YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.25))
    expected_visible_colors = {"center": set(), "offset": set(), "near": set()}
    for name in ("center", "offset", "near"):
        path = source_dir / f"{name}.png"
        with Image.open(path) as source:
            frame = np.asarray(source.convert("RGB"), dtype=np.uint8)
        detections = pipeline.process(frame)
        colors = set()
        print(f"[FRAME] view={name} path={path} shape={frame.shape}")
        for item in detections:
            if item.class_name == "chair" and item.color in {"red", "green"}:
                colors.add(item.color)
            print(
                "[DETECTION] "
                f"class={item.class_name} confidence={item.confidence:.4f} "
                f"color={item.color or 'none'} center=({item.bbox_center[0]:.1f},"
                f"{item.bbox_center[1]:.1f}) bbox={tuple(round(v, 1) for v in item.bbox_xyxy)}"
            )
        expected_visible_colors[name] = colors
        annotated_path = output_dir / f"saved_{name}_annotated.png"
        save_annotated_frame(frame, detections, annotated_path)
        print(f"[ANNOTATED] saved={annotated_path}")
        print(f"[FRAME-RESULT] view={name} chair_colors={sorted(colors)}")
    print("[SAVED-RESULTS] " + " ".join(
        f"{name}={','.join(sorted(colors)) or 'none'}"
        for name, colors in expected_visible_colors.items()
    ))


if __name__ == "__main__":
    main()
