#!/usr/bin/env python3
"""Run a pretrained COCO detector on the saved dog_front_camera scene image.

This is a one-image smoke test only. It does not classify object colors, read
evaluation ground truth, or control the robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_IMAGE = PROJECT_ROOT / "outputs/scene_test/front_camera_scene.png"
OUTPUT_IMAGE = PROJECT_ROOT / "outputs/yolo_smoke_test/front_camera_annotated.png"
WEIGHTS = "yolo11n.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=INPUT_IMAGE)
    parser.add_argument("--annotated", type=Path, default=OUTPUT_IMAGE)
    args = parser.parse_args()

    # Load and validate the camera image before initializing the detector.
    with Image.open(args.image) as source:
        rgb_image = source.convert("RGB")
        image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    print(f"[IMAGE] source=dog_front_camera path={args.image}")
    print(f"[IMAGE] shape={image.shape} dtype={image.dtype}")

    model = YOLO(WEIGHTS)
    result = model.predict(source=image, verbose=False)[0]

    detections: list[tuple[str, float, list[float]]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = [float(value) for value in box.xyxy[0].tolist()]
            class_name = str(result.names[class_id])
            detections.append((class_name, confidence, bbox))
            print(
                f"[DETECT] class={class_name} confidence={confidence:.4f} "
                f"bbox={[round(value, 1) for value in bbox]}"
            )

    args.annotated.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics plot() returns BGR; convert it back to RGB before saving.
    annotated_bgr = result.plot()
    Image.fromarray(annotated_bgr[:, :, ::-1]).save(args.annotated)
    print(f"[YOLO] weights={WEIGHTS} detections={len(detections)}")
    print(f"[YOLO] annotated={args.annotated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
