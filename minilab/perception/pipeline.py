"""YOLO detection plus conservative per-chair HSV color augmentation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .color import ColorResult, HSVColorClassifier
from .detector import Detection, YOLODetector


class PerceptionPipeline:
    """Return project-owned detections, optionally including chair color."""

    def __init__(self, detector: Optional[YOLODetector] = None,
                 color_classifier: Optional[HSVColorClassifier] = None):
        self.detector = detector or YOLODetector()
        self.color_classifier = color_classifier or HSVColorClassifier()
        self.last_color_results: dict[int, ColorResult] = {}

    def process(self, rgb_frame: np.ndarray) -> list[Detection]:
        detections = self.detector.detect(rgb_frame)
        augmented = []
        self.last_color_results = {}
        for index, item in enumerate(detections):
            if item.class_name == "chair":
                color = self.color_classifier.classify(rgb_frame, item.bbox_xyxy)
                self.last_color_results[index] = color
                augmented.append(replace(
                    item,
                    color=color.color,
                    red_fraction=color.red_fraction,
                    green_fraction=color.green_fraction,
                ))
            else:
                augmented.append(item)
        return augmented


def save_annotated_frame(rgb_frame: np.ndarray, detections: list[Detection], path):
    """Draw project-owned boxes/labels and save an RGB debug image."""
    frame = YOLODetector.validate_frame(rgb_frame)
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = {"red": (245, 55, 50), "green": (45, 220, 80), "unknown": (255, 220, 70)}
    for item in detections:
        x1, y1, x2, y2 = item.bbox_xyxy
        label = f"{item.class_name} {item.confidence:.2f}"
        if item.color is not None:
            label += f" / {item.color}"
        outline = colors.get(item.color, (50, 190, 255))
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=3)
        text_box = draw.textbbox((0, 0), label, font=ImageFont.load_default())
        label_h = text_box[3] - text_box[1]
        label_y = max(0, y1 - label_h - 5)
        draw.rectangle((x1, label_y, x1 + (text_box[2] - text_box[0]) + 6,
                        label_y + label_h + 4), fill=outline)
        draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0),
                  font=ImageFont.load_default())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target
