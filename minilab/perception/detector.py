"""Ultralytics-backed COCO detector with project-owned result objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional

import numpy as np


DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.25


@dataclass(frozen=True)
class Detection:
    """Detector output independent of Ultralytics Results objects."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    color: Optional[str] = None
    red_fraction: Optional[float] = None
    green_fraction: Optional[float] = None

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bbox_width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def bbox_height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def bbox_area(self) -> float:
        return self.bbox_width * self.bbox_height


def detections_from_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics-like result to plain immutable detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = result.names
    converted = []
    for box in boxes:
        class_id = int(np.asarray(box.cls).reshape(-1)[0])
        confidence = float(np.asarray(box.conf).reshape(-1)[0])
        coords = np.asarray(box.xyxy, dtype=np.float64).reshape(-1, 4)[0]
        if isinstance(names, Mapping):
            class_name = str(names[class_id])
        else:
            class_name = str(names[class_id])
        converted.append(Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=tuple(float(value) for value in coords),
        ))
    return converted


class YOLODetector:
    """Run COCO YOLO11n on HxWx3 RGB uint8 images.

    The model name is resolved by Ultralytics, which loads a local file if
    present and otherwise retrieves/caches its official pretrained weights.
    All classes above ``confidence_threshold`` are returned.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 confidence_threshold: float = DEFAULT_CONFIDENCE,
                 *, device: str = "cpu"):
        if not np.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        from ultralytics import YOLO

        self.model_name = str(model_name)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.model = YOLO(self.model_name)
        self.last_inference_seconds: Optional[float] = None

    @staticmethod
    def validate_frame(frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"frame must have shape HxWx3 RGB, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"frame must have dtype uint8, got {image.dtype}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("frame dimensions must be non-zero")
        return image

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image = self.validate_frame(frame)
        started = perf_counter()
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        self.last_inference_seconds = perf_counter() - started
        return detections_from_result(results[0])

    @staticmethod
    def filter_by_class(
        detections: Iterable[Detection], class_name: str
    ) -> list[Detection]:
        """Select a class while leaving the original all-class result intact."""
        return [item for item in detections if item.class_name == class_name]
