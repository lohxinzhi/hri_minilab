"""Detect objects in camera frames with COCO-pretrained YOLOv8."""

import numpy as np
from ultralytics import YOLO


class VisionModule:
    """Detect objects with a pretrained model reused across frames."""

    def __init__(self, model: str = "yolov8n.pt") -> None:
        """Accept a pretrained model name or a path to custom model weights."""
        self._model_path = model
        self._model: YOLO | None = None

    def get_bbox(self, frame: np.ndarray) -> list[list[float]]:
        """Return detected boxes as [x_min, y_min, x_max, y_max] pixel coordinates.

        Args:
            frame: One HWC uint8 image with three BGR channels. Convert RGB
                camera frames to BGR before calling this method.

        Returns:
            One coordinate list per detected object, or [] if none are detected.
            Coordinates are relative to the original frame size.

        The selected model loads on first use. The default COCO-pretrained
        yolov8n.pt weights download automatically if not available locally.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.ndim != 3 or frame.shape[2] != 3 or 0 in frame.shape:
            raise ValueError("frame must be a non-empty HWC image with three channels")
        if frame.dtype != np.uint8:
            raise ValueError("frame must have dtype uint8")

        if self._model is None:
            self._model = YOLO(self._model_path)
        result = self._model.predict(source=frame, verbose=False)[0]
        if result.boxes is None:
            return []
        return result.boxes.xyxy.cpu().tolist()
