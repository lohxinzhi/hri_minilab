"""Detect objects in camera frames with COCO-pretrained YOLOv8."""

import cv2
import mujoco
import numpy as np
from ultralytics import YOLO


class VisionModule:
    """Detect objects with a pretrained model reused across frames."""

    COCO_SCENE_CLASSES = frozenset({"chair", "bench", "car", "bicycle"})

    def __init__(self, model: str = "yolov8n.pt") -> None:
        """Accept a pretrained model name or a path to custom model weights."""
        self._model_path = model
        self._model: YOLO | None = None
        self._renderer: mujoco.Renderer | None = None
        self._window_name = "Robot dog FPV"
        self._window_open = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def stream_fpv(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        camera: str = "dog_front_camera",
    ) -> bool:
        """Render one live FPV frame and display it in an OpenCV window.

        Call repeatedly from the simulation loop on the main thread. Returns
        False when Q, Escape, or the window close button is pressed; the caller
        can stop streaming while continuing simulation. Labels all detections;
        selected classes get green boxes and labels, and others get grey ones.
        """
        if not self._window_open:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._window_open = True

        frame = self.render_fpv(mj_model, mj_data, camera)
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if (
            key in (ord("q"), ord("Q"), 27)
            or cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1
        ):
            self.close()
            return False
        return True

    def render_frame(self, mj_model, mj_data, camera):
        """Render an existing camera as BGR pixels without opening a window."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(mj_model, height=480, width=640)
        self._renderer.update_scene(mj_data, camera=camera)
        return cv2.cvtColor(self._renderer.render(), cv2.COLOR_RGB2BGR)

    def render_fpv(self, mj_model, mj_data, camera="dog_front_camera"):
        """Return the annotated FPV frame for a browser feed."""
        frame = self.render_frame(mj_model, mj_data, camera)
        detections = self.get_bbox(frame, include_labels=True)
        for detection in detections:
            selected = detection["label"] in self.COCO_SCENE_CLASSES
            x_min, y_min, x_max, y_max = map(int, detection["bbox"])
            color = (0, 255, 0) if selected else (64, 64, 64)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{detection['label']} {detection['confidence']:.2f}"
            cv2.putText(
                frame,
                label,
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def close(self) -> None:
        """Release FPV rendering resources and close the OpenCV window."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._window_open:
            self._window_open = False
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                # The user may already have closed the window through the OS.
                pass

    def get_bbox(
        self, frame: np.ndarray, *, include_labels: bool = False
    ) -> list[list[float]] | list[dict]:
        """Return detected boxes as [x_min, y_min, x_max, y_max] pixel coordinates.

        Args:
            frame: One HWC uint8 image with three BGR channels. Convert RGB
                camera frames to BGR before calling this method.
            include_labels: Include the class name and confidence with each box.

        Returns:
            One coordinate list per detected object, or [] if none are detected.
            Coordinates are relative to the original frame size.
            With include_labels=True, returns dictionaries containing bbox,
            label, and confidence instead of coordinate lists.

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
        coordinates = result.boxes.xyxy.cpu().tolist()
        if not include_labels:
            return coordinates
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        return [
            {
                "bbox": bbox,
                "label": result.names[int(class_id)],
                "confidence": confidence,
            }
            for bbox, class_id, confidence in zip(
                coordinates, class_ids, confidences, strict=True
            )
        ]
