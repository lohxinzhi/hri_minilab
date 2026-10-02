"""Detect objects in camera frames with COCO-pretrained YOLOv8."""

import time

import cv2
import mujoco
import numpy as np
from ultralytics import YOLO


class VisionModule:
    """Detect objects with a pretrained model reused across frames."""

    # OpenCV HSV: hue 0..179, saturation/value 0..255. Restrict the
    # background exclusion to dark blues so brighter blue objects remain.
    DARK_BLUE_HSV_LOWER = (100, 40, 0)
    DARK_BLUE_HSV_UPPER = (130, 255, 120)
    # Birch texture is predominantly H=17, S=75. Allow lighting variation,
    # while retaining saturated orange/yellow objects and neutral pixels.
    BIRCH_HSV_LOWER = (12, 25, 80)
    BIRCH_HSV_UPPER = (25, 160, 255)

    def __init__(
        self, model: str = "yolov8n.pt", *, confidence_threshold: float = 0.5
    ) -> None:
        """Accept a pretrained model name or a path to custom model weights."""
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold
        self._model_path = model
        self._model: YOLO | None = None
        self._renderer: mujoco.Renderer | None = None
        self._window_name = "Robot dog FPV"
        self._window_open = False
        self.detections = []
        self.frame_size = (640, 480)
        self._last_detection_log = None

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
        can stop streaming while continuing simulation. Labels all classes above
        the confidence threshold, with boxes matching their estimated colors.
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
        # Estimate every color before drawing, so overlapping boxes cannot
        # contaminate the pixels used for a later object's color estimate.
        colors = [self._object_color(frame, item["bbox"]) for item in detections]
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.detections = [
            {**item, "color": color_name}
            for item, (color_name, _) in zip(detections, colors, strict=True)
        ]
        now = time.monotonic()
        log_detections = bool(detections) and (
            self._last_detection_log is None or now - self._last_detection_log >= 1.0
        )
        if log_detections:
            self._last_detection_log = now
        for detection, (color_name, color) in zip(detections, colors, strict=True):
            x_min, y_min, x_max, y_max = map(int, detection["bbox"])
            if log_detections:
                print(
                    f"[DETECT] class={detection['label']} color={color_name} "
                    f"conf={detection['confidence']:.2f} "
                    f"bbox={[x_min, y_min, x_max, y_max]}"
                )
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{detection['label']} {detection['confidence']:.2f} {color_name}"
            cv2.putText(
                frame,
                label,
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _object_color(frame: np.ndarray, bbox) -> tuple[str, tuple[int, int, int]]:
        """Estimate color from median HSV inside a clipped BGR bounding box.

        Exclude dark-blue and pale birch background pixels before HSV medians.
        Low saturation/value identify neutral colors, whose hue alone is not
        meaningful. Return unknown when no usable pixels remain.
        """
        height, width = frame.shape[:2]
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min, x_max = np.clip([x_min, x_max], 0, width)
        y_min, y_max = np.clip([y_min, y_max], 0, height)
        crop = frame[
            y_min:y_max,
            x_min:x_max,
        ]
        if crop.size == 0:
            return "unknown", (128, 128, 128)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        background = cv2.inRange(
            hsv, VisionModule.DARK_BLUE_HSV_LOWER, VisionModule.DARK_BLUE_HSV_UPPER
        )
        birch = cv2.inRange(
            hsv, VisionModule.BIRCH_HSV_LOWER, VisionModule.BIRCH_HSV_UPPER
        )
        background = cv2.bitwise_or(background, birch)
        hsv = hsv[background == 0]
        if hsv.size == 0:
            return "unknown", (128, 128, 128)
        hues = hsv[:, 0].astype(float)
        # Red spans both ends of OpenCV's 0..179 hue scale. Unwrap that
        # boundary when red pixels dominate, before taking the median.
        if np.mean((hues < 10) | (hues >= 170)) > 0.5:
            hues[hues >= 170] -= 180
        hue = float(np.median(hues)) % 180
        saturation, value = np.median(hsv[:, 1:], axis=0)
        if value < 50:
            name = "black"
        elif saturation < 40:
            name = "white" if value >= 200 else "gray"
        elif hue < 10 or hue >= 170:
            name = "red"
        elif hue < 25:
            name = "brown" if value < 180 else "orange"
        elif hue < 35:
            name = "yellow"
        elif hue < 85:
            name = "green"
        elif hue < 100:
            name = "cyan"
        elif hue < 130:
            name = "blue"
        elif hue < 150:
            name = "purple"
        else:
            name = "pink"
        median_hsv = np.array([[[round(hue) % 180, saturation, value]]], dtype=np.uint8)
        bgr = cv2.cvtColor(median_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return name, tuple(int(channel) for channel in bgr)

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
            One coordinate list per object strictly above confidence_threshold,
            or [] if none pass the threshold.
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
        result = self._model.predict(
            source=frame, conf=self.confidence_threshold, verbose=False
        )[0]
        if result.boxes is None:
            return []
        coordinates = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        if not include_labels:
            return [
                bbox
                for bbox, confidence in zip(coordinates, confidences, strict=True)
                if confidence > self.confidence_threshold
            ]
        class_ids = result.boxes.cls.cpu().tolist()
        return [
            {
                "bbox": bbox,
                "label": result.names[int(class_id)],
                "confidence": confidence,
            }
            for bbox, class_id, confidence in zip(
                coordinates, class_ids, confidences, strict=True
            )
            if confidence > self.confidence_threshold
        ]
