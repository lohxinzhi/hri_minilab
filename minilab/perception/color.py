"""Conservative HSV color classification for detected object crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVColorConfig:
    # OpenCV HSV hue is [0, 179]. Red wraps across its 0/179 boundary.
    red_hue_ranges: tuple[tuple[int, int], ...] = ((0, 12), (168, 179))
    green_hue_ranges: tuple[tuple[int, int], ...] = ((35, 90),)
    saturation_min: int = 70
    value_min: int = 45
    inner_width_fraction: float = 0.70
    inner_height_fraction: float = 0.72
    min_color_fraction: float = 0.08
    min_fraction_margin: float = 0.04
    dominance_ratio: float = 1.5
    minimum_roi_pixels: int = 25


@dataclass(frozen=True)
class ColorResult:
    color: str
    red_fraction: float
    green_fraction: float
    eligible_fraction: float
    pixels_sampled: int


def clip_bbox_xyxy(bbox_xyxy, image_width: int, image_height: int):
    """Clip float xyxy box to valid image bounds; reject empty boxes."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    coords = np.asarray(bbox_xyxy, dtype=np.float64)
    if coords.shape != (4,) or not np.all(np.isfinite(coords)):
        raise ValueError("bbox_xyxy must contain four finite coordinates")
    x1, y1, x2, y2 = coords
    x1 = int(np.clip(np.floor(x1), 0, image_width))
    y1 = int(np.clip(np.floor(y1), 0, image_height))
    x2 = int(np.clip(np.ceil(x2), 0, image_width))
    y2 = int(np.clip(np.ceil(y2), 0, image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox does not intersect the image")
    return x1, y1, x2, y2


class HSVColorClassifier:
    """Classify red/green from a centered inner bbox region.

    The central 70% x 72% region reduces edge background and neighboring
    pixels. Fractions use all pixels in that ROI as denominator; low-saturation
    and low-value pixels are excluded from red/green counts, not treated as a
    color. Weak or ambiguous evidence returns ``unknown``.
    """

    def __init__(self, config: HSVColorConfig = HSVColorConfig()):
        self.config = config

    def classify(self, rgb_frame: np.ndarray, bbox_xyxy) -> ColorResult:
        frame = np.asarray(rgb_frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("RGB frame must be HxWx3 uint8")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = clip_bbox_xyxy(bbox_xyxy, width, height)
        box_w, box_h = x2 - x1, y2 - y1
        inner_w = max(1, int(round(box_w * self.config.inner_width_fraction)))
        inner_h = max(1, int(round(box_h * self.config.inner_height_fraction)))
        left = x1 + (box_w - inner_w) // 2
        top = y1 + (box_h - inner_h) // 2
        roi = frame[top:top + inner_h, left:left + inner_w]
        pixel_count = int(roi.shape[0] * roi.shape[1])
        if pixel_count < self.config.minimum_roi_pixels:
            return ColorResult("unknown", 0.0, 0.0, 0.0, pixel_count)

        # This explicitly uses RGB conversion; cv2's BGR default would swap red/blue.
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        eligible = (saturation >= self.config.saturation_min) & (value >= self.config.value_min)
        red = np.zeros(hue.shape, dtype=bool)
        for low, high in self.config.red_hue_ranges:
            red |= (hue >= low) & (hue <= high)
        green = np.zeros(hue.shape, dtype=bool)
        for low, high in self.config.green_hue_ranges:
            green |= (hue >= low) & (hue <= high)
        red &= eligible
        green &= eligible
        red_fraction = float(np.count_nonzero(red) / pixel_count)
        green_fraction = float(np.count_nonzero(green) / pixel_count)
        eligible_fraction = float(np.count_nonzero(eligible) / pixel_count)

        color = "unknown"
        difference = abs(red_fraction - green_fraction)
        if difference >= self.config.min_fraction_margin:
            if (red_fraction >= self.config.min_color_fraction
                    and red_fraction >= self.config.dominance_ratio * green_fraction):
                color = "red"
            elif (green_fraction >= self.config.min_color_fraction
                    and green_fraction >= self.config.dominance_ratio * red_fraction):
                color = "green"
        return ColorResult(color, red_fraction, green_fraction,
                           eligible_fraction, pixel_count)
