# Task 2 perception smoke-test pipeline

The project-owned `minilab.perception` pipeline accepts only `dog_front_camera`
frames supplied by `minilab.platform.camera.FrontCamera`: RGB channel order,
`uint8`, shape `(height, width, 3)`. `YOLODetector` loads Ultralytics' pretrained
COCO `yolo11n.pt` weights from the local path or the normal Ultralytics cache.
The tested Ultralytics version is 8.4.168. The default confidence threshold is
0.25; all detections returned above that threshold remain available as plain
project `Detection` records, independent of Ultralytics result objects.

For `chair` detections, `HSVColorClassifier` analyzes the centered inner 70% by
72% of the clipped bbox to reduce floor/background pixels around the object.
OpenCV conversion uses `COLOR_RGB2HSV` (not BGR conversion). OpenCV hue is
0–179: red ranges are 0–12 and 168–179 to handle hue wraparound; green is
35–90. Saturation must be at least 70 and value at least 45. A color must cover
at least 8% of the ROI and exceed the other candidate by a 4 percentage-point
margin and 1.5x dominance; otherwise it is reported as `unknown`. These are
general configurable thresholds, not detector tuning.

`scripts/test_perception_saved.py` exercises the existing center, offset, and
near front-camera smoke-test images. `scripts/test_perception_live.py` wires
the same pipeline to RGB frames emitted by the existing custom-scene launcher.
Its annotated images and downloaded model weights are temporary/local outputs,
not assignment source files. This is a perception smoke test only; it does not
implement target search, motion, steering, or approach behavior.
