# Task 2 scene detector smoke test

These are smoke-test results from the 640×480 `dog_front_camera` frames and
the pretrained Ultralytics YOLO11n COCO model (Ultralytics 8.4.168), using its
default confidence threshold. They are not the final Task 4 evaluation
dataset and do not guarantee detection in other views or conditions.

| View | Green chair → `chair` | Red chair → `chair` | `ball_01` → `sports ball` |
|---|---:|---:|---:|
| Center, normal start | 0.8529 | 0.8710 | 0.7003 |
| Lateral offset | 0.8935 | 0.8186 | 0.8081 |
| Closer, about 2.9 m | not in frame | 0.8312 | 0.8575 |

The ball was classified as `sports ball` in all three tested views. The green
chair was outside the closer ball-centered view; both chairs were detected in
the normal and lateral-offset views. This is a limited smoke test, not a
claim of perfect YOLO performance.

Re-run one frame with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quadruped_mujoco
python scripts/test_yolo_smoke.py \
  --image outputs/yolo_smoke_test_v3/center.png \
  --annotated outputs/yolo_smoke_test_v3/center_annotated.png
```

The script loads the named `yolo11n.pt` model through Ultralytics, which
downloads the pretrained weights when they are not already available. The
local weight file and test images are excluded from Git.
