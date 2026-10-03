# Robot dog browser dashboard

## Setup on another machine

The reference setup is Ubuntu 24.04 x86-64 with Python 3.11.16. The browser
GUI still needs local OpenGL rendering for its camera feeds. A GPU is optional;
CPU inference and software rendering are available. Other operating systems have
not been validated for this project.

### Required packages and tools

These are the versions installed in the working simulation environment, rather
than minimum supported versions or a complete lockfile. Python package dependencies
are installed automatically by pip. All environment requirements are declared
in [environment.yml](environment.yml).

| Package/tool | Reference version | Purpose |
| --- | --- | --- |
| Python | 3.11.16 | Application runtime |
| Conda | 26.5.3 | Isolated environment; Miniconda or Anaconda |
| pip | 26.1.2 | Python package installation |
| setuptools | 83.0.0 | Building/installing the runtime package |
| Git | 2.43.0 | Downloading both repositories |
| mujoco | 3.14.0 | Physics and camera rendering |
| mujoco-runtime-control | 0.6.0 | `runtime_control`, installed from `quadruped_mujoco` |
| numpy | 2.4.6 | Numerical operations |
| onnxruntime | 1.30.0 | Robot locomotion policy inference |
| PyYAML | 6.0.3 | Reading `dog.yaml` |
| ultralytics | 8.4.166 | COCO-pretrained YOLOv8 detector |
| torch | 2.14.0 | YOLO inference backend |
| torchvision | 0.29.0 | PyTorch vision utilities |
| opencv-python | 5.0.0.93 | Camera frames, annotations and JPEG encoding |
| matplotlib | 3.11.2 | Saved heading/angular-velocity plots |
| Pillow | 12.3.0 | Image utilities used by the runtime package |
| glfw | 2.10.2 | MuJoCo's default rendering backend |
| openai | 3.22.1 | Cloud LLM client |

A modern browser supporting JavaScript, fetch and MJPEG streams is required.
The dashboard was checked in Chromium; no fixed browser version is required.
Ubuntu OpenGL/Mesa libraries come from the OS package manager and are not pinned.
The dashboard uses Python's standard-library HTTP server: Node.js, npm, ROS and
Isaac Gym are not needed to run this simulation. The MuJoCo Python wheel includes
the MuJoCo library; no separate MuJoCo installation is required
([MuJoCo installation documentation](https://mujoco.readthedocs.io/en/stable/python.html#installation)).

### 1. Download the application and runtime

After installing Git and Miniconda/Anaconda, run:

```bash
mkdir -p workspace/src
cd workspace/src
git clone https://github.com/lohxinzhi/hri_minilab.git
git clone https://github.com/aoqianz/quadruped_mujoco.git
```

The runtime revision inspected for these instructions is
`dd40180f1121a66373d261e64a9a09eb69b1b2a7` (package version 0.6.0).
For that revision, run:

```bash
git -C quadruped_mujoco checkout dd40180f1121a66373d261e64a9a09eb69b1b2a7
```

Keep the application assets in their existing relative locations:

```text
workspace/src/
├── quadruped_mujoco/          # Provides the installed runtime_control package
└── hri_minilab/
    ├── play.py
    ├── environment.yml      # Conda environment and Python dependencies
    ├── dog.yaml
    ├── model_3400.onnx       # Pretrained locomotion policy
    ├── yolov8n.pt            # COCO-pretrained detector
    ├── dog/                 # Robot MJCF and meshes
    ├── map/                 # Scene XML and meshes/ subfolders
    └── web/index.html       # Dashboard
```

The robot model, ONNX policy, YOLO weights and scene assets are included in the
application repository. There is no need to download the original vehicle GLBs
or install mesh-conversion tools. See [map/README.md](map/README.md) for asset
sources and attribution. If `yolov8n.pt` is missing, Ultralytics downloads it on
first use, which requires internet access.

### 2. Create the Python environment

From `workspace/src`, enter the application directory before creating the environment:

```bash
cd hri_minilab
conda env create -f environment.yml
conda activate quad_mujuco
python -m pip check
```

The file installs Python, pip, the pinned application packages, and the sibling
`quadruped_mujoco` checkout in editable mode. Run these commands from
`hri_minilab` so the `../quadruped_mujoco` dependency resolves correctly.
If the environment already exists, update it from the same file:

```bash
conda env update -n quad_mujuco -f environment.yml
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if your platform needs a CPU-specific or CUDA-specific wheel index. Keep the
`torch`/`torchvision` versions paired. This project uses CPU ONNX Runtime;
`onnxruntime-gpu` and a CUDA toolkit are not required for the locomotion policy.
Do not install `opencv-python-headless` alongside `opencv-python` because both
provide the same `cv2` module.

### 3. Configure rendering on Ubuntu

For a desktop session with working graphics drivers, MuJoCo's default GLFW
backend can be used. If OpenGL libraries are missing, install:

```bash
sudo apt-get update
sudo apt-get install libgl1 libegl1 libglfw3 libosmesa6
```

For offscreen rendering without a desktop display, select EGL before launching
Python (this is the backend used in the browser smoke checks):

```bash
export MUJOCO_GL=egl
```

EGL needs a working graphics driver. If EGL cannot initialize, use Mesa software
rendering instead:

```bash
export MUJOCO_GL=osmesa
```

Select one backend per launch. These Linux settings should not be copied to
Windows or macOS. Running without `--gui` does not render camera feeds.

### 4. Configure the cloud LLM

Set the key in the same shell that launches Python. This example reads it without
echoing it or placing its value in shell history:

```bash
read -r -s -p "API key: " OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
export OPENAI_MODEL="gpt-6-luna"
```

`gpt-6-luna` is the application's default model name. Set `OPENAI_MODEL` to a
model available to your account that supports the strict JSON schema used by
the dialogue manager. For an OpenAI-compatible cloud provider, also set
`OPENAI_BASE_URL` to its API base URL. Credentials and model access are supplied
by the user; internet access is needed for chat requests. No API key belongs in
the repository. The active model name is shown in the dashboard.

### 5. Launch and verify

From `hri_minilab`, with the environment activated:

```bash
python -c "import mujoco, numpy, onnxruntime, yaml, cv2, ultralytics, openai, matplotlib, runtime_control; print('Imports OK')"
python -m unittest discover -s tests -v
python play.py --map coco_scene --gui
```

The dashboard opens automatically at `http://127.0.0.1:8765`. Open that URL
manually if automatic browser launch is unavailable. Use `--gui-port 8766` for
a different port. The server listens only on the local machine.

The detector defaults to YOLOv8 nano (`yolov8n.pt`). Select another YOLO model
or a custom weights path with `--vision-model`:

```bash
python play.py --map coco_scene --gui --vision-model yolov8s.pt
python play.py --map coco_scene --gui --vision-model /path/to/custom_weights.pt
```

Describe/VQA uses a separate OpenAI VLM, defaulting to `gpt-6-luna`.
Use `--vlm-model` to select a model available to your account that supports
image input and Chat Completions. See the [official OpenAI vision guide](https://developers.openai.com/api/docs/guides/images-vision).
This model uses `OPENAI_API_KEY` and the optional `OPENAI_BASE_URL`;
`OPENAI_MODEL` configures the dialogue planner separately.

```bash
python play.py --map coco_scene --gui --vision-model yolo11n.pt --vlm-model gpt-4o
```

The dashboard header displays the planner LLM, VLM, and object detection model.
You can also set `VisionModule(vlm_model="gpt-4o")` directly in Python.

Goto search/approach uses YOLO bounding boxes by default. Add `--use-vlm`
to use the configured `--vlm-model` instead:

```bash
python play.py --map coco_scene --gui --use-vlm --vlm-model gpt-4o
```

VLM detection uses `DETECTION_PROMPT` to locate the requested object and color
in an unannotated FPV image. It runs on the visual worker with at most one
bounding-box request in flight. A VLM goto remains stationary until its first
valid detection response arrives: a matching target starts approach, and a
not-found response starts search. This initial wait resets for each new mission;
API failures do not count as a detection result. Search then keeps turning
through one continuous 360-degree revolution while requests are pending. If the revolution completes
before the last request, the robot waits for that response before reporting
not found. During VLM approach, bounding boxes refresh in the background after
0.25 seconds while the last valid box keeps steering. Forward speed is 0.2 m/s
and yaw correction is `-0.8 * horizontal_error` rad/s, both 40% of the YOLO
approach speeds. A completed observation that no longer matches the target
returns the mission to search. Invalid responses or a box older than five
seconds since receipt stop approach until a usable result arrives, without
changing phase. Both YOLO and VLM modes stop below a world-frame distance of 1.5 m.
YOLO approach remains at 0.5 m/s with `-2 * horizontal_error` yaw correction.
Invalid boxes and API failures retry without being treated as a target loss;
the existing 60-second goto deadline still applies. Boxes are checked for finite, ordered coordinates
within the image. Colors retain the existing HSV estimation. The VLM does not
provide confidence scores; `--vision-confidence` applies only to YOLO. Normal FPV boxes and labels always
use YOLO; VLM boxes are separate observations used only for goto search/approach.
During a goto mission, the VLM box is drawn over the YOLO boxes in purple,
labeled `VLM: <object>` without confidence. The dashboard shows the goto
detector separately from the YOLO display model.
Detection requests share the worker with describe/VQA.

VLM localization is slower and less precise than a dedicated detector;
[OpenAI documents limitations in precise spatial localization](https://developers.openai.com/api/docs/guides/images-vision#limitations).
Live VLM bounding-box accuracy has not been validated in this project.

The FPV feed labels all detected COCO classes with confidence strictly above
50%. Use `--vision-confidence 0.65` to change the threshold (range 0 to 1), or
pass `confidence_threshold=0.65` to `VisionModule` in Python. Labels include
the color name; boxes and text use the median HSV color inside each bounding
box. This estimate includes background pixels. Neutral colors use saturation
and brightness, and red hues are unwrapped across the hue boundary.

Check that both camera feeds appear, then send a chat request such as
`Move forward at 0.5 m/s for 2 seconds.` The simulation defaults to 300 seconds;
use `--duration 600` to run for ten minutes. **End simulation** or Ctrl+C exits
through the normal cleanup path and saves a timestamped plot under `plot/`.
Closing the browser tab alone does not end the simulation.

For an offline physics/policy smoke test without rendering or cloud requests:

```bash
python play.py --map coco_scene --headless --duration 5
```

If imports fail, check `which python`, activate `quad_mujuco`, and reinstall the
runtime with `python -m pip install -e ../quadruped_mujoco` from `hri_minilab`.
If camera rendering fails, check the graphics driver and select EGL or OSMesa
before starting Python. If chat fails, check the API key, model access and
optional base URL; API errors appear in the chat panel.

## Using the dashboard

The page shows the configured LLM, VLM, and object detection models and contains:

- The existing third-person follow view and robot FPV, streamed side by side.
- Labeled detection boxes in the FPV: green for selected scene classes, gray
  for other detections. No cv2 window is opened.
- A messaging panel showing user requests, queued/thinking status, robot action
  plans and API errors. Enter sends a message; Shift + Enter adds a line.
- A bottom console panel streaming Python stdout/stderr, including `print()`
  output, independently of the chat input. Logs are also printed to the terminal.
- A scrollable generated-actions panel beside the logs, showing every action
  generated since simulation startup, with one JSON line per action and all
  action properties. The currently executing action is highlighted; the highlight
  clears when motion completes or is cancelled. Chat reply text is excluded.
- An **Export Chat** button downloads chat history and generated plans with
  `role`, `text`, `action`, and `estimated_api_cost_usd` columns. Each assistant row contains its full plan
  as JSON and an estimated per-call cost in USD; user rows have empty action and
  cost fields. Costs are exported only, not displayed in the GUI. Export before
  ending the simulation.
- A top-right **End simulation** button that exits and saves the heading plot.
- A **Stop robot** button that cancels the current motion and remaining actions.

Send requests in the chat panel, for example:

- `Move forward at 0.5 m/s for 2 seconds, then turn left 90 degrees.`
- `Turn right 45 degrees, then walk backward for one second.`
- `Stop.`

`DialogueManager.submit_prompt()` queues each browser message. The dialogue
worker appends it to conversation history, sends the full history to the cloud
API and returns a validated action list. Neither input nor API work blocks the
physics thread. Chat history and logs remain available while an API call is
pending. User prompts are not read from the console.

Actions run in order:

```json
{"actions": [
  {"action": "move", "velocity": {"vx": 0.5, "vy": 0.0, "wz": 0.0}, "duration": 2.0},
  {"action": "turn", "angle": 90.0},
  {"action": "stop"}
]}
```

Move velocities use m/s for `vx` and `vy`, rad/s for `wz`. Positive values mean
forward, left and counterclockwise. Turn angles are relative degrees. Turns
complete within `TURN_TOLERANCE_DEG` in `play.py` (default 3 degrees). A zero-speed
move waits for its duration. `stop` cancels remaining actions; `chat` displays a
reply. A new completed plan replaces any unfinished plan. The robot continues
its current plan while the API processes another request. The simulation issues
motion commands only through `motion_api.py`.

Both camera feeds update in the shared low-rate task block. `--low_hz 10`
limits updates to 10 Hz; the default is 20 Hz. Rendering and encoding run on
the simulation thread; HTTP handlers serve stored frames without accessing
MuJoCo data. No new simulation cameras are created.

The dialogue manager has no camera input or object navigation; visual questions
and approach requests produce an explanation rather than an unsupported action.
Malformed, refused or failed API responses are shown in chat without executing
their plans. Ctrl+C stops the simulation and saves heading/angular-velocity plots
under `plot/`; it does not wait for an in-flight cloud request to finish.

Without `--gui`, the simulation runs without display windows. `--headless`
is also available for automated checks. No browser chat is available in that mode.

The API uses [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
with a strict action schema and local validation before execution.

Run offline tests:

```bash
python -m unittest discover -s tests -v
```

### Exported API cost estimates

Costs use the API response's prompt/completion token counts, including conversation
history and reasoning tokens. Cached input and cache-write counts are used when
reported. Estimates are attached to assistant rows, including rejected or malformed
responses when usage is available. User rows and calls with unknown usage/pricing
have blank cost fields; a blank is not a zero-cost call.

The built-in GPT-6 Luna estimate uses standard USD rates recorded on 2026-10-02:
$0.10 input, $0.01 cached input, $0.125 cache writes and $0.50 output per million
tokens, with the published long-context multipliers above 272,000 prompt tokens.
Sources: [GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [OpenAI pricing](https://developers.openai.com/api/docs/pricing).
This is a token-based estimate, not an invoice; taxes, regional premiums, tool
fees and account discounts are excluded. Nonstandard service tiers, other models
and custom base URLs require explicit pricing rather than guessed rates.

For custom pricing, set both `LLM_INPUT_USD_PER_MILLION` and
`LLM_OUTPUT_USD_PER_MILLION`. Optional `LLM_CACHED_INPUT_USD_PER_MILLION` and
`LLM_CACHE_WRITE_USD_PER_MILLION` default to the custom input rate when omitted.
Custom rates are applied directly without automatic long-context multipliers;
configure them for your provider's applicable rate. All rates must be finite and
non-negative. No additional Python packages are needed.

## License

The project code is licensed under the [Apache License 2.0](LICENSE).
Third-party code, pretrained model weights and object meshes retain their
respective licenses; this license does not replace those terms. Map asset
attributions and licenses are listed in [map/ASSET_SOURCES.md](map/ASSET_SOURCES.md).
The `quadruped_mujoco` repository is a separate dependency with its own terms.
