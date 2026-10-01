# Robot dog browser dashboard

Run from this directory in the `quad_mujuco` conda environment:

```bash
conda activate quad_mujuco
python play.py --map coco_scene --gui
```

The dashboard opens automatically at `http://127.0.0.1:8765`. Use
`--gui-port 8766` to change the port. `OPENAI_API_KEY` must be set in the
OS environment. The dialogue integration requires the `openai` Python package
(`pip install openai`). `OPENAI_MODEL` defaults to `gpt-6-luna`; the SDK also
accepts `OPENAI_BASE_URL` for an OpenAI-compatible cloud endpoint.

The page contains:

- The existing third-person follow view and robot FPV, streamed side by side.
- Labeled detection boxes in the FPV: green for selected scene classes, gray
  for other detections. No cv2 window is opened.
- A messaging panel showing user requests, queued/thinking status, robot action
  plans and API errors. Enter sends a message; Shift + Enter adds a line.
- A bottom console panel streaming Python stdout/stderr, including `print()`
  output, independently of the chat input. Logs are also printed to the terminal.
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
