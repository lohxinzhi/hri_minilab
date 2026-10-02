"""Translate browser chat requests into validated action plans on a worker thread."""

import json
import math
import os
import threading
from copy import deepcopy
from queue import Empty, Queue

from openai import OpenAI, OpenAIError

from llm_cost import configured_rates, estimate_cost

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
OBJECT_COLORS = [
    "any",
    "red",
    "orange",
    "brown",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "white",
    "gray",
    "black",
]

SYSTEM_PROMPT = (
    """You are the dialogue manager of a robot dog in MuJoCo.
Convert the latest request into a JSON object containing an ordered actions list.
Only generate the actions needed for this request; do not repeat earlier plans.
Allowed actions:
- {"action":"move","velocity":{"vx":0.0,"vy":0.0,"wz":0.0},"duration":1.0}
- {"action":"turn","angle":90.0}
- {"action":"goto","object_type":"chair","object_color":"red"}
- {"action": "describe", "mode": "describe"}
- {"action": "describe", "mode": "vqa", "question": "<specific question about the current view>"}
- {"action":"stop"}
- {"action":"chat","reply":"Your answer or clarification"}
vx and vy are m/s; wz is rad/s. Positive vx is forward, positive vy is left,
and positive wz is counterclockwise. Turns are RELATIVE degrees, positive left,
negative right. Moves last for duration seconds (default 1 second when unspecified).
Use ordinary walking speeds, normally at most 1 m/s and 1 rad/s, unless specified.
Actions run sequentially; stop cancels the remainder of the list. A new plan
replaces any unfinished earlier plan only when it contains a motion or stop
action. A plan containing only chat and/or describe leaves the current motion task running. Ask a
clarification using chat when needed.
Use goto for requests to go to, approach, or find an object. Map the requested
object to exactly one supported COCO class: e.g. bike -> bicycle, sofa -> couch,
Toyota Supra or Lamborghini -> car. Reject unmappable objects (e.g. a door)
with a chat reply explaining the reason; never invent a class or approximate
an unrelated object. Ask for clarification through chat when mapping is ambiguous.
Use object_color="any" if the user does not specify a color. Normalize grey to
gray and light/dark color descriptions to the supported base color. Reject
unsupported or ambiguous colors with chat rather than inventing a color.
goto uses live vision to search one revolution and approach the matched object,
with a configured mission timeout. It does not accept coordinates or a fabricated bbox.
You have no camera images: do not invent current observations or locations
Use describe mode for a general account of the current view.
Use vqa mode for a specific question about what the robot can see;
preserve the question in the question field. Do not answer visual
questions from memory with chat. Return JSON only.
Supported COCO classes: """
    + ", ".join(COCO_CLASSES)
    + "\nSupported colors: "
    + ", ".join(OBJECT_COLORS)
)


def _object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


ACTION_SCHEMA = _object_schema(
    {
        "actions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "anyOf": [
                    _object_schema(
                        {
                            "action": {"type": "string", "enum": ["move"]},
                            "velocity": _object_schema(
                                {key: {"type": "number"} for key in ("vx", "vy", "wz")}
                            ),
                            "duration": {"type": "number", "minimum": 0},
                        }
                    ),
                    _object_schema(
                        {
                            "action": {"type": "string", "enum": ["turn"]},
                            "angle": {"type": "number"},
                        }
                    ),
                    _object_schema({"action": {"type": "string", "enum": ["stop"]}}),
                    _object_schema(
                        {
                            "action": {"type": "string", "enum": ["goto"]},
                            "object_type": {"type": "string", "enum": COCO_CLASSES},
                            "object_color": {"type": "string", "enum": OBJECT_COLORS},
                        }
                    ),
                    _object_schema(
                        {
                            "action": {"type": "string", "enum": ["chat"]},
                            "reply": {"type": "string"},
                        }
                    ),
                    _object_schema(
                        {
                            "action": {"type": "string", "enum": ["describe"]},
                            "mode": {"type": "string", "enum": ["describe"]},
                        }
                    ),
                    _object_schema(
                        {
                            "action": {"type": "string", "enum": ["describe"]},
                            "mode": {"type": "string", "enum": ["vqa"]},
                            "question": {"type": "string"},
                        }
                    ),
                ]
            },
        }
    }
)


def validate_actions(actions):
    """Reject malformed plans before any action can be executed."""
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("action values must be finite numbers")
        if not math.isfinite(value):
            raise ValueError("action values must be finite numbers")

    for item in actions:
        if not isinstance(item, dict):
            raise TypeError("each action must be an object")
        kind = item.get("action")
        fields = {
            "move": {"action", "velocity", "duration"},
            "turn": {"action", "angle"},
            "stop": {"action"},
            "chat": {"action", "reply"},
            "goto": {"action", "object_type", "object_color"},
            "describe": {"action", "mode"}
            | ({"question"} if item.get("mode") == "vqa" else set()),
        }
        if not isinstance(kind, str) or kind not in fields or set(item) != fields[kind]:
            raise ValueError("unsupported action or invalid action fields")
        if kind == "move":
            velocity = item["velocity"]
            if not isinstance(velocity, dict) or set(velocity) != {"vx", "vy", "wz"}:
                raise ValueError("move velocity must contain vx, vy, wz")
            for value in velocity.values():
                number(value)
            number(item["duration"])
            if item["duration"] < 0:
                raise ValueError("move duration must be non-negative")
        elif kind == "turn":
            number(item["angle"])
        elif kind == "goto":
            if (
                not isinstance(item["object_type"], str)
                or item["object_type"] not in COCO_CLASSES
            ):
                raise ValueError("goto object_type must be a supported COCO class")
            if (
                not isinstance(item["object_color"], str)
                or item["object_color"] not in OBJECT_COLORS
            ):
                raise ValueError("goto object_color must be a supported color or any")
        elif kind == "describe":
            if item["mode"] not in ("describe", "vqa"):
                raise ValueError("describe mode must be describe or vqa")
            if item["mode"] == "vqa" and (
                not isinstance(item["question"], str) or not item["question"].strip()
            ):
                raise ValueError("vqa question must be nonempty text")
        elif kind == "chat" and (
            not isinstance(item["reply"], str) or not item["reply"].strip()
        ):
            raise ValueError("chat reply must be nonempty text")
    return deepcopy(actions)


class DialogueManager:
    """Keep conversation history and process submitted prompts off the main loop."""

    def __init__(self, model=None, *, client=None):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-6-luna")
        self.pricing = configured_rates(self.model)
        self._last_api_cost = None
        self.client = client
        self._owns_client = client is None
        self._history_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._plans = Queue()
        self._errors = Queue()
        self._requests = Queue()
        self._chat_lock = threading.Lock()
        self._chat = []
        self._generated_actions = []
        self._plan_offsets = {}
        self._plan_request_ids = {}
        self._export_actions = {}
        self._export_costs = {}
        self._next_id = 1

    def submit_prompt(self, prompt):
        """Queue a browser prompt without waiting for the cloud response."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if len(prompt) > 8000:
            raise ValueError("prompt must be at most 8000 characters")
        with self._chat_lock:
            if self._stop.is_set():
                raise RuntimeError("dialogue manager is closed")
            request_id = self._next_id
            self._next_id += 1
            self._chat.append(
                {
                    "id": request_id,
                    "role": "user",
                    "content": prompt.strip(),
                    "status": "queued",
                }
            )
            self._requests.put((request_id, prompt.strip()))
        return request_id

    def chat_snapshot(self):
        """Copy chat messages without acquiring the API/history lock."""
        with self._chat_lock:
            return deepcopy(self._chat)

    def export_snapshot(self):
        """Copy chat and the full generated plan for each assistant reply."""
        with self._chat_lock:
            return [
                {
                    "role": message["role"],
                    "text": message["content"],
                    "estimated_api_cost_usd": message.get("estimated_cost")
                    if message.get("kind") == "visual_response"
                    else self._export_costs.get(message["id"])
                    if message["role"] == "assistant"
                    and message.get("kind") not in {"task_result", "visual_response"}
                    else None,
                    "action": deepcopy(self._export_actions.get(message["id"], []))
                    if message["role"] == "assistant"
                    and message.get("kind") not in {"task_result", "visual_response"}
                    else []
                    if message["role"] == "assistant"
                    else None,
                }
                for message in self._chat
            ]

    def action_history_start(self, actions):
        """Consume the history offset associated with a queued plan."""
        with self._chat_lock:
            return self._plan_offsets.pop(id(actions), None)

    def actions_snapshot(self):
        """Return all generated actions since startup with chat replies omitted."""
        with self._chat_lock:
            return deepcopy(self._generated_actions)

    def action_request_id(self, actions):
        """Consume the prompt ID associated with this exact queued plan."""
        with self._chat_lock:
            return self._plan_request_ids.pop(id(actions), None)

    def update_task_status(self, request_id, status, *, action=None, reason=None):
        """Update both chat entries as the simulation executes a prompt's plan."""
        if status not in {"executing", "completed", "failed", "cancelled"}:
            raise ValueError("unsupported task status")
        with self._chat_lock:
            changed = False
            for message in self._chat:
                if message["id"] == request_id and message["status"] in {
                    "planned",
                    "executing",
                }:
                    message["status"] = status
                    changed = True
            actions = self._export_actions.get(request_id, [])
            if (
                changed
                and status in {"completed", "failed", "cancelled"}
                and any(item["action"] != "chat" for item in actions)
            ):
                content = self._task_result(
                    actions, status, action=action, reason=reason
                )
                if not content:
                    return
                self._chat.append(
                    {
                        "id": request_id,
                        "role": "assistant",
                        "kind": "task_result",
                        "content": content,
                        "status": status,
                    }
                )

    @staticmethod
    def _task_result(actions, status, *, action=None, reason=None):
        """Describe execution outcomes without another model request."""
        if status == "cancelled":
            return "Task cancelled before all actions were completed."
        if status == "failed":
            action = action or next(
                (item for item in actions if item["action"] == "goto"), None
            )
            if action is not None and action["action"] == "goto":
                color = action["object_color"]
                target = (
                    action["object_type"]
                    if color == "any"
                    else f"{color} {action['object_type']}"
                )
                if reason and "not found" in reason:
                    return f"Failed to find the {target} after a full search."
                if reason and "timeout" in reason:
                    limit = reason.removesuffix("-second timeout")
                    return f"Failed to find or approach the {target}: timed out after {limit} seconds."
                if reason and "position" in reason:
                    return f"Failed to approach the {target}: its position is unavailable in this map."
                return f"Failed to approach the {target}: {reason or 'mission unsuccessful'}."
            return f"Failed to complete the task: {reason or 'execution unsuccessful'}."
        lines = []
        for item in actions:
            kind = item["action"]
            if kind == "turn":
                angle = item["angle"]
                direction = "left" if angle > 0 else "right"
                lines.append(
                    f"Successfully turned {abs(angle):g} degrees to the {direction}."
                    if angle
                    else "Completed the zero-degree turn."
                )
            elif kind == "move":
                verb = "moved" if any(item["velocity"].values()) else "waited"
                lines.append(f"Successfully {verb} for {item['duration']:g} seconds.")
            elif kind == "goto":
                color = item["object_color"]
                target = (
                    item["object_type"]
                    if color == "any"
                    else f"{color} {item['object_type']}"
                )
                lines.append(f"Successfully approached the {target}.")
            elif kind == "stop":
                lines.append("Successfully stopped.")
                break
        return "\n".join(lines)

    def add_visual_response(self, request_id, content, estimated_cost=None):
        """Save a VLM answer in chat and the planner's conversational history.

        Called by the vision worker, never the simulation thread.
        """
        with self._history_lock:
            self.history.append({"role": "assistant", "content": content})
        with self._chat_lock:
            status = next(
                (
                    message["status"]
                    for message in self._chat
                    if message["id"] == request_id
                ),
                "executing",
            )
            self._chat.append(
                {
                    "id": request_id,
                    "role": "assistant",
                    "kind": "visual_response",
                    "content": content,
                    "status": status,
                    "estimated_cost": estimated_cost,
                }
            )

    def _reply(self, request_id, content, status, actions=None, estimated_cost=None):
        with self._chat_lock:
            self._export_costs[request_id] = estimated_cost
            if actions is not None:
                self._export_actions[request_id] = deepcopy(actions)
                self._plan_offsets[id(actions)] = len(self._generated_actions)
                self._plan_request_ids[id(actions)] = request_id
                self._generated_actions.extend(
                    deepcopy(
                        [
                            {"action": "chat"} if action["action"] == "chat" else action
                            for action in actions
                        ]
                    )
                )
            for message in self._chat:
                if message["id"] == request_id and message["role"] == "user":
                    message["status"] = status
                    break
            if not content:
                return
            self._chat.append(
                {
                    "id": request_id,
                    "role": "assistant",
                    "content": content,
                    "status": status,
                }
            )

    @staticmethod
    def _summarize(actions):
        lines = []
        for action in actions:
            kind = action["action"]
            if kind == "chat":
                lines.append(action["reply"])
            elif kind == "move":
                v = action["velocity"]
                lines.append(
                    f"Move at vx={v['vx']:g}, vy={v['vy']:g} m/s, wz={v['wz']:g} rad/s for {action['duration']:g} s."
                )
            elif kind == "turn":
                lines.append(
                    f"Turn {action['angle']:g}° relative to the current heading."
                )
            elif kind == "goto":
                color = action["object_color"]
                target = (
                    action["object_type"]
                    if color == "any"
                    else f"{color} {action['object_type']}"
                )
                lines.append(f"Search for and approach {target}.")
            elif kind == "describe":
                continue
            else:
                lines.append("Stop and cancel remaining actions.")
                break
        return "\n".join(lines)

    def user_cmd(self, user_prompt):
        """Append a prompt to history, call the API and return validated actions.

        start() runs this on a daemon worker; browser callers use submit_prompt().
        OPENAI_API_KEY and optional OPENAI_BASE_URL configure the SDK client.
        """
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user prompt must be nonempty text")
        with self._command_lock:
            self._last_api_cost = None
            with self._history_lock:
                self.history.append({"role": "user", "content": user_prompt.strip()})
                messages = deepcopy(self.history)
            if self.client is None:
                self.client = OpenAI(timeout=30.0, max_retries=0)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "robot_actions",
                        "strict": True,
                        "schema": ACTION_SCHEMA,
                    },
                },
            )
            self._last_api_cost = estimate_cost(completion, self.pricing)
            choice = completion.choices[0]
            if choice.finish_reason != "stop" or choice.message.refusal:
                raise ValueError("LLM response was incomplete or refused")
            payload = json.loads(choice.message.content)
            if not isinstance(payload, dict) or set(payload) != {"actions"}:
                raise ValueError("LLM response must contain only an actions list")
            actions = validate_actions(payload["actions"])
            with self._history_lock:
                self.history.append(
                    {"role": "assistant", "content": json.dumps({"actions": actions})}
                )
            return actions

    def start(self):
        """Start background API processing; browser submissions remain nonblocking."""
        if self._thread is not None:
            raise RuntimeError("dialogue worker already started")
        self._thread = threading.Thread(
            target=self._run, name="robot-dialogue", daemon=True
        )
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    request = self._requests.get(timeout=0.1)
                except Empty:
                    continue
                if request is None or self._stop.is_set():
                    break
                request_id, prompt = request
                with self._chat_lock:
                    for message in self._chat:
                        if message["id"] == request_id and message["role"] == "user":
                            message["status"] = "thinking"
                            break
                try:
                    actions = self.user_cmd(prompt)
                except (OpenAIError, ValueError, TypeError, IndexError) as exc:
                    if not self._stop.is_set():
                        error = f"{type(exc).__name__}: {exc}"
                        print(
                            f"[CMD] rejected reasons={json.dumps([error], ensure_ascii=False)}"
                        )
                        self._errors.put(error)
                        self._reply(
                            request_id,
                            error,
                            "error",
                            estimated_cost=self._last_api_cost,
                        )
                    continue
                if not self._stop.is_set():
                    if all(action["action"] == "chat" for action in actions):
                        reasons = [action["reply"] for action in actions]
                        print(
                            f"[CMD] rejected reasons={json.dumps(reasons, ensure_ascii=False)}"
                        )
                    else:
                        print(
                            f"[CMD] actions={json.dumps(actions, ensure_ascii=False)} "
                            f"n={len(actions)}"
                        )
                    self._reply(
                        request_id,
                        self._summarize(actions),
                        "planned",
                        actions=actions,
                        estimated_cost=self._last_api_cost,
                    )
                    self._plans.put(actions)
        finally:
            if self._owns_client and self.client is not None:
                self.client.close()

    def poll_actions(self):
        """Return the next action list, or None immediately if no reply is ready."""
        try:
            return self._plans.get_nowait()
        except Empty:
            return None

    def poll_error(self):
        """Return the next API error without blocking the simulation."""
        try:
            return self._errors.get_nowait()
        except Empty:
            return None

    def close(self):
        """Stop accepting replies; bounded join lets Ctrl+C exit during an API call."""
        self._stop.set()
        self._requests.put(None)
        if self._thread is not None:
            self._thread.join(timeout=0.2)
