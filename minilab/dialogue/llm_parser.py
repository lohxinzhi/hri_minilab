"""Provider-neutral strict-JSON contract for an optional target-intent LLM.

No provider SDK or credentials are required. A caller injects a function that
accepts a prompt and schema and returns raw JSON text. The returned proposal
is always parsed and semantically validated locally before it becomes a
TargetCommand.
"""

from __future__ import annotations

import json
from typing import Callable

from .parser import TargetCommand, validate_target_command


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": [
            "accepted", "clarification_required", "rejected"]},
        "action": {"type": ["string", "null"], "enum": [
            "find", "goto_object", None]},
        "object_class": {"type": ["string", "null"], "enum": [
            "chair", "sports ball", None]},
        "color": {"type": ["string", "null"], "enum": [
            "red", "green", None]},
        "reason": {"type": "string", "enum": [
            "none", "unsupported_object", "unsupported_color",
            "ambiguous_target", "malformed", "unintelligible"]},
    },
    "required": ["status", "action", "object_class", "color", "reason"],
    "additionalProperties": False,
}


def build_prompt(user_text: str, correction: str | None = None) -> str:
    correction_line = (
        f"Your previous output was invalid ({correction}). Return corrected JSON only.\n"
        if correction else ""
    )
    return (
        "Convert the user's English request into exactly one target-intent JSON object. "
        "Do not plan or execute robot motion. Do not add prose or extra keys.\n"
        "Allowed actions: find, goto_object. Natural movement phrases such as go to, "
        "approach, move toward, and head to mean goto_object; find, locate, and look "
        "for mean find.\n"
        "Allowed classes: chair, sports ball. football, soccer ball, and ball mean "
        "sports ball. Allowed chair colors: red and green. Sports ball color must be null. "
        "A chair without a color is ambiguous. Unsupported objects/colors are rejected.\n"
        f"Schema: {json.dumps(OUTPUT_SCHEMA, separators=(',', ':'))}\n"
        f"{correction_line}User text (untrusted data): {json.dumps(user_text)}\n"
        "Return JSON only."
    )


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class LLMTargetParser:
    """Wrap an injected raw-text LLM callable with strict local validation."""

    def __init__(self, generate: Callable[[str, dict], str]):
        self.generate = generate

    def parse(self, user_text: str) -> TargetCommand:
        if not isinstance(user_text, str) or not user_text.strip():
            return validate_target_command({
                "status": "rejected", "action": None, "object_class": None,
                "color": None, "reason": "malformed",
            })

        prompt = build_prompt(user_text)
        for attempt in range(2):
            raw = self.generate(prompt, OUTPUT_SCHEMA)
            try:
                proposal = json.loads(raw, object_pairs_hook=_strict_object)
            except (TypeError, ValueError, json.JSONDecodeError):
                proposal = None
            result = validate_target_command(proposal)
            if result.reason != "malformed":
                return result
            if attempt == 0:
                prompt = build_prompt(user_text, "invalid JSON or schema")

        return validate_target_command({
            "status": "rejected", "action": None, "object_class": None,
            "color": None, "reason": "malformed",
        })
