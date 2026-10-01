"""Deterministic parser and authoritative validator for target commands.

This module produces descriptions only. It has no simulator, perception,
ground-truth, or motion-control dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


STATUSES = {"accepted", "clarification_required", "rejected"}
ACTIONS = {"find", "goto_object"}
OBJECT_CLASSES = {"chair", "sports ball"}
COLORS = {"red", "green"}
REASONS = {
    "none", "unsupported_object", "unsupported_color", "ambiguous_target",
    "malformed", "unintelligible",
}
COMMAND_FIELDS = {"status", "action", "object_class", "color", "reason"}


@dataclass(frozen=True)
class TargetCommand:
    """Validated target intent; it does not represent executable robot control."""

    status: str
    action: str | None
    object_class: str | None
    color: str | None
    reason: str


def _result(status: str, reason: str) -> TargetCommand:
    return TargetCommand(status, None, None, None, reason)


def validate_target_command(proposal: Mapping[str, Any] | Any) -> TargetCommand:
    """Validate exact command fields and enforce scene-specific semantics.

    Unexpected/missing fields never pass. Values are normalized only for
    schema enum comparison; arbitrary strings are not forwarded to consumers.
    """
    if not isinstance(proposal, Mapping) or set(proposal) != COMMAND_FIELDS:
        return _result("rejected", "malformed")

    status = proposal.get("status")
    action = proposal.get("action")
    object_class = proposal.get("object_class")
    color = proposal.get("color")
    reason = proposal.get("reason")

    if (not isinstance(status, str) or status not in STATUSES
            or not isinstance(reason, str) or reason not in REASONS):
        return _result("rejected", "malformed")
    if action is not None and (not isinstance(action, str) or action not in ACTIONS):
        return _result("rejected", "malformed")
    if object_class is not None and (
            not isinstance(object_class, str) or object_class not in OBJECT_CLASSES):
        return _result("rejected", "unsupported_object")
    if color is not None and (not isinstance(color, str) or color not in COLORS):
        return _result("rejected", "unsupported_color")

    if status == "accepted":
        if action not in ACTIONS or object_class not in OBJECT_CLASSES:
            return _result("rejected", "malformed")
        if object_class == "chair" and color is None:
            return TargetCommand("clarification_required", action, "chair", None,
                                "ambiguous_target")
        if object_class == "sports ball" and color is not None:
            return _result("rejected", "unsupported_color")
        if reason != "none":
            return _result("rejected", "malformed")
        return TargetCommand("accepted", action, object_class, color, "none")

    if status == "clarification_required":
        if reason == "none":
            reason = "ambiguous_target"
        return TargetCommand("clarification_required", action, object_class,
                             color, reason)
    if reason == "none":
        return _result("rejected", "malformed")
    return TargetCommand("rejected", action, object_class, color, reason)


_ACTION_PATTERNS = (
    (re.compile(r"\b(?:go\s+over\s+to|go\s+to|head\s+to|move\s+toward(?:s)?|approach)\b"),
     "goto_object"),
    (re.compile(r"\b(?:look\s+for|locate|find)\b"), "find"),
)
_TARGET_ALIASES = (
    (re.compile(r"\b(?:sports\s+ball|soccer\s+ball|football|ball)\b"),
     "sports ball"),
    (re.compile(r"\bchair\b"), "chair"),
)
_SUPPORTED_COLOR = re.compile(r"\b(red|green)\b")
_OTHER_COLOR = re.compile(
    r"\b(blue|yellow|orange|purple|violet|pink|black|white|gray|grey|brown)\b"
)
_ARTICLES_AND_POLITENESS = re.compile(
    r"\b(?:please|can|could|would|you|the|a|an|to|over)\b"
)


def parse_target_command(text: str) -> TargetCommand:
    """Parse one obvious English target phrase into a validated command.

    Supported target synonyms: football, soccer ball, and ball map to
    ``sports ball``. This intentionally recognizes a small phrase grammar;
    paraphrases outside it are rejected rather than guessed.
    """
    if not isinstance(text, str) or not text.strip():
        return _result("rejected", "malformed")

    normalized = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return _result("rejected", "malformed")

    action = None
    target_text = None
    for pattern, candidate_action in _ACTION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            action = candidate_action
            target_text = normalized[match.end():].strip()
            break
    if action is None or not target_text:
        return _result("rejected", "unintelligible")

    # Strip only harmless trailing conversational punctuation/words. Remaining
    # unknown nouns are handled as unsupported targets below.
    target_text = re.sub(r"\b(?:please|now|for\s+me)\b", " ", target_text)
    target_text = re.sub(r"\s+", " ", target_text).strip()
    target_matches = [(match.start(), match.end(), name)
                      for pattern, name in _TARGET_ALIASES
                      for match in pattern.finditer(target_text)]
    if len(target_matches) != 1:
        return _result("rejected", "unsupported_object")
    start, end, object_class = target_matches[0]
    prefix = target_text[:start]
    suffix = target_text[end:]
    # A single target with unrecognized modifiers is not silently interpreted.
    remainder = _ARTICLES_AND_POLITENESS.sub(" ", prefix + " " + suffix)
    remainder = _SUPPORTED_COLOR.sub(" ", remainder)
    remainder = _OTHER_COLOR.sub(" ", remainder)
    if re.sub(r"\s+", " ", remainder).strip():
        return _result("rejected", "unsupported_object")

    unsupported_color = _OTHER_COLOR.search(target_text)
    supported_colors = _SUPPORTED_COLOR.findall(target_text)
    if unsupported_color:
        return _result("rejected", "unsupported_color")
    if len(set(supported_colors)) > 1:
        return _result("rejected", "malformed")
    color = supported_colors[0] if supported_colors else None
    proposal = {
        "status": "accepted",
        "action": action,
        "object_class": object_class,
        "color": color,
        "reason": "none",
    }
    return validate_target_command(proposal)
