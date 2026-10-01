"""Weight- and provider-free tests for target command parsing."""

import json
import unittest

from minilab.dialogue.llm_parser import LLMTargetParser, OUTPUT_SCHEMA
from minilab.dialogue.parser import parse_target_command, validate_target_command


class DeterministicParserTests(unittest.TestCase):
    def test_supported_phrases(self):
        cases = {
            "Go to the red chair.": ("goto_object", "chair", "red"),
            "Please approach the green chair.": ("goto_object", "chair", "green"),
            "Find the red chair.": ("find", "chair", "red"),
            "Locate the green chair.": ("find", "chair", "green"),
            "Go to the football.": ("goto_object", "sports ball", None),
            "Approach the sports ball.": ("goto_object", "sports ball", None),
            "Find the ball.": ("find", "sports ball", None),
            "Look for a soccer ball.": ("find", "sports ball", None),
            "Head to the red chair.": ("goto_object", "chair", "red"),
            "Move toward a green chair.": ("goto_object", "chair", "green"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                command = parse_target_command(text)
                self.assertEqual(command.status, "accepted")
                self.assertEqual((command.action, command.object_class, command.color), expected)

    def test_chair_without_color_needs_clarification(self):
        for text in ("Go to the chair", "Find a chair", "Find the chair"):
            with self.subTest(text=text):
                command = parse_target_command(text)
                self.assertEqual(command.status, "clarification_required")
                self.assertEqual(command.reason, "ambiguous_target")

    def test_rejections(self):
        cases = {
            "Go to the blue chair": "unsupported_color",
            "Find the table": "unsupported_object",
            "Go to the dog": "unsupported_object",
            "": "malformed",
            "     ": "malformed",
            "asdf qwer !": "unintelligible",
            "Go to the red blue chair": "unsupported_color",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                command = parse_target_command(text)
                self.assertEqual(command.status, "rejected")
                self.assertEqual(command.reason, reason)


class ValidatorTests(unittest.TestCase):
    def proposal(self, **updates):
        value = {
            "status": "accepted", "action": "goto_object",
            "object_class": "chair", "color": "red", "reason": "none",
        }
        value.update(updates)
        return value

    def test_rejects_extra_and_missing_fields(self):
        self.assertEqual(validate_target_command(
            {**self.proposal(), "extra": "execute"}).reason, "malformed")
        missing = self.proposal()
        del missing["reason"]
        self.assertEqual(validate_target_command(missing).reason, "malformed")

    def test_rejects_invalid_enums(self):
        self.assertEqual(validate_target_command(
            self.proposal(action="move")).reason, "malformed")
        self.assertEqual(validate_target_command(
            self.proposal(object_class="table")).reason, "unsupported_object")
        self.assertEqual(validate_target_command(
            self.proposal(color="blue")).reason, "unsupported_color")

    def test_chair_null_color_is_clarification(self):
        result = validate_target_command(self.proposal(color=None))
        self.assertEqual((result.status, result.reason),
                         ("clarification_required", "ambiguous_target"))

    def test_sports_ball_rejects_color(self):
        result = validate_target_command(self.proposal(
            object_class="sports ball", color="red"))
        self.assertEqual((result.status, result.reason),
                         ("rejected", "unsupported_color"))

    def test_accepted_requires_action_and_class(self):
        self.assertEqual(validate_target_command(
            self.proposal(action=None)).reason, "malformed")
        self.assertEqual(validate_target_command(
            self.proposal(object_class=None)).reason, "malformed")


class MockLLMParserTests(unittest.TestCase):
    def test_schema_excludes_additional_properties(self):
        self.assertFalse(OUTPUT_SCHEMA["additionalProperties"])
        self.assertEqual(set(OUTPUT_SCHEMA["required"]), {
            "status", "action", "object_class", "color", "reason"})

    def test_one_schema_correction_retry_then_accept(self):
        responses = ["not json", json.dumps({
            "status": "accepted", "action": "find", "object_class": "chair",
            "color": "green", "reason": "none",
        })]
        calls = []

        def fake_generate(prompt, schema):
            calls.append((prompt, schema))
            return responses.pop(0)

        result = LLMTargetParser(fake_generate).parse("Locate the green chair")
        self.assertEqual((result.status, result.action, result.color),
                         ("accepted", "find", "green"))
        self.assertEqual(len(calls), 2)
        self.assertIn("previous output was invalid", calls[1][0])

    def test_bad_outputs_are_rejected_without_partial_execution(self):
        bad_outputs = [
            "{broken",  # invalid JSON
            json.dumps({  # extra key
                "status": "accepted", "action": "goto_object",
                "object_class": "chair", "color": "red", "reason": "none",
                "exec": "move forward",
            }),
        ]

        for first in bad_outputs:
            with self.subTest(first=first):
                calls = iter([first, first])
                result = LLMTargetParser(lambda prompt, schema: next(calls)).parse("go chair")
                self.assertEqual((result.status, result.reason),
                                 ("rejected", "malformed"))
                self.assertIsNone(result.action)

    def test_semantically_ambiguous_llm_output_is_clarification(self):
        response = json.dumps({
            "status": "accepted", "action": "find", "object_class": "chair",
            "color": None, "reason": "none",
        })
        result = LLMTargetParser(lambda prompt, schema: response).parse("find chair")
        self.assertEqual((result.status, result.reason),
                         ("clarification_required", "ambiguous_target"))


if __name__ == "__main__":
    unittest.main()
