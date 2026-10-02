"""Validate pricing calculations and per-request export attribution offline."""

import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import OpenAIError

from dialogue_manager import DialogueManager
from llm_cost import LUNA_RATES, configured_rates, estimate_cost


def response(prompt=1000, output=200, cached=100, writes=200, refusal=None):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=output,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached, cache_write_tokens=writes
            ),
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    refusal=refusal,
                    content=json.dumps({"actions": [{"action": "stop"}]}),
                ),
            )
        ],
    )


class CostTests(unittest.TestCase):
    def test_standard_and_long_context_pricing(self):
        pricing = (LUNA_RATES, True)
        self.assertAlmostEqual(estimate_cost(response(), pricing), 0.000196)
        self.assertAlmostEqual(
            estimate_cost(response(prompt=272001, cached=0, writes=0), pricing),
            (272001 * 0.20 + 200 * 0.75) / 1_000_000,
        )
        self.assertAlmostEqual(
            estimate_cost(response(prompt=272000, cached=0, writes=0), pricing),
            (272000 * 0.10 + 200 * 0.50) / 1_000_000,
        )

    def test_unknown_or_invalid_usage_is_not_zero_cost(self):
        for item in [
            SimpleNamespace(),
            response(prompt=-1),
            response(cached=1001),
            response(prompt=1.5),
        ]:
            self.assertIsNone(estimate_cost(item, (LUNA_RATES, True)))
        self.assertIsNone(estimate_cost(response(), None))
        item = response()
        item.service_tier = "flex"
        self.assertIsNone(estimate_cost(item, (LUNA_RATES, True)))
        self.assertEqual(estimate_cost(response(0, 0, 0, 0), (LUNA_RATES, True)), 0)

    def test_missing_cache_details_use_uncached_rate(self):
        item = response()
        item.usage.prompt_tokens_details = None
        self.assertAlmostEqual(estimate_cost(item, (LUNA_RATES, True)), 0.0002)

    def test_custom_pricing_and_unknown_models(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_rates("gpt-6-luna"), (LUNA_RATES, True))
            self.assertIsNone(configured_rates("another-model"))
            os.environ["OPENAI_BASE_URL"] = "https://example.test/v1"
            self.assertIsNone(configured_rates("gpt-6-luna"))
            os.environ.update(
                LLM_INPUT_USD_PER_MILLION="2", LLM_OUTPUT_USD_PER_MILLION="4"
            )
            self.assertEqual(configured_rates("another-model"), ((2, 2, 2, 4), False))
            self.assertAlmostEqual(
                estimate_cost(response(), configured_rates("another-model")), 0.0028
            )
            os.environ["LLM_CACHED_INPUT_USD_PER_MILLION"] = "nan"
            with self.assertRaises(ValueError):
                configured_rates("another-model")
        with (
            patch.dict(os.environ, {"LLM_INPUT_USD_PER_MILLION": "1"}, clear=True),
            self.assertRaises(ValueError),
        ):
            configured_rates("another-model")

    def test_worker_attributes_cost_to_reply_including_refusal(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response(),
            response(refusal="No"),
            OpenAIError("network failed"),
        ]
        with patch.dict(os.environ, {}, clear=True):
            manager = DialogueManager(client=client)
        self.addCleanup(manager.close)
        manager.start()
        manager.submit_prompt("stop")
        manager._plans.get(timeout=1)
        manager.submit_prompt("refused request")
        manager.submit_prompt("failed request")
        deadline = time.monotonic() + 2
        while len(manager.export_snapshot()) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        rows = manager.export_snapshot()
        self.assertEqual(len(rows), 6)
        assistant = [row for row in rows if row["role"] == "assistant"]
        self.assertAlmostEqual(assistant[0]["estimated_api_cost_usd"], 0.000196)
        self.assertAlmostEqual(assistant[1]["estimated_api_cost_usd"], 0.000196)
        self.assertIsNone(assistant[2]["estimated_api_cost_usd"])
        self.assertTrue(
            all(
                row["estimated_api_cost_usd"] is None
                for row in rows
                if row["role"] == "user"
            )
        )
        self.assertTrue(
            all("estimated_api_cost_usd" not in row for row in manager.chat_snapshot())
        )
