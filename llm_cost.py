"""Estimate standard text-generation cost from provider-reported token usage."""

import math
import os

# USD per million tokens; verified 2026-10-02:
# https://developers.openai.com/api/docs/models/gpt-6-luna
LUNA_RATES = (0.10, 0.01, 0.125, 0.50)
RATE_ENV = (
    "LLM_INPUT_USD_PER_MILLION",
    "LLM_CACHED_INPUT_USD_PER_MILLION",
    "LLM_CACHE_WRITE_USD_PER_MILLION",
    "LLM_OUTPUT_USD_PER_MILLION",
)


def configured_rates(model):
    """Return (rates, apply_luna_long_context), or None for unknown pricing."""
    values = [os.environ.get(name) for name in RATE_ENV]
    if any(value is not None for value in values):
        if values[0] is None or values[3] is None:
            raise ValueError("Custom LLM pricing requires input and output rates")
        rates = tuple(
            float(value if value is not None else values[0]) for value in values
        )
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("LLM pricing rates must be finite and non-negative")
        return rates, False
    if model == "gpt-6-luna" and not os.environ.get("OPENAI_BASE_URL"):
        return LUNA_RATES, True
    return None


def estimate_cost(completion, pricing):
    """Return estimated USD, or None when usage/pricing cannot be determined.

    Includes the full prompt (history/schema), cached input, cache writes and
    all completion tokens (including reasoning). No local token counting.
    """
    if pricing is None:
        return None
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    output = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    writes = getattr(details, "cache_write_tokens", 0) or 0
    counts = (prompt, output, cached, writes)
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    if cached + writes > prompt:
        return None
    rates, long_context = pricing
    if long_context and getattr(completion, "service_tier", None) not in {
        None,
        "auto",
        "default",
    }:
        return None
    input_rate, cached_rate, write_rate, output_rate = rates
    if long_context and prompt > 272000:
        input_rate *= 2
        cached_rate *= 2
        write_rate *= 2
        output_rate *= 1.5
    return (
        (prompt - cached - writes) * input_rate
        + cached * cached_rate
        + writes * write_rate
        + output * output_rate
    ) / 1_000_000
