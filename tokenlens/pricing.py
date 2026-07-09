"""Model pricing and cost estimation for Anthropic models.

Rates are USD per 1,000,000 tokens, cached 2026-06-24 from the Anthropic
pricing table. Update these as pricing changes. Cache-read and cache-write
multipliers follow Anthropic's published prompt-caching economics.
"""

from __future__ import annotations

# model-id prefix -> (input $/MTok, output $/MTok)
_RATES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Prompt-caching multipliers relative to the base input rate.
_CACHE_READ_MULT = 0.10   # cached tokens served at ~0.1x input
_CACHE_WRITE_MULT = 1.25  # 5-minute TTL cache writes at ~1.25x input

_PER_TOKEN = 1_000_000.0


def rates_for(model: str) -> tuple[float, float] | None:
    """Return (input_rate, output_rate) per MTok for a model id, or None.

    Uses longest-prefix match so dated/suffixed ids still resolve.
    """
    best: str | None = None
    for prefix in _RATES:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return _RATES[best] if best else None


def cost_usd(model: str, usage: dict) -> float | None:
    """Estimate the USD cost of a single response from its `usage` block.

    `usage` is Anthropic's usage object with any of:
      input_tokens, output_tokens,
      cache_read_input_tokens, cache_creation_input_tokens
    Returns None if the model is unknown.
    """
    rates = rates_for(model)
    if rates is None:
        return None
    in_rate, out_rate = rates

    uncached_in = usage.get("input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0

    total = (
        uncached_in * in_rate
        + cache_read * in_rate * _CACHE_READ_MULT
        + cache_write * in_rate * _CACHE_WRITE_MULT
        + out * out_rate
    )
    return total / _PER_TOKEN
