"""Rung 5: LLMLingua-2 token pruning (optional).

LLMLingua-2 is a small (~500M) token-classification model that scores each
token's information content and drops the low-value ones. It is CPU-capable and
purpose-built for prompt compression.

This module lazy-imports `llmlingua`; if it (or torch) is not installed, the
scorer is unavailable and the pipeline falls back to the safe floor. Nothing
here is imported at proxy startup, so the base proxy keeps its zero-dependency
footprint.

NOTE: model quality on real traffic must be verified on-machine — this code
follows the documented `llmlingua.PromptCompressor` API but is not exercised in
the dependency-free test suite.
"""

from __future__ import annotations

import threading

_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

_lock = threading.Lock()
_compressor = None
_load_error: Exception | None = None


def available() -> bool:
    """True if the LLMLingua-2 scorer can be loaded."""
    return _get_compressor() is not None


def _get_compressor():
    global _compressor, _load_error
    if _compressor is not None or _load_error is not None:
        return _compressor
    with _lock:
        if _compressor is not None or _load_error is not None:
            return _compressor
        try:
            from llmlingua import PromptCompressor  # type: ignore

            _compressor = PromptCompressor(
                model_name=_MODEL,
                use_llmlingua2=True,
            )
        except Exception as e:  # torch/llmlingua missing, download failure, etc.
            _load_error = e
            _compressor = None
    return _compressor


def load_error() -> Exception | None:
    return _load_error


def compress(text: str, rate: float) -> str:
    """Prune `text` to roughly `rate` of its tokens (0<rate<=1).

    Returns the pruned prose, or the original text if the scorer is unavailable
    or errors (fail-open — never lose content on a compression failure).
    """
    comp = _get_compressor()
    if comp is None:
        return text
    try:
        result = comp.compress_prompt(text, rate=rate, force_tokens=["\n"])
        return result.get("compressed_prompt", text) or text
    except Exception:
        return text
