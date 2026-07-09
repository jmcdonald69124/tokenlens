"""Rungs 0-2: safe, deterministic, model-free compression of prose.

These transforms are near-lossless and cache-neutral (we only ever apply them
to the volatile tail, never the cached prefix). They are the always-on floor.
"""

from __future__ import annotations

import re

_TRAILING_WS = re.compile(r"[ \t]+(\n|$)")
_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")
_MANY_NEWLINES = re.compile(r"\n{3,}")
# Decorative separator lines like ==== or ---- or ****
_DECORATIVE = re.compile(r"(?m)^[ \t]*([=*_\-]){6,}[ \t]*$")


def safe_compress(text: str) -> str:
    """Rung 0-1: normalize whitespace and strip decorative noise. Idempotent."""
    if not text:
        return text
    out = _DECORATIVE.sub("", text)
    out = _TRAILING_WS.sub(r"\1", out)
    out = _RUNS_OF_SPACES.sub(" ", out)
    out = _MANY_NEWLINES.sub("\n\n", out)
    return out
