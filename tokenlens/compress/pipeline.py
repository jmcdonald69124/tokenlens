"""Cache-aware, content-safe compression of an Anthropic /v1/messages body.

The two hard rules for a coding-agent workload like Claude Code:

  1. NEVER mutate the cached prefix. Anything up to and including the last
     `cache_control` breakpoint (in render order tools -> system -> messages)
     is frozen — changing one byte busts the cache and re-bills the whole
     prefix, which can cost MORE limit than compression saves.

  2. NEVER touch non-prose. Only plain natural-language text in the volatile
     tail is eligible. Code, tool_use / tool_result, images, documents, and
     assistant/history turns are left byte-for-byte intact.

Everything is fail-open: any parse or compression error returns the original
body unchanged, so the proxy never breaks a request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import llmlingua2, safe
from .estimate import estimate_tokens, is_prose


@dataclass
class CompressionResult:
    body: bytes
    method: str            # "none" | "safe" | "llmlingua2"
    eligible_blocks: int
    compressed_blocks: int
    original_est_tokens: int
    compressed_est_tokens: int

    @property
    def est_tokens_saved(self) -> int:
        return max(0, self.original_est_tokens - self.compressed_est_tokens)


def _has_cache_control(block) -> bool:
    return isinstance(block, dict) and block.get("cache_control") is not None


def _find_cache_boundary(system, messages) -> tuple[int, int]:
    """Return (sys_idx, msg_idx) — indices *at or before which* content is frozen.

    Returns the position of the last cache_control breakpoint. If it is in a
    system block, msg boundary is -1 (nothing in messages is frozen by it, but
    a later message breakpoint would override). We scan messages last (render
    order), so the message boundary wins when present.
    """
    sys_boundary = -1
    if isinstance(system, list):
        for i, b in enumerate(system):
            if _has_cache_control(b):
                sys_boundary = i

    msg_boundary = -1
    if isinstance(messages, list):
        for i, m in enumerate(messages):
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for b in content:
                    if _has_cache_control(b):
                        msg_boundary = i
            # a string-content message can't carry cache_control
    return sys_boundary, msg_boundary


def compress_request(body: bytes, *, method: str = "safe", rate: float = 0.7) -> CompressionResult:
    """Compress eligible prose in the volatile tail of an Anthropic request body.

    method: "none" | "safe" (rungs 0-2) | "llmlingua2" (safe floor + rung 5).
    rate:   target fraction of tokens to keep for llmlingua2 (0<rate<=1).
    """
    if method == "none":
        return CompressionResult(body, "none", 0, 0, 0, 0)

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return CompressionResult(body, "none", 0, 0, 0, 0)
    if not isinstance(payload, dict):
        return CompressionResult(body, "none", 0, 0, 0, 0)

    messages = payload.get("messages")
    system = payload.get("system")
    _, msg_boundary = _find_cache_boundary(system, messages)

    if not isinstance(messages, list):
        return CompressionResult(body, "none", 0, 0, 0, 0)

    eligible = 0
    compressed = 0
    orig_tokens = 0
    new_tokens = 0
    changed = False

    for m_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        # Only compress content in the volatile tail (strictly after the last
        # cache breakpoint) and only in user turns (never assistant history).
        if m_idx <= msg_boundary:
            continue
        if message.get("role") != "user":
            continue

        content = message.get("content")
        if not isinstance(content, list):
            # string content: eligible only if it's prose in the tail
            if isinstance(content, str) and is_prose(content):
                eligible += 1
                orig_tokens += estimate_tokens(content)
                out = _apply(content, method, rate)
                new_tokens += estimate_tokens(out)
                if out != content:
                    message["content"] = out
                    compressed += 1
                    changed = True
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            # never touch anything with a cache breakpoint, or non-text blocks
            if block.get("type") != "text" or _has_cache_control(block):
                continue
            text = block.get("text", "")
            if not is_prose(text):
                continue
            eligible += 1
            orig_tokens += estimate_tokens(text)
            out = _apply(text, method, rate)
            new_tokens += estimate_tokens(out)
            if out != text:
                block["text"] = out
                compressed += 1
                changed = True

    if not changed:
        return CompressionResult(body, method, eligible, 0, orig_tokens, orig_tokens)

    new_body = json.dumps(payload).encode("utf-8")
    return CompressionResult(new_body, method, eligible, compressed, orig_tokens, new_tokens)


def _apply(text: str, method: str, rate: float) -> str:
    out = safe.safe_compress(text)          # rungs 0-2 always
    if method == "llmlingua2":
        out = llmlingua2.compress(out, rate)  # rung 5 on top
    return out
