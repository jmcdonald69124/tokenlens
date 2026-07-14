"""A minimal Anthropic Messages API client built on the standard library.

TokenLens is deliberately zero-dependency, and the proxy and bench already
speak the Messages API by hand over `urllib` (see proxy.py / bench.py). The
harness follows the same rule rather than pulling the official SDK in for a
handful of calls. If TokenLens ever grows a dependency list, this file is the
first thing that should be replaced by `anthropic`.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE = "https://api.anthropic.com"

# Retry on transient upstream conditions only.
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
_MAX_ATTEMPTS = 4


class ApiError(RuntimeError):
    """A non-retryable (or retry-exhausted) upstream failure."""

    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(f"{status or 'network'}: {message}")
        self.status = status
        self.message = message


class Client:
    """POSTs JSON to the Messages API. Not thread-affine — safe to share."""

    def __init__(self, headers: dict, base: str = DEFAULT_BASE, timeout: int = 600) -> None:
        self._headers = headers
        self.base = base.rstrip("/")
        self.timeout = timeout

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_env(cls, base: str = DEFAULT_BASE, timeout: int = 600) -> "Client":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ApiError(None, "ANTHROPIC_API_KEY is not set")
        return cls(
            {
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            base,
            timeout,
        )

    @classmethod
    def from_request_headers(cls, headers: dict, base: str = DEFAULT_BASE,
                             timeout: int = 600) -> "Client":
        """Build a client from a caller's own forwarded auth headers (BYOK).

        Used by the proxy's shadow judge: the caller's key is borrowed for the
        lifetime of the request it arrived on and never persisted anywhere.
        """
        keep = {"x-api-key", "authorization", "anthropic-version", "anthropic-beta"}
        out = {k.lower(): v for k, v in headers.items() if k.lower() in keep}
        out["content-type"] = "application/json"
        out.setdefault("anthropic-version", ANTHROPIC_VERSION)
        if "x-api-key" not in out and "authorization" not in out:
            raise ApiError(None, "no credentials on the incoming request")
        return cls(out, base, timeout)

    # -- transport ---------------------------------------------------------
    def post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        url = self.base + path
        last: Exception | None = None

        for attempt in range(_MAX_ATTEMPTS):
            req = urllib.request.Request(url, data=data, headers=self._headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code not in _RETRY_STATUS or attempt == _MAX_ATTEMPTS - 1:
                    raise ApiError(e.code, _error_message(body)) from None
                last = e
            except urllib.error.URLError as e:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise ApiError(None, str(e.reason)) from None
                last = e
            # exponential backoff with jitter
            time.sleep((2 ** attempt) + random.random())

        raise ApiError(None, str(last))

    # -- endpoints ---------------------------------------------------------
    def messages(self, payload: dict) -> dict:
        return self.post("/v1/messages", payload)

    _COUNTABLE = ("model", "messages", "system", "tools", "tool_choice")

    def count_tokens(self, payload: dict) -> int | None:
        """Ground-truth input token count. Free endpoint; None on failure."""
        countable = {k: payload[k] for k in self._COUNTABLE if k in payload}
        try:
            return self.post("/v1/messages/count_tokens", countable).get("input_tokens")
        except ApiError:
            return None


def _error_message(body: str) -> str:
    try:
        return json.loads(body).get("error", {}).get("message", body)[:300]
    except (json.JSONDecodeError, AttributeError):
        return body[:300]


def text_of(message: dict) -> str:
    """Concatenate the text blocks of a Messages API response."""
    parts = []
    for block in message.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()
