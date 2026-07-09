"""Correctness tests for the cache-aware, content-safe compression pipeline.

These exercise the deterministic parts (no model). The safe-floor `_apply`
never drops semantic content, so we assert on *what is left untouched* — the
critical safety properties — plus that whitespace normalization happens.
"""

import json

from tokenlens.compress import compress_request
from tokenlens.compress.estimate import is_prose
from tokenlens.compress.safe import safe_compress


def _msg(role, blocks):
    return {"role": role, "content": blocks}


def _text(t, cache=False):
    b = {"type": "text", "text": t}
    if cache:
        b["cache_control"] = {"type": "ephemeral"}
    return b


def run(payload, method="safe", rate=0.7):
    res = compress_request(json.dumps(payload).encode(), method=method, rate=rate)
    return json.loads(res.body), res


def test_none_is_passthrough():
    payload = {"model": "m", "messages": [_msg("user", "hello   world")]}
    out, res = run(payload, method="none")
    assert out == payload
    assert res.method == "none"


def test_cached_prefix_is_never_touched():
    # A cached user block, then a fresh (tail) user block after the boundary.
    payload = {
        "model": "m",
        "system": [_text("big frozen system prompt with   spaces", cache=True)],
        "messages": [
            _msg("user", [_text("cached    context block", cache=True)]),
            _msg("user", [_text("fresh    tail    prose here")]),
        ],
    }
    out, res = run(payload)
    # system + first message are at/before the last cache breakpoint -> frozen
    assert out["system"][0]["text"] == "big frozen system prompt with   spaces"
    assert out["messages"][0]["content"][0]["text"] == "cached    context block"
    # the fresh tail prose IS normalized
    assert out["messages"][1]["content"][0]["text"] == "fresh tail prose here"
    assert res.compressed_blocks == 1


def test_tool_blocks_never_touched():
    payload = {
        "model": "m",
        "messages": [
            _msg("user", [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "raw    output    keep    spaces"},
                _text("please    summarize the result"),
            ]),
        ],
    }
    out, res = run(payload)
    assert out["messages"][0]["content"][0]["content"] == "raw    output    keep    spaces"
    assert out["messages"][0]["content"][1]["text"] == "please summarize the result"


def test_assistant_history_never_touched():
    payload = {
        "model": "m",
        "messages": [
            _msg("assistant", [_text("earlier    assistant    turn")]),
            _msg("user", [_text("new    user    turn")]),
        ],
    }
    out, _ = run(payload)
    assert out["messages"][0]["content"][0]["text"] == "earlier    assistant    turn"
    assert out["messages"][1]["content"][0]["text"] == "new user turn"


def test_code_blocks_never_touched():
    code = "def f(x):\n    return   x + 1\n"
    payload = {"model": "m", "messages": [_msg("user", [_text(code)])]}
    out, res = run(payload)
    assert out["messages"][0]["content"][0]["text"] == code  # untouched
    assert res.compressed_blocks == 0


def test_output_is_valid_json_and_roundtrips_structure():
    payload = {
        "model": "m",
        "max_tokens": 100,
        "messages": [_msg("user", [_text("hello    there    friend")])],
    }
    _, res = run(payload)
    reparsed = json.loads(res.body)
    assert reparsed["model"] == "m"
    assert reparsed["max_tokens"] == 100


def test_prose_heuristic():
    assert is_prose("Please review the auth module and tell me what you find.")
    # prose containing code-ish English words must still count as prose
    assert is_prose("Can you return the summary and remove the class from the report?")
    assert is_prose("I got this from the logs and it did not end well for us.")
    assert not is_prose('{"key": "value"}')
    assert not is_prose("```python\nprint(1)\n```")
    assert not is_prose("@@ -1,2 +1,3 @@")
    assert not is_prose("def handle(req):\n    return req")   # indented code
    assert not is_prose("import os")                          # bare import
    assert not is_prose("")


def test_safe_compress_idempotent():
    t = "a   b\n\n\n\nc  \n"
    once = safe_compress(t)
    assert safe_compress(once) == once
    assert "   " not in once
    assert "\n\n\n" not in once


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\nALL {len(fns)} PIPELINE TESTS PASSED")
    sys.exit(0)
