"""`tokenlens bench <file>` — measure compression savings on a document.

Wraps the file as a single-user-message request (no cache_control, so the whole
thing is eligible tail), runs it through each compression method, and reports
original-vs-compressed token counts. Token counts come from Anthropic's
count_tokens endpoint when --measure is set and ANTHROPIC_API_KEY is available;
otherwise a local estimate is used and clearly labelled.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from .compress import compress_request, llmlingua2
from .compress.estimate import estimate_tokens


def _extract_text(payload: dict) -> str:
    out = []
    for m in payload.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    out.append(b.get("text", ""))
    return "\n".join(out)


def _count_tokens_api(payload: dict, model: str, base: str, api_key: str) -> int:
    countable = {"model": model, "messages": payload["messages"]}
    if "system" in payload:
        countable["system"] = payload["system"]
    data = json.dumps(countable).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/messages/count_tokens",
        data=data,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["input_tokens"]


def run(path: str, rate: float, model: str, measure: bool, base: str) -> int:
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        print(f"bench: cannot read {path}: {e}", file=sys.stderr)
        return 1
    if not text.strip():
        print(f"bench: {path} is empty", file=sys.stderr)
        return 1

    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }
    body = json.dumps(payload).encode()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_api = measure and bool(api_key)
    if measure and not api_key:
        print("[bench] --measure needs ANTHROPIC_API_KEY set; using local estimate.\n",
              file=sys.stderr)

    def count(p: dict, txt: str) -> int:
        if use_api:
            try:
                return _count_tokens_api(p, model, base, api_key)
            except Exception as e:
                print(f"[bench] count_tokens failed ({e}); using local estimate",
                      file=sys.stderr)
        return estimate_tokens(txt)

    source = "count_tokens (exact)" if use_api else "local estimate (~chars/4)"
    original = count(payload, text)

    methods = ["safe"]
    if llmlingua2.available():
        methods.append("llmlingua2")
    else:
        ll_note = f"(llmlingua2 not installed: {llmlingua2.load_error()})"

    print(f"TokenLens bench — {path}")
    print(f"  model: {model}   counts: {source}")
    print(f"  original: {original:,} tokens\n")
    print(f"  {'method':<12}{'tokens':>10}{'saved':>10}{'reduction':>12}")
    print(f"  {'-'*44}")

    preview = None
    for method in methods:
        res = compress_request(body, method=method, rate=rate)
        comp_payload = json.loads(res.body)
        comp_text = _extract_text(comp_payload)
        c = count(comp_payload, comp_text)
        saved = original - c
        pct = f"{round(100 * saved / original, 1)}%" if original else "0%"
        label = method + (f" r={rate}" if method == "llmlingua2" else "")
        print(f"  {label:<12}{c:>10,}{saved:>10,}{pct:>12}")
        if method == "llmlingua2":
            preview = comp_text

    if "llmlingua2" not in methods:
        print(f"\n  {ll_note}\n  install for real pruning:  pip install llmlingua")
    elif preview is not None:
        snippet = preview.strip().replace("\n", " ")[:220]
        print(f"\n  llmlingua2 preview: {snippet}…")

    print()
    return 0
