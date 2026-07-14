"""The TokenLens local proxy.

Mirrors the Anthropic Messages API surface, forwards each request to the
upstream Anthropic endpoint (BYOK — the caller's own x-api-key is passed
straight through and never stored), compresses the volatile tail when asked to,
and logs token usage + estimated cost.

With --judge it also runs the eval harness's judge in shadow mode on a sample of
live traffic: the original, uncompressed request is replayed alongside the
compressed one and a model grades both answers, so the dashboard can show what
compression is *costing* you, not just what it is saving. That is expensive by
construction — see serve() for the warning it prints.

Point any Anthropic SDK or app at http://localhost:8787 as the base URL and it
works transparently.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, pricing
from .compress import compress_request
from .dashboard import DASHBOARD_HTML
from .eval import judge as judge_mod
from .eval.api import ApiError, Client, text_of
from .stats import Stats, format_request

# Headers we must not copy verbatim between the two hops.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",  # we force identity upstream to avoid gzip handling
}


def _extract_usage_from_sse(buffer: str) -> dict:
    """Parse accumulated SSE text and return the best-known usage dict.

    message_start carries input/cache tokens; message_delta carries the
    cumulative output_tokens. We merge whatever we've seen.
    """
    usage: dict = {}
    for line in buffer.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "message_start":
            u = evt.get("message", {}).get("usage", {})
            usage.update({k: v for k, v in u.items() if v is not None})
        elif etype == "message_delta":
            u = evt.get("usage", {})
            usage.update({k: v for k, v in u.items() if v is not None})
    return usage


def _extract_text_from_sse(buffer: str) -> str:
    """Reassemble the assistant's text from an accumulated SSE stream."""
    parts = []
    for line in buffer.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "content_block_delta":
            delta = evt.get("delta", {})
            if delta.get("type") == "text_delta":
                parts.append(delta.get("text", ""))
    return "".join(parts).strip()


def _user_text(payload: dict) -> str:
    """The text of the user turns — what the judge grades the answers against."""
    out = []
    for m in payload.get("messages", []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    out.append(b.get("text", ""))
    return "\n".join(out)


class _Handler(BaseHTTPRequestHandler):
    # set by the server factory
    upstream: str = "https://api.anthropic.com"
    stats: Stats = Stats()
    compress_method: str = "none"   # "none" | "safe" | "llmlingua2"
    compress_rate: float = 0.7
    measure: bool = False           # ground-truth savings via count_tokens
    judge: bool = False             # shadow-mode LLM-as-judge on live traffic
    judge_model: str = judge_mod.DEFAULT_JUDGE_MODEL
    judge_sample: float = 0.25      # fraction of compressed requests to judge
    eval_report: dict | None = None  # calibration curve from `tokenlens eval`
    _req_model: str = "unknown"
    _req_saved: int = 0

    protocol_version = "HTTP/1.1"
    server_version = f"tokenlens/{__version__}"

    def log_message(self, *args) -> None:  # silence default noisy logging
        pass

    # --- local control endpoints ------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/tokenlens", "/tokenlens/dashboard"):
            self._send(200, {"content-type": "text/html; charset=utf-8"},
                       DASHBOARD_HTML.encode())
        elif path == "/tokenlens/feed":
            limit = 50
            if "?" in self.path:
                from urllib.parse import parse_qs
                q = parse_qs(self.path.split("?", 1)[1])
                try:
                    limit = max(1, min(200, int(q.get("limit", ["50"])[0])))
                except ValueError:
                    pass
            feed = self.stats.feed(limit)
            feed["eval"] = self.eval_report
            feed["mode"] = {
                "compress": self.compress_method,
                "rate": self.compress_rate,
                "judge": self.judge,
                "judge_model": self.judge_model if self.judge else None,
                "judge_sample": self.judge_sample if self.judge else None,
            }
            body = json.dumps(feed).encode()
            self._send(200, {"content-type": "application/json", "cache-control": "no-store"}, body)
        elif path == "/tokenlens/stats":
            body = json.dumps(self.stats.snapshot(), indent=2).encode()
            self._send(200, {"content-type": "application/json"}, body)
        else:
            self._send(404, {"content-type": "text/plain"}, b"tokenlens: not found\n")

    # --- the proxy path ---------------------------------------------------
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length) if length else b""

        model = "unknown"
        streaming = False
        try:
            parsed = json.loads(body) if body else {}
            model = parsed.get("model", "unknown")
            streaming = bool(parsed.get("stream"))
        except (json.JSONDecodeError, AttributeError):
            pass

        self._req_model = model
        self._req_saved = 0
        self._judge_original = None   # set only when this request will be judged

        # Cache-aware, content-safe compression of the request body. Fail-open:
        # any error here leaves the original body untouched.
        forward_body = body
        if self.compress_method != "none" and self.path.endswith("/v1/messages"):
            try:
                result = compress_request(
                    body, method=self.compress_method, rate=self.compress_rate
                )
                forward_body = result.body
                if result.eligible_blocks:
                    self._req_saved = result.est_tokens_saved
                    print(
                        f"[tokenlens] compress {self.compress_method}: "
                        f"{result.compressed_blocks}/{result.eligible_blocks} prose "
                        f"blocks, ~{self._req_saved} tok saved (est)",
                        file=sys.stderr, flush=True,
                    )
            except Exception as e:  # never break the user's request
                # Type only, not the message: an exception raised inside a
                # third-party tokenizer can quote the text it choked on, and
                # "prompt bodies are never logged" has to survive the error path
                # too or it isn't a guarantee. Reproduce with `tokenlens bench`.
                print(f"[tokenlens] compression skipped ({type(e).__name__}); "
                      f"request forwarded unchanged",
                      file=sys.stderr, flush=True)
                forward_body = body

        req_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        req_headers["accept-encoding"] = "identity"

        # Ground-truth savings: count original vs compressed off the hot path.
        if (self.measure and self.compress_method != "none"
                and self.path.endswith("/v1/messages") and forward_body is not body):
            auth = {k: v for k, v in req_headers.items() if k.lower() != "content-length"}
            threading.Thread(
                target=self._measure_savings,
                args=(body, forward_body, auth),
                daemon=True,
            ).start()

        # Shadow-mode judging: only ever on a request we actually compressed —
        # judging an untouched request would grade the model against itself.
        if (self.judge and self.path.endswith("/v1/messages")
                and forward_body is not body
                and random.random() < self.judge_sample):
            self._judge_original = body
            self._judge_headers = dict(req_headers)

        url = self.upstream.rstrip("/") + self.path
        upstream_req = urllib.request.Request(
            url, data=forward_body, headers=req_headers, method="POST"
        )

        start = time.monotonic()
        try:
            resp = urllib.request.urlopen(upstream_req, timeout=600)
        except urllib.error.HTTPError as e:
            # Forward upstream error responses faithfully.
            err_body = e.read()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.stats.record_event(model, {}, None, self._req_saved, elapsed_ms, e.code)
            self._send(
                e.code,
                {"content-type": e.headers.get("content-type", "application/json")},
                err_body,
            )
            return
        except urllib.error.URLError as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.stats.record_event(model, {}, None, self._req_saved, elapsed_ms, 502)
            msg = json.dumps({
                "type": "error",
                "error": {"type": "tokenlens_upstream_error", "message": str(e)},
            }).encode()
            self._send(502, {"content-type": "application/json"}, msg)
            return

        if streaming:
            self._relay_stream(resp, model, start)
        else:
            self._relay_full(resp, model, start)

    # --- non-streaming ----------------------------------------------------
    def _relay_full(self, resp, model: str, start: float) -> None:
        data = resp.read()
        elapsed_ms = int((time.monotonic() - start) * 1000)

        usage = {}
        try:
            usage = json.loads(data).get("usage", {}) or {}
        except (json.JSONDecodeError, AttributeError):
            pass
        self._account(model, usage, elapsed_ms, resp.status)

        headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
        }
        self._send(resp.status, headers, data)

        if self._judge_original and resp.status == 200:
            try:
                answer = text_of(json.loads(data))
            except (json.JSONDecodeError, AttributeError):
                answer = ""
            self._spawn_judge(model, answer)

    # --- streaming (SSE) --------------------------------------------------
    def _relay_stream(self, resp, model: str, start: float) -> None:
        headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
        }
        headers.setdefault("content-type", "text/event-stream")
        # We don't know the length ahead of time; stream with chunked encoding.
        self.send_response(resp.status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

        text_buffer = ""
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self._write_chunk(chunk)
                text_buffer += chunk.decode("utf-8", errors="replace")
            self._write_chunk(b"")  # terminating 0-length chunk
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            usage = _extract_usage_from_sse(text_buffer)
            self._account(model, usage, elapsed_ms, resp.status)
            if self._judge_original and resp.status == 200:
                self._spawn_judge(model, _extract_text_from_sse(text_buffer))

    # --- model judgement --------------------------------------------------
    def _spawn_judge(self, model: str, compressed_answer: str) -> None:
        if not compressed_answer:
            return
        threading.Thread(
            target=self._judge_request,
            args=(model, self._judge_original, self._judge_headers, compressed_answer),
            daemon=True,
        ).start()

    def _judge_request(self, model: str, original: bytes, headers: dict,
                       compressed_answer: str) -> None:
        """Replay the uncompressed request and have a model grade both answers.

        This runs off the hot path, after the client already has its response.
        It costs a full extra completion plus a judge call, which is why it is
        sampled and off by default. The caller's key is borrowed for the life of
        this thread and never written anywhere.
        """
        try:
            payload = json.loads(original)
        except (json.JSONDecodeError, ValueError):
            return
        payload.pop("stream", None)  # the control run is non-streaming

        try:
            client = Client.from_request_headers(headers, self.upstream)
            cleartext_answer = text_of(client.messages(payload))
            if not cleartext_answer:
                return
            j = judge_mod.grade(
                client,
                case_key=f"{model}:{time.time()}",
                request_text=_user_text(payload),
                cleartext_answer=cleartext_answer,
                compressed_answer=compressed_answer,
                judge_model=self.judge_model,
            )
        except ApiError as e:
            print(f"[tokenlens] judge skipped: {e}", file=sys.stderr, flush=True)
            return

        self.stats.record_judgement(
            model, j.cleartext_grade, j.compressed_grade, j.retention, j.note
        )
        flag = "DEGRADED" if j.retention < self.stats.judge_tolerance else "ok"
        print(f"[tokenlens] judged: cleartext {j.cleartext_grade} vs compressed "
              f"{j.compressed_grade} — {j.retention:.0%} quality retained [{flag}] "
              f"— {j.note}", file=sys.stderr, flush=True)

    # --- helpers ----------------------------------------------------------
    def _account(self, model: str, usage: dict, elapsed_ms: int, status: int) -> None:
        cost = pricing.cost_usd(model, usage) if usage else None
        saved = getattr(self, "_req_saved", 0)
        self.stats.record_event(model, usage, cost, saved, elapsed_ms, status)
        print(format_request(model, usage, cost, elapsed_ms), file=sys.stderr, flush=True)

    # count_tokens accepts only these top-level fields
    _COUNTABLE = ("model", "messages", "system", "tools", "tool_choice")

    def _count_tokens(self, payload: dict, headers: dict) -> int | None:
        countable = {k: payload[k] for k in self._COUNTABLE if k in payload}
        data = json.dumps(countable).encode()
        url = self.upstream.rstrip("/") + "/v1/messages/count_tokens"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read()).get("input_tokens")
        except Exception:
            return None

    def _measure_savings(self, original: bytes, compressed: bytes, headers: dict) -> None:
        try:
            orig = json.loads(original)
            comp = json.loads(compressed)
        except (json.JSONDecodeError, ValueError):
            return
        o = self._count_tokens(orig, headers)
        c = self._count_tokens(comp, headers)
        if o is None or c is None:
            return
        self.stats.record_measurement(o, c)
        print(f"[tokenlens] measured: {o} -> {c} tokens "
              f"(-{o - c}, {round(100*(o-c)/o,1) if o else 0}% smaller)",
              file=sys.stderr, flush=True)

    def _write_chunk(self, chunk: bytes) -> None:
        self.wfile.write(f"{len(chunk):X}\r\n".encode())
        self.wfile.write(chunk)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _send(self, status: int, headers: dict, body: bytes) -> None:
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _load_eval_report(path: str) -> dict | None:
    """Load a `tokenlens eval` report and keep only what the dashboard renders."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[tokenlens] could not read eval report {path}: {e}", file=sys.stderr)
        return None
    return {
        "model": data.get("model"),
        "judge_model": data.get("judge_model"),
        "tolerance": data.get("tolerance"),
        "tasks": data.get("tasks"),
        "generated_at": data.get("generated_at"),
        "noise_floor": data.get("noise_floor"),
        "repeats": data.get("repeats"),
        "curve": [
            {k: s.get(k) for k in
             ("arm", "savings_pct", "mean_retention", "worst_retention",
              "passes", "resolved", "is_control")}
            for s in data.get("curve", [])
        ],
        "policy": data.get("policy", {}),
        "task_catalog": data.get("task_catalog", []),
    }


def serve(host: str, port: int, upstream: str,
          compress_method: str = "none", compress_rate: float = 0.7,
          measure: bool = False, judge: bool = False,
          judge_model: str = judge_mod.DEFAULT_JUDGE_MODEL,
          judge_sample: float = 0.25, judge_tolerance: float = 0.99,
          eval_report: str | None = None) -> None:
    _Handler.upstream = upstream
    _Handler.stats = Stats(judge_tolerance=judge_tolerance)
    _Handler.compress_method = compress_method
    _Handler.compress_rate = compress_rate
    _Handler.measure = measure
    _Handler.judge = judge and compress_method != "none"
    _Handler.judge_model = judge_model
    _Handler.judge_sample = judge_sample
    _Handler.eval_report = _load_eval_report(eval_report) if eval_report else None

    mode = "measure only" if compress_method == "none" else f"compress={compress_method}"
    httpd = ThreadingHTTPServer((host, port), _Handler)
    base = f"http://{host}:{port}"
    print(f"tokenlens {__version__} — local Anthropic proxy ({mode})")
    print(f"  listening on {base}")
    print(f"  forwarding to {upstream}")
    print(f"  dashboard:    {base}/tokenlens/   <- open this in a browser")
    print(f"  stats (json): {base}/tokenlens/stats")
    if measure:
        print(f"  measure:      ON (real savings via count_tokens)")
    if judge and compress_method == "none":
        print("  WARNING: --judge does nothing without --compress; "
              "there is no compressed answer to grade.")
    elif _Handler.judge:
        print(f"  judge:        ON ({judge_model}, sampling "
              f"{judge_sample:.0%} of compressed requests)")
        print(f"                COSTS MONEY: each judged request replays the "
              f"uncompressed prompt upstream")
        print(f"                and adds one judge call. Lower --judge-sample to spend less.")
    if _Handler.eval_report:
        print(f"  eval report:  {eval_report} (calibration curve shown on the dashboard)")

    # The dashboard feed is unauthenticated — anyone who can reach the port can
    # read it. That is fine on loopback and only on loopback. With --judge on it
    # carries the judge's one-line notes, which are derived from your prompts, so
    # binding this to a routable interface publishes your content to the network.
    if host not in ("127.0.0.1", "::1", "localhost"):
        print()
        print(f"  WARNING: bound to {host}, not loopback. {base}/tokenlens/ is")
        print( "           unauthenticated: anyone who can reach this port can read your")
        print( "           token counts, and — with --judge on — the judge's notes, which")
        print( "           are derived from your prompt content. Bind 127.0.0.1 unless you")
        print( "           genuinely mean to serve this.")
    if compress_method == "llmlingua2":
        from .compress import llmlingua2 as _ll
        if not _ll.available():
            print(f"  WARNING: llmlingua2 unavailable ({_ll.load_error()}); "
                  f"falling back to safe floor. Install: pip install llmlingua")
    print()
    print("Point your Anthropic client at this base URL, e.g.:")
    print(f"  export ANTHROPIC_BASE_URL={base}")
    print("  (your own ANTHROPIC_API_KEY is passed straight through — BYOK)")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down; final totals:")
        print(json.dumps(_Handler.stats.snapshot(), indent=2))
