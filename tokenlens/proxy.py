"""The TokenLens local proxy.

Mirrors the Anthropic Messages API surface, forwards each request unchanged to
the upstream Anthropic endpoint (BYOK — the caller's own x-api-key is passed
straight through and never stored), and logs token usage + estimated cost.

Milestone 1 does NOT compress anything yet. Its job is to prove the plumbing
end to end and establish the baseline (tokens in / out / cost) that later
compression rungs are measured against.

Point any Anthropic SDK or app at http://localhost:8787 as the base URL and it
works transparently.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, pricing
from .compress import compress_request
from .dashboard import DASHBOARD_HTML
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


class _Handler(BaseHTTPRequestHandler):
    # set by the server factory
    upstream: str = "https://api.anthropic.com"
    stats: Stats = Stats()
    compress_method: str = "none"   # "none" | "safe" | "llmlingua2"
    compress_rate: float = 0.7
    measure: bool = False           # ground-truth savings via count_tokens
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
            body = json.dumps(self.stats.feed(limit)).encode()
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
                print(f"[tokenlens] compression skipped: {e}", file=sys.stderr, flush=True)
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


def serve(host: str, port: int, upstream: str,
          compress_method: str = "none", compress_rate: float = 0.7,
          measure: bool = False) -> None:
    _Handler.upstream = upstream
    _Handler.stats = Stats()
    _Handler.compress_method = compress_method
    _Handler.compress_rate = compress_rate
    _Handler.measure = measure

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
