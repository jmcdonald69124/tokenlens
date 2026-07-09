"""Thread-safe running totals + a rolling event buffer for the live dashboard."""

from __future__ import annotations

import threading
import time
from collections import deque

_RECENT_MAX = 200


class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0
        self.input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.est_tokens_saved = 0
        # ground-truth measurement (via count_tokens), when --measure is on
        self.measured_requests = 0
        self.real_original_tokens = 0
        self.real_compressed_tokens = 0
        self._recent: deque[dict] = deque(maxlen=_RECENT_MAX)
        # cumulative-saved series for the dashboard sparkline
        self._series: deque[dict] = deque(maxlen=300)

    def record_event(
        self,
        model: str,
        usage: dict,
        cost: float | None,
        saved: int,
        elapsed_ms: int,
        status: int,
    ) -> None:
        ts = time.time()
        inp = usage.get("input_tokens", 0) or 0
        cr = usage.get("cache_read_input_tokens", 0) or 0
        cw = usage.get("cache_creation_input_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        event = {
            "ts": ts,
            "time": time.strftime("%H:%M:%S", time.localtime(ts)),
            "model": model,
            "status": status,
            "input": inp,
            "cache_read": cr,
            "cache_write": cw,
            "output": out,
            "saved": saved,
            "cost": round(cost, 6) if cost is not None else None,
            "ms": elapsed_ms,
        }
        with self._lock:
            self.requests += 1
            self.input_tokens += inp
            self.cache_read_tokens += cr
            self.cache_write_tokens += cw
            self.output_tokens += out
            self.est_tokens_saved += saved
            if cost is not None:
                self.cost_usd += cost
            self._recent.appendleft(event)
            # sparkline point: cumulative saved (real if we have it, else est)
            cum = self.real_original_tokens - self.real_compressed_tokens
            if self.measured_requests == 0:
                cum = self.est_tokens_saved
            self._series.append({"t": ts, "cum": cum})

    def record_measurement(self, original_tokens: int, compressed_tokens: int) -> None:
        """Record a ground-truth original-vs-compressed token count."""
        with self._lock:
            self.measured_requests += 1
            self.real_original_tokens += original_tokens
            self.real_compressed_tokens += compressed_tokens
            self._series.append({
                "t": time.time(),
                "cum": self.real_original_tokens - self.real_compressed_tokens,
            })

    def snapshot(self) -> dict:
        with self._lock:
            real_saved = self.real_original_tokens - self.real_compressed_tokens
            pct = None
            if self.real_original_tokens > 0:
                pct = round(100 * real_saved / self.real_original_tokens, 1)
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "est_tokens_saved": self.est_tokens_saved,
                "measured_requests": self.measured_requests,
                "real_tokens_saved": real_saved,
                "real_savings_pct": pct,
            }

    def feed(self, limit: int = 50) -> dict:
        with self._lock:
            recent = list(self._recent)[:limit]
            series = [p["cum"] for p in self._series]
        return {
            "totals": self.snapshot(),
            "recent": recent,
            "series": series,
            "uptime_s": round(time.time() - self.started_at, 1),
            "server_time": time.time(),
        }


def format_request(model: str, usage: dict, cost: float | None, elapsed_ms: int) -> str:
    """One-line human-readable summary of a completed request (stderr log)."""
    parts = [f"in={usage.get('input_tokens', 0) or 0}"]
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    if cr or cw:
        parts.append(f"cache_r={cr} cache_w={cw}")
    parts.append(f"out={usage.get('output_tokens', 0) or 0}")
    cost_str = f"${cost:.5f}" if cost is not None else "$?(unknown model)"
    return f"[tokenlens] {model:<22} {'  '.join(parts)}  {cost_str}  {elapsed_ms}ms"
