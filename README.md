# TokenLens

**A local, cache-aware proxy that shows you exactly where your frontier-model tokens go — and safely compresses the parts that are safe to compress.**

Point any Anthropic client at TokenLens instead of `api.anthropic.com`. It
forwards every request through, measures the tokens and cost, optionally
compresses the request, and streams live telemetry to a dashboard in your
browser. Your API key is passed straight through and never stored.

> ⚠️ **Experimental / research-stage.** The proxy, the cache-aware safety rules,
> and the measurement all work and are tested — but there is **no output-quality
> eval harness yet**, so don't turn `--rate` up on important work and assume the
> result is identical. Measure first (`--measure`, or `tokenlens bench`). The
> subscription (OAuth) routing path is not covered by automated tests.
> Feedback and PRs welcome.

See [`DESIGN.md`](./DESIGN.md) for the full architecture and roadmap.

---

## Why

Frontier models bill by the **token**, and they read **tokens, not bytes** — so
classic byte compression (gzip/zstd) is useless: the model can't read it, and
decoding it before sending saves nothing. Real savings only come from sending a
**shorter token sequence the model still interprets correctly**.

TokenLens does that carefully. The critical constraint is **prompt caching**:
cached tokens are cheap but must stay byte-identical, so TokenLens **never
touches the cached prefix, tool calls, code, or history** — it only compresses
plain prose in the volatile tail. This is what makes it safe to put in front of
a real coding agent without busting the cache.

---

## Features

- **Drop-in Anthropic proxy** — mirror of `/v1/messages`, streaming and
  non-streaming. Just change the base URL.
- **BYOK & private** — your key is forwarded, never stored; prompt/response
  bodies are never logged.
- **Cache-aware compression** — safe floor (whitespace/dedup, zero deps) plus an
  optional local LLMLingua-2 token-pruning model.
- **Real savings measurement** — `--measure` counts original vs compressed with
  Anthropic's own `count_tokens` (free, off the hot path).
- **Live dashboard** — token usage, cost, savings, and a cache-health indicator
  that flags if compression ever starts busting the cache.
- **`bench` command** — measure compression on any file in one shot.

---

## Requirements

- **Python 3.9+** (standard library only — no dependencies for the core proxy).
- Optional: `pip install llmlingua` for the LLMLingua-2 compression model
  (pulls in PyTorch).

---

## Install

```bash
git clone https://github.com/jmcdonald69124/tokenlens.git
cd tokenlens
```

No build step needed — run it as a module from the repo root.

---

## Quick start

**1. Start the proxy:**

```bash
python3 -m tokenlens serve          # listens on http://127.0.0.1:8787
```

**2. Open the dashboard:** http://localhost:8787/tokenlens/

**3. Point any Anthropic client at it** (in another terminal):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY=sk-ant-...        # your own key (BYOK)
```

Now use the SDK, CLI, or `curl` exactly as normal — traffic flows through
TokenLens and appears live on the dashboard.

Each request also prints a line:

```
[tokenlens] claude-opus-4-8   in=1240  cache_r=1000 cache_w=0  out=530  $0.01705  842ms
```

Ctrl-C prints the session totals.

### `serve` options

```
python3 -m tokenlens serve \
  --host 127.0.0.1 --port 8787 \
  --upstream https://api.anthropic.com \
  --compress none|safe|llmlingua2 \
  --rate 0.7 \
  --measure
```

---

## Compression (opt-in)

Off by default (pure measurement). Turn it on with `--compress`:

```bash
python3 -m tokenlens serve --compress safe         # deterministic, zero deps
python3 -m tokenlens serve --compress llmlingua2   # safe floor + local model
```

- **`safe`** — collapses whitespace and strips decorative noise. Lossless-ish,
  no dependencies.
- **`llmlingua2 [--rate 0.7]`** — adds the LLMLingua-2 token-pruning model on top
  of the safe floor (`--rate` is the fraction of tokens to keep; lower = more
  aggressive). Requires `pip install llmlingua`; if it's not installed, TokenLens
  warns and falls back to the safe floor.

Compression is **cache-aware and content-safe**: it only touches plain prose in
the volatile tail (after the last `cache_control` breakpoint) and never the
cached prefix, tool calls, code, images, or assistant history.

---

## Seeing real savings (`--measure`)

The default saved-token number is a local estimate. For the true figure, add
`--measure`: every compressed request is counted (original and compressed) via
Anthropic's `count_tokens` endpoint — free, separate from your usage limits, and
run in the background so it never slows you down.

```bash
python3 -m tokenlens serve --compress llmlingua2 --rate 0.6 --measure
```

The dashboard then shows a **measured** tokens-saved headline, a **% reduction**,
and a **cumulative-savings sparkline**.

**Reading the numbers:** on a heavily-cached workload the % looks small because
it's diluted by the cached prefix (which we never touch). The absolute
tokens-saved on the *uncached* content is the real signal.

---

## Measure savings on a file (`bench`)

See whether a given document is worth compressing — no client wiring needed:

```bash
python3 -m tokenlens bench mydoc.txt                 # local estimate
python3 -m tokenlens bench mydoc.txt --measure       # exact (needs ANTHROPIC_API_KEY)
python3 -m tokenlens bench mydoc.txt --rate 0.5      # more aggressive llmlingua2
```

Example output:

```
TokenLens bench — mydoc.txt
  model: claude-opus-4-8   counts: count_tokens (exact)
  original: 2,480 tokens

  method          tokens     saved   reduction
  --------------------------------------------
  safe             2,410        70        2.8%
  llmlingua2 r=0.5 1,290     1,190       48.0%

  llmlingua2 preview: The auth module handles login and session management...
```

Point it at **prose** (docs, specs, articles) for real savings; point it at
**code** and you'll correctly see ~0 — the safety rules skipping it.

---

## Using it with Claude Code

Claude Code honors `ANTHROPIC_BASE_URL`, so it can route through TokenLens:

```bash
python3 -m tokenlens serve --compress safe --measure   # terminal 1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787        # terminal 2, then: claude
```

Because Claude Code caches its system prompt and history, most of its input is
cached prefix that TokenLens (correctly) never touches — so on typical warm-cache
coding turns there's little to compress. The bigger wins are **uncached, prose-
heavy** turns (pasting a large document to summarize or discuss).

Watch the **cache-health banner** on the dashboard: green means compression isn't
disturbing the cache. If it goes amber (cache reads stuck at zero), stop and
check — that means the cache is being busted, which is counterproductive.

> The subscription (OAuth) routing path isn't covered by automated tests. If
> Claude Code errors with a custom base URL, open an issue with the message.

---

## Dashboard & endpoints

| Path | What |
|------|------|
| `GET /tokenlens/` | Live telemetry dashboard (open in a browser) |
| `GET /tokenlens/feed?limit=50` | JSON feed powering the dashboard |
| `GET /tokenlens/stats` | Cumulative totals as JSON |
| `POST /v1/messages` | Proxied to the upstream Anthropic API |

---

## Privacy & security

- **BYOK** — your API key is forwarded to Anthropic and never stored or written
  to disk.
- **No content logging** — only token counts, status, latency, and cost are
  recorded. Prompts and responses are never logged.
- **Localhost by default** — binds to `127.0.0.1`. Don't expose it on an
  untrusted network.

---

## How it works (short version)

1. Parse the request and find the last `cache_control` breakpoint.
2. Treat everything up to and including it as the **frozen cached prefix** — never
   modified.
3. In the volatile tail, compress only plain-prose text blocks in user turns;
   skip tool calls, code, images, documents, and assistant history.
4. Forward the (possibly smaller) request upstream unchanged otherwise; stream
   the response straight back.
5. Record token usage from the real response; optionally `count_tokens` the
   before/after for exact savings.

Everything is fail-open: any parse or compression error forwards the original
request untouched, so the proxy never breaks a call.

---

## Roadmap

- **Eval harness** — score compressed vs. cleartext output quality so aggressive
  compression rates can be trusted (the next priority).
- More compression rungs (structural minify, dedup, retrieval/selection).
- Pre-cache compression (shrink large context *before* it's cached, cutting both
  the cache-write and every future read).

See [`DESIGN.md`](./DESIGN.md) for the full plan.

---

## Contributing

It's a few hundred lines of dependency-light Python. The compression pipeline
(`tokenlens/compress/`), the safety rules (`estimate.py`), and the dashboard
(`dashboard.py`) are all easy to read and extend. Run the tests:

```bash
PYTHONPATH=. python3 tests/test_pipeline.py
```

Better compressor rungs and an eval harness especially welcome.

---

## License

MIT — see [`LICENSE`](./LICENSE).
