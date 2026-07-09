# TokenLens

**A local, cache-aware proxy that shows you exactly where your frontier-model tokens go — and safely compresses the parts that are safe to compress.**

Point any Anthropic client at TokenLens instead of `api.anthropic.com`. It
forwards every request through, measures the tokens and cost, optionally
compresses the request, and streams live telemetry to a dashboard in your
browser. Your API key is passed straight through and never stored.

> ⚠️ **Experimental / research-stage.** The proxy, the cache-aware safety rules,
> and the measurement all work and are tested. Output quality is now measurable
> too — `tokenlens eval` grades compressed answers against cleartext answers with
> an LLM judge — but **the bundled golden set is a 10-task starter set, not a
> benchmark**. Calibrate on your own traffic before trusting a rate. The
> Claude Code routing path is not covered by automated tests.
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
- **Eval harness** — `tokenlens eval` runs a golden task set through the model
  twice, cleartext and compressed, and has a judge model grade both answers. Out
  comes a compression-ratio vs quality curve and a per-task-class policy: the
  most aggressive setting whose quality still lands within tolerance.
- **Model judgement on live traffic** — `serve --judge` runs that same judge in
  shadow mode on a sample of real requests, so the dashboard shows what
  compression is *costing* you, not just what it is saving.
- **Live dashboard** — token usage, cost, savings, quality retained, and a
  cache-health indicator that flags if compression starts busting the cache.
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
  --measure \
  --judge --judge-sample 0.25 --judge-model claude-opus-4-8 \
  --eval-report tokenlens-eval.json
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
Anthropic's `count_tokens` endpoint — free, doesn't add to your billed token
usage, and run in the background so it never slows you down.

```bash
python3 -m tokenlens serve --compress llmlingua2 --rate 0.6 --measure
```

The dashboard then shows a **measured** tokens-saved headline, a **% reduction**,
and a **cumulative-savings sparkline**.

**Reading the numbers:** on a heavily-cached workload the % looks small because
it's diluted by the cached prefix (which we never touch). The absolute
tokens-saved on the *uncached* content is the real signal.

---

## Is it still any good? (`eval`)

Tokens saved is the easy half. The half that decides whether you can actually
turn compression up is: **did the answer get worse?** `tokenlens eval` answers
that.

For every task in a golden set it sends the same request twice — once cleartext,
once through the exact `compress_request()` the proxy uses — and gives both
answers to a judge model, which grades them against a reference answer without
being told which is which. The metric is **retention**: the compressed answer's
grade as a fraction of the cleartext answer's grade on the same request. 100%
means the judge could not tell them apart.

```bash
python3 -m tokenlens eval --dry-run          # free: what would be compressed, and by how much
python3 -m tokenlens eval                    # the real thing (needs ANTHROPIC_API_KEY, costs money)
python3 -m tokenlens eval --rates 0.8,0.6,0.4 --tolerance 0.99 --out tokenlens-eval.json
python3 -m tokenlens eval --tasks my-traffic.jsonl --class extraction
```

Output is a compression-ratio vs quality curve, one row per rung/rate, and then
the thing you actually wanted — a **policy**: per task class, the most aggressive
setting whose mean quality still lands within tolerance (99% of cleartext by
default). If nothing clears the bar for a class, the policy for that class is
`none`, and that is a real answer, not a failure.

Costs: about `tasks × (1 + 2 × arms)` model calls. Token counting is free
(`count_tokens`); the completions and judge calls are not.

**Bring your own golden set.** The bundled 10-task set
(`tokenlens/eval/goldens/default.jsonl`) spans long-doc QA, extraction,
summarization, chat history, instruction-following, numeric reasoning, and one
code case that must come back untouched. It is a smoke test for the harness, not
a benchmark for your workload. A golden task is four fields:

```json
{"id": "…", "task_class": "extraction", "question": "…", "context": "…", "reference": "…"}
```

### Judging live traffic (`serve --judge`)

The same judge can run in shadow mode against real requests:

```bash
python3 -m tokenlens serve --compress llmlingua2 --rate 0.6 --measure --judge
```

On a sampled fraction of the requests it actually compressed (25% by default),
TokenLens replays the **original, uncompressed** prompt upstream, has the judge
grade both answers, and shows **quality retained** on the dashboard next to
tokens saved — plus a running feed of the judge's one-line notes, so a regression
tells you *what* it dropped.

This **costs money**: every judged request is one extra full completion plus one
judge call. It is off by default, sampled, and only ever fires on requests that
were actually compressed. Turn it on to calibrate or to watch for drift, not to
leave running forever. Point `serve --eval-report tokenlens-eval.json` at a
report to also show the calibration curve on the dashboard.

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

> The Claude Code routing path isn't covered by automated tests. If it errors
> with a custom base URL, open an issue with the message.

---

## Dashboard & endpoints

| Path | What |
|------|------|
| `GET /tokenlens/` | Live telemetry dashboard (open in a browser) |
| `GET /tokenlens/feed?limit=50` | JSON feed powering the dashboard |
| `GET /tokenlens/stats` | Cumulative totals as JSON (incl. `quality_retained_pct`) |
| `POST /v1/messages` | Proxied to the upstream Anthropic API |

---

## Privacy & security

- **BYOK** — your API key is forwarded to Anthropic and never stored or written
  to disk. `--judge` borrows the key from the request it arrived on, for the life
  of that request only.
- **No content logging** — only token counts, status, latency, and cost are
  recorded. Prompts and responses are never logged. The one exception is
  `--judge`, which you opt into: the judge's one-line note about how the two
  answers differed is derived from response content and appears in your terminal
  and your local dashboard. It never leaves your machine.
- **`--judge` sends more, not less** — a judged request replays the *original,
  uncompressed* prompt upstream and adds a judge call. Same destination as the
  request you already made, at roughly double the cost.
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

- **Feed the policy back into the proxy** — `tokenlens eval` produces a per-task-class
  policy; the proxy can't yet read it back and apply a rate per request. Today you
  read the curve and set `--rate` yourself.
- **A golden set worth the name** — the bundled 10 tasks prove the harness runs. Real
  calibration needs tasks shaped like your traffic.
- More compression rungs (structural minify, dedup, retrieval/selection) — each one
  now has a bar to clear before it ships.
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

Better compressor rungs are especially welcome — and now there's a way to prove
they work: `tokenlens eval` has to say yes before a rung is worth shipping.
Golden tasks (`tokenlens/eval/goldens/default.jsonl`) are welcome too.

---

## Disclaimer

TokenLens is an independent, community project. It is **not affiliated with,
endorsed by, or sponsored by Anthropic**. "Claude", "Claude Code", and
"Anthropic" are referenced only to describe compatibility.

Use TokenLens with your own credentials and in accordance with your provider's
terms of service and acceptable use policy. You are responsible for how you use
it. The software is provided "as is", without warranty of any kind (see the MIT
license below).

## License

MIT — see [`LICENSE`](./LICENSE).
