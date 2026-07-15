# TokenLens — Design

**A compression router/gateway that cuts frontier-model token cost while holding output quality equal to cleartext.**

---

## 1. Problem & vision

Frontier LLM APIs bill **per token**, and the model consumes **tokens, not bytes**.
Classic byte compression (gzip/zstd/base64) is therefore useless: the model can't
decode it, and decoding it ourselves before sending saves zero tokens. Any real
saving must come from sending a **shorter token sequence that the frontier model
still interprets as well as the original**.

TokenLens is not a single algorithm — it is a **gateway** that sits between an
application and the frontier API. For each request it:

1. Classifies the *ask* (task type, structure, risk tolerance).
2. Selects the optimal compression policy for that class.
3. Applies compression.
4. Routes to the chosen model.
5. Logs tokens saved and a quality signal, closing the loop.

**Guarantee we sell:** predictable token savings with a measured promise that
output quality never drops below a configured tolerance of cleartext.

```
app ──▶ TokenLens gateway ──▶ frontier model ──▶ response ──▶ app
              │
              ├─ 1. classify the "ask"
              ├─ 2. pick compression policy for that class
              ├─ 3. apply compression (ladder rungs 0–4)
              ├─ 4. route to chosen model
              └─ 5. log tokens saved + quality signal
```

---

## 2. Two loops

**Serving loop (online, milliseconds, transparent):**
classify → compress → route. Drop-in proxy that mirrors the OpenAI/Anthropic
request/response shape so existing apps point at TokenLens with a base-URL change.

**Calibration loop (offline):**
run the eval harness across task classes to learn *how far each class can be
compressed before quality drops*. Output is the **policy table** the serving loop
reads. This is what makes compression "optimal for the ask" rather than guessed.

---

## 3. Compression ladder

Ordered safe → aggressive. The router picks the highest rung a task class
tolerates (proven by the eval harness).

| Rung | Technique | Model needed? | Typical saving | Risk |
|------|-----------|---------------|----------------|------|
| 0 | **Normalize** — collapse whitespace/newlines, strip decorative separators | No | low | ~none |
| 1 | **Structural minify** — compact JSON/markdown/tables/code, strip HTML boilerplate | No | low–med | low (unless formatting is semantic) |
| 2 | **Dedup** — remove exact/near-exact repeated blocks & redundant boilerplate | No | med | low |
| 3 | **Statistical prune** — drop low-information tokens/sentences by entropy/TF-IDF, optionally query-aware | No (statistical) | med–high | needs eval per class |
| 4 | **Selection (RAG-style)** — keep only query-relevant chunks of large context | No (retrieval) | high | needs eval per class |
| 5 | **Model-scored prune (LLMLingua-style)** — tiny local model scores token importance; drop lowest | **Yes, tiny local model** | **high (3–5×)** | needs eval per class |

Rungs 0–2 are the **safe universal floor** (always applied). Rungs 3–5 are
**gated behind eval results** — only enabled for a class where measured quality
stays within tolerance.

### Notes on the tiny local scorer (rung 5)
- A small (~0.5–3B) local LM estimates per-token information content; low-value
  tokens (filler, redundancy, boilerplate) are dropped. LLMs are robust to
  telegraphic, partially-deleted text — this is the state of the art
  (LLMLingua / LLMLingua-2 / LongLLMLingua).
- Economics: pay cheap local inference to save expensive frontier tokens — the
  math favors us strongly at scale.
- **Question-aware mode:** when the downstream query is known, score tokens
  relative to it. Beats task-blind compression and reduces "lost in the middle."
- **Out of scope:** learned soft-prompt / "gist token" compression (Gisting,
  AutoCompressor) requires access to model internals or fine-tuning of the
  *target* model — impossible against a closed frontier endpoint.

---

## 4. Measuring "same performance as cleartext"

Quality measurement is the definition of "optimal," so it is built **first**.

- **Calibration set per task class:** (question + input + checkable answer).
- For each rung: run compressed vs. cleartext through the target model, score
  both, record the **compression-ratio-vs-accuracy curve**.
- **Policy = highest rung where accuracy stays within tolerance** (e.g. ≥99% of
  cleartext) for that class.
- **Production drift check:** sample a fraction of live traffic through both
  paths (shadow mode) to detect regression.

Scoring options (start with the first, add others as needed):
- Automatic task accuracy (QA / extraction / classification with ground truth) — most rigorous.
- LLM-as-judge similarity between compressed-input and cleartext-input answers.
- Exact/semantic match, ROUGE/embedding similarity for generative tasks.

---

## 5. Architecture (components)

- **Ingress / proxy** — API-compatible endpoint; parses request, extracts the
  compressible parts (system prompt, context, history) vs. must-not-touch parts.
- **Classifier** — labels the ask into a task class (fast, rule/statistical to
  start; upgradeable).
- **Policy store** — task class → {rungs enabled, params, target model}. Written
  by the calibration loop, read by serving.
- **Compressor** — the ladder (rungs 0–5) as a standalone, independently testable
  library.
- **Router** — picks target model, forwards, streams response back.
- **Telemetry** — per-request tokens in/out (pre & post compression), $ saved,
  latency, and (sampled) quality signal.
- **Eval harness** — offline runner producing ratio/accuracy curves and policies.

---

## 5a. Deployment strategy (resolved)

One engine, three form factors. The compression + routing core is a **library**;
every deployment is a thin wrapper around it:

- **Local runner (built first)** — the library wrapped in a local proxy/CLI the
  user points their own frontier API key at (BYOK). Runs entirely on the user's
  machine; prompts and keys never leave it except to go to the frontier provider.
  This is what lets us (and early testers) run real requests end-to-end and see
  actual token savings before any hosted infrastructure exists.
- **Self-hosted proxy library** — the same core dropped into a developer's stack
  as middleware. Lowest trust barrier; the primary developer/business offering.
- **Hosted SaaS gateway (later)** — the library + hosting + billing for teams who
  won't self-host. Where a "cut of measured savings" business model lives.
- **Desktop app (optional later)** — the local runner + GUI for BYOK power users.

**Why local-first:** a local runner has zero trust barrier (nothing leaves the
machine) and is the fastest path to a testable, real product. The rung-5 tiny
local scorer runs comfortably on desktop hardware.

**Who benefits.** Compression helps any client billed by tokens — *provided the
client can be pointed at the proxy*. Per-token / API users (including BYOK) pay
by the token, so compressing the request cuts cost directly.

The reachability caveat is separate from the economics: a client only benefits
if it exposes a base-URL / proxy hook. **Claude Code honors `ANTHROPIC_BASE_URL`**,
so it can route through the proxy. The **consumer Claude desktop chat app** has
**no proxy hook** and talks to claude.ai directly, so its traffic can't be
intercepted.

## 6. MVP milestones

1. ✅ **Local runner skeleton** — a local proxy/CLI (BYOK) that forwards requests
   unchanged to the frontier provider and logs token counts. Runnable and
   testable by the user on their own machine. Proves plumbing + measures baseline
   savings potential.
2. ✅ **Rungs 0–2** — the safe, always-wins compression tier.
3. ✅ **Eval harness + one calibration set** — `tokenlens eval`: calibration tasks run
   cleartext and compressed, an LLM judge grades both answers blind, out comes a
   ratio/quality curve and a per-task-class policy. The same judge runs in shadow
   mode on live traffic (`serve --judge`) and surfaces on the dashboard. The
   bundled 10-task set is a starter, not a benchmark.
4. **Classifier + policy table** — the actual router. The harness now *emits* a
   policy per task class; the proxy cannot yet read one back and apply a rate per
   request. That gap is the next piece of work.
5. **Rungs 3–5** — statistical, selection, and tiny-local-model pruning, each
   gated behind eval results. (Rung 5, LLMLingua-2, exists but is ungated: it
   ships behind an explicit `--rate` you set yourself.)

---

## 7. Open questions

- Which frontier model(s) are the primary routing targets?
- Which task classes matter first (RAG/long-doc QA, chat history, code, extraction)?
- Tiny local scorer: which model, and CPU vs. GPU serving budget?
- Latency budget per request (caps how heavy rung 3–5 can be online)?

---

## 8. Key design decisions (locked so far)

- **Product shape:** compression *router/gateway*, not a bare algorithm.
- **Compression must be token-level** and read directly by the frontier model
  (family A); no decompress-before-send.
- **Safe floor always on** (rungs 0–2); aggressive rungs gated by measured quality.
- **Tiny local scorer is in scope** for the high-value pruning tier.
- **Eval harness is built first** — it defines "optimal."
- **Deployment: one library core, deployed local-first** (BYOK local runner the
  user can test on their own machine), then self-hosted proxy, then hosted SaaS,
  with a desktop GUI as an optional later form. Target audience is any client
  billed by tokens that can be pointed at the proxy (per-token / API, incl. BYOK).
