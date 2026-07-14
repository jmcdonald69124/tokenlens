# You can't compress what you can't grade

*Adding an eval harness to TokenLens — and why it had to come before the
interesting compression. A follow-up to "I Built a Prompt Compressor. The Most
Useful Thing It Found Was Nothing."*

---

The last piece ended on a promise: *"TokenLens has no output-quality eval
harness, and building one is the next thing on the roadmap. Until then I keep the
keep-rate conservative on anything that matters."*

This is that harness. It found four things. Two of them were mistakes in the
article you just read, and the fourth one nearly invalidated the harness itself.

## The number that lies

TokenLens is a local proxy that sits between your Anthropic client and the API,
compresses the parts of your prompt that are safe to compress, and shows you a
dashboard of what it saved. Until this week that dashboard had a headline metric:

> **Tokens saved: 41,208**

It is a great number. It is also, on its own, worthless — because I can trivially
beat it. Set `--rate 0.1` and TokenLens will throw away 90% of your prompt. Set
`--rate 0.0` and it saves *everything*. The savings metric is maximised by a
compressor that sends nothing at all, and a metric you maximise by doing the
worst possible thing is not a metric, it is a temptation.

The number that was missing is the denominator: **did the answer get worse?**

If you have ever trained a classifier, you have already met this bug — it is
**recall with no precision**. Build a tumour detector that answers "yes" for
every scan and you have a model with *perfect recall*. It catches every case. It
is also useless, and nothing *inside* that number will ever tell you so. This is
what the confusion matrix is actually for: not thoroughness, but the fact that any
single cell of it can be maximised by a degenerate model, and only the cells *next
to* each other constrain the lie. Precision and recall aren't two facts about a
classifier. They're one fact, and either half on its own is propaganda.

Tokens saved is recall. It counts what the compressor removed and never asks what
it destroyed — and exactly like the classifier that says yes to everything, the
way to maximise it is to do the worst possible thing. Retention — *did the answer
survive?* — is the precision term. I had shipped a dashboard with one and not the
other.

The analogy runs deeper than the joke, and it turned out to be load-bearing.
`--rate` is a decision threshold: turn it down and the compressor deletes more
aggressively, exactly as dropping a classifier's threshold makes it fire more
freely. So the thing the harness prints — reduction on one axis, quality on the
other, one point per rate — **is a precision–recall curve**. The per-class policy
is the operating point. And the reason the curve has to be per class is the same
reason you never trust a single F1 across a skewed dataset: the trade-off is
different in each region, and an average over all of them describes nothing that
exists.

Which leaves one question the classifier analogy asks and I hadn't: *how noisy is
the label?* Hold that thought.

The README said so out loud. It shipped with a warning that read, in part:
"there is **no output-quality eval harness yet**, so don't turn `--rate` up on
important work and assume the result is identical." The design doc was blunter.
Milestone 3 was "eval harness + one golden task set," and buried in the locked
decisions was the line that should have been the first sentence of the project:

> **Eval harness is built first — it defines "optimal."**

It was not built first. It was built fifth, which is what this article is about.

## What "harness" actually means here

TokenLens has two loops. The **serving loop** is the proxy: it runs on the hot
path, it must never break your request, and it fails open on every error. The
**calibration loop** runs offline, costs money, and answers one question: *for
this task class, how hard can I compress before quality drops?*

The harness is the calibration loop, and it is simple to state:

```
golden task ──▶ cleartext request ──▶ model ──▶ answer A
            └─▶ compressed request ─▶ model ──▶ answer B
                                                    │
                       answer A + answer B + reference ──▶ judge ──▶ retention
```

Run every task through every rung of the compression ladder at every rate, and
you get a compression-ratio-versus-quality curve. From the curve you get the
thing you actually wanted, which is a **policy**: per task class, the most
aggressive setting whose quality still lands within tolerance of cleartext.

## The metric is retention, not quality

The first design mistake I nearly made was to grade the compressed answer on its
own. Score it 0–100, set a bar, ship it.

This is meaningless. A 71/100 is alarming if the cleartext answer scored 95 and
completely fine if the cleartext answer also scored 71 because the task is hard
and the model is only so good. Compression is never accused of the *absolute*
quality of an answer. It is only ever accused of the **difference**. So the judge
always sees both answers, from the same request, and the metric is a ratio:

```python
retention = grade(answer from compressed input) / grade(answer from cleartext input)
```

One wrinkle, in `judge.py`:

```python
@property
def retention(self) -> float:
    if self.cleartext_grade <= 0:
        # Cleartext scored zero — the task is broken, not the compression.
        return 1.0
    return min(1.0, self.compressed_grade / self.cleartext_grade)
```

Retention is **capped at 1.0**. Sometimes the compressed answer scores *higher*
than the cleartext one — pruning removed a distractor, or the model got lucky.
That is noise, and if you let it above 1.0, one lucky case silently pays for a
real regression somewhere else in the mean. Compression is not allowed to earn
credit. It is only allowed to avoid losing any.

## Designing a judge that can't cheat

An LLM-as-judge is a measuring instrument, and measuring instruments have to be
built so they cannot quietly agree with you. Four things matter.

**The judge is blind.** It is never told which answer came from the compressed
side. If you tell it, you are no longer measuring compression — you are measuring
the judge's prior about compression, which is that compression is probably bad.
There is a test for exactly this:

```python
def test_judge_never_sees_which_answer_was_compressed():
    ...
    prompt = client.judge_prompts[0].lower()
    assert "compress" not in prompt
```

**The A/B position is randomised, deterministically.** Judges have position bias.
So the compressed answer is shown as A or B based on a hash of the case key:

```python
def _swap_for(key: str) -> bool:
    return hashlib.sha256(key.encode("utf-8")).digest()[0] % 2 == 1
```

Position bias can no longer systematically favour one arm, and the shuffle is
identical on a rerun, so the eval is reproducible.

**The judge sees the *original* request.** Not the compressed one. This is the
subtle one. If you show the judge the pruned prompt, an answer that faithfully
reflects the pruned prompt looks *correct* — you have graded the compressed
answer against the compressed question and confirmed that the two agree. Of
course they agree. The whole point is to find out what the compression threw
away, so the judge grades both answers against what you *actually asked*, plus a
reference answer as ground truth.

**The judge has to think before it commits to a number.** The response is
constrained with a JSON schema, and `reasoning` is deliberately the first
property in it:

```python
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Two or three sentences comparing…"},
        "grade_a":   {"type": "integer", "description": "Quality of ANSWER A on a 0-100 scale…"},
        "grade_b":   {"type": "integer", "description": "…the same standard as ANSWER A."},
        "note":      {"type": "string", "description": "One short sentence naming the single…"},
    },
    "required": ["reasoning", "grade_a", "grade_b", "note"],
    "additionalProperties": False,
}
```

The schema fills in order, so the judge is made to articulate the comparison
before it can emit a grade. Grades come back clamped to 0–100 on our side,
because a schema can express "integer" but not "between 0 and 100."

## The golden set is the actual work

The code is a few hundred lines. The golden set is the part that decides whether
any of it means anything.

A golden task is four fields: a `question`, a `context` (the bulk prose the
compressor is allowed to chew on), a `reference` answer that a competent human
would accept, and a `task_class`. The class matters more than it looks. Long-doc
QA and extraction fail *completely differently* under pruning: a summary survives
losing a sentence, an extraction task dies the moment the pruner eats a date.
Averaging them produces a policy that is too aggressive for one and too timid for
the other, so the policy is chosen per class, and there is a test that says so:

```python
def test_policy_is_chosen_per_class_not_globally():
    """Extraction can be far more fragile than summarisation. Averaging the two
    is exactly the mistake the harness exists to prevent."""
```

The bundled set is ten tasks — long-doc QA, extraction, summarization, chat
history, instruction-following, numeric reasoning, and one code task. It is a
**smoke test for the harness, not a benchmark for your workload**, and the README
now says that in as many words. If you use TokenLens on your own traffic, the
golden set is the thing you should be writing, not the compressor.

That last code task earns its place. Its context is an indented Python function,
which TokenLens's `is_prose()` gate should refuse to touch. The correct result is
0% savings and 100% retention — the harness proving the *safety gate* works, not
the pruner. And when nothing was eligible, the harness doesn't spend your money
asking a judge to confirm the model agrees with itself:

```python
if result.compressed_blocks == 0:
    # The request went upstream byte-identical, so quality is cleartext quality
    # by construction — spending a judge call to confirm that is a waste.
```

## The harness must not have its own compressor

The one architectural rule: the harness calls `compress_request()` — the exact
function the proxy calls. It does not reimplement compression, it does not have a
"test mode" pipeline. If the harness and the proxy ever disagree about what gets
compressed, the harness is lying to you, and a lying eval is worse than no eval.

For the same reason, the request the harness builds has a `cache_control`
breakpoint on its system prompt. That is the shape real traffic has, and it means
the eval exercises the cache-boundary logic — the rule that the cached prefix is
frozen and only the volatile tail is compressible — rather than testing the
pruner in a vacuum where every byte is fair game.

## What it found before it graded a single answer

Three things, and the first two were free.

The dry run (`tokenlens eval --dry-run`, no API calls, no key needed) printed
this on my machine:

```
  task                class                 arm               est. tokens   blocks
  --------------------------------------------------------------------------------
  num-budget-01       numeric-reasoning     cleartext                 470        —
                                            safe                      470      0/2
                                            llmlingua2 r=0.8          400      2/2
                                            llmlingua2 r=0.6          323      2/2
                                            llmlingua2 r=0.4          220      2/2
  code-safety-01      code-context          cleartext                 188        —
                                            safe                      188      0/1
                                            llmlingua2 r=0.8          184      1/1
```

**The safe floor does nothing to clean prose.** `safe` saves 0 tokens on every
golden task, because it only strips whitespace and decorative junk, and my golden
contexts are tidy. That is correct behaviour, and it is also a quiet admission:
the "always-on, model-free safe floor" that ships by default is, on well-formed
input, a no-op. It earns its keep on messy real-world prompts, not on prose
someone wrote carefully.

**LLMLingua-2 was dead on arrival on any machine without an NVIDIA GPU.** The
first dry run reported `llmlingua2 unavailable (Torch not compiled with CUDA
enabled)`. `PromptCompressor` defaults to `device_map="cuda"` and hard-fails
otherwise, so the entire top rung of the compression ladder had been silently
falling back to the safe floor — which, per the previous paragraph, does nothing
— on every Mac and every CPU-only Linux box. The proxy's fail-open design meant
nobody's request ever broke. It also meant nobody ever found out. Four lines fixed
it:

```python
def _device() -> str:
    """PromptCompressor defaults to CUDA and hard-fails on machines without it.
    The model is small enough to run on CPU, which is where most people running
    a local proxy actually are, so fall back rather than lose the whole rung."""
```

Neither of these is a compression result. They are the harness catching the
project lying about what it was doing, which is what a harness is for, and it did
it before spending a cent on a model call.

The third finding is the one I owe you a correction for.

## Correcting the record: the 40% I never measured

The first article ended with a claim: *"Run one of those through the bench
command and the language-model codec prunes 40% or more of the tokens."*

That number was an **assumption, not a measurement**, and I should have said so
at the time. It came from the LLMLingua-2 literature and from what I expected the
model to do. It could not have come from my machine, because — per the finding
above — the model would not load on my machine. `bench` would have printed the
safe row and the words `llmlingua2 not installed`. I published a compression
figure produced by a compressor that never ran.

With the CUDA bug fixed, I ran it. Here is the real curve, on the text of that
very article, at the local `chars/4` estimate:

| `--rate` (keep fraction) | tokens | reduction |
|---|---|---|
| cleartext | 1,816 | — |
| 0.8 | 1,540 | 15.2% |
| 0.7 | 1,384 | **23.8%** |
| 0.6 | 1,228 | 32.4% |
| 0.5 | 1,057 | 41.8% |
| 0.4 | 856 | 52.9% |

So 40% is reachable — at `--rate 0.5`, which means **deleting half the tokens**.
The command I actually printed in that article was `--rate 0.7`, and `--rate 0.7`
gives 23.8%. The two numbers in my own article contradicted each other, and the
arithmetic was always going to say so: the rate *is* the keep fraction, so 0.7
cannot exceed ~30% reduction no matter how good the model is. Nobody caught it,
including me, because there was nothing in the repo whose job was to catch it.

## And the payoff case pays nothing

The first article said the savings live in "long documents, RAG context, a big
spec you paste in once." So I benched a big spec I actually paste in: this
project's own `DESIGN.md`.

```
  original: 2,563 tokens

  method          tokens     saved   reduction
  --------------------------------------------
  safe             2,563         0        0.0%
  llmlingua2 r=0.5 2,563         0        0.0%
```

Zero. At every rate. Because `is_prose()` — the safety gate that refuses to touch
anything that looks like code — disqualifies a block on a single code fence, or
on two "codey" characters, and a real spec is markdown: it has fenced blocks and
it has tables, and a table is a wall of pipe characters. One fence anywhere in
the document throws out the whole document.

That is the gate doing exactly what it was built to do, and it is also the
payoff case evaporating. The shape that pays is narrower than I claimed: not
"long documents," but *long unstructured natural-language prose* — a transcript,
an article, an email thread, raw retrieved passages. Put a markdown table in it
and TokenLens correctly, silently, saves you nothing.

Which is the same lesson as the first article, arriving by a different road. The
most useful thing the tool told me, again, was where not to bother. The
difference is that this time something in the repo was built to ask.

## The judge fails its own exam

Then I ran it, and got a beautiful curve. At keep-rate 0.6 the prompts came out
32.6% smaller and the judge said quality held at 95.7%. Thirty-two percent off,
four percent worse. That is a *shippable* trade, and I was about ten minutes from
writing it up as one.

What stopped me was a boring question: how much does that 95.7% move if I run it
again? I didn't know. I had one sample per cell. The model answers differently
every time you call it, and the judge grades differently every time you call it,
and I had built an instrument with no idea what its own error bar was.

So I added an arm that does nothing.

```python
@dataclass(frozen=True)
class Arm:
    """One rung of the ladder at one setting — a candidate policy.

    `control` is not a rung. It compresses nothing and asks the judge to compare
    two independently sampled cleartext answers to the same request. Its
    retention *should* be 100%; whatever it actually is, is the noise floor of
    the instrument — the model's own sampling variance plus the judge's
    inconsistency. Any arm scoring inside that band has not been measured, it has
    been guessed at. Without this row you cannot read the others.
    """
```

The control arm sends the prompt uncompressed, samples a *second* answer to the
same prompt, and asks the judge to grade the two against each other. Nothing was
deleted. Nothing was touched. The two answers came from the same model on the
same input, and the true retention is 100% by construction.

The judge gave it **93.8%**.

That is the whole ballgame. Six point two percent of "quality loss" was available
for free, on prompts that were never compressed, from a compressor that did not
run. My tolerance — the bar the entire harness was gating on — was one percent.
I had been measuring a 1% effect with an instrument whose needle wobbles by 6%.

Every number in the run had to be re-read against that:

| arm | reduction | quality | what it actually means |
|---|---|---|---|
| control | 0% | **93.8%** | the error bar — nothing was compressed |
| llmlingua2 r=0.8 | 14.1% | 95.2% | a 4.8% loss, *inside* the noise. Unresolved. |
| llmlingua2 r=0.6 | 33.9% | 90.5% | a 9.5% loss. Real, but only just. |
| llmlingua2 r=0.4 | 50.6% | 60.5% | a 40% loss against a 6% error bar. **Real.** |

The pretty result I nearly published — 32.6% smaller at 95.7% quality — was one
draw from a distribution I hadn't looked at. Run with `--repeats 3` it came back
as a 9.5% loss. It was never a shippable trade. It was a coin landing well.

This is the question I left hanging earlier, and it has a name in the older field
too: **label noise**. My judge is the annotator, and I had skipped the one thing
nobody running a labelling pipeline would dream of skipping — I never measured
inter-annotator agreement. Hand the same pair to the same grader twice and it
doesn't agree with *itself*. And once the labels are noisy there is a floor under
your error that no amount of model improvement gets beneath: you cannot resolve a
difference smaller than the disagreement in the thing doing the resolving.
Reporting a number below that floor isn't precision. It's fiction with decimal
places.

And the noise is not a constant. It is a property of the *task*:

| task class | control (should be 100%) |
|---|---|
| extraction, chat-history, code-context | **100.0%** |
| instruction-following | 97.5% |
| numeric-reasoning | 94.7% |
| long-doc-qa | 88.9% |
| summarization | **84.2%** |

Look at the bottom row. On summarization, the judge marked an *uncompressed*
answer down by 16% — and the compressed arm at keep-rate 0.8 scored 86.4%, which
is *better* than the control. On that class the noise doesn't just contaminate the
signal, it exceeds it. No result on summarization means anything, at any rate,
and the harness now says so instead of printing a number.

The classes at the top are the mirror image. Extraction has one right answer, the
model produces it consistently, the judge grades it consistently, and the control
comes back at a clean 100%. On that class a quality reading is worth something —
because I checked.

## Three guards, learned the hard way

The control arm didn't just add a row to the table; it invalidated the logic that
read the table. Three fixes came out of it.

**The bar has to move.** A 99% tolerance means "keep 99% of cleartext quality,"
which is a coherent thing to ask of a perfect judge and an incoherent thing to ask
of one that scores the uncompressed answer at 93.8%. Holding the compressed arm to
a standard the *uncompressed* arm cannot meet isn't strictness — it's measuring the
judge and billing the compressor for it. So the bar drops by the floor, per class,
and only the loss beyond it is charged to compression:

```python
def quality_threshold(tolerance: float, floor: float) -> float:
    return max(0.0, min(1.0, tolerance - floor))
```

On extraction, where the floor is zero, this changes nothing and the user gets
exactly the 99% they asked for. On numeric-reasoning the bar becomes 93.7%. That
asymmetry is correct: the classes with steady judges get held to a strict standard
because they *can* be.

**Quality cannot improve as you delete more tokens.** My first policy picker took
the arm with the biggest savings that passed. Given a noisy curve, it cheerfully
selected keep-rate 0.6 *over* keep-rate 0.8 on numeric-reasoning — an arm that
deleted more tokens and scored better, which is not a finding, it's an artifact.
The picker now walks the curve from the gentlest arm upward and stops dead at the
first failure. Anything that "passes" on the far side of a failure is noise, and
promoting it is how a lucky sample becomes the rate you ship.

**And an unmeasurable class gets refused, not certified.** If the correction above
were applied blindly to summarization, its 15.8% floor would drag the bar down to
83% and *everything* would pass. So there's a ceiling: a control arm that loses
more than 10% isn't an instrument, it's a coin, and the class comes back `none`
with the reason attached.

## From curve to policy

What survives all three guards is the payoff, and it is a much shorter list than
the one I started with:

```
  ✓ extraction            llmlingua2 rate=0.8   14.9% smaller at 99.8% quality
  ✓ code-context          llmlingua2 rate=0.6    2.9% smaller at 100.0% quality
  ? numeric-reasoning     llmlingua2 rate=0.8   14.0% smaller at 96.8% quality
                          provisional — inside the 5.3% noise floor
  ? instruction-following llmlingua2 rate=0.8   15.3% smaller at 99.0% quality
                          provisional — inside the 2.5% noise floor
  — long-doc-qa           none — unmeasurable, control lost 11%
  — summarization         none — unmeasurable, control lost 16%
  — chat-history          none — no arm held quality
```

One line in there is a real, defensible result: **extraction at keep-rate 0.8 —
14.9% smaller at 99.8% quality.** It is trustworthy for exactly one reason, which
is that extraction's control arm reads 100.0%. The instrument was clean when it
took that reading. That is the only sentence in this project I would put in
production.

The `?` rows are the interesting ones, and I want to be precise about what they
mean, because it is the easiest place in the whole system to lie to yourself. A
provisional pass does **not** mean "safe." It means *we could not detect harm* —
which is a statement about the instrument, not about the compressor. Adopting one
is a bet that the loss really is near zero. The report says so in those words,
every time, rather than printing a checkmark and letting you infer the rest.

And `none` being a legitimate output is the part I want to defend hardest. An eval
that can only ever tell you "yes, compress" is a rubber stamp. This one is allowed
to come back and say *don't* — on three of seven classes it did — and
`tokenlens eval` exits non-zero when no arm holds quality, so you can put it in CI
and let it fail the build.

The default policy, across everything, is a provisional 14% at keep-rate 0.8. The
first article shipped at keep-rate 0.7 and told you it was getting 40%. It was
getting 23.8%, at a quality cost nobody had measured, because nobody had built the
thing that could.

## Putting judgement on the dashboard

The offline harness tells you what *was* true when you calibrated. Traffic
drifts. So the same judge now runs in shadow mode on live traffic:

```bash
python3 -m tokenlens serve --compress llmlingua2 --rate 0.6 --measure --judge
```

On a sampled fraction of the requests it actually compressed, TokenLens replays
the original, uncompressed prompt upstream, hands both answers to the judge, and
the dashboard grows a **Quality retained** card sitting right next to *tokens
saved* — mean retention across everything judged, how many requests fell below
your tolerance, and a live feed of the judge's one-line notes, so a regression
tells you *what* it dropped rather than just that something got worse. Green when
compression is free. Red when it isn't.

This costs real money — a judged request is a full extra completion plus a judge
call, roughly double — so it is off by default, it is sampled, it only ever fires
on requests that were genuinely compressed, and the proxy prints a warning at
startup that says so in capital letters. A quality metric you can only get by
paying for it is an honest quality metric. The alternative is the one I started
with: a big purple number that goes up when you make the product worse.

One promise from the last article needs amending, narrowly and in public. I wrote
that "prompt bodies are never logged; it records counts, not content." With
`--judge` on, that stops being strictly true: the judge's one-line note is
derived from your content — *"one answer drops the deadline"* — and it is
displayed on the dashboard. It is the only content-derived thing TokenLens ever
surfaces. It is off by default, it never leaves your machine, and it exists
because a quality number that can't tell you *what* broke isn't much of a quality
number. But it is an exception to a sentence I published without one, so: there
it is.

---

*TokenLens is a local, BYOK, cache-aware token-measurement and compression proxy:
about fourteen hundred lines of dependency-free Python, and now a thousand more
that exist only to tell it when it is wrong. The harness is `tokenlens eval`; the
judge lives in `tokenlens/eval/judge.py`; the golden set is ten tasks and wants
to be yours.*

*Every quality number above comes from one run: ten synthetic tasks, three
repeats, Haiku 4.5 as both the model under test and the judge, 330 calls. That is
enough to establish that the noise floor exists and roughly how big it is. It is
not enough to pin any single cell to a decimal place, and a curve from my golden
set is not a curve from yours — the whole argument of this article is that
unmeasured compression is a guess, and someone else's measurement is just a more
expensive guess. Run it on tasks shaped like your traffic. Reduction percentages
are local `chars/4` estimates; the quality percentages are the judge's.*
