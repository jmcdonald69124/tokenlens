"""The calibration runner: calibration tasks in, ratio-vs-quality curve and policy out.

For every calibration task the harness builds one request and sends it twice —
once cleartext, once through the real `compress_request()` the proxy uses — and
hands both answers to the judge. Nothing here re-implements compression; if the
harness and the proxy ever disagree about what gets compressed, the harness is
lying, so it calls the same function.

The request shape is deliberately the shape the proxy has to survive in
production: a cached system prefix (a `cache_control` breakpoint) followed by a
volatile user turn. That means the eval exercises the cache-boundary logic, not
just the pruner.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from ..compress import compress_request, llmlingua2
from ..compress.estimate import estimate_tokens
from . import judge as judge_mod
from .api import ApiError, Client, text_of
from .tasks import CalibrationTask

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_TOLERANCE = 0.99
DEFAULT_MAX_TOKENS = 1024

# A control arm that loses more than this much quality while compressing nothing
# is not an instrument, it is a coin. Correcting for a floor that large would
# widen the pass band until every arm fits inside it, so refuse to certify the
# class at all and say why. More --repeats is the only way out.
MAX_USABLE_FLOOR = 0.10

# Neutral: the harness must not coach the model into being robust to
# compression, or it measures the system prompt instead of the compressor.
SYSTEM_PROMPT = (
    "Answer the user's request using the material they provide. "
    "Follow every instruction in the request exactly. Be accurate and concise."
)


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

    method: str          # "control" | "safe" | "llmlingua2"
    rate: float = 1.0

    @property
    def label(self) -> str:
        if self.method == "llmlingua2":
            return f"llmlingua2 r={self.rate:g}"
        return self.method

    @property
    def is_control(self) -> bool:
        return self.method == "control"

    @classmethod
    def parse(cls, spec: str) -> "Arm":
        """'safe' or 'llmlingua2@0.6'."""
        spec = spec.strip()
        if "@" in spec:
            method, _, rate = spec.partition("@")
            return cls(method.strip(), float(rate))
        return cls(spec)


@dataclass
class CaseResult:
    task_id: str
    task_class: str
    arm: str
    method: str
    rate: float
    baseline_tokens: int = 0
    compressed_tokens: int = 0
    cleartext_grade: int = 0
    compressed_grade: int = 0
    retention: float = 1.0
    eligible_blocks: int = 0
    compressed_blocks: int = 0
    note: str = ""
    error: str | None = None
    repeats: int = 1
    retentions: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def spread(self) -> float:
        """Widest disagreement between repeats of the same cell — the noise."""
        if len(self.retentions) < 2:
            return 0.0
        return round(max(self.retentions) - min(self.retentions), 4)

    @property
    def saved(self) -> int:
        return max(0, self.baseline_tokens - self.compressed_tokens)

    @property
    def savings_pct(self) -> float:
        if not self.baseline_tokens:
            return 0.0
        return 100.0 * self.saved / self.baseline_tokens


@dataclass
class ArmStat:
    """One row of the compression-ratio vs quality curve."""

    arm: str
    method: str
    rate: float
    n: int
    baseline_tokens: int
    compressed_tokens: int
    savings_pct: float
    mean_retention: float
    worst_retention: float
    mean_cleartext_grade: float
    mean_compressed_grade: float
    degraded: int          # cases below the threshold
    passes: bool           # mean retention at or above the threshold
    untouched: int         # cases where nothing was eligible for compression
    is_control: bool = False
    mean_spread: float = 0.0     # mean within-cell disagreement across repeats
    resolved: bool = True        # is this arm's loss bigger than the noise floor?
    threshold: float = 0.0       # the retention this arm actually had to clear


@dataclass
class EvalReport:
    model: str
    judge_model: str
    tolerance: float
    tasks: int
    generated_at: float
    cases: list[CaseResult]
    curve: list[ArmStat]
    by_class: dict[str, list[ArmStat]]
    policy: dict[str, dict]
    errors: list[str] = field(default_factory=list)
    repeats: int = 1
    task_catalog: list[dict] = field(default_factory=list)

    @property
    def noise_floor(self) -> float:
        return noise_floor(self.curve)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "model": self.model,
            "judge_model": self.judge_model,
            "tolerance": self.tolerance,
            "tasks": self.tasks,
            "repeats": self.repeats,
            "noise_floor": self.noise_floor,
            "task_catalog": self.task_catalog,
            "curve": [asdict(s) for s in self.curve],
            "by_class": {k: [asdict(s) for s in v] for k, v in self.by_class.items()},
            "policy": self.policy,
            "cases": [
                {**asdict(c), "saved": c.saved, "savings_pct": round(c.savings_pct, 2)}
                for c in self.cases
            ],
            "errors": self.errors,
        }


# --- request construction -------------------------------------------------
def build_body(task: CalibrationTask, model: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> bytes:
    """A cached system prefix plus a volatile user turn — production shape."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": task.context},
                {"type": "text", "text": task.question},
            ],
        }],
    }
    return json.dumps(payload).encode("utf-8")


def _payload_text(payload: dict) -> str:
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


def _count(client: Client, payload: dict) -> int:
    """Ground truth from count_tokens (free); local estimate if it fails."""
    n = client.count_tokens(payload)
    if n is None:
        return estimate_tokens(_payload_text(payload))
    return n


# --- the run --------------------------------------------------------------
def run_eval(
    tasks: list[CalibrationTask],
    client: Client,
    *,
    arms: list[Arm],
    model: str = DEFAULT_MODEL,
    judge_model: str = judge_mod.DEFAULT_JUDGE_MODEL,
    tolerance: float = DEFAULT_TOLERANCE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    concurrency: int = 4,
    repeats: int = 1,
    progress: bool = True,
) -> EvalReport:
    started = time.time()
    errors: list[str] = []
    repeats = max(1, repeats)

    def log(msg: str) -> None:
        if progress:
            print(f"[eval] {msg}", file=sys.stderr, flush=True)

    def run_task(task: CalibrationTask) -> list[CaseResult]:
        """Baseline once per repeat, then every arm against it.

        Each repeat gets its own cleartext answer. Pairing repeat i's compressed
        answer with repeat i's cleartext answer keeps the comparison honest: both
        sides carry the model's sampling variance, so the ratio isn't measuring
        one lucky baseline.
        """
        body = build_body(task, model, max_tokens)
        payload = json.loads(body)

        try:
            baseline_tokens = _count(client, payload)
            cleartext_answers = [
                text_of(client.messages(payload)) for _ in range(repeats)
            ]
        except ApiError as e:
            errors.append(f"{task.id}: baseline failed ({e})")
            log(f"{task.id}: baseline FAILED — {e}")
            return [
                CaseResult(task.id, task.task_class, a.label, a.method, a.rate,
                           error=f"baseline failed: {e}")
                for a in arms
            ]

        results = []
        for arm in arms:
            try:
                results.append(_run_arm(
                    task, arm, body, baseline_tokens, cleartext_answers,
                    client=client, judge_model=judge_model,
                ))
            except ApiError as e:
                errors.append(f"{task.id} / {arm.label}: {e}")
                log(f"{task.id} / {arm.label}: FAILED — {e}")
                results.append(CaseResult(
                    task.id, task.task_class, arm.label, arm.method, arm.rate,
                    baseline_tokens=baseline_tokens, error=str(e),
                ))
        done = [f"{r.arm} {r.savings_pct:.0f}%/{r.retention:.0%}" for r in results if r.ok]
        log(f"{task.id}: {', '.join(done) if done else 'no successful arms'}")
        return results

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        cases = [c for batch in pool.map(run_task, tasks) for c in batch]

    curve = build_curve(cases, tolerance)
    by_class = {
        cls: build_curve([c for c in cases if c.task_class == cls], tolerance)
        for cls in sorted({c.task_class for c in cases})
    }
    return EvalReport(
        model=model,
        judge_model=judge_model,
        tolerance=tolerance,
        tasks=len(tasks),
        generated_at=started,
        cases=cases,
        curve=curve,
        by_class=by_class,
        policy=build_policy(by_class, curve, tolerance),
        errors=errors,
        repeats=repeats,
        task_catalog=[t.summary for t in tasks],
    )


def _run_arm(
    task: CalibrationTask,
    arm: Arm,
    body: bytes,
    baseline_tokens: int,
    cleartext_answers: list[str],
    *,
    client: Client,
    judge_model: str,
) -> CaseResult:
    repeats = len(cleartext_answers)

    if arm.is_control:
        # Compress nothing. Sample a second cleartext answer and judge it against
        # the first. Retention here *should* be 100%; whatever comes back is the
        # instrument's noise floor, and every other row has to be read against it.
        payload = json.loads(body)
        judgements = []
        for i in range(repeats):
            rival = text_of(client.messages(payload))
            judgements.append(judge_mod.grade(
                client,
                case_key=f"{task.id}:control:{i}",
                request_text=f"{task.context}\n\n{task.question}",
                reference=task.reference,
                cleartext_answer=cleartext_answers[i],
                compressed_answer=rival,
                judge_model=judge_model,
            ))
        return _fold(task, arm, baseline_tokens, baseline_tokens, judgements,
                     eligible=0, compressed=0,
                     note="control — cleartext judged against cleartext")

    # The same call the proxy makes — no harness-only compression path exists.
    result = compress_request(body, method=arm.method, rate=arm.rate)
    payload = json.loads(result.body)
    compressed_tokens = _count(client, payload)

    if result.compressed_blocks == 0:
        # Nothing was eligible (non-prose, or frozen behind the cache boundary).
        # The request goes upstream byte-identical, so quality is cleartext
        # quality by construction — spending a judge call to confirm that is a
        # waste of the user's money.
        return CaseResult(
            task.id, task.task_class, arm.label, arm.method, arm.rate,
            baseline_tokens=baseline_tokens,
            compressed_tokens=compressed_tokens,
            cleartext_grade=100, compressed_grade=100, retention=1.0,
            eligible_blocks=result.eligible_blocks,
            compressed_blocks=0,
            note="untouched — nothing eligible for compression",
            repeats=repeats, retentions=[1.0] * repeats,
        )

    judgements = []
    for i in range(repeats):
        compressed_answer = text_of(client.messages(payload))
        judgements.append(judge_mod.grade(
            client,
            # the repeat index is in the key, so the A/B position reshuffles
            # between repeats and position bias averages out rather than
            # compounding.
            case_key=f"{task.id}:{arm.label}:{i}",
            request_text=f"{task.context}\n\n{task.question}",
            reference=task.reference,
            cleartext_answer=cleartext_answers[i],
            compressed_answer=compressed_answer,
            judge_model=judge_model,
        ))
    return _fold(task, arm, baseline_tokens, compressed_tokens, judgements,
                 eligible=result.eligible_blocks,
                 compressed=result.compressed_blocks)


def _fold(task: CalibrationTask, arm: Arm, baseline_tokens: int, compressed_tokens: int,
          judgements: list, *, eligible: int, compressed: int,
          note: str = "") -> CaseResult:
    """Average the repeats of one cell into a single case."""
    n = len(judgements)
    retentions = [round(j.retention, 4) for j in judgements]
    # The note comes from the worst repeat: if quality moved, that's the one that
    # says why. A note from the best repeat would flatter the arm.
    worst = min(judgements, key=lambda j: j.retention)
    return CaseResult(
        task.id, task.task_class, arm.label, arm.method, arm.rate,
        baseline_tokens=baseline_tokens,
        compressed_tokens=compressed_tokens,
        cleartext_grade=round(sum(j.cleartext_grade for j in judgements) / n),
        compressed_grade=round(sum(j.compressed_grade for j in judgements) / n),
        retention=round(sum(retentions) / n, 4),
        eligible_blocks=eligible,
        compressed_blocks=compressed,
        note=note or worst.note,
        repeats=n,
        retentions=retentions,
    )


# --- aggregation ----------------------------------------------------------
def build_curve(cases: list[CaseResult], tolerance: float) -> list[ArmStat]:
    """Collapse cases into one row per arm, ordered most to least aggressive."""
    stats: list[ArmStat] = []
    by_arm: dict[str, list[CaseResult]] = {}
    for c in cases:
        if c.ok:
            by_arm.setdefault(c.arm, []).append(c)

    # The control arm has to be scored before anything else can be, because it
    # sets the bar the others are held to. It is measured on exactly these cases,
    # so a class with a steady judge and a class with a wild one get different
    # bars — which is the point. The floor is a property of the task, not of the
    # compressor.
    control = next((g for g in by_arm.values() if g[0].method == "control"), None)
    floor = 0.0
    if control:
        floor = round(1.0 - sum(c.retention for c in control) / len(control), 4)
    threshold = quality_threshold(tolerance, floor)

    for arm, group in by_arm.items():
        n = len(group)
        base = sum(c.baseline_tokens for c in group)
        comp = sum(c.compressed_tokens for c in group)
        retentions = [c.retention for c in group]
        mean_ret = sum(retentions) / n
        stats.append(ArmStat(
            arm=arm,
            method=group[0].method,
            rate=group[0].rate,
            n=n,
            baseline_tokens=base,
            compressed_tokens=comp,
            savings_pct=round(100.0 * (base - comp) / base, 2) if base else 0.0,
            mean_retention=round(mean_ret, 4),
            worst_retention=round(min(retentions), 4),
            mean_cleartext_grade=round(sum(c.cleartext_grade for c in group) / n, 1),
            mean_compressed_grade=round(sum(c.compressed_grade for c in group) / n, 1),
            degraded=sum(1 for c in group if c.retention < threshold),
            passes=mean_ret >= threshold,
            untouched=sum(1 for c in group if c.compressed_blocks == 0),
            is_control=group[0].method == "control",
            mean_spread=round(sum(c.spread for c in group) / n, 4),
            threshold=threshold,
        ))

    # Anything whose quality loss is smaller than the control arm's own loss has
    # not been measured — the instrument cannot see a difference that small. A
    # floor of zero is the exception: a judge that reproduced itself exactly can
    # resolve anything, including an arm that lost nothing.
    for s in stats:
        s.resolved = s.is_control or floor <= 0 or (1.0 - s.mean_retention) > floor

    # Most aggressive first: biggest token reduction at the top.
    stats.sort(key=lambda s: -s.savings_pct)
    return stats


def quality_threshold(tolerance: float, floor: float) -> float:
    """The retention an arm has to clear *on this instrument*.

    The tolerance is stated against a perfect judge: keep 99% of cleartext
    quality. This judge is not perfect. The control arm compressed nothing — its
    true retention is 1.0 by construction — and still scored `1 - floor`. Holding
    a compressed arm to a standard the *uncompressed* arm cannot meet is not
    strictness, it is measuring the judge and billing the compressor for it.

    So the bar drops by the floor, and only the loss beyond it is charged to
    compression. On a class where the judge is steady (floor 0) this is exactly
    the tolerance the user asked for, and nothing changes.
    """
    return max(0.0, min(1.0, tolerance - floor))


def noise_floor(stats: list[ArmStat]) -> float:
    """How much quality the judge 'loses' when nothing was compressed at all.

    The control arm compresses nothing, so its true retention is 1.0 by
    construction. Whatever it actually measured is error: the model's sampling
    variance plus the judge's inconsistency. That gap is the smallest difference
    this instrument can resolve.
    """
    for s in stats:
        if s.is_control:
            return round(1.0 - s.mean_retention, 4)
    return 0.0


def build_policy(by_class: dict[str, list[ArmStat]], overall: list[ArmStat],
                 tolerance: float) -> dict[str, dict]:
    """The whole point of the harness: the most aggressive arm we can defend.

    Two guards stop noise being promoted into a production setting.

    *Monotonicity.* Quality cannot improve as you delete more tokens, so we walk
    the curve from the gentlest arm upward and stop at the first failure. An arm
    that "passes" on the far side of a failure is a fluke, not a policy, and
    picking it — as the naive `max(savings)` did — is how a lucky sample becomes
    the rate you ship.

    *Resolution.* An arm whose quality loss is smaller than the control arm's own
    loss has not been shown to be safe; it has been shown to be unmeasurable at
    this sample size. It can still be adopted — it was not shown to be worse than
    doing nothing, and doing nothing is the best case — but it ships marked
    provisional, never as a measured result.

    *Usable instrument.* And if the control arm itself lost more than
    MAX_USABLE_FLOOR, none of the above means anything: refuse the class outright
    rather than hand back a number the run cannot support.
    """
    policy: dict[str, dict] = {}
    for name, stats in list(by_class.items()) + [("default", overall)]:
        floor = noise_floor(stats)
        rungs = [s for s in stats if not s.is_control]
        bar = quality_threshold(tolerance, floor)

        if floor > MAX_USABLE_FLOOR:
            policy[name] = {
                "method": "none",
                "rate": None,
                "reason": (
                    f"unmeasurable — the control arm compressed nothing and still "
                    f"lost {floor:.0%}. This judge cannot resolve a "
                    f"{1 - tolerance:.0%} tolerance here at any compression rate"
                ),
                "savings_pct": 0.0,
                "mean_retention": None,
                "noise_floor": floor,
                "unmeasurable": True,
            }
            continue

        # Gentlest first. Stop at the first arm that fails: nothing beyond it is
        # trustworthy, however well it happened to score.
        best = None
        blocked = False
        for s in sorted(rungs, key=lambda s: s.savings_pct):
            if not s.passes:
                blocked = True
                break
            best = s
        skipped = [s.arm for s in rungs
                   if blocked and s.passes and best and s.savings_pct > best.savings_pct]

        if best is None or best.savings_pct <= 0:
            policy[name] = {
                "method": "none",
                "rate": None,
                "reason": (f"no arm held quality at or above {bar:.0%} of cleartext"
                           + (f" (a {tolerance:.0%} tolerance, less the {floor:.0%} "
                              f"the control arm lost on its own)" if floor else "")),
                "savings_pct": 0.0,
                "mean_retention": None,
                "noise_floor": floor,
            }
            continue

        entry = {
            "method": best.method,
            "rate": best.rate if best.method == "llmlingua2" else None,
            "arm": best.arm,
            "savings_pct": best.savings_pct,
            "mean_retention": best.mean_retention,
            "worst_retention": best.worst_retention,
            "n": best.n,
            "noise_floor": floor,
            "threshold": bar,
            "confidence": "measured" if best.resolved else "provisional",
        }
        if not best.resolved and floor > 0:
            entry["caveat"] = (
                f"provisional — it lost {1 - best.mean_retention:.1%}, which is "
                f"inside the {floor:.1%} noise floor. That is not a clean bill of "
                f"health, it is a failure to detect harm: adopt it and you are "
                f"betting the loss really is near zero"
            )
        if skipped:
            entry["ignored"] = skipped
            entry["ignored_reason"] = (
                f"{', '.join(skipped)} scored higher while deleting more — "
                f"impossible, so treated as noise, not policy"
            )
        policy[name] = entry
    return policy


def default_arms(rates: list[float]) -> list[Arm]:
    """The control, the safe floor, and one pruning arm per requested keep-rate."""
    arms = [Arm("control"), Arm("safe")]
    if llmlingua2.available():
        arms += [Arm("llmlingua2", r) for r in rates]
    return arms
