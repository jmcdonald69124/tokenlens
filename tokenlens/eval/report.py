"""`tokenlens eval` — run the harness, print the curve, write the report."""

from __future__ import annotations

import json
import sys
import textwrap
import time

from ..compress import compress_request, llmlingua2
from .api import ApiError, Client
from .harness import (
    Arm,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TOLERANCE,
    MAX_USABLE_FLOOR,
    EvalReport,
    build_body,
    default_arms,
    noise_floor,
    quality_threshold,
    run_eval,
)
from .judge import DEFAULT_JUDGE_MODEL
from .tasks import load_tasks

_BAR_WIDTH = 18


def _bar(fraction: float) -> str:
    filled = max(0, min(_BAR_WIDTH, round(fraction * _BAR_WIDTH)))
    return "█" * filled + "·" * (_BAR_WIDTH - filled)


def _wrap(text: str | None, width: int) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width)


def _verdict(s, floor: float) -> str:
    if s.is_control:
        return "— noise floor (nothing was compressed)"
    if s.untouched == s.n:
        # Byte-identical upstream: 100% is construction, not measurement.
        return "— untouched (nothing was eligible)"
    if not s.passes:
        return "✗ degrades quality"
    if not s.resolved and floor > 0:
        # It cleared the bar, but the bar is inside the error bar.
        return "? provisional — loss is inside the noise"
    verdict = "✓ within tolerance"
    if s.degraded:
        verdict += f" ({s.degraded} case{'s' if s.degraded > 1 else ''} below)"
    return verdict


def render(report: EvalReport) -> str:
    tol = report.tolerance
    floor = report.noise_floor
    out: list[str] = []
    out.append("")
    out.append(f"TokenLens eval — {report.tasks} tasks × {len(report.curve)} arms"
               f"{f' × {report.repeats} repeats' if report.repeats > 1 else ''}")
    out.append(f"  target: {report.model}   judge: {report.judge_model}   "
               f"tolerance: quality ≥ {tol:.0%} of cleartext")
    out.append("")
    out.append(f"  {'arm':<17}{'tokens':>9}{'saved':>9}{'reduction':>11}"
               f"{'quality':>10}  {'':<{_BAR_WIDTH}}  verdict")
    out.append(f"  {'-' * 76}")

    if report.curve:
        base = report.curve[0].baseline_tokens
        out.append(f"  {'cleartext':<17}{base:>9,}{'—':>9}{'—':>11}{'100%':>10}  "
                   f"{_bar(1.0)}  baseline")

    for s in report.curve:
        out.append(
            f"  {s.arm:<17}{s.compressed_tokens:>9,}"
            f"{s.baseline_tokens - s.compressed_tokens:>9,}"
            f"{s.savings_pct:>10.1f}%{s.mean_retention:>9.1%}  "
            f"{_bar(s.mean_retention)}  {_verdict(s, floor)}"
        )

    out.append("")
    out.append("  Quality is the judge's grade for the compressed answer as a fraction of")
    out.append("  its grade for the cleartext answer, on the same request. 100% means the")
    out.append("  judge could not tell the difference.")

    if floor > 0:
        bar = quality_threshold(tol, floor)
        out.append("")
        out.append(f"  ⚠ Noise floor: {floor:.1%}. The control arm compressed nothing and")
        out.append(f"    still lost {floor:.1%} of its quality, so that is this run's")
        out.append(f"    measurement error — the model answers differently each time and the")
        out.append(f"    judge grades differently each time. Your {1 - tol:.0%} tolerance is")
        out.append(f"    smaller than that error bar, so arms are scored against {bar:.1%} —")
        out.append(f"    the tolerance less the floor. Anything losing under {floor:.1%} is")
        out.append(f"    marked provisional: not proven equal, just not proven worse.")
        out.append(f"    More --repeats shrinks the floor; nothing else will.")

    # per-class curve
    out.append("")
    out.append("  By task class — each class is scored against its own noise floor")
    out.append(f"  {'-' * 76}")
    for cls, stats in report.by_class.items():
        cls_floor = noise_floor(stats)
        bar = quality_threshold(tol, cls_floor)
        if cls_floor > MAX_USABLE_FLOOR:
            head = f"  {cls}  — noise floor {cls_floor:.1%}: unmeasurable, no bar can be set"
        elif cls_floor > 0:
            head = f"  {cls}  — noise floor {cls_floor:.1%}, bar {bar:.1%}"
        else:
            head = f"  {cls}  — noise floor 0%, bar {bar:.1%} (a steady judge)"
        out.append(head)
        for s in stats:
            flag = "—" if s.is_control else ("✓" if s.passes else "✗")
            if not s.is_control and s.passes and not s.resolved and cls_floor > 0:
                flag = "?"
            if s.untouched == s.n and not s.is_control:
                flag = "—"
            untouched = "  (untouched — not eligible)" if s.untouched == s.n else ""
            out.append(f"    {flag} {s.arm:<15}{s.savings_pct:>6.1f}% smaller"
                       f"{s.mean_retention:>9.1%} quality{untouched}")
    out.append("")

    # the payoff
    out.append("  Policy — most aggressive setting that held quality")
    out.append(f"  {'-' * 76}")
    for cls, p in report.policy.items():
        if p["method"] == "none":
            out.append(f"  — {cls:<21} none — {p['reason']}")
        else:
            rate = f" rate={p['rate']:g}" if p.get("rate") else ""
            mark = "?" if p.get("confidence") == "provisional" else "✓"
            out.append(
                f"  {mark} {cls:<21} {p['method']}{rate:<12} "
                f"{p['savings_pct']:.1f}% smaller at {p['mean_retention']:.1%} quality "
                f"(worst case {p['worst_retention']:.0%})"
            )
        for line in _wrap(p.get("caveat"), 68):
            out.append(f"  {'':<23} {line}")
        if p.get("ignored"):
            for line in _wrap(f"ignored {p['ignored_reason']}", 68):
                out.append(f"  {'':<23} {line}")
    out.append("")
    out.append("  ✓ measured   ? provisional (loss is inside the noise floor)   — no policy")
    out.append("")

    if report.errors:
        out.append(f"  {len(report.errors)} case(s) errored:")
        for e in report.errors[:10]:
            out.append(f"    ! {e}")
        out.append("")
    return "\n".join(out)


def render_dry_run(tasks, model: str, arms: list[Arm]) -> str:
    """No API calls: show what the compressor would do, locally, for free."""
    from ..compress.estimate import estimate_tokens

    out = ["", f"TokenLens eval — dry run ({len(tasks)} tasks, no API calls)", ""]
    out.append(f"  {'task':<20}{'class':<22}{'arm':<17}{'est. tokens':>12}{'blocks':>9}")
    out.append(f"  {'-' * 80}")
    for task in tasks:
        body = build_body(task, model)
        base = estimate_tokens(task.context) + estimate_tokens(task.question)
        out.append(f"  {task.id:<20}{task.task_class:<22}{'cleartext':<17}{base:>12,}{'—':>9}")
        for arm in arms:
            r = compress_request(body, method=arm.method, rate=arm.rate)
            # est_tokens_saved covers only the eligible blocks, so subtract it
            # from the whole-payload baseline rather than reporting it directly.
            blocks = f"{r.compressed_blocks}/{r.eligible_blocks}"
            out.append(f"  {'':<20}{'':<22}{arm.label:<17}"
                       f"{base - r.est_tokens_saved:>12,}{blocks:>9}")
    out.append("")
    out.append("  Token counts are local estimates and quality is unmeasured — a dry run")
    out.append("  proves the plumbing, not the policy. Run without --dry-run to grade.")
    out.append("")
    return "\n".join(out)


def run(
    *,
    tasks_path: str | None = None,
    model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    rates: list[float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    out_path: str | None = None,
    upstream: str = "https://api.anthropic.com",
    classes: list[str] | None = None,
    dry_run: bool = False,
    repeats: int = 1,
) -> int:
    rates = rates or [0.8, 0.6, 0.4]
    try:
        tasks = load_tasks(tasks_path, classes)
    except (OSError, ValueError) as e:
        print(f"eval: {e}", file=sys.stderr)
        return 1

    arms = default_arms(rates)
    if not llmlingua2.available():
        print(f"[eval] llmlingua2 unavailable ({llmlingua2.load_error()}); "
              f"grading the safe floor only. Install: pip install llmlingua",
              file=sys.stderr)

    if dry_run:
        print(render_dry_run(tasks, model, arms))
        return 0

    try:
        client = Client.from_env(upstream)
    except ApiError as e:
        print(f"eval: {e}. The harness calls the real model — set your key and retry.",
              file=sys.stderr)
        return 1

    repeats = max(1, repeats)
    # per task: `repeats` baselines, plus (answer + judge) per arm per repeat.
    n_calls = len(tasks) * repeats * (1 + 2 * len(arms))
    print(f"[eval] {len(tasks)} tasks × {len(arms)} arms × {repeats} repeat(s) — "
          f"about {n_calls} model calls. This costs real money.",
          file=sys.stderr)
    if repeats == 1:
        print("[eval] --repeats 1: one sample per cell. The control arm will tell you "
              "how much of the resulting curve is noise.", file=sys.stderr)

    report = run_eval(
        tasks, client,
        arms=arms, model=model, judge_model=judge_model,
        tolerance=tolerance, max_tokens=max_tokens, concurrency=concurrency,
        repeats=repeats,
    )
    print(render(report))

    path = out_path or "tokenlens-eval.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"  report written to {path}"
              f"   (serve it on the dashboard: tokenlens serve --eval-report {path})\n")
    except OSError as e:
        print(f"eval: could not write {path}: {e}", file=sys.stderr)
        return 1

    # A run where every arm degraded quality is a successful eval and a failed
    # policy. Exit non-zero so CI can gate on it.
    return 0 if any(s.passes for s in report.curve) else 2
