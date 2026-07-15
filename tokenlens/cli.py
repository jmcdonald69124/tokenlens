"""Command-line entry point for TokenLens."""

from __future__ import annotations

import argparse

from . import __version__, bench, proxy
from .eval.harness import DEFAULT_MODEL, DEFAULT_TOLERANCE
from .eval.judge import DEFAULT_JUDGE_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tokenlens",
        description="Local BYOK proxy that measures (and later compresses) "
                    "tokens sent to a frontier model.",
    )
    parser.add_argument("--version", action="version", version=f"tokenlens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local proxy")
    serve.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8787, help="bind port (default 8787)")
    serve.add_argument(
        "--upstream",
        default="https://api.anthropic.com",
        help="upstream API base URL (default https://api.anthropic.com)",
    )
    serve.add_argument(
        "--compress",
        choices=["none", "safe", "llmlingua2"],
        default="none",
        help="compression method: none (measure only, default), "
             "safe (rungs 0-2, model-free), llmlingua2 (safe floor + local model)",
    )
    serve.add_argument(
        "--rate",
        type=float,
        default=0.7,
        help="llmlingua2 target keep-rate, 0<rate<=1 (default 0.7)",
    )
    serve.add_argument(
        "--measure",
        action="store_true",
        help="measure real savings via the count_tokens endpoint (free, off the "
             "hot path). Adds two background count_tokens calls per compressed request.",
    )
    serve.add_argument(
        "--judge",
        action="store_true",
        help="shadow-mode model judgement: on a sample of compressed requests, replay "
             "the uncompressed prompt and have a model grade both answers. Surfaces "
             "'quality retained' on the dashboard. COSTS MONEY (one extra completion "
             "plus one judge call per judged request).",
    )
    serve.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                       help=f"model that grades the answers (default {DEFAULT_JUDGE_MODEL})")
    serve.add_argument("--judge-sample", type=float, default=0.25,
                       help="fraction of compressed requests to judge (default 0.25)")
    serve.add_argument("--judge-tolerance", type=float, default=DEFAULT_TOLERANCE,
                       help="quality floor, as a fraction of cleartext quality, below "
                            "which a judged request counts as degraded (default 0.99)")
    serve.add_argument("--eval-report", metavar="PATH",
                       help="show the calibration curve from a `tokenlens eval` run "
                            "(tokenlens-eval.json) on the dashboard")

    e = sub.add_parser(
        "eval",
        help="calibrate: grade compressed vs cleartext answers on a calibration set",
        description="Runs every calibration task through the model twice — cleartext and "
                    "compressed — and has a judge model grade both answers. Prints a "
                    "compression-ratio vs quality curve and the most aggressive policy "
                    "that holds quality within tolerance.",
    )
    e.add_argument("--tasks", metavar="PATH",
                   help="JSONL calibration set (default: the bundled starter set)")
    e.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model under test (default {DEFAULT_MODEL})")
    e.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"model that grades the answers (default {DEFAULT_JUDGE_MODEL})")
    e.add_argument("--rates", default="0.8,0.6,0.4",
                   help="llmlingua2 keep-rates to test, comma separated (default 0.8,0.6,0.4)")
    e.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help="minimum mean quality, as a fraction of cleartext, for an arm to "
                        "pass (default 0.99)")
    e.add_argument("--class", dest="classes", action="append", metavar="NAME",
                   help="only run this task class (repeatable)")
    e.add_argument("--repeats", type=int, default=1, metavar="N",
                   help="judge each cell N times and average (default 1). The model "
                        "answers differently every call and the judge grades "
                        "differently every call; one sample cannot tell a 5%% quality "
                        "loss from a coin flip. Costs N× as much.")
    e.add_argument("--concurrency", type=int, default=4,
                   help="tasks graded in parallel (default 4)")
    e.add_argument("--max-tokens", type=int, default=1024,
                   help="max_tokens for the answers under test (default 1024)")
    e.add_argument("--out", default="tokenlens-eval.json",
                   help="where to write the JSON report (default tokenlens-eval.json)")
    e.add_argument("--upstream", default="https://api.anthropic.com",
                   help="API base URL")
    e.add_argument("--dry-run", action="store_true",
                   help="no API calls: show what the compressor would do to each task")

    b = sub.add_parser("bench", help="measure compression savings on a text file")
    b.add_argument("file", help="path to a text/prose document")
    b.add_argument("--rate", type=float, default=0.6,
                   help="llmlingua2 keep-rate, 0<rate<=1 (default 0.6)")
    b.add_argument("--model", default="claude-opus-4-8", help="model id for token counts")
    b.add_argument("--measure", action="store_true",
                   help="use count_tokens for exact counts (needs ANTHROPIC_API_KEY)")
    b.add_argument("--upstream", default="https://api.anthropic.com",
                   help="API base URL for count_tokens")

    args = parser.parse_args(argv)
    if args.command == "serve":
        proxy.serve(args.host, args.port, args.upstream,
                    args.compress, args.rate, args.measure,
                    judge=args.judge, judge_model=args.judge_model,
                    judge_sample=args.judge_sample,
                    judge_tolerance=args.judge_tolerance,
                    eval_report=args.eval_report)
    elif args.command == "eval":
        try:
            rates = [float(r) for r in args.rates.split(",") if r.strip()]
        except ValueError:
            parser.error(f"--rates: not a comma-separated list of numbers: {args.rates!r}")
        from .eval import report as eval_report
        return eval_report.run(
            tasks_path=args.tasks, model=args.model, judge_model=args.judge_model,
            rates=rates, tolerance=args.tolerance, concurrency=args.concurrency,
            max_tokens=args.max_tokens, out_path=args.out, upstream=args.upstream,
            classes=args.classes, dry_run=args.dry_run, repeats=args.repeats,
        )
    elif args.command == "bench":
        return bench.run(args.file, args.rate, args.model, args.measure, args.upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
