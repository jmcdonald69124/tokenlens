"""Command-line entry point for TokenLens."""

from __future__ import annotations

import argparse

from . import __version__, bench, proxy


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
                    args.compress, args.rate, args.measure)
    elif args.command == "bench":
        return bench.run(args.file, args.rate, args.model, args.measure, args.upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
