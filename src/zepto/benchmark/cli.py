"""CLI for horizon simulation benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zepto.benchmark.apertus_training import (
    DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    WallClockLimitExceeded,
    run_apertus_8b_training_horizon,
    run_apertus_training_smoke,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zepto-bench",
        description="Run Zepto horizon simulation wall-clock benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser(
        "smoke",
        help="Tiny Apertus model (~1s); checks timing harness and structural cache.",
    )
    smoke.set_defaults(handler=_cmd_smoke)

    full = sub.add_parser(
        "apertus-8b-training",
        help="Full Apertus 8B training horizon (slow; default 300s wall clock).",
    )
    full.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
        metavar="SECS",
        help=f"Wall-clock limit (default: {DEFAULT_WALL_CLOCK_LIMIT_SECONDS:.0f}).",
    )
    full.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="Write timing JSON to this path.",
    )
    full.set_defaults(handler=_cmd_8b)

    return parser


def _print_result(title: str, timing, *, json_path: Path | None = None) -> None:
    print(f"\n--- {title} ---")
    print(timing.format_summary())
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(timing.to_json(), encoding="utf-8")
        print(f"\nWrote {json_path}")


def _cmd_smoke(_args: argparse.Namespace) -> int:
    result = run_apertus_training_smoke()
    _print_result("Apertus training smoke", result.timing)
    return 0


def _cmd_8b(args: argparse.Namespace) -> int:
    try:
        result = run_apertus_8b_training_horizon(
            wall_clock_limit_seconds=args.timeout,
        )
    except WallClockLimitExceeded as exc:
        print(f"\nTimed out: {exc}", file=sys.stderr)
        return 1

    sim = result.sim
    timing = result.timing
    last = sim.timeline[-1]
    if not last.lowered.nodes:
        print("Benchmark produced no lowered nodes.", file=sys.stderr)
        return 1
    if sum(n.forward_flops for n in last.lowered.nodes) <= 0:
        print("Benchmark produced zero forward FLOPs.", file=sys.stderr)
        return 1

    _print_result("Apertus 8B training simulation timing", timing, json_path=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
