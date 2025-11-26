#!/usr/bin/env python3
"""Run the async worker soak harness outside of pytest."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Callable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esper.core.async_runner import AsyncWorker  # noqa: E402
from tests.helpers.async_worker_harness import SoakConfig, run_soak  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async worker soak harness")
    parser.add_argument("--iterations", type=int, default=5, help="Number of soak iterations")
    parser.add_argument("--jobs", type=int, default=96, help="Jobs per iteration")
    parser.add_argument("--seed", type=int, default=0, help="Randomness seed")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="*",
        default=[2, 4],
        help="Concurrency levels to cycle through",
    )
    return parser


def _log_line(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = SoakConfig(
        iterations=args.iterations,
        jobs_per_iteration=args.jobs,
        concurrency_levels=tuple(args.concurrency),
    )

    def factory(concurrency: int) -> AsyncWorker:
        return AsyncWorker(max_concurrency=concurrency)

    result = run_soak(factory, seed=args.seed, config=cfg, log_fn=_log_line)

    total = (
        result.jobs_completed
        + result.jobs_cancelled
        + result.jobs_failed
        + result.jobs_timed_out
    )
    _log_line("--- Soak Summary ---")
    _log_line(f"iterations: {result.iterations_run}")
    _log_line(f"submitted: {result.jobs_submitted}")
    _log_line(
       f"completed={result.jobs_completed} cancelled={result.jobs_cancelled} "
       f"failed={result.jobs_failed} timed_out={result.jobs_timed_out}"
    )
    _log_line(f"accounting-balanced: {total == result.jobs_submitted}")
    return 0 if total == result.jobs_submitted else 1


if __name__ == "__main__":
    raise SystemExit(main())

