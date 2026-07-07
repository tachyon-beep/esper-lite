#!/usr/bin/env python3
"""Generate the Stage-2 HRA MAJOR-1 acceptance packet from Karn telemetry views.

Two subcommands enforce the §10 freeze order at the CLI surface
(gate doc: docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md):

  calibrate  read the OFF arms ONLY and print the δ the owner freezes (§10 step 2).
             Spec JSON: {"off_run_dirs": [...], "w": <int>, "budget": <int>}
  score      score the paired ON/OFF spec against ALREADY-FROZEN thresholds (§10 step 4)
             and emit the §-structured verdict packet.
             Spec JSON: {
               "pairs": [{"seed": int, "on_run_dir": str, "off_run_dir": str}, ...],
               "host_params": int, "n": 5|10, "budget": int,
               "thresholds": {"delta": float, "eps_rel": float, "tau_acc": float,
                              "delta_param_max": float, "w": int},
               "g3_ratio_max": float, "g4_ratio_max": float, "g4_abs_floor": int
             }

FAIL CLOSED on ingestion corruption: the raw_events view ingests with ignore_errors=true,
silently dropping malformed JSONL — a §1 completeness gate cannot be trusted over silently
dropped rows, so corruption aborts (exit 2) before any scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from esper.karn.mcp.views import create_views, scan_ingestion_integrity
from esper.simic.telemetry.stage2_acceptance_io import calibrate_from_spec, packet_from_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage-2 HRA MAJOR-1 acceptance packet (calibrate OFF -> freeze -> score)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("calibrate", "freeze δ from the OFF arms only (§10 step 2)"),
        ("score", "score the paired spec against frozen thresholds (§10 step 4)"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument(
            "--telemetry-dir", required=True, help="Telemetry root containing */events.jsonl"
        )
        cmd.add_argument("--spec", required=True, help="Path to the spec JSON (see module docstring)")
        cmd.add_argument(
            "--output", default=None, help="Output path (default: stdout)"
        )
    args = parser.parse_args(argv)

    ingestion = scan_ingestion_integrity(args.telemetry_dir)
    if not ingestion.is_clean:
        for line in ingestion.malformed_lines:
            print(f"MALFORMED: {line}", file=sys.stderr)
        print(
            "telemetry ingestion is not clean — malformed JSONL would be silently dropped "
            "by the view layer, so the §1 completeness gates cannot be trusted; aborting",
            file=sys.stderr,
        )
        return 2

    spec = json.loads(Path(args.spec).read_text())
    conn = duckdb.connect(":memory:")
    try:
        create_views(conn, args.telemetry_dir)
        try:
            if args.command == "calibrate":
                text = calibrate_from_spec(conn, spec)
            else:
                text = packet_from_spec(conn, spec)
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        conn.close()

    if args.output is None:
        print(text)
    else:
        Path(args.output).write_text(text + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
