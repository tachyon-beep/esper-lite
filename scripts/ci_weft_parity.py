#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from esper.utils.weft_parity import (
    build_phase_a_report,
    inspect_loomweave_db,
    parse_defensive_output,
    parse_leyline_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase A Weft parity report."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--wardline-output",
        type=Path,
        help="Wardline JSONL output. Defaults to ARTIFACTS_DIR/wardline.jsonl.",
    )
    parser.add_argument(
        "--loomweave-db",
        type=Path,
        default=Path(".weft/loomweave/loomweave.db"),
    )
    parser.add_argument(
        "--fail-on-homegrown-only",
        action="store_true",
        help="Exit 1 when the parity report contains homegrown-only findings.",
    )
    args = parser.parse_args()

    artifacts_dir: Path = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    defensive_result = _run([sys.executable, "scripts/lint_defensive_patterns.py"])
    leyline_result = _run([sys.executable, "scripts/lint_leyline_types.py"])
    _write_command_output(
        artifacts_dir / "homegrown-defensive-patterns.txt", defensive_result
    )
    _write_command_output(artifacts_dir / "homegrown-leyline-types.txt", leyline_result)

    wardline_output = _wardline_output_path(args.wardline_output, artifacts_dir)

    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = build_phase_a_report(
        wardline_output=wardline_output,
        loomweave=inspect_loomweave_db(args.loomweave_db),
        head_commit=head_commit,
        defensive_findings=parse_defensive_output(
            defensive_result.stdout + defensive_result.stderr
        ),
        defensive_exit_code=defensive_result.returncode,
        leyline_findings=parse_leyline_output(
            leyline_result.stdout + leyline_result.stderr
        ),
        leyline_exit_code=leyline_result.returncode,
    )
    report["metadata"] = {
        "schema": "esper-weft-parity-metadata-v1",
        "homegrown_exit_codes": {
            "defensive-patterns": defensive_result.returncode,
            "leyline-types": leyline_result.returncode,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.fail_on_homegrown_only and _has_homegrown_only(report):
        return 1
    return 0


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_command_output(
    path: Path, result: subprocess.CompletedProcess[str]
) -> None:
    path.write_text(result.stdout + result.stderr)


def _wardline_output_path(raw_path: Path | None, artifacts_dir: Path) -> Path:
    if raw_path is None:
        return artifacts_dir / "wardline.jsonl"
    return raw_path


def _has_homegrown_only(report: dict[str, Any]) -> bool:
    for check in report["checks"]:
        if len(check["homegrown_only"]) > 0:
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
