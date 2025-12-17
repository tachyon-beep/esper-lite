#!/usr/bin/env python3
# src/esper/scripts/overwatch.py
"""Overwatch TUI Entry Point.

Launch the Overwatch training monitor.

Usage:
    # Replay mode (Stage 1+)
    uv run python -m esper.scripts.overwatch --replay training.jsonl

    # Live mode (Stage 6)
    uv run python -m esper.scripts.overwatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="overwatch",
        description="Esper Overwatch - Training Monitor TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View a replay file
    python -m esper.scripts.overwatch --replay training.jsonl

    # Live monitoring (requires training to be running)
    python -m esper.scripts.overwatch  # Coming in Stage 6

Keyboard shortcuts:
    q       Quit
    ?       Toggle help overlay
    j/k     Navigate flight board
    Enter   Expand environment
    Esc     Collapse / Dismiss
""",
    )

    parser.add_argument(
        "--replay",
        type=str,
        required=True,  # Required in Stage 1, optional in Stage 6
        metavar="FILE",
        help="Path to JSONL replay file (from SnapshotWriter)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay playback speed multiplier (default: 1.0)",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate replay file exists
    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"Error: Replay file not found: {replay_path}", file=sys.stderr)
        return 1

    if not replay_path.suffix == ".jsonl":
        print(f"Warning: Expected .jsonl file, got: {replay_path.suffix}", file=sys.stderr)

    # Import app here to avoid slow import on --help
    try:
        from esper.karn.overwatch import OverwatchApp
    except ImportError as e:
        print(f"Error: Failed to import Overwatch: {e}", file=sys.stderr)
        print("Hint: Install with `uv sync --extra overwatch`", file=sys.stderr)
        return 1

    # Launch the app
    app = OverwatchApp(replay_path=replay_path)
    app.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
