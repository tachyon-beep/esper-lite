"""Tests for Overwatch CLI entry point."""

from __future__ import annotations

import pytest


class TestOverwatchCLI:
    """Tests for overwatch CLI."""

    def test_cli_module_imports(self) -> None:
        """CLI module can be imported."""
        from esper.scripts import overwatch

        assert callable(getattr(overwatch, "main", None))
        assert callable(getattr(overwatch, "build_parser", None))

    def test_parser_has_replay_arg(self) -> None:
        """Parser accepts --replay argument."""
        from esper.scripts.overwatch import build_parser

        parser = build_parser()
        args = parser.parse_args(["--replay", "test.jsonl"])

        assert args.replay == "test.jsonl"

    def test_parser_replay_required_message(self) -> None:
        """Parser shows helpful message when --replay missing."""
        from esper.scripts.overwatch import build_parser

        parser = build_parser()
        # In Stage 1, --replay is required (live mode comes in Stage 6)
        with pytest.raises(SystemExit):
            parser.parse_args([])
