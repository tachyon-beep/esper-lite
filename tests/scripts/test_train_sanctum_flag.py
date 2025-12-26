"""Tests for --sanctum CLI flag."""

import subprocess
import sys


class TestSanctumCLIFlag:
    """Test --sanctum CLI integration."""

    def test_sanctum_flag_exists(self):
        """--sanctum flag should be recognized."""
        result = subprocess.run(
            [sys.executable, "-m", "esper.scripts.train", "ppo", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--sanctum" in result.stdout

    def test_sanctum_flag_accepted_alone(self):
        """--sanctum alone should be accepted (will fail without config, but flag parses)."""
        # Just test the flag is parsed, not that training runs
        result = subprocess.run(
            [
                sys.executable, "-m", "esper.scripts.train",
                "ppo", "--sanctum", "--help",
            ],
            capture_output=True,
            text=True,
        )
        # Help should work with --sanctum present
        assert result.returncode == 0
