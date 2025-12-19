"""Final verification tests for Sanctum port completion."""

import pytest


class TestSanctumPortComplete:
    """Verify Sanctum is a complete port of the Rich TUI."""

    def test_tui_py_deleted(self):
        """Legacy tui.py should no longer exist."""
        from pathlib import Path
        tui_path = Path(__file__).parents[3] / "src" / "esper" / "karn" / "tui.py"
        assert not tui_path.exists(), "tui.py should be deleted after Sanctum port"

    def test_sanctum_exports_available(self):
        """Sanctum exports should be available from karn."""
        from esper.karn.sanctum import (
            SanctumSnapshot,
            SanctumBackend,
            SanctumAggregator,
        )
        assert SanctumSnapshot is not None
        assert SanctumBackend is not None
        assert SanctumAggregator is not None

    def test_sanctum_app_importable(self):
        """SanctumApp should be importable (may be None without Textual)."""
        from esper.karn.sanctum import SanctumApp
        # SanctumApp may be None if Textual not installed, but import should work

    def test_cli_flag_works(self):
        """--sanctum flag should be recognized in train.py."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "esper.scripts.train", "ppo", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--sanctum" in result.stdout

    def test_sanctum_backend_in_karn_exports(self):
        """SanctumBackend should be exported from esper.karn."""
        from esper.karn import SanctumBackend
        assert SanctumBackend is not None

    def test_tui_exports_removed_from_karn(self):
        """Legacy TUI exports should be removed from esper.karn."""
        import esper.karn as karn

        # These should NOT be in __all__
        assert "TUIOutput" not in karn.__all__
        assert "TUIState" not in karn.__all__
        assert "ThresholdConfig" not in karn.__all__
        assert "HealthStatus" not in karn.__all__
