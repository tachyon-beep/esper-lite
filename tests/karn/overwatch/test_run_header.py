"""Tests for RunHeader widget."""

from __future__ import annotations

import pytest

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


@pytest.fixture
def sample_snapshot() -> TuiSnapshot:
    """Create a sample snapshot with run info."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T14:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(),
        run_id="exp-001",
        task_name="cifar10",
        episode=5,
        batch=150,
        best_metric=0.823,
        runtime_s=3725.0,
        envs_ok=3,
        envs_warn=1,
        envs_crit=0,
    )


class TestRunHeader:
    """Tests for RunHeader widget."""

    def test_run_header_imports(self) -> None:
        """RunHeader can be imported."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        assert RunHeader is not None

    def test_run_header_renders_run_id(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays run ID."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "exp-001" in content

    def test_run_header_renders_task(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays task name."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "cifar10" in content

    def test_run_header_renders_episode(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays episode number."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "5" in content or "Ep 5" in content or "ep:5" in content.lower()

    def test_run_header_renders_runtime(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays formatted runtime."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "1h" in content  # 3725s = 1h 2m

    def test_run_header_renders_connection_live(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader shows Live when connected with low staleness."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line2()
        assert "Live" in content or "●" in content

    def test_run_header_renders_connection_stale(self) -> None:
        """RunHeader shows Stale when staleness is high."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        snapshot = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T14:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 10.0),  # 10s stale
            tamiyo=TamiyoState(),
        )
        header = RunHeader()
        header.update_snapshot(snapshot)

        content = header.render_line2()
        assert "Stale" in content or "10" in content

    def test_run_header_renders_env_counts(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader shows environment health counts."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line2()
        # Should show OK:3 WARN:1 CRIT:0 or similar
        assert "3" in content  # 3 OK
        assert "1" in content  # 1 WARN

    def test_run_header_renders_best_metric(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader shows best metric achieved."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line2()
        assert "82" in content or "0.82" in content  # best_metric=0.823

    def test_run_header_empty_state(self) -> None:
        """RunHeader handles no snapshot gracefully."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()

        content = header.render_line1()
        assert "Waiting" in content or "No data" in content or "--" in content
