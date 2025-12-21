"""Tests for TamiyoStrip widget."""

from __future__ import annotations

import pytest

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


@pytest.fixture
def tamiyo_snapshot() -> TuiSnapshot:
    """Create a snapshot with Tamiyo data."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T14:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(
            kl_divergence=0.015,
            entropy=1.5,
            explained_variance=0.75,
            clip_fraction=0.08,
            grad_norm=0.5,
            learning_rate=3e-4,
            kl_trend=0.002,
            entropy_trend=-0.05,
            ev_trend=0.01,
            entropy_collapsed=False,
            ev_warning=False,
            action_counts={"GERMINATE": 10, "BLEND": 20, "PRUNE": 5, "WAIT": 65},
            recent_actions=["G", "B", "W", "W", "P"],
        ),
    )


class TestTamiyoStrip:
    """Tests for TamiyoStrip widget."""

    def test_tamiyo_strip_imports(self) -> None:
        """TamiyoStrip can be imported."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        assert TamiyoStrip is not None

    def test_tamiyo_strip_renders_kl(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip displays KL divergence."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        assert "KL" in content
        assert "0.015" in content or "0.02" in content

    def test_tamiyo_strip_renders_entropy(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip displays entropy."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        assert "Ent" in content or "H" in content
        assert "1.5" in content

    def test_tamiyo_strip_renders_ev(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip displays explained variance."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        assert "EV" in content
        assert "0.75" in content or "75" in content

    def test_tamiyo_strip_renders_trend_arrows(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip shows trend arrows."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        # kl_trend=0.002 (stable →), entropy_trend=-0.05 (↓), ev_trend=0.01 (stable/↑)
        assert "↓" in content  # entropy falling

    def test_tamiyo_strip_renders_action_counts(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip shows action distribution."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_actions()
        # Should show action names and counts/percentages
        assert "G" in content or "GERM" in content
        assert "B" in content or "BLEND" in content

    def test_tamiyo_strip_renders_recent_actions(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip shows recent action sequence."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_actions()
        # recent_actions = ["G", "B", "W", "W", "C"]
        assert "G" in content
        assert "W" in content

    def test_tamiyo_strip_health_coloring(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip applies health-based Rich markup coloring."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        health = strip.get_vitals_health()
        # kl=0.015 is ok, entropy=1.5 is ok, ev=0.75 is ok
        assert health["kl"] == "ok"
        assert health["entropy"] == "ok"
        assert health["ev"] == "ok"

        # Verify Rich markup is applied to vitals
        content = strip.render_vitals()
        assert "[green]KL" in content  # KL is healthy (ok)
        assert "[green]Ent" in content  # Entropy is healthy (ok)
        assert "[green]EV" in content  # EV is healthy (ok)

    def test_tamiyo_strip_empty_state(self) -> None:
        """TamiyoStrip handles no snapshot gracefully."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()

        content = strip.render_vitals()
        assert "Waiting" in content or "--" in content

    def test_tamiyo_strip_entropy_collapsed_warning(self) -> None:
        """TamiyoStrip shows warning when entropy is collapsed."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        snapshot = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T14:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(
                entropy=0.1,
                entropy_collapsed=True,
            ),
        )
        strip = TamiyoStrip()
        strip.update_snapshot(snapshot)

        health = strip.get_vitals_health()
        assert health["entropy"] == "crit"

        # Verify critical health shows as red
        content = strip.render_vitals()
        assert "[red]Ent" in content

    def test_tamiyo_strip_warning_health_coloring(self) -> None:
        """TamiyoStrip shows yellow for warning health levels."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        # Create snapshot with warning-level metrics
        snapshot = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T14:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(
                kl_divergence=0.03,  # warn level (0.025-0.05)
                entropy=0.4,  # warn level (0.2-0.5)
                explained_variance=0.5,  # warn level (0.3-0.6)
            ),
        )
        strip = TamiyoStrip()
        strip.update_snapshot(snapshot)

        health = strip.get_vitals_health()
        assert health["kl"] == "warn"
        assert health["entropy"] == "warn"
        assert health["ev"] == "warn"

        # Verify warning health shows as yellow
        content = strip.render_vitals()
        assert "[yellow]KL" in content
        assert "[yellow]Ent" in content
        assert "[yellow]EV" in content

    def test_tamiyo_strip_critical_health_coloring(self) -> None:
        """TamiyoStrip shows red for critical health levels."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        # Create snapshot with critical-level metrics
        snapshot = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T14:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(
                kl_divergence=0.08,  # crit level (>0.05)
                entropy=0.1,  # crit level (<0.2)
                explained_variance=0.2,  # crit level (<0.3)
            ),
        )
        strip = TamiyoStrip()
        strip.update_snapshot(snapshot)

        health = strip.get_vitals_health()
        assert health["kl"] == "crit"
        assert health["entropy"] == "crit"
        assert health["ev"] == "crit"

        # Verify critical health shows as red
        content = strip.render_vitals()
        assert "[red]KL" in content
        assert "[red]Ent" in content
        assert "[red]EV" in content
