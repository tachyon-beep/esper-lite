"""Tests for ContextPanel widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import (
    EnvSummary,
    SlotChipState,
)


class TestContextPanel:
    """Tests for ContextPanel widget."""

    def test_context_panel_imports(self) -> None:
        """ContextPanel can be imported."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        assert ContextPanel is not None

    def test_context_panel_renders_anomaly_reasons(self) -> None:
        """ContextPanel displays anomaly reasons as bullet list."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="WARN",
            anomaly_score=0.72,
            anomaly_reasons=["High gradient ratio (3.2x)", "Memory pressure (94%)"],
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "WHY FLAGGED" in content or "Why Flagged" in content
        assert "High gradient ratio" in content
        assert "Memory pressure" in content
        assert "•" in content or "-" in content  # Bullet points

    def test_context_panel_renders_env_header(self) -> None:
        """ContextPanel shows env ID and status."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="CRIT",
            anomaly_score=0.85,
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "Env 3" in content
        assert "CRIT" in content

    def test_context_panel_renders_slot_details(self) -> None:
        """ContextPanel shows selected slot details."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            slots={
                "r0c1": SlotChipState(
                    slot_id="r0c1",
                    stage="BLENDING",
                    blueprint_id="conv_light",
                    alpha=0.78,
                    gate_last="G2",
                    gate_passed=True,
                ),
            },
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "r0c1" in content
        assert "BLENDING" in content or "Blending" in content
        assert "conv_light" in content
        assert "0.78" in content or "78" in content
        assert "G2" in content

    def test_context_panel_renders_metrics(self) -> None:
        """ContextPanel shows env metrics."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            throughput_fps=98.5,
            reward_last=0.42,
            task_metric=0.823,
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "98" in content  # throughput
        assert "0.42" in content or "42" in content  # reward

    def test_context_panel_empty_state(self) -> None:
        """ContextPanel shows empty state when no env selected."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        panel = ContextPanel()

        content = panel.render_content()
        assert "Select" in content or "select" in content or "No env" in content

    def test_context_panel_no_anomaly_reasons(self) -> None:
        """ContextPanel handles env with no anomaly reasons."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            anomaly_score=0.1,
            anomaly_reasons=[],
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "No issues" in content or "Healthy" in content or "OK" in content
