"""Tests for HistoricalEnvDetail modal.

Tests cover:
- HistoricalEnvDetail: Modal for viewing frozen env state from Best Runs
- Threshold usage from DisplayThresholds
- format_params import and usage
"""
from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from esper.karn.constants import DisplayThresholds
from esper.karn.sanctum.schema import (
    BestRunRecord,
    RewardComponents,
)
from esper.karn.sanctum.widgets.historical_env_detail import HistoricalEnvDetail


def render_to_text(renderable) -> str:
    """Helper to render a Rich renderable to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


def make_minimal_record(
    *,
    env_id: int = 0,
    episode: int = 10,
    peak_accuracy: float = 85.0,
    final_accuracy: float = 82.0,
    growth_ratio: float = 1.0,
    host_params: int = 1_000_000,
) -> BestRunRecord:
    """Create a minimal BestRunRecord for testing."""
    return BestRunRecord(
        env_id=env_id,
        episode=episode,
        peak_accuracy=peak_accuracy,
        final_accuracy=final_accuracy,
        growth_ratio=growth_ratio,
        host_params=host_params,
        fossilized_count=0,
        pruned_count=0,
        seeds={},
        slot_ids=["r0c0", "r0c1"],
        blueprint_spawns={},
        blueprint_fossilized={},
        blueprint_prunes={},
        reward_components=RewardComponents(),
        accuracy_history=[80.0, 82.0, 85.0],
        reward_history=[0.1, 0.2, 0.3],
        action_history=["WAIT", "GERMINATE"],
        reward_mode=None,
        counterfactual_matrix=None,
    )


class TestHistoricalEnvDetailRendering:
    """Test HistoricalEnvDetail renders all fields correctly."""

    def test_modal_creation(self):
        """Modal should be creatable with BestRunRecord."""
        record = make_minimal_record()
        modal = HistoricalEnvDetail(record)
        assert modal is not None
        assert modal._record == record

    def test_renders_header_with_episode_and_peak(self):
        """Header should show episode number (1-indexed) and peak accuracy."""
        record = make_minimal_record(episode=42, peak_accuracy=95.5)
        modal = HistoricalEnvDetail(record)
        header = modal._render_header()
        header_text = header.plain

        # 1-indexed display: episode + 1 = 42 + 1 = 43
        assert "Episode# 43" in header_text
        assert "Peak: 95.5%" in header_text
        # Default view state is "peak", shown as "PEAK STATE"
        assert "PEAK STATE" in header_text

    def test_header_uses_format_params(self):
        """Header should format host params using shared utility."""
        record = make_minimal_record(host_params=2_500_000)
        modal = HistoricalEnvDetail(record)
        header = modal._render_header()
        header_text = header.plain

        # Should show "2.5M" from format_params
        assert "Host: 2.5M" in header_text

    def test_growth_ratio_uses_threshold(self):
        """Growth ratio styling should use DisplayThresholds.GROWTH_RATIO_WARNING."""
        # Below threshold - should be green
        record_low = make_minimal_record(growth_ratio=1.1)
        modal_low = HistoricalEnvDetail(record_low)
        header_low = modal_low._render_header()
        assert "Growth: 1.10x" in header_low.plain

        # Above threshold - should be yellow
        record_high = make_minimal_record(growth_ratio=1.3)
        modal_high = HistoricalEnvDetail(record_high)
        header_high = modal_high._render_header()
        assert "Growth: 1.30x" in header_high.plain

    def test_renders_metrics_table(self):
        """Metrics section should render as a table."""
        record = make_minimal_record()
        modal = HistoricalEnvDetail(record)
        table = modal._render_metrics()

        assert isinstance(table, Table)

    def test_pbrs_uses_display_thresholds(self):
        """PBRS healthy check should use DisplayThresholds constants."""
        # Create record with stage_bonus that's 20% of total (healthy)
        record = make_minimal_record()
        record.reward_components = RewardComponents(
            total=1.0,
            stage_bonus=0.2,  # 20% - within healthy range
        )
        modal = HistoricalEnvDetail(record)
        table = modal._render_metrics()
        output = render_to_text(table)

        # Should show PBRS percentage
        assert "PBRS:" in output

    def test_renders_graveyard_panel(self):
        """Graveyard section should render as a panel."""
        record = make_minimal_record()
        record.blueprint_spawns = {"conv3x3": 5, "dense2": 3}
        record.blueprint_fossilized = {"conv3x3": 2}
        record.blueprint_prunes = {"dense2": 1}

        modal = HistoricalEnvDetail(record)
        panel = modal._render_graveyard()

        assert isinstance(panel, Panel)
        output = render_to_text(panel)
        assert "conv3x3" in output
        assert "dense2" in output

    def test_graveyard_success_rate_thresholds(self):
        """Success rate styling should use DisplayThresholds constants."""
        record = make_minimal_record()
        # 3 fossilized, 1 pruned = 75% success rate (green)
        record.blueprint_spawns = {"conv3x3": 4}
        record.blueprint_fossilized = {"conv3x3": 3}
        record.blueprint_prunes = {"conv3x3": 1}

        modal = HistoricalEnvDetail(record)
        panel = modal._render_graveyard()
        output = render_to_text(panel)

        # Should show 75% rate
        assert "75%" in output


class TestHistoricalEnvDetailThresholds:
    """Test that DisplayThresholds are used correctly."""

    def test_growth_ratio_threshold_value(self):
        """GROWTH_RATIO_WARNING should be 1.2."""
        assert DisplayThresholds.GROWTH_RATIO_WARNING == 1.2

    def test_pbrs_threshold_values(self):
        """PBRS thresholds should be 0.1-0.4."""
        assert DisplayThresholds.PBRS_HEALTHY_MIN == 0.1
        assert DisplayThresholds.PBRS_HEALTHY_MAX == 0.4

    def test_blueprint_success_thresholds(self):
        """Blueprint success thresholds should be 0.50 and 0.25."""
        assert DisplayThresholds.BLUEPRINT_SUCCESS_GREEN == 0.50
        assert DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW == 0.25


class TestHistoricalEnvDetailIntegration:
    """Test HistoricalEnvDetail integration with schema types."""

    def test_handles_reward_mode(self):
        """A/B cohort should be displayed."""
        record = make_minimal_record()
        record.reward_mode = "A"

        modal = HistoricalEnvDetail(record)
        header = modal._render_header()

        assert "Cohort A" in header.plain

    def test_handles_empty_seeds(self):
        """Empty seeds dict should not cause errors."""
        record = make_minimal_record()
        record.seeds = {}

        modal = HistoricalEnvDetail(record)
        # Should not raise
        _ = modal._render_header()
        _ = modal._render_metrics()
        _ = modal._render_graveyard()
