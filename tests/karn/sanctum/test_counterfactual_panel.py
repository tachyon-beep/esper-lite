"""Tests for CounterfactualPanel widget."""
import pytest
from esper.karn.sanctum.schema import CounterfactualConfig, CounterfactualSnapshot
from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel


class TestCounterfactualSnapshot:
    """Test CounterfactualSnapshot dataclass methods."""

    def test_baseline_accuracy(self):
        """Baseline is config with all False."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        assert snapshot.baseline_accuracy == 25.0

    def test_combined_accuracy(self):
        """Combined is config with all True."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        assert snapshot.combined_accuracy == 65.0

    def test_individual_contributions(self):
        """Individual contribution is solo accuracy minus baseline."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=30.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        contribs = snapshot.individual_contributions()
        assert contribs["r0c0"] == 10.0  # 35 - 25
        assert contribs["r0c1"] == 5.0   # 30 - 25

    def test_total_synergy_positive(self):
        """Synergy when combined > sum of individuals."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),   # +40 total
            ],
            strategy="full_factorial",
        )
        # Expected: 25 + 10 + 10 = 45, Actual: 65, Synergy: 20
        assert snapshot.total_synergy() == 20.0

    def test_total_synergy_negative(self):
        """Interference when combined < sum of individuals."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(True, True), accuracy=40.0),   # +15 total (interference!)
            ],
            strategy="full_factorial",
        )
        # Expected: 25 + 10 + 10 = 45, Actual: 40, Synergy: -5
        assert snapshot.total_synergy() == -5.0


class TestCounterfactualPanel:
    """Test CounterfactualPanel widget rendering."""

    def test_renders_unavailable_when_no_data(self):
        """Panel shows unavailable message when strategy is unavailable."""
        matrix = CounterfactualSnapshot(strategy="unavailable")
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()
        assert "unavailable" in str(rendered.renderable).lower()

    def test_renders_waterfall_with_data(self):
        """Panel renders waterfall when data available."""
        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()
        # Should have correct title and cyan border (not dim like unavailable)
        assert rendered.title == "Counterfactual Analysis"
        assert rendered.border_style == "cyan"

    def test_renders_ablation_only_with_indicator(self):
        """Panel shows 'Live Ablation Analysis' for ablation_only strategy."""
        from io import StringIO
        from rich.console import Console

        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=30.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="ablation_only",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()

        # Render to string to check content
        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(rendered)
        content = console.file.getvalue()

        # Should show live ablation header
        assert "Live Ablation" in content
        # Should not show pairs (not available for ablation)
        assert "Pairs:" not in content
        # Should show episode end note
        assert "episode end" in content
