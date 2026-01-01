"""Tests for CounterfactualPanel widget."""
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
        """Panel shows stable layout with placeholders when unavailable.

        The unavailable state uses a dim border and shows the same structure
        as the waterfall view but with '--' placeholders to prevent layout shifts.
        """
        matrix = CounterfactualSnapshot(strategy="unavailable")
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()
        # Unavailable state has dim border (vs cyan for available)
        assert rendered.border_style == "dim"
        assert rendered.title == "Counterfactual Analysis"

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
        """Panel shows 'Live Ablation Analysis' for ablation_only strategy.

        When pair data IS present, pairs are shown and synergy is computed.
        The 'episode end' message only shows when pair data is NOT available.
        """
        from io import StringIO
        from rich.console import Console

        # Ablation mode WITH pair data (the (True, True) config is the pair for 2 seeds)
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
        # When pair data is available, pairs ARE shown (even in ablation mode)
        assert "Pairs:" in content
        # When pair data is available, synergy is computed (not the "episode end" message)
        assert "Synergy:" in content
        assert "episode end" not in content

    def test_renders_ablation_only_without_pairs(self):
        """Panel shows 'episode end' message when ablation mode and NO pair data."""
        from io import StringIO
        from rich.console import Console

        # Ablation mode WITHOUT pair data (only solo configs, no pair config)
        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1", "r0c2"),  # 3 seeds
            configs=[
                CounterfactualConfig(seed_mask=(False, False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True, False), accuracy=30.0),
                CounterfactualConfig(seed_mask=(False, False, True), accuracy=32.0),
                # No pair configs like (True, True, False) - not computed yet
                CounterfactualConfig(seed_mask=(True, True, True), accuracy=65.0),
            ],
            strategy="ablation_only",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(rendered)
        content = console.file.getvalue()

        # Should show live ablation header
        assert "Live Ablation" in content
        # Without pair data, no pairs section
        assert "Pairs:" not in content
        assert "Top Combinations" not in content
        # Should show episode end message (pairs not available)
        assert "episode end" in content

    def test_single_seed_never_shows_interference(self):
        """Single seed should never show interference, even with stale data.

        This guards against stale matrix data from when there were 2+ seeds.
        With 1 seed, synergy is mathematically 0 - any non-zero value is from
        stale data and should not trigger interference warnings.
        """
        from io import StringIO
        from rich.console import Console

        # Single seed with artificially negative "synergy" (stale data scenario)
        # This shouldn't happen mathematically, but could if matrix is stale
        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0",),  # Single seed
            configs=[
                CounterfactualConfig(seed_mask=(False,), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True,), accuracy=30.0),
            ],
            strategy="full_factorial",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(rendered)
        content = console.file.getvalue()

        # Single seed should NEVER show interference - mathematically impossible
        assert "INTERFERENCE" not in content
        assert "Seeds are hurting each other" not in content
        # But it should still show the synergy calculation (which should be 0)
        assert "Interaction:" in content

    def test_two_seeds_with_interference_shows_warning(self):
        """Two seeds with negative synergy should show interference warning."""
        from io import StringIO
        from rich.console import Console

        # Two seeds where combined is WORSE than sum of individuals (interference)
        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(True, True), accuracy=40.0),   # +15 total (interference!)
            ],
            strategy="full_factorial",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(rendered)
        content = console.file.getvalue()

        # Two seeds with interference SHOULD show the warning
        assert "INTERFERENCE" in content
        assert "Seeds are hurting each other" in content

    def test_single_seed_baseline_not_zero(self):
        """Regression: 1-seed baseline must NOT be 0.0 (B-TUI-01).

        Bug: For 1 seed, all_disabled_accs.get(i, 0.0) returned 0.0 because
        the 'all_off' config was only generated for 2-4 seeds. The emitter's
        fallback to min(baseline_accs.values()) was never triggered.

        Fix: Pass None instead of 0.0 so the emitter uses ablation accuracy.
        """
        from io import StringIO
        from rich.console import Console

        # 1 seed with proper baseline (ablation = host-only accuracy = 45%)
        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0",),
            configs=[
                CounterfactualConfig(seed_mask=(False,), accuracy=45.0),  # NOT 0.0!
                CounterfactualConfig(seed_mask=(True,), accuracy=55.0),   # With seed
            ],
            strategy="ablation_only",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(rendered)
        content = console.file.getvalue()

        # Baseline should show 45% (the ablation accuracy), NOT 0%
        assert "45.0%" in content
        # Combined should show 55%
        assert "55.0%" in content
        # The bar should NOT start from 0 (which would be 100% filled)
        assert "Baseline (Host only)" in content
