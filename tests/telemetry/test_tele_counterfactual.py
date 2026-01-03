"""End-to-end tests for counterfactual analysis telemetry metrics (TELE-520 to TELE-525).

Verifies CounterfactualSnapshot schema and CounterfactualPanel consumer correctly
handle counterfactual data flow. Tests cover:

- TELE-520: strategy (str enum: "unavailable", "full_factorial", "ablation_only")
- TELE-521: baseline_accuracy (host-only accuracy, 0.0-100.0)
- TELE-522: combined_accuracy (all seeds enabled, 0.0-100.0)
- TELE-523: total_synergy (>0.5 green, <-0.5 red)
- TELE-524: individual_contributions (dict[str, float])
- TELE-525: pair_contributions (dict[tuple[str, str], float])

These test the CounterfactualSnapshot dataclass and CounterfactualPanel widget.
"""

import pytest
from rich.text import Text

from esper.karn.sanctum.schema import CounterfactualConfig, CounterfactualSnapshot, SeedState
from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def baseline_config():
    """CounterfactualConfig with all seeds disabled (baseline)."""
    return CounterfactualConfig(
        seed_mask=(False, False, False),
        accuracy=70.0,
    )


@pytest.fixture
def combined_config():
    """CounterfactualConfig with all seeds enabled (combined)."""
    return CounterfactualConfig(
        seed_mask=(True, True, True),
        accuracy=85.0,
    )


@pytest.fixture
def individual_configs():
    """CounterfactualConfigs for individual seed evaluations."""
    return [
        CounterfactualConfig(seed_mask=(True, False, False), accuracy=74.0),  # +4 over baseline
        CounterfactualConfig(seed_mask=(False, True, False), accuracy=72.0),  # +2 over baseline
        CounterfactualConfig(seed_mask=(False, False, True), accuracy=75.0),  # +5 over baseline
    ]


@pytest.fixture
def pair_configs():
    """CounterfactualConfigs for pair evaluations."""
    return [
        CounterfactualConfig(seed_mask=(True, True, False), accuracy=78.0),  # slot 0+1
        CounterfactualConfig(seed_mask=(True, False, True), accuracy=82.0),  # slot 0+2
        CounterfactualConfig(seed_mask=(False, True, True), accuracy=79.0),  # slot 1+2
    ]


@pytest.fixture
def full_snapshot(baseline_config, combined_config, individual_configs, pair_configs):
    """Complete CounterfactualSnapshot with all configurations."""
    all_configs = [baseline_config] + individual_configs + pair_configs + [combined_config]
    return CounterfactualSnapshot(
        slot_ids=("r0c0", "r0c1", "r0c2"),
        configs=all_configs,
        strategy="full_factorial",
        compute_time_ms=150.0,
    )


@pytest.fixture
def ablation_snapshot():
    """CounterfactualSnapshot in ablation_only mode (no pairs)."""
    return CounterfactualSnapshot(
        slot_ids=("r0c0", "r0c1"),
        configs=[
            CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),  # baseline
            CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),   # slot 0 only
            CounterfactualConfig(seed_mask=(False, True), accuracy=73.0),   # slot 1 only
            CounterfactualConfig(seed_mask=(True, True), accuracy=80.0),    # combined
        ],
        strategy="ablation_only",
    )


@pytest.fixture
def unavailable_snapshot():
    """CounterfactualSnapshot with no data available."""
    return CounterfactualSnapshot(
        slot_ids=(),
        configs=[],
        strategy="unavailable",
    )


@pytest.fixture
def mock_seeds():
    """Mock SeedState objects for interaction metrics testing."""
    return {
        "r0c0": SeedState(
            slot_id="r0c0",
            stage="TRAINING",
            interaction_sum=0.8,
            boost_received=0.5,
            contribution_velocity=0.02,
        ),
        "r0c1": SeedState(
            slot_id="r0c1",
            stage="BLENDING",
            interaction_sum=-0.3,
            boost_received=0.0,
            contribution_velocity=-0.015,
        ),
        "r0c2": SeedState(
            slot_id="r0c2",
            stage="DORMANT",  # Should be excluded from active seeds
            interaction_sum=0.0,
            boost_received=0.0,
            contribution_velocity=0.0,
        ),
    }


# -----------------------------------------------------------------------------
# TELE-520: strategy
# -----------------------------------------------------------------------------


class TestTELE520Strategy:
    """TELE-520: Counterfactual strategy enum field."""

    def test_strategy_field_exists_in_schema(self):
        """TELE-520: CounterfactualSnapshot has strategy field."""
        snapshot = CounterfactualSnapshot()
        assert hasattr(snapshot, "strategy")

    def test_strategy_default_value_is_unavailable(self):
        """TELE-520: Default strategy is 'unavailable'."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.strategy == "unavailable"

    def test_strategy_full_factorial_value(self):
        """TELE-520: Strategy can be set to 'full_factorial'."""
        snapshot = CounterfactualSnapshot(strategy="full_factorial")
        assert snapshot.strategy == "full_factorial"

    def test_strategy_ablation_only_value(self):
        """TELE-520: Strategy can be set to 'ablation_only'."""
        snapshot = CounterfactualSnapshot(strategy="ablation_only")
        assert snapshot.strategy == "ablation_only"

    def test_strategy_unavailable_value(self):
        """TELE-520: Strategy can be explicitly set to 'unavailable'."""
        snapshot = CounterfactualSnapshot(strategy="unavailable")
        assert snapshot.strategy == "unavailable"

    def test_strategy_all_enum_values(self):
        """TELE-520: All three strategy enum values are valid."""
        valid_strategies = ["unavailable", "full_factorial", "ablation_only"]
        for strategy in valid_strategies:
            snapshot = CounterfactualSnapshot(strategy=strategy)
            assert snapshot.strategy == strategy

    def test_strategy_type_is_str(self):
        """TELE-520: Strategy field type is str."""
        snapshot = CounterfactualSnapshot(strategy="full_factorial")
        assert isinstance(snapshot.strategy, str)


# -----------------------------------------------------------------------------
# TELE-521: baseline_accuracy
# -----------------------------------------------------------------------------


class TestTELE521BaselineAccuracy:
    """TELE-521: Counterfactual baseline accuracy property."""

    def test_baseline_accuracy_property_exists(self):
        """TELE-521: CounterfactualSnapshot has baseline_accuracy property."""
        snapshot = CounterfactualSnapshot()
        assert hasattr(snapshot, "baseline_accuracy")

    def test_baseline_accuracy_default_is_zero(self):
        """TELE-521: Default baseline_accuracy is 0.0 when no configs."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.baseline_accuracy == 0.0

    def test_baseline_accuracy_returns_all_false_config(self, full_snapshot):
        """TELE-521: baseline_accuracy returns accuracy from all-false mask config."""
        assert full_snapshot.baseline_accuracy == 70.0

    def test_baseline_accuracy_two_seeds(self):
        """TELE-521: baseline_accuracy works with two seeds."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=65.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=75.0),
            ],
        )
        assert snapshot.baseline_accuracy == 65.0

    def test_baseline_accuracy_single_seed(self):
        """TELE-521: baseline_accuracy works with single seed."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0",),
            configs=[
                CounterfactualConfig(seed_mask=(False,), accuracy=60.0),
                CounterfactualConfig(seed_mask=(True,), accuracy=68.0),
            ],
        )
        assert snapshot.baseline_accuracy == 60.0

    def test_baseline_accuracy_zero_when_no_matching_config(self):
        """TELE-521: baseline_accuracy returns 0.0 when no all-false config exists."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(True, False), accuracy=72.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=80.0),
            ],
        )
        assert snapshot.baseline_accuracy == 0.0


# -----------------------------------------------------------------------------
# TELE-522: combined_accuracy
# -----------------------------------------------------------------------------


class TestTELE522CombinedAccuracy:
    """TELE-522: Counterfactual combined accuracy property."""

    def test_combined_accuracy_property_exists(self):
        """TELE-522: CounterfactualSnapshot has combined_accuracy property."""
        snapshot = CounterfactualSnapshot()
        assert hasattr(snapshot, "combined_accuracy")

    def test_combined_accuracy_default_is_zero(self):
        """TELE-522: Default combined_accuracy is 0.0 when no configs."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.combined_accuracy == 0.0

    def test_combined_accuracy_returns_all_true_config(self, full_snapshot):
        """TELE-522: combined_accuracy returns accuracy from all-true mask config."""
        assert full_snapshot.combined_accuracy == 85.0

    def test_combined_accuracy_two_seeds(self):
        """TELE-522: combined_accuracy works with two seeds."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=65.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=78.0),
            ],
        )
        assert snapshot.combined_accuracy == 78.0

    def test_combined_accuracy_single_seed(self):
        """TELE-522: combined_accuracy works with single seed."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0",),
            configs=[
                CounterfactualConfig(seed_mask=(False,), accuracy=60.0),
                CounterfactualConfig(seed_mask=(True,), accuracy=68.0),
            ],
        )
        assert snapshot.combined_accuracy == 68.0

    def test_combined_accuracy_zero_when_no_matching_config(self):
        """TELE-522: combined_accuracy returns 0.0 when no all-true config exists."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=65.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=72.0),
            ],
        )
        assert snapshot.combined_accuracy == 0.0


# -----------------------------------------------------------------------------
# TELE-523: total_synergy
# -----------------------------------------------------------------------------


class TestTELE523TotalSynergy:
    """TELE-523: Counterfactual total synergy method."""

    def test_total_synergy_method_exists(self):
        """TELE-523: CounterfactualSnapshot has total_synergy method."""
        snapshot = CounterfactualSnapshot()
        assert hasattr(snapshot, "total_synergy")
        assert callable(snapshot.total_synergy)

    def test_total_synergy_default_is_zero(self):
        """TELE-523: Default total_synergy is 0.0 when no configs."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.total_synergy() == 0.0

    def test_total_synergy_formula(self, full_snapshot):
        """TELE-523: total_synergy = combined - baseline - sum(individuals)."""
        # baseline = 70, combined = 85
        # individuals: slot0 = 74-70 = 4, slot1 = 72-70 = 2, slot2 = 75-70 = 5
        # sum of individuals = 4 + 2 + 5 = 11
        # expected = baseline + sum = 70 + 11 = 81
        # synergy = combined - expected = 85 - 81 = 4
        synergy = full_snapshot.total_synergy()
        assert synergy == pytest.approx(4.0)

    def test_total_synergy_negative_interference(self):
        """TELE-523: Negative synergy indicates interference."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),  # baseline
                CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),   # +5
                CounterfactualConfig(seed_mask=(False, True), accuracy=74.0),   # +4
                CounterfactualConfig(seed_mask=(True, True), accuracy=77.0),    # combined
            ],
        )
        # expected = 70 + 5 + 4 = 79
        # synergy = 77 - 79 = -2 (interference!)
        synergy = snapshot.total_synergy()
        assert synergy == pytest.approx(-2.0)
        assert synergy < -0.5  # Below threshold for interference warning

    def test_total_synergy_positive_synergy(self, full_snapshot):
        """TELE-523: Positive synergy indicates seeds working together."""
        synergy = full_snapshot.total_synergy()
        assert synergy > 0.5  # Above threshold for synergy indicator

    def test_total_synergy_neutral(self):
        """TELE-523: Near-zero synergy indicates independent seeds."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),  # baseline
                CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),   # +5
                CounterfactualConfig(seed_mask=(False, True), accuracy=74.0),   # +4
                CounterfactualConfig(seed_mask=(True, True), accuracy=79.0),    # expected = 70+5+4=79
            ],
        )
        synergy = snapshot.total_synergy()
        assert synergy == pytest.approx(0.0)
        assert -0.5 <= synergy <= 0.5  # Within neutral range

    def test_total_synergy_health_threshold_green(self, full_snapshot):
        """TELE-523: Synergy > 0.5 is healthy (green threshold)."""
        synergy = full_snapshot.total_synergy()
        # From fixture: synergy = 4.0
        assert synergy > 0.5

    def test_total_synergy_health_threshold_red(self):
        """TELE-523: Synergy < -0.5 is warning (red threshold)."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=74.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=76.0),  # Expected 79, actual 76
            ],
        )
        synergy = snapshot.total_synergy()
        # expected = 70+5+4 = 79, actual = 76, synergy = -3
        assert synergy == pytest.approx(-3.0)
        assert synergy < -0.5


# -----------------------------------------------------------------------------
# TELE-524: individual_contributions
# -----------------------------------------------------------------------------


class TestTELE524IndividualContributions:
    """TELE-524: Counterfactual individual contributions method."""

    def test_individual_contributions_method_exists(self):
        """TELE-524: CounterfactualSnapshot has individual_contributions method."""
        snapshot = CounterfactualSnapshot()
        assert hasattr(snapshot, "individual_contributions")
        assert callable(snapshot.individual_contributions)

    def test_individual_contributions_default_is_empty(self):
        """TELE-524: Default individual_contributions is empty dict when no configs."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.individual_contributions() == {}

    def test_individual_contributions_returns_dict(self, full_snapshot):
        """TELE-524: individual_contributions returns dict[str, float]."""
        result = full_snapshot.individual_contributions()
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())

    def test_individual_contributions_formula(self, full_snapshot):
        """TELE-524: contribution = seed_solo_accuracy - baseline."""
        result = full_snapshot.individual_contributions()
        # baseline = 70.0
        # r0c0: 74.0 - 70.0 = 4.0
        # r0c1: 72.0 - 70.0 = 2.0
        # r0c2: 75.0 - 70.0 = 5.0
        assert result["r0c0"] == pytest.approx(4.0)
        assert result["r0c1"] == pytest.approx(2.0)
        assert result["r0c2"] == pytest.approx(5.0)

    def test_individual_contributions_keys_match_slot_ids(self, full_snapshot):
        """TELE-524: Result keys are slot_ids."""
        result = full_snapshot.individual_contributions()
        assert set(result.keys()) == set(full_snapshot.slot_ids)

    def test_individual_contributions_negative_value(self):
        """TELE-524: Negative contribution when seed hurts model."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0",),
            configs=[
                CounterfactualConfig(seed_mask=(False,), accuracy=70.0),  # baseline
                CounterfactualConfig(seed_mask=(True,), accuracy=68.0),   # worse!
            ],
        )
        result = snapshot.individual_contributions()
        assert result["s0"] == pytest.approx(-2.0)

    def test_individual_contributions_missing_config(self):
        """TELE-524: Slot omitted from result if no matching config."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),  # baseline
                CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),   # s0 only
                # No s1-only config!
            ],
        )
        result = snapshot.individual_contributions()
        assert "s0" in result
        assert "s1" not in result


# -----------------------------------------------------------------------------
# TELE-525: pair_contributions
# -----------------------------------------------------------------------------


class TestTELE525PairContributions:
    """TELE-525: Counterfactual pair contributions method."""

    def test_pair_contributions_method_exists(self):
        """TELE-525: CounterfactualSnapshot has pair_contributions method."""
        snapshot = CounterfactualSnapshot()
        assert hasattr(snapshot, "pair_contributions")
        assert callable(snapshot.pair_contributions)

    def test_pair_contributions_default_is_empty(self):
        """TELE-525: Default pair_contributions is empty dict when no configs."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.pair_contributions() == {}

    def test_pair_contributions_returns_dict(self, full_snapshot):
        """TELE-525: pair_contributions returns dict[tuple[str, str], float]."""
        result = full_snapshot.pair_contributions()
        assert isinstance(result, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())

    def test_pair_contributions_formula(self, full_snapshot):
        """TELE-525: pair_contribution = pair_accuracy - baseline."""
        result = full_snapshot.pair_contributions()
        # baseline = 70.0
        # r0c0 + r0c1: 78.0 - 70.0 = 8.0
        # r0c0 + r0c2: 82.0 - 70.0 = 12.0
        # r0c1 + r0c2: 79.0 - 70.0 = 9.0
        assert result[("r0c0", "r0c1")] == pytest.approx(8.0)
        assert result[("r0c0", "r0c2")] == pytest.approx(12.0)
        assert result[("r0c1", "r0c2")] == pytest.approx(9.0)

    def test_pair_contributions_count_for_three_seeds(self, full_snapshot):
        """TELE-525: Three seeds = 3 pairs (n*(n-1)/2)."""
        result = full_snapshot.pair_contributions()
        assert len(result) == 3  # 3*2/2 = 3

    def test_pair_contributions_count_for_two_seeds(self):
        """TELE-525: Two seeds = 1 pair."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=80.0),
            ],
        )
        result = snapshot.pair_contributions()
        assert len(result) == 1
        assert ("s0", "s1") in result

    def test_pair_contributions_synergy_calculation(self, full_snapshot):
        """TELE-525: Pair synergy = pair_contrib - individual_a - individual_b."""
        pairs = full_snapshot.pair_contributions()
        individuals = full_snapshot.individual_contributions()

        # Pair (r0c0, r0c1): contrib=8, ind0=4, ind1=2, synergy=8-4-2=2
        pair_01 = pairs[("r0c0", "r0c1")]
        ind_0 = individuals["r0c0"]
        ind_1 = individuals["r0c1"]
        synergy_01 = pair_01 - ind_0 - ind_1
        assert synergy_01 == pytest.approx(2.0)

        # Pair (r0c0, r0c2): contrib=12, ind0=4, ind2=5, synergy=12-4-5=3
        pair_02 = pairs[("r0c0", "r0c2")]
        ind_2 = individuals["r0c2"]
        synergy_02 = pair_02 - ind_0 - ind_2
        assert synergy_02 == pytest.approx(3.0)

    def test_pair_contributions_health_threshold_synergy(self, full_snapshot):
        """TELE-525: Pair synergy > 0.5 is highlighted (green threshold)."""
        pairs = full_snapshot.pair_contributions()
        individuals = full_snapshot.individual_contributions()

        # All pairs in our fixture should have positive synergy > 0.5
        for (s1, s2), contrib in pairs.items():
            pair_synergy = contrib - individuals[s1] - individuals[s2]
            assert pair_synergy > 0.5


# -----------------------------------------------------------------------------
# Consumer tests - CounterfactualPanel
# -----------------------------------------------------------------------------


class TestCounterfactualPanelReadsFields:
    """Test CounterfactualPanel correctly reads CounterfactualSnapshot fields."""

    def test_panel_reads_strategy_unavailable(self, unavailable_snapshot):
        """Panel renders unavailable state based on strategy field."""
        panel = CounterfactualPanel(unavailable_snapshot)
        rendered = panel.render()
        # Should use dim border for unavailable state
        assert rendered.border_style == "dim"

    def test_panel_reads_strategy_full_factorial(self, full_snapshot):
        """Panel renders waterfall for full_factorial strategy."""
        panel = CounterfactualPanel(full_snapshot)
        rendered = panel.render()
        # Should use cyan border for available data
        assert rendered.border_style == "cyan"

    def test_panel_reads_strategy_ablation_only(self, ablation_snapshot):
        """Panel renders ablation indicator for ablation_only strategy."""
        panel = CounterfactualPanel(ablation_snapshot)
        rendered = panel.render()
        # Ablation mode should still show waterfall (cyan border)
        assert rendered.border_style == "cyan"

    def test_panel_reads_baseline_accuracy(self, full_snapshot):
        """Panel accesses baseline_accuracy for rendering baseline bar."""
        panel = CounterfactualPanel(full_snapshot)
        # Verify the matrix attribute is correctly stored
        assert panel._matrix.baseline_accuracy == 70.0

    def test_panel_reads_combined_accuracy(self, full_snapshot):
        """Panel accesses combined_accuracy for bar scaling."""
        panel = CounterfactualPanel(full_snapshot)
        assert panel._matrix.combined_accuracy == 85.0

    def test_panel_reads_individual_contributions(self, full_snapshot):
        """Panel accesses individual_contributions for individual bars."""
        panel = CounterfactualPanel(full_snapshot)
        individuals = panel._matrix.individual_contributions()
        assert len(individuals) == 3
        assert individuals["r0c0"] == pytest.approx(4.0)

    def test_panel_reads_pair_contributions(self, full_snapshot):
        """Panel accesses pair_contributions for pairs section."""
        panel = CounterfactualPanel(full_snapshot)
        pairs = panel._matrix.pair_contributions()
        assert len(pairs) == 3
        assert ("r0c0", "r0c1") in pairs

    def test_panel_reads_total_synergy(self, full_snapshot):
        """Panel accesses total_synergy for synergy indicator."""
        panel = CounterfactualPanel(full_snapshot)
        synergy = panel._matrix.total_synergy()
        assert synergy == pytest.approx(4.0)

    def test_panel_update_matrix_method(self, unavailable_snapshot, full_snapshot):
        """Panel update_matrix method replaces matrix correctly."""
        panel = CounterfactualPanel(unavailable_snapshot)
        assert panel._matrix.strategy == "unavailable"

        panel.update_matrix(full_snapshot)
        assert panel._matrix.strategy == "full_factorial"
        assert panel._matrix.baseline_accuracy == 70.0

    def test_panel_reads_seeds_for_interaction_metrics(self, full_snapshot, mock_seeds):
        """Panel uses seeds dict for interaction metrics."""
        panel = CounterfactualPanel(full_snapshot, seeds=mock_seeds)
        assert panel._seeds == mock_seeds

    def test_panel_update_seeds_via_update_matrix(self, full_snapshot, mock_seeds):
        """Panel update_matrix can update seeds dict."""
        panel = CounterfactualPanel(full_snapshot)
        assert panel._seeds == {}

        panel.update_matrix(full_snapshot, seeds=mock_seeds)
        assert panel._seeds == mock_seeds


# -----------------------------------------------------------------------------
# CounterfactualSnapshot schema tests
# -----------------------------------------------------------------------------


class TestCounterfactualSnapshotSchema:
    """Verify CounterfactualSnapshot schema is correct."""

    def test_default_values(self):
        """CounterfactualSnapshot has correct defaults per TELE records."""
        snapshot = CounterfactualSnapshot()

        assert snapshot.slot_ids == ()
        assert snapshot.configs == []
        assert snapshot.strategy == "unavailable"  # TELE-520 default
        assert snapshot.compute_time_ms == 0.0

    def test_baseline_accuracy_default(self):
        """baseline_accuracy defaults to 0.0."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.baseline_accuracy == 0.0  # TELE-521 default

    def test_combined_accuracy_default(self):
        """combined_accuracy defaults to 0.0."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.combined_accuracy == 0.0  # TELE-522 default

    def test_total_synergy_default(self):
        """total_synergy defaults to 0.0."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.total_synergy() == 0.0  # TELE-523 default

    def test_individual_contributions_default(self):
        """individual_contributions defaults to empty dict."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.individual_contributions() == {}  # TELE-524 default

    def test_pair_contributions_default(self):
        """pair_contributions defaults to empty dict."""
        snapshot = CounterfactualSnapshot()
        assert snapshot.pair_contributions() == {}  # TELE-525 default

    def test_all_fields_settable(self):
        """CounterfactualSnapshot fields can be set."""
        configs = [
            CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
            CounterfactualConfig(seed_mask=(True, True), accuracy=80.0),
        ]
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=configs,
            strategy="full_factorial",
            compute_time_ms=100.5,
        )

        assert snapshot.slot_ids == ("s0", "s1")
        assert len(snapshot.configs) == 2
        assert snapshot.strategy == "full_factorial"
        assert snapshot.compute_time_ms == 100.5


# -----------------------------------------------------------------------------
# CounterfactualConfig schema tests
# -----------------------------------------------------------------------------


class TestCounterfactualConfigSchema:
    """Verify CounterfactualConfig schema is correct."""

    def test_seed_mask_field(self):
        """CounterfactualConfig has seed_mask field."""
        config = CounterfactualConfig(seed_mask=(True, False, True), accuracy=75.0)
        assert config.seed_mask == (True, False, True)

    def test_accuracy_field(self):
        """CounterfactualConfig has accuracy field."""
        config = CounterfactualConfig(seed_mask=(True,), accuracy=82.5)
        assert config.accuracy == 82.5

    def test_accuracy_default(self):
        """CounterfactualConfig accuracy defaults to 0.0."""
        config = CounterfactualConfig(seed_mask=(False,))
        assert config.accuracy == 0.0


# -----------------------------------------------------------------------------
# Integration tests - synergy thresholds
# -----------------------------------------------------------------------------


class TestSynergyHealthThresholds:
    """Integration tests for synergy health threshold logic."""

    def test_synergy_above_positive_threshold(self):
        """Synergy > 0.5 is healthy (green)."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=73.0),  # +3
                CounterfactualConfig(seed_mask=(False, True), accuracy=72.0),  # +2
                CounterfactualConfig(seed_mask=(True, True), accuracy=76.0),   # expected=75, synergy=1
            ],
        )
        synergy = snapshot.total_synergy()
        assert synergy > 0.5
        assert synergy == pytest.approx(1.0)

    def test_synergy_below_negative_threshold(self):
        """Synergy < -0.5 is interference (red)."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),  # +5
                CounterfactualConfig(seed_mask=(False, True), accuracy=74.0),  # +4
                CounterfactualConfig(seed_mask=(True, True), accuracy=78.0),   # expected=79, synergy=-1
            ],
        )
        synergy = snapshot.total_synergy()
        assert synergy < -0.5
        assert synergy == pytest.approx(-1.0)

    def test_synergy_within_neutral_range(self):
        """-0.5 <= synergy <= 0.5 is neutral (dim)."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=75.0),  # +5
                CounterfactualConfig(seed_mask=(False, True), accuracy=74.0),  # +4
                CounterfactualConfig(seed_mask=(True, True), accuracy=79.3),   # expected=79, synergy=0.3
            ],
        )
        synergy = snapshot.total_synergy()
        assert -0.5 <= synergy <= 0.5
        assert synergy == pytest.approx(0.3)

    def test_synergy_exactly_at_positive_threshold(self):
        """Synergy = 0.5 is at the edge of healthy range."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=73.0),  # +3
                CounterfactualConfig(seed_mask=(False, True), accuracy=72.0),  # +2
                CounterfactualConfig(seed_mask=(True, True), accuracy=75.5),   # expected=75, synergy=0.5
            ],
        )
        synergy = snapshot.total_synergy()
        assert synergy == pytest.approx(0.5)
        # At exactly 0.5, it's at the threshold boundary

    def test_synergy_exactly_at_negative_threshold(self):
        """Synergy = -0.5 is at the edge of interference range."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("s0", "s1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=70.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=73.0),  # +3
                CounterfactualConfig(seed_mask=(False, True), accuracy=72.0),  # +2
                CounterfactualConfig(seed_mask=(True, True), accuracy=74.5),   # expected=75, synergy=-0.5
            ],
        )
        synergy = snapshot.total_synergy()
        assert synergy == pytest.approx(-0.5)
        # At exactly -0.5, it's at the threshold boundary
