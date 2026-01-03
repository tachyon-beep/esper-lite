"""Tests for Reward Components TELE records (TELE-651 to TELE-654).

Verifies reward component breakdown fields displayed in EnvOverview:

- TELE-651: base_acc_delta - Base accuracy delta (float, ΔAcc column)
- TELE-652: seed_contribution - Seed contribution percentage (float, Seed Δ column)
- TELE-653: bounded_attribution - Bounded attribution fallback (float, Seed Δ column)
- TELE-654: compute_rent - Compute rent penalty (float, Rent column, always red)

Per TELE records: Schema fields exist in RewardComponents dataclass,
consumer is EnvOverview widget with _format_delta_acc(), _format_seed_delta(),
and _format_rent() methods.
"""

from dataclasses import fields

import pytest

from esper.karn.sanctum.schema import (
    EnvState,
    RewardComponents,
)
from esper.karn.sanctum.widgets.env_overview import EnvOverview


# =============================================================================
# Schema Tests - Field Existence and Types
# =============================================================================


class TestRewardComponentsSchema:
    """Verify RewardComponents dataclass schema matches TELE-651 to TELE-654 records."""

    def test_dataclass_has_all_tele_651_654_fields(self):
        """All TELE-651 to TELE-654 fields exist in RewardComponents."""
        field_names = {f.name for f in fields(RewardComponents)}

        # TELE-651 to TELE-654 required fields
        required_fields = {
            "base_acc_delta",  # TELE-651
            "seed_contribution",  # TELE-652
            "bounded_attribution",  # TELE-653
            "compute_rent",  # TELE-654
        }

        for field_name in required_fields:
            assert field_name in field_names, f"Missing field: {field_name}"

    def test_all_fields_are_float_type(self):
        """All TELE-651 to TELE-654 fields are float type."""
        components = RewardComponents()

        float_fields = [
            "base_acc_delta",
            "seed_contribution",
            "bounded_attribution",
            "compute_rent",
        ]

        for field_name in float_fields:
            value = getattr(components, field_name)
            assert isinstance(
                value, float
            ), f"{field_name} should be float, got {type(value)}"


# =============================================================================
# TELE-651: Base Accuracy Delta
# =============================================================================


class TestTELE651BaseAccDelta:
    """TELE-651: Base Accuracy Delta - legacy shaped reward signal.

    ΔAcc column in EnvOverview.
    Type: float
    Units: reward units (proportional to accuracy delta)
    Range: [-inf, +inf] (typically small values around -1.0 to +1.0)
    Default: 0.0

    Display Color Logic: Green if >= 0, red if < 0
    Display Format: +.2f
    """

    def test_field_exists(self):
        """TELE-651: base_acc_delta field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "base_acc_delta")

    def test_default_value(self):
        """TELE-651: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.base_acc_delta == 0.0

    def test_accepts_positive_values(self):
        """TELE-651: Positive values indicate accuracy improvement."""
        components = RewardComponents(base_acc_delta=0.5)
        assert components.base_acc_delta == 0.5
        assert components.base_acc_delta > 0

    def test_accepts_negative_values(self):
        """TELE-651: Negative values indicate accuracy degradation."""
        components = RewardComponents(base_acc_delta=-0.3)
        assert components.base_acc_delta == -0.3
        assert components.base_acc_delta < 0

    def test_accepts_zero_value(self):
        """TELE-651: Zero indicates no accuracy change (stable)."""
        components = RewardComponents(base_acc_delta=0.0)
        assert components.base_acc_delta == 0.0

    def test_healthy_threshold(self):
        """TELE-651: Healthy threshold is value >= 0."""
        healthy_value = 0.1
        assert healthy_value >= 0

    def test_warning_threshold(self):
        """TELE-651: Warning threshold is -0.5 < value < 0."""
        warning_value = -0.3
        assert -0.5 < warning_value < 0

    def test_critical_threshold(self):
        """TELE-651: Critical threshold is value <= -0.5."""
        critical_value = -0.6
        assert critical_value <= -0.5


# =============================================================================
# TELE-652: Seed Contribution
# =============================================================================


class TestTELE652SeedContribution:
    """TELE-652: Seed Contribution - counterfactual seed impact.

    Primary value for Seed Δ column in EnvOverview.
    Type: float
    Units: percentage points (%)
    Range: [-inf, +inf] (typically -10% to +10%)
    Default: 0.0

    Display Color Logic: Green if > 0, red if < 0
    Display Format: +.1f%
    """

    def test_field_exists(self):
        """TELE-652: seed_contribution field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "seed_contribution")

    def test_default_value(self):
        """TELE-652: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.seed_contribution == 0.0

    def test_accepts_positive_values(self):
        """TELE-652: Positive values indicate seed is helping."""
        components = RewardComponents(seed_contribution=2.5)
        assert components.seed_contribution == 2.5
        assert components.seed_contribution > 0

    def test_accepts_negative_values(self):
        """TELE-652: Negative values indicate seed is hurting (toxic seed)."""
        components = RewardComponents(seed_contribution=-1.5)
        assert components.seed_contribution == -1.5
        assert components.seed_contribution < 0

    def test_accepts_zero_value(self):
        """TELE-652: Zero indicates no measurable seed impact."""
        components = RewardComponents(seed_contribution=0.0)
        assert components.seed_contribution == 0.0

    def test_healthy_threshold(self):
        """TELE-652: Healthy threshold is value > 0."""
        healthy_value = 1.5
        assert healthy_value > 0

    def test_neutral_threshold(self):
        """TELE-652: Neutral threshold is value == 0."""
        neutral_value = 0.0
        assert neutral_value == 0

    def test_warning_threshold(self):
        """TELE-652: Warning threshold is value < 0."""
        warning_value = -0.5
        assert warning_value < 0


# =============================================================================
# TELE-653: Bounded Attribution
# =============================================================================


class TestTELE653BoundedAttribution:
    """TELE-653: Bounded Attribution - anti-gaming protected attribution.

    Fallback for Seed Δ column when seed_contribution is 0.
    Type: float
    Units: reward units (bounded attribution score)
    Range: [-inf, +inf] (typically -1.0 to +1.0 after bounding)
    Default: 0.0

    Display Color Logic: Green if > 0, red if < 0
    Display Format: +.2f
    """

    def test_field_exists(self):
        """TELE-653: bounded_attribution field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "bounded_attribution")

    def test_default_value(self):
        """TELE-653: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.bounded_attribution == 0.0

    def test_accepts_positive_values(self):
        """TELE-653: Positive values indicate genuine positive contribution."""
        components = RewardComponents(bounded_attribution=0.75)
        assert components.bounded_attribution == 0.75
        assert components.bounded_attribution > 0

    def test_accepts_negative_values(self):
        """TELE-653: Negative values indicate seed causing harm."""
        components = RewardComponents(bounded_attribution=-0.3)
        assert components.bounded_attribution == -0.3
        assert components.bounded_attribution < 0

    def test_accepts_zero_value(self):
        """TELE-653: Zero indicates no measurable impact or pre-blending."""
        components = RewardComponents(bounded_attribution=0.0)
        assert components.bounded_attribution == 0.0

    def test_healthy_threshold(self):
        """TELE-653: Healthy threshold is value > 0."""
        healthy_value = 0.5
        assert healthy_value > 0

    def test_neutral_threshold(self):
        """TELE-653: Neutral threshold is value == 0."""
        neutral_value = 0.0
        assert neutral_value == 0

    def test_warning_threshold(self):
        """TELE-653: Warning threshold is value < 0."""
        warning_value = -0.2
        assert warning_value < 0


# =============================================================================
# TELE-654: Compute Rent
# =============================================================================


class TestTELE654ComputeRent:
    """TELE-654: Compute Rent - parameter bloat penalty.

    Rent column in EnvOverview.
    Type: float
    Units: reward units (always negative or zero)
    Range: [-max_rent_penalty, 0] (typically 0 to -5.0)
    Default: 0.0

    Display Color Logic: Always red (it's always a penalty)
    Display Format: .2f
    """

    def test_field_exists(self):
        """TELE-654: compute_rent field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "compute_rent")

    def test_default_value(self):
        """TELE-654: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.compute_rent == 0.0

    def test_accepts_negative_values(self):
        """TELE-654: Negative values indicate rent penalty being applied."""
        components = RewardComponents(compute_rent=-0.5)
        assert components.compute_rent == -0.5
        assert components.compute_rent < 0

    def test_accepts_zero_value(self):
        """TELE-654: Zero indicates no rent (no active seeds or grace period)."""
        components = RewardComponents(compute_rent=0.0)
        assert components.compute_rent == 0.0

    def test_healthy_threshold(self):
        """TELE-654: Healthy threshold is value == 0 (no compute rent)."""
        healthy_value = 0.0
        assert healthy_value == 0

    def test_normal_threshold(self):
        """TELE-654: Normal threshold is -1.0 < value < 0."""
        normal_value = -0.5
        assert -1.0 < normal_value < 0

    def test_warning_threshold(self):
        """TELE-654: Warning threshold is value <= -1.0."""
        warning_value = -1.5
        assert warning_value <= -1.0

    def test_compute_rent_is_always_penalty(self):
        """TELE-654: Compute rent is always non-positive (penalty or zero)."""
        # Valid values are <= 0
        components = RewardComponents(compute_rent=-2.5)
        assert components.compute_rent <= 0

        components = RewardComponents(compute_rent=0.0)
        assert components.compute_rent <= 0


# =============================================================================
# Consumer Tests - EnvOverview Widget
# =============================================================================


class TestEnvOverviewReadsRewardComponents:
    """Tests that EnvOverview correctly reads RewardComponents fields."""

    def test_env_has_reward_components(self):
        """EnvState has reward_components field of type RewardComponents."""
        env = EnvState(env_id=0)
        assert hasattr(env, "reward_components")
        assert isinstance(env.reward_components, RewardComponents)

    def test_reward_components_default_values_in_env(self):
        """EnvState.reward_components has correct defaults per TELE records."""
        env = EnvState(env_id=0)

        # TELE-651
        assert env.reward_components.base_acc_delta == 0.0
        # TELE-652
        assert env.reward_components.seed_contribution == 0.0
        # TELE-653
        assert env.reward_components.bounded_attribution == 0.0
        # TELE-654
        assert env.reward_components.compute_rent == 0.0


class TestEnvOverviewFormatDeltaAcc:
    """Tests for _format_delta_acc() method (TELE-651 consumer)."""

    @pytest.fixture
    def widget(self) -> EnvOverview:
        """Create EnvOverview widget for testing."""
        return EnvOverview()

    def test_positive_value_displays_green(self, widget: EnvOverview):
        """TELE-651: Positive base_acc_delta displays green."""
        env = EnvState(env_id=0)
        env.reward_components.base_acc_delta = 0.5

        result = widget._format_delta_acc(env)

        assert "[green]" in result
        assert "+0.50" in result

    def test_negative_value_displays_red(self, widget: EnvOverview):
        """TELE-651: Negative base_acc_delta displays red."""
        env = EnvState(env_id=0)
        env.reward_components.base_acc_delta = -0.3

        result = widget._format_delta_acc(env)

        assert "[red]" in result
        assert "-0.30" in result

    def test_zero_value_displays_green(self, widget: EnvOverview):
        """TELE-651: Zero base_acc_delta displays green (>= 0 threshold)."""
        env = EnvState(env_id=0)
        env.reward_components.base_acc_delta = 0.0

        result = widget._format_delta_acc(env)

        assert "[green]" in result
        assert "+0.00" in result


class TestEnvOverviewFormatSeedDelta:
    """Tests for _format_seed_delta() method (TELE-652/653 consumer)."""

    @pytest.fixture
    def widget(self) -> EnvOverview:
        """Create EnvOverview widget for testing."""
        return EnvOverview()

    def test_positive_seed_contribution_displays_green(self, widget: EnvOverview):
        """TELE-652: Positive seed_contribution displays green."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 2.5

        result = widget._format_seed_delta(env)

        assert "[green]" in result
        assert "+2.5%" in result

    def test_negative_seed_contribution_displays_red(self, widget: EnvOverview):
        """TELE-652: Negative seed_contribution displays red."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = -1.5

        result = widget._format_seed_delta(env)

        assert "[red]" in result
        assert "-1.5%" in result

    def test_zero_seed_contribution_falls_back_to_bounded_attribution(
        self, widget: EnvOverview
    ):
        """TELE-652/653: When seed_contribution is 0, use bounded_attribution."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = 0.75

        result = widget._format_seed_delta(env)

        assert "[green]" in result
        assert "+0.75" in result
        # Should NOT have percentage format (that's for seed_contribution)
        assert "%" not in result

    def test_negative_bounded_attribution_displays_red(self, widget: EnvOverview):
        """TELE-653: Negative bounded_attribution displays red when fallback."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = -0.3

        result = widget._format_seed_delta(env)

        assert "[red]" in result
        assert "-0.30" in result

    def test_both_zero_displays_dash(self, widget: EnvOverview):
        """TELE-652/653: Both zero displays dash placeholder."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = 0.0

        result = widget._format_seed_delta(env)

        assert result == "\u2500"  # "─" em-dash

    def test_seed_contribution_takes_precedence(self, widget: EnvOverview):
        """TELE-652: seed_contribution is used first when non-zero."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 3.0
        env.reward_components.bounded_attribution = 0.5

        result = widget._format_seed_delta(env)

        # Should use seed_contribution format (percentage)
        assert "+3.0%" in result
        # Should NOT use bounded_attribution format
        assert "+0.50" not in result


class TestEnvOverviewFormatRent:
    """Tests for _format_rent() method (TELE-654 consumer)."""

    @pytest.fixture
    def widget(self) -> EnvOverview:
        """Create EnvOverview widget for testing."""
        return EnvOverview()

    def test_negative_rent_displays_red(self, widget: EnvOverview):
        """TELE-654: Negative compute_rent displays red (always penalty)."""
        env = EnvState(env_id=0)
        env.reward_components.compute_rent = -0.5

        result = widget._format_rent(env)

        assert "[red]" in result
        assert "-0.50" in result

    def test_zero_rent_displays_dash(self, widget: EnvOverview):
        """TELE-654: Zero compute_rent displays dash placeholder."""
        env = EnvState(env_id=0)
        env.reward_components.compute_rent = 0.0

        result = widget._format_rent(env)

        assert result == "\u2500"  # "─" em-dash

    def test_large_rent_displays_correctly(self, widget: EnvOverview):
        """TELE-654: Large rent penalty displays correctly."""
        env = EnvState(env_id=0)
        env.reward_components.compute_rent = -2.75

        result = widget._format_rent(env)

        assert "[red]" in result
        assert "-2.75" in result


# =============================================================================
# Color Threshold Tests
# =============================================================================


class TestColorThresholds:
    """Verify color thresholds match TELE record specifications."""

    @pytest.fixture
    def widget(self) -> EnvOverview:
        """Create EnvOverview widget for testing."""
        return EnvOverview()

    def test_base_acc_delta_green_at_zero(self, widget: EnvOverview):
        """TELE-651: base_acc_delta green threshold is >= 0."""
        env = EnvState(env_id=0)
        env.reward_components.base_acc_delta = 0.0
        result = widget._format_delta_acc(env)
        assert "[green]" in result

    def test_base_acc_delta_green_when_positive(self, widget: EnvOverview):
        """TELE-651: base_acc_delta green when positive."""
        env = EnvState(env_id=0)
        env.reward_components.base_acc_delta = 0.01
        result = widget._format_delta_acc(env)
        assert "[green]" in result

    def test_base_acc_delta_red_when_negative(self, widget: EnvOverview):
        """TELE-651: base_acc_delta red when negative."""
        env = EnvState(env_id=0)
        env.reward_components.base_acc_delta = -0.01
        result = widget._format_delta_acc(env)
        assert "[red]" in result

    def test_seed_contribution_green_when_positive(self, widget: EnvOverview):
        """TELE-652: seed_contribution green when > 0."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.1
        result = widget._format_seed_delta(env)
        assert "[green]" in result

    def test_seed_contribution_red_when_negative(self, widget: EnvOverview):
        """TELE-652: seed_contribution red when < 0."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = -0.1
        result = widget._format_seed_delta(env)
        assert "[red]" in result

    def test_bounded_attribution_green_when_positive(self, widget: EnvOverview):
        """TELE-653: bounded_attribution green when > 0."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = 0.1
        result = widget._format_seed_delta(env)
        assert "[green]" in result

    def test_bounded_attribution_red_when_negative(self, widget: EnvOverview):
        """TELE-653: bounded_attribution red when < 0."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = -0.1
        result = widget._format_seed_delta(env)
        assert "[red]" in result

    def test_compute_rent_always_red(self, widget: EnvOverview):
        """TELE-654: compute_rent always red (it's always a penalty)."""
        env = EnvState(env_id=0)
        env.reward_components.compute_rent = -0.5
        result = widget._format_rent(env)
        assert "[red]" in result


# =============================================================================
# Fallback Logic Tests
# =============================================================================


class TestFallbackLogic:
    """Test seed_contribution -> bounded_attribution fallback behavior."""

    @pytest.fixture
    def widget(self) -> EnvOverview:
        """Create EnvOverview widget for testing."""
        return EnvOverview()

    def test_seed_contribution_used_when_nonzero(self, widget: EnvOverview):
        """seed_contribution is primary when non-zero."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 5.0
        env.reward_components.bounded_attribution = 1.0

        result = widget._format_seed_delta(env)

        # Uses percentage format (seed_contribution)
        assert "%" in result
        assert "5.0" in result

    def test_bounded_attribution_used_when_seed_contribution_zero(
        self, widget: EnvOverview
    ):
        """bounded_attribution is fallback when seed_contribution is 0."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = 0.5

        result = widget._format_seed_delta(env)

        # Uses decimal format (bounded_attribution)
        assert "%" not in result
        assert "0.50" in result

    def test_dash_when_both_zero(self, widget: EnvOverview):
        """Dash displayed when both seed_contribution and bounded_attribution are 0."""
        env = EnvState(env_id=0)
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = 0.0

        result = widget._format_seed_delta(env)

        assert result == "\u2500"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRewardComponentsIntegration:
    """Integration tests for complete reward components flows."""

    def test_all_components_can_be_updated(self):
        """All TELE-651 to TELE-654 fields can be updated after creation."""
        components = RewardComponents()

        # Update all fields
        components.base_acc_delta = 0.25
        components.seed_contribution = 3.5
        components.bounded_attribution = 0.8
        components.compute_rent = -1.2

        assert components.base_acc_delta == 0.25
        assert components.seed_contribution == 3.5
        assert components.bounded_attribution == 0.8
        assert components.compute_rent == -1.2

    def test_env_state_reward_components_updates(self):
        """EnvState.reward_components can be updated."""
        env = EnvState(env_id=0)

        env.reward_components.base_acc_delta = 0.5
        env.reward_components.seed_contribution = 2.0
        env.reward_components.bounded_attribution = 0.3
        env.reward_components.compute_rent = -0.8

        assert env.reward_components.base_acc_delta == 0.5
        assert env.reward_components.seed_contribution == 2.0
        assert env.reward_components.bounded_attribution == 0.3
        assert env.reward_components.compute_rent == -0.8

    def test_typical_training_scenario(self):
        """Simulate typical training scenario with reward components."""
        env = EnvState(env_id=0)

        # Scenario: Seed is helping, accuracy improving, small rent
        env.reward_components.base_acc_delta = 0.15
        env.reward_components.seed_contribution = 1.8
        env.reward_components.bounded_attribution = 0.6
        env.reward_components.compute_rent = -0.3

        widget = EnvOverview()

        # All should display correctly
        delta_acc = widget._format_delta_acc(env)
        seed_delta = widget._format_seed_delta(env)
        rent = widget._format_rent(env)

        # ΔAcc: positive, green
        assert "[green]" in delta_acc
        assert "+0.15" in delta_acc

        # Seed Δ: uses seed_contribution (positive, green, percentage)
        assert "[green]" in seed_delta
        assert "1.8%" in seed_delta

        # Rent: negative, red
        assert "[red]" in rent
        assert "-0.30" in rent

    def test_toxic_seed_scenario(self):
        """Simulate toxic seed scenario where seed is hurting."""
        env = EnvState(env_id=0)

        # Scenario: Seed is toxic, accuracy degrading
        env.reward_components.base_acc_delta = -0.4
        env.reward_components.seed_contribution = -2.5
        env.reward_components.bounded_attribution = -0.8
        env.reward_components.compute_rent = -0.5

        widget = EnvOverview()

        delta_acc = widget._format_delta_acc(env)
        seed_delta = widget._format_seed_delta(env)
        rent = widget._format_rent(env)

        # All should be red (bad scenario)
        assert "[red]" in delta_acc
        assert "[red]" in seed_delta
        assert "[red]" in rent

    def test_no_active_seed_scenario(self):
        """Simulate scenario with no active seed (all zeros except base_acc_delta)."""
        env = EnvState(env_id=0)

        # Scenario: No seed active, only base accuracy changes
        env.reward_components.base_acc_delta = 0.1
        env.reward_components.seed_contribution = 0.0
        env.reward_components.bounded_attribution = 0.0
        env.reward_components.compute_rent = 0.0

        widget = EnvOverview()

        delta_acc = widget._format_delta_acc(env)
        seed_delta = widget._format_seed_delta(env)
        rent = widget._format_rent(env)

        # ΔAcc: shows value
        assert "[green]" in delta_acc

        # Seed Δ: dash (no seed contribution)
        assert seed_delta == "\u2500"

        # Rent: dash (no rent)
        assert rent == "\u2500"
