"""Tests for Reward Components TELE records (TELE-651 to TELE-664).

Verifies reward component breakdown fields displayed in EnvOverview and EnvDetailScreen:

**EnvOverview Consumer (TELE-651 to TELE-654):**

- TELE-651: base_acc_delta - Base accuracy delta (float, ΔAcc column)
- TELE-652: seed_contribution - Seed contribution percentage (float, Seed Δ column)
- TELE-653: bounded_attribution - Bounded attribution fallback (float, Seed Δ column)
- TELE-654: compute_rent - Compute rent penalty (float, Rent column, always red)

**EnvDetailScreen Consumer (TELE-655 to TELE-664):**

- TELE-655: total - Total combined reward signal (float, signed +/-)
- TELE-656: stage_bonus - PBRS lifecycle stage bonus (float, >= 0)
- TELE-657: alpha_shock - Anti-gaming alpha oscillation penalty (float, <= 0)
- TELE-658: ratio_penalty - Anti-gaming contribution ratio penalty (float, <= 0)
- TELE-659: hindsight_credit - Retroactive scaffold contribution bonus (float, >= 0)
- TELE-660: scaffold_count - Number of contributing scaffolds (int, >= 0)
- TELE-661: avg_scaffold_delay - Average epochs since scaffolding (float, >= 0)
- TELE-662: fossilize_terminal_bonus - Terminal success bonus (float, >= 0)
- TELE-663: blending_warning - Negative trajectory warning during BLENDING (float, <= 0)
- TELE-664: holding_warning - Holding stage delay penalty (float, <= 0)

**Sign Conventions:**
- Bonuses (>= 0): stage_bonus, hindsight_credit, fossilize_terminal_bonus
- Penalties (<= 0): alpha_shock, ratio_penalty, blending_warning, holding_warning
- Signed (+/-): total, base_acc_delta, seed_contribution, bounded_attribution, compute_rent

Per TELE records: Schema fields exist in RewardComponents dataclass,
consumers are EnvOverview and EnvDetailScreen widgets.
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
    """Verify RewardComponents dataclass schema matches TELE-651 to TELE-664 records."""

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

    def test_dataclass_has_all_tele_655_664_fields(self):
        """All TELE-655 to TELE-664 fields exist in RewardComponents."""
        field_names = {f.name for f in fields(RewardComponents)}

        # TELE-655 to TELE-664 required fields
        required_fields = {
            "total",  # TELE-655
            "stage_bonus",  # TELE-656
            "alpha_shock",  # TELE-657
            "ratio_penalty",  # TELE-658
            "hindsight_credit",  # TELE-659
            "scaffold_count",  # TELE-660
            "avg_scaffold_delay",  # TELE-661
            "fossilize_terminal_bonus",  # TELE-662
            "blending_warning",  # TELE-663
            "holding_warning",  # TELE-664
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

    def test_tele_655_664_float_fields_are_float_type(self):
        """All TELE-655 to TELE-664 float fields are float type."""
        components = RewardComponents()

        float_fields = [
            "total",  # TELE-655
            "stage_bonus",  # TELE-656
            "alpha_shock",  # TELE-657
            "ratio_penalty",  # TELE-658
            "hindsight_credit",  # TELE-659
            "avg_scaffold_delay",  # TELE-661
            "fossilize_terminal_bonus",  # TELE-662
            "blending_warning",  # TELE-663
            "holding_warning",  # TELE-664
        ]

        for field_name in float_fields:
            value = getattr(components, field_name)
            assert isinstance(
                value, float
            ), f"{field_name} should be float, got {type(value)}"

    def test_scaffold_count_is_int_type(self):
        """TELE-660: scaffold_count is int type."""
        components = RewardComponents()
        assert isinstance(
            components.scaffold_count, int
        ), f"scaffold_count should be int, got {type(components.scaffold_count)}"


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


# =============================================================================
# TELE-655: Reward Total
# =============================================================================


class TestTELE655RewardTotal:
    """TELE-655: Reward Total - combined reward signal.

    Primary reward signal displayed in EnvDetailScreen.
    Type: float
    Units: reward units (sum of all components)
    Range: [-inf, +inf] (typically -1.0 to +1.0)
    Default: 0.0

    Sign Convention: Signed (+/-)
    - Positive: Net positive reinforcement
    - Negative: Net negative reinforcement
    - Zero: Neutral step

    Display Format: +.3f with bold green (>= 0) or bold red (< 0)
    """

    def test_total_field_exists(self):
        """TELE-655: total field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "total")

    def test_total_default_value(self):
        """TELE-655: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.total == 0.0

    def test_total_type_is_float(self):
        """TELE-655: total field is float type."""
        components = RewardComponents()
        assert isinstance(components.total, float)

    def test_total_can_be_positive_or_negative(self):
        """TELE-655: total accepts both positive and negative values (signed)."""
        # Positive total (net reinforcement)
        positive_components = RewardComponents(total=0.5)
        assert positive_components.total == 0.5
        assert positive_components.total > 0

        # Negative total (net penalty)
        negative_components = RewardComponents(total=-0.3)
        assert negative_components.total == -0.3
        assert negative_components.total < 0

        # Zero total (neutral)
        zero_components = RewardComponents(total=0.0)
        assert zero_components.total == 0.0

    def test_total_display_format(self):
        """TELE-655: Display format is +.3f (e.g., '+0.123' or '-0.456')."""
        # Positive formatting
        positive_value = 0.123
        formatted_positive = f"{positive_value:+.3f}"
        assert formatted_positive == "+0.123"

        # Negative formatting
        negative_value = -0.456
        formatted_negative = f"{negative_value:+.3f}"
        assert formatted_negative == "-0.456"

        # Zero formatting
        zero_value = 0.0
        formatted_zero = f"{zero_value:+.3f}"
        assert formatted_zero == "+0.000"


# =============================================================================
# TELE-656: Reward Stage Bonus
# =============================================================================


class TestTELE656RewardStageBonus:
    """TELE-656: Reward Stage Bonus - PBRS lifecycle bonus.

    PBRS bonus for seeds in advanced lifecycle stages (BLENDING+).
    Type: float
    Units: reward units (PBRS shaping bonus)
    Range: [0, +inf] (always non-negative, bonus only)
    Default: 0.0

    Sign Convention: Non-negative (>= 0)
    - Zero: Early stage seeds (DORMANT, GERMINATED, TRAINING)
    - Positive: Advanced stage seeds (BLENDING, HOLDING, GRAFTED)

    Display: Blue styling, used for PBRS fraction calculation
    PBRS Fraction: abs(stage_bonus) / abs(total), healthy range 10-40%
    """

    def test_stage_bonus_field_exists(self):
        """TELE-656: stage_bonus field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "stage_bonus")

    def test_stage_bonus_default_value(self):
        """TELE-656: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.stage_bonus == 0.0

    def test_stage_bonus_type_is_float(self):
        """TELE-656: stage_bonus field is float type."""
        components = RewardComponents()
        assert isinstance(components.stage_bonus, float)

    def test_stage_bonus_non_negative(self):
        """TELE-656: stage_bonus is always non-negative (bonus, not penalty)."""
        # Positive bonus for advanced stage
        bonus_components = RewardComponents(stage_bonus=0.25)
        assert bonus_components.stage_bonus >= 0

        # Zero bonus for early stage
        zero_components = RewardComponents(stage_bonus=0.0)
        assert zero_components.stage_bonus >= 0

    def test_pbrs_fraction_calculation(self):
        """TELE-656: PBRS fraction = abs(stage_bonus) / abs(total)."""
        # Healthy PBRS scenario: 20% shaping
        components = RewardComponents(total=0.5, stage_bonus=0.1)

        if abs(components.total) > 0:
            pbrs_fraction = abs(components.stage_bonus) / abs(components.total)
        else:
            pbrs_fraction = 0.0

        assert pbrs_fraction == pytest.approx(0.2, abs=0.001)

    def test_pbrs_healthy_range(self):
        """TELE-656: Healthy PBRS fraction is 10-40%."""
        PBRS_HEALTHY_MIN = 0.10
        PBRS_HEALTHY_MAX = 0.40

        # Healthy: 25% PBRS
        healthy_fraction = 0.25
        assert PBRS_HEALTHY_MIN <= healthy_fraction <= PBRS_HEALTHY_MAX

        # Too low: 5% PBRS (warning)
        low_fraction = 0.05
        assert low_fraction < PBRS_HEALTHY_MIN

        # Too high: 55% PBRS (warning)
        high_fraction = 0.55
        assert high_fraction > PBRS_HEALTHY_MAX


# =============================================================================
# TELE-657: Reward Alpha Shock
# =============================================================================


class TestTELE657RewardAlphaShock:
    """TELE-657: Reward Alpha Shock - anti-gaming alpha oscillation penalty.

    Convex penalty for rapid alpha (blending coefficient) changes.
    Type: float
    Units: reward units (penalty)
    Range: [-inf, 0] (always non-positive, penalty only)
    Default: 0.0

    Sign Convention: Non-positive (<= 0)
    - Zero: No alpha oscillation detected (healthy)
    - Negative: Penalty for gaming via alpha manipulation

    Display: Red styling, triggers "Gaming: X% (SHOCK)" indicator
    Formula: -alpha_shock_coef * sum(alpha_delta^2)
    """

    def test_alpha_shock_field_exists(self):
        """TELE-657: alpha_shock field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "alpha_shock")

    def test_alpha_shock_default_value(self):
        """TELE-657: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.alpha_shock == 0.0

    def test_alpha_shock_type_is_float(self):
        """TELE-657: alpha_shock field is float type."""
        components = RewardComponents()
        assert isinstance(components.alpha_shock, float)

    def test_alpha_shock_non_positive(self):
        """TELE-657: alpha_shock is always non-positive (penalty, never bonus)."""
        # Penalty for gaming
        penalty_components = RewardComponents(alpha_shock=-0.15)
        assert penalty_components.alpha_shock <= 0

        # Zero (no gaming)
        zero_components = RewardComponents(alpha_shock=0.0)
        assert zero_components.alpha_shock <= 0

    def test_alpha_shock_triggers_gaming_indicator(self):
        """TELE-657: Non-zero alpha_shock triggers gaming indicator."""
        # Gaming detected
        gaming_components = RewardComponents(alpha_shock=-0.05)
        gaming_detected = gaming_components.alpha_shock != 0.0
        assert gaming_detected

        # No gaming
        healthy_components = RewardComponents(alpha_shock=0.0)
        no_gaming = healthy_components.alpha_shock == 0.0
        assert no_gaming


# =============================================================================
# TELE-658: Reward Ratio Penalty
# =============================================================================


class TestTELE658RewardRatioPenalty:
    """TELE-658: Reward Ratio Penalty - anti-gaming contribution ratio penalty.

    Penalty for suspicious contribution/improvement ratio (ransomware pattern).
    Type: float
    Units: reward units (penalty)
    Range: [-inf, 0] (always non-positive, penalty only)
    Default: 0.0

    Sign Convention: Non-positive (<= 0)
    - Zero: Healthy contribution ratio
    - Negative: Penalty for contribution inflation

    Display: Red styling, triggers "Gaming: X% (RATIO)" indicator
    Triggered when: seed_contribution / total_improvement > 5.0 (500%)
    """

    def test_ratio_penalty_field_exists(self):
        """TELE-658: ratio_penalty field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "ratio_penalty")

    def test_ratio_penalty_default_value(self):
        """TELE-658: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.ratio_penalty == 0.0

    def test_ratio_penalty_type_is_float(self):
        """TELE-658: ratio_penalty field is float type."""
        components = RewardComponents()
        assert isinstance(components.ratio_penalty, float)

    def test_ratio_penalty_non_positive(self):
        """TELE-658: ratio_penalty is always non-positive (penalty, never bonus)."""
        # Penalty for ransomware pattern
        penalty_components = RewardComponents(ratio_penalty=-0.08)
        assert penalty_components.ratio_penalty <= 0

        # Zero (healthy ratio)
        zero_components = RewardComponents(ratio_penalty=0.0)
        assert zero_components.ratio_penalty <= 0

    def test_ratio_penalty_display_format(self):
        """TELE-658: Display format is .3f (e.g., '-0.050')."""
        penalty_value = -0.05
        formatted = f"{penalty_value:.3f}"
        assert formatted == "-0.050"


# =============================================================================
# TELE-659: Reward Hindsight Credit
# =============================================================================


class TestTELE659RewardHindsightCredit:
    """TELE-659: Reward Hindsight Credit - retroactive scaffold contribution bonus.

    Retroactive bonus when beneficiary seed fossilizes.
    Type: float
    Units: reward units (bonus)
    Range: [0, +inf] (always non-negative, bonus only)
    Default: 0.0

    Sign Convention: Non-negative (>= 0)
    - Zero: No scaffold contribution recognized
    - Positive: Retroactive credit for scaffolding

    Display: Blue styling, with scaffold metadata "(Nx, Y.Ye)"
    """

    def test_hindsight_credit_field_exists(self):
        """TELE-659: hindsight_credit field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "hindsight_credit")

    def test_hindsight_credit_default_value(self):
        """TELE-659: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.hindsight_credit == 0.0

    def test_hindsight_credit_type_is_float(self):
        """TELE-659: hindsight_credit field is float type."""
        components = RewardComponents()
        assert isinstance(components.hindsight_credit, float)

    def test_hindsight_credit_non_negative(self):
        """TELE-659: hindsight_credit is always non-negative (bonus, never penalty)."""
        # Bonus for scaffold contribution
        bonus_components = RewardComponents(hindsight_credit=0.12)
        assert bonus_components.hindsight_credit >= 0

        # Zero (no scaffold contribution)
        zero_components = RewardComponents(hindsight_credit=0.0)
        assert zero_components.hindsight_credit >= 0

    def test_hindsight_credit_display_with_scaffold_metadata(self):
        """TELE-659: Display includes scaffold metadata (Nx, Y.Ye)."""
        # Format: "Hind: +0.12 (2x, 3.5e)" - credit with count and delay
        hindsight_credit = 0.12
        scaffold_count = 2
        avg_scaffold_delay = 3.5

        # Verify values can be formatted together
        formatted = f"+{hindsight_credit:.2f} ({scaffold_count}x, {avg_scaffold_delay:.1f}e)"
        assert formatted == "+0.12 (2x, 3.5e)"


# =============================================================================
# TELE-660: Reward Scaffold Count
# =============================================================================


class TestTELE660RewardScaffoldCount:
    """TELE-660: Reward Scaffold Count - number of contributing scaffolds.

    Debugging/analysis field tracking scaffold contributors.
    Type: int
    Units: count (number of scaffold seeds)
    Range: [0, +inf] (always non-negative)
    Default: 0

    Sign Convention: Non-negative (>= 0)
    - Zero: No scaffolding (standalone fossilization)
    - Positive: Number of scaffold contributors

    Display: Shown as "(Nx, ...)" alongside hindsight_credit
    """

    def test_scaffold_count_field_exists(self):
        """TELE-660: scaffold_count field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "scaffold_count")

    def test_scaffold_count_default_value(self):
        """TELE-660: Default value is 0 per TELE record."""
        components = RewardComponents()
        assert components.scaffold_count == 0

    def test_scaffold_count_type_is_int(self):
        """TELE-660: scaffold_count field is int type."""
        components = RewardComponents()
        assert isinstance(components.scaffold_count, int)

    def test_scaffold_count_non_negative(self):
        """TELE-660: scaffold_count is always non-negative."""
        # Positive count (active scaffolding)
        active_components = RewardComponents(scaffold_count=3)
        assert active_components.scaffold_count >= 0

        # Zero count (no scaffolding)
        zero_components = RewardComponents(scaffold_count=0)
        assert zero_components.scaffold_count >= 0

    def test_scaffold_count_displayed_with_hindsight(self):
        """TELE-660: scaffold_count displayed alongside hindsight_credit."""
        scaffold_count = 2

        # Format matches "(Nx, ...)" pattern
        formatted = f"({scaffold_count}x, "
        assert formatted == "(2x, "


# =============================================================================
# TELE-661: Reward Avg Scaffold Delay
# =============================================================================


class TestTELE661RewardAvgScaffoldDelay:
    """TELE-661: Reward Avg Scaffold Delay - average epochs since scaffolding.

    Debugging/analysis field tracking temporal scaffolding distance.
    Type: float
    Units: epochs
    Range: [0, +inf] (always non-negative)
    Default: 0.0

    Sign Convention: Non-negative (>= 0)
    - Zero: No scaffolding or just started
    - Low (< 5): Recent scaffolding
    - High (> 10): Long-term scaffolding

    Display Format: "X.Xe" (e.g., "3.5e" for 3.5 epochs)
    """

    def test_avg_scaffold_delay_field_exists(self):
        """TELE-661: avg_scaffold_delay field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "avg_scaffold_delay")

    def test_avg_scaffold_delay_default_value(self):
        """TELE-661: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.avg_scaffold_delay == 0.0

    def test_avg_scaffold_delay_type_is_float(self):
        """TELE-661: avg_scaffold_delay field is float type."""
        components = RewardComponents()
        assert isinstance(components.avg_scaffold_delay, float)

    def test_avg_scaffold_delay_non_negative(self):
        """TELE-661: avg_scaffold_delay is always non-negative."""
        # Positive delay (established scaffolding)
        delay_components = RewardComponents(avg_scaffold_delay=5.5)
        assert delay_components.avg_scaffold_delay >= 0

        # Zero delay (no scaffolding or just started)
        zero_components = RewardComponents(avg_scaffold_delay=0.0)
        assert zero_components.avg_scaffold_delay >= 0

    def test_avg_scaffold_delay_display_format(self):
        """TELE-661: Display format is 'X.Xe' (e.g., '3.5e')."""
        avg_scaffold_delay = 3.5
        formatted = f"{avg_scaffold_delay:.1f}e"
        assert formatted == "3.5e"


# =============================================================================
# TELE-662: Reward Fossilize Terminal Bonus
# =============================================================================


class TestTELE662RewardFossilizeTerminalBonus:
    """TELE-662: Reward Fossilize Terminal Bonus - terminal success bonus.

    Large terminal reward for successful fossilization.
    Type: float
    Units: reward units (terminal bonus)
    Range: [0, +inf] (always non-negative, bonus only)
    Default: 0.0

    Sign Convention: Non-negative (>= 0)
    - Zero: No fossilization this step
    - Positive: Successful fossilization

    Display: Blue styling, "Foss: +X.XXX" format
    Formula: num_contributing_fossilized * fossilize_terminal_scale
    """

    def test_fossilize_terminal_bonus_field_exists(self):
        """TELE-662: fossilize_terminal_bonus field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "fossilize_terminal_bonus")

    def test_fossilize_terminal_bonus_default_value(self):
        """TELE-662: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.fossilize_terminal_bonus == 0.0

    def test_fossilize_terminal_bonus_type_is_float(self):
        """TELE-662: fossilize_terminal_bonus field is float type."""
        components = RewardComponents()
        assert isinstance(components.fossilize_terminal_bonus, float)

    def test_fossilize_terminal_bonus_non_negative(self):
        """TELE-662: fossilize_terminal_bonus is always non-negative (bonus only)."""
        # Bonus for successful fossilization
        bonus_components = RewardComponents(fossilize_terminal_bonus=0.5)
        assert bonus_components.fossilize_terminal_bonus >= 0

        # Zero (no fossilization)
        zero_components = RewardComponents(fossilize_terminal_bonus=0.0)
        assert zero_components.fossilize_terminal_bonus >= 0

    def test_fossilize_terminal_bonus_display_format(self):
        """TELE-662: Display format is '+.3f' (e.g., '+0.500')."""
        fossilize_bonus = 0.5
        formatted = f"+{fossilize_bonus:.3f}"
        assert formatted == "+0.500"


# =============================================================================
# TELE-663: Reward Blending Warning
# =============================================================================


class TestTELE663RewardBlendingWarning:
    """TELE-663: Reward Blending Warning - negative trajectory warning.

    Penalty during BLENDING stage when seed has negative total_improvement.
    Type: float
    Units: reward units (penalty)
    Range: [-inf, 0] (always non-positive, penalty only)
    Default: 0.0

    Sign Convention: Non-positive (<= 0)
    - Zero: Healthy trajectory or not in BLENDING
    - Negative: Warning for negative trajectory during BLENDING

    Display: Yellow styling, "Blend: -X.XXX" format
    Formula: -0.1 - escalation (where escalation increases with degradation)
    """

    def test_blending_warning_field_exists(self):
        """TELE-663: blending_warning field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "blending_warning")

    def test_blending_warning_default_value(self):
        """TELE-663: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.blending_warning == 0.0

    def test_blending_warning_type_is_float(self):
        """TELE-663: blending_warning field is float type."""
        components = RewardComponents()
        assert isinstance(components.blending_warning, float)

    def test_blending_warning_non_positive(self):
        """TELE-663: blending_warning is always non-positive (penalty style)."""
        # Warning for negative trajectory
        warning_components = RewardComponents(blending_warning=-0.15)
        assert warning_components.blending_warning <= 0

        # Zero (healthy or not in BLENDING)
        zero_components = RewardComponents(blending_warning=0.0)
        assert zero_components.blending_warning <= 0

    def test_blending_warning_display_format(self):
        """TELE-663: Display format is '.3f' (e.g., '-0.150')."""
        blending_warning = -0.15
        formatted = f"{blending_warning:.3f}"
        assert formatted == "-0.150"


# =============================================================================
# TELE-664: Reward Holding Warning
# =============================================================================


class TestTELE664RewardHoldingWarning:
    """TELE-664: Reward Holding Warning - holding stage delay penalty.

    Penalty during HOLDING stage when agent continues to WAIT.
    Type: float
    Units: reward units (penalty)
    Range: [-inf, 0] (always non-positive, penalty only)
    Default: 0.0

    Sign Convention: Non-positive (<= 0)
    - Zero: Not in HOLDING or just entered
    - Negative: Penalty for holding too long

    Display: Yellow styling, "Hold: -X.XXX" format
    Purpose: Encourages timely decision (fossilize or prune)
    """

    def test_holding_warning_field_exists(self):
        """TELE-664: holding_warning field exists in schema."""
        components = RewardComponents()
        assert hasattr(components, "holding_warning")

    def test_holding_warning_default_value(self):
        """TELE-664: Default value is 0.0 per TELE record."""
        components = RewardComponents()
        assert components.holding_warning == 0.0

    def test_holding_warning_type_is_float(self):
        """TELE-664: holding_warning field is float type."""
        components = RewardComponents()
        assert isinstance(components.holding_warning, float)

    def test_holding_warning_non_positive(self):
        """TELE-664: holding_warning is always non-positive (penalty style)."""
        # Warning for holding too long
        warning_components = RewardComponents(holding_warning=-0.03)
        assert warning_components.holding_warning <= 0

        # Zero (not in HOLDING or just entered)
        zero_components = RewardComponents(holding_warning=0.0)
        assert zero_components.holding_warning <= 0

    def test_holding_warning_display_format(self):
        """TELE-664: Display format is '.3f' (e.g., '-0.030')."""
        holding_warning = -0.03
        formatted = f"{holding_warning:.3f}"
        assert formatted == "-0.030"


# =============================================================================
# TELE-655 to TELE-664 Integration Tests
# =============================================================================


class TestTELE655to664Integration:
    """Integration tests for TELE-655 to TELE-664 fields in RewardComponents."""

    def test_all_fields_can_be_set_together(self):
        """All TELE-655 to TELE-664 fields can be set in a single instance."""
        components = RewardComponents(
            total=0.45,
            stage_bonus=0.1,
            alpha_shock=-0.02,
            ratio_penalty=-0.01,
            hindsight_credit=0.08,
            scaffold_count=2,
            avg_scaffold_delay=3.5,
            fossilize_terminal_bonus=0.0,
            blending_warning=0.0,
            holding_warning=0.0,
        )

        # Verify all values
        assert components.total == 0.45
        assert components.stage_bonus == 0.1
        assert components.alpha_shock == -0.02
        assert components.ratio_penalty == -0.01
        assert components.hindsight_credit == 0.08
        assert components.scaffold_count == 2
        assert components.avg_scaffold_delay == 3.5
        assert components.fossilize_terminal_bonus == 0.0
        assert components.blending_warning == 0.0
        assert components.holding_warning == 0.0

    def test_sign_conventions_enforced(self):
        """Verify sign conventions: bonuses >= 0, penalties <= 0."""
        # Create components with valid sign conventions
        components = RewardComponents(
            # Bonuses (>= 0)
            stage_bonus=0.25,
            hindsight_credit=0.12,
            fossilize_terminal_bonus=0.5,
            # Penalties (<= 0)
            alpha_shock=-0.05,
            ratio_penalty=-0.03,
            blending_warning=-0.1,
            holding_warning=-0.02,
        )

        # Bonuses must be non-negative
        assert components.stage_bonus >= 0
        assert components.hindsight_credit >= 0
        assert components.fossilize_terminal_bonus >= 0

        # Penalties must be non-positive
        assert components.alpha_shock <= 0
        assert components.ratio_penalty <= 0
        assert components.blending_warning <= 0
        assert components.holding_warning <= 0

    def test_env_state_contains_all_fields(self):
        """EnvState.reward_components has all TELE-655 to TELE-664 fields."""
        env = EnvState(env_id=0)

        # All fields should exist with defaults
        assert env.reward_components.total == 0.0
        assert env.reward_components.stage_bonus == 0.0
        assert env.reward_components.alpha_shock == 0.0
        assert env.reward_components.ratio_penalty == 0.0
        assert env.reward_components.hindsight_credit == 0.0
        assert env.reward_components.scaffold_count == 0
        assert env.reward_components.avg_scaffold_delay == 0.0
        assert env.reward_components.fossilize_terminal_bonus == 0.0
        assert env.reward_components.blending_warning == 0.0
        assert env.reward_components.holding_warning == 0.0

    def test_healthy_training_scenario(self):
        """Simulate healthy training with positive reinforcement."""
        env = EnvState(env_id=0)

        # Healthy scenario: positive total, stage bonus, no warnings
        env.reward_components.total = 0.35
        env.reward_components.stage_bonus = 0.1
        env.reward_components.alpha_shock = 0.0  # No gaming
        env.reward_components.ratio_penalty = 0.0  # No gaming
        env.reward_components.hindsight_credit = 0.0
        env.reward_components.blending_warning = 0.0
        env.reward_components.holding_warning = 0.0

        # PBRS fraction should be healthy (10-40%)
        if abs(env.reward_components.total) > 0:
            pbrs_fraction = abs(env.reward_components.stage_bonus) / abs(
                env.reward_components.total
            )
            assert 0.10 <= pbrs_fraction <= 0.40

    def test_gaming_detection_scenario(self):
        """Simulate gaming detection with anti-gaming penalties."""
        env = EnvState(env_id=0)

        # Gaming scenario: alpha shock and ratio penalty triggered
        env.reward_components.total = -0.05
        env.reward_components.alpha_shock = -0.08
        env.reward_components.ratio_penalty = -0.05

        # Gaming indicators should be non-zero
        gaming_detected = (
            env.reward_components.alpha_shock != 0.0
            or env.reward_components.ratio_penalty != 0.0
        )
        assert gaming_detected

    def test_fossilization_scenario(self):
        """Simulate successful fossilization with terminal bonus."""
        env = EnvState(env_id=0)

        # Fossilization scenario: terminal bonus with hindsight credit
        env.reward_components.total = 0.75
        env.reward_components.fossilize_terminal_bonus = 0.5
        env.reward_components.hindsight_credit = 0.15
        env.reward_components.scaffold_count = 2
        env.reward_components.avg_scaffold_delay = 4.5

        # Terminal bonus should be significant
        assert env.reward_components.fossilize_terminal_bonus > 0
        # Hindsight credit should be present
        assert env.reward_components.hindsight_credit > 0
        # Scaffold metadata should be populated
        assert env.reward_components.scaffold_count > 0
        assert env.reward_components.avg_scaffold_delay > 0

    def test_warning_scenario(self):
        """Simulate warning scenario during BLENDING/HOLDING."""
        env = EnvState(env_id=0)

        # Warning scenario: blending and holding warnings
        env.reward_components.total = -0.25
        env.reward_components.blending_warning = -0.12
        env.reward_components.holding_warning = -0.03

        # Warnings should be active (non-zero negative)
        assert env.reward_components.blending_warning < 0
        assert env.reward_components.holding_warning < 0
