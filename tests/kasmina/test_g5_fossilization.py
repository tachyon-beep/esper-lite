"""Tests for G5 fossilization gate minimum contribution threshold."""

from esper.kasmina.slot import (
    QualityGates,
    SeedState,
    SeedMetrics,
)
from esper.leyline import SeedStage, DEFAULT_MIN_FOSSILIZE_CONTRIBUTION


def create_probationary_state(contribution: float, healthy: bool = True) -> SeedState:
    """Create a SeedState in HOLDING stage with given contribution.

    SeedState is a dataclass with required fields seed_id and blueprint_id.
    SeedMetrics.counterfactual_contribution is a settable field.
    """
    metrics = SeedMetrics()
    metrics.counterfactual_contribution = contribution

    state = SeedState(
        seed_id="test-seed",
        blueprint_id="test-blueprint",
        stage=SeedStage.HOLDING,
        metrics=metrics,
        is_healthy=healthy,
    )
    return state


class TestG5MinimumContribution:
    """Test G5 gate enforces minimum contribution threshold."""

    def test_g5_rejects_below_threshold(self):
        """G5 gate should reject seeds with contribution below 1%."""
        gates = QualityGates()
        state = create_probationary_state(contribution=0.5)  # Below 1%

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_rejects_zero_contribution(self):
        """G5 gate should reject seeds with zero contribution."""
        gates = QualityGates()
        state = create_probationary_state(contribution=0.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_rejects_negative_contribution(self):
        """G5 gate should reject seeds with negative contribution."""
        gates = QualityGates()
        state = create_probationary_state(contribution=-5.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_accepts_at_threshold(self):
        """G5 gate should accept seeds with contribution exactly at 1%."""
        gates = QualityGates()
        state = create_probationary_state(contribution=1.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert result.passed
        assert any("sufficient_contribution" in c for c in result.checks_passed)

    def test_g5_accepts_above_threshold(self):
        """G5 gate should accept seeds with contribution above 1%."""
        gates = QualityGates()
        state = create_probationary_state(contribution=5.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert result.passed
        assert any("sufficient_contribution" in c for c in result.checks_passed)

    def test_min_fossilize_contribution_constant(self):
        """DEFAULT_MIN_FOSSILIZE_CONTRIBUTION should be 1.0%."""
        assert DEFAULT_MIN_FOSSILIZE_CONTRIBUTION == 1.0
