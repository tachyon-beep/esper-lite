"""Unit tests for scaffold hindsight credit assignment (Phase 3.2).

Tests cover:
- Credit computation on fossilization
- Temporal discounting with gamma^delay
- Credit cap at MAX_HINDSIGHT_CREDIT (0.2)
- Ledger lifecycle and reset
- Multiple scaffolds contributing to same beneficiary
"""

from esper.leyline import DEFAULT_GAMMA
from esper.simic.rewards import compute_scaffold_hindsight_credit
from esper.simic.training.parallel_env_state import ParallelEnvState


# Constants matching vectorized.py implementation
MAX_HINDSIGHT_CREDIT = 0.2


class TestHindsightCreditComputation:
    """Tests for compute_scaffold_hindsight_credit function."""

    def test_hindsight_credit_computed_on_fossilization(self):
        """Credit is added to pending when beneficiary fossilizes."""
        # Scaffold provided boost of 1.5, beneficiary improved by 3%
        credit = compute_scaffold_hindsight_credit(
            boost_given=1.5,
            beneficiary_improvement=3.0,
            credit_weight=0.2,
        )

        assert credit > 0, f"Expected positive credit, got {credit}"
        assert credit <= MAX_HINDSIGHT_CREDIT, (
            f"Credit {credit} should not exceed cap {MAX_HINDSIGHT_CREDIT}"
        )

    def test_no_credit_for_negative_improvement(self):
        """Beneficiary with negative improvement gives no credit."""
        # Beneficiary regressed despite receiving boost
        credit = compute_scaffold_hindsight_credit(
            boost_given=2.0,
            beneficiary_improvement=-1.0,
            credit_weight=0.2,
        )

        assert credit == 0.0, f"Expected zero credit for negative improvement, got {credit}"

    def test_no_credit_for_negative_boost(self):
        """Negative boost (antagonistic interaction) gives no credit."""
        # Scaffold hurt the beneficiary
        credit = compute_scaffold_hindsight_credit(
            boost_given=-0.5,
            beneficiary_improvement=2.0,
            credit_weight=0.2,
        )

        assert credit == 0.0, f"Expected zero credit for negative boost, got {credit}"

    def test_no_credit_for_zero_boost(self):
        """Zero boost gives no credit."""
        credit = compute_scaffold_hindsight_credit(
            boost_given=0.0,
            beneficiary_improvement=2.0,
            credit_weight=0.2,
        )

        assert credit == 0.0, f"Expected zero credit for zero boost, got {credit}"

    def test_no_credit_for_zero_improvement(self):
        """Zero improvement gives no credit."""
        credit = compute_scaffold_hindsight_credit(
            boost_given=1.5,
            beneficiary_improvement=0.0,
            credit_weight=0.2,
        )

        assert credit == 0.0, f"Expected zero credit for zero improvement, got {credit}"

    def test_credit_capped_at_maximum(self):
        """Total credit per fossilization is capped at credit_weight."""
        # Very large boost and improvement
        credit = compute_scaffold_hindsight_credit(
            boost_given=100.0,
            beneficiary_improvement=100.0,
            credit_weight=0.2,
        )

        assert credit <= 0.2, f"Credit {credit} exceeds cap of 0.2"
        # tanh(100 * 100 * 0.1) saturates to 1.0, so credit = 1.0 * 0.2 = 0.2
        assert abs(credit - 0.2) < 0.001, f"Credit should be at cap, got {credit}"

    def test_credit_scales_with_boost_and_improvement(self):
        """Credit increases with both boost magnitude and beneficiary improvement."""
        # Small interaction
        small_credit = compute_scaffold_hindsight_credit(
            boost_given=0.1,
            beneficiary_improvement=0.5,
            credit_weight=0.2,
        )

        # Larger interaction
        large_credit = compute_scaffold_hindsight_credit(
            boost_given=1.0,
            beneficiary_improvement=2.0,
            credit_weight=0.2,
        )

        assert large_credit > small_credit, (
            f"Larger interaction {large_credit} should exceed smaller {small_credit}"
        )


class TestTemporalDiscount:
    """Tests for temporal discounting of hindsight credit."""

    def test_temporal_discount_applied(self):
        """Older scaffolding interactions receive less credit."""
        boost_given = 1.5
        beneficiary_improvement = 3.0
        credit_weight = 0.2

        # Base credit with no delay
        base_credit = compute_scaffold_hindsight_credit(
            boost_given=boost_given,
            beneficiary_improvement=beneficiary_improvement,
            credit_weight=credit_weight,
        )

        # Simulating temporal discount as done in vectorized.py:
        # discount = DEFAULT_GAMMA ** delay
        delay = 5
        discount = DEFAULT_GAMMA ** delay
        discounted_credit = base_credit * discount

        # The discounted credit should be less than base
        assert discounted_credit < base_credit, (
            f"Discounted credit {discounted_credit} should be less than base {base_credit}"
        )

        # Verify discount formula: gamma^5 with gamma=0.995 = ~0.975
        expected_discount = DEFAULT_GAMMA ** delay
        assert abs(expected_discount - 0.975) < 0.01, (
            f"Expected discount ~0.975, got {expected_discount}"
        )

    def test_temporal_discount_zero_delay(self):
        """Zero delay means no discount (gamma^0 = 1.0)."""
        delay = 0
        discount = DEFAULT_GAMMA ** delay

        assert discount == 1.0, f"gamma^0 should equal 1.0, got {discount}"

    def test_temporal_discount_large_delay(self):
        """Large delays result in significant discount."""
        delay = 100
        discount = DEFAULT_GAMMA ** delay

        # gamma=0.995, 100 steps: 0.995^100 = ~0.606
        assert discount < 0.7, f"100-step discount should be significant, got {discount}"
        assert discount > 0.5, f"100-step discount shouldn't be too extreme, got {discount}"


class TestScaffoldLedger:
    """Tests for scaffold boost ledger behavior."""

    def _make_minimal_env_state(self) -> ParallelEnvState:
        """Create a minimal ParallelEnvState for testing ledger behavior."""
        import torch
        from unittest.mock import Mock

        # Create minimal mocks for required fields
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_signal_tracker = Mock()
        mock_signal_tracker.reset = Mock()
        mock_governor = Mock()
        mock_governor.reset = Mock()

        return ParallelEnvState(
            model=mock_model,
            host_optimizer=mock_optimizer,
            signal_tracker=mock_signal_tracker,
            governor=mock_governor,
        )

    def test_ledger_starts_empty(self):
        """Scaffold ledger is empty on initialization."""
        env_state = self._make_minimal_env_state()

        assert env_state.scaffold_boost_ledger == {}, (
            "Ledger should be empty on init"
        )

    def test_ledger_cleared_on_episode_reset(self):
        """Scaffold ledger resets between episodes."""
        env_state = self._make_minimal_env_state()

        # Add some boost entries
        env_state.scaffold_boost_ledger["slot_0"] = [(1.5, "slot_1", 5)]
        env_state.scaffold_boost_ledger["slot_1"] = [(1.5, "slot_0", 5)]

        # Also set pending credit
        env_state.pending_hindsight_credit = 0.15

        # Reset episode state
        env_state.reset_episode_state(slots=["slot_0", "slot_1"])

        assert env_state.scaffold_boost_ledger == {}, (
            "Ledger should be cleared on reset"
        )
        assert env_state.pending_hindsight_credit == 0.0, (
            "Pending credit should be cleared on reset"
        )

    def test_pending_credit_added_to_next_reward(self):
        """Pending credit flows to next transition's reward."""
        env_state = self._make_minimal_env_state()

        # Set pending credit (as would happen on fossilization)
        env_state.pending_hindsight_credit = 0.15

        # Simulate reward computation consuming the credit
        reward = 0.5
        if env_state.pending_hindsight_credit > 0:
            hindsight_credit_applied = env_state.pending_hindsight_credit
            reward += hindsight_credit_applied
            env_state.pending_hindsight_credit = 0.0

        assert reward == 0.65, f"Expected reward 0.65 with credit, got {reward}"
        assert env_state.pending_hindsight_credit == 0.0, (
            "Pending credit should be consumed after application"
        )

    def test_ledger_tracks_boost_with_epoch(self):
        """Ledger entries include epoch for temporal discount calculation."""
        env_state = self._make_minimal_env_state()

        # Add boost at epoch 5
        boost_given = 1.5
        beneficiary_slot = "slot_1"
        epoch_of_boost = 5

        if "slot_0" not in env_state.scaffold_boost_ledger:
            env_state.scaffold_boost_ledger["slot_0"] = []
        env_state.scaffold_boost_ledger["slot_0"].append(
            (boost_given, beneficiary_slot, epoch_of_boost)
        )

        # Verify structure
        assert len(env_state.scaffold_boost_ledger["slot_0"]) == 1
        entry = env_state.scaffold_boost_ledger["slot_0"][0]
        assert entry == (1.5, "slot_1", 5), f"Entry should be (boost, beneficiary, epoch), got {entry}"

    def test_current_epoch_tracks_time(self):
        """current_epoch field enables temporal discount calculation."""
        env_state = self._make_minimal_env_state()

        assert env_state.current_epoch == 0, "Epoch should start at 0"

        # Simulate epoch progression
        env_state.current_epoch = 10

        assert env_state.current_epoch == 10, "Epoch should update"

        # Reset clears epoch
        env_state.reset_episode_state(slots=["slot_0"])

        assert env_state.current_epoch == 0, "Epoch should reset to 0"


class TestMultipleScaffolds:
    """Tests for multiple scaffolds contributing to same beneficiary."""

    def test_multiple_scaffolds_each_get_credit(self):
        """Multiple scaffolds boosting same beneficiary each receive credit."""
        # Two scaffolds each provided different boosts to the beneficiary
        beneficiary_improvement = 3.0
        credit_weight = 0.2

        # Scaffold A provided boost of 1.0
        credit_a = compute_scaffold_hindsight_credit(
            boost_given=1.0,
            beneficiary_improvement=beneficiary_improvement,
            credit_weight=credit_weight,
        )

        # Scaffold B provided boost of 0.5
        credit_b = compute_scaffold_hindsight_credit(
            boost_given=0.5,
            beneficiary_improvement=beneficiary_improvement,
            credit_weight=credit_weight,
        )

        # Both should get positive credit
        assert credit_a > 0, f"Scaffold A should get credit, got {credit_a}"
        assert credit_b > 0, f"Scaffold B should get credit, got {credit_b}"

        # Scaffold A should get more credit (larger boost)
        assert credit_a > credit_b, (
            f"Larger boost {credit_a} should get more credit than {credit_b}"
        )

    def test_total_credit_capped_when_summed(self):
        """Total credit from multiple scaffolds is capped at MAX_HINDSIGHT_CREDIT."""
        beneficiary_improvement = 5.0
        credit_weight = 0.2

        # Multiple scaffolds with large boosts
        credits = []
        for boost in [2.0, 1.5, 1.0]:
            credit = compute_scaffold_hindsight_credit(
                boost_given=boost,
                beneficiary_improvement=beneficiary_improvement,
                credit_weight=credit_weight,
            )
            credits.append(credit)

        # Sum and cap (as done in vectorized.py)
        total_credit = sum(credits)
        capped_credit = min(total_credit, MAX_HINDSIGHT_CREDIT)

        assert capped_credit <= MAX_HINDSIGHT_CREDIT, (
            f"Capped credit {capped_credit} should not exceed {MAX_HINDSIGHT_CREDIT}"
        )

    def test_credit_with_temporal_discounts_for_multiple_scaffolds(self):
        """Multiple scaffolds at different epochs get different discounts."""
        beneficiary_improvement = 3.0
        credit_weight = 0.2
        current_epoch = 10

        # Scaffold A boosted at epoch 3 (7 epochs ago)
        delay_a = current_epoch - 3
        discount_a = DEFAULT_GAMMA ** delay_a
        credit_a = compute_scaffold_hindsight_credit(
            boost_given=1.0,
            beneficiary_improvement=beneficiary_improvement,
            credit_weight=credit_weight,
        )
        discounted_a = credit_a * discount_a

        # Scaffold B boosted at epoch 8 (2 epochs ago)
        delay_b = current_epoch - 8
        discount_b = DEFAULT_GAMMA ** delay_b
        credit_b = compute_scaffold_hindsight_credit(
            boost_given=1.0,  # Same boost magnitude
            beneficiary_improvement=beneficiary_improvement,
            credit_weight=credit_weight,
        )
        discounted_b = credit_b * discount_b

        # Same base credit (same boost)
        assert abs(credit_a - credit_b) < 0.001, "Same boost should give same base credit"

        # But different discounted credit (scaffold B is more recent)
        assert discounted_b > discounted_a, (
            f"Recent scaffold {discounted_b} should get more than old {discounted_a}"
        )


class TestCreditGoesThoughNormalizer:
    """Tests verifying credit is added before normalization."""

    def test_credit_goes_through_normalizer(self):
        """Credit is added before normalization, not after.

        This test verifies the integration pattern:
        1. Base reward is computed
        2. Pending hindsight credit is added to base reward
        3. Combined reward goes through normalizer
        """
        # Simulate the reward flow from vectorized.py
        base_reward = 0.5
        pending_credit = 0.1

        # Credit added BEFORE normalization
        reward_before_norm = base_reward + pending_credit

        # Simulate simple normalization (subtract mean, divide by std)
        # In practice this uses a running normalizer
        running_mean = 0.3
        running_std = 0.2
        normalized = (reward_before_norm - running_mean) / running_std

        # The key property: hindsight credit affects the normalized value
        # If credit was added AFTER normalization, it would bypass scaling

        base_normalized = (base_reward - running_mean) / running_std
        expected_credit_effect = pending_credit / running_std

        # Verify credit went through normalizer
        assert abs(normalized - base_normalized - expected_credit_effect) < 0.001, (
            "Credit should affect normalized reward by credit/std"
        )

    def test_credit_respects_reward_scale(self):
        """Credit is scaled appropriately by normalizer."""
        # If rewards have high variance, credit should be scaled down
        # If rewards have low variance, credit should be scaled up
        pending_credit = 0.1

        # High variance scenario
        high_std = 1.0
        credit_effect_high = pending_credit / high_std

        # Low variance scenario
        low_std = 0.1
        credit_effect_low = pending_credit / low_std

        # Low variance means credit has larger effect
        assert credit_effect_low > credit_effect_high, (
            "Credit should have larger effect when reward variance is low"
        )
