"""Integration tests for max_seeds wiring through training pipeline."""

from esper.tamiyo.policy.action_masks import compute_action_masks
from esper.leyline import LifecycleOp


class TestMaxSeedsWiring:
    """Test max_seeds flows correctly through the pipeline."""

    # NOTE: The test_seed_utilization_in_features test was removed because
    # seed_utilization is no longer exposed as a feature in Obs V3.
    # Seed limit enforcement is now handled via compute_action_masks,
    # which is tested below.

    def test_germinate_masked_at_limit(self):
        """Verify GERMINATE is masked when at seed limit."""
        # No active seed (empty slot)
        slot_states = {"r0c1": None}

        # At limit: 3 seeds out of 3 max
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["r0c1"],
            total_seeds=3,
            max_seeds=3,
        )

        # GERMINATE op should be masked (False)
        assert masks["op"][LifecycleOp.GERMINATE].item() is False, "GERMINATE should be masked at seed limit"

    def test_germinate_allowed_under_limit(self):
        """Verify GERMINATE is allowed when under seed limit."""
        # No active seed (empty slot)
        slot_states = {"r0c1": None}

        # Under limit: 2 seeds out of 3 max
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["r0c1"],
            total_seeds=2,
            max_seeds=3,
        )

        # GERMINATE op should be allowed (True)
        assert masks["op"][LifecycleOp.GERMINATE].item() is True, "GERMINATE should be allowed under limit"

    def test_unlimited_seeds_when_max_zero(self):
        """Verify max_seeds=0 means unlimited."""
        # No active seed (empty slot)
        slot_states = {"r0c1": None}

        # max_seeds=0 means unlimited
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["r0c1"],
            total_seeds=100,  # Many seeds
            max_seeds=0,      # Unlimited
        )

        # GERMINATE should still be allowed (slot is empty)
        assert masks["op"][LifecycleOp.GERMINATE].item() is True, "GERMINATE should be allowed with max_seeds=0"
