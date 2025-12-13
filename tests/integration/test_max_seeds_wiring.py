"""Integration tests for max_seeds wiring through training pipeline."""
import pytest
import torch

from esper.simic.ppo import signals_to_features
from esper.simic.action_masks import compute_action_masks
from esper.simic.features import MULTISLOT_FEATURE_SIZE
from esper.leyline.factored_actions import LifecycleOp


class TestMaxSeedsWiring:
    """Test max_seeds flows correctly through the pipeline."""

    def test_seed_utilization_in_features(self):
        """Verify seed_utilization appears in feature vector."""
        class MockMetrics:
            epoch = 10
            global_step = 100
            train_loss = 0.5
            val_loss = 0.6
            loss_delta = -0.1
            train_accuracy = 85.0
            val_accuracy = 82.0
            accuracy_delta = 0.5
            plateau_epochs = 2
            best_val_accuracy = 83.0
            best_val_loss = 0.55
            grad_norm_host = 1.0

        class MockSignals:
            metrics = MockMetrics()
            loss_history = [0.8, 0.7, 0.6, 0.5, 0.5]
            accuracy_history = [70.0, 75.0, 80.0, 82.0, 85.0]
            active_seeds = []
            available_slots = 3
            seed_stage = 0
            seed_epochs_in_stage = 0
            seed_alpha = 0.0
            seed_improvement = 0.0
            seed_counterfactual = 0.0

        # With 1 seed out of 3 max -> utilization = 0.333...
        features = signals_to_features(
            signals=MockSignals(),
            model=None,
            use_telemetry=False,
            slots=["mid"],
            total_seeds=1,
            max_seeds=3,
        )

        # Feature index 22 is seed_utilization per obs_to_multislot_features layout
        assert len(features) == MULTISLOT_FEATURE_SIZE
        assert abs(features[22] - (1/3)) < 0.01, f"Expected ~0.333, got {features[22]}"

    def test_germinate_masked_at_limit(self):
        """Verify GERMINATE is masked when at seed limit."""
        # No active seed (empty slot)
        slot_states = {"mid": None}

        # At limit: 3 seeds out of 3 max
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["mid"],
            total_seeds=3,
            max_seeds=3,
        )

        # GERMINATE op should be masked (False)
        assert masks["op"][LifecycleOp.GERMINATE].item() is False, "GERMINATE should be masked at seed limit"

    def test_germinate_allowed_under_limit(self):
        """Verify GERMINATE is allowed when under seed limit."""
        # No active seed (empty slot)
        slot_states = {"mid": None}

        # Under limit: 2 seeds out of 3 max
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["mid"],
            total_seeds=2,
            max_seeds=3,
        )

        # GERMINATE op should be allowed (True)
        assert masks["op"][LifecycleOp.GERMINATE].item() is True, "GERMINATE should be allowed under limit"

    def test_unlimited_seeds_when_max_zero(self):
        """Verify max_seeds=0 means unlimited."""
        # No active seed (empty slot)
        slot_states = {"mid": None}

        # max_seeds=0 means unlimited
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["mid"],
            total_seeds=100,  # Many seeds
            max_seeds=0,      # Unlimited
        )

        # GERMINATE should still be allowed (slot is empty)
        assert masks["op"][LifecycleOp.GERMINATE].item() is True, "GERMINATE should be allowed with max_seeds=0"
