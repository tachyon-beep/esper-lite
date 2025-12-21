"""Multi-seed scenario tests for Kasmina.

Tests verify correct behavior with multiple seeds:
- Sequential seeds in same slot (germinate → cull → germinate)
- Concurrent seeds in different slots
- Gradient independence between seeds
- Alpha independence
- Multi-seed forward order
- Cull one keeps other
- Fossilize one affects active count
- All stages simultaneously across slots
"""

import pytest
import torch
import torch.nn as nn

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel
from esper.kasmina.slot import SeedSlot, SeedState, SeedMetrics
from esper.leyline import SeedStage, DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE


class TestSequentialSeedsInSameSlot:
    """Tests for sequential seeds in the same slot."""

    def test_germinate_cull_germinate_same_slot(self):
        """Second seed should work correctly after first is culled."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # First seed lifecycle
        state1 = slot.germinate("noop", seed_id="seed1")
        assert state1.seed_id == "seed1"
        slot.state.transition(SeedStage.TRAINING)

        # Cull first seed
        slot.prune()
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

        # Cooldown must complete before slot is available again.
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            slot.step_epoch()
        assert slot.state is None

        # Second seed should work correctly
        state2 = slot.germinate("noop", seed_id="seed2")
        assert state2.seed_id == "seed2"
        assert state2.stage == SeedStage.GERMINATED

        # Verify fresh metrics
        assert state2.metrics.epochs_total == 0
        assert state2.metrics.best_val_accuracy == 0.0

    def test_recycled_slot_no_contamination(self):
        """Metrics and state should not carry over between seeds."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # First seed accumulates metrics
        slot.germinate("noop", seed_id="seed1")
        slot.state.metrics.record_accuracy(75.0)
        slot.state.metrics.record_accuracy(80.0)
        slot.state.metrics.seed_gradient_norm_ratio = 0.5
        slot.state.transition(SeedStage.TRAINING)

        # Cull and create new seed
        slot.prune()
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            slot.step_epoch()
        assert slot.state is None
        slot.germinate("noop", seed_id="seed2")

        # Second seed should have fresh state
        assert slot.state.metrics.epochs_total == 0
        assert slot.state.metrics.best_val_accuracy == 0.0
        assert slot.state.metrics.seed_gradient_norm_ratio == 0.0

    def test_multiple_germinate_cull_cycles(self):
        """Slot should handle many germinate/cull cycles without issues."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        for i in range(10):
            state = slot.germinate("noop", seed_id=f"seed_{i}")
            assert state.seed_id == f"seed_{i}"
            assert state.stage == SeedStage.GERMINATED

            # Progress through some states
            slot.state.transition(SeedStage.TRAINING)
            slot.state.metrics.record_accuracy(50.0 + i)

            # Cull
            slot.prune()
            for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
                slot.step_epoch()
            assert slot.state is None

    def test_different_blueprints_same_slot(self):
        """Same slot should accept different blueprints after cull."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        blueprints = ["noop", "norm", "depthwise"]

        for bp in blueprints:
            state = slot.germinate(bp, seed_id=f"seed_{bp}")
            assert state.blueprint_id == bp

            # Forward pass should work
            x = torch.randn(2, 64, 8, 8)
            output = slot(x)
            assert output.shape == x.shape

            slot.state.transition(SeedStage.TRAINING)
            slot.prune()
            for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
                slot.step_epoch()
            assert slot.state is None


class TestConcurrentSeedsDifferentSlots:
    """Tests for concurrent seeds in different slots."""

    def test_two_seeds_different_slots_independent(self):
        """Seeds in different slots should have independent lifecycles."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Germinate seeds in two slots
        model.germinate_seed("noop", "seed_slot0", slot="r0c0")
        model.germinate_seed("norm", "seed_slot1", slot="r0c1")

        # Verify independent states
        slot0 = model.seed_slots["r0c0"]
        slot1 = model.seed_slots["r0c1"]

        assert slot0.state.seed_id == "seed_slot0"
        assert slot0.state.blueprint_id == "noop"
        assert slot1.state.seed_id == "seed_slot1"
        assert slot1.state.blueprint_id == "norm"

    def test_three_seeds_all_active(self):
        """Three concurrent seeds should all be active."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("norm", "seed1", slot="r0c1")
        model.germinate_seed("depthwise", "seed2", slot="r0c2")

        assert model.has_active_seed
        assert model.count_active_seeds() == 3

    def test_stage_independence_different_slots(self):
        """Each slot can be in a different stage."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Germinate all three
        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        # Progress them to different stages
        slot0 = model.seed_slots["r0c0"]
        slot1 = model.seed_slots["r0c1"]
        slot2 = model.seed_slots["r0c2"]

        # Keep slot0 at GERMINATED
        # Advance slot1 to TRAINING
        slot1.state.transition(SeedStage.TRAINING)
        # Advance slot2 to BLENDING
        slot2.state.transition(SeedStage.TRAINING)
        slot2.state.transition(SeedStage.BLENDING)

        assert slot0.state.stage == SeedStage.GERMINATED
        assert slot1.state.stage == SeedStage.TRAINING
        assert slot2.state.stage == SeedStage.BLENDING


class TestGradientIndependence:
    """Tests for gradient independence between seeds."""

    def test_seed_gradients_independent(self):
        """Gradients for each seed should be independent."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("norm", "seed0", slot="r0c0")
        model.germinate_seed("norm", "seed1", slot="r0c1")

        # Progress to training
        model.seed_slots["r0c0"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)

        # Set alpha > 0 for blending
        model.seed_slots["r0c0"].set_alpha(0.5)
        model.seed_slots["r0c1"].set_alpha(0.5)

        # Progress to BLENDING for gradient flow
        model.seed_slots["r0c0"].state.transition(SeedStage.BLENDING)
        model.seed_slots["r0c1"].state.transition(SeedStage.BLENDING)

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Each seed should have its own gradients
        seed0_grads = []
        seed1_grads = []

        for param in model.seed_slots["r0c0"].get_parameters():
            if param.grad is not None:
                seed0_grads.append(param.grad.clone())

        for param in model.seed_slots["r0c1"].get_parameters():
            if param.grad is not None:
                seed1_grads.append(param.grad.clone())

        # Both should have gradients (norm blueprint has params)
        assert len(seed0_grads) > 0
        assert len(seed1_grads) > 0

    def test_cull_one_preserves_other_gradients(self):
        """Culling one seed should not affect gradients of another."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("norm", "seed0", slot="r0c0")
        model.germinate_seed("norm", "seed1", slot="r0c1")

        # Transition to TRAINING
        model.seed_slots["r0c0"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)

        # Cull seed0
        model.prune_seed(slot="r0c0")

        # Verify seed1 still works
        model.seed_slots["r0c1"].state.transition(SeedStage.BLENDING)
        model.seed_slots["r0c1"].set_alpha(0.5)

        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Seed1 should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.seed_slots["r0c1"].get_parameters()
        )
        assert has_grad


class TestAlphaIndependence:
    """Tests for alpha independence between slots."""

    def test_different_alphas_per_slot(self):
        """Each slot should maintain its own alpha."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Germinate all
        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        # Set different alphas
        model.seed_slots["r0c0"].set_alpha(0.0)
        model.seed_slots["r0c1"].set_alpha(0.5)
        model.seed_slots["r0c2"].set_alpha(1.0)

        assert model.seed_slots["r0c0"].state.alpha == 0.0
        assert model.seed_slots["r0c1"].state.alpha == 0.5
        assert model.seed_slots["r0c2"].state.alpha == 1.0

    def test_alpha_change_does_not_affect_other_slots(self):
        """Changing alpha in one slot should not affect others."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")

        # Set initial alphas
        model.seed_slots["r0c0"].set_alpha(0.3)
        model.seed_slots["r0c1"].set_alpha(0.7)

        # Change slot0 alpha
        model.seed_slots["r0c0"].set_alpha(0.9)

        # Slot1 should be unchanged
        assert model.seed_slots["r0c0"].state.alpha == 0.9
        assert model.seed_slots["r0c1"].state.alpha == 0.7


class TestMultiSeedForwardOrder:
    """Tests for correct forward pass order with multiple seeds."""

    def test_seeds_applied_in_injection_order(self):
        """Seeds should be applied in network injection order."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Track forward pass order via a custom module
        call_order = []

        class OrderTracker(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def forward(self, x):
                call_order.append(self.name)
                return x

        # Override slots with order tracking
        model.seed_slots["r0c0"].seed = OrderTracker("r0c0")
        model.seed_slots["r0c1"].seed = OrderTracker("r0c1")
        model.seed_slots["r0c2"].seed = OrderTracker("r0c2")

        # Create dummy states (use state, not _state)
        model.seed_slots["r0c0"].state = SeedState(
            seed_id="s0", blueprint_id="noop", stage=SeedStage.TRAINING
        )
        model.seed_slots["r0c1"].state = SeedState(
            seed_id="s1", blueprint_id="noop", stage=SeedStage.TRAINING
        )
        model.seed_slots["r0c2"].state = SeedState(
            seed_id="s2", blueprint_id="noop", stage=SeedStage.TRAINING
        )

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)

        # Verify order
        assert call_order == ["r0c0", "r0c1", "r0c2"]


class TestCullOneKeepsOther:
    """Tests for culling one seed while keeping others."""

    def test_cull_first_keeps_others(self):
        """Culling first slot should not affect later slots."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        # Transition all to TRAINING
        for slot in model.seed_slots.values():
            slot.state.transition(SeedStage.TRAINING)

        # Cull first
        model.prune_seed(slot="r0c0")

        # Others should be unaffected
        assert model.seed_slots["r0c0"].state is not None
        assert model.seed_slots["r0c0"].state.stage == SeedStage.PRUNED
        assert model.seed_slots["r0c0"].seed is None
        assert model.seed_slots["r0c1"].state.seed_id == "seed1"
        assert model.seed_slots["r0c2"].state.seed_id == "seed2"
        assert model.count_active_seeds() == 2

    def test_cull_middle_keeps_first_and_last(self):
        """Culling middle slot should not affect first or last."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        for slot in model.seed_slots.values():
            slot.state.transition(SeedStage.TRAINING)

        model.prune_seed(slot="r0c1")

        assert model.seed_slots["r0c0"].state.seed_id == "seed0"
        assert model.seed_slots["r0c1"].state is not None
        assert model.seed_slots["r0c1"].state.stage == SeedStage.PRUNED
        assert model.seed_slots["r0c1"].seed is None
        assert model.seed_slots["r0c2"].state.seed_id == "seed2"

    def test_cull_all_sequentially(self):
        """Culling all seeds sequentially should work."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        for slot in model.seed_slots.values():
            slot.state.transition(SeedStage.TRAINING)

        model.prune_seed(slot="r0c0")
        assert model.count_active_seeds() == 2

        model.prune_seed(slot="r0c1")
        assert model.count_active_seeds() == 1

        model.prune_seed(slot="r0c2")
        assert model.count_active_seeds() == 0
        assert not model.has_active_seed


class TestFossilizeAffectsActiveCount:
    """Tests for fossilization affecting active seed count."""

    def test_fossilize_decrements_active_count(self):
        """Fossilizing a seed should decrement active count."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")

        for slot in model.seed_slots.values():
            slot.state.transition(SeedStage.TRAINING)

        assert model.count_active_seeds() == 2

        # Force one to FOSSILIZED
        model.seed_slots["r0c0"].state.stage = SeedStage.FOSSILIZED

        # Active count should decrement (fossilized seeds are not counted as active)
        assert model.count_active_seeds() == 1
        assert model.count_fossilized_seeds() == 1
        assert model.total_seeds() == 2

    def test_fossilized_still_participates_in_forward(self):
        """Fossilized seed should still apply its transformation in forward."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.seed_slots["r0c0"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c0"].state.stage = SeedStage.FOSSILIZED
        model.seed_slots["r0c0"].set_alpha(1.0)

        # Forward should still work
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        assert output.shape == (2, 10)


class TestAllStagesSimultaneously:
    """Tests for seeds at different stages simultaneously."""

    def test_three_slots_three_stages(self):
        """Three slots can be at GERMINATED, TRAINING, BLENDING simultaneously."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        # r0c0: GERMINATED (no transition)
        # r0c1: TRAINING
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)
        # r0c2: BLENDING
        model.seed_slots["r0c2"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c2"].state.transition(SeedStage.BLENDING)

        assert model.seed_slots["r0c0"].state.stage == SeedStage.GERMINATED
        assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING
        assert model.seed_slots["r0c2"].state.stage == SeedStage.BLENDING

        # Forward should work
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_mixed_fossilized_and_training(self):
        """One fossilized and one training seed should coexist."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")

        # Make r0c0 fossilized
        model.seed_slots["r0c0"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c0"].state.stage = SeedStage.FOSSILIZED
        model.seed_slots["r0c0"].set_alpha(1.0)

        # Keep r0c1 in training
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)

        assert model.seed_slots["r0c0"].state.stage == SeedStage.FOSSILIZED
        assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)


class TestMetricsIndependence:
    """Tests for metrics independence between slots."""

    def test_metrics_accumulate_independently(self):
        """Metrics should accumulate independently per slot."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")

        # Record different accuracies
        model.seed_slots["r0c0"].state.metrics.record_accuracy(50.0)
        model.seed_slots["r0c0"].state.metrics.record_accuracy(60.0)

        model.seed_slots["r0c1"].state.metrics.record_accuracy(70.0)
        model.seed_slots["r0c1"].state.metrics.record_accuracy(80.0)
        model.seed_slots["r0c1"].state.metrics.record_accuracy(90.0)

        assert model.seed_slots["r0c0"].state.metrics.epochs_total == 2
        assert model.seed_slots["r0c0"].state.metrics.best_val_accuracy == 60.0

        assert model.seed_slots["r0c1"].state.metrics.epochs_total == 3
        assert model.seed_slots["r0c1"].state.metrics.best_val_accuracy == 90.0

    def test_gradient_ratio_independent_per_slot(self):
        """Gradient ratio should be tracked independently per slot."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")

        model.seed_slots["r0c0"].state.metrics.seed_gradient_norm_ratio = 0.3
        model.seed_slots["r0c1"].state.metrics.seed_gradient_norm_ratio = 0.8

        assert model.seed_slots["r0c0"].state.metrics.seed_gradient_norm_ratio == 0.3
        assert model.seed_slots["r0c1"].state.metrics.seed_gradient_norm_ratio == 0.8


class TestTransformerMultiSeed:
    """Tests for multi-seed scenarios with TransformerHost."""

    def test_transformer_two_seeds(self):
        """TransformerHost should support multiple concurrent seeds."""
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=6,
            n_head=2,
            block_size=32,
            num_segments=3,
        )
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")

        assert model.count_active_seeds() == 2

        x = torch.randint(0, 1000, (2, 8))
        output = model(x)

        assert output.shape == (2, 8, 1000)

    def test_transformer_mixed_stages(self):
        """TransformerHost with seeds at different stages."""
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=6,
            n_head=2,
            block_size=32,
            num_segments=3,
        )
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("noop", "seed0", slot="r0c0")
        model.germinate_seed("noop", "seed1", slot="r0c1")
        model.germinate_seed("noop", "seed2", slot="r0c2")

        # Different stages
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c2"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c2"].state.transition(SeedStage.BLENDING)

        x = torch.randint(0, 1000, (2, 8))
        output = model(x)

        assert output.shape == (2, 8, 1000)


class TestEmptyAndPartialSlots:
    """Tests for scenarios with empty or partially filled slots."""

    def test_one_slot_active_two_empty(self):
        """Model should work with one active slot and two empty."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Only germinate in r0c1
        model.germinate_seed("noop", "seed1", slot="r0c1")

        assert model.seed_slots["r0c0"].state is None
        assert model.seed_slots["r0c1"].state is not None
        assert model.seed_slots["r0c2"].state is None

        assert model.count_active_seeds() == 1

        # Forward should work
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_no_active_seeds_uses_host_directly(self):
        """With no active seeds, model should use host forward directly."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # No seeds germinated
        assert model.count_active_seeds() == 0
        assert not model.has_active_seed

        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_germinate_middle_slot_only(self):
        """Germinating only the middle slot should work correctly."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        model.germinate_seed("norm", "middle_seed", slot="r0c1")

        assert model.seed_slots["r0c0"].state is None
        assert model.seed_slots["r0c1"].state.seed_id == "middle_seed"
        assert model.seed_slots["r0c2"].state is None

        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
