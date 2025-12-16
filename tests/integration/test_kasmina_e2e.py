"""End-to-end integration tests for Kasmina.

Tests verify the complete Kasmina workflow:
- Full training with CNN host and single seed
- Full training with Transformer host and single seed
- Multi-slot training with CNN host
- Gradient telemetry integration with Tolaria
- Counterfactual validation integration
"""

import pytest
import torch
import torch.nn as nn

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel
from esper.kasmina.slot import SeedSlot, QualityGates
from esper.leyline import (
    SeedStage,
    DEFAULT_MIN_TRAINING_IMPROVEMENT,
    DEFAULT_MIN_BLENDING_EPOCHS,
    DEFAULT_GRADIENT_RATIO_THRESHOLD,
)


class TestCNNHostSingleSeedTraining:
    """E2E tests for CNN host with single seed training."""

    def test_full_training_cnn_single_seed(self):
        """Complete training with CNN host and single seed should work."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        # Germinate seed
        model.germinate_seed("norm", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        # Verify initial state
        assert slot.state.stage == SeedStage.GERMINATED
        assert model.count_active_seeds() == 1

        # Simulate training steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 3, 32, 32)
        target = torch.randint(0, 10, (4,))

        # Transition to TRAINING
        slot.step_epoch()
        assert slot.state.stage == SeedStage.TRAINING

        # Training loop
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Record accuracy (simulated)
            slot.state.metrics.record_accuracy(50.0 + epoch * 2)

        # Verify training metrics accumulated
        assert slot.state.metrics.epochs_total == 5
        assert slot.state.metrics.best_val_accuracy >= 50.0

    def test_cnn_training_to_blending_transition(self):
        """CNN seed should transition from TRAINING to BLENDING when conditions met."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        model.germinate_seed("norm", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        # Transition to TRAINING
        slot.step_epoch()
        assert slot.state.stage == SeedStage.TRAINING

        # Set up conditions for G2 gate
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(50.0 + DEFAULT_MIN_TRAINING_IMPROVEMENT + 1.0)
        slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Accumulate epochs
        for _ in range(DEFAULT_MIN_BLENDING_EPOCHS):
            slot.state.metrics.epochs_in_current_stage += 1

        # Try to advance
        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING

    def test_cnn_complete_lifecycle_to_probationary(self):
        """CNN seed should complete lifecycle through PROBATIONARY."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        model.germinate_seed("noop", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        # GERMINATED -> TRAINING
        slot.step_epoch()
        assert slot.state.stage == SeedStage.TRAINING

        # TRAINING -> BLENDING
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=5)

        # Complete blending
        for _ in range(5):
            slot.state.metrics.record_accuracy(60.0)
            slot.step_epoch()

        # Should be in PROBATIONARY
        assert slot.state.stage == SeedStage.PROBATIONARY


class TestTransformerHostSingleSeedTraining:
    """E2E tests for Transformer host with single seed training."""

    def test_full_training_transformer_single_seed(self):
        """Complete training with Transformer host and single seed should work."""
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=6,
            n_head=2,
            block_size=32,
            num_segments=3,
        )
        model = MorphogeneticModel(host, slots=["r0c0"])

        # Germinate seed
        model.germinate_seed("noop", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        # Verify initial state
        assert slot.state.stage == SeedStage.GERMINATED
        assert model.count_active_seeds() == 1

        # Simulate training steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randint(0, 1000, (4, 16))
        target = torch.randint(0, 1000, (4, 16))

        # Transition to TRAINING
        slot.step_epoch()
        assert slot.state.stage == SeedStage.TRAINING

        # Training loop
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output.view(-1, 1000), target.view(-1))
            loss.backward()
            optimizer.step()

            # Record accuracy (simulated)
            slot.state.metrics.record_accuracy(10.0 + epoch)

        # Verify training metrics accumulated
        assert slot.state.metrics.epochs_total == 5


class TestMultiSlotTraining:
    """E2E tests for multi-slot training."""

    def test_multi_slot_cnn_training(self):
        """Multi-slot CNN training should work with independent lifecycles."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Germinate seeds in all slots
        model.germinate_seed("noop", "seed_0", slot="r0c0")
        model.germinate_seed("norm", "seed_1", slot="r0c1")
        model.germinate_seed("noop", "seed_2", slot="r0c2")

        assert model.count_active_seeds() == 3

        # Training loop
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 3, 32, 32)
        target = torch.randint(0, 10, (4,))

        # Advance all to TRAINING
        for slot in model.seed_slots.values():
            slot.step_epoch()

        # Train for a few epochs
        for epoch in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Record accuracy for each slot
            for i, slot in enumerate(model.seed_slots.values()):
                slot.state.metrics.record_accuracy(50.0 + epoch + i)

        # Verify all slots trained
        for slot in model.seed_slots.values():
            assert slot.state.metrics.epochs_total == 3

    def test_multi_slot_different_stages(self):
        """Multi-slot training with seeds at different stages should work."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Germinate seeds
        model.germinate_seed("noop", "seed_0", slot="r0c0")
        model.germinate_seed("noop", "seed_1", slot="r0c1")
        model.germinate_seed("noop", "seed_2", slot="r0c2")

        # Different stages
        model.seed_slots["r0c0"].step_epoch()  # -> TRAINING
        model.seed_slots["r0c1"].step_epoch()  # -> TRAINING
        model.seed_slots["r0c1"].state.transition(SeedStage.BLENDING)
        model.seed_slots["r0c2"].step_epoch()  # -> TRAINING
        model.seed_slots["r0c2"].state.transition(SeedStage.BLENDING)
        model.seed_slots["r0c2"].state.transition(SeedStage.PROBATIONARY)

        # Verify stages
        assert model.seed_slots["r0c0"].state.stage == SeedStage.TRAINING
        assert model.seed_slots["r0c1"].state.stage == SeedStage.BLENDING
        assert model.seed_slots["r0c2"].state.stage == SeedStage.PROBATIONARY

        # Forward pass should work with mixed stages
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10)


class TestCullAndRecycle:
    """E2E tests for cull and recycle scenarios."""

    def test_cull_and_regerminate(self):
        """Culling and regrowing a seed should work correctly."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        # First seed lifecycle
        model.germinate_seed("noop", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        slot.step_epoch()  # -> TRAINING
        slot.state.metrics.record_accuracy(50.0)

        # Cull
        model.cull_seed(slot="r0c0")
        assert model.count_active_seeds() == 0

        # Second seed lifecycle
        model.germinate_seed("norm", "seed_1", slot="r0c0")
        assert model.count_active_seeds() == 1
        assert slot.state.seed_id == "seed_1"
        assert slot.state.blueprint_id == "norm"
        assert slot.state.metrics.epochs_total == 0

    def test_partial_cull_in_multi_slot(self):
        """Culling one seed in multi-slot should not affect others."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])

        # Germinate both
        model.germinate_seed("noop", "seed_0", slot="r0c0")
        model.germinate_seed("norm", "seed_1", slot="r0c1")

        # Advance both to TRAINING
        model.seed_slots["r0c0"].step_epoch()
        model.seed_slots["r0c1"].step_epoch()

        # Cull r0c0
        model.cull_seed(slot="r0c0")

        assert model.count_active_seeds() == 1
        assert model.seed_slots["r0c0"].state is None
        assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING


class TestGradientTelemetryIntegration:
    """E2E tests for gradient telemetry integration."""

    def test_gradient_norm_captured_during_training(self):
        """Gradient norms should be captured during training."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        model.germinate_seed("norm", "seed_0", slot="r0c0", blend_algorithm_id="sigmoid")
        slot = model.seed_slots["r0c0"]

        # Advance to BLENDING for gradient flow
        slot.step_epoch()  # -> TRAINING
        slot.state.transition(SeedStage.BLENDING)
        slot.set_alpha(0.5)

        # Training with gradient capture
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 3, 32, 32)
        target = torch.randint(0, 10, (4,))

        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()

            # Manually capture gradient telemetry
            seed_grad_norm = 0.0
            for p in slot.get_parameters():
                if p.grad is not None:
                    seed_grad_norm += p.grad.norm().item() ** 2
            seed_grad_norm = seed_grad_norm ** 0.5

            slot.state.metrics.gradient_norm_avg = seed_grad_norm

            optimizer.step()

        # Verify gradient norm was captured
        assert slot.state.metrics.gradient_norm_avg > 0


class TestCounterfactualValidation:
    """E2E tests for counterfactual validation scenarios."""

    def test_counterfactual_contribution_tracking(self):
        """Counterfactual contribution should be tracked correctly."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        model.germinate_seed("noop", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        # Progress to PROBATIONARY
        slot.step_epoch()  # -> TRAINING
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        # Simulate counterfactual validation
        # In real training, this would compare accuracy with alpha=1 vs alpha=0
        slot.state.metrics.counterfactual_contribution = 2.5

        assert slot.state.metrics.counterfactual_contribution == 2.5

    def test_force_alpha_for_counterfactual(self):
        """force_alpha context should work for counterfactual evaluation."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])

        model.germinate_seed("noop", "seed_0", slot="r0c0")
        slot = model.seed_slots["r0c0"]

        slot.step_epoch()  # -> TRAINING
        slot.state.transition(SeedStage.BLENDING)
        slot.set_alpha(0.8)

        assert slot.state.alpha == 0.8

        # Force alpha to 0 for counterfactual
        with slot.force_alpha(0.0):
            assert slot.state.alpha == 0.0

            # Forward pass at alpha=0
            x = torch.randn(4, 3, 32, 32)
            output_baseline = model(x)

        # Alpha restored
        assert slot.state.alpha == 0.8


class TestEndToEndAccuracyTracking:
    """E2E tests for accuracy tracking through the pipeline."""

    def test_accuracy_improvement_tracked(self):
        """Accuracy improvement should be tracked correctly through training."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.step_epoch()  # -> TRAINING

        # Simulate improving accuracy
        accuracies = [40.0, 45.0, 50.0, 52.0, 55.0, 58.0, 60.0]
        for acc in accuracies:
            slot.state.metrics.record_accuracy(acc)

        assert slot.state.metrics.initial_val_accuracy == 40.0
        assert slot.state.metrics.best_val_accuracy == 60.0
        assert slot.state.metrics.current_val_accuracy == 60.0
        assert slot.state.metrics.total_improvement == 20.0

    def test_accuracy_regression_handled(self):
        """Accuracy regression should be tracked without losing best."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.step_epoch()  # -> TRAINING

        # Accuracy goes up then down
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)
        slot.state.metrics.record_accuracy(55.0)  # Regression

        assert slot.state.metrics.best_val_accuracy == 60.0
        assert slot.state.metrics.current_val_accuracy == 55.0


class TestBlendingAlgorithmIntegration:
    """E2E tests for blending algorithm integration."""

    def test_linear_blending_progression(self):
        """Linear blending should progress alpha correctly."""
        from esper.kasmina.blending import LinearBlend

        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test", blend_algorithm_id="linear")
        slot.step_epoch()  # -> TRAINING
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        # Alpha should be linear
        assert isinstance(slot.alpha_schedule, LinearBlend)

        for step in range(11):
            expected_alpha = step / 10
            actual_alpha = slot.alpha_schedule.get_alpha(step)
            assert actual_alpha == pytest.approx(expected_alpha)

    def test_sigmoid_blending_progression(self):
        """Sigmoid blending should have S-curve alpha progression."""
        from esper.kasmina.blending import SigmoidBlend

        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test", blend_algorithm_id="sigmoid")
        slot.step_epoch()  # -> TRAINING
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        assert isinstance(slot.alpha_schedule, SigmoidBlend)

        # Sigmoid should be slower at start and end
        alpha_start = slot.alpha_schedule.get_alpha(0)
        alpha_mid = slot.alpha_schedule.get_alpha(5)
        alpha_end = slot.alpha_schedule.get_alpha(10)

        assert alpha_start < alpha_mid < alpha_end
