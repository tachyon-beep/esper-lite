"""Tests for DiagnosticTracker - telemetry collection and analysis."""

from __future__ import annotations

import torch
import torch.nn as nn


from esper.nissa.config import TelemetryConfig
from esper.nissa.tracker import DiagnosticTracker


class _SimpleModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTrainingModePreservation:
    """Verify telemetry collection doesn't leak training state."""

    def test_end_epoch_preserves_training_mode_with_loss_landscape(self) -> None:
        """Model.training is unchanged after end_epoch with loss_landscape enabled.

        Regression test for: _estimate_sharpness() calls model.eval() but must
        restore original training mode to avoid corrupting subsequent training.
        """
        # Create config with loss_landscape enabled
        config = TelemetryConfig.from_profile("diagnostic")
        config.loss_landscape.enabled = True
        config.loss_landscape.perturbation_samples = 1  # Minimize test time

        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Ensure model starts in training mode
        model.train()
        assert model.training, "Precondition: model should be in training mode"

        # Create a minimal validation loader
        val_data = [(torch.randn(4, 10), torch.randint(0, 2, (4,))) for _ in range(2)]
        criterion = nn.CrossEntropyLoss()

        # Call end_epoch with val_loader/criterion to trigger _estimate_sharpness
        tracker.end_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_accuracy=60.0,
            val_loader=val_data,
            criterion=criterion,
        )

        # Assert: training mode must be preserved
        assert model.training, (
            "model.training was changed by end_epoch(). "
            "Telemetry collection must not mutate training state."
        )

        tracker.cleanup()

    def test_end_epoch_preserves_eval_mode_with_loss_landscape(self) -> None:
        """If model starts in eval mode, it stays in eval mode after end_epoch."""
        config = TelemetryConfig.from_profile("diagnostic")
        config.loss_landscape.enabled = True
        config.loss_landscape.perturbation_samples = 1

        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Start in eval mode
        model.eval()
        assert not model.training, "Precondition: model should be in eval mode"

        val_data = [(torch.randn(4, 10), torch.randint(0, 2, (4,))) for _ in range(2)]
        criterion = nn.CrossEntropyLoss()

        tracker.end_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_accuracy=60.0,
            val_loader=val_data,
            criterion=criterion,
        )

        assert not model.training, "model.training changed from False to True"
        tracker.cleanup()


class TestPlateauDetection:
    """Verify plateau detection triggers at expected boundaries."""

    def test_plateau_detected_after_three_similar_epochs(self) -> None:
        """Plateau flag triggers when 3+ epochs have similar loss.

        Regression test for: off-by-one error where plateau detection
        was delayed due to history window issues.
        """
        config = TelemetryConfig.from_profile("minimal")
        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Simulate 4 epochs with nearly identical validation loss
        constant_loss = 0.500
        for epoch in range(4):
            snapshot = tracker.end_epoch(
                epoch=epoch,
                train_loss=0.4,
                val_loss=constant_loss + 0.001 * epoch,  # Within threshold
                val_accuracy=60.0,
            )

        # After 4 identical epochs, plateau should be detected
        assert tracker.plateau_detected, (
            f"plateau_detected should be True after 4 epochs with similar loss. "
            f"History: {[s.val_loss for s in tracker.history]}"
        )
        assert "sustained_plateau" in snapshot.red_flags, (
            "sustained_plateau flag missing from red_flags"
        )

        tracker.cleanup()

    def test_no_plateau_with_improving_loss(self) -> None:
        """No plateau detected when loss is consistently improving."""
        config = TelemetryConfig.from_profile("minimal")
        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Simulate 4 epochs with improving loss
        for epoch in range(4):
            snapshot = tracker.end_epoch(
                epoch=epoch,
                train_loss=0.4,
                val_loss=0.500 - 0.05 * epoch,  # Clearly improving
                val_accuracy=60.0 + 5 * epoch,
            )

        assert not tracker.plateau_detected, "Should not detect plateau with improving loss"
        assert "sustained_plateau" not in snapshot.red_flags

        tracker.cleanup()

    def test_plateau_length_includes_current_epoch(self) -> None:
        """_plateau_length with current_loss counts from current epoch."""
        config = TelemetryConfig.from_profile("minimal")
        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Build history: 3 epochs at 0.5
        for epoch in range(3):
            tracker.end_epoch(epoch=epoch, train_loss=0.4, val_loss=0.5, val_accuracy=60.0)

        # Now query with current_loss=0.5 (not yet in history)
        # Should count: current + 3 in history = 4
        length = tracker._plateau_length(current_loss=0.5)
        assert length == 4, f"Expected plateau length 4 (1 current + 3 history), got {length}"

        # With a different current_loss, should break the streak
        length_break = tracker._plateau_length(current_loss=0.3)
        assert length_break == 1, f"Expected plateau length 1 (just current), got {length_break}"

        tracker.cleanup()

    def test_narrative_reports_plateau_at_epoch_3(self) -> None:
        """Narrative mentions plateau starting at epoch 3 (not delayed).

        Regression test for: generate_narrative() skipping epoch1->epoch2
        comparison due to off-by-one history check.
        """
        config = TelemetryConfig.from_profile("minimal")
        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        constant_loss = 0.5
        narratives = []
        for epoch in range(5):
            snapshot = tracker.end_epoch(
                epoch=epoch,
                train_loss=0.4,
                val_loss=constant_loss,
                val_accuracy=60.0,
            )
            narratives.append(snapshot.narrative)

        # By epoch 3 (4th call), we should see plateau mentioned
        # Epochs: 0, 1, 2, 3 -> 4 epochs with same loss = plateau
        assert any("plateau" in n.lower() for n in narratives[3:]), (
            f"Plateau should be mentioned by epoch 3. Narratives: {narratives}"
        )

        tracker.cleanup()


class TestConfigToggles:
    """Test that config flags are honored by DiagnosticTracker."""

    def test_track_norm_false_skips_norm_computation(self) -> None:
        """When track_norm=False, gradient stats.norm stays at default 0.0.

        Regression test for: config flags existed but were not honored.
        """
        config = TelemetryConfig.from_profile("standard")
        config.gradients.enabled = True
        config.gradients.layers = "all"  # Track all layers (test model doesn't have 'host.*' prefix)
        config.gradients.track_norm = False
        config.gradients.track_std = True  # keep std on for comparison

        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Run a backward pass to trigger gradient hooks
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradient stats
        assert len(tracker._grad_stats) > 0, "Should have captured gradient stats"
        for stats in tracker._grad_stats.values():
            # norm should be 0.0 (not computed) since track_norm=False
            assert stats.norm == 0.0, f"norm should be 0.0 when track_norm=False, got {stats.norm}"
            # std should be non-zero since track_std=True
            assert stats.std != 0.0, "std should be computed when track_std=True"

        tracker.cleanup()

    def test_track_std_false_skips_std_computation(self) -> None:
        """When track_std=False, gradient stats.std stays at default 0.0."""
        config = TelemetryConfig.from_profile("standard")
        config.gradients.enabled = True
        config.gradients.layers = "all"  # Track all layers (test model doesn't have 'host.*' prefix)
        config.gradients.track_norm = True  # keep norm on for comparison
        config.gradients.track_std = False

        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Run a backward pass
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradient stats
        assert len(tracker._grad_stats) > 0, "Should have captured gradient stats"
        for stats in tracker._grad_stats.values():
            # std should be 0.0 (not computed) since track_std=False
            assert stats.std == 0.0, f"std should be 0.0 when track_std=False, got {stats.std}"
            # norm should be non-zero since track_norm=True
            assert stats.norm != 0.0, "norm should be computed when track_norm=True"

        tracker.cleanup()

    def test_estimate_sharpness_false_skips_sharpness(self) -> None:
        """When estimate_sharpness=False, snapshot.sharpness is None.

        Regression test for: estimate_sharpness flag existed but only
        loss_landscape.enabled was checked.
        """
        config = TelemetryConfig.from_profile("diagnostic")
        config.loss_landscape.enabled = True  # Enable loss landscape analysis
        config.loss_landscape.estimate_sharpness = False  # But disable sharpness specifically

        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Create minimal validation loader
        val_data = [(torch.randn(4, 10), torch.randint(0, 2, (4,))) for _ in range(2)]
        criterion = nn.CrossEntropyLoss()

        snapshot = tracker.end_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_accuracy=60.0,
            val_loader=val_data,
            criterion=criterion,
        )

        # Sharpness should be None since estimate_sharpness=False
        assert snapshot.sharpness is None, (
            f"sharpness should be None when estimate_sharpness=False, got {snapshot.sharpness}"
        )

        tracker.cleanup()

    def test_estimate_sharpness_true_computes_sharpness(self) -> None:
        """When estimate_sharpness=True (default), snapshot.sharpness is computed."""
        config = TelemetryConfig.from_profile("diagnostic")
        config.loss_landscape.enabled = True
        config.loss_landscape.estimate_sharpness = True  # Explicitly enable
        config.loss_landscape.perturbation_samples = 1  # Minimize test time

        model = _SimpleModel()
        tracker = DiagnosticTracker(model, config, device="cpu")

        # Create minimal validation loader
        val_data = [(torch.randn(4, 10), torch.randint(0, 2, (4,))) for _ in range(2)]
        criterion = nn.CrossEntropyLoss()

        snapshot = tracker.end_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_accuracy=60.0,
            val_loader=val_data,
            criterion=criterion,
        )

        # Sharpness should be computed (non-None float)
        assert snapshot.sharpness is not None, (
            "sharpness should be computed when estimate_sharpness=True"
        )
        assert isinstance(snapshot.sharpness, float), (
            f"sharpness should be float, got {type(snapshot.sharpness)}"
        )

        tracker.cleanup()
