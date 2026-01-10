"""Tests for PPO update loss functions."""

import pytest
import torch

from esper.simic.agent.ppo_update import compute_contribution_aux_loss


class TestComputeContributionAuxLoss:
    """Tests for auxiliary loss computation for contribution predictions."""

    def test_compute_contribution_auxiliary_loss(self) -> None:
        """Auxiliary loss computes MSE between predictions and targets."""
        batch_size = 32
        seq_len = 10
        num_slots = 6

        pred = torch.randn(batch_size, seq_len, num_slots)
        targets = torch.randn(batch_size, seq_len, num_slots)
        mask = torch.ones(batch_size, seq_len, num_slots, dtype=torch.bool)
        mask[:, :, 4:] = False  # Last 2 slots inactive
        has_fresh = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        has_fresh[:, ::2] = True  # Every other timestep has measurement

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh)

        assert loss.shape == ()  # Scalar
        assert loss >= 0  # MSE is non-negative

    def test_contribution_aux_loss_zero_when_no_measurements(self) -> None:
        """Returns 0.0 when no fresh measurements in batch."""
        pred = torch.randn(4, 10, 6)
        targets = torch.randn(4, 10, 6)
        mask = torch.ones(4, 10, 6, dtype=torch.bool)
        has_fresh = torch.zeros(4, 10, dtype=torch.bool)  # No measurements!

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh)

        assert loss.item() == 0.0

    def test_contribution_aux_loss_with_clip(self) -> None:
        """Large target values are clipped to prevent outlier domination."""
        pred = torch.zeros(4, 10, 6)
        targets = torch.zeros(4, 10, 6)
        targets[0, 0, 0] = 100.0  # Outlier
        mask = torch.ones(4, 10, 6, dtype=torch.bool)
        has_fresh = torch.ones(4, 10, dtype=torch.bool)

        loss_unclipped = compute_contribution_aux_loss(pred, targets, mask, has_fresh, clip=None)
        loss_clipped = compute_contribution_aux_loss(pred, targets, mask, has_fresh, clip=10.0)

        assert loss_clipped < loss_unclipped

    def test_contribution_aux_loss_respects_active_mask(self) -> None:
        """Loss should only be computed over active slots."""
        pred = torch.zeros(2, 5, 4)
        targets = torch.zeros(2, 5, 4)
        targets[:, :, 0] = 10.0  # Large error in slot 0
        targets[:, :, 1] = 10.0  # Large error in slot 1

        mask = torch.zeros(2, 5, 4, dtype=torch.bool)
        mask[:, :, 2:] = True  # Only slots 2,3 active (which have 0 error)

        has_fresh = torch.ones(2, 5, dtype=torch.bool)

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh, clip=None)

        # Since active slots have zero error, loss should be zero
        assert loss.item() == pytest.approx(0.0)

    def test_contribution_aux_loss_respects_fresh_mask(self) -> None:
        """Loss should only be computed at timesteps with fresh measurements."""
        pred = torch.zeros(2, 4, 3)
        targets = torch.zeros(2, 4, 3)
        targets[:, 0, :] = 10.0  # Large error at timestep 0
        targets[:, 1, :] = 10.0  # Large error at timestep 1

        mask = torch.ones(2, 4, 3, dtype=torch.bool)
        has_fresh = torch.zeros(2, 4, dtype=torch.bool)
        has_fresh[:, 2:] = True  # Only timesteps 2,3 have fresh measurements (zero error)

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh, clip=None)

        # Since fresh timesteps have zero error, loss should be zero
        assert loss.item() == pytest.approx(0.0)

    def test_contribution_aux_loss_combined_masks(self) -> None:
        """Both active_mask and has_fresh_target must be true for loss computation."""
        batch_size, seq_len, num_slots = 2, 4, 3
        pred = torch.zeros(batch_size, seq_len, num_slots)
        targets = torch.ones(batch_size, seq_len, num_slots) * 5.0  # All have error

        # Only one specific position should contribute
        mask = torch.zeros(batch_size, seq_len, num_slots, dtype=torch.bool)
        mask[0, 1, 0] = True  # Only slot 0 at batch 0, timestep 1

        has_fresh = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        has_fresh[0, 1] = True  # Only timestep 1 at batch 0

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh, clip=None)

        # MSE for single element: (0 - 5)^2 = 25
        assert loss.item() == pytest.approx(25.0)

    def test_contribution_aux_loss_device_preservation(self) -> None:
        """Loss should be on same device as inputs."""
        pred = torch.randn(4, 10, 6)
        targets = torch.randn(4, 10, 6)
        mask = torch.ones(4, 10, 6, dtype=torch.bool)
        has_fresh = torch.zeros(4, 10, dtype=torch.bool)  # No measurements

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh)

        assert loss.device == pred.device

    def test_contribution_aux_loss_clip_symmetric(self) -> None:
        """Clip should apply symmetrically to positive and negative targets."""
        pred = torch.zeros(2, 2, 2)
        targets = torch.zeros(2, 2, 2)
        targets[0, 0, 0] = 100.0   # Large positive outlier
        targets[1, 1, 1] = -100.0  # Large negative outlier
        mask = torch.ones(2, 2, 2, dtype=torch.bool)
        has_fresh = torch.ones(2, 2, dtype=torch.bool)

        loss = compute_contribution_aux_loss(pred, targets, mask, has_fresh, clip=10.0)

        # Both outliers clipped to +/-10, so errors are 100 each
        # Total: 200 / 8 elements = 25
        assert loss.item() == pytest.approx(25.0)
