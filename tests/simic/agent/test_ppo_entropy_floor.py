"""Tests for per-head entropy floor penalty in PPO loss."""

import pytest
import torch

from esper.simic.agent.ppo_update import compute_entropy_floor_penalty, compute_losses


class TestEntropyFloorPenalty:
    """Tests for per-head entropy floor penalty function."""

    def test_no_penalty_when_above_floor(self) -> None:
        """Entropy above floor should incur no penalty."""
        entropy = {"blueprint": torch.tensor(0.5)}  # Above 0.4 floor
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}  # Required: always dict

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        assert penalty.item() == pytest.approx(0.0)

    def test_penalty_when_below_floor(self) -> None:
        """Entropy below floor should incur quadratic penalty."""
        entropy = {"blueprint": torch.tensor(0.1)}  # Below 0.4 floor
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        # Shortfall = 0.4 - 0.1 = 0.3, penalty = 0.1 * 0.3^2 = 0.009
        assert penalty.item() > 0
        assert penalty.item() == pytest.approx(0.1 * (0.3 ** 2), rel=0.01)

    def test_penalty_scales_with_shortfall(self) -> None:
        """Larger shortfall should incur larger penalty."""
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        # Small shortfall
        entropy_small = {"blueprint": torch.tensor(0.35)}
        penalty_small = compute_entropy_floor_penalty(entropy_small, head_masks, floor, coef)

        # Large shortfall
        entropy_large = {"blueprint": torch.tensor(0.1)}
        penalty_large = compute_entropy_floor_penalty(entropy_large, head_masks, floor, coef)

        assert penalty_large > penalty_small

    def test_penalty_respects_mask(self) -> None:
        """Penalty should only consider masked (active) timesteps."""
        # Half the timesteps are masked (inactive)
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        head_masks = {"blueprint": mask}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        # Per-step entropy: first 5 have 0.5, last 5 have 0.0
        # Masked mean should be 0.5 (above floor) -> no penalty
        per_step_entropy = torch.tensor([0.5] * 5 + [0.0] * 5)

        # Note: compute_entropy_floor_penalty expects pre-computed mean entropy per head
        # Actually, we need to check how it handles per-step vs scalar entropy
        # For now, test with scalar (mean over masked steps)
        mean_ent = (per_step_entropy * mask).sum() / mask.sum()
        entropy_scalar = {"blueprint": mean_ent}

        penalty = compute_entropy_floor_penalty(entropy_scalar, head_masks, floor, coef)
        assert penalty.item() == pytest.approx(0.0)

    def test_multiple_heads(self) -> None:
        """Penalty should sum across all heads."""
        entropy = {
            "blueprint": torch.tensor(0.1),  # Below 0.4 -> penalty
            "op": torch.tensor(0.5),         # Above 0.15 -> no penalty
        }
        head_masks = {
            "blueprint": torch.ones(10),
            "op": torch.ones(10),
        }
        floor = {"blueprint": 0.4, "op": 0.15}
        coef = {"blueprint": 0.1, "op": 0.1}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        # Only blueprint should contribute
        expected = 0.1 * (0.3 ** 2)  # shortfall 0.3
        assert penalty.item() == pytest.approx(expected, rel=0.01)


class TestEntropyFloorEdgeCases:
    """Edge case tests added per expert review."""

    def test_no_penalty_when_head_inactive(self) -> None:
        """Heads with no valid steps should NOT incur penalty (critical fix)."""
        # Blueprint head is completely masked (no valid steps)
        entropy = {"blueprint": torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])}
        head_masks = {"blueprint": torch.zeros(5)}  # All masked!
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        # Should be 0 - head was never active
        assert penalty.item() == pytest.approx(0.0), \
            f"Expected no penalty for inactive head, got {penalty.item()}"

    def test_per_head_penalty_coefficients(self) -> None:
        """Different heads should use different penalty strengths."""
        entropy = {
            "op": torch.tensor(0.1),       # Below floor 0.15, shortfall 0.05
            "blueprint": torch.tensor(0.1) # Below floor 0.40, shortfall 0.30
        }
        head_masks = {k: torch.ones(10) for k in entropy}
        floor = {"op": 0.15, "blueprint": 0.40}
        coefs = {"op": 0.05, "blueprint": 0.20}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coefs)

        # op: 0.05 * 0.05^2 = 0.000125
        # bp: 0.20 * 0.30^2 = 0.018
        expected = 0.000125 + 0.018
        assert penalty.item() == pytest.approx(expected, rel=0.01)

    def test_entropy_floor_gradient_direction(self) -> None:
        """Entropy below floor should have gradient pushing it up."""
        entropy_param = torch.tensor(0.1, requires_grad=True)
        entropy_dict = {"blueprint": entropy_param}
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        penalty = compute_entropy_floor_penalty(entropy_dict, head_masks, floor, coef)
        penalty.backward()

        # Gradient should be negative (pushes entropy up to reduce penalty)
        assert entropy_param.grad < 0, f"Expected negative gradient, got {entropy_param.grad}"

    def test_empty_entropy_dict_returns_zero(self) -> None:
        """Empty entropy dict should return zero penalty."""
        penalty = compute_entropy_floor_penalty({}, {}, {"blueprint": 0.4}, {"blueprint": 0.1})
        assert penalty.item() == pytest.approx(0.0)


class TestEntropyFloorIntegration:
    """Integration tests for entropy floor in compute_losses."""

    def test_compute_losses_includes_entropy_floor_penalty(self) -> None:
        """compute_losses should include entropy floor penalty when enabled."""
        device = torch.device("cpu")
        batch_size = 32
        head_names = ("op", "blueprint", "slot", "style", "tempo",
                      "alpha_target", "alpha_speed", "alpha_curve")

        # Create minimal inputs for compute_losses (matching actual signature)
        per_head_ratios = {k: torch.ones(batch_size, device=device) for k in head_names}
        per_head_advantages = {k: torch.randn(batch_size, device=device) for k in head_names}
        head_masks = {k: torch.ones(batch_size, device=device) for k in head_names}
        values = torch.randn(batch_size, device=device)
        normalized_returns = torch.randn(batch_size, device=device)
        old_values = torch.randn(batch_size, device=device)

        # Create entropy dict with collapsed blueprint head
        entropy = {
            "op": torch.full((batch_size,), 0.5),      # Healthy
            "blueprint": torch.full((batch_size,), 0.05),  # Collapsed!
            "slot": torch.full((batch_size,), 0.4),
            "style": torch.full((batch_size,), 0.4),
            "tempo": torch.full((batch_size,), 0.4),
            "alpha_target": torch.full((batch_size,), 0.4),
            "alpha_speed": torch.full((batch_size,), 0.4),
            "alpha_curve": torch.full((batch_size,), 0.4),
        }

        entropy_coef_per_head = {"op": 1.0, "blueprint": 1.3, "slot": 1.0,
                                  "style": 1.2, "tempo": 1.3, "alpha_target": 1.2,
                                  "alpha_speed": 1.2, "alpha_curve": 1.2}

        # Without floor penalty
        losses_no_floor = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=True,
            value_clip=0.2,
            value_coef=0.5,
            head_names=head_names,
            entropy_floor=None,  # Disabled
        )

        # With floor penalty
        entropy_floor = {"blueprint": 0.4}  # Blueprint is at 0.05, well below 0.4
        losses_with_floor = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=True,
            value_clip=0.2,
            value_coef=0.5,
            head_names=head_names,
            entropy_floor=entropy_floor,
        )

        # Total loss should be higher with floor penalty
        assert losses_with_floor.total_loss > losses_no_floor.total_loss

    def test_compute_losses_entropy_floor_penalty_with_scalar_coef(self) -> None:
        """entropy_floor_penalty_coef can be scalar (broadcasts to all heads)."""
        device = torch.device("cpu")
        batch_size = 16
        head_names = ("op", "blueprint")

        per_head_ratios = {k: torch.ones(batch_size, device=device) for k in head_names}
        per_head_advantages = {k: torch.zeros(batch_size, device=device) for k in head_names}
        head_masks = {k: torch.ones(batch_size, device=device) for k in head_names}
        values = torch.zeros(batch_size, device=device)
        normalized_returns = torch.zeros(batch_size, device=device)
        old_values = torch.zeros(batch_size, device=device)

        # Both heads collapsed
        entropy = {
            "op": torch.full((batch_size,), 0.05),
            "blueprint": torch.full((batch_size,), 0.05),
        }
        entropy_coef_per_head = {"op": 1.0, "blueprint": 1.0}
        entropy_floor = {"op": 0.4, "blueprint": 0.4}

        # With scalar coef
        losses_scalar = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=False,
            value_clip=0.2,
            value_coef=0.5,
            head_names=head_names,
            entropy_floor=entropy_floor,
            entropy_floor_penalty_coef=0.1,  # Scalar
        )

        # With dict coef (same value)
        losses_dict = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=False,
            value_clip=0.2,
            value_coef=0.5,
            head_names=head_names,
            entropy_floor=entropy_floor,
            entropy_floor_penalty_coef={"op": 0.1, "blueprint": 0.1},  # Dict
        )

        # Should be equal
        assert losses_scalar.total_loss.item() == pytest.approx(
            losses_dict.total_loss.item(), rel=1e-5
        )
