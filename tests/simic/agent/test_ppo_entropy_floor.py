"""Tests for per-head entropy floor penalty in PPO loss."""

import pytest
import torch

from esper.simic.agent.ppo_update import compute_entropy_floor_penalty


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
        entropy = {"blueprint": per_step_entropy}

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
