"""Tests for action masking functionality."""

import pytest
import torch

from esper.simic.features import compute_action_mask
from esper.simic.networks import MaskedCategorical, InvalidStateMachineError


class TestComputeActionMask:
    """Tests for compute_action_mask function."""

    def test_no_active_seed_allows_germinate(self):
        """Without active seed, GERMINATE actions should be allowed."""
        # 4 germinate actions (like CNN topology)
        mask = compute_action_mask(has_active_seed=0.0, seed_stage=0, num_germinate_actions=4)

        # Total actions: WAIT + 4 GERMINATE + FOSSILIZE + CULL = 7
        assert len(mask) == 7

        # WAIT (0) always valid
        assert mask[0] == 1.0

        # GERMINATE_* (1-4) valid without active seed
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 1.0

        # FOSSILIZE (5) not valid - need PROBATIONARY stage
        assert mask[5] == 0.0

        # CULL (6) not valid - need active seed
        assert mask[6] == 0.0

    def test_active_seed_blocks_germinate(self):
        """With active seed, GERMINATE actions should be blocked."""
        # Seed in TRAINING stage (3)
        mask = compute_action_mask(has_active_seed=1.0, seed_stage=3, num_germinate_actions=4)

        # WAIT (0) always valid
        assert mask[0] == 1.0

        # GERMINATE_* (1-4) blocked with active seed
        assert mask[1] == 0.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        assert mask[4] == 0.0

        # FOSSILIZE (5) not valid - need PROBATIONARY stage (6)
        assert mask[5] == 0.0

        # CULL (6) valid with active seed
        assert mask[6] == 1.0

    def test_probationary_allows_fossilize(self):
        """In PROBATIONARY stage, FOSSILIZE should be allowed."""
        # PROBATIONARY = stage 6
        mask = compute_action_mask(has_active_seed=1.0, seed_stage=6, num_germinate_actions=4)

        # WAIT (0) always valid
        assert mask[0] == 1.0

        # GERMINATE_* (1-4) blocked with active seed
        assert mask[1] == 0.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        assert mask[4] == 0.0

        # FOSSILIZE (5) valid in PROBATIONARY
        assert mask[5] == 1.0

        # CULL (6) valid with active seed
        assert mask[6] == 1.0

    def test_wait_always_valid(self):
        """WAIT should always be valid regardless of state."""
        # Various states
        masks = [
            compute_action_mask(0.0, 0, 4),  # No seed
            compute_action_mask(1.0, 3, 4),  # TRAINING
            compute_action_mask(1.0, 6, 4),  # PROBATIONARY
            compute_action_mask(1.0, 7, 4),  # FOSSILIZED
        ]

        for mask in masks:
            assert mask[0] == 1.0, "WAIT should always be valid"


class TestMaskedCategorical:
    """Tests for MaskedCategorical distribution."""

    def test_all_masked_raises_error(self):
        """Should raise error when all actions are masked."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[0.0, 0.0, 0.0]])  # All invalid

        with pytest.raises(InvalidStateMachineError):
            MaskedCategorical(logits=logits, mask=mask)

    def test_single_valid_action(self):
        """With single valid action, sampling should always return it."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[0.0, 1.0, 0.0]])  # Only action 1 valid

        dist = MaskedCategorical(logits=logits, mask=mask)

        # Sample many times - should always be action 1
        for _ in range(10):
            action = dist.sample()
            assert action.item() == 1

    def test_masked_probs_near_zero(self):
        """Masked actions should have near-zero probability."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # 0 and 2 valid

        dist = MaskedCategorical(logits=logits, mask=mask)
        probs = dist.probs[0]

        # Invalid actions should have near-zero prob
        assert probs[1].item() < 1e-6
        assert probs[3].item() < 1e-6

        # Valid actions should have non-zero prob
        assert probs[0].item() > 0.0
        assert probs[2].item() > 0.0

    def test_entropy_only_considers_valid_actions(self):
        """Entropy should only be computed over valid actions."""
        logits = torch.zeros(1, 4)  # Uniform logits

        # All 4 valid - entropy should be log(4) ≈ 1.386
        all_valid = MaskedCategorical(logits, torch.ones(1, 4))
        entropy_4 = all_valid.entropy()

        # Only 2 valid - entropy should be log(2) ≈ 0.693
        two_valid = MaskedCategorical(logits, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
        entropy_2 = two_valid.entropy()

        # With uniform logits over valid actions:
        # 4 valid: entropy = log(4) = 1.386
        # 2 valid: entropy = log(2) = 0.693
        assert abs(entropy_4.item() - 1.386) < 0.01
        assert abs(entropy_2.item() - 0.693) < 0.01
