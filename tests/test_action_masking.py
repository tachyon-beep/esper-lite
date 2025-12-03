"""Tests for action masking functionality."""

import pytest
import torch

from esper.simic.features import (
    compute_action_mask,
    MIN_CULL_AGE,
    MIN_GERMINATE_EPOCH,
    MIN_PLATEAU_TO_GERMINATE,
)
from esper.simic.networks import MaskedCategorical, InvalidStateMachineError


class TestComputeActionMask:
    """Tests for compute_action_mask function."""

    def test_no_active_seed_with_plateau_allows_germinate(self):
        """Without active seed AND plateau detected, GERMINATE should be allowed."""
        # 4 germinate actions, plateau conditions met
        mask = compute_action_mask(
            has_active_seed=0.0,
            seed_stage=0,
            num_germinate_actions=4,
            epoch=MIN_GERMINATE_EPOCH,
            plateau_epochs=MIN_PLATEAU_TO_GERMINATE,
        )

        # Total actions: WAIT + 4 GERMINATE + FOSSILIZE + CULL = 7
        assert len(mask) == 7

        # WAIT (0) always valid
        assert mask[0] == 1.0

        # GERMINATE_* (1-4) valid without active seed when plateau met
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 1.0

        # FOSSILIZE (5) not valid - need PROBATIONARY stage
        assert mask[5] == 0.0

        # CULL (6) not valid - need active seed
        assert mask[6] == 0.0

    def test_active_seed_blocks_germinate(self):
        """With active seed, GERMINATE actions should be blocked even if plateau met."""
        # Seed in TRAINING stage (3), old enough to cull, plateau met
        mask = compute_action_mask(
            has_active_seed=1.0,
            seed_stage=3,
            num_germinate_actions=4,
            seed_age_epochs=MIN_CULL_AGE,
            epoch=MIN_GERMINATE_EPOCH,
            plateau_epochs=MIN_PLATEAU_TO_GERMINATE,
        )

        # WAIT (0) always valid
        assert mask[0] == 1.0

        # GERMINATE_* (1-4) blocked with active seed (even with plateau)
        assert mask[1] == 0.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        assert mask[4] == 0.0

        # FOSSILIZE (5) not valid - need PROBATIONARY stage (6)
        assert mask[5] == 0.0

        # CULL (6) valid with active, mature seed
        assert mask[6] == 1.0

    def test_probationary_allows_fossilize(self):
        """In PROBATIONARY stage, FOSSILIZE should be allowed."""
        # PROBATIONARY = stage 6, seed is mature (reached PROBATIONARY takes time)
        mask = compute_action_mask(
            has_active_seed=1.0, seed_stage=6, num_germinate_actions=4,
            seed_age_epochs=10  # Seeds reaching PROBATIONARY are always mature
        )

        # WAIT (0) always valid
        assert mask[0] == 1.0

        # GERMINATE_* (1-4) blocked with active seed
        assert mask[1] == 0.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        assert mask[4] == 0.0

        # FOSSILIZE (5) valid in PROBATIONARY
        assert mask[5] == 1.0

        # CULL (6) valid with active, mature seed
        assert mask[6] == 1.0

    def test_wait_always_valid(self):
        """WAIT should always be valid regardless of state."""
        # Various states - WAIT is valid in all of them
        masks = [
            # No seed, no plateau (only WAIT valid)
            compute_action_mask(0.0, 0, 4, epoch=0, plateau_epochs=0),
            # No seed, plateau met (WAIT + GERMINATE valid)
            compute_action_mask(0.0, 0, 4, epoch=5, plateau_epochs=3),
            # Active seed in TRAINING
            compute_action_mask(1.0, 3, 4, seed_age_epochs=5),
            # Active seed in PROBATIONARY
            compute_action_mask(1.0, 6, 4, seed_age_epochs=10),
            # Active seed FOSSILIZED
            compute_action_mask(1.0, 7, 4, seed_age_epochs=15),
        ]

        for mask in masks:
            assert mask[0] == 1.0, "WAIT should always be valid"

    def test_young_seed_cannot_be_culled(self):
        """Seeds younger than MIN_CULL_AGE cannot be culled.

        This prevents culling seeds before we have enough signal to evaluate
        their quality. MIN_CULL_AGE=10 gives ~3% random survival rate.
        """
        # Young seed at age 0-9 - CULL should be blocked
        for age in range(MIN_CULL_AGE):
            mask = compute_action_mask(
                has_active_seed=1.0, seed_stage=3, num_germinate_actions=4,
                seed_age_epochs=age
            )
            # CULL (index 6) should be invalid for young seeds
            assert mask[6] == 0.0, f"CULL should be blocked at age {age}"

            # WAIT should still be valid
            assert mask[0] == 1.0, "WAIT should always be valid"

    def test_mature_seed_can_be_culled(self):
        """Seeds at or above MIN_CULL_AGE (10) can be culled."""
        # Mature seed at exactly MIN_CULL_AGE
        mask = compute_action_mask(
            has_active_seed=1.0, seed_stage=3, num_germinate_actions=4,
            seed_age_epochs=MIN_CULL_AGE
        )
        assert mask[6] == 1.0, f"CULL should be valid at age {MIN_CULL_AGE}"

        # Even older seed
        mask_older = compute_action_mask(
            has_active_seed=1.0, seed_stage=3, num_germinate_actions=4,
            seed_age_epochs=MIN_CULL_AGE + 5
        )
        assert mask_older[6] == 1.0, "CULL should be valid for older seeds"


class TestPlateauGating:
    """Tests for plateau gating of GERMINATE actions.

    Plateau gating ensures seeds only get credit for improvements AFTER
    natural training gains have exhausted. This fixes credit misattribution
    from early germination and matches h-tamiyo behavior.
    """

    def test_early_epoch_blocks_germinate(self):
        """GERMINATE blocked before MIN_GERMINATE_EPOCH even with plateau."""
        for epoch in range(MIN_GERMINATE_EPOCH):
            mask = compute_action_mask(
                has_active_seed=0.0,
                seed_stage=0,
                num_germinate_actions=4,
                epoch=epoch,
                plateau_epochs=MIN_PLATEAU_TO_GERMINATE,  # Plateau met
            )
            # GERMINATE_* (1-4) should be blocked
            assert mask[1] == 0.0, f"GERMINATE blocked at epoch {epoch}"
            assert mask[2] == 0.0
            assert mask[3] == 0.0
            assert mask[4] == 0.0
            # WAIT should still be valid
            assert mask[0] == 1.0

    def test_insufficient_plateau_blocks_germinate(self):
        """GERMINATE blocked without sufficient plateau epochs."""
        for plateau in range(MIN_PLATEAU_TO_GERMINATE):
            mask = compute_action_mask(
                has_active_seed=0.0,
                seed_stage=0,
                num_germinate_actions=4,
                epoch=MIN_GERMINATE_EPOCH,  # Epoch requirement met
                plateau_epochs=plateau,
            )
            # GERMINATE_* (1-4) should be blocked
            assert mask[1] == 0.0, f"GERMINATE blocked at plateau_epochs={plateau}"
            assert mask[2] == 0.0
            assert mask[3] == 0.0
            assert mask[4] == 0.0
            # WAIT should still be valid
            assert mask[0] == 1.0

    def test_plateau_met_allows_germinate(self):
        """GERMINATE allowed when both epoch and plateau thresholds met."""
        mask = compute_action_mask(
            has_active_seed=0.0,
            seed_stage=0,
            num_germinate_actions=4,
            epoch=MIN_GERMINATE_EPOCH,
            plateau_epochs=MIN_PLATEAU_TO_GERMINATE,
        )
        # GERMINATE_* (1-4) should be valid
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 1.0

    def test_plateau_exceeded_allows_germinate(self):
        """GERMINATE allowed when thresholds exceeded."""
        mask = compute_action_mask(
            has_active_seed=0.0,
            seed_stage=0,
            num_germinate_actions=4,
            epoch=MIN_GERMINATE_EPOCH + 5,
            plateau_epochs=MIN_PLATEAU_TO_GERMINATE + 2,
        )
        # GERMINATE_* (1-4) should be valid
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 1.0

    def test_only_wait_valid_before_plateau(self):
        """Before plateau, only WAIT should be valid (no seed)."""
        mask = compute_action_mask(
            has_active_seed=0.0,
            seed_stage=0,
            num_germinate_actions=4,
            epoch=0,
            plateau_epochs=0,
        )
        # Only WAIT (0) should be valid
        assert mask[0] == 1.0
        # All others should be invalid
        for i in range(1, len(mask)):
            assert mask[i] == 0.0, f"Action {i} should be blocked before plateau"


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
