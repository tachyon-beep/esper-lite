"""Tests for reward shaping functions."""

import pytest

from esper.simic.rewards import compute_seed_potential


class TestComputeSeedPotential:
    """Tests for potential-based reward shaping."""

    def test_no_seed_returns_zero(self):
        """Test that no active seed has zero potential."""
        obs = {'has_active_seed': 0, 'seed_stage': 0, 'seed_epochs_in_stage': 0}
        assert compute_seed_potential(obs) == 0.0

    def test_germinated_has_low_potential(self):
        """Test GERMINATED stage has low potential."""
        obs = {'has_active_seed': 1, 'seed_stage': 1, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 5.0

    def test_training_has_higher_potential(self):
        """Test TRAINING stage has higher potential than GERMINATED."""
        germ = {'has_active_seed': 1, 'seed_stage': 1, 'seed_epochs_in_stage': 0}
        train = {'has_active_seed': 1, 'seed_stage': 2, 'seed_epochs_in_stage': 0}

        assert compute_seed_potential(train) > compute_seed_potential(germ)

    def test_blending_has_highest_potential(self):
        """Test BLENDING stage has highest potential."""
        obs = {'has_active_seed': 1, 'seed_stage': 3, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 25.0

    def test_progress_bonus_capped(self):
        """Test that progress bonus is capped."""
        obs = {'has_active_seed': 1, 'seed_stage': 2, 'seed_epochs_in_stage': 100}
        potential = compute_seed_potential(obs)
        # Base 15.0 + max 3.0 progress = 18.0
        assert potential == 18.0
