"""Tests for TrainingConfig hyperparameter validation."""

import pytest
from esper.simic.training import TrainingConfig


class TestHyperparameterValidation:
    """Test that invalid hyperparameters raise clear errors."""

    def test_gamma_must_be_in_range(self):
        """Gamma must be in (0, 1]."""
        with pytest.raises(ValueError, match="gamma"):
            TrainingConfig(gamma=1.5)
        with pytest.raises(ValueError, match="gamma"):
            TrainingConfig(gamma=0.0)
        with pytest.raises(ValueError, match="gamma"):
            TrainingConfig(gamma=-0.1)
        # Valid edge case
        TrainingConfig(gamma=1.0)  # Should not raise
        TrainingConfig(gamma=0.001)  # Should not raise

    def test_clip_ratio_must_be_positive(self):
        """Clip ratio must be > 0."""
        with pytest.raises(ValueError, match="clip_ratio"):
            TrainingConfig(clip_ratio=0.0)
        with pytest.raises(ValueError, match="clip_ratio"):
            TrainingConfig(clip_ratio=-0.1)

    def test_lr_must_be_positive(self):
        """Learning rate must be > 0."""
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=0.0)
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=-1e-4)

    def test_entropy_coef_must_be_non_negative(self):
        """Entropy coefficient must be >= 0."""
        with pytest.raises(ValueError, match="entropy_coef"):
            TrainingConfig(entropy_coef=-0.1)
        # Zero is valid (no entropy bonus)
        TrainingConfig(entropy_coef=0.0)  # Should not raise
