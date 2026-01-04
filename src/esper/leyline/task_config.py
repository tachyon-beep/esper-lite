"""Leyline Task Config - Task-specific configuration for training.

TaskConfig defines task-specific parameters for observation normalization
and training behavior. It is used across multiple subsystems:
- tamiyo: Feature extraction and observation encoding
- runtime: Task specification and wiring
- tolaria: Model factory
- simic: Training loops
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TaskConfig:
    """Task-specific configuration for observation normalization.

    This dataclass captures task-specific parameters that affect how
    observations are normalized and how training progresses.

    Attributes:
        task_type: Type of task - "classification" or "lm" (language modeling)
        topology: Network topology - "cnn" or "transformer"
        baseline_loss: Expected loss at random initialization
        target_loss: Achievable loss with good training
        typical_loss_delta_std: Standard deviation of loss changes
        max_epochs: Maximum epochs for the task
        max_steps: Maximum training steps
        train_to_blend_fraction: Fraction of max_epochs to stay in TRAINING before blending
        blending_steps: Steps for alpha ramp during blending stage
    """

    task_type: str  # "classification" or "lm"
    topology: str  # "cnn" or "transformer"
    baseline_loss: float  # Random init loss
    target_loss: float  # Achievable loss
    typical_loss_delta_std: float
    max_epochs: int
    max_steps: int = 10000
    train_to_blend_fraction: float = 0.1  # Fraction of max_epochs to stay in TRAINING before blending
    blending_steps: int = 5  # Steps for alpha ramp during blending

    @property
    def achievable_range(self) -> float:
        """Range of achievable loss reduction (baseline - target)."""
        return self.baseline_loss - self.target_loss

    @staticmethod
    def for_cifar10() -> "TaskConfig":
        """Create TaskConfig preset for CIFAR-10 classification."""
        return TaskConfig(
            task_type="classification",
            topology="cnn",
            baseline_loss=2.3,  # ln(10)
            target_loss=0.3,
            typical_loss_delta_std=0.05,
            max_epochs=25,
            max_steps=10000,
        )

    @staticmethod
    def for_tinystories() -> "TaskConfig":
        """Create TaskConfig preset for TinyStories language modeling."""
        return TaskConfig(
            task_type="lm",
            topology="transformer",
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,
            typical_loss_delta_std=0.15,
            max_epochs=50,
            max_steps=50000,
        )


__all__ = ["TaskConfig"]
