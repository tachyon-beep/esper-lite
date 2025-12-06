"""Unified Hyperparameter Configuration for Simic Training.

Consolidates all training hyperparameters into a single configuration object
for easier tuning and experiment management. Supports task-specific presets
and YAML serialization.

Usage:
    from esper.simic.config import TrainingConfig

    # Default config (CIFAR-10 optimized)
    config = TrainingConfig()

    # Task-specific presets
    config = TrainingConfig.for_cifar10()
    config = TrainingConfig.for_tinystories()

    # Custom config
    config = TrainingConfig(
        lr=1e-4,
        target_kl=0.02,
        ppo_updates_per_batch=2,
    )

    # Use with train_ppo_vectorized
    agent, history = train_ppo_vectorized(**config.to_train_kwargs())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TrainingConfig:
    """Unified training configuration for PPO-based seed lifecycle control.

    Groups hyperparameters by category:
    - PPO core: Learning rate, clip ratio, discount, etc.
    - Entropy: Exploration via entropy bonus and annealing
    - KL stopping: Early stopping based on policy divergence
    - Sample efficiency: Multiple updates per batch
    - Normalization: Observation/reward normalization
    - Task-specific: Governor, stabilization thresholds

    All parameters have sensible defaults optimized for CIFAR-10.
    Use class methods for task-specific presets.
    """

    # === PPO Core ===
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    clip_value: bool = True

    # === Entropy (Exploration) ===
    entropy_coef: float = 0.05  # Unified default (validated in for_tinystories preset)
    entropy_coef_start: float | None = None
    entropy_coef_end: float | None = None
    entropy_coef_min: float = 0.01
    entropy_anneal_episodes: int = 0

    # === KL Early Stopping ===
    # Stop PPO epoch iteration when approx_kl > 1.5 * target_kl
    # None disables early stopping (not recommended)
    target_kl: float | None = 0.015

    # === Sample Efficiency ===
    # Multiple PPO updates per batch of episodes
    # Higher values improve sample efficiency but risk policy divergence
    # With KL early stopping, values of 2-4 are often safe
    ppo_updates_per_batch: int = 1

    # === Training Scale ===
    n_episodes: int = 100
    n_envs: int = 4
    max_epochs: int = 25

    # === Task-Specific: Governor ===
    # Random guess loss for lobotomy detection
    # CIFAR-10: ln(10) ≈ 2.3, TinyStories: ln(50257) ≈ 10.8
    random_guess_loss: float = field(default_factory=lambda: math.log(10))
    governor_sensitivity: float = 6.0
    governor_absolute_threshold: float = 12.0
    governor_death_penalty: float = 10.0

    # === Task-Specific: Stabilization ===
    # Block germination until host training stabilizes
    # Lower threshold = stricter = stabilizes later
    stabilization_threshold: float = 0.03  # 3% relative improvement
    stabilization_epochs: int = 3  # Consecutive stable epochs required

    # === Anomaly Detection Thresholds ===
    # Ratio explosion/collapse detection
    anomaly_max_ratio_threshold: float = 5.0
    anomaly_min_ratio_threshold: float = 0.1
    # Value function collapse detection
    # Lower threshold = triggers only at very poor EV = LESS SENSITIVE (default 0.1)
    # Higher threshold = triggers at mediocre EV = MORE SENSITIVE (e.g., 0.3)
    anomaly_min_explained_variance: float = 0.1

    # === Telemetry ===
    use_telemetry: bool = True

    # === Recurrence (LSTM Policy) ===
    recurrent: bool = False
    lstm_hidden_dim: int = 128
    chunk_length: int | None = None  # None = auto-match max_epochs; set explicitly if different

    def __post_init__(self):
        """Validate and set defaults for recurrent config."""
        import logging
        import warnings

        logger = logging.getLogger(__name__)

        # Auto-match chunk_length to max_epochs if not set
        if self.chunk_length is None:
            self.chunk_length = self.max_epochs

        # Warn if chunk_length < max_epochs (will cause mid-episode chunking)
        if self.recurrent and self.chunk_length < self.max_epochs:
            warnings.warn(
                f"chunk_length={self.chunk_length} < max_epochs={self.max_epochs}. "
                f"Mid-episode chunking will occur, losing temporal context at chunk "
                f"boundaries. For optimal BPTT, set chunk_length >= max_epochs.",
                RuntimeWarning,
            )

        # Note about learning rate for recurrent policies
        if self.recurrent and self.lr > 2.5e-4:
            logger.info(
                f"Using lr={self.lr} with recurrent=True. Recurrent policies can be "
                f"more sensitive to learning rate. If training is unstable, consider "
                f"reducing to lr=2.5e-4 or lr=1e-4."
            )

    @staticmethod
    def for_cifar10() -> "TrainingConfig":
        """Optimized configuration for CIFAR-10 image classification."""
        return TrainingConfig(
            random_guess_loss=math.log(10),  # ln(10) ≈ 2.3
            stabilization_threshold=0.03,
            stabilization_epochs=3,
            entropy_coef=0.1,
            target_kl=0.015,
        )

    @staticmethod
    def for_tinystories() -> "TrainingConfig":
        """Optimized configuration for TinyStories language modeling."""
        return TrainingConfig(
            random_guess_loss=math.log(50257),  # ln(50257) ≈ 10.8
            stabilization_threshold=0.01,  # LLMs have smaller relative improvements
            stabilization_epochs=5,  # More patience for LM stabilization
            governor_absolute_threshold=15.0,  # Higher threshold for LM loss scale
            entropy_coef=0.05,  # Lower entropy for more deterministic LM control
            target_kl=0.02,  # Slightly higher KL tolerance
        )

    @staticmethod
    def for_imagenet() -> "TrainingConfig":
        """Configuration for ImageNet (1000 classes)."""
        return TrainingConfig(
            random_guess_loss=math.log(1000),  # ln(1000) ≈ 6.9
            stabilization_threshold=0.02,
            stabilization_epochs=4,
            max_epochs=50,  # More epochs for larger dataset
        )

    def to_ppo_kwargs(self) -> dict[str, Any]:
        """Extract PPOAgent constructor kwargs."""
        return {
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_start": self.entropy_coef_start,
            "entropy_coef_end": self.entropy_coef_end,
            "entropy_coef_min": self.entropy_coef_min,
            "value_coef": self.value_coef,
            "clip_value": self.clip_value,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "target_kl": self.target_kl,
            "recurrent": self.recurrent,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "chunk_length": self.chunk_length,
        }

    def to_train_kwargs(self) -> dict[str, Any]:
        """Extract train_ppo_vectorized kwargs."""
        return {
            "n_episodes": self.n_episodes,
            "n_envs": self.n_envs,
            "max_epochs": self.max_epochs,
            "use_telemetry": self.use_telemetry,
            "lr": self.lr,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_start": self.entropy_coef_start,
            "entropy_coef_end": self.entropy_coef_end,
            "entropy_coef_min": self.entropy_coef_min,
            "entropy_anneal_episodes": self.entropy_anneal_episodes,
            "gamma": self.gamma,
            "ppo_updates_per_batch": self.ppo_updates_per_batch,
            "recurrent": self.recurrent,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "chunk_length": self.chunk_length,
        }

    def to_governor_kwargs(self) -> dict[str, Any]:
        """Extract TolariaGovernor constructor kwargs."""
        return {
            "sensitivity": self.governor_sensitivity,
            "absolute_threshold": self.governor_absolute_threshold,
            "death_penalty": self.governor_death_penalty,
            "random_guess_loss": self.random_guess_loss,
        }

    def to_tracker_kwargs(self) -> dict[str, Any]:
        """Extract SignalTracker constructor kwargs."""
        return {
            "stabilization_threshold": self.stabilization_threshold,
            "stabilization_epochs": self.stabilization_epochs,
        }

    def to_anomaly_kwargs(self) -> dict[str, Any]:
        """Extract AnomalyDetector constructor kwargs."""
        return {
            "max_ratio_threshold": self.anomaly_max_ratio_threshold,
            "min_ratio_threshold": self.anomaly_min_ratio_threshold,
            "min_explained_variance": self.anomaly_min_explained_variance,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**data)

    def summary(self) -> str:
        """Human-readable summary of configuration."""
        lines = [
            "TrainingConfig:",
            f"  PPO: lr={self.lr}, gamma={self.gamma}, clip={self.clip_ratio}",
            f"  Epochs: {self.n_epochs} PPO epochs, {self.n_episodes} episodes, {self.max_epochs} env epochs",
            f"  Entropy: {self.entropy_coef}" + (f" -> {self.entropy_coef_end}" if self.entropy_coef_end else ""),
            f"  KL stopping: {'enabled' if self.target_kl else 'disabled'}" + (f" (target={self.target_kl})" if self.target_kl else ""),
            f"  Updates/batch: {self.ppo_updates_per_batch}",
            f"  Recurrent: {'LSTM' if self.recurrent else 'MLP'}" + (f" (hidden={self.lstm_hidden_dim}, chunk={self.chunk_length})" if self.recurrent else ""),
            f"  Governor: random_guess_loss={self.random_guess_loss:.2f}",
            f"  Stabilization: threshold={self.stabilization_threshold}, epochs={self.stabilization_epochs}",
        ]
        return "\n".join(lines)


__all__ = ["TrainingConfig"]
