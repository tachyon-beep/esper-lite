"""Simic offline policy trainer package.

Implements the replay buffer and PPO+LoRA training loop described in
`docs/design/detailed_design/04-simic.md`.
"""

from .replay import FieldReportReplayBuffer, SimicExperience
from .trainer import SimicTrainer, SimicTrainerConfig
from .validation import PolicyValidator, ValidationConfig, ValidationResult
from .registry import EmbeddingRegistry, EmbeddingRegistryConfig

__all__ = [
    "FieldReportReplayBuffer",
    "SimicExperience",
    "SimicTrainer",
    "SimicTrainerConfig",
    "EmbeddingRegistry",
    "EmbeddingRegistryConfig",
    "PolicyValidator",
    "ValidationConfig",
    "ValidationResult",
]
