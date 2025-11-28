"""Configuration dataclasses for data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EnvironmentConfig:
    """Defines a training environment setup."""

    config_id: str
    architecture: str
    learning_rate: float
    batch_size: int
    optimizer: str
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_epochs: int = 25

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "architecture": self.architecture,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentConfig:
        return cls(**data)

    @classmethod
    def from_preset(cls, preset_id: str) -> EnvironmentConfig:
        if preset_id not in ENVIRONMENT_PRESETS:
            raise ValueError(f"Unknown preset: {preset_id}. Available: {list(ENVIRONMENT_PRESETS.keys())}")
        return cls(**ENVIRONMENT_PRESETS[preset_id])


@dataclass
class BehaviorPolicyConfig:
    """Defines a Kasmina variant."""

    policy_id: str
    min_epochs_before_germinate: int = 5
    plateau_epochs_to_germinate: int = 3
    cull_after_epochs_without_improvement: int = 5
    blueprint_preference: list[str] | None = None
    epsilon: float = 0.0
    temperature: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "min_epochs_before_germinate": self.min_epochs_before_germinate,
            "plateau_epochs_to_germinate": self.plateau_epochs_to_germinate,
            "cull_after_epochs_without_improvement": self.cull_after_epochs_without_improvement,
            "blueprint_preference": self.blueprint_preference,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehaviorPolicyConfig:
        return cls(**data)

    @classmethod
    def from_preset(cls, preset_id: str) -> BehaviorPolicyConfig:
        if preset_id not in POLICY_PRESETS:
            raise ValueError(f"Unknown preset: {preset_id}. Available: {list(POLICY_PRESETS.keys())}")
        return cls(**POLICY_PRESETS[preset_id])

    def with_epsilon(self, epsilon: float) -> BehaviorPolicyConfig:
        """Return a copy with epsilon-greedy exploration."""
        new_id = f"{self.policy_id}-eps{epsilon}"
        return BehaviorPolicyConfig(
            policy_id=new_id,
            min_epochs_before_germinate=self.min_epochs_before_germinate,
            plateau_epochs_to_germinate=self.plateau_epochs_to_germinate,
            cull_after_epochs_without_improvement=self.cull_after_epochs_without_improvement,
            blueprint_preference=self.blueprint_preference,
            epsilon=epsilon,
            temperature=self.temperature,
        )


@dataclass
class ActionProbabilities:
    """Logged per decision with explicit behavior policy probability."""

    greedy_probs: dict[str, float]
    behavior_probs: dict[str, float]
    behavior_prob: float
    greedy_action: str
    sampled_action: str
    was_exploratory: bool
    epsilon: float

    @staticmethod
    def compute_behavior_prob(greedy_probs: dict[str, float], action: str, epsilon: float) -> float:
        """Compute μ(a|s) accounting for ε-greedy."""
        num_actions = len(greedy_probs)
        return (1 - epsilon) * greedy_probs[action] + epsilon / num_actions

    @classmethod
    def from_decision(
        cls,
        greedy_probs: dict[str, float],
        sampled_action: str,
        epsilon: float,
    ) -> ActionProbabilities:
        """Create from a decision."""
        greedy_action = max(greedy_probs, key=greedy_probs.get)
        behavior_probs = {
            a: cls.compute_behavior_prob(greedy_probs, a, epsilon)
            for a in greedy_probs
        }
        behavior_prob = behavior_probs[sampled_action]
        was_exploratory = sampled_action != greedy_action

        return cls(
            greedy_probs=greedy_probs,
            behavior_probs=behavior_probs,
            behavior_prob=behavior_prob,
            greedy_action=greedy_action,
            sampled_action=sampled_action,
            was_exploratory=was_exploratory,
            epsilon=epsilon,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "greedy_probs": self.greedy_probs,
            "behavior_probs": self.behavior_probs,
            "behavior_prob": self.behavior_prob,
            "greedy_action": self.greedy_action,
            "sampled_action": self.sampled_action,
            "was_exploratory": self.was_exploratory,
            "epsilon": self.epsilon,
        }


@dataclass
class RewardComponents:
    """Stored per step for future relabeling."""

    accuracy_delta: float
    loss_delta: float
    potential_prev: float
    potential_next: float
    intervention_cost: float
    sparse: float = 0.0
    return_to_go: float = 0.0

    # Action-specific costs
    INTERVENTION_COSTS = {
        "WAIT": 0.0,
        "GERMINATE": -0.02,
        "ADVANCE": -0.01,
        "CULL": -0.005,
    }

    def total(self, gamma: float = 0.99, shaping_weight: float = 1.0) -> float:
        """Reconstruct reward with configurable shaping."""
        shaping = gamma * self.potential_next - self.potential_prev
        return self.accuracy_delta * 10 + shaping_weight * shaping + self.intervention_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_delta": self.accuracy_delta,
            "loss_delta": self.loss_delta,
            "potential_prev": self.potential_prev,
            "potential_next": self.potential_next,
            "intervention_cost": self.intervention_cost,
            "sparse": self.sparse,
            "return_to_go": self.return_to_go,
        }


@dataclass
class StepMetadata:
    """Additional per-step fields."""

    timestep: int
    done: bool
    truncated: bool
    state_hash: str
    active_seed_count: int
    best_seed_accuracy: float | None
    training_budget_remaining: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "done": self.done,
            "truncated": self.truncated,
            "state_hash": self.state_hash,
            "active_seed_count": self.active_seed_count,
            "best_seed_accuracy": self.best_seed_accuracy,
            "training_budget_remaining": self.training_budget_remaining,
        }


# =============================================================================
# Presets
# =============================================================================

ENVIRONMENT_PRESETS: dict[str, dict] = {
    "baseline": {
        "config_id": "baseline",
        "architecture": "HostCNN",
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "fast-lr": {
        "config_id": "fast-lr",
        "architecture": "HostCNN",
        "learning_rate": 0.1,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "slow-lr": {
        "config_id": "slow-lr",
        "architecture": "HostCNN",
        "learning_rate": 0.001,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "wide": {
        "config_id": "wide",
        "architecture": "HostCNN-Wide",
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "deep": {
        "config_id": "deep",
        "architecture": "HostCNN-Deep",
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "adam": {
        "config_id": "adam",
        "architecture": "HostCNN",
        "learning_rate": 0.001,
        "batch_size": 128,
        "optimizer": "Adam",
    },
    "small-batch": {
        "config_id": "small-batch",
        "architecture": "HostCNN",
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "SGD",
    },
    "large-batch": {
        "config_id": "large-batch",
        "architecture": "HostCNN",
        "learning_rate": 0.02,
        "batch_size": 256,
        "optimizer": "SGD",
    },
    "resnet18": {
        "config_id": "resnet18",
        "architecture": "ResNet-18",
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "resnet34": {
        "config_id": "resnet34",
        "architecture": "ResNet-34",
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "resnet34-adam": {
        "config_id": "resnet34-adam",
        "architecture": "ResNet-34",
        "learning_rate": 0.001,
        "batch_size": 128,
        "optimizer": "Adam",
    },
    "resnet34-slow": {
        "config_id": "resnet34-slow",
        "architecture": "ResNet-34",
        "learning_rate": 0.005,
        "batch_size": 128,
        "optimizer": "SGD",
    },
    "resnet34-small-batch": {
        "config_id": "resnet34-small-batch",
        "architecture": "ResNet-34",
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "SGD",
    },
}

POLICY_PRESETS: dict[str, dict] = {
    "baseline": {
        "policy_id": "baseline",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 5,
    },
    "early-intervener": {
        "policy_id": "early-intervener",
        "min_epochs_before_germinate": 3,
        "plateau_epochs_to_germinate": 2,
        "cull_after_epochs_without_improvement": 5,
    },
    "late-intervener": {
        "policy_id": "late-intervener",
        "min_epochs_before_germinate": 8,
        "plateau_epochs_to_germinate": 5,
        "cull_after_epochs_without_improvement": 5,
    },
    "quick-culler": {
        "policy_id": "quick-culler",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 3,
    },
    "patient-culler": {
        "policy_id": "patient-culler",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 8,
    },
    "blueprint-explorer": {
        "policy_id": "blueprint-explorer",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 5,
        "blueprint_preference": ["conv_enhance", "attention", "norm", "depthwise"],
    },
    "aggressive": {
        "policy_id": "aggressive",
        "min_epochs_before_germinate": 3,
        "plateau_epochs_to_germinate": 2,
        "cull_after_epochs_without_improvement": 3,
    },
    "conservative": {
        "policy_id": "conservative",
        "min_epochs_before_germinate": 8,
        "plateau_epochs_to_germinate": 5,
        "cull_after_epochs_without_improvement": 8,
    },
    "random": {
        "policy_id": "random",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 5,
        "epsilon": 1.0,  # Full random
    },
    "anti-kasmina": {
        "policy_id": "anti-kasmina",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 5,
        # Special flag handled in policy wrapper
    },
    "periodic": {
        "policy_id": "periodic",
        "min_epochs_before_germinate": 5,
        "plateau_epochs_to_germinate": 3,
        "cull_after_epochs_without_improvement": 5,
        # Special flag handled in policy wrapper
    },
}
