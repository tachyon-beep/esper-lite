"""Generation orchestrator for managing episode generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from esper.datagen.configs import (
    EnvironmentConfig,
    BehaviorPolicyConfig,
    ENVIRONMENT_PRESETS,
    POLICY_PRESETS,
)


@dataclass
class GenerationPlan:
    """Plan for generating a single episode."""

    env_config_id: str
    policy_config_id: str
    epsilon: float
    episode_idx: int
    random_seed: int | None = None

    @property
    def episode_id(self) -> str:
        """Generate unique episode ID."""
        policy_suffix = f"{self.policy_config_id}"
        if self.epsilon > 0:
            policy_suffix = f"{self.policy_config_id}-eps{self.epsilon}"
        return f"{self.env_config_id}_{policy_suffix}_{self.episode_idx:04d}"

    def to_dict(self) -> dict:
        return {
            "env_config_id": self.env_config_id,
            "policy_config_id": self.policy_config_id,
            "epsilon": self.epsilon,
            "episode_idx": self.episode_idx,
            "episode_id": self.episode_id,
            "random_seed": self.random_seed,
        }

    def get_env_config(self) -> EnvironmentConfig:
        return EnvironmentConfig.from_preset(self.env_config_id)

    def get_policy_config(self) -> BehaviorPolicyConfig:
        config = BehaviorPolicyConfig.from_preset(self.policy_config_id)
        if self.epsilon > 0:
            config = config.with_epsilon(self.epsilon)
        return config


def create_generation_matrix(
    env_ids: list[str] | None = None,
    policy_ids: list[str] | None = None,
    episodes_per_combo: int = 10,
    epsilon_values: list[float] | None = None,
    base_seed: int = 42,
) -> list[GenerationPlan]:
    """Create the full generation matrix.

    Args:
        env_ids: Environment config IDs (default: all presets)
        policy_ids: Policy config IDs (default: all presets)
        episodes_per_combo: Episodes per (env, policy, epsilon) combo
        epsilon_values: Epsilon values to use (default: [0.0] for threshold policies, [1.0] for random)
        base_seed: Base random seed

    Returns:
        List of GenerationPlan objects
    """
    if env_ids is None:
        env_ids = list(ENVIRONMENT_PRESETS.keys())
    if policy_ids is None:
        policy_ids = list(POLICY_PRESETS.keys())

    plans = []
    seed_counter = base_seed

    for env_id in env_ids:
        for policy_id in policy_ids:
            # Determine epsilon values for this policy
            if epsilon_values is not None:
                epsilons = epsilon_values
            elif policy_id == "random":
                epsilons = [1.0]  # Random policy always uses epsilon=1.0
            elif policy_id in ("anti-kasmina", "periodic"):
                epsilons = [0.0]  # Special policies don't use epsilon
            else:
                epsilons = [0.0]  # Default: no exploration for base policy

            for epsilon in epsilons:
                for ep_idx in range(episodes_per_combo):
                    plan = GenerationPlan(
                        env_config_id=env_id,
                        policy_config_id=policy_id,
                        epsilon=epsilon,
                        episode_idx=ep_idx,
                        random_seed=seed_counter,
                    )
                    plans.append(plan)
                    seed_counter += 1

    return plans


class GenerationOrchestrator:
    """Manages episode generation across the matrix."""

    def __init__(
        self,
        output_dir: str | Path,
        env_ids: list[str] | None = None,
        policy_ids: list[str] | None = None,
        episodes_per_combo: int = 10,
        epsilon_values: list[float] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create generation matrix
        self.plans = create_generation_matrix(
            env_ids=env_ids,
            policy_ids=policy_ids,
            episodes_per_combo=episodes_per_combo,
            epsilon_values=epsilon_values,
        )

        # Track progress
        self.completed: set[str] = set()
        self._load_progress()

    @property
    def total_planned(self) -> int:
        return len(self.plans)

    @property
    def completed_count(self) -> int:
        return len(self.completed)

    @property
    def remaining_count(self) -> int:
        return self.total_planned - self.completed_count

    def get_next_batch(self, batch_size: int = 10) -> list[GenerationPlan]:
        """Get next batch of episodes to generate."""
        batch = []
        for plan in self.plans:
            if plan.episode_id not in self.completed:
                batch.append(plan)
                if len(batch) >= batch_size:
                    break
        return batch

    def mark_complete(self, episode_id: str) -> None:
        """Mark an episode as complete."""
        self.completed.add(episode_id)
        self._save_progress()

    def _progress_file(self) -> Path:
        return self.output_dir / ".generation_progress.json"

    def _load_progress(self) -> None:
        """Load progress from file."""
        progress_file = self._progress_file()
        if progress_file.exists():
            with open(progress_file) as f:
                data = json.load(f)
                self.completed = set(data.get("completed", []))

    def _save_progress(self) -> None:
        """Save progress to file."""
        progress_file = self._progress_file()
        with open(progress_file, "w") as f:
            json.dump({"completed": list(self.completed)}, f)

    def get_progress_summary(self) -> dict:
        """Get progress summary."""
        return {
            "total_planned": self.total_planned,
            "completed": self.completed_count,
            "remaining": self.remaining_count,
            "progress_pct": self.completed_count / self.total_planned * 100 if self.total_planned > 0 else 0,
        }
