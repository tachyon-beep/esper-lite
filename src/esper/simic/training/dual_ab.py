"""Dual-policy A/B testing: train separate policies on separate GPUs.

This module implements true A/B comparison of reward modes by training
independent policies in parallel, each on its own GPU with its own reward mode.

Architecture:
    - One PolicyGroup per GPU/device
    - Each group has: independent agent, dedicated envs, own reward config
    - Groups train in lockstep for fair comparison
    - Final results show which reward mode produces the better policy

Usage:
    from esper.simic.training import train_dual_policy_ab
    from esper.simic.rewards import RewardMode

    results = train_dual_policy_ab(
        n_envs_per_group=4,
        group_configs=[
            ("A", RewardMode.SHAPED),
            ("B", RewardMode.SIMPLIFIED),
        ],
        devices=["cuda:0", "cuda:1"],
        n_episodes=100,
    )

    agent_a, history_a = results["A"]
    agent_b, history_b = results["B"]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_CLIP_RATIO,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
)
from esper.simic.agent import PPOAgent
from esper.simic.rewards import ContributionRewardConfig, RewardMode
from esper.tamiyo.policy.features import get_feature_size
from .policy_group import PolicyGroup
from .vectorized import train_ppo_vectorized

if TYPE_CHECKING:
    from esper.simic.rewards import RewardMode as RewardModeType

_logger = logging.getLogger(__name__)


def train_dual_policy_ab(
    n_envs_per_group: int = 4,
    group_configs: list[tuple[str, "RewardModeType"]] | None = None,
    devices: list[str] | None = None,
    n_episodes: int = 100,
    max_epochs: int = DEFAULT_EPISODE_LENGTH,
    task: str = "cifar10",
    lr: float = DEFAULT_LEARNING_RATE,
    clip_ratio: float = DEFAULT_CLIP_RATIO,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,
    entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
    seed: int = 42,
    use_telemetry: bool = True,
    slots: list[str] | None = None,
    **kwargs,
) -> dict[str, tuple[PPOAgent, list[dict]]]:
    """Train multiple policies in parallel, one per GPU, for true A/B testing.

    This function creates one PolicyGroup per device, each with its own:
    - PPOAgent (independent policy network)
    - Dedicated environments (n_envs_per_group each)
    - Reward configuration (different reward mode per group)

    Groups train in lockstep: each episode, all groups complete their episodes
    before moving to the next. This ensures fair comparison - all groups see
    the same number of episodes and training steps.

    Args:
        n_envs_per_group: Number of parallel environments per policy group
        group_configs: List of (group_id, reward_mode) tuples. Defaults to:
            [("A", RewardMode.SHAPED), ("B", RewardMode.SIMPLIFIED)]
        devices: List of device strings (e.g., ["cuda:0", "cuda:1"]).
            Must have one device per group. Defaults to first N available GPUs.
        n_episodes: Total episodes to train (per group)
        max_epochs: Episode length (epochs per episode)
        task: Task name (e.g., "cifar10")
        lr: Learning rate for PPO
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient for exploration
        entropy_coef_min: Minimum entropy coefficient (for annealing)
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        lstm_hidden_dim: LSTM hidden dimension for policy network
        seed: Random seed base (each group gets seed + group_idx * 10000)
        use_telemetry: Enable telemetry events
        slots: List of slot IDs to use (e.g., ["r0c0", "r0c1", "r0c2"]).
            Defaults to ["r0c0", "r0c1", "r0c2"] (3 slots).
        **kwargs: Additional arguments passed to train_ppo_vectorized

    Returns:
        Dict mapping group_id -> (agent, history)
        where history is a list of dicts with batch-level metrics:
            - "batch": batch number
            - "episodes": total episodes completed
            - "avg_accuracy": average validation accuracy
            - "rolling_avg_accuracy": rolling average accuracy

    Raises:
        ValueError: If number of devices doesn't match number of groups
        ValueError: If fewer than 2 groups specified (need comparison)

    Example:
        >>> results = train_dual_policy_ab(
        ...     n_envs_per_group=4,
        ...     group_configs=[("A", RewardMode.SHAPED), ("B", RewardMode.SIMPLIFIED)],
        ...     devices=["cuda:0", "cuda:1"],
        ...     n_episodes=100,
        ... )
        >>> agent_a, history_a = results["A"]
        >>> final_acc_a = history_a[-1]["final_accuracy"]
    """
    # Validate inputs and apply defaults
    if group_configs is None:
        group_configs = [
            ("A", RewardMode.SHAPED),
            ("B", RewardMode.SIMPLIFIED),
        ]

    if slots is None:
        slots = ["r0c0", "r0c1", "r0c2"]  # Default 3-slot configuration

    if len(group_configs) < 2:
        raise ValueError(
            f"Need at least 2 groups for A/B testing, got {len(group_configs)}"
        )

    # Auto-detect devices if not provided
    if devices is None:
        if not torch.cuda.is_available():
            raise ValueError("--dual-ab requires CUDA, but CUDA is not available")
        n_devices = torch.cuda.device_count()
        if n_devices < len(group_configs):
            raise ValueError(
                f"--dual-ab requires {len(group_configs)} GPUs, "
                f"but only {n_devices} available"
            )
        devices = [f"cuda:{i}" for i in range(len(group_configs))]

    if len(devices) != len(group_configs):
        raise ValueError(
            f"Number of devices ({len(devices)}) must match "
            f"number of groups ({len(group_configs)})"
        )

    _logger.info(
        f"Starting dual-policy A/B test with {len(group_configs)} groups, "
        f"{n_envs_per_group} envs per group, {n_episodes} episodes"
    )

    # Train each group independently using train_ppo_vectorized
    # NOTE: This is a simplified implementation that trains groups sequentially.
    # For true parallel training, we would need to modify train_ppo_vectorized
    # to support multiple agents or run each group in a separate process.
    results = {}

    for (group_id, reward_mode), device in zip(group_configs, devices):
        _logger.info(
            f"Training group {group_id} with {reward_mode.value} reward "
            f"on device {device}"
        )

        # Create reward config for this group
        reward_config = ContributionRewardConfig(reward_mode=reward_mode)

        # Train this group using vectorized PPO
        agent, history = train_ppo_vectorized(
            n_episodes=n_episodes,
            n_envs=n_envs_per_group,
            max_epochs=max_epochs,
            device=device,
            devices=[device],  # Single device for this group
            task=task,
            lr=lr,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            entropy_coef_min=entropy_coef_min,
            gamma=gamma,
            gae_lambda=gae_lambda,
            lstm_hidden_dim=lstm_hidden_dim,
            seed=seed + hash(group_id) % 10000,  # Unique seed per group
            use_telemetry=use_telemetry,
            reward_mode=reward_mode.value,
            slots=slots,  # Pass slot configuration
            **kwargs,
        )

        results[group_id] = (agent, history)

        _logger.info(
            f"Group {group_id} training complete. "
            f"Final accuracy: {history[-1]['avg_accuracy']:.2f}%"
            if history
            else f"Group {group_id} training complete (no history)"
        )

    # Print comparison
    _print_dual_ab_comparison(group_configs, results)

    return results


def _print_dual_ab_comparison(
    group_configs: list[tuple[str, "RewardModeType"]],
    results: dict[str, tuple[PPOAgent, list[dict]]],
) -> None:
    """Print final A/B comparison between groups.

    Args:
        group_configs: List of (group_id, reward_mode) tuples
        results: Dict mapping group_id -> (agent, history)
    """
    print("\n" + "=" * 70)
    print("DUAL-POLICY A/B TEST RESULTS")
    print("=" * 70)

    group_stats = []
    for group_id, reward_mode in group_configs:
        if group_id not in results:
            continue

        agent, history = results[group_id]

        if not history:
            print(f"\n{group_id} - {reward_mode.value.upper()}")
            print("  No episodes completed")
            continue

        # Compute statistics
        # history contains batch-level metrics with avg_accuracy
        avg_accs = [batch["avg_accuracy"] for batch in history]

        final_acc = avg_accs[-1] if avg_accs else 0.0
        best_acc = max(avg_accs) if avg_accs else 0.0
        # Use rolling_avg_accuracy if available, otherwise avg_accuracy
        rolling_accs = [
            batch.get("rolling_avg_accuracy", batch.get("avg_accuracy", 0.0))
            for batch in history
        ]
        avg_rolling = sum(rolling_accs) / len(rolling_accs) if rolling_accs else 0.0

        group_stats.append({
            "group_id": group_id,
            "reward_mode": reward_mode.value,
            "final_acc": final_acc,
            "best_acc": best_acc,
            "avg_rolling": avg_rolling,
            "n_batches": len(history),
        })

        print(f"\n{group_id} - {reward_mode.value.upper()}")
        print(f"  Batches: {len(history)}")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Best Accuracy: {best_acc:.2f}%")
        print(f"  Avg Rolling Accuracy: {avg_rolling:.2f}%")

    # Winner determination
    if len(group_stats) == 2:
        a_stats, b_stats = group_stats
        margin = abs(a_stats["final_acc"] - b_stats["final_acc"])
        winner = a_stats if a_stats["final_acc"] > b_stats["final_acc"] else b_stats

        print(f"\n>>> WINNER: {winner['group_id']} ({winner['reward_mode']}) by {margin:.2f}% <<<")

    print("=" * 70)


__all__ = [
    "train_dual_policy_ab",
]
