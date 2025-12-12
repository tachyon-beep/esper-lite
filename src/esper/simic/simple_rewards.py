"""Simplified Reward Function for Multi-Slot Seed Control.

Design principle: The agent should LEARN behavior, not have it hard-coded.

Four signals only:
1. Contribution: did the seed help or hurt? (counterfactual)
2. Size: big = bad (logarithmic parameter penalty)
3. Budget pressure: discourage GERMINATE actions as budget fills (quadratic)
4. Terminal: final accuracy matters

That's it. ~40 lines. No hard gates, no ransomware detection, no PBRS farming
prevention, no ratio penalties, no legitimacy discounts. Let the agent learn.
"""

from __future__ import annotations

import math


def compute_simple_reward(
    seed_contributions: dict[str, float | None],
    total_params: int,
    host_params: int,
    is_terminal: bool,
    val_acc: float,
    seed_utilization: float = 0.0,
    action_is_germinate: bool = False,
    contribution_scale: float = 1.0,
    rent_scale: float = 0.1,
    terminal_acc_scale: float = 0.05,
    budget_pressure_scale: float = 0.1,
) -> float:
    """Compute reward with minimal shaping.

    Args:
        seed_contributions: Per-slot counterfactual deltas (real_acc - baseline_acc).
                           None means no counterfactual available for that slot.
        total_params: Total model params (host + all seeds)
        host_params: Baseline host params (for computing bloat ratio)
        is_terminal: Whether this is the final step
        val_acc: Current validation accuracy
        seed_utilization: Fraction of seed slots currently occupied (0.0 to 1.0)
        action_is_germinate: Whether the current action is GERMINATE (budget penalty only applies here)
        contribution_scale: Weight for contribution signal (default 1.0)
        rent_scale: Weight for parameter rent penalty (default 0.1)
        terminal_acc_scale: Weight for terminal accuracy bonus (default 0.05)
        budget_pressure_scale: Weight for budget pressure penalty (default 0.1)

    Returns:
        Reward value (unbounded, but typically in [-10, +10])
    """
    reward = 0.0

    # 1. Contribution: sum across all slots (help = positive, hurt = negative)
    for contrib in seed_contributions.values():
        if contrib is not None:
            reward += contribution_scale * contrib

    # 2. Size penalty: logarithmic so first params matter more than 10th million
    if total_params > 0 and host_params > 0:
        growth_ratio = total_params / host_params
        reward -= rent_scale * math.log(1.0 + growth_ratio)

    # 3. Budget pressure: quadratic penalty on GERMINATE actions as slots fill
    #    Early seeds cheap (0.1^2 = 0.01), later seeds expensive (0.9^2 = 0.81)
    #    Only penalize when GERMINATE is chosen - creates clear causal link
    if action_is_germinate:
        reward -= budget_pressure_scale * (seed_utilization ** 2)

    # 4. Terminal: final accuracy matters
    if is_terminal:
        reward += terminal_acc_scale * val_acc

    return reward


__all__ = ["compute_simple_reward"]
