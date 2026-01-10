"""Utilities for escrow reward tests.

These helpers keep the test cases extremely explicit: each test focuses on a
single escrow concept (credit target, delta, clawback, etc.) without drowning in
unrelated shaping terms.
"""

from __future__ import annotations

from dataclasses import replace

from esper.leyline import MIN_PRUNE_AGE, LifecycleOp, SeedStage
from esper.simic.rewards import ContributionRewardConfig, RewardMode, SeedInfo


def escrow_config(
    *,
    contribution_weight: float = 1.0,
    escrow_delta_clip: float = 0.0,
    disable_anti_gaming: bool = True,
) -> ContributionRewardConfig:
    """Return a config that isolates escrow mechanics.

    We explicitly zero-out or disable shaping terms so `reward == bounded_attribution`
    for the scenarios these tests cover.
    """
    return ContributionRewardConfig(
        reward_mode=RewardMode.ESCROW,
        contribution_weight=contribution_weight,
        escrow_delta_clip=escrow_delta_clip,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=disable_anti_gaming,
        # Remove unrelated shaping pressure.
        rent_weight=0.0,
        alpha_shock_coef=0.0,
        germinate_cost=0.0,
        fossilize_cost=0.0,
        prune_cost=0.0,
        set_alpha_target_cost=0.0,
        germinate_with_seed_penalty=0.0,
        invalid_fossilize_penalty=0.0,
        prune_fossilized_penalty=0.0,
        fossilize_base_bonus=0.0,
        fossilize_contribution_scale=0.0,
        fossilize_noncontributing_penalty=0.0,
        prune_hurting_bonus=0.0,
        prune_acceptable_bonus=0.0,
        prune_good_seed_penalty=0.0,
        early_prune_threshold=0,
        early_prune_penalty=0.0,
    )


def seed_info(
    *,
    stage: SeedStage = SeedStage.TRAINING,
    total_improvement: float = 0.0,
    improvement_since_stage_start: float = 0.0,
    epochs_in_stage: int = 1,
    seed_age_epochs: int = MIN_PRUNE_AGE,
    interaction_sum: float = 0.0,
    boost_received: float = 0.0,
) -> SeedInfo:
    """Convenience constructor with safe defaults for escrow tests."""
    return SeedInfo(
        stage=stage.value,
        improvement_since_stage_start=improvement_since_stage_start,
        total_improvement=total_improvement,
        epochs_in_stage=epochs_in_stage,
        seed_params=0,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=0,
        seed_age_epochs=seed_age_epochs,
        interaction_sum=interaction_sum,
        boost_received=boost_received,
    )


def stable_val_acc_from_history(acc_history: list[float], window: int) -> float:
    """Mirror the stable accuracy computation used in vectorized training."""
    if window <= 0:
        raise ValueError(f"escrow_stable_window must be positive, got {window}")
    if not acc_history:
        raise RuntimeError(
            "ESCROW stable accuracy requested before any accuracy history exists"
        )
    k = window if window <= len(acc_history) else len(acc_history)
    return min(acc_history[-k:])


def apply_terminal_escrow_forfeit(
    *,
    reward: float,
    credits_by_slot: dict[str, float],
    is_fossilized_by_slot: dict[str, bool] | None = None,
    stage_by_slot: dict[str, SeedStage] | None = None,
) -> tuple[float, float]:
    """Apply the vectorized training terminal clawback.

    Returns:
        (reward_after_forfeit, escrow_forfeit_component)

    Args:
        credits_by_slot: Escrow credit balance per slot.
        is_fossilized_by_slot: Legacy API - True if slot is fossilized. Use stage_by_slot
            for more precise control.
        stage_by_slot: Preferred API - SeedStage per slot. FOSSILIZED and PRUNED slots
            are excluded from forfeit (FOSSILIZED earned it, PRUNED already clawed back).

    Note:
        In production, this happens at `epoch == max_epochs` and is recorded as:
        - `reward -= sum(non_terminal_credits)`
        - `components.escrow_forfeit = -sum(non_terminal_credits)`
    """
    if stage_by_slot is None and is_fossilized_by_slot is None:
        raise ValueError("Must provide either stage_by_slot or is_fossilized_by_slot")

    # Terminal stages that don't forfeit escrow:
    # - FOSSILIZED: Successfully integrated, escrow is earned
    # - PRUNED: Already ordered removal, escrow was clawed back at prune time
    terminal_stages = {SeedStage.FOSSILIZED, SeedStage.PRUNED}

    escrow_forfeit = 0.0
    for slot_id, credit in credits_by_slot.items():
        if stage_by_slot is not None:
            if stage_by_slot[slot_id] in terminal_stages:
                continue
        elif is_fossilized_by_slot is not None:
            # Legacy path: only checks fossilized (doesn't know about PRUNED)
            if is_fossilized_by_slot[slot_id]:
                continue
        escrow_forfeit += credit
    if escrow_forfeit == 0.0:
        return reward, 0.0
    return reward - escrow_forfeit, -escrow_forfeit


def with_prune_good_seed_penalty(
    config: ContributionRewardConfig, *, prune_good_seed_penalty: float
) -> ContributionRewardConfig:
    """Return a copy of config with a different prune_good_seed_penalty."""
    return replace(config, prune_good_seed_penalty=prune_good_seed_penalty)


__all__ = [
    "LifecycleOp",
    "RewardMode",
    "SeedStage",
    "apply_terminal_escrow_forfeit",
    "escrow_config",
    "seed_info",
    "stable_val_acc_from_history",
    "with_prune_good_seed_penalty",
]

