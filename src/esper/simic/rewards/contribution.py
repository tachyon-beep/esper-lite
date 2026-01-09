"""Contribution-primary reward computation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    LifecycleOp,
    MIN_HOLDING_EPOCHS,
    MIN_PRUNE_AGE,
    SeedStage,
)
from esper.nissa import get_hub
from .reward_telemetry import RewardComponentsTelemetry
from .shaping import STAGE_POTENTIALS
from .types import (
    SeedInfo,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
    STAGE_HOLDING,
)

_logger = logging.getLogger(__name__)


class RewardMode(Enum):
    """Reward function variant for experimentation.

    SHAPED: Current dense shaping with PBRS, attribution, warnings (default)
    ESCROW: Dense, reversible attribution (anti-peak / anti-thrash)
    BASIC: Accuracy improvement minus parameter rent (minimal, no lifecycle shaping)
    SPARSE: Terminal-only ground truth (accuracy - param_cost)
    MINIMAL: Sparse + early-prune penalty only
    SIMPLIFIED: DRL Expert recommended - PBRS + intervention cost + terminal only
    """

    SHAPED = "shaped"
    ESCROW = "escrow"
    BASIC = "basic"
    SPARSE = "sparse"
    MINIMAL = "minimal"
    SIMPLIFIED = "simplified"


@dataclass
class ContributionRewardConfig:
    """Configuration for contribution-primary reward computation.

    Uses counterfactual validation (seed_contribution) as the primary signal,
    eliminating heuristics that conflate host drift with seed impact.

    This is the recommended reward function when counterfactual validation
    is enabled in vectorized training.

    Note:
        `proxy_contribution_weight` is derived from `contribution_weight * proxy_confidence_factor`.
        This ensures the proxy signal automatically scales with contribution_weight changes.
    """

    # Primary signal: seed contribution weight
    # Reduced from 3.0 to 1.0 for stable PPO (per-step rewards should be in [-10, +10])
    contribution_weight: float = 1.0

    # M2: Proxy confidence factor - how trustworthy is proxy vs counterfactual signal?
    # 0.3 means "proxy is 30% as reliable as true counterfactual"
    # proxy_contribution_weight is derived as: contribution_weight * proxy_confidence_factor
    proxy_confidence_factor: float = 0.3

    @property
    def proxy_contribution_weight(self) -> float:
        """Derived proxy weight: contribution_weight * proxy_confidence_factor."""
        return self.contribution_weight * self.proxy_confidence_factor

    # === Escrow Attribution (RewardMode.ESCROW) ===
    # Soft escrow: pay the CHANGE in an "unrealised credit target" so transient spikes are clawed back.
    # Stable accuracy uses min(last_k) to require sustained improvement ("prove it held").
    escrow_stable_window: int = 3
    # Optional per-step cap on escrow delta (0 disables). Useful if reward spikes destabilize PPO.
    escrow_delta_clip: float = 0.0

    # PBRS stage progression
    pbrs_weight: float = 0.3
    epoch_progress_bonus: float = 0.3
    max_progress_bonus: float = 2.0

    # Compute rent (logarithmic scaling, per-step)
    # rent = min(rent_weight * log(1 + growth_ratio), max_rent)
    # With growth_ratio=2.5 (typical): rent = 0.5 * log(3.5) ≈ 0.63 per step
    rent_weight: float = 0.5
    # Per-step cap (previously 8.0 for per-episode, now scaled for per-step use)
    # Limits rent to ~80% of avg attribution to prevent crushing small improvements
    max_rent: float = 1.5
    # Floor for host_params normalization in rent/shock calculations.
    # Tiny hosts (e.g., 17 trainable params) would otherwise be crushed by any non-trivial seed.
    rent_host_params_floor: int = 200
    # Alpha-weighted rent floor (BaseSlotRent): ratio of host params per occupied slot.
    # Calibrated from telemetry_2025-12-20_044944: ~0.0039.
    base_slot_rent_ratio: float = 0.0039
    # Convex alpha shock coefficient (penalizes fast alpha changes).
    # Calibrated from telemetry_2025-12-20_044944: ~0.1958.
    alpha_shock_coef: float = 0.1958
    # Maximum alpha shock penalty (prevents reward variance explosion).
    # Without this cap, rapid alpha oscillations can produce unbounded negative rewards
    # (e.g., 10 oscillations of Δ=1.0 → -1.958, 100 → -19.58), destabilizing PPO.
    alpha_shock_cap: float = 1.0

    # Enforcement penalties (state machine compliance)
    invalid_fossilize_penalty: float = -1.0
    prune_fossilized_penalty: float = -1.0
    germinate_with_seed_penalty: float = -0.3

    # Intervention costs (action friction)
    germinate_cost: float = -0.05  # Increased from -0.02 (C3 fix: anti-farming rebalance)
    fossilize_cost: float = -0.01
    prune_cost: float = -0.005
    set_alpha_target_cost: float = -0.005

    # Fossilize shaping
    fossilize_base_bonus: float = 0.5
    fossilize_contribution_scale: float = 0.1
    fossilize_noncontributing_penalty: float = -0.2

    # Prune shaping
    prune_hurting_bonus: float = 0.15  # Reduced from 0.3 (C3 fix: anti-farming rebalance)
    prune_acceptable_bonus: float = 0.1
    prune_good_seed_penalty: float = -0.3
    prune_hurting_threshold: float = -0.5
    # Minimum age (epochs) before a seed can receive prune_hurting_bonus.
    # Prevents germinate→hurt→prune farming where agent creates harmful seeds
    # just to get the bonus for removing them. (C3 fix: age gate)
    min_prune_bonus_age: int = 3

    # Anti-gaming: attribution discount and ratio penalty thresholds
    # Prevents seeds from gaming counterfactual by creating dependencies
    # - improvement_safe_threshold: below this, high contribution is suspicious
    # - hacking_ratio_threshold: contribution/improvement ratio triggering penalty
    # - attribution_sigmoid_steepness: controls discount curve for regressing seeds
    #   Lower values are more forgiving of normal training variance (±0.1-0.3%)
    #   steepness=10: -0.1% regression → 27% credit (too aggressive)
    #   steepness=3:  -0.1% regression → 43% credit, -0.5% → 18% (balanced)
    improvement_safe_threshold: float = 0.1
    hacking_ratio_threshold: float = 5.0
    attribution_sigmoid_steepness: float = 3.0

    # Terminal bonus
    terminal_acc_weight: float = 0.05
    # Terminal bonus per fossilized seed (incentivizes completion over farming)
    # DRL Expert review 2025-12-10: set to 3.0 to compete with post-scale attribution
    fossilize_terminal_scale: float = 3.0

    # Gamma for PBRS (uses module constant for consistency)
    gamma: float = DEFAULT_GAMMA

    # === BASIC Mode Parameters ===
    # BASIC mode uses only two signals:
    # - Accuracy improvement (acc_delta) scaled into a stable range
    # - Parameter rent scaled by param_budget
    basic_acc_delta_weight: float = 5.0

    # === Experiment Mode ===
    reward_mode: RewardMode = RewardMode.SHAPED

    # === Ablation Flags ===
    # Used for systematic reward function experiments.
    # These disable specific reward components to measure their contribution.
    disable_pbrs: bool = False  # Disable PBRS stage advancement shaping
    disable_terminal_reward: bool = False  # Disable terminal accuracy bonus
    disable_anti_gaming: bool = False  # Disable ratio_penalty and alpha_shock

    # === Sparse Reward Parameters ===
    # Parameter budget for efficiency calculation (sparse/minimal modes)
    param_budget: int = 500_000
    # Weight for parameter penalty in sparse reward
    param_penalty_weight: float = 0.1
    # Sparse reward scale (DRL Expert recommended if learning fails)
    sparse_reward_scale: float = 1.0

    # === Minimal Reward Parameters ===
    # Early-prune threshold (epochs before pruning is penalized)
    early_prune_threshold: int = 5
    # Penalty for pruning too early (discourages degenerate prune-all policies)
    early_prune_penalty: float = -0.1

    # === Action shaping weights ===
    # Special-case penalties for actions that violate lifecycle constraints
    advance_from_training_penalty: float = -0.1

    # === Auto-prune Penalty (degenerate policy prevention) ===
    # Penalty applied when environment auto-prunes a seed (safety or timeout)
    # instead of the policy explicitly choosing to prune.
    # (DRL Expert review 2025-12-17: prevents WAIT-spam policies that rely on
    # environment cleanup rather than learning proactive lifecycle management)
    auto_prune_penalty: float = -0.2

    # === D2: Capacity Economics (slot saturation prevention) ===
    # Threshold-based rent discourages early slot saturation and encourages
    # efficient use of capacity. First N slots are "free" (no occupancy rent),
    # excess slots incur per-epoch cost.
    #
    # DRL Expert Review 2025-01-08:
    # - seed_occupancy_cost: Per-epoch cost per seed above free_slots threshold
    # - free_slots: First N slots incur no occupancy rent (encourages some activity)
    # - fossilized_maintenance_cost: Per-epoch cost per fossilized seed (they still consume capacity)
    # - first_germinate_bonus: One-time bonus for first germination (breaks "do nothing" symmetry)
    seed_occupancy_cost: float = 0.01
    free_slots: int = 1
    fossilized_maintenance_cost: float = 0.002
    first_germinate_bonus: float = 0.2

    # === D3: Anti-Timing-Gaming (early germination discount) ===
    # Seeds germinated before warmup period receive discounted attribution.
    # This prevents "germinate early to claim host drift" gaming pattern.
    # Linear discount: epoch 1 = discount_floor, epoch warmup = 1.0
    germination_warmup_epochs: int = 10
    germination_discount_floor: float = 0.4
    disable_timing_discount: bool = False

    # === D3: Attribution formula variant ===
    # Controls how progress and seed_contribution combine into attributed value.
    # - "geometric": sqrt(progress * contribution) - current default, rewards host drift
    # - "harmonic": 2*p*c/(p+c) - dominated by smaller value, conservative
    # - "minimum": min(progress, contribution) - very conservative
    attribution_formula: Literal["geometric", "harmonic", "minimum"] = "geometric"

    @staticmethod
    def default() -> "ContributionRewardConfig":
        """Return default configuration."""
        return ContributionRewardConfig()


# Default config singleton (avoid repeated allocations)
_DEFAULT_CONTRIBUTION_CONFIG = ContributionRewardConfig()

# =============================================================================
# Contribution-Primary Reward (uses counterfactual validation)
# =============================================================================
#
# P2-2 Design Decision: This function is intentionally large (~300 lines, 7 components)
# because it implements sophisticated reward engineering where each component addresses
# a specific failure mode:
#   1. Bounded attribution - prevents ransomware signatures (gaming via model corruption)
#   2. Blending warning - early signal for problematic seeds before they cause damage
#   3. Holding indecision - prevents farming WAIT actions for positive rewards
#   4. PBRS stage progression - potential-based shaping preserves optimal policy
#   5. Compute rent - logarithmic parameter cost prevents model bloat
#   6. Action shaping - state machine compliance penalties
#   7. Terminal bonus - episode completion incentive
#
# Splitting into smaller functions would obscure the reward design and make it
# harder to reason about component interactions. Property tests in
# tests/simic/properties/test_pbrs_properties.py verify PBRS guarantees hold.
# =============================================================================


def compute_contribution_reward(
    action: LifecycleOp,
    seed_contribution: float | None,
    val_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: ContributionRewardConfig | None = None,
    acc_at_germination: float | None = None,
    acc_delta: float | None = None,
    return_components: bool = False,
    num_fossilized_seeds: int = 0,
    num_contributing_fossilized: int = 0,
    slot_id: str | None = None,
    seed_id: str | None = None,
    effective_seed_params: float | None = None,
    alpha_delta_sq_sum: float = 0.0,
    stable_val_acc: float | None = None,
    escrow_credit_prev: float = 0.0,
    # D2: Capacity economics parameters
    n_active_seeds: int = 0,
    seeds_germinated_this_episode: int = 0,
) -> float | tuple[float, RewardComponentsTelemetry]:
    """Compute reward using bounded attribution (ransomware-resistant)."""
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    if config.gamma != DEFAULT_GAMMA:
        raise ValueError(
            f"PBRS gamma mismatch: config.gamma={config.gamma} != DEFAULT_GAMMA={DEFAULT_GAMMA}. "
            "This breaks policy invariance guarantees (Ng et al., 1999). Use DEFAULT_GAMMA from leyline."
        )

    components = RewardComponentsTelemetry() if return_components else None
    if components:
        components.action_name = action.name
        components.epoch = epoch
        components.seed_stage = seed_info.stage if seed_info else None
        components.val_acc = val_acc
        components.acc_at_germination = acc_at_germination
        components.base_acc_delta = acc_delta if acc_delta is not None else 0.0

    reward = 0.0

    bounded_attribution = 0.0
    progress = None
    escrow_credit_target = 0.0
    escrow_delta = 0.0
    timing_discount = 1.0  # D3: Default to full credit (no discount)

    seed_is_fossilized = seed_info is not None and seed_info.stage == STAGE_FOSSILIZED

    escrow_mode = config.reward_mode == RewardMode.ESCROW
    if escrow_mode and stable_val_acc is None:
        raise ValueError(
            "RewardMode.ESCROW requires stable_val_acc (provide via SignalTracker history)."
        )
    progress_acc = stable_val_acc if escrow_mode and stable_val_acc is not None else val_acc

    attribution_discount = 1.0
    ratio_penalty = 0.0
    if seed_contribution is not None and seed_info is not None:
        total_imp = seed_info.total_improvement

        if not escrow_mode:
            if total_imp < 0:
                exp_arg = min(-config.attribution_sigmoid_steepness * total_imp, 700.0)
                attribution_discount = 1.0 / (1.0 + math.exp(exp_arg))

        if seed_contribution > 1.0 and attribution_discount >= 0.5 and not config.disable_anti_gaming:
            safe_threshold = max(config.improvement_safe_threshold, 1e-8)
            if total_imp > safe_threshold:
                ratio = seed_contribution / total_imp
                if ratio > config.hacking_ratio_threshold:
                    ratio_penalty = -min(
                        -config.prune_good_seed_penalty,
                        0.1 * (ratio - config.hacking_ratio_threshold) / config.hacking_ratio_threshold,
                    )
            elif total_imp <= config.improvement_safe_threshold:
                ratio_penalty = config.prune_good_seed_penalty * min(1.0, seed_contribution / 10.0)

        if slot_id is not None and seed_id is not None:
            hub = get_hub()
            if total_imp > 0 and ratio_penalty != 0:
                _check_reward_hacking(
                    hub,
                    seed_contribution=seed_contribution,
                    total_improvement=total_imp,
                    hacking_ratio_threshold=config.hacking_ratio_threshold,
                    slot_id=slot_id,
                    seed_id=seed_id,
                )
            _check_ransomware_signature(
                hub,
                seed_contribution=seed_contribution,
                total_improvement=total_imp,
                slot_id=slot_id,
                seed_id=seed_id,
            )

    if acc_at_germination is not None:
        progress = progress_acc - acc_at_germination

    if escrow_mode:
        if not return_components:
            raise ValueError(
                "RewardMode.ESCROW requires return_components=True so escrow state can be updated."
            )
        if action == LifecycleOp.PRUNE:
            escrow_credit_target = 0.0
            escrow_delta = escrow_credit_target - escrow_credit_prev
            bounded_attribution = escrow_delta
        elif seed_is_fossilized:
            escrow_credit_target = escrow_credit_prev
            escrow_delta = 0.0
            bounded_attribution = 0.0
        elif seed_contribution is not None:
            if seed_contribution < 0:
                bounded_attribution = config.contribution_weight * seed_contribution
                escrow_credit_target = 0.0
                escrow_delta = escrow_credit_target - escrow_credit_prev
                bounded_attribution += escrow_delta
            else:
                attributed = 0.0
                if progress is None:
                    attributed = seed_contribution * 0.5
                elif progress > 0:
                    if seed_contribution >= progress:
                        attributed = math.sqrt(progress * seed_contribution)
                    else:
                        attributed = seed_contribution

                escrow_credit_target = max(
                    0.0,
                    (config.contribution_weight * attributed) + ratio_penalty,
                )
                escrow_delta = escrow_credit_target - escrow_credit_prev
                if config.escrow_delta_clip > 0:
                    escrow_delta = max(
                        -config.escrow_delta_clip,
                        min(config.escrow_delta_clip, escrow_delta),
                    )
                bounded_attribution = escrow_delta
    else:
        if seed_contribution is not None and not seed_is_fossilized:
            if seed_contribution < 0:
                bounded_attribution = config.contribution_weight * seed_contribution
            else:
                if progress is not None:
                    if progress <= 0:
                        attributed = 0.0
                    elif seed_contribution >= progress:
                        # Use configurable formula when contribution exceeds progress
                        attributed = _compute_attributed_value(
                            progress=progress,
                            seed_contribution=seed_contribution,
                            formula=config.attribution_formula,
                        )
                    else:
                        # contribution < progress: cap at contribution (unchanged)
                        attributed = seed_contribution
                else:
                    attributed = seed_contribution * 0.5

                attributed *= attribution_discount

                bounded_attribution = (config.contribution_weight * attributed) + ratio_penalty

                # D3: Apply timing discount for early germination
                if not config.disable_timing_discount and seed_info is not None:
                    germination_epoch = epoch - seed_info.seed_age_epochs
                    timing_discount = _compute_timing_discount(
                        germination_epoch=germination_epoch,
                        warmup_epochs=config.germination_warmup_epochs,
                        discount_floor=config.germination_discount_floor,
                    )
                    bounded_attribution *= timing_discount
        elif seed_info is not None and not seed_is_fossilized:
            if acc_delta is not None and acc_delta > 0:
                bounded_attribution = config.proxy_contribution_weight * acc_delta

    if action == LifecycleOp.FOSSILIZE and seed_info is not None:
        if seed_info.total_improvement < 0:
            bounded_attribution = 0.0

    if action == LifecycleOp.PRUNE and not escrow_mode:
        bounded_attribution = -bounded_attribution

    reward += bounded_attribution

    if components:
        components.seed_contribution = seed_contribution
        components.bounded_attribution = bounded_attribution
        components.progress_since_germination = progress
        components.stable_val_acc = stable_val_acc if escrow_mode else None
        components.attribution_discount = attribution_discount
        components.ratio_penalty = ratio_penalty
        components.escrow_credit_prev = escrow_credit_prev
        components.escrow_credit_target = escrow_credit_target
        components.escrow_delta = escrow_delta
        components.escrow_credit_next = escrow_credit_prev + escrow_delta
        components.timing_discount = timing_discount

    blending_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_BLENDING:
        total_imp = seed_info.total_improvement
        if total_imp < 0:
            escalation = min(seed_info.epochs_in_stage * 0.05, 0.3)
            blending_warning = -0.1 - escalation
            reward += blending_warning
    if components:
        components.blending_warning = blending_warning

    # === Holding indecision penalty ===
    # In HOLDING stage, penalize actions that don't resolve the decision.
    # Terminal actions (FOSSILIZE, PRUNE) are exempt - they commit to a decision.
    # Non-terminal actions (WAIT, SET_ALPHA_TARGET, etc.) incur penalty.
    #
    # Bug fix (2026-01-08): Previously only WAIT triggered this penalty, allowing
    # Tamiyo to "turntable" SET_ALPHA_TARGET to avoid penalty while collecting
    # dense positives. Now all non-terminal actions in HOLDING are penalized.
    holding_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_HOLDING:
        # Terminal actions that resolve HOLDING - exempt from penalty
        terminal_actions = (LifecycleOp.FOSSILIZE, LifecycleOp.PRUNE)
        if action not in terminal_actions:
            if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
                has_counterfactual = (
                    seed_info.total_improvement is not None
                    or seed_info.improvement_since_stage_start is not None
                )
                if has_counterfactual:
                    epochs_waiting = seed_info.epochs_in_stage - 1
                    base_penalty = 0.1
                    ramp_penalty = max(0, epochs_waiting - 1) * 0.05
                    per_epoch_penalty = min(base_penalty + ramp_penalty, 0.3)
                    holding_warning = -per_epoch_penalty
                    reward += holding_warning
    if components:
        components.holding_warning = holding_warning

    pbrs_bonus = 0.0
    if seed_info is not None and not config.disable_pbrs:
        pbrs_bonus = _contribution_pbrs_bonus(seed_info, config)
        reward += pbrs_bonus
    if components:
        components.pbrs_bonus = pbrs_bonus

    synergy_bonus = 0.0
    if seed_info is not None and attribution_discount >= 0.5 and bounded_attribution > 0:
        synergy_bonus = _compute_synergy_bonus(
            seed_info.interaction_sum,
        )
        reward += synergy_bonus
    if components:
        components.synergy_bonus = synergy_bonus

    rent_penalty = 0.0
    growth_ratio = 0.0
    if host_params > 0:
        if config.rent_host_params_floor < 1:
            raise ValueError(
                f"rent_host_params_floor must be >= 1 (got {config.rent_host_params_floor})"
            )
        effective_overhead = (
            effective_seed_params
            if effective_seed_params is not None
            else max(total_params - host_params, 0)
        )
        if effective_overhead > 0:
            denom = max(host_params, config.rent_host_params_floor)
            growth_ratio = effective_overhead / denom
            scaled_cost = math.log(1.0 + growth_ratio)
            # Per-step rent penalty (same scale as attribution for balanced signal)
            # Previously divided by max_epochs which created 150:1 asymmetry vs attribution
            rent_penalty = min(config.rent_weight * scaled_cost, config.max_rent)
            reward -= rent_penalty
    if components:
        components.compute_rent = -rent_penalty
        components.growth_ratio = growth_ratio

    alpha_shock = 0.0
    if alpha_delta_sq_sum > 0 and config.alpha_shock_coef != 0.0 and not config.disable_anti_gaming:
        raw_shock = -config.alpha_shock_coef * alpha_delta_sq_sum
        # Cap to prevent reward variance explosion (C2 fix: unbounded alpha_shock)
        alpha_shock = max(raw_shock, -config.alpha_shock_cap)
        reward += alpha_shock
    if components:
        components.alpha_shock = alpha_shock

    # === D2: Capacity Economics (slot saturation prevention) ===
    # Threshold-based occupancy rent: first N slots are free, excess incurs per-epoch cost.
    # This discourages early slot saturation and encourages efficient capacity use.
    #
    # IMPORTANT: Use n_occupied (active + fossilized) for threshold calculation.
    # ChatGPT Pro review 2025-01-08: Using only n_active made fossilizing an "escape hatch"
    # from occupancy cost (0.01 → 0.002). Since fossilized seeds still occupy slots, they
    # must count against the free_slots threshold until D3 (audit) creates proper incentives.
    occupancy_rent = 0.0
    fossilized_rent = 0.0
    first_germ_bonus = 0.0

    # Occupancy rent for occupied slots above free_slots threshold
    # n_occupied includes both active AND fossilized seeds (they all consume slots)
    n_occupied = n_active_seeds + num_fossilized_seeds
    excess_occupied = max(0, n_occupied - config.free_slots)
    if excess_occupied > 0:
        occupancy_rent = config.seed_occupancy_cost * excess_occupied
        reward -= occupancy_rent

    # Fossilized maintenance rent (additional small cost for frozen compute)
    if num_fossilized_seeds > 0:
        fossilized_rent = config.fossilized_maintenance_cost * num_fossilized_seeds
        reward -= fossilized_rent

    # First-germination bonus (breaks "do nothing" symmetry)
    # One-time bonus when agent takes first germination action
    if action == LifecycleOp.GERMINATE and seeds_germinated_this_episode == 0:
        first_germ_bonus = config.first_germinate_bonus
        reward += first_germ_bonus

    if components:
        components.occupancy_rent = occupancy_rent
        components.fossilized_rent = fossilized_rent
        components.first_germinate_bonus = first_germ_bonus
        components.n_active_seeds = n_active_seeds

    action_shaping = 0.0

    if action == LifecycleOp.GERMINATE:
        if seed_info is not None:
            action_shaping += config.germinate_with_seed_penalty
        elif not config.disable_pbrs:
            if epoch < max_epochs:
                phi_germinated = STAGE_POTENTIALS[SeedStage.GERMINATED]
                phi_no_seed = 0.0
                pbrs_germinate = config.gamma * phi_germinated - phi_no_seed
                action_shaping += config.pbrs_weight * pbrs_germinate
        action_shaping += config.germinate_cost

    elif action == LifecycleOp.FOSSILIZE:
        action_shaping += _contribution_fossilize_shaping(seed_info, seed_contribution, config)
        action_shaping += config.fossilize_cost

    elif action == LifecycleOp.PRUNE:
        if seed_info is not None and not config.disable_pbrs:
            phi_current = STAGE_POTENTIALS[SeedStage(seed_info.stage)]
            phi_current += min(
                seed_info.epochs_in_stage * config.epoch_progress_bonus,
                config.max_progress_bonus,
            )
            pbrs_cull = config.gamma * 0.0 - phi_current
            action_shaping += config.pbrs_weight * pbrs_cull
        action_shaping += _contribution_prune_shaping(seed_info, seed_contribution, config)
        action_shaping += config.prune_cost
    elif action == LifecycleOp.SET_ALPHA_TARGET:
        action_shaping += config.set_alpha_target_cost

    reward += action_shaping
    if components:
        components.action_shaping = action_shaping

    terminal_bonus = 0.0
    fossilize_terminal_bonus = 0.0
    if epoch == max_epochs and not config.disable_terminal_reward:
        terminal_bonus = val_acc * config.terminal_acc_weight
        fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
        terminal_bonus += fossilize_terminal_bonus
        reward += terminal_bonus
    if components:
        components.terminal_bonus = terminal_bonus
        components.fossilize_terminal_bonus = fossilize_terminal_bonus
        components.num_fossilized_seeds = num_fossilized_seeds
        components.num_contributing_fossilized = num_contributing_fossilized

    if components:
        components.total_reward = reward
        return reward, components
    return reward


def compute_sparse_reward(
    committed_val_acc: float,
    fossilized_seed_params: int,
    epoch: int,
    max_epochs: int,
    config: ContributionRewardConfig,
) -> float:
    """Compute sparse (terminal-only) reward."""
    if epoch != max_epochs:
        return 0.0

    if config.param_budget <= 0:
        raise ValueError("param_budget must be positive")
    accuracy_reward = committed_val_acc / 100.0
    param_cost = config.param_penalty_weight * (fossilized_seed_params / config.param_budget)

    base_reward = accuracy_reward - param_cost
    clamped_base = max(-1.0, min(1.0, base_reward))

    return config.sparse_reward_scale * clamped_base


def compute_minimal_reward(
    committed_val_acc: float,
    fossilized_seed_params: int,
    epoch: int,
    max_epochs: int,
    action: LifecycleOp,
    seed_age: int | None,
    config: ContributionRewardConfig,
) -> float:
    """Compute minimal reward (sparse + early-prune penalty)."""
    reward = compute_sparse_reward(
        committed_val_acc=committed_val_acc,
        fossilized_seed_params=fossilized_seed_params,
        epoch=epoch,
        max_epochs=max_epochs,
        config=config,
    )

    if action == LifecycleOp.PRUNE and seed_age is not None:
        if seed_age < config.early_prune_threshold:
            reward += config.early_prune_penalty

    return reward


def compute_basic_reward(
    *,
    acc_delta: float,
    effective_seed_params: float | None,
    total_params: int,
    host_params: int,
    config: ContributionRewardConfig,
) -> tuple[float, float, float]:
    """Compute BASIC reward: accuracy improvement minus parameter rent."""
    if config.param_budget <= 0:
        raise ValueError("param_budget must be positive")

    accuracy_improvement = config.basic_acc_delta_weight * (acc_delta / 100.0)

    effective_overhead = (
        effective_seed_params
        if effective_seed_params is not None
        else max(total_params - host_params, 0)
    )
    rent_penalty = config.param_penalty_weight * (effective_overhead / config.param_budget)

    growth_ratio = 0.0
    if host_params > 0:
        denom = max(host_params, config.rent_host_params_floor)
        growth_ratio = effective_overhead / denom

    reward = accuracy_improvement - rent_penalty
    return reward, rent_penalty, growth_ratio


def compute_simplified_reward(
    action: LifecycleOp,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    val_acc: float,
    num_contributing_fossilized: int,
    config: ContributionRewardConfig | None = None,
) -> float:
    """Compute simplified 3-component reward (DRL Expert recommended)."""
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    reward = 0.0

    if seed_info is not None and not config.disable_pbrs:
        reward += _contribution_pbrs_bonus(seed_info, config)

    if action != LifecycleOp.WAIT:
        reward -= 0.01

    if epoch == max_epochs and not config.disable_terminal_reward:
        accuracy_bonus = (val_acc / 100.0) * 3.0
        fossilize_bonus = num_contributing_fossilized * 2.0
        reward += accuracy_bonus + fossilize_bonus

    return reward


def _contribution_pbrs_bonus(
    seed_info: SeedInfo,
    config: ContributionRewardConfig,
) -> float:
    """Compute PBRS bonus for stage progression."""
    phi_current = STAGE_POTENTIALS[SeedStage(seed_info.stage)]
    phi_current += min(
        seed_info.epochs_in_stage * config.epoch_progress_bonus,
        config.max_progress_bonus,
    )

    if seed_info.epochs_in_stage == 0:
        if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
            _logger.warning(
                "PBRS telescoping risk: transition from stage %d with previous_epochs_in_stage=0. "
                "phi_prev will be underestimated. This indicates SeedInfo was constructed incorrectly.",
                seed_info.previous_stage,
            )
        phi_prev = STAGE_POTENTIALS[SeedStage(seed_info.previous_stage)]
        phi_prev += min(
            seed_info.previous_epochs_in_stage * config.epoch_progress_bonus,
            config.max_progress_bonus,
        )
    else:
        phi_prev = STAGE_POTENTIALS[SeedStage(seed_info.stage)]
        phi_prev += min(
            (seed_info.epochs_in_stage - 1) * config.epoch_progress_bonus,
            config.max_progress_bonus,
        )

    return config.pbrs_weight * (config.gamma * phi_current - phi_prev)


def _compute_synergy_bonus(
    interaction_sum: float,
    synergy_weight: float = 0.1,
) -> float:
    """Compute synergy bonus for scaffolding behavior."""
    if interaction_sum <= 0:
        return 0.0

    raw_bonus = math.tanh(interaction_sum * 0.5)
    return raw_bonus * synergy_weight


def compute_scaffold_hindsight_credit(
    boost_given: float,
    beneficiary_improvement: float,
    credit_weight: float = 0.2,
) -> float:
    """Compute retroactive credit for scaffold seeds."""
    if boost_given <= 0 or beneficiary_improvement <= 0:
        return 0.0

    raw_credit = math.tanh(boost_given * beneficiary_improvement * 0.1)
    return raw_credit * credit_weight


def _contribution_fossilize_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for FOSSILIZE action - with legitimacy and ransomware checks."""
    if seed_info is None:
        return config.invalid_fossilize_penalty

    if seed_info.stage != STAGE_HOLDING:
        return config.invalid_fossilize_penalty

    total_delta = seed_info.total_improvement
    if total_delta < 0:
        base_penalty = -0.5
        damage_scale = min(abs(total_delta) * 0.2, 1.0)

        ransomware_signature = (
            seed_contribution is not None
            and seed_contribution > 0.1
            and total_delta < -0.2
        )
        ransomware_penalty = -0.3 if ransomware_signature else 0.0

        return base_penalty - damage_scale + ransomware_penalty

    legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_HOLDING_EPOCHS)

    if seed_contribution is not None and seed_contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION:
        base_bonus = (
            config.fossilize_base_bonus
            + config.fossilize_contribution_scale * seed_contribution
        )
        return base_bonus * legitimacy_discount

    return config.fossilize_noncontributing_penalty


def _contribution_prune_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for PRUNE action - simplified with counterfactual."""
    if seed_info is None:
        return -0.2

    if seed_info.stage == STAGE_FOSSILIZED:
        return config.prune_fossilized_penalty

    if seed_info.seed_age_epochs < MIN_PRUNE_AGE:
        return -0.3 * (MIN_PRUNE_AGE - seed_info.seed_age_epochs)

    if seed_contribution is None and seed_info.seed_age_epochs < config.early_prune_threshold:
        return config.early_prune_penalty

    if seed_contribution is not None:
        if seed_contribution < config.prune_hurting_threshold:
            # C3 fix: Age gate prevents germinate→hurt→prune farming.
            # Only give bonus for pruning hurting seeds if they've existed long enough
            # to rule out intentional harm-seeding for reward farming.
            if seed_info.seed_age_epochs < config.min_prune_bonus_age:
                # Young hurting seed: small penalty (discourages quick-cycle farming)
                return -0.1
            return config.prune_hurting_bonus
        if seed_contribution < 0:
            return config.prune_acceptable_bonus
        base_penalty = config.prune_good_seed_penalty
        scaled_penalty = base_penalty - 0.05 * seed_contribution
        max_penalty = 3.0 * base_penalty
        return max(scaled_penalty, max_penalty)

    return 0.0


def _compute_timing_discount(
    germination_epoch: int,
    warmup_epochs: int,
    discount_floor: float,
) -> float:
    """Compute timing discount for early germination.

    Seeds germinated before warmup_epochs receive discounted attribution.
    Linear interpolation from discount_floor (epoch 0) to 1.0 (epoch >= warmup).

    Args:
        germination_epoch: Epoch when seed was germinated
        warmup_epochs: Number of epochs before full credit
        discount_floor: Minimum discount (applied at epoch 0)

    Returns:
        Discount factor in [discount_floor, 1.0]
    """
    if germination_epoch >= warmup_epochs:
        return 1.0

    # Linear interpolation: epoch 0 = floor, epoch warmup = 1.0
    progress = germination_epoch / warmup_epochs
    return discount_floor + (1.0 - discount_floor) * progress


def _compute_attributed_value(
    progress: float,
    seed_contribution: float,
    formula: Literal["geometric", "harmonic", "minimum"],
) -> float:
    """Compute attributed value using the specified formula.

    Args:
        progress: Accuracy improvement since germination (val_acc - acc_at_germination)
        seed_contribution: Counterfactual contribution of the seed
        formula: One of "geometric", "harmonic", "minimum"

    Returns:
        Attributed value combining progress and contribution

    Formulas:
        - geometric: sqrt(progress * contribution) - rewards host drift
        - harmonic: 2*p*c/(p+c) - dominated by smaller value, anti-gaming
        - minimum: min(progress, contribution) - very conservative
    """
    if progress <= 0 or seed_contribution <= 0:
        return 0.0

    if formula == "geometric":
        return math.sqrt(progress * seed_contribution)

    elif formula == "harmonic":
        # Harmonic mean: 2ab/(a+b), dominated by smaller value
        return 2 * progress * seed_contribution / (progress + seed_contribution)

    elif formula == "minimum":
        return min(progress, seed_contribution)

    else:
        raise ValueError(f"Unknown attribution formula: {formula}")


def _check_reward_hacking(
    hub: Any,
    *,
    seed_contribution: float,
    total_improvement: float,
    hacking_ratio_threshold: float = 5.0,
    slot_id: str,
    seed_id: str,
) -> bool:
    """Emit REWARD_HACKING_SUSPECTED if attribution ratio is anomalous."""
    from esper.leyline import RewardHackingSuspectedPayload, TelemetryEvent, TelemetryEventType

    if total_improvement <= 0 or seed_contribution <= 0:
        return False

    ratio = seed_contribution / total_improvement

    if ratio < hacking_ratio_threshold:
        return False

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.REWARD_HACKING_SUSPECTED,
        severity="warning",
        seed_id=seed_id,
        slot_id=slot_id,
        data=RewardHackingSuspectedPayload(
            pattern="attribution_ratio",
            ratio=ratio,
            seed_contribution=seed_contribution,
            total_improvement=total_improvement,
            threshold=hacking_ratio_threshold,
            slot_id=slot_id,
            seed_id=seed_id,
        ),
    ))
    return True


def _check_ransomware_signature(
    hub: Any,
    *,
    seed_contribution: float,
    total_improvement: float,
    contribution_threshold: float = 0.1,
    degradation_threshold: float = -0.2,
    slot_id: str,
    seed_id: str,
) -> bool:
    """Emit REWARD_HACKING_SUSPECTED if seed shows ransomware signature."""
    from esper.leyline import RewardHackingSuspectedPayload, TelemetryEvent, TelemetryEventType

    if seed_contribution <= contribution_threshold:
        return False

    if total_improvement >= degradation_threshold:
        return False

    severity = "critical" if seed_contribution >= 1.0 else "warning"
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.REWARD_HACKING_SUSPECTED,
        severity=severity,
        seed_id=seed_id,
        slot_id=slot_id,
        data=RewardHackingSuspectedPayload(
            pattern="ransomware_signature",
            seed_contribution=seed_contribution,
            total_improvement=total_improvement,
            contribution_threshold=contribution_threshold,
            degradation_threshold=degradation_threshold,
            slot_id=slot_id,
            seed_id=seed_id,
        ),
    ))
    return True


# M5: Derive from ContributionRewardConfig defaults to avoid value duplication.
# This ensures INTERVENTION_COSTS stays in sync with config defaults.
_default_config = ContributionRewardConfig()
INTERVENTION_COSTS: dict[LifecycleOp, float] = {
    LifecycleOp.WAIT: 0.0,
    LifecycleOp.GERMINATE: _default_config.germinate_cost,
    LifecycleOp.FOSSILIZE: _default_config.fossilize_cost,
    LifecycleOp.PRUNE: _default_config.prune_cost,
    LifecycleOp.SET_ALPHA_TARGET: _default_config.set_alpha_target_cost,
    LifecycleOp.ADVANCE: 0.0,
}
del _default_config  # Don't pollute module namespace


def get_intervention_cost(action: LifecycleOp) -> float:
    """Get intervention cost for an action using default config values."""
    return INTERVENTION_COSTS.get(action, 0.0)


__all__ = [
    "RewardMode",
    "ContributionRewardConfig",
    "compute_contribution_reward",
    "compute_sparse_reward",
    "compute_minimal_reward",
    "compute_basic_reward",
    "compute_simplified_reward",
    "compute_scaffold_hindsight_credit",
    "_contribution_pbrs_bonus",
    "_contribution_prune_shaping",
    "_contribution_fossilize_shaping",
    "_compute_timing_discount",
    "_compute_attributed_value",
    "_check_reward_hacking",
    "_check_ransomware_signature",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
]
