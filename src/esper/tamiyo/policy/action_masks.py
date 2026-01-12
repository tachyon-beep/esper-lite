"""Action Masking for Multi-Slot Control.

Only masks actions that are invalid under lifecycle constraints:
- SLOT: only enabled slots (from --slots arg) are selectable
- SLOT_BY_OP: op-conditioned slot validity mask (hierarchical sampling support)
- GERMINATE: blocked if ALL enabled slots occupied OR at seed limit OR
             D3: any seed in GERMINATED/TRAINING/PRUNED/EMBARGOED/RESETTING stages
             (enforces sequential development, allows scaffolding during BLENDING)
- ADVANCE: blocked if NO enabled slot is in GERMINATED/TRAINING/BLENDING, or if disabled by config
- FOSSILIZE: blocked if NO enabled slot has a HOLDING seed
- PRUNE: blocked if NO enabled slot has a prunable seed in GERMINATED/TRAINING/BLENDING/HOLDING
         with age >= MIN_PRUNE_AGE and alpha_mode == HOLD (unless governor override)
- WAIT: always valid
- BLUEPRINT: NOOP always blocked (0 trainable parameters)

Does NOT mask timing heuristics (epoch, plateau, stabilization).
Tamiyo learns optimal timing from counterfactual reward signals.

Multi-slot execution: The sampled slot determines which slot is targeted.
The op mask is computed optimistically (valid if ANY enabled slot allows it).
Invalid slot+op combinations are rejected at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from esper.leyline import (
    AlphaMode,
    AlphaTargetAction,
    BlueprintAction,
    CNN_BLUEPRINTS,
    GerminationStyle,
    LifecycleOp,
    MASKED_LOGIT_VALUE,
    MIN_PRUNE_AGE,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    SeedStage,
    Topology,
    TRANSFORMER_BLUEPRINTS,
    VALID_TOPOLOGIES,
)
from esper.leyline.stages import VALID_TRANSITIONS
from esper.leyline.slot_config import SlotConfig

if TYPE_CHECKING:
    from esper.leyline import SeedStateReport

# Stage sets for validation - derived from VALID_TRANSITIONS (single source of truth)
# Stages from which FOSSILIZED is a valid transition
_FOSSILIZABLE_STAGES = frozenset({
    stage.value for stage, transitions in VALID_TRANSITIONS.items()
    if SeedStage.FOSSILIZED in transitions
})

# Stages from which PRUNED is a valid transition
_PRUNABLE_STAGES = frozenset({
    stage.value
    for stage, transitions in VALID_TRANSITIONS.items()
    if SeedStage.PRUNED in transitions
})

# Stages from which ADVANCE is meaningful (explicit policy decision)
_ADVANCABLE_STAGES = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
})

# D3: Stages that block GERMINATE - enforces sequential seed development
# When any seed is in these stages, GERMINATE is masked out to prevent
# PBRS farming (collecting +0.25 germination bonuses repeatedly).
# Note: BLENDING explicitly excluded to enable scaffolding pattern
# (old seed blending while new seed trains).
_BLOCKS_GERMINATION = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
})

# D3: Cleanup stages that also block GERMINATE to close PRUNE→GERMINATE exploit
# Without this, agent could: GERMINATE (+0.25) → WAIT (MIN_PRUNE_AGE) → PRUNE → repeat
# By blocking until slot fully resets to DORMANT, we force genuine sequential development.
_CLEANUP_BLOCKS_GERMINATION = frozenset({
    SeedStage.PRUNED.value,
    SeedStage.EMBARGOED.value,
    SeedStage.RESETTING.value,
})


@dataclass(frozen=True, slots=True)
class MaskSeedInfo:
    """Minimal seed info for action masking only.

    Uses int for stage (not enum) for torch.compile safety.
    """

    stage: int  # SeedStage.value
    seed_age_epochs: int
    alpha_mode: int = AlphaMode.HOLD.value


def build_slot_states(
    slot_reports: dict[str, "SeedStateReport"],
    slots: list[str],
) -> dict[str, MaskSeedInfo | None]:
    """Build slot_states dict for action masking from slot reports.

    This function ensures all slots have entries, setting None for empty slots.
    The resulting dict is suitable for compute_action_masks() which requires
    all enabled slots to be present as keys.

    Args:
        slot_reports: Slot -> SeedStateReport for slots where `SeedSlot.state is not None`.
            This includes non-active lifecycle states (e.g. PRUNED/EMBARGOED/RESETTING) where
            the underlying seed module may already be freed; missing keys indicate an empty
            slot (`state is None`).
        slots: List of slot IDs to include in output (may include empty slots)

    Returns:
        Dict mapping slot_id to MaskSeedInfo or None if slot is empty (`state is None`).
        Guaranteed to have an entry for every slot_id in slots.
    """
    slot_states: dict[str, MaskSeedInfo | None] = {}
    for slot_id in slots:
        report = slot_reports.get(slot_id)
        if report is None:
            slot_states[slot_id] = None
        else:
            slot_states[slot_id] = MaskSeedInfo(
                stage=report.stage.value,
                seed_age_epochs=report.metrics.epochs_total,
                alpha_mode=report.alpha_mode,
            )
    return slot_states


def compute_action_masks(
    slot_states: dict[str, MaskSeedInfo | None],
    enabled_slots: list[str],
    total_seeds: int = 0,
    max_seeds: int = 0,
    slot_config: SlotConfig | None = None,
    device: torch.device | None = None,
    topology: Topology = "cnn",
    allow_governor_override: bool = False,
    disable_advance: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute action masks based on slot states.

    Only masks PHYSICALLY IMPOSSIBLE actions. Does not mask timing heuristics.

    Args:
        slot_states: Dict mapping slot_id to MaskSeedInfo or None
        enabled_slots: List of slot IDs that are enabled (from --slots arg)
        total_seeds: Total number of active seeds across all slots
        max_seeds: Maximum allowed seeds (0 = unlimited)
        slot_config: Slot configuration (defaults to SlotConfig.default())
        device: Torch device for tensors
        topology: Task topology ("cnn" or "transformer") for blueprint masking
        allow_governor_override: Allow PRUNE even if alpha_mode != HOLD
        disable_advance: If True, mask out ADVANCE regardless of seed stages.

    Returns:
        Dict of boolean tensors for each action head:
        - "slot": [num_slots] - which slots can be targeted (only enabled slots)
        - "slot_by_op": [NUM_OPS, num_slots] - op-conditioned slot validity mask
        - "blueprint": [NUM_BLUEPRINTS] - which blueprints can be used
        - "style": [NUM_STYLES] - which germination styles can be used
        - "tempo": [NUM_TEMPO] - which tempo values can be used (all valid)
        - "alpha_target": [NUM_ALPHA_TARGETS] - which alpha targets can be used
        - "alpha_speed": [NUM_ALPHA_SPEEDS] - which alpha speeds can be used
        - "alpha_curve": [NUM_ALPHA_CURVES] - which alpha curves can be used
        - "op": [NUM_OPS] - which operations are valid (ANY enabled slot)

    Raises:
        ValueError: If topology is not "cnn" or "transformer"
        ValueError: If enabled_slots is empty
        ValueError: If enabled_slots contains unknown slot IDs
    """
    # === Input Validation (fail fast on misconfiguration) ===
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(
            f"Unknown topology: {topology!r}. Valid topologies: {sorted(VALID_TOPOLOGIES)}"
        )

    if not enabled_slots:
        raise ValueError("enabled_slots cannot be empty")

    if slot_config is None:
        slot_config = SlotConfig.default()

    # Validate enabled_slots against slot_config
    enabled_set = set(enabled_slots)
    valid_slots = set(slot_config.slot_ids)
    unknown_slots = enabled_set - valid_slots
    if unknown_slots:
        raise ValueError(
            f"enabled_slots contains unknown slot IDs: {sorted(unknown_slots)}. "
            f"Valid slot IDs: {slot_config.slot_ids}"
        )

    device = device if device is not None else torch.device("cpu")

    # Order enabled slots according to slot_config order
    ordered = tuple(slot_id for slot_id in slot_config.slot_ids if slot_id in enabled_set)

    # Validate slot_states contains all enabled slots (fail-fast on misconfiguration)
    # This prevents .get() from silently treating missing keys as empty slots
    missing_slots = set(ordered) - slot_states.keys()
    if missing_slots:
        raise ValueError(
            f"slot_states missing entries for enabled slots: {sorted(missing_slots)}. "
            f"slot_states keys: {sorted(slot_states.keys())}. "
            f"Use build_slot_states() to ensure all enabled slots have entries."
        )

    # Slot mask: only enabled slots are selectable in canonical order
    slot_mask = torch.zeros(slot_config.num_slots, dtype=torch.bool, device=device)
    for slot_id in ordered:
        idx = slot_config.index_for_slot_id(slot_id)
        slot_mask[idx] = True

    # Op-conditioned slot mask (hierarchical validity)
    # Shape: [NUM_OPS, num_slots]; each op selects from a restricted set of slots.
    slot_by_op = torch.zeros((NUM_OPS, slot_config.num_slots), dtype=torch.bool, device=device)
    # WAIT: slot is irrelevant; allow all enabled slots (network canonicalizes later)
    slot_by_op[LifecycleOp.WAIT] = slot_mask

    # Blueprint mask: only allow blueprints valid for this topology
    blueprint_mask = torch.zeros(NUM_BLUEPRINTS, dtype=torch.bool, device=device)
    
    valid_blueprints = TRANSFORMER_BLUEPRINTS if topology == "transformer" else CNN_BLUEPRINTS
    for bp in valid_blueprints:
        # NOOP is technically in the sets but we force it masked out anyway below
        blueprint_mask[bp] = True

    # NOOP is a placeholder seed with no trainable parameters - always disable
    blueprint_mask[BlueprintAction.NOOP] = False

    # Style mask: relevant when GERMINATE is possible, or when HOLD retargeting is possible
    # (SET_ALPHA_TARGET can change alpha_algorithm). Default to SIGMOID_ADD when irrelevant
    # so the head remains well-defined.
    style_mask = torch.zeros(NUM_STYLES, dtype=torch.bool, device=device)
    style_mask[GerminationStyle.SIGMOID_ADD] = True

    # Tempo mask: all values valid. Gradient noise from irrelevant sampling is handled
    # by causal masking in PPO (leyline/causal_masks.py) which zeros gradients when
    # tempo doesn't affect execution (non-GERMINATE ops).
    tempo_mask = torch.ones(NUM_TEMPO, dtype=torch.bool, device=device)

    # Alpha heads: target changes are HOLD-only or germinate.
    # alpha_speed and alpha_curve are all-True for same reason as tempo: causal masks
    # zero their gradients when SET_ALPHA_TARGET is not the selected op.
    alpha_target_mask = torch.zeros(NUM_ALPHA_TARGETS, dtype=torch.bool, device=device)
    alpha_speed_mask = torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device)
    alpha_curve_mask = torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool, device=device)

    # Op mask: depends on slot states across ALL enabled slots
    op_mask = torch.zeros(NUM_OPS, dtype=torch.bool, device=device)
    op_mask[LifecycleOp.WAIT] = True  # WAIT always valid

    # Check slot states for enabled slots only
    # Direct access is safe here - we validated all keys exist above
    has_empty_enabled_slot = any(
        slot_states[slot_id] is None
        for slot_id in ordered
    )

    seed_limit_reached = max_seeds > 0 and total_seeds >= max_seeds

    # D3: Check for seeds in active development or cleanup stages
    # This blocks GERMINATE until the current seed lifecycle completes, enforcing
    # sequential development and preventing PBRS germination farming.
    has_developing_seed = any(
        seed_info is not None and (
            seed_info.stage in _BLOCKS_GERMINATION
            or seed_info.stage in _CLEANUP_BLOCKS_GERMINATION
        )
        for slot_id in ordered
        for seed_info in [slot_states[slot_id]]
    )

    can_germinate = (
        has_empty_enabled_slot
        and not seed_limit_reached
        and not has_developing_seed
    )

    # GERMINATE: valid if ANY enabled slot is empty AND under seed limit AND no developing seed
    if can_germinate:
        op_mask[LifecycleOp.GERMINATE] = True
        for slot_id in ordered:
            if slot_states[slot_id] is None:
                idx = slot_config.index_for_slot_id(slot_id)
                slot_by_op[LifecycleOp.GERMINATE, idx] = True

    # ADVANCE/FOSSILIZE/PRUNE: valid if ANY enabled slot has a valid state
    # (optimistic masking - network learns slot+op associations)
    has_retargetable_hold_slot = False
    for slot_id in ordered:
        seed_info = slot_states[slot_id]  # Safe - validated above
        if seed_info is not None:
            stage = seed_info.stage
            age = seed_info.seed_age_epochs

            # ADVANCE: only from explicit policy-controlled stages
            if not disable_advance and stage in _ADVANCABLE_STAGES:
                op_mask[LifecycleOp.ADVANCE] = True
                idx = slot_config.index_for_slot_id(slot_id)
                slot_by_op[LifecycleOp.ADVANCE, idx] = True

            # FOSSILIZE: only from HOLDING
            if stage in _FOSSILIZABLE_STAGES:
                op_mask[LifecycleOp.FOSSILIZE] = True
                idx = slot_config.index_for_slot_id(slot_id)
                slot_by_op[LifecycleOp.FOSSILIZE, idx] = True

            # PRUNE: only from prunable stages, seed age >= MIN_PRUNE_AGE, and HOLD-only
            if stage in _PRUNABLE_STAGES and age >= MIN_PRUNE_AGE:
                if allow_governor_override or seed_info.alpha_mode == AlphaMode.HOLD.value:
                    op_mask[LifecycleOp.PRUNE] = True
                    idx = slot_config.index_for_slot_id(slot_id)
                    slot_by_op[LifecycleOp.PRUNE, idx] = True
            # SET_ALPHA_TARGET: HOLD-only, only when a seed is in a retargetable stage.
            if stage in (SeedStage.BLENDING.value, SeedStage.HOLDING.value):
                if seed_info.alpha_mode == AlphaMode.HOLD.value:
                    op_mask[LifecycleOp.SET_ALPHA_TARGET] = True
                    has_retargetable_hold_slot = True
                    idx = slot_config.index_for_slot_id(slot_id)
                    slot_by_op[LifecycleOp.SET_ALPHA_TARGET, idx] = True

    if can_germinate or has_retargetable_hold_slot:
        style_mask[:] = True
        alpha_target_mask[:] = True
    else:
        alpha_target_mask[AlphaTargetAction.FULL] = True

    return {
        "slot": slot_mask,
        "slot_by_op": slot_by_op,
        "blueprint": blueprint_mask,
        "style": style_mask,
        "tempo": tempo_mask,
        "alpha_target": alpha_target_mask,
        "alpha_speed": alpha_speed_mask,
        "alpha_curve": alpha_curve_mask,
        "op": op_mask,
    }


def compute_batch_masks(
    batch_slot_states: list[dict[str, MaskSeedInfo | None]],
    enabled_slots: list[str],
    total_seeds_list: list[int] | None = None,
    max_seeds: int = 0,
    slot_config: SlotConfig | None = None,
    device: torch.device | None = None,
    topology: Topology = "cnn",
    allow_governor_override: bool = False,
    disable_advance: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute action masks for a batch of observations.

    Delegates to compute_action_masks for each env, then stacks results.
    This ensures single source of truth for masking logic.

    Args:
        batch_slot_states: List of slot state dicts, one per env
        enabled_slots: List of enabled slot IDs (same for all envs, from --slots arg)
        total_seeds_list: List of total seeds per env (None = all 0)
        max_seeds: Maximum allowed seeds (0 = unlimited)
        slot_config: Slot configuration (defaults to SlotConfig.default())
        device: Torch device for tensors
        topology: Task topology ("cnn" or "transformer") for blueprint masking
        allow_governor_override: Allow PRUNE even if alpha_mode != HOLD
        disable_advance: If True, mask out ADVANCE regardless of seed stages.

    Returns:
        Dict of boolean tensors (batch_size, num_actions) for each head

    Raises:
        ValueError: If batch_slot_states is empty
        ValueError: If total_seeds_list length doesn't match batch_slot_states length
    """
    # === Input Validation (fail fast on misconfiguration) ===
    if not batch_slot_states:
        raise ValueError(
            "compute_batch_masks requires at least one observation (batch_slot_states is empty)"
        )

    if total_seeds_list is not None and len(total_seeds_list) != len(batch_slot_states):
        raise ValueError(
            f"total_seeds_list length ({len(total_seeds_list)}) must match "
            f"batch_slot_states length ({len(batch_slot_states)})"
        )

    device = device if device is not None else torch.device("cpu")

    # Delegate to compute_action_masks for each env
    # (topology/enabled_slots validation happens in compute_action_masks)
    masks_list = [
        compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled_slots,
            total_seeds=total_seeds_list[i] if total_seeds_list else 0,
            max_seeds=max_seeds,
            slot_config=slot_config,
            device=device,
            topology=topology,
            allow_governor_override=allow_governor_override,
            disable_advance=disable_advance,
        )
        for i, slot_states in enumerate(batch_slot_states)
    ]

    # Stack into batch tensors
    return {
        key: torch.stack([m[key] for m in masks_list])
        for key in masks_list[0]
    }


def slot_id_to_index(slot_id: str, slot_config: SlotConfig | None = None) -> int:
    """Convert canonical slot ID to action index.

    Args:
        slot_id: Canonical slot ID (e.g., "r0c0")
        slot_config: Slot configuration (defaults to SlotConfig.default())

    Returns:
        Index in slot_config.slot_ids tuple

    Raises:
        ValueError: If slot_id not in slot_config or uses legacy format

    Examples:
        >>> slot_id_to_index("r0c0")
        0
        >>> slot_id_to_index("r0c1")
        1
    """
    if slot_config is None:
        slot_config = SlotConfig.default()

    from esper.leyline.slot_id import parse_slot_id, SlotIdError

    # Validate format (will raise SlotIdError for legacy names like "early")
    try:
        parse_slot_id(slot_id)
    except SlotIdError as e:
        raise ValueError(str(e)) from e

    # Use slot_config's index method
    try:
        return slot_config.index_for_slot_id(slot_id)
    except ValueError:
        raise ValueError(f"Unknown slot_id: {slot_id}. Valid: {slot_config.slot_ids}")


__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "slot_id_to_index",
    "MaskedCategorical",
    "InvalidStateMachineError",
]


# =============================================================================
# Masked Distribution (moved from networks.py during dead code cleanup)
# =============================================================================

class InvalidStateMachineError(RuntimeError):
    """Raised when action mask has no valid actions (state machine bug)."""
    pass


@torch.compiler.disable  # type: ignore[untyped-decorator]
def _validate_action_mask(mask: torch.Tensor) -> None:
    """Validate that at least one action is valid per batch element.

    Isolated from torch.compile to prevent graph breaks in the main forward path.
    The .any() call forces CPU sync, but this safety check is worth the cost.
    """
    valid_count = mask.sum(dim=-1)
    if (valid_count == 0).any():
        raise InvalidStateMachineError(
            f"No valid actions available. Mask: {mask}. "
            "This indicates a bug in the Kasmina state machine."
        )


@torch.compiler.disable  # type: ignore[untyped-decorator]
def _validate_logits(logits: torch.Tensor) -> None:
    """Validate that logits don't contain inf/nan (indicates network instability).

    Isolated from torch.compile to prevent graph breaks in the main forward path.
    If logits contain inf/nan, training has already gone wrong upstream.
    """
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise ValueError(
            f"MaskedCategorical received logits with inf/nan values. "
            f"This indicates network instability (gradient explosion, numerical overflow). "
            f"Logits stats: min={logits.min().item():.4g}, max={logits.max().item():.4g}, "
            f"nan_count={torch.isnan(logits).sum().item()}, inf_count={torch.isinf(logits).sum().item()}"
        )


class MaskedCategorical:
    """Categorical distribution with action masking and correct entropy calculation.

    Masks invalid actions by setting their logits to MASKED_LOGIT_VALUE (-1e4)
    before softmax. This value is chosen to be:
    - Large enough to effectively zero the probability after softmax
    - Small enough to avoid numerical overflow in FP16/BF16 (finfo.min can cause issues)
    - Consistent across all dtypes for deterministic behavior

    Computes entropy only over valid actions to avoid penalizing restricted states.

    Optionally applies a probability floor to guarantee minimum exploration mass,
    which ensures gradient flow even when entropy would otherwise collapse.

    Attributes:
        validate: Class-level toggle for validation. Set to False for production
            performance (disables CUDA sync from .any()/.sum() calls).
            Default: True (validation enabled for safety during development).
    """

    validate: bool = True  # Class-level validation toggle

    def __init__(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        min_prob: float | None = None,
    ):
        """Initialize masked categorical distribution.

        Args:
            logits: Raw policy logits [batch, num_actions]
            mask: Boolean mask, True = valid, False = invalid [batch, num_actions]
            min_prob: Optional minimum probability for valid actions. If provided,
                all valid actions are clamped to at least this probability, then
                renormalized. This guarantees gradient flow even when the policy
                would otherwise collapse to a near-deterministic distribution.
                Typical values: 0.02-0.10 depending on head sparsity.

        Raises:
            InvalidStateMachineError: If any batch element has no valid actions
                (only when validate=True)
            ValueError: If logits contain inf or nan (only when validate=True)

        Note:
            The validation check is isolated via @torch.compiler.disable to prevent
            graph breaks in the main forward path while preserving safety checks.
            Disable with MaskedCategorical.validate = False for production.
        """
        if MaskedCategorical.validate:
            _validate_action_mask(mask)
            _validate_logits(logits)

        self.mask = mask
        # Upcast logits to float32 for numerical stability.
        # Under AMP, logits may arrive as float16/bfloat16. The log_softmax in
        # Categorical can produce numerically unstable results in reduced precision.
        # This is defensive - the main fix is in ppo_agent.py which runs evaluate_actions
        # outside autocast. But this upcast provides belt-and-suspenders safety.
        logits_f32 = logits.float()
        self.masked_logits = logits_f32.masked_fill(~mask, MASKED_LOGIT_VALUE)

        # Apply probability floor if specified (guarantees gradient flow)
        if min_prob is not None and min_prob > 0:
            self.masked_logits = self._apply_probability_floor(
                self.masked_logits, mask, min_prob
            )

        self._dist = Categorical(logits=self.masked_logits)

    def _apply_probability_floor(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        min_prob: float,
    ) -> torch.Tensor:
        """Apply probability floor to valid actions and renormalize.

        Ensures all valid actions have at least min_prob probability,
        which guarantees gradient flow even when entropy would otherwise collapse.

        The algorithm (floor-preserving renormalization):
        1. Compute softmax probabilities from logits
        2. Cap floor at 1/num_valid * 0.99 (can't exceed uniform distribution)
        3. Identify "underweight" actions (below floor) and "overweight" actions (above floor)
        4. Set underweight actions to floor, scale overweight actions to fill remaining mass
        5. Convert back to logits via log(prob)

        This preserves the floor constraint after renormalization, unlike naive
        clamp-then-renormalize which can violate the floor.

        Args:
            logits: Masked logits [batch, num_actions] (already masked)
            mask: Boolean mask [batch, num_actions], True = valid
            min_prob: Minimum probability for valid actions

        Returns:
            New logits with probability floor applied [batch, num_actions]
        """
        probs = F.softmax(logits, dim=-1)

        # Cap floor at uniform distribution (can't exceed 1/num_valid)
        num_valid = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
        max_floor = 1.0 / num_valid
        effective_floor = torch.minimum(
            torch.tensor(min_prob, device=logits.device, dtype=logits.dtype),
            max_floor * 0.99,
        )

        # Identify underweight (need boosting) vs overweight (can be scaled down)
        # Only valid actions participate in floor logic
        valid_probs = probs * mask.float()
        is_underweight = mask & (probs < effective_floor)  # Below floor
        is_overweight = mask & (probs >= effective_floor)  # At or above floor

        # Count underweight actions and compute mass needed for flooring
        num_underweight = is_underweight.sum(dim=-1, keepdim=True).float()
        floor_mass_needed = num_underweight * effective_floor

        # Mass available from overweight actions (above their floor allocation)
        overweight_probs = torch.where(is_overweight, valid_probs, torch.zeros_like(valid_probs))
        overweight_total = overweight_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Remaining mass for overweight actions after floor allocation
        remaining_mass = (1.0 - floor_mass_needed).clamp(min=1e-8)

        # Scale factor for overweight actions to fit in remaining mass
        scale = remaining_mass / overweight_total

        # Build final probabilities:
        # - Underweight: set to floor
        # - Overweight: scale down proportionally
        # - Masked: keep at original (will be masked out anyway)
        floored_probs = torch.where(
            is_underweight,
            effective_floor.expand_as(probs),
            torch.where(
                is_overweight,
                probs * scale,
                probs,  # Masked actions keep original prob (irrelevant)
            ),
        )

        # Convert back to logits
        # Use safe_probs to avoid log(0) for masked actions
        safe_probs = torch.where(
            mask,
            floored_probs.clamp(min=1e-8),
            torch.ones_like(floored_probs),
        )
        new_logits = torch.log(safe_probs)
        new_logits = torch.where(mask, new_logits, torch.full_like(new_logits, MASKED_LOGIT_VALUE))

        return new_logits

    @property
    def probs(self) -> torch.Tensor:
        """Action probabilities (masked actions have ~0 probability)."""
        return self._dist.probs

    def sample(self) -> torch.Tensor:
        """Sample actions from the masked distribution."""
        return self._dist.sample()  # type: ignore[no-any-return,no-untyped-call]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions."""
        return self._dist.log_prob(actions)  # type: ignore[no-any-return,no-untyped-call]

    def entropy(self) -> torch.Tensor:
        """Compute normalized entropy over valid actions.

        Returns entropy normalized to [0, 1] by dividing by max entropy
        (log of number of valid actions). This makes exploration incentives
        comparable across states with different action restrictions.

        When only one action is valid, entropy is exactly 0 (no choice = no uncertainty).

        Entropy Coefficient Guidance:
            Since this returns NORMALIZED entropy [0, 1], the entropy coefficient
            in PPO should be higher than typical values for unnormalized entropy.

            - Unnormalized entropy: typical coef ~0.01 (entropy ranges 0 to ~3)
            - Normalized entropy: typical coef ~0.05-0.1 (entropy ranges 0 to 1)

            Example: If you want exploration equivalent to coef=0.01 with unnormalized
            entropy for a 10-action space (max_entropy = ln(10) ≈ 2.3), use:
            normalized_coef = unnormalized_coef * max_entropy ≈ 0.023

            The Esper default of entropy_coef=0.05 is appropriate for normalized entropy.
        """
        probs = self._dist.probs
        log_probs = self._dist.logits - self._dist.logits.logsumexp(dim=-1, keepdim=True)
        raw_entropy = -(probs * log_probs * self.mask).sum(dim=-1)
        num_valid = self.mask.sum(dim=-1).clamp(min=1)
        max_entropy = torch.log(num_valid.float())
        # Guard division to prevent FP16 overflow when num_valid == 1 (max_entropy = 0)
        # Use 1.0 as safe divisor for single-action case; result is discarded by where().
        safe_max_entropy = torch.where(num_valid > 1, max_entropy, torch.ones_like(max_entropy))
        normalized = raw_entropy / safe_max_entropy.clamp(min=1e-8)
        # Single valid action = zero entropy (no choice = no uncertainty)
        return torch.where(num_valid == 1, torch.zeros_like(normalized), normalized)
