"""SUPPRESS-SLOT intervention mask logic (estimand-invariant core).

Pure tensor logic for the causal-contribution harness's primary intervention:
a stationary mask that makes one slot un-committable to every non-WAIT op for
the whole run, WITH the op-validity repair from design §5.1.

Why the repair is load-bearing (verified firsthand against the sampler):
``factored_lstm.get_action`` gathers the final slot mask from the op-conditioned
``slot_by_op`` row of the *sampled* op (``factored_lstm.py:1258-1272``), NOT from
the marginal ``slot`` head. So zeroing the suppressed slot's ``slot_by_op``
column is what removes it — but if we stop there and the suppressed slot was the
ONLY valid slot for some op (e.g. GERMINATE), ``op_mask`` for that op stays True
(``action_masks.py:300-306``), the controller can sample it, and the gathered
slot row is ALL FALSE. Under ``MaskedCategorical.validate=False`` in training
(``vectorized.py:1102``) the rollout does not raise — it uniformly samples an
INVALID slot (possibly the suppressed slot itself, silently defeating
suppression for that step) and the next PPO update hard-crashes on the empty-mask
guard (``factored_lstm.py:1547``). The repair recomputes each non-WAIT op's
validity from its (now-zeroed) ``slot_by_op`` row, routing
"suppressed-slot-was-the-only-option" steps to WAIT-only.

These functions are pure and operate on the batched mask dict in place; they own
no telemetry or run context (the caller emits). In-place bool-mask edits are
autograd-safe: masks are rebuilt fresh per step, carry no grad history, and
sampling runs under ``torch.inference_mode`` (design §5.1 note).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from esper.leyline import (
    CONTROL_MODE,
    INTERVENTION_DETERMINISM_CLASS,
    SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY,
    SUPPRESS_SLOT_TARGET_SLOT_ID,
    InterventionConfiguredPayload,
    LifecycleOp,
    ProofBaselineMode,
)


@dataclass(slots=True, frozen=True)
class SuppressSlotResult:
    """Outcome of one SUPPRESS-SLOT mask application over a batch of envs.

    Attributes:
        suppressed_slot_index: The slot index zeroed out of every non-WAIT op.
        intervention_forced: bool[num_envs] — envs the suppression flipped to
            WAIT-only (had a non-WAIT op pre-suppression, WAIT-only after).
        wait_saturation_forced: bool[num_envs] — envs that were ALREADY WAIT-only
            before any intervention (natural WAIT saturation; arm-invariant).
    """

    suppressed_slot_index: int
    intervention_forced: torch.Tensor
    wait_saturation_forced: torch.Tensor

    @property
    def intervention_forced_count(self) -> int:
        return int(self.intervention_forced.sum().item())

    @property
    def wait_saturation_count(self) -> int:
        return int(self.wait_saturation_forced.sum().item())


def split_forced_step_reasons(
    pre_op_mask: torch.Tensor,
    post_op_mask: torch.Tensor,
    *,
    wait_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split forced (WAIT-only) steps into intervention vs WAIT-saturation.

    A step is "forced" when the env has exactly one valid op and it is WAIT.
    This is the same condition ``execute_actions`` uses
    (``action_execution.py:552``), evaluated here on the POST-intervention mask.

    Args:
        pre_op_mask: bool[num_envs, num_ops] op mask BEFORE the intervention.
        post_op_mask: bool[num_envs, num_ops] op mask AFTER the intervention/repair.
        wait_idx: index of the WAIT op.

    Returns:
        (intervention_forced, wait_saturation_forced), each bool[num_envs].
        intervention_forced: forced now, NOT forced before (the mask did it).
        wait_saturation_forced: forced before (and still) — natural saturation.
        The two are mutually exclusive; their union is the set of forced steps.
    """
    if pre_op_mask.shape != post_op_mask.shape:
        raise ValueError(
            f"pre/post op masks must match: {tuple(pre_op_mask.shape)} vs "
            f"{tuple(post_op_mask.shape)}"
        )
    # WAIT is always valid in both masks, so "WAIT-only" == exactly one valid op.
    post_wait_only = post_op_mask.sum(dim=-1) == 1
    pre_wait_only = pre_op_mask.sum(dim=-1) == 1
    intervention_forced = post_wait_only & ~pre_wait_only
    wait_saturation_forced = pre_wait_only
    return intervention_forced, wait_saturation_forced


def apply_suppress_slot_mask(
    masks_batch: dict[str, torch.Tensor],
    *,
    suppressed_slot_index: int,
) -> SuppressSlotResult:
    """Make ``suppressed_slot_index`` un-committable, with the op-validity repair.

    Mutates ``masks_batch["op"]`` and ``masks_batch["slot_by_op"]`` in place.

    Steps (design §5.1):
      1. Read the trigger rows from ``slot_by_op`` BEFORE the zeroing mutation
         (snapshot ``op`` mask) so the forced-step split is computed against the
         pre-intervention state.
      2. Zero the suppressed slot's column across every op (incl. GERMINATE).
      3. Recompute each NON-WAIT op's validity as
         ``op_mask[:, op] &= slot_by_op[:, op, :].any(dim=-1)``; WAIT stays True.
         This routes "suppressed-slot-was-the-only-option" steps to WAIT-only and
         guarantees no op survives with an all-false slot row (no silent
         invalid-sample, no empty-mask crash).

    Args:
        masks_batch: batched mask dict with ``op`` [E, O] and ``slot_by_op``
            [E, O, S] (the layout produced at ``vectorized_trainer.py:2041-2047``).
        suppressed_slot_index: slot column to zero (e.g.
            ``slot_config.index_for_slot_id("r0c0")``).

    Returns:
        SuppressSlotResult with the per-env forced-step split.
    """
    op_mask = masks_batch["op"]            # [E, O] bool
    slot_by_op = masks_batch["slot_by_op"]  # [E, O, S] bool

    if op_mask.dim() != 2:
        raise ValueError(f"op mask must be [num_envs, num_ops], got {tuple(op_mask.shape)}")
    if slot_by_op.dim() != 3:
        raise ValueError(
            f"slot_by_op must be [num_envs, num_ops, num_slots], got {tuple(slot_by_op.shape)}"
        )
    num_slots = slot_by_op.shape[2]
    if suppressed_slot_index < 0 or suppressed_slot_index >= num_slots:
        raise ValueError(
            f"suppressed_slot_index {suppressed_slot_index} out of range for "
            f"num_slots={num_slots}"
        )
    if op_mask.shape[0] != slot_by_op.shape[0] or op_mask.shape[1] != slot_by_op.shape[1]:
        raise ValueError(
            "op mask and slot_by_op disagree on [num_envs, num_ops]: "
            f"{tuple(op_mask.shape)} vs {tuple(slot_by_op.shape[:2])}"
        )

    wait_idx = LifecycleOp.WAIT.value

    # (1) snapshot the pre-intervention op mask (trigger rows read from the
    #     pre-mutation state, per the §5.1 ordering note).
    pre_op_mask = op_mask.clone()

    # (2) zero the suppressed slot's column across ALL ops (incl. GERMINATE).
    slot_by_op[:, :, suppressed_slot_index] = False

    # (3) recompute non-WAIT op validity from the repaired slot rows; WAIT stays True.
    op_row_has_slot = slot_by_op.any(dim=-1)   # [E, O]
    repaired = op_mask & op_row_has_slot
    repaired[:, wait_idx] = True
    masks_batch["op"] = repaired

    intervention_forced, wait_saturation_forced = split_forced_step_reasons(
        pre_op_mask, repaired, wait_idx=wait_idx
    )
    return SuppressSlotResult(
        suppressed_slot_index=suppressed_slot_index,
        intervention_forced=intervention_forced,
        wait_saturation_forced=wait_saturation_forced,
    )


def build_intervention_configured_payload(
    *,
    lifecycle_policy: str | None,
    suppression_enabled: bool,
    slot_config: object,
    rng_split_enabled: bool,
    master_seed: int,
) -> InterventionConfiguredPayload:
    """Build the run-start INTERVENTION_CONFIGURED payload (no action_id, §5.7).

    Pure: takes the run's intervention posture and returns the typed payload. The
    caller emits. A non-SUPPRESS-SLOT instrumented arm (matched control) reports
    mode="control", suppression OFF, with the RNG split still enabled.
    """
    is_suppress_slot = lifecycle_policy == SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY
    suppressed_index = (
        slot_config.index_for_slot_id(SUPPRESS_SLOT_TARGET_SLOT_ID)  # type: ignore[attr-defined]
        if is_suppress_slot
        else -1
    )
    mode = (
        ProofBaselineMode.SUPPRESS_SLOT.value if is_suppress_slot else CONTROL_MODE
    )
    return InterventionConfiguredPayload(
        proof_baseline_mode=mode,
        lifecycle_policy=lifecycle_policy or "none",
        suppressed_slot_id=(
            SUPPRESS_SLOT_TARGET_SLOT_ID if is_suppress_slot else "none"
        ),
        suppressed_slot_index=suppressed_index,
        suppression_enabled=bool(is_suppress_slot and suppression_enabled),
        determinism_class=INTERVENTION_DETERMINISM_CLASS.value,
        rng_split_enabled=rng_split_enabled,
        master_seed=master_seed,
        description=(
            "SUPPRESS-SLOT stationary mask: r0c0 un-committable to every "
            "non-WAIT op for the whole run, with op-validity repair routing "
            "r0c0-only steps to WAIT."
            if is_suppress_slot
            else "Matched control / no stationary intervention."
        ),
    )


class OffsetParityError(AssertionError):
    """Raised when a no-offset / pairing invariant fails. Pre-verdict HARD gate."""


def assert_constant_controller_stride(draw_counts: list[int]) -> int:
    """Assert the per-step controller draw-count DELTA is constant (within a run).

    ``draw_counts`` is the per-step cumulative ``controller_draw_count`` sequence
    (in epoch order) from one run's INTERVENTION_STEP records. The controller
    draws a fixed number of times per step, so consecutive deltas must be equal;
    a deviation means the controller stream was offset inside the run. Returns the
    pinned stride. Raises ``OffsetParityError`` on any deviation.
    """
    if len(draw_counts) < 2:
        return 0
    deltas = [b - a for a, b in zip(draw_counts, draw_counts[1:])]
    stride = deltas[0]
    for i, d in enumerate(deltas[1:], start=1):
        if d != stride:
            raise OffsetParityError(
                f"controller draw-count stride broke at step {i}: {d} != {stride} "
                "(controller stream offset within the run)"
            )
    return stride


def assert_offset_free_parity(
    control_draw_counts: list[int],
    other_draw_counts: list[int],
    *,
    control_state_hashes: list[str | None] | None = None,
    other_state_hashes: list[str | None] | None = None,
) -> None:
    """HARD pre-verdict gate: paired arms share the controller stream position.

    The dispositive, device-portable check is cumulative ``controller_draw_count``
    parity across the matched arms (control vs suppress_slot_off, or the
    pre-first-suppression prefix of control vs suppress_slot_on): equal counts
    prove the controller stream was never offset by differing blueprint/host draw
    counts. Where BOTH arms emitted a state hash at the same step (stride-aligned),
    those must also match (a secondary cross-check). Raises ``OffsetParityError``.

    The caller is responsible for trimming both sequences to the comparable prefix
    (e.g. up to the first suppression for an ON arm).
    """
    if control_draw_counts != other_draw_counts:
        raise OffsetParityError(
            "controller draw-count parity failed across paired arms: "
            f"{control_draw_counts!r} != {other_draw_counts!r} "
            "(an RNG offset would inject an artifact into the primary Δ_struct)"
        )
    if control_state_hashes is not None and other_state_hashes is not None:
        for i, (a, b) in enumerate(zip(control_state_hashes, other_state_hashes)):
            if a is not None and b is not None and a != b:
                raise OffsetParityError(
                    f"controller state-hash parity failed at step {i}: {a} != {b}"
                )


def assert_target_slot_never_committed(
    germination_slot_ids: list[str], *, suppressed_slot_id: str = SUPPRESS_SLOT_TARGET_SLOT_ID
) -> None:
    """Enforce the from-step-0 invariant: the suppressed slot never germinates.

    ``germination_slot_ids`` is the list of ``slot_id`` from the ON run's
    SEED_GERMINATED records. The stationary mask zeroes the suppressed slot every
    step including step 0, so a single occurrence here is a suppression breach.
    Raises ``OffsetParityError``.
    """
    breaches = [s for s in germination_slot_ids if s == suppressed_slot_id]
    if breaches:
        raise OffsetParityError(
            f"suppression breached: {len(breaches)} germination(s) at suppressed "
            f"slot {suppressed_slot_id!r} (the from-step-0 invariant did not hold)"
        )


__all__ = [
    "OffsetParityError",
    "SuppressSlotResult",
    "apply_suppress_slot_mask",
    "assert_constant_controller_stride",
    "assert_offset_free_parity",
    "assert_target_slot_never_committed",
    "build_intervention_configured_payload",
    "split_forced_step_reasons",
]
