"""Subtractive reward-stream partition for EV-stabilization (Stage 0 + Stage 2).

The PPO step reward is finalized in ``action_execution.py`` (after
``compute_contribution_reward`` applies the terminal corrections). This helper
splits that finalized ``reward_raw`` into two streams that sum back to it exactly:

- ``R_cf``   — the dense counterfactual stream = ``components.bounded_attribution``.
  ``bounded_attribution`` is the SOLE counterfactual addend hitting the reward and
  it already contains ``escrow_delta`` and ``ratio_penalty`` (see
  ``contribution.py``), so it must not be re-added from those mirror fields.
- ``R_main`` — ``reward_raw - R_cf`` = every shaping term, the terminal corrections,
  and the telemetry-less ``pending_auto_prune_penalty``, by subtraction.

EXHAUSTIVE BY CONSTRUCTION: ``r_main + r_cf == reward_raw`` for any input, so no
reward term can be dropped or double-counted. Stage 0 uses this for per-stream EV
diagnostics; Stage 2 routes ``R_cf`` to a dedicated cf value head. They share this
single source so the two stages cannot diverge.
"""

from esper.leyline.telemetry_contracts import RewardComponentsTelemetry


# Phase 0 (Stage-0): the SHAPED/ESCROW additend set and its SIGN MAP — each entry is
# the term's signed contribution to ``reward_raw`` (the finalized PPO step reward).
# Reviewer-mandated (Phase 0 evidence packet §4): component fields are stored in mixed
# sign conventions, so the decomposition must apply the correct sign, not the raw field.
#
#   +1  the field IS the signed contribution. For ``compute_rent`` (stored = −rent_penalty)
#       and ``escrow_forfeit`` (stored = −escrow_forfeit) this is ALREADY negative.
#   −1  the field is a POSITIVE magnitude the reward SUBTRACTS (``reward -= field``), so the
#       signed contribution is the negated field (occupancy_rent / fossilized_rent).
#
# EXCLUDED on purpose:
#   * ``ratio_penalty``           — folded into ``bounded_attribution`` (contribution.py:680);
#                                    a separate additend would double-count it.
#   * ``fossilize_terminal_bonus`` — always 0.0 (the real immediate bonus is inside
#                                    ``action_shaping``); a dead field, not an additend.
# Telemetry-less terms (no component field — e.g. ``pending_auto_prune_penalty``) are
# captured by the ``residual`` (``reward_raw − Σ(named signed)``), exhaustive by construction.
ADDITEND_SIGN_MAP: dict[str, int] = {
    "bounded_attribution": 1,  # float | None; None ⇒ 0.0 (no attribution this step)
    "blending_warning": 1,
    "holding_warning": 1,
    "pbrs_bonus": 1,
    "synergy_bonus": 1,
    "alpha_shock": 1,
    "action_shaping": 1,
    "terminal_bonus": 1,
    "hindsight_credit": 1,
    "compute_rent": 1,        # stored already-signed (= −rent_penalty)
    "escrow_forfeit": 1,      # stored already-signed (= −escrow_forfeit)
    "occupancy_rent": -1,     # stored positive; reward subtracts
    "fossilized_rent": -1,    # stored positive; reward subtracts
}

RESIDUAL_KEY = "residual"


def decompose_additends(
    reward_raw: float, components: RewardComponentsTelemetry
) -> dict[str, float]:
    """Decompose a finalized SHAPED/ESCROW step reward into per-term SIGNED additends.

    Returns a dict mapping each additend name (``ADDITEND_SIGN_MAP`` keys) to its signed
    contribution to ``reward_raw``, plus a ``residual`` key (``reward_raw − Σ(named)``)
    that captures any telemetry-less correction (e.g. ``pending_auto_prune_penalty``).
    ``sum(result.values()) == reward_raw`` exhaustively by construction.

    Args:
        reward_raw: the finalized PPO step reward (``action_outcome.reward_raw``).
        components: the populated reward-component breakdown for this step
            (CONTRIBUTION family with ``return_components``; callers guarantee non-None).

    Returns:
        ``{term: signed_contribution, ..., "residual": reward_raw − Σ(named)}``.
    """
    decomposed: dict[str, float] = {}
    named_sum = 0.0
    for term, sign in ADDITEND_SIGN_MAP.items():
        raw = getattr(components, term)
        # Explicit None branch (NOT a defensive ``or 0.0``): only ``bounded_attribution``
        # is Optional, and None means "no attribution this step" — a real 0.0-of-cf state,
        # the same semantics as ``split_reward_streams``.
        value = 0.0 if raw is None else float(raw)
        contribution = sign * value
        decomposed[term] = contribution
        named_sum += contribution
    decomposed[RESIDUAL_KEY] = reward_raw - named_sum
    return decomposed


def split_reward_streams(
    reward_raw: float, components: RewardComponentsTelemetry
) -> tuple[float, float]:
    """Split a finalized step reward into ``(r_main, r_cf)``.

    Args:
        reward_raw: The finalized PPO step reward (``action_outcome.reward_raw``).
        components: The reward-component breakdown for this step. Must be populated
            (callers guarantee non-None; CONTRIBUTION-family with ``return_components``).

    Returns:
        ``(r_main, r_cf)`` where ``r_cf`` is the counterfactual stream
        (``bounded_attribution``, or ``0.0`` when there is no attribution this step)
        and ``r_main = reward_raw - r_cf``.
    """
    # Explicit None branch (NOT a defensive ``or 0.0``): None means "no attribution
    # this step", which is a real, expected state for SHAPED steps with no
    # contribution signal — semantically 0.0 of counterfactual reward.
    if components.bounded_attribution is None:
        r_cf = 0.0
    else:
        r_cf = components.bounded_attribution
    return reward_raw - r_cf, r_cf
