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
