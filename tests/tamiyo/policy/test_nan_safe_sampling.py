# tests/tamiyo/policy/test_nan_safe_sampling.py
"""Regression tests for the NaN-safe rollout-sampling guard.

Reproduces (on CPU) the morphogenetic-PPO crash where a non-finite policy logit
makes ``F.softmax`` emit NaN, which trips ``torch.multinomial``'s device-side assert
('probability tensor contains either inf, nan or element < 0') and aborts the rollout.

The guard ``_nan_safe_sampling_probs`` sits only at the rollout multinomial sites; it must
be byte-identical for well-conditioned inputs and produce a finite, sampleable distribution
for any non-finite row, while leaving the loss-path logits/log-probs (and therefore the PPO
finiteness gate) untouched.
"""
import pytest
import torch
import torch.nn.functional as F

from esper.leyline import MASKED_LOGIT_VALUE
from esper.tamiyo.policy.action_masks import (
    _apply_floor_to_logits,
    _nan_safe_sampling_probs,
)

_MULTINOMIAL_CRASH_MSG = "probability tensor contains either"


def _floor_sample_probs(logits: torch.Tensor, mask: torch.Tensor, min_prob: float) -> torch.Tensor:
    """Replicate the rollout call-site: masked_fill -> floor -> softmax (pre-guard)."""
    masked = logits.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    floored = _apply_floor_to_logits(masked, mask, min_prob)
    return F.softmax(floored, dim=-1)


# --------------------------------------------------------------------------------------
# Byte-identity for well-conditioned inputs (the guard must be a no-op)
# --------------------------------------------------------------------------------------

def test_guard_is_identity_on_well_conditioned_probs():
    """A row whose probs are all finite is returned bit-for-bit unchanged."""
    torch.manual_seed(0)
    cases = [
        # (logits, mask, min_prob)
        (torch.tensor([[2.0, -1.0, 0.5]]), torch.ones(1, 3, dtype=torch.bool), 0.05),  # slot
        (torch.randn(8, 6), torch.ones(8, 6, dtype=torch.bool), 0.15),                  # op, batch
        (torch.randn(4, 3), torch.ones(4, 3, dtype=torch.bool), 0.05),                  # slot, batch
    ]
    for logits, mask, min_prob in cases:
        probs = _floor_sample_probs(logits, mask, min_prob)
        guarded = _nan_safe_sampling_probs(probs, mask)
        assert torch.equal(guarded, probs), "guard perturbed a well-conditioned distribution"


def test_collapsed_but_finite_logits_are_identity_and_sampleable():
    """Extreme-but-finite (collapsed) logits floor to a valid dist; guard is a no-op."""
    mask3 = torch.ones(1, 3, dtype=torch.bool)
    for logits in (
        torch.tensor([[50.0, -20.0, -20.0]]),
        torch.tensor([[400.0, -400.0, -400.0]]),
        torch.tensor([[1e30, -1e30, -1e30]]),  # near fp32 max, still finite
    ):
        probs = _floor_sample_probs(logits, mask3, 0.05)
        assert torch.isfinite(probs).all(), "floor produced non-finite probs from FINITE input"
        guarded = _nan_safe_sampling_probs(probs, mask3)
        assert torch.equal(guarded, probs), "guard perturbed a collapsed-but-finite distribution"
        # The 0.05 floor must survive the round-trip (exploration mass guaranteed).
        assert probs.min().item() >= 0.05 - 1e-6
        torch.multinomial(guarded, num_samples=1)  # must not raise


# --------------------------------------------------------------------------------------
# The failure mode: a non-finite logit crashes multinomial WITHOUT the guard
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_logit_crashes_multinomial_without_guard(bad):
    """Documents the bug: floor PROPAGATES a non-finite logit -> multinomial raises."""
    mask3 = torch.ones(1, 3, dtype=torch.bool)
    probs = _floor_sample_probs(torch.tensor([[bad, 1.0, 2.0]]), mask3, 0.05)
    # Floor does not sanitize: NaN fails both '<' and '>=', falls through, clamp+log keep NaN.
    assert not torch.isfinite(probs).all()
    with pytest.raises(RuntimeError, match=_MULTINOMIAL_CRASH_MSG):
        torch.multinomial(probs, num_samples=1)


# --------------------------------------------------------------------------------------
# The fix: the guard makes any non-finite row finite, valid, and sampleable
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_guard_makes_nonfinite_row_sampleable(bad):
    mask3 = torch.ones(1, 3, dtype=torch.bool)
    probs = _floor_sample_probs(torch.tensor([[bad, 1.0, 2.0]]), mask3, 0.05)
    guarded = _nan_safe_sampling_probs(probs, mask3)
    assert torch.isfinite(guarded).all()
    assert (guarded >= 0).all()
    assert guarded.sum(dim=-1).item() > 0
    torch.multinomial(guarded, num_samples=1)  # must not raise


def test_all_nan_row_falls_back_to_uniform_over_valid():
    mask3 = torch.ones(1, 3, dtype=torch.bool)
    probs = _floor_sample_probs(torch.full((1, 3), float("nan")), mask3, 0.05)
    guarded = _nan_safe_sampling_probs(probs, mask3)
    assert torch.allclose(guarded, torch.full((1, 3), 1.0 / 3.0))


def test_guard_fallback_respects_mask():
    """The uniform fallback must put ZERO mass on masked-out (invalid) actions."""
    # 4-action head, only actions 0 and 2 valid; a non-finite logit on a valid action.
    mask = torch.tensor([[True, False, True, False]])
    logits = torch.tensor([[float("nan"), -5.0, 0.5, -5.0]])
    probs = _floor_sample_probs(logits, mask, 0.05)
    guarded = _nan_safe_sampling_probs(probs, mask)
    assert torch.isfinite(guarded).all()
    # Masked positions get exactly zero mass; valid positions split the mass uniformly.
    assert guarded[0, 1].item() == 0.0
    assert guarded[0, 3].item() == 0.0
    assert torch.allclose(guarded[0, [0, 2]], torch.tensor([0.5, 0.5]))
    torch.multinomial(guarded, num_samples=1)


def test_guard_mixed_batch_preserves_finite_rows_exactly():
    """In a batch, finite rows are byte-identical; only the non-finite row is replaced."""
    mask = torch.ones(2, 3, dtype=torch.bool)
    logits = torch.tensor([[2.0, -1.0, 0.5], [float("nan"), 1.0, 2.0]])
    probs = _floor_sample_probs(logits, mask, 0.05)
    guarded = _nan_safe_sampling_probs(probs, mask)
    # Row 0 (finite) unchanged bit-for-bit; row 1 (non-finite) repaired.
    assert torch.equal(guarded[0], probs[0])
    assert not torch.equal(guarded[1], probs[1])
    assert torch.isfinite(guarded).all()
    torch.multinomial(guarded, num_samples=1)  # whole batch samples without raising


# --------------------------------------------------------------------------------------
# End-to-end wiring: a non-finite head logit no longer aborts get_action's rollout sampling
# --------------------------------------------------------------------------------------

def _poison_head_bias(head: torch.nn.Module) -> None:
    """Force a Sequential head's output projection to emit +inf logits."""
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        linears[-1].bias.fill_(float("inf"))


@pytest.mark.parametrize("head_name", ["op_head", "slot_head"])
def test_get_action_survives_nonfinite_head_logit(head_name):
    """Reproduces the crash through the REAL network: a non-finite logit on the op head
    (forward() op-sampling site) or the slot head (_sample_head site) used to trip the
    multinomial device-side assert. With the guard, get_action completes and returns a
    valid in-range action for every head.

    Runs with ``MaskedCategorical.validate = False`` to replicate the TRAINING process,
    where the per-head inf/nan ``_validate_logits`` guard is intentionally disabled for
    performance (vectorized.py). With validate=True (dev/eval) the inf is instead caught
    by a clean ValueError before sampling -- which is precisely why this crashed only in
    training, at multinomial, rather than raising earlier."""
    from esper.leyline import NUM_BLUEPRINTS
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
    from esper.tamiyo.policy.action_masks import MaskedCategorical

    torch.manual_seed(0)
    net = FactoredRecurrentActorCritic(state_dim=50)
    net.eval()
    state = torch.randn(4, 50)
    bp_idx = torch.randint(0, NUM_BLUEPRINTS, (4, net.num_slots))

    saved_validate = MaskedCategorical.validate
    MaskedCategorical.validate = False  # replicate the training process
    try:
        # Sanity: well-conditioned get_action works before poisoning.
        net.get_action(state, bp_idx, deterministic=False)

        _poison_head_bias(getattr(net, head_name))

        # Stochastic rollout sampling must NOT raise the multinomial assert anymore.
        result = net.get_action(state, bp_idx, deterministic=False)
    finally:
        MaskedCategorical.validate = saved_validate

    for key, action in result.actions.items():
        assert torch.isfinite(action.float()).all(), f"{key} produced a non-finite action"
        assert (action >= 0).all(), f"{key} produced a negative action index"
