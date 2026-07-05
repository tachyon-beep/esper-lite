"""Regression: the telemetry op-probs readout must not crash on a transient non-finite
op-logit (rare BF16 rollout artifact). A diagnostic must never kill a training run — the
real non-finite is recorded by the PPO update finiteness gate, not here. (Seed-44 crash,
2026-07-01: env-8 all-NaN op-logits killed a 144-episode run at MaskedCategorical.)
"""
import torch

from esper.simic.training.vectorized_trainer import _masked_op_probs_for_telemetry
from esper.tamiyo.policy.action_masks import MaskedCategorical


def test_all_nan_env_does_not_crash_and_is_finite():
    op_logits = torch.zeros(3, 6)
    op_logits[1] = float("nan")  # the seed-44 crash shape: one env all-NaN
    op_mask = torch.ones(3, 6, dtype=torch.bool)
    probs = _masked_op_probs_for_telemetry(
        op_logits=op_logits, op_mask=op_mask, probability_floor={"op": 0.02}
    )
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1), torch.ones(3))


def test_partial_mask_with_nan_row_is_finite():
    op_logits = torch.randn(3, 6)
    op_logits[2] = float("nan")
    op_mask = torch.zeros(3, 6, dtype=torch.bool)
    op_mask[:, 0] = True  # WAIT always valid
    op_mask[0, 1] = True
    probs = _masked_op_probs_for_telemetry(
        op_logits=op_logits, op_mask=op_mask, probability_floor={"op": 0.02}
    )
    assert torch.isfinite(probs).all()


def test_finite_input_is_byte_identical():
    """The sanitize is skipped for all-finite input — telemetry is unchanged for the
    common case (byte-identical to a direct MaskedCategorical readout)."""
    op_logits = torch.randn(4, 6)
    op_mask = torch.ones(4, 6, dtype=torch.bool)
    guarded = _masked_op_probs_for_telemetry(
        op_logits=op_logits, op_mask=op_mask, probability_floor=None
    )
    direct = MaskedCategorical(op_logits, op_mask).probs
    assert torch.equal(guarded, direct)
