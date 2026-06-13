"""Gates for P1-VALID (empty-mask guard) and P1-SYNC (finiteness fold equivalence).

P1-VALID: with MaskedCategorical.validate=False (training), an all-masked head must STILL be
caught -- the PPO finiteness gate does not catch it (uniform-over-invalid log_probs are
finite). evaluate_actions keeps a sync-free folded empty-mask guard for this.

P1-SYNC: the fused finiteness reduction (one sync) must make the SAME gate decision as the
original per-head OR-of-.all() loop, on every NaN/Inf placement. The attribution slow-path is
byte-identical (verbatim), so matching the gate decision is the contract.
"""

import pytest
import torch

from esper.leyline import HEAD_NAMES
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
from esper.tamiyo.policy.action_masks import (
    MASKED_LOGIT_VALUE,
    InvalidStateMachineError,
    MaskedCategorical,
    _masked_log_prob,
)


# --- P1-VALID: the premise -- an empty mask yields FINITE log_probs ------------------
def test_empty_mask_logprob_is_finite_not_caught_by_finiteness():
    # All actions masked -> all logits = MASKED_LOGIT_VALUE -> uniform-over-invalid ->
    # log_prob is finite. This is why the finiteness gate cannot catch empty masks.
    logits = torch.randn(4, 6)
    mask = torch.zeros(4, 6, dtype=torch.bool)  # empty
    masked = logits.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    action = torch.zeros(4, dtype=torch.long)
    lp = _masked_log_prob(masked, action)
    assert torch.isfinite(lp).all(), "empty-mask log_prob is finite (the unguarded hazard)"


# --- P1-VALID: the guard catches it when validate=False ------------------------------
def _build_policy_and_inputs(batch=4):
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)
    policy = FactoredRecurrentActorCritic(state_dim=64, slot_config=slot_config)
    policy.eval()
    states = torch.randn(batch, 1, 64)
    blueprint_indices = torch.full((batch, 1, num_slots), -1, dtype=torch.long)
    # Sample real actions so the action indices are valid for gather.
    res = policy.get_action(states[:, 0, :], blueprint_indices[:, 0, :], deterministic=True)
    actions = {k: v.clone().unsqueeze(1) for k, v in res.actions.items()}
    return policy, states, blueprint_indices, actions, policy.num_ops


def test_empty_op_mask_raises_when_validate_false():
    prev = MaskedCategorical.validate
    try:
        MaskedCategorical.validate = False
        policy, states, bp, actions, num_ops = _build_policy_and_inputs()
        empty_op_mask = torch.zeros(states.shape[0], 1, num_ops, dtype=torch.bool)
        with pytest.raises(InvalidStateMachineError):
            policy.evaluate_actions(states, bp, actions, op_mask=empty_op_mask)
    finally:
        MaskedCategorical.validate = prev


def test_valid_masks_do_not_raise_when_validate_false():
    prev = MaskedCategorical.validate
    try:
        MaskedCategorical.validate = False
        policy, states, bp, actions, _ = _build_policy_and_inputs()
        # No explicit masks -> all-valid -> must not raise and must return finite log_probs.
        log_probs, _, _, _, _ = policy.evaluate_actions(states, bp, actions)
        for k in HEAD_NAMES:
            assert torch.isfinite(log_probs[k]).all()
    finally:
        MaskedCategorical.validate = prev


# --- P1-SYNC: fused finiteness gate decision == original per-head loop ---------------
def _gate_old(new_lp: dict, old_lp: dict, values: torch.Tensor) -> bool:
    """Original per-head OR-of-.all() logic -> nonfinite_found."""
    found = False
    for k in HEAD_NAMES:
        if not torch.isfinite(new_lp[k]).all():
            found = True
    for k in HEAD_NAMES:
        if not torch.isfinite(old_lp[k]).all():
            found = True
    if not torch.isfinite(values).all():
        found = True
    return found


def _gate_new(new_lp: dict, old_lp: dict, values: torch.Tensor) -> bool:
    """P1-SYNC fused reduction -> not all_finite."""
    new_stack = torch.stack([new_lp[k].float() for k in HEAD_NAMES])
    old_stack = torch.stack([old_lp[k].float() for k in HEAD_NAMES])
    all_finite = (
        torch.isfinite(new_stack).all()
        & torch.isfinite(old_stack).all()
        & torch.isfinite(values).all()
    )
    return not bool(all_finite)


def test_p1sync_fold_matches_original_gate_decision():
    torch.manual_seed(0)
    n = 10
    base_new = {k: torch.randn(n) for k in HEAD_NAMES}
    base_old = {k: torch.randn(n) for k in HEAD_NAMES}
    base_val = torch.randn(n)

    # All finite -> no gate.
    assert _gate_old(base_new, base_old, base_val) is False
    assert _gate_new(base_new, base_old, base_val) is False

    # Single-head NaN in new log_probs.
    for inject_head in [HEAD_NAMES[0], HEAD_NAMES[3], HEAD_NAMES[-1]]:
        nn_new = {k: v.clone() for k, v in base_new.items()}
        nn_new[inject_head][2] = float("nan")
        assert _gate_old(nn_new, base_old, base_val) == _gate_new(nn_new, base_old, base_val) is True

    # Single-head Inf in old log_probs.
    oo = {k: v.clone() for k, v in base_old.items()}
    oo[HEAD_NAMES[1]][5] = float("inf")
    assert _gate_old(base_new, oo, base_val) == _gate_new(base_new, oo, base_val) is True

    # NaN in values.
    vv = base_val.clone()
    vv[0] = float("nan")
    assert _gate_old(base_new, base_old, vv) == _gate_new(base_new, base_old, vv) is True
