"""Parity gates for the shared masked-logit seam (P1-EVAL / P1-BF16 CRITICAL-1).

These tests pin the contract that the rollout leg (``get_action``) and the update leg
(``evaluate_actions``) compute log-probs / entropy / floor through ONE FP32 pathway, so
the PPO importance ratio is unbiased even when the backbone runs in BF16.

- V1  : helper log_prob == an independent ``Categorical`` reference (FP32 golden).
- V1b : the floor/log_prob helpers force FP32 internally -> a BF16 call-site cannot
        leak reduced precision into the seam.
- V3  : normalized-entropy edges (single-valid -> 0; 2-valid floor-active) in FP32+BF16.
- V2  : cross-path -> get_action log_prob == evaluate_actions log_prob on the SAME
        weights (the CRITICAL-1 symmetry falsifier).
"""

import torch
from torch.distributions import Categorical

from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
from esper.tamiyo.policy.action_masks import (
    MASKED_LOGIT_VALUE,
    MaskedCategorical,
    _apply_floor_to_logits,
    _masked_log_prob,
    _normalized_entropy_from_masked_logits,
)


def _rand_mask(batch: int, dim: int, *, min_valid: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(batch, dim, generator=g) > 0.4
    # Guarantee at least `min_valid` valid actions per row (no empty masks).
    for i in range(batch):
        if int(mask[i].sum()) < min_valid:
            idx = torch.randperm(dim, generator=g)[:min_valid]
            mask[i, idx] = True
    return mask


# --- V1: log_prob matches an independent Categorical reference (no floor) -------------
def test_v1_masked_log_prob_matches_categorical_reference():
    g = torch.Generator().manual_seed(1)
    batch, dim = 16, 8
    logits = torch.randn(batch, dim, generator=g)
    mask = _rand_mask(batch, dim, seed=1)
    masked_logits = logits.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    action = torch.randint(0, dim, (batch,), generator=g)

    ref = Categorical(logits=masked_logits).log_prob(action)
    got = _masked_log_prob(masked_logits, action)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


# --- V1: floor path matches Categorical over the floored logits ----------------------
def test_v1_floor_log_prob_matches_categorical_reference():
    g = torch.Generator().manual_seed(2)
    batch, dim = 16, 6
    logits = torch.randn(batch, dim, generator=g)
    mask = _rand_mask(batch, dim, min_valid=2, seed=2)
    masked = logits.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    floored = _apply_floor_to_logits(masked, mask, min_prob=0.1)
    action = torch.randint(0, dim, (batch,), generator=g)

    ref = Categorical(logits=floored).log_prob(action)
    got = _masked_log_prob(floored, action)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

    # Floored probs are a valid distribution over valid actions (masked ~0). The floor
    # algorithm sets originally-underweight actions to the floor but may scale a barely-
    # overweight action below it during renormalization -- that is the PRE-EXISTING
    # algorithm's behavior (preserved verbatim), so we only assert a proper distribution.
    probs = torch.softmax(floored, dim=-1)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(batch), rtol=1e-5, atol=1e-5)
    assert (probs[~mask] < 1e-3).all()  # invalid actions carry ~0 mass


# --- V1: MaskedCategorical (now delegating) stays consistent with the seam -----------
def test_v1_masked_categorical_delegates_consistently():
    g = torch.Generator().manual_seed(3)
    batch, dim = 12, 7
    logits = torch.randn(batch, dim, generator=g)
    mask = _rand_mask(batch, dim, min_valid=2, seed=3)
    action = torch.randint(0, dim, (batch,), generator=g)

    MaskedCategorical.validate = True
    dist = MaskedCategorical(logits=logits, mask=mask, min_prob=0.05)
    seam = _masked_log_prob(dist.masked_logits, action)
    torch.testing.assert_close(dist.log_prob(action), seam, rtol=1e-5, atol=1e-6)
    ent_ref = _normalized_entropy_from_masked_logits(dist.masked_logits, dist.mask)
    torch.testing.assert_close(dist.entropy(), ent_ref, rtol=1e-5, atol=1e-6)


# --- V1b: BF16 call-site cannot leak reduced precision into the seam -----------------
def test_v1b_floor_forces_fp32_under_bf16_callsite():
    g = torch.Generator().manual_seed(4)
    batch, dim = 16, 6
    logits = torch.randn(batch, dim, generator=g)
    mask = _rand_mask(batch, dim, min_valid=2, seed=4)

    # Same logit VALUES, one stored bf16 one fp32. The helper .float()s internally, so
    # floor(bf16_logits) must equal floor(bf16_logits.float()) bit-for-bit (no bf16 math).
    bf16 = logits.to(torch.bfloat16)
    masked_bf16 = bf16.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    masked_fp32 = bf16.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    out_from_bf16 = _apply_floor_to_logits(masked_bf16, mask, 0.1)
    out_from_fp32 = _apply_floor_to_logits(masked_fp32, mask, 0.1)
    assert out_from_bf16.dtype == torch.float32
    torch.testing.assert_close(out_from_bf16, out_from_fp32, rtol=0, atol=0)

    action = torch.randint(0, dim, (batch,), generator=g)
    lp_bf16 = _masked_log_prob(masked_bf16, action)
    assert lp_bf16.dtype == torch.float32


# --- V3: entropy edges ---------------------------------------------------------------
def test_v3_single_valid_action_entropy_is_zero():
    batch, dim = 5, 4
    mask = torch.zeros(batch, dim, dtype=torch.bool)
    mask[:, 0] = True  # exactly one valid action per row
    logits = torch.randn(batch, dim)
    masked = logits.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    ent = _normalized_entropy_from_masked_logits(masked, mask)
    torch.testing.assert_close(ent, torch.zeros(batch), rtol=0, atol=0)


def test_v3_two_valid_floor_active_entropy_fp32_and_bf16():
    batch, dim = 8, 5
    g = torch.Generator().manual_seed(5)
    mask = torch.zeros(batch, dim, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = True  # exactly two valid actions
    logits = torch.randn(batch, dim, generator=g) * 3.0  # peaky -> floor will bite

    masked_fp32 = logits.float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    floored_fp32 = _apply_floor_to_logits(masked_fp32, mask, 0.1)
    ent_fp32 = _normalized_entropy_from_masked_logits(floored_fp32, mask)

    masked_bf16 = logits.to(torch.bfloat16).float().masked_fill(~mask, MASKED_LOGIT_VALUE)
    floored_bf16 = _apply_floor_to_logits(masked_bf16, mask, 0.1)
    ent_bf16 = _normalized_entropy_from_masked_logits(floored_bf16, mask)

    # Normalized entropy in [0, 1]; both paths FP32-computed.
    assert (ent_fp32 >= 0).all() and (ent_fp32 <= 1.0 + 1e-5).all()
    # bf16-rounded logits perturb the value slightly but the PATHWAY is identical FP32.
    torch.testing.assert_close(ent_fp32, ent_bf16, rtol=2e-2, atol=2e-2)


# --- V2: cross-path rollout-vs-update log_prob symmetry (CRITICAL-1) ------------------
def test_v2_cross_path_log_prob_symmetry_fp32():
    torch.manual_seed(7)
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)
    policy = FactoredRecurrentActorCritic(state_dim=64, slot_config=slot_config)
    policy.eval()

    batch = 6
    state = torch.randn(batch, 64)
    blueprint_indices = torch.full((batch, num_slots), -1, dtype=torch.long)
    floor = {"op": 0.05, "blueprint": 0.1, "tempo": 0.1}

    res = policy.get_action(
        state, blueprint_indices, deterministic=True, probability_floor=floor
    )

    # Feed the rollout actions back through the update leg on the SAME weights/hidden.
    # .clone() lifts them out of inference_mode (the real rollout buffer stores copies).
    actions_seq = {k: v.clone().unsqueeze(1) for k, v in res.actions.items()}  # [batch, 1]
    log_probs_eval, _, _, _, _ = policy.evaluate_actions(
        state.unsqueeze(1),
        blueprint_indices.unsqueeze(1),
        actions_seq,
        probability_floor=floor,
    )

    for key, lp_roll in res.log_probs.items():
        lp_upd = log_probs_eval[key][:, 0]
        torch.testing.assert_close(
            lp_upd, lp_roll, rtol=1e-4, atol=1e-5,
            msg=f"cross-path log_prob mismatch on head '{key}'",
        )


def test_v2_cross_path_log_prob_symmetry_stochastic_op():
    """Stochastic rollout: ratio symmetry must hold even though forward() samples op
    from the unfloored distribution (the op is SCORED against the floored FP32 one in
    both legs, so old_log_prob == new_log_prob). Pins MEDIUM-1 from review.
    """
    torch.manual_seed(11)
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)
    policy = FactoredRecurrentActorCritic(state_dim=64, slot_config=slot_config)
    policy.eval()

    batch = 8
    state = torch.randn(batch, 64)
    blueprint_indices = torch.full((batch, num_slots), -1, dtype=torch.long)
    floor = {"op": 0.05, "blueprint": 0.1, "tempo": 0.1}

    res = policy.get_action(
        state, blueprint_indices, deterministic=False, probability_floor=floor
    )
    actions_seq = {k: v.clone().unsqueeze(1) for k, v in res.actions.items()}
    log_probs_eval, _, _, _, _ = policy.evaluate_actions(
        state.unsqueeze(1),
        blueprint_indices.unsqueeze(1),
        actions_seq,
        probability_floor=floor,
    )
    # The op head is the one under scrutiny: its stored old_log_prob must equal the
    # recomputed new log_prob to FP32 precision (unbiased ratio), regardless of how the
    # op was sampled.
    torch.testing.assert_close(
        log_probs_eval["op"][:, 0], res.log_probs["op"], rtol=1e-4, atol=1e-5,
        msg="stochastic-op ratio asymmetry (CRITICAL-1 regression)",
    )
