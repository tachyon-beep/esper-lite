"""Tests for the shared PPO policy-AMP context (P1-BF16 HIGH-1 / MEDIUM-1 / NOTE-1).

`policy_amp_context` is the SINGLE precision decision shared by the rollout sampling, the
bootstrap value, and the PPO update. These tests pin:
- HIGH-1: FP16 must NOT autocast the policy path (the policy optimizer has no GradScaler;
  FP16 would underflow the unscaled backward). FP16 -> FP32 (nullcontext).
- MEDIUM-1: a single factory drives both legs, so they cannot silently desync.
- NOTE-1: under BF16 autocast the LSTM carry (h, c) stays FP32 (cuDNN RNN is on autocast's
  fp32 list), so reduced-precision drift does not compound across a rollout.
"""

from contextlib import nullcontext

import pytest
import torch

from esper.simic.training.helpers import policy_amp_context


def test_fp16_never_autocasts_policy_path():
    # FP16 is unsafe for the policy path (no GradScaler) -> must fall back to FP32.
    assert isinstance(policy_amp_context(True, torch.float16), nullcontext)


def test_amp_disabled_or_none_is_fp32():
    assert isinstance(policy_amp_context(False, torch.bfloat16), nullcontext)
    assert isinstance(policy_amp_context(True, None), nullcontext)


def test_bf16_autocasts_only_on_cuda():
    ctx = policy_amp_context(True, torch.bfloat16)
    if torch.cuda.is_available():
        assert isinstance(ctx, torch.autocast)
        # And it is BF16, matching what both legs will use.
        with ctx:
            assert torch.get_autocast_dtype("cuda") == torch.bfloat16
    else:
        # No CUDA -> FP32 even if BF16 requested (autocast device_type=cuda needs a device).
        assert isinstance(ctx, nullcontext)


def test_both_legs_share_one_decision():
    # MEDIUM-1: rollout and update both call the SAME factory with the SAME inputs, so the
    # decision is identical by construction. Pin that identical inputs -> identical type.
    for amp_enabled in (True, False):
        for dtype in (torch.bfloat16, torch.float16, None):
            a = policy_amp_context(amp_enabled, dtype)
            b = policy_amp_context(amp_enabled, dtype)
            assert type(a) is type(b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for BF16 autocast")
def test_bf16_rollout_carry_bounded_and_logprob_fp32():
    """MEDIUM-2 stability gate (replaces the falsified NOTE-1 dtype assumption).

    Empirically the custom ResidualLSTM under BF16 autocast carries the hidden state in
    FLOAT16 (cuDNN RNN autocast quirk), NOT FP32. That is fine here: per-layer LayerNorm
    (FP32) keeps activations O(1), so over a 150-step rollout the carry magnitude tracks
    FP32 and stays far below both float16's range and the +/-50 clamp boundary. What MUST
    hold for CRITICAL-1 is that the log_prob is FP32 (the ratio seam) and the carry never
    diverges/NaNs across the horizon.
    """
    from esper.leyline.slot_config import SlotConfig
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic

    device = torch.device("cuda:0")
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)
    policy = FactoredRecurrentActorCritic(state_dim=64, slot_config=slot_config).to(device)
    policy.eval()
    blueprint_indices = torch.full((12, num_slots), -1, dtype=torch.long, device=device)

    hidden = None
    max_abs_c = 0.0
    torch.manual_seed(0)
    for _ in range(150):  # full rollout horizon
        state = torch.randn(12, 64, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
            res = policy.get_action(state, blueprint_indices, hidden=hidden, deterministic=False)
        hidden = res.hidden
        c = hidden[1].float()
        assert torch.isfinite(c).all(), "LSTM carry went non-finite under BF16 autocast"
        max_abs_c = max(max_abs_c, c.abs().max().item())
        # The ratio seam: log_prob MUST be FP32 regardless of the BF16 backbone.
        assert res.log_probs["op"].dtype == torch.float32

    # Carry stays well below the tanh clamp boundary (~50) and float16 range (~65504),
    # so the rollout/reconstruction clamp asymmetry (MEDIUM-2) does not bite in practice.
    assert max_abs_c < 50.0, f"carry max|c|={max_abs_c:.2f} approached the clamp boundary"
