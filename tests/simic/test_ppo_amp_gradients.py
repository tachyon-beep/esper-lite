"""Regression test for B11-PT-01: AMP breaks gradient flow to action heads.

This test verifies that head gradient norms are finite (not NaN) when PPO
update runs under torch.amp.autocast. The bug manifests as all head gradients
being None because log_prob computation in float16 breaks the gradient path.

The fix is to upcast logits to float32 in MaskedCategorical before creating
the Categorical distribution.
"""
import math

import pytest
import torch

from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for AMP gradient test"
)


def _fill_buffer_with_synthetic_data(agent: PPOAgent, slot_config: SlotConfig) -> None:
    """Fill agent buffer with synthetic data for testing update()."""
    state_dim = get_feature_size(slot_config)
    device = agent.device

    hidden = agent.policy.network.get_initial_hidden(1, torch.device(device))

    for env_id in range(agent.buffer.num_envs):
        agent.buffer.start_episode(env_id)
        for step in range(agent.buffer.max_steps_per_env):
            state = torch.randn(1, state_dim, device=device)
            masks = {
                "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool, device=device),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=device),
            }
            bp_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
            pre_hidden = hidden

            result = agent.policy.network.get_action(
                state, bp_indices, hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )
            hidden = result.hidden

            is_done = step == agent.buffer.max_steps_per_env - 1
            agent.buffer.add(
                env_id=env_id,
                state=state.squeeze(0),
                slot_action=result.actions["slot"].item(),
                blueprint_action=result.actions["blueprint"].item(),
                style_action=result.actions["style"].item(),
                tempo_action=result.actions["tempo"].item(),
                alpha_target_action=result.actions["alpha_target"].item(),
                alpha_speed_action=result.actions["alpha_speed"].item(),
                alpha_curve_action=result.actions["alpha_curve"].item(),
                op_action=result.actions["op"].item(),
                effective_op_action=result.actions["op"].item(),
                slot_log_prob=result.log_probs["slot"].item(),
                blueprint_log_prob=result.log_probs["blueprint"].item(),
                style_log_prob=result.log_probs["style"].item(),
                tempo_log_prob=result.log_probs["tempo"].item(),
                alpha_target_log_prob=result.log_probs["alpha_target"].item(),
                alpha_speed_log_prob=result.log_probs["alpha_speed"].item(),
                alpha_curve_log_prob=result.log_probs["alpha_curve"].item(),
                op_log_prob=result.log_probs["op"].item(),
                value=result.values.item(),
                reward=1.0,
                done=is_done,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=pre_hidden[0],
                hidden_c=pre_hidden[1],
                bootstrap_value=0.0,
                blueprint_indices=bp_indices.squeeze(0),
            )
        agent.buffer.end_episode(env_id)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_head_gradients_finite_under_amp(amp_dtype: torch.dtype) -> None:
    """B11-PT-01: Head gradient norms must be finite under AMP.

    This test reproduces the production training loop where agent.update()
    runs inside torch.amp.autocast(). The bug causes all head gradient norms
    to be NaN because gradient flow breaks through float16 log_prob.

    The test FAILS until the fix is applied (upcast logits to float32 in
    MaskedCategorical).
    """
    # Skip bfloat16 if not supported by GPU
    if amp_dtype == torch.bfloat16:
        props = torch.cuda.get_device_properties(0)
        if props.major < 8:  # Ampere+ required for bfloat16
            pytest.skip("bfloat16 requires Ampere or newer GPU")

    slot_config = SlotConfig.default()
    device = "cuda:0"

    # Create policy WITHOUT compile to isolate the AMP issue
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device=device,
        compile_mode="off",
    )

    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=5,
        device=device,
    )

    # Fill buffer with synthetic data
    _fill_buffer_with_synthetic_data(agent, slot_config)

    # Run update inside autocast - this is what vectorized.py does
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        metrics = agent.update(clear_buffer=True)

    # The key assertion: head gradient norms must be finite
    head_grad_norms = metrics.get("head_grad_norms", {})

    # These are the action heads that must have finite gradients
    required_heads = ["slot", "blueprint", "style", "tempo",
                      "alpha_target", "alpha_speed", "alpha_curve", "op"]

    for head_name in required_heads:
        norms = head_grad_norms.get(head_name, [])
        assert len(norms) > 0, f"No gradient norms recorded for {head_name}_head"

        # Check that at least one norm is finite (not NaN)
        finite_norms = [n for n in norms if math.isfinite(n)]
        assert len(finite_norms) > 0, (
            f"B11-PT-01 REGRESSION: {head_name}_head gradient norms are all NaN under AMP. "
            f"Got norms: {norms}. "
            f"This indicates gradient flow is broken through log_prob in {amp_dtype}. "
            f"Fix: upcast logits to float32 in MaskedCategorical before creating Categorical."
        )


def test_head_gradients_finite_without_amp() -> None:
    """Control test: head gradients work correctly without AMP.

    This verifies that the gradient collection logic itself is correct.
    If this test passes but test_head_gradients_finite_under_amp fails,
    the issue is specifically the AMP/float16 interaction.
    """
    slot_config = SlotConfig.default()
    device = "cuda:0"

    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device=device,
        compile_mode="off",
    )

    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=5,
        device=device,
    )

    _fill_buffer_with_synthetic_data(agent, slot_config)

    # Run update WITHOUT autocast
    metrics = agent.update(clear_buffer=True)

    head_grad_norms = metrics.get("head_grad_norms", {})
    required_heads = ["slot", "blueprint", "style", "op"]

    for head_name in required_heads:
        norms = head_grad_norms.get(head_name, [])
        assert len(norms) > 0, f"No gradient norms for {head_name}_head"
        finite_norms = [n for n in norms if math.isfinite(n)]
        assert len(finite_norms) > 0, f"{head_name}_head has no finite gradient norms: {norms}"
