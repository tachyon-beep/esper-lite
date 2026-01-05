"""Integration test for B11-PT-01: Sanctum UX receives NaN for head gradient averages.

This test reproduces the user's actual complaint:
  When running: uv run python -m esper.scripts.train ppo --devices cuda:0 ...
  The Sanctum UX shows NaN for all action head gradient averages.

The bug is fixed when this test passes - meaning Sanctum receives finite values.
"""
import math

import pytest

# Skip if CUDA not available
pytest.importorskip("torch")
import torch
import torch.amp as torch_amp

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
from esper.tamiyo.policy.action_masks import MaskedCategorical
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - this tests the actual training command"
)


def _fill_buffer(agent: PPOAgent, slot_config: SlotConfig) -> None:
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


def _assert_head_grads_finite(metrics: dict, test_name: str) -> None:
    """Assert that head gradient norms are finite (what Sanctum displays)."""
    head_grad_norms = metrics.get("head_grad_norms", {})

    # These are the heads whose gradients Sanctum displays
    required_heads = ["slot", "op", "blueprint", "style", "tempo",
                      "alpha_target", "alpha_speed", "alpha_curve"]

    for head_name in required_heads:
        norms = head_grad_norms.get(head_name, [])
        assert len(norms) > 0, f"No gradient norms for {head_name}_head"

        finite_norms = [n for n in norms if math.isfinite(n)]
        assert len(finite_norms) > 0, (
            f"B11-PT-01: {head_name}_head gradient norms are all NaN. "
            f"Sanctum will display NaN for head_{head_name}_grad_norm. "
            f"Test: {test_name}"
        )


def test_head_gradients_finite_cuda_default():
    """B11-PT-01: Head grads must be finite on CUDA (default settings).

    This tests the baseline case: CUDA, no compile, no AMP.
    If this fails, there's a fundamental gradient flow issue.
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

    _fill_buffer(agent, slot_config)
    metrics = agent.update(clear_buffer=True)
    _assert_head_grads_finite(metrics, "cuda_default")


def test_head_gradients_finite_cuda_compiled():
    """B11-PT-01: Head grads must be finite with torch.compile.

    Production uses compile_mode="default". If grads are NaN here but not
    in test_head_gradients_finite_cuda_default, the issue is torch.compile.
    """
    slot_config = SlotConfig.default()
    device = "cuda:0"

    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device=device,
        compile_mode="default",
    )

    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=5,
        device=device,
    )

    _fill_buffer(agent, slot_config)
    metrics = agent.update(clear_buffer=True)
    _assert_head_grads_finite(metrics, "cuda_compiled")


def test_head_gradients_finite_cuda_gpu_preload_scenario():
    """B11-PT-01: Head grads must be finite with --gpu-preload --experimental-gpu-preload-gather.

    This tests the user's actual command:
    uv run python -m esper.scripts.train ppo --devices cuda:0 cuda:1 --telemetry-dir ./telemetry
    --sanctum --telemetry-level debug --config-json config.json --gpu-preload --rounds 2000
    --envs 2 --experimental-gpu-preload-gather --task cifar_impaired

    The experimental GPU preload gather path may have different behavior.
    This test runs the PPO agent update in the same way the training loop does.
    """
    slot_config = SlotConfig.default()
    device = "cuda:0"

    # Use compile_mode="default" to match production
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device=device,
        compile_mode="default",
    )

    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=5,
        device=device,
    )

    _fill_buffer(agent, slot_config)

    # No autocast - user's command doesn't have --amp
    metrics = agent.update(clear_buffer=True)

    _assert_head_grads_finite(metrics, "cuda_gpu_preload_scenario")


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_head_gradients_finite_cuda_with_amp(amp_dtype: torch.dtype):
    """B11-PT-01: Head grads must be finite under AMP (if --amp is used).

    This tests what happens when training uses --amp flag.
    The vectorized training loop wraps agent.update() in autocast().
    """
    if amp_dtype == torch.bfloat16:
        props = torch.cuda.get_device_properties(0)
        if props.major < 8:
            pytest.skip("bfloat16 requires Ampere or newer")

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

    _fill_buffer(agent, slot_config)

    # This is exactly what vectorized.py does when amp=True
    with torch_amp.autocast(device_type="cuda", dtype=amp_dtype):
        metrics = agent.update(clear_buffer=True)

    _assert_head_grads_finite(metrics, f"cuda_amp_{amp_dtype}")


def test_gradient_flow_isolation():
    """B11-PT-01 DIAGNOSTIC: Trace exactly where gradients break under AMP.

    Tests the gradient chain: head → logits → MaskedCategorical → log_prob → loss → backward.
    This isolates whether the issue is in MaskedCategorical or elsewhere.
    """
    import torch.nn as nn

    device = "cuda:0"

    # Minimal head mimicking network structure
    head = nn.Linear(32, 5).to(device)

    # Test 1: No AMP - baseline that should always work
    h = torch.randn(4, 32, device=device, requires_grad=True)
    logits = head(h)
    mask = torch.ones(4, 5, dtype=torch.bool, device=device)
    dist = MaskedCategorical(logits=logits, mask=mask)
    actions = torch.randint(0, 5, (4,), device=device)
    log_probs = dist.log_prob(actions)
    loss = -log_probs.mean()
    loss.backward()
    assert head.weight.grad is not None, "No AMP: head.weight.grad is None"
    assert head.weight.grad.isfinite().all(), "No AMP: head.weight.grad has non-finite values"

    # Test 2: With AMP autocast around entire forward + backward
    head = nn.Linear(32, 5).to(device)  # Fresh head
    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        h = torch.randn(4, 32, device=device, requires_grad=True)
        logits = head(h)
        assert logits.dtype == torch.float16, f"Expected float16 logits, got {logits.dtype}"

        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        dist = MaskedCategorical(logits=logits, mask=mask)
        actions = torch.randint(0, 5, (4,), device=device)
        log_probs = dist.log_prob(actions)

        # Check that our fix upcasted to float32
        assert log_probs.dtype == torch.float32, f"Expected float32 log_probs after fix, got {log_probs.dtype}"

        loss = -log_probs.mean()
        loss.backward()

    assert head.weight.grad is not None, (
        "B11-PT-01: head.weight.grad is None under AMP autocast. "
        "The upcast fix in MaskedCategorical is not sufficient."
    )
    assert head.weight.grad.isfinite().all(), (
        "B11-PT-01: head.weight.grad has non-finite values under AMP autocast."
    )

    # Test 3: AMP forward, backward outside autocast (recommended pattern)
    head = nn.Linear(32, 5).to(device)  # Fresh head
    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        h = torch.randn(4, 32, device=device, requires_grad=True)
        logits = head(h)
        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        dist = MaskedCategorical(logits=logits, mask=mask)
        actions = torch.randint(0, 5, (4,), device=device)
        log_probs = dist.log_prob(actions)
        loss = -log_probs.mean()
    # Backward OUTSIDE autocast
    loss.backward()

    assert head.weight.grad is not None, (
        "B11-PT-01: head.weight.grad is None with backward outside autocast."
    )
    assert head.weight.grad.isfinite().all(), (
        "B11-PT-01: head.weight.grad has non-finite values with backward outside autocast."
    )


def test_gradient_flow_with_ratio_computation():
    """B11-PT-01 DIAGNOSTIC: Test gradient flow through PPO ratio computation.

    The PPO update computes: ratio = exp(log_probs - old_log_probs).
    This tests if the subtraction or exp breaks gradient flow.
    """
    import torch.nn as nn

    device = "cuda:0"

    # Test: Ratio computation under AMP (mimics PPO update)
    head = nn.Linear(32, 5).to(device)

    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        h = torch.randn(4, 32, device=device, requires_grad=True)
        logits = head(h)

        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        dist = MaskedCategorical(logits=logits, mask=mask)
        actions = torch.randint(0, 5, (4,), device=device)
        log_probs = dist.log_prob(actions)

        # Simulate old log probs (detached, from buffer - no requires_grad)
        old_log_probs = torch.randn(4, device=device)

        # PPO ratio computation
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Simple policy loss (advantage = 1 for simplicity)
        policy_loss = -(ratio).mean()
        loss = policy_loss

    loss.backward()

    assert head.weight.grad is not None, (
        "B11-PT-01: head.weight.grad is None with ratio computation. "
        "The gradient breaks somewhere in: log_probs → log_ratio → ratio → loss."
    )
    assert head.weight.grad.isfinite().all(), (
        "B11-PT-01: head.weight.grad has non-finite values with ratio computation."
    )


def test_gradient_flow_with_buffer_data():
    """B11-PT-01 DIAGNOSTIC: Test gradient flow when input comes from buffer (no requires_grad).

    In PPO update, states come from the buffer and don't have requires_grad=True.
    This tests if that breaks gradient flow to the head.
    """
    import torch.nn as nn

    device = "cuda:0"

    # Simulate network with feature net + head
    class MiniNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_net = nn.Linear(10, 32)
            self.head = nn.Linear(32, 5)

        def forward(self, x):
            features = self.feature_net(x)
            return self.head(features)

    net = MiniNetwork().to(device)

    # States from buffer DON'T have requires_grad (this is realistic)
    states_from_buffer = torch.randn(4, 10, device=device)  # No requires_grad=True!

    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        logits = net(states_from_buffer)

        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        dist = MaskedCategorical(logits=logits, mask=mask)
        actions = torch.randint(0, 5, (4,), device=device)
        log_probs = dist.log_prob(actions)
        loss = -log_probs.mean()

    loss.backward()

    assert net.head.weight.grad is not None, (
        "B11-PT-01: head.weight.grad is None when input lacks requires_grad. "
        "This suggests buffer data breaks gradient flow."
    )
    assert net.head.weight.grad.isfinite().all(), (
        "B11-PT-01: head.weight.grad has non-finite values."
    )
    assert net.feature_net.weight.grad is not None, (
        "B11-PT-01: feature_net.weight.grad is None too."
    )


def test_gradient_flow_with_full_network():
    """B11-PT-01 DIAGNOSTIC: Test gradient flow with the actual FactoredRecurrentActorCritic.

    This uses the real network to see if the issue is network-specific.
    """
    device = "cuda:0"
    slot_config = SlotConfig.default()
    state_dim = get_feature_size(slot_config)

    # Create the actual network
    net = FactoredRecurrentActorCritic(
        state_dim=state_dim,
        slot_config=slot_config,
    ).to(device)

    # States from buffer (no requires_grad)
    batch, seq = 4, 5
    states = torch.randn(batch, seq, state_dim, device=device)
    bp_indices = torch.zeros(batch, seq, slot_config.num_slots, dtype=torch.long, device=device)
    hidden = net.get_initial_hidden(batch, torch.device(device))

    # Create action dicts
    actions = {
        "slot": torch.randint(0, slot_config.num_slots, (batch, seq), device=device),
        "blueprint": torch.randint(0, NUM_BLUEPRINTS, (batch, seq), device=device),
        "style": torch.randint(0, NUM_STYLES, (batch, seq), device=device),
        "tempo": torch.randint(0, NUM_TEMPO, (batch, seq), device=device),
        "alpha_target": torch.randint(0, NUM_ALPHA_TARGETS, (batch, seq), device=device),
        "alpha_speed": torch.randint(0, NUM_ALPHA_SPEEDS, (batch, seq), device=device),
        "alpha_curve": torch.randint(0, NUM_ALPHA_CURVES, (batch, seq), device=device),
        "op": torch.randint(0, NUM_OPS, (batch, seq), device=device),
    }

    # All-true masks
    masks = {
        "slot": torch.ones(batch, seq, slot_config.num_slots, dtype=torch.bool, device=device),
        "blueprint": torch.ones(batch, seq, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(batch, seq, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(batch, seq, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(batch, seq, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(batch, seq, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(batch, seq, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(batch, seq, NUM_OPS, dtype=torch.bool, device=device),
    }

    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        log_probs, values, entropy, _ = net.evaluate_actions(
            states, bp_indices, actions,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            op_mask=masks["op"],
            hidden=hidden,
        )

        # Simple loss using all log probs
        total_loss = torch.tensor(0.0, device=device)
        for key in log_probs:
            total_loss = total_loss - log_probs[key].mean()
        total_loss = total_loss + values.mean()  # Add value loss

    total_loss.backward()

    # Check slot_head gradients
    slot_head_grads_exist = any(p.grad is not None for p in net.slot_head.parameters())
    slot_head_grads_finite = all(
        p.grad.isfinite().all() for p in net.slot_head.parameters() if p.grad is not None
    )

    assert slot_head_grads_exist, (
        "B11-PT-01: slot_head has no gradients with full network. "
        "Issue is specific to FactoredRecurrentActorCritic."
    )
    assert slot_head_grads_finite, (
        "B11-PT-01: slot_head has non-finite gradients with full network."
    )

    # Check op_head too
    op_head_grads_exist = any(p.grad is not None for p in net.op_head.parameters())
    assert op_head_grads_exist, (
        "B11-PT-01: op_head has no gradients with full network."
    )


def test_gradient_flow_nested_autocast_disabled():
    """B11-PT-01 DIAGNOSTIC: Test if autocast(enabled=False) inside autocast breaks gradients.

    PPO uses: autocast(enabled=True) { ... autocast(enabled=False) { backward() } }
    This tests if that pattern causes gradient issues.
    """
    import torch.nn as nn

    device = "cuda:0"
    head = nn.Linear(32, 5).to(device)

    # Pattern used in PPO: outer autocast + inner autocast(enabled=False) for backward
    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        h = torch.randn(4, 32, device=device)
        logits = head(h)
        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        dist = MaskedCategorical(logits=logits, mask=mask)
        actions = torch.randint(0, 5, (4,), device=device)
        log_probs = dist.log_prob(actions)
        loss = -log_probs.mean()

        # This is EXACTLY what PPO does
        with torch_amp.autocast(device_type="cuda", enabled=False):
            loss.backward()

    assert head.weight.grad is not None, (
        "B11-PT-01: Nested autocast(enabled=False) breaks gradient flow!"
    )
    assert head.weight.grad.isfinite().all(), (
        "B11-PT-01: Nested autocast(enabled=False) causes non-finite gradients!"
    )


def test_log_prob_infinity_under_amp():
    """B11-PT-01 DIAGNOSTIC: Check if -inf log_probs are the real issue under AMP.

    The hypothesis: FP16 masking produces -inf log_probs, leading to
    log_ratio = (-inf) - (-inf) = NaN, which poisons the loss and gradients.

    This test verifies whether MaskedCategorical's upcast fix is sufficient
    to prevent -inf values.
    """
    import torch.nn as nn

    device = "cuda:0"
    head = nn.Linear(32, 5).to(device)

    # Test WITHOUT our upcast fix (simulated by using raw Categorical)
    from torch.distributions import Categorical

    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        h = torch.randn(4, 32, device=device)
        logits = head(h)

        # Apply masking like we do (set invalid to very negative)
        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        mask[:, 3:] = False  # Mask out last 2 actions
        masked_logits = logits.masked_fill(~mask, -1e4)

        # Check if raw fp16 Categorical produces -inf
        raw_dist = Categorical(logits=masked_logits)
        actions = torch.randint(0, 3, (4,), device=device)  # Only valid actions
        raw_log_probs = raw_dist.log_prob(actions)

        has_neg_inf_raw = (raw_log_probs == float('-inf')).any().item()
        min_raw = raw_log_probs.min().item()

    # Test WITH our upcast fix (MaskedCategorical)
    head2 = nn.Linear(32, 5).to(device)

    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        h = torch.randn(4, 32, device=device)
        logits = head2(h)
        mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        mask[:, 3:] = False

        # Use MaskedCategorical which upcasts to float32
        fixed_dist = MaskedCategorical(logits=logits, mask=mask)
        actions = torch.randint(0, 3, (4,), device=device)
        fixed_log_probs = fixed_dist.log_prob(actions)

        has_neg_inf_fixed = (fixed_log_probs == float('-inf')).any().item()
        min_fixed = fixed_log_probs.min().item()

    print("\n=== LOG PROB INFINITY DIAGNOSTIC ===")
    print(f"Raw FP16 Categorical: min={min_raw:.4f}, has_neg_inf={has_neg_inf_raw}")
    print(f"MaskedCategorical (upcast): min={min_fixed:.4f}, has_neg_inf={has_neg_inf_fixed}")

    # The fix should prevent -inf
    assert not has_neg_inf_fixed, (
        f"MaskedCategorical still produces -inf log_probs even with upcast! "
        f"min_log_prob={min_fixed}"
    )


def test_gradient_flow_full_ppo_loss():
    """B11-PT-01 DIAGNOSTIC: Test gradient flow with PPO-style loss computation.

    This mimics PPO's loss more closely:
    - ratio = exp(log_probs - old_log_probs)
    - policy_loss with clipping
    - value_loss
    - entropy_loss
    """
    device = "cuda:0"
    slot_config = SlotConfig.default()
    state_dim = get_feature_size(slot_config)

    net = FactoredRecurrentActorCritic(
        state_dim=state_dim,
        slot_config=slot_config,
    ).to(device)

    batch, seq = 4, 5
    states = torch.randn(batch, seq, state_dim, device=device)
    bp_indices = torch.zeros(batch, seq, slot_config.num_slots, dtype=torch.long, device=device)
    hidden = net.get_initial_hidden(batch, torch.device(device))

    actions = {
        "slot": torch.randint(0, slot_config.num_slots, (batch, seq), device=device),
        "blueprint": torch.randint(0, NUM_BLUEPRINTS, (batch, seq), device=device),
        "style": torch.randint(0, NUM_STYLES, (batch, seq), device=device),
        "tempo": torch.randint(0, NUM_TEMPO, (batch, seq), device=device),
        "alpha_target": torch.randint(0, NUM_ALPHA_TARGETS, (batch, seq), device=device),
        "alpha_speed": torch.randint(0, NUM_ALPHA_SPEEDS, (batch, seq), device=device),
        "alpha_curve": torch.randint(0, NUM_ALPHA_CURVES, (batch, seq), device=device),
        "op": torch.randint(0, NUM_OPS, (batch, seq), device=device),
    }

    masks = {
        "slot": torch.ones(batch, seq, slot_config.num_slots, dtype=torch.bool, device=device),
        "blueprint": torch.ones(batch, seq, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(batch, seq, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(batch, seq, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(batch, seq, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(batch, seq, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(batch, seq, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(batch, seq, NUM_OPS, dtype=torch.bool, device=device),
    }

    # Simulate old log probs (from buffer, no requires_grad)
    old_log_probs = {key: torch.randn(batch, seq, device=device) for key in actions}

    with torch_amp.autocast(device_type="cuda", dtype=torch.float16):
        log_probs, values, entropy, _ = net.evaluate_actions(
            states, bp_indices, actions,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            op_mask=masks["op"],
            hidden=hidden,
        )

        # PPO-style loss computation
        clip_ratio = 0.2
        policy_loss = torch.tensor(0.0, device=device)
        for key in log_probs:
            log_ratio = log_probs[key] - old_log_probs[key]
            ratio = torch.exp(log_ratio)
            advantage = torch.ones_like(ratio)  # Simplified

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
            clipped_surr = torch.min(surr1, surr2)
            head_loss = -clipped_surr.mean()
            policy_loss = policy_loss + head_loss

        value_loss = values.mean() ** 2
        entropy_loss = torch.tensor(0.0, device=device)
        for key in entropy:
            entropy_loss = entropy_loss - entropy[key].mean()

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # PPO pattern: autocast(enabled=False) for backward
        with torch_amp.autocast(device_type="cuda", enabled=False):
            total_loss.backward()

    slot_head_grads = [p.grad for p in net.slot_head.parameters()]
    slot_grads_exist = any(g is not None for g in slot_head_grads)

    assert slot_grads_exist, (
        f"B11-PT-01: slot_head has no gradients with PPO-style loss. "
        f"Grads: {[g is not None for g in slot_head_grads]}"
    )

    op_head_grads = [p.grad for p in net.op_head.parameters()]
    op_grads_exist = any(g is not None for g in op_head_grads)

    assert op_grads_exist, (
        f"B11-PT-01: op_head has no gradients with PPO-style loss. "
        f"Grads: {[g is not None for g in op_head_grads]}"
    )
