"""Regression test for PPO observation normalization bug.

This test verifies that observations stored in the rollout buffer are normalized,
ensuring that policy.evaluate_actions() during PPO update uses the same observation
distribution as during rollout collection.

Bug: Previously, rollout collection normalized observations before calling
policy.get_action(), but stored raw unnormalized observations in the buffer.
This caused PPO updates to evaluate actions on a different observation distribution,
corrupting importance sampling ratios and violating the on-policy assumption.

Fix: Store normalized observations in the buffer (src/esper/simic/training/vectorized.py:3145).
"""

import torch

from esper.simic.agent.ppo import PPOAgent
from esper.simic.control import RunningMeanStd
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size
from esper.leyline import (
    NUM_BLUEPRINTS,
    NUM_STYLES,
    NUM_TEMPO,
    NUM_ALPHA_TARGETS,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_CURVES,
    NUM_OPS,
)


def test_buffer_stores_normalized_observations():
    """Verify that buffer stores normalized observations used during rollout.

    This test:
    1. Creates a non-identity normalizer (mean ≠ 0, std ≠ 1)
    2. Creates a minimal agent with a buffer
    3. Simulates rollout: normalize obs, get action, store in buffer
    4. Verifies that re-evaluating the policy on buffered states reproduces the stored log_probs

    If the buffer stores raw observations, the log_probs will differ significantly.
    If the buffer stores normalized observations, the log_probs will match (within tolerance).
    """
    device = "cpu"
    batch_size = 4

    # Create slot config for 2-slot architecture
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    state_dim = get_feature_size(slot_config)

    # Create policy
    policy = create_policy(
        policy_type="lstm",
        state_dim=state_dim,
        slot_config=slot_config,
        device=device,
        compile_mode="off",
    )

    # Create agent with buffer
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device=device,
        num_envs=batch_size,
        max_steps_per_env=128,
    )

    # Create observation normalizer with NON-IDENTITY statistics
    # This is critical - if mean=0, var=1, normalization would be a no-op
    obs_normalizer = RunningMeanStd((state_dim,), device=device)
    # Manually set non-identity stats to force actual normalization
    obs_normalizer.mean.fill_(5.0)   # Non-zero mean
    obs_normalizer.var.fill_(4.0)    # var = std^2, so std = 2.0
    obs_normalizer.count = torch.tensor(1000.0, device=device)  # Pretend we've seen enough samples

    # Generate raw observations (these would come from feature extraction)
    raw_obs = torch.randn(batch_size, state_dim, device=device) * 10 + 5  # Range roughly [-25, 35]

    # Normalize observations (this is what rollout collection does)
    normalized_obs = obs_normalizer.normalize(raw_obs)

    # Verify normalization actually changed the observations
    assert not torch.allclose(raw_obs, normalized_obs), \
        "Normalization should change observations (check normalizer stats)"

    # Get actions from policy on NORMALIZED observations (mimics rollout collection)
    # Create dummy blueprint indices and masks
    blueprint_indices = torch.zeros(batch_size, 2, dtype=torch.long, device=device)
    masks = {
        "slot": torch.ones(batch_size, 2, dtype=torch.bool, device=device),
        "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool, device=device),
    }

    with torch.no_grad():
        action_result = agent.policy.get_action(
            normalized_obs,
            blueprint_indices=blueprint_indices,
            masks=masks,
            deterministic=False,
        )

    # Store in buffer (should store NORMALIZED observations, not raw)
    # Simulate what vectorized.py:3145 does after the fix
    for env_idx in range(batch_size):
        agent.buffer.add(
            env_id=env_idx,
            state=normalized_obs[env_idx].detach(),  # THE FIX: Store normalized, not raw
            blueprint_indices=blueprint_indices[env_idx].detach(),
            slot_action=int(action_result.action["slot"][env_idx]),
            blueprint_action=int(action_result.action["blueprint"][env_idx]),
            style_action=int(action_result.action["style"][env_idx]),
            tempo_action=int(action_result.action["tempo"][env_idx]),
            alpha_target_action=int(action_result.action["alpha_target"][env_idx]),
            alpha_speed_action=int(action_result.action["alpha_speed"][env_idx]),
            alpha_curve_action=int(action_result.action["alpha_curve"][env_idx]),
            op_action=int(action_result.action["op"][env_idx]),
            slot_log_prob=float(action_result.log_prob["slot"][env_idx]),
            blueprint_log_prob=float(action_result.log_prob["blueprint"][env_idx]),
            style_log_prob=float(action_result.log_prob["style"][env_idx]),
            tempo_log_prob=float(action_result.log_prob["tempo"][env_idx]),
            alpha_target_log_prob=float(action_result.log_prob["alpha_target"][env_idx]),
            alpha_speed_log_prob=float(action_result.log_prob["alpha_speed"][env_idx]),
            alpha_curve_log_prob=float(action_result.log_prob["alpha_curve"][env_idx]),
            op_log_prob=float(action_result.log_prob["op"][env_idx]),
            value=float(action_result.value[env_idx]),
            reward=1.0,
            done=False,
            slot_mask=masks["slot"][env_idx],
            blueprint_mask=masks["blueprint"][env_idx],
            style_mask=masks["style"][env_idx],
            tempo_mask=masks["tempo"][env_idx],
            alpha_target_mask=masks["alpha_target"][env_idx],
            alpha_speed_mask=masks["alpha_speed"][env_idx],
            alpha_curve_mask=masks["alpha_curve"][env_idx],
            op_mask=masks["op"][env_idx],
            hidden_h=torch.zeros(1, 1, agent.lstm_hidden_dim, device=device),
            hidden_c=torch.zeros(1, 1, agent.lstm_hidden_dim, device=device),
            truncated=False,
            bootstrap_value=0.0,
        )

    # Get data from buffer (this is what PPO update does)
    buffer_data = agent.buffer.get_batched_sequences(device=device)

    # Re-evaluate actions on buffered states (mimics PPO update)
    # This should reproduce the original log_probs if observations are normalized
    stored_actions = {
        "slot": buffer_data["slot_actions"],
        "blueprint": buffer_data["blueprint_actions"],
        "style": buffer_data["style_actions"],
        "tempo": buffer_data["tempo_actions"],
        "alpha_target": buffer_data["alpha_target_actions"],
        "alpha_speed": buffer_data["alpha_speed_actions"],
        "alpha_curve": buffer_data["alpha_curve_actions"],
        "op": buffer_data["op_actions"],
    }
    stored_masks = {
        "slot": buffer_data["slot_masks"],
        "blueprint": buffer_data["blueprint_masks"],
        "style": buffer_data["style_masks"],
        "tempo": buffer_data["tempo_masks"],
        "alpha_target": buffer_data["alpha_target_masks"],
        "alpha_speed": buffer_data["alpha_speed_masks"],
        "alpha_curve": buffer_data["alpha_curve_masks"],
        "op": buffer_data["op_masks"],
    }

    with torch.no_grad():
        eval_result = agent.policy.evaluate_actions(
            buffer_data["states"],  # Should be normalized observations
            buffer_data["blueprint_indices"],
            stored_actions,
            stored_masks,
            hidden=None,  # Initial hidden state (will use zeros internally)
        )

    # Compare re-evaluated log_probs with stored log_probs
    # They should match closely if observations are normalized
    stored_log_probs = {
        "slot": buffer_data["slot_log_probs"],
        "blueprint": buffer_data["blueprint_log_probs"],
        "style": buffer_data["style_log_probs"],
        "tempo": buffer_data["tempo_log_probs"],
        "alpha_target": buffer_data["alpha_target_log_probs"],
        "alpha_speed": buffer_data["alpha_speed_log_probs"],
        "alpha_curve": buffer_data["alpha_curve_log_probs"],
        "op": buffer_data["op_log_probs"],
    }

    # Verify log_probs match for all action heads
    for head_name in stored_log_probs.keys():
        # Allow small tolerance for floating point errors
        assert torch.allclose(
            eval_result.log_prob[head_name],
            stored_log_probs[head_name],
            atol=1e-5,
            rtol=1e-4,
        ), (
            f"{head_name} log_probs don't match! "
            f"This indicates buffer is storing unnormalized observations. "
            f"Expected: {stored_log_probs[head_name][:3]}, "
            f"Got: {eval_result.log_prob[head_name][:3]}"
        )


def test_raw_observations_would_fail():
    """Verify that storing RAW observations would cause log_prob mismatch.

    This is a negative test that demonstrates the bug would occur if we stored
    raw observations instead of normalized ones.
    """
    device = "cpu"
    batch_size = 4

    # Create slot config
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    state_dim = get_feature_size(slot_config)

    # Create policy
    policy = create_policy(
        policy_type="lstm",
        state_dim=state_dim,
        slot_config=slot_config,
        device=device,
        compile_mode="off",
    )

    # Create agent
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device=device,
        num_envs=batch_size,
        max_steps_per_env=128,
    )

    # Create normalizer with non-identity stats
    obs_normalizer = RunningMeanStd((state_dim,), device=device)
    obs_normalizer.mean.fill_(10.0)  # Large mean shift
    obs_normalizer.var.fill_(25.0)   # var = std^2, so std = 5.0
    obs_normalizer.count = torch.tensor(1000.0, device=device)

    # Generate raw observations
    raw_obs = torch.randn(batch_size, state_dim, device=device) * 20 + 10
    normalized_obs = obs_normalizer.normalize(raw_obs)

    # Get actions on normalized observations
    blueprint_indices = torch.zeros(batch_size, 2, dtype=torch.long, device=device)
    masks = {
        "slot": torch.ones(batch_size, 2, dtype=torch.bool, device=device),
        "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool, device=device),
    }

    with torch.no_grad():
        action_result = agent.policy.get_action(
            normalized_obs,  # Get actions on NORMALIZED
            blueprint_indices=blueprint_indices,
            masks=masks,
            deterministic=False,
        )

    # BUGGY CODE: Store RAW observations (this is the bug we fixed)
    for env_idx in range(batch_size):
        agent.buffer.add(
            env_id=env_idx,
            state=raw_obs[env_idx].detach(),  # BUG: Storing raw instead of normalized
            blueprint_indices=blueprint_indices[env_idx].detach(),
            slot_action=int(action_result.action["slot"][env_idx]),
            blueprint_action=int(action_result.action["blueprint"][env_idx]),
            style_action=int(action_result.action["style"][env_idx]),
            tempo_action=int(action_result.action["tempo"][env_idx]),
            alpha_target_action=int(action_result.action["alpha_target"][env_idx]),
            alpha_speed_action=int(action_result.action["alpha_speed"][env_idx]),
            alpha_curve_action=int(action_result.action["alpha_curve"][env_idx]),
            op_action=int(action_result.action["op"][env_idx]),
            slot_log_prob=float(action_result.log_prob["slot"][env_idx]),
            blueprint_log_prob=float(action_result.log_prob["blueprint"][env_idx]),
            style_log_prob=float(action_result.log_prob["style"][env_idx]),
            tempo_log_prob=float(action_result.log_prob["tempo"][env_idx]),
            alpha_target_log_prob=float(action_result.log_prob["alpha_target"][env_idx]),
            alpha_speed_log_prob=float(action_result.log_prob["alpha_speed"][env_idx]),
            alpha_curve_log_prob=float(action_result.log_prob["alpha_curve"][env_idx]),
            op_log_prob=float(action_result.log_prob["op"][env_idx]),
            value=float(action_result.value[env_idx]),
            reward=1.0,
            done=False,
            slot_mask=masks["slot"][env_idx],
            blueprint_mask=masks["blueprint"][env_idx],
            style_mask=masks["style"][env_idx],
            tempo_mask=masks["tempo"][env_idx],
            alpha_target_mask=masks["alpha_target"][env_idx],
            alpha_speed_mask=masks["alpha_speed"][env_idx],
            alpha_curve_mask=masks["alpha_curve"][env_idx],
            op_mask=masks["op"][env_idx],
            hidden_h=torch.zeros(1, 1, agent.lstm_hidden_dim, device=device),
            hidden_c=torch.zeros(1, 1, agent.lstm_hidden_dim, device=device),
            truncated=False,
            bootstrap_value=0.0,
        )

    # Re-evaluate on buffered states (which are RAW in this buggy case)
    buffer_data = agent.buffer.get_batched_sequences(device=device)
    stored_actions = {
        "slot": buffer_data["slot_actions"],
        "blueprint": buffer_data["blueprint_actions"],
        "style": buffer_data["style_actions"],
        "tempo": buffer_data["tempo_actions"],
        "alpha_target": buffer_data["alpha_target_actions"],
        "alpha_speed": buffer_data["alpha_speed_actions"],
        "alpha_curve": buffer_data["alpha_curve_actions"],
        "op": buffer_data["op_actions"],
    }
    stored_masks = {
        "slot": buffer_data["slot_masks"],
        "blueprint": buffer_data["blueprint_masks"],
        "style": buffer_data["style_masks"],
        "tempo": buffer_data["tempo_masks"],
        "alpha_target": buffer_data["alpha_target_masks"],
        "alpha_speed": buffer_data["alpha_speed_masks"],
        "alpha_curve": buffer_data["alpha_curve_masks"],
        "op": buffer_data["op_masks"],
    }

    with torch.no_grad():
        eval_result = agent.policy.evaluate_actions(
            buffer_data["states"],  # These are RAW observations
            buffer_data["blueprint_indices"],
            stored_actions,
            stored_masks,
            hidden=None,
        )

    # With raw observations, log_probs should NOT match
    # (unless by extreme coincidence)
    stored_log_probs = {
        "slot": buffer_data["slot_log_probs"],
    }

    assert not torch.allclose(
        eval_result.log_prob["slot"],
        stored_log_probs["slot"],
        atol=1e-5,
        rtol=1e-4,
    ), "Log probs unexpectedly matched despite raw observations in buffer"
