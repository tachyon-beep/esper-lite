"""Tests for PPO module."""
import math
import warnings
import pytest
import torch

from esper.leyline import (
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_VALUE_CLIP,
    LifecycleOp,
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
from esper.tamiyo.policy.features import get_feature_size, batch_obs_to_features


def test_ppo_agent_architecture():
    """PPOAgent should use FactoredRecurrentActorCritic and TamiyoRolloutBuffer."""
    from esper.simic.agent import TamiyoRolloutBuffer
    from esper.tamiyo.networks import FactoredRecurrentActorCritic

    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=4,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
    )

    # Direct type checks
    assert isinstance(agent.buffer, TamiyoRolloutBuffer)
    assert isinstance(agent.policy.network, FactoredRecurrentActorCritic)


def test_kl_early_stopping_triggers():
    """Verify approx_kl is computed and can trigger early stopping."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=5,
        target_kl=0.001,  # Very low to ensure triggering
        recurrent_n_epochs=5,  # Multiple epochs to allow early stop
        device="cpu",
    )

    # Fill buffer with synthetic data
    state_dim = get_feature_size(slot_config)
    hidden = agent.policy.network.get_initial_hidden(1, torch.device(agent.device))
    for env_id in range(2):
        agent.buffer.start_episode(env_id)
        for step in range(5):
            state = torch.randn(1, state_dim, device=agent.device)
            masks = {
                "slot": torch.ones(1, 3, dtype=torch.bool, device=agent.device),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=agent.device),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=agent.device),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=agent.device),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=agent.device),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=agent.device),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=agent.device),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=agent.device),
            }
            pre_hidden = hidden
            bp_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=agent.device)
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
            hidden = result.hidden  # Update hidden for next step
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
                done=step == 4,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                # Store PRE-step hidden (input to get_action) for BPTT reconstruction.
                hidden_h=pre_hidden[0],
                hidden_c=pre_hidden[1],
                bootstrap_value=0.0,
                blueprint_indices=bp_indices.squeeze(0),
            )
        agent.buffer.end_episode(env_id)

    metrics = agent.update(clear_buffer=True)

    # approx_kl must be computed (not always 0.0)
    assert "approx_kl" in metrics, "approx_kl should be in metrics"
    # With very low target_kl, early stopping should trigger
    assert "early_stop_epoch" in metrics or metrics.get("approx_kl", 0) > 0, \
        "Either early stopping triggered or KL was computed"


def test_kl_early_stopping_aborts_on_multiepoch_drift():
    """target_kl must abort a later epoch once the policy drifts off the θ0 anchor.

    Under the anchored-reference-pass design, the PPO importance ratio is measured
    against ref_log_probs captured by a frozen-θ0 no_grad anchor pass at the START of
    update() — NOT against the rollout-collection log_probs in the buffer. This makes
    the epoch-0 ratio identically 1.0 (scored θ == anchor θ0), so approx_kl ≈ 0 and
    epoch 0 can never early-stop. The KL trust region is therefore exercised across
    MULTIPLE epochs: epoch 0 takes a real optimizer step, the weights drift away from
    the frozen anchor, and at a later epoch approx_kl crosses 1.5 * target_kl and the
    update is aborted BEFORE that epoch's optimizer.step().

    (The previous version of this test injected fake rollout log_probs of -10.0 to
    force an epoch-0 abort. That premise — a rollout-vs-network log_prob mismatch —
    is exactly what the anchor deliberately eliminates: the rollout log_probs no longer
    feed the ratio baseline at all, so the injection can no longer move approx_kl. The
    drift must now come from genuine weight updates across epochs.)
    """
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=5,
        # Extremely low so the first real optimizer step's drift already crosses it.
        target_kl=1e-8,
        recurrent_n_epochs=4,  # Need >1 epoch so post-step drift can be measured.
        device="cpu",
    )

    # Fill the buffer by scoring with the live network so the stored rollout log_probs
    # are genuine. Their values are irrelevant to the ratio under the anchor design,
    # but using real ones keeps the rollout self-consistent.
    state_dim = get_feature_size(slot_config)
    hidden = agent.policy.network.get_initial_hidden(1, torch.device(agent.device))
    for env_id in range(2):
        agent.buffer.start_episode(env_id)
        for step in range(5):
            state = torch.randn(1, state_dim, device=agent.device)
            masks = {
                "slot": torch.ones(1, 3, dtype=torch.bool, device=agent.device),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=agent.device),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=agent.device),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=agent.device),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=agent.device),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=agent.device),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=agent.device),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=agent.device),
            }
            pre_hidden = hidden
            bp_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=agent.device)
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
                done=step == 4,
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

    metrics = agent.update(clear_buffer=True)

    # Early stop must fire, and it must be at a LATER epoch (>=1): epoch 0 is anchored
    # to θ0 so its ratio is 1.0 and approx_kl ≈ 0 — it can never trip the trust region.
    assert "early_stop_epoch" in metrics, \
        "Anchor design: post-θ0 drift must trip the KL trust region within recurrent_n_epochs"
    assert metrics["early_stop_epoch"] >= 1, \
        "Epoch 0 is anchored to θ0 (ratio==1.0, approx_kl≈0); abort must occur at a later epoch"

    # The aborting epoch (>=1) short-circuits BEFORE its own optimizer.step(). But epoch 0
    # DID complete a real step (that step is what produced the drift), so the update as a
    # whole performed work: ppo_update_performed is True and policy_loss reflects the
    # completed epoch(s), not a NaN. (The all-NaN / ppo_update_performed=False no-step
    # contract only applies when ZERO epochs complete a step — impossible here because the
    # θ0 anchor guarantees epoch 0's ratio==1.0, so epoch 0 never self-aborts.)
    assert metrics["ppo_update_performed"] is True, \
        "Epoch 0 completed a real optimizer step before the later-epoch KL abort"
    assert math.isfinite(metrics["policy_loss"]), \
        "policy_loss should reflect the completed pre-abort epoch(s), not NaN"


def test_value_clipping_uses_appropriate_range():
    """Verify value clipping doesn't use the policy clip ratio."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        clip_ratio=0.2,  # Policy clip
        clip_value=True,
        value_clip=DEFAULT_VALUE_CLIP,
        device="cpu",
    )

    # Value clip should be much larger than policy clip
    assert agent.value_clip == DEFAULT_VALUE_CLIP, "Agent should have value_clip matching leyline default"
    assert agent.value_clip > agent.clip_ratio, "Value clip should be larger than policy clip"


def test_value_clipping_disabled_option():
    """Verify clip_value=False disables value clipping entirely."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        clip_value=False,
        device="cpu",
    )
    assert agent.clip_value is False, "clip_value should be configurable to False"


def test_no_recurrent_staleness_warning_at_k4():
    """Under the anchored full-recompute-TBPTT design, multi-epoch recurrent PPO is
    mathematically exact, so the old hidden-state staleness UserWarning must be gone."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PPOAgent(
            policy=policy,
            slot_config=slot_config,
            recurrent_n_epochs=4,
            clip_value=False,
            device="cpu",
        )
    staleness = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and (
            "Hidden states stored from" in str(w.message)
            or "stale value comparisons" in str(w.message)
            or "may slow learning" in str(w.message)
        )
    ]
    assert not staleness, (
        f"No recurrent staleness UserWarning expected at K=4, got: "
        f"{[str(w.message) for w in staleness]}"
    )


def test_recurrent_n_epochs_below_one_raises() -> None:
    """recurrent_n_epochs < 1 must fail fast (PR2 exposes this at config level).

    Without the guard, recurrent_n_epochs=0 makes the epoch loop run zero times -> no
    optimizer step, ppo_update_performed=False -> silent no-op training. Fail loudly instead.
    """
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm", slot_config=slot_config, device="cpu", compile_mode="off"
    )
    with pytest.raises(ValueError, match="recurrent_n_epochs must be >= 1"):
        PPOAgent(
            policy=policy, slot_config=slot_config, recurrent_n_epochs=0, device="cpu"
        )


def test_clip_value_with_multiepoch_raises() -> None:
    """clip_value=True is disallowed under recurrent_n_epochs > 1 (design constraint).

    Value clipping anchors on rollout old_values not recomputed at theta0; under multi-epoch
    it would clip against a stale reference. Replaces the deleted staleness UserWarning with a
    hard fail-fast guard (re-enabling needs an anchored ref_values pass, a separate task).
    """
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm", slot_config=slot_config, device="cpu", compile_mode="off"
    )
    with pytest.raises(ValueError, match="clip_value=True is incompatible with recurrent_n_epochs"):
        PPOAgent(
            policy=policy, slot_config=slot_config,
            recurrent_n_epochs=4, clip_value=True, device="cpu",
        )


def test_weight_decay_optimizer_covers_all_network_params() -> None:
    """Weight-decay optimizer groups must include every network parameter.

    Regression guard: when weight_decay>0, PPOAgent uses custom AdamW param groups
    (actor/shared/critic). Missing a module in the grouping silently freezes it.
    """
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        weight_decay=0.01,
        device="cpu",
    )

    network_params = {id(p) for p in agent.policy.network.parameters()}

    opt_params = [p for group in agent.optimizer.param_groups for p in group["params"]]
    opt_param_ids = [id(p) for p in opt_params]

    assert len(opt_param_ids) == len(set(opt_param_ids)), (
        "Optimizer has duplicate parameters across param groups"
    )

    optimizer_params = set(opt_param_ids)

    missing = network_params - optimizer_params
    extra = optimizer_params - network_params

    missing_names = [
        name for name, p in agent.policy.network.named_parameters() if id(p) in missing
    ]

    assert not missing_names, f"Optimizer missing network params: {missing_names}"
    assert not extra, "Optimizer has params not in network"


def test_head_grad_norms_includes_tempo_head() -> None:
    """Per-head gradient norm telemetry must include tempo head values (P4-6)."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=3,
        target_kl=None,  # Ensure we reach backward() (no early stopping)
        device="cpu",
    )

    device = torch.device(agent.device)
    state_dim = get_feature_size(slot_config)
    hidden = agent.policy.network.get_initial_hidden(1, device)

    agent.buffer.start_episode(env_id=0)
    for step in range(3):
        state = torch.randn(1, state_dim, device=device)
        masks = {
            "slot": torch.ones(1, 3, dtype=torch.bool, device=device),
            "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
            "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
            "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
            "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
            "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
            "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
            # Force GERMINATE so tempo head is causally relevant.
            "op": torch.zeros(1, NUM_OPS, dtype=torch.bool, device=device),
        }
        masks["op"][:, LifecycleOp.GERMINATE] = True

        pre_hidden = hidden
        bp_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
        result = agent.policy.network.get_action(
            state,
            bp_indices,
            hidden,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            op_mask=masks["op"],
        )
        hidden = result.hidden  # Update hidden for next step

        agent.buffer.add(
            env_id=0,
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
            done=step == 2,
            truncated=False,
            slot_mask=masks["slot"].squeeze(0),
            blueprint_mask=masks["blueprint"].squeeze(0),
            style_mask=masks["style"].squeeze(0),
            tempo_mask=masks["tempo"].squeeze(0),
            alpha_target_mask=masks["alpha_target"].squeeze(0),
            alpha_speed_mask=masks["alpha_speed"].squeeze(0),
            alpha_curve_mask=masks["alpha_curve"].squeeze(0),
            op_mask=masks["op"].squeeze(0),
            # Store PRE-step hidden (input to get_action) for BPTT reconstruction.
            hidden_h=pre_hidden[0],
            hidden_c=pre_hidden[1],
            bootstrap_value=0.0,
            blueprint_indices=bp_indices.squeeze(0),
        )
    agent.buffer.end_episode(env_id=0)

    metrics = agent.update(clear_buffer=True)
    head_grad_norms = metrics["head_grad_norms"]

    assert head_grad_norms["tempo"], "tempo grad norm history must not be empty"
    assert len(head_grad_norms["tempo"]) == len(head_grad_norms["slot"])


def test_head_grad_norms_are_finite_on_first_update() -> None:
    """Per-head gradient norms must be finite (not NaN) after first PPO update.

    BUG: On first episode, all head_*_grad_norm values were NaN in Sanctum UX.
    This test verifies that gradients flow correctly to all heads and produce
    finite values after backward(), not NaN.

    The fix is successful when this test passes: all heads have finite gradient norms.
    """
    import math

    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=3,
        target_kl=None,  # Ensure we reach backward() (no early stopping)
        device="cpu",
    )

    device = torch.device(agent.device)
    state_dim = get_feature_size(slot_config)
    hidden = agent.policy.network.get_initial_hidden(1, device)

    agent.buffer.start_episode(env_id=0)
    for step in range(3):
        state = torch.randn(1, state_dim, device=device)
        masks = {
            "slot": torch.ones(1, 3, dtype=torch.bool, device=device),
            "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
            "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
            "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
            "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
            "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
            "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
            # Force GERMINATE so all heads are causally relevant.
            "op": torch.zeros(1, NUM_OPS, dtype=torch.bool, device=device),
        }
        masks["op"][:, LifecycleOp.GERMINATE] = True

        pre_hidden = hidden
        bp_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
        result = agent.policy.network.get_action(
            state,
            bp_indices,
            hidden,
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

        agent.buffer.add(
            env_id=0,
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
            done=step == 2,
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
    agent.buffer.end_episode(env_id=0)

    metrics = agent.update(clear_buffer=True)
    head_grad_norms = metrics["head_grad_norms"]

    # All heads must have finite gradient norms (not NaN, not Inf)
    expected_heads = ["slot", "blueprint", "style", "tempo",
                      "alpha_target", "alpha_speed", "alpha_curve", "op", "value"]

    for head_name in expected_heads:
        assert head_name in head_grad_norms, f"Missing head: {head_name}"
        norms = head_grad_norms[head_name]
        assert len(norms) > 0, f"Empty gradient norm list for {head_name}"

        for i, norm in enumerate(norms):
            assert math.isfinite(norm), (
                f"head_grad_norms[{head_name}][{i}] = {norm} is not finite. "
                f"BUG: Gradients not flowing to {head_name} head after backward(). "
                f"Full list: {norms}"
            )


def test_signals_to_features_with_multislot_params():
    """Test batch_obs_to_features V3 API with 3-slot config."""
    from esper.leyline import LifecycleOp

    # Create minimal signals mock
    class MockMetrics:
        epoch = 10
        global_step = 100
        train_loss = 0.5
        val_loss = 0.6
        loss_delta = -0.1
        train_accuracy = 85.0
        val_accuracy = 82.0
        accuracy_delta = 0.5
        plateau_epochs = 2
        best_val_accuracy = 83.0
        best_val_loss = 0.55
        grad_norm_host = 1.0

    class MockSignals:
        metrics = MockMetrics()
        loss_history = [0.8, 0.7, 0.6, 0.5, 0.5]
        accuracy_history = [70.0, 75.0, 80.0, 82.0, 85.0]

    class MockEnvState:
        last_action_success = True
        last_action_op = LifecycleOp.WAIT.value
        gradient_health_prev = {}
        epochs_since_counterfactual = {}

    slot_config = SlotConfig.default()  # 3 slots
    batch_signals = [MockSignals()]
    batch_slot_reports = [{}]  # Empty slot reports (all inactive)
    batch_env_states = [MockEnvState()]

    obs, blueprint_indices = batch_obs_to_features(
        batch_signals=batch_signals,
        batch_slot_reports=batch_slot_reports,
        batch_env_states=batch_env_states,
        slot_config=slot_config,
        device=torch.device("cpu"),
        max_epochs=100,
    )

    # Obs V3: 23 base + 31*3 slots = 116 dims (excluding blueprint embeddings)
    expected_dim = get_feature_size(slot_config)
    assert obs.shape == (1, expected_dim)
    assert obs.shape[1] == 116


def test_signals_to_features_telemetry_slot_alignment() -> None:
    """Telemetry features embedded in slot features align to slot config order."""
    from esper.leyline import SeedMetrics, SeedStage, SeedStateReport, SeedTelemetry, LifecycleOp

    class MockMetrics:
        epoch = 7
        global_step = 100
        train_loss = 0.5
        val_loss = 0.6
        loss_delta = -0.1
        train_accuracy = 85.0
        val_accuracy = 82.0
        accuracy_delta = 0.5
        plateau_epochs = 2
        best_val_accuracy = 83.0
        best_val_loss = 0.55
        grad_norm_host = 1.0

    class MockSignals:
        metrics = MockMetrics()
        loss_history = []
        accuracy_history = []

    class MockEnvState:
        last_action_success = True
        last_action_op = LifecycleOp.WAIT.value
        gradient_health_prev = {"r0c1": 0.8}  # Previous gradient health
        epochs_since_counterfactual = {"r0c1": 2}  # 2 epochs since last CF

    mid_telemetry = SeedTelemetry(seed_id="s1", blueprint_id="norm")
    mid_telemetry.gradient_norm = 2.0
    mid_telemetry.gradient_health = 0.7
    mid_telemetry.has_vanishing = True
    mid_telemetry.has_exploding = False
    mid_telemetry.epochs_in_stage = 4
    mid_telemetry.accuracy = 65.0
    mid_telemetry.accuracy_delta = 1.0
    mid_telemetry.stage = SeedStage.TRAINING.value
    mid_telemetry.alpha = 0.3
    mid_telemetry.epoch = 7
    mid_telemetry.max_epochs = 25

    slot_reports = {
        "r0c1": SeedStateReport(
            seed_id="s1",
            slot_id="r0c1",
            blueprint_id="norm",
            blueprint_index=3,  # norm blueprint index
            stage=SeedStage.TRAINING,
            metrics=SeedMetrics(
                epochs_total=7,
                current_alpha=0.3,
                counterfactual_contribution=1.5,
                contribution_velocity=0.2,
                epochs_in_current_stage=4,
                interaction_sum=0.0,
            ),
            telemetry=mid_telemetry,
            blend_tempo_epochs=5,
            alpha_target=0.5,
            alpha_mode=0,
            alpha_steps_total=10,
            alpha_steps_done=3,
            time_to_target=7,
            alpha_velocity=0.05,
            alpha_algorithm=0,
        )
    }

    slot_config = SlotConfig.default()  # 3 slots: r0c0, r0c1, r0c2
    batch_signals = [MockSignals()]
    batch_slot_reports = [slot_reports]
    batch_env_states = [MockEnvState()]

    obs, blueprint_indices = batch_obs_to_features(
        batch_signals=batch_signals,
        batch_slot_reports=batch_slot_reports,
        batch_env_states=batch_env_states,
        slot_config=slot_config,
        device=torch.device("cpu"),
        max_epochs=100,
    )

    # V3: Telemetry is embedded in slot features (4 dims: gradient_norm, gradient_health, has_vanishing, has_exploding)
    # Base features: 23 dims
    # Slot 0 (r0c0): 31 dims - all zeros (inactive)
    # Slot 1 (r0c1): 31 dims - active with telemetry
    # Slot 2 (r0c2): 31 dims - all zeros (inactive)

    # Extract slot 1 features (r0c1) - starts at index 23 + 31 = 54
    slot1_start = 23 + 31  # Skip base + slot0
    slot1_features = obs[0, slot1_start:slot1_start + 31].tolist()

    # Slot features layout (31 dims):
    # [0] is_active = 1.0
    # [1-10] stage one-hot
    # [11] current_alpha
    # [12] contribution
    # [13] contribution_velocity
    # [14] blend_tempo_epochs
    # [15-22] alpha scaffolding (8 dims)
    # [23-26] telemetry merged (4 dims): gradient_norm, gradient_health, has_vanishing, has_exploding
    # [27] gradient_health_prev
    # [28] epochs_in_stage_norm
    # [29] counterfactual_fresh
    # [30] seed_age_norm

    # Check telemetry fields are present (indices 23-26 in slot features)
    assert slot1_features[23] > 0.0  # gradient_norm (normalized, should be > 0)
    assert 0.0 <= slot1_features[24] <= 1.0  # gradient_health
    assert slot1_features[25] == 1.0  # has_vanishing = True
    assert slot1_features[26] == 0.0  # has_exploding = False
    assert abs(slot1_features[30] - 0.07) < 1e-6  # seed_age_norm (epochs_total=7, max_epochs=100)

    # Check slot 0 and slot 2 are all zeros (inactive)
    slot0_start = 23
    slot0_features = obs[0, slot0_start:slot0_start + 31].tolist()
    assert slot0_features == [0.0] * 31  # r0c0 (disabled)

    slot2_start = 23 + 62  # Skip base + slot0 + slot1
    slot2_features = obs[0, slot2_start:slot2_start + 31].tolist()
    assert slot2_features == [0.0] * 31  # r0c2 (disabled)


def test_ppo_agent_accepts_slot_config():
    """PPOAgent should accept slot_config and derive state_dim from it."""
    slot_config = SlotConfig.default()  # 3 slots
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
    )

    expected_state_dim = get_feature_size(slot_config)  # 23 + 3*9 = 50
    assert agent.slot_config == slot_config
    assert agent.policy.network.state_dim == expected_state_dim


def test_ppo_agent_with_3_slot_config():
    """PPOAgent with 3-slot config should have state_dim=50."""
    slot_config = SlotConfig.default()  # 3 slots (r0c0, r0c1, r0c2)
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
    )

    expected_state_dim = get_feature_size(slot_config)  # 23 + 3*9 = 50
    assert agent.policy.network.state_dim == expected_state_dim
    assert agent.policy.network.num_slots == 3


def test_ppo_agent_with_5_slot_config():
    """PPOAgent with 5-slot config should have state_dim=68."""
    # Create a 5-slot config
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
    )

    expected_state_dim = get_feature_size(slot_config)  # 23 + 5*9 = 68
    assert agent.policy.network.state_dim == expected_state_dim
    assert agent.policy.network.num_slots == 5


def test_ppo_agent_network_slot_head_matches_config():
    """Network's slot head size should match slot_config.num_slots."""
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0"))  # 4 slots
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
    )

    # Verify network was initialized with correct num_slots
    assert agent.policy.network.num_slots == 4


def test_ppo_agent_buffer_matches_slot_config():
    """PPOAgent buffer should match slot_config passed during initialization."""
    slot_config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device="cpu",
        num_envs=2,
        max_steps_per_env=10,
    )

    # Buffer should have matching slot_config
    assert agent.buffer.num_slots == 5
    assert agent.buffer.slot_config == slot_config
    # slot_masks should have correct shape
    assert agent.buffer.slot_masks.shape == (2, 10, 5)


def test_ppo_agent_full_update_with_5_slots():
    """Full PPO update cycle with 5-slot config should work correctly."""
    slot_config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device="cpu",
        num_envs=1,
        max_steps_per_env=5,
        target_kl=None,  # Disable KL early stopping - test uses fake log_probs
    )

    # Add some transitions to buffer
    agent.buffer.start_episode(env_id=0)
    for i in range(5):
        agent.buffer.add(
            env_id=0,
            state=torch.randn(agent.policy.network.state_dim),
            slot_action=i % 5,  # Use all 5 slots
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
            effective_op_action=0,
            slot_log_prob=-1.0,
            blueprint_log_prob=-1.0,
            style_log_prob=-1.0,
            tempo_log_prob=-1.0,
            alpha_target_log_prob=-1.0,
            alpha_speed_log_prob=-1.0,
            alpha_curve_log_prob=-1.0,
            op_log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=(i == 4),
            slot_mask=torch.ones(5, dtype=torch.bool),  # 5 slots
            blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
            hidden_h=torch.zeros(1, 1, agent.policy.hidden_dim),
            hidden_c=torch.zeros(1, 1, agent.policy.hidden_dim),
            blueprint_indices=torch.zeros(5, dtype=torch.long),  # 5 slots
        )
    agent.buffer.end_episode(env_id=0)

    # Run PPO update - should not crash
    metrics = agent.update()

    # Should have produced metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_ppo_update_collects_q_values():
    """PPO update collects Q(s,op) for all ops and computes variance."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=5,
        target_kl=None,  # Disable KL early stopping
        device="cpu",
    )

    # Add some transitions to buffer
    agent.buffer.start_episode(env_id=0)
    op_mask = torch.tensor([True, True, False, False, True, False], dtype=torch.bool)
    for i in range(5):
        agent.buffer.add(
            env_id=0,
            state=torch.randn(agent.policy.network.state_dim),
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
            effective_op_action=0,
            slot_log_prob=-1.0,
            blueprint_log_prob=-1.0,
            style_log_prob=-1.0,
            tempo_log_prob=-1.0,
            alpha_target_log_prob=-1.0,
            alpha_speed_log_prob=-1.0,
            alpha_curve_log_prob=-1.0,
            op_log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=(i == 4),
            truncated=False,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=op_mask,
            hidden_h=torch.zeros(1, 1, agent.policy.hidden_dim),
            hidden_c=torch.zeros(1, 1, agent.policy.hidden_dim),
            bootstrap_value=0.0,
            blueprint_indices=torch.zeros(3, dtype=torch.long),
        )
    agent.buffer.end_episode(env_id=0)

    # Trigger update
    metrics = agent.update()

    # Verify Q-values were collected
    assert "op_q_values" in metrics
    assert "op_valid_mask" in metrics
    assert "q_variance" in metrics
    assert "q_spread" in metrics

    assert metrics["op_valid_mask"] == tuple(op_mask.tolist())
    assert len(metrics["op_q_values"]) == NUM_OPS

    valid_qs = [
        q for q, is_valid in zip(metrics["op_q_values"], metrics["op_valid_mask"]) if is_valid
    ]
    assert len(valid_qs) >= 2

    valid_q_tensor = torch.tensor(valid_qs)
    expected_variance = valid_q_tensor.var().item()
    expected_spread = (valid_q_tensor.max() - valid_q_tensor.min()).item()
    assert math.isclose(metrics["q_variance"], expected_variance, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["q_spread"], expected_spread, rel_tol=1e-6, abs_tol=1e-6)


def test_ppo_update_singleton_valid_op_q_metrics_are_finite():
    """Singleton op masks should not look like numerical instability."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=5,
        target_kl=None,
        device="cpu",
    )

    agent.buffer.start_episode(env_id=0)
    op_mask = torch.tensor([True, False, False, False, False, False], dtype=torch.bool)
    for i in range(5):
        agent.buffer.add(
            env_id=0,
            state=torch.randn(agent.policy.network.state_dim),
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
            effective_op_action=0,
            slot_log_prob=-1.0,
            blueprint_log_prob=-1.0,
            style_log_prob=-1.0,
            tempo_log_prob=-1.0,
            alpha_target_log_prob=-1.0,
            alpha_speed_log_prob=-1.0,
            alpha_curve_log_prob=-1.0,
            op_log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=(i == 4),
            truncated=False,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=op_mask,
            hidden_h=torch.zeros(1, 1, agent.policy.hidden_dim),
            hidden_c=torch.zeros(1, 1, agent.policy.hidden_dim),
            bootstrap_value=0.0,
            blueprint_indices=torch.zeros(3, dtype=torch.long),
        )
    agent.buffer.end_episode(env_id=0)

    metrics = agent.update()

    assert metrics["op_valid_mask"] == tuple(op_mask.tolist())
    assert math.isfinite(metrics["q_variance"])
    assert math.isfinite(metrics["q_spread"])
    assert metrics["q_variance"] == 0.0
    assert metrics["q_spread"] == 0.0


def test_bptt_invariant_assertion_fires():
    """PPOAgent must reject max_steps_per_env > chunk_length (BPTT invariant guard).

    Regression for esper-lite-415bf1347d: a buffer row longer than chunk_length would
    pack multiple episodes into one BPTT sequence; evaluate_actions does not reset
    hidden state at done boundaries, so recurrent gradients would leak across episode
    boundaries (present in the update leg but absent in rollout) -> silent gradient bias.
    The guard converts that latent trap into a loud construction-time failure.
    """
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )

    with pytest.raises(ValueError, match="BPTT INVARIANT VIOLATED"):
        PPOAgent(
            policy=policy,
            slot_config=slot_config,
            num_envs=2,
            chunk_length=5,
            max_steps_per_env=6,  # > chunk_length: one row could hold >1 episode
            device="cpu",
        )


def test_bptt_invariant_allows_equal_boundary():
    """max_steps_per_env == chunk_length is valid (exactly one episode per row).

    Confirms the guard is '>' not '>=': the boundary-equal config (the production
    invariant max_steps_per_env == chunk_length == max_epochs) must construct cleanly.
    """
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )

    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        chunk_length=5,
        max_steps_per_env=5,  # equal: exactly one episode per row
        device="cpu",
    )
    assert agent.max_steps_per_env == 5
    assert agent.chunk_length == 5
