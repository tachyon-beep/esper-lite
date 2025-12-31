"""Tests for PPO module."""
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
from esper.simic.training.parallel_env_state import ParallelEnvState


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


def test_kl_early_stopping_with_single_epoch():
    """BUG-003 regression: target_kl must work with recurrent_n_epochs=1.

    Previously, KL early stopping only checked at the START of the next epoch.
    With n_epochs=1, there was no "next epoch" so target_kl was a no-op.

    The fix moves the KL check BEFORE optimizer.step(), so even with n_epochs=1,
    an update can be skipped if KL is already too high (e.g., from drift since rollout).
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
        target_kl=0.0001,  # Extremely low to ensure triggering
        recurrent_n_epochs=1,  # The critical case from BUG-003
        device="cpu",
    )

    # Fill buffer with FAKE log_probs that differ from network's actual output
    # This simulates policy drift since rollout collection
    state_dim = get_feature_size(slot_config)
    for env_id in range(2):
        agent.buffer.start_episode(env_id)
        for step in range(5):
            agent.buffer.add(
                env_id=env_id,
                state=torch.randn(state_dim),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
                alpha_target_action=0,
                alpha_speed_action=0,
                alpha_curve_action=0,
                op_action=0,
                # Fake log_probs that are very different from network's actual output
                slot_log_prob=-10.0,  # Network won't produce this
                blueprint_log_prob=-10.0,
                style_log_prob=-10.0,
                tempo_log_prob=-10.0,
                alpha_target_log_prob=-10.0,
                alpha_speed_log_prob=-10.0,
                alpha_curve_log_prob=-10.0,
                op_log_prob=-10.0,
                value=1.0,
                reward=1.0,
                done=step == 4,
                truncated=False,
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, agent.policy.hidden_dim),
                hidden_c=torch.zeros(1, 1, agent.policy.hidden_dim),
                bootstrap_value=0.0,
                blueprint_indices=torch.zeros(slot_config.num_slots, dtype=torch.long),
            )
        agent.buffer.end_episode(env_id)

    metrics = agent.update(clear_buffer=True)

    # With BUG-003 fix: KL check happens BEFORE optimizer.step()
    # So with n_epochs=1 and extreme policy drift, we should early stop
    assert "early_stop_epoch" in metrics, \
        "BUG-003 regression: early stopping should work with n_epochs=1"
    assert metrics["early_stop_epoch"] == 0, \
        "Should have early stopped at epoch 0 (the only epoch)"

    # Key assertion: NO update should have happened
    # (policy_loss won't be in metrics because we broke before computing it)
    assert "policy_loss" not in metrics, \
        "No policy_loss should be computed when early stopping at epoch 0"


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
    )

    # Obs V3: 24 base + 30*3 slots = 114 dims (excluding blueprint embeddings)
    expected_dim = get_feature_size(slot_config)
    assert obs.shape == (1, expected_dim)
    assert obs.shape[1] == 114


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
    )

    # V3: Telemetry is embedded in slot features (4 dims: gradient_norm, gradient_health, has_vanishing, has_exploding)
    # Base features: 24 dims
    # Slot 0 (r0c0): 30 dims - all zeros (inactive)
    # Slot 1 (r0c1): 30 dims - active with telemetry
    # Slot 2 (r0c2): 30 dims - all zeros (inactive)

    # Extract slot 1 features (r0c1) - starts at index 24 + 30 = 54
    slot1_start = 24 + 30  # Skip base + slot0
    slot1_features = obs[0, slot1_start:slot1_start + 30].tolist()

    # Slot features layout (30 dims):
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

    # Check telemetry fields are present (indices 23-26 in slot features)
    assert slot1_features[23] > 0.0  # gradient_norm (normalized, should be > 0)
    assert 0.0 <= slot1_features[24] <= 1.0  # gradient_health
    assert slot1_features[25] == 1.0  # has_vanishing = True
    assert slot1_features[26] == 0.0  # has_exploding = False

    # Check slot 0 and slot 2 are all zeros (inactive)
    slot0_start = 24
    slot0_features = obs[0, slot0_start:slot0_start + 30].tolist()
    assert slot0_features == [0.0] * 30  # r0c0 (disabled)

    slot2_start = 24 + 60  # Skip base + slot0 + slot1
    slot2_features = obs[0, slot2_start:slot2_start + 30].tolist()
    assert slot2_features == [0.0] * 30  # r0c2 (disabled)


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
