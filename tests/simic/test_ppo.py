"""Tests for PPO module."""
import pytest
import torch

from esper.leyline import DEFAULT_EPISODE_LENGTH, DEFAULT_VALUE_CLIP
from esper.simic.ppo import signals_to_features, PPOAgent
from esper.simic.control import MULTISLOT_FEATURE_SIZE


def test_ppo_agent_architecture():
    """PPOAgent should use FactoredRecurrentActorCritic and TamiyoRolloutBuffer."""
    from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer
    from esper.simic.tamiyo_network import FactoredRecurrentActorCritic

    agent = PPOAgent(
        state_dim=50,
        num_envs=4,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
        compile_network=False,  # Avoid compilation overhead in test
    )

    # Direct type checks
    assert isinstance(agent.buffer, TamiyoRolloutBuffer)
    assert isinstance(agent._base_network, FactoredRecurrentActorCritic)


def test_kl_early_stopping_triggers():
    """Verify approx_kl is computed and can trigger early stopping."""
    agent = PPOAgent(
        state_dim=35,
        num_envs=2,
        max_steps_per_env=5,
        target_kl=0.001,  # Very low to ensure triggering
        recurrent_n_epochs=5,  # Multiple epochs to allow early stop
        compile_network=False,
        device="cpu",
    )

    # Fill buffer with synthetic data
    hidden = agent._base_network.get_initial_hidden(1, torch.device(agent.device))
    for env_id in range(2):
        agent.buffer.start_episode(env_id)
        for step in range(5):
            state = torch.randn(1, 35, device=agent.device)
            masks = {
                "slot": torch.ones(1, 3, dtype=torch.bool, device=agent.device),
                "blueprint": torch.ones(1, 5, dtype=torch.bool, device=agent.device),
                "blend": torch.ones(1, 3, dtype=torch.bool, device=agent.device),
                "op": torch.ones(1, 4, dtype=torch.bool, device=agent.device),  # 4 lifecycle ops
            }
            actions, log_probs, value, hidden = agent._base_network.get_action(
                state, hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                op_mask=masks["op"],
            )
            agent.buffer.add(
                env_id=env_id,
                state=state.squeeze(0),
                slot_action=actions["slot"].item(),
                blueprint_action=actions["blueprint"].item(),
                blend_action=actions["blend"].item(),
                op_action=actions["op"].item(),
                slot_log_prob=log_probs["slot"].item(),
                blueprint_log_prob=log_probs["blueprint"].item(),
                blend_log_prob=log_probs["blend"].item(),
                op_log_prob=log_probs["op"].item(),
                value=value.item(),
                reward=1.0,
                done=step == 4,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                blend_mask=masks["blend"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=hidden[0],  # [num_layers, batch, hidden_dim]
                hidden_c=hidden[1],
                bootstrap_value=0.0,
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
    agent = PPOAgent(
        state_dim=35,
        num_envs=2,
        max_steps_per_env=5,
        target_kl=0.0001,  # Extremely low to ensure triggering
        recurrent_n_epochs=1,  # The critical case from BUG-003
        compile_network=False,
        device="cpu",
    )

    # Fill buffer with FAKE log_probs that differ from network's actual output
    # This simulates policy drift since rollout collection
    for env_id in range(2):
        agent.buffer.start_episode(env_id)
        for step in range(5):
            agent.buffer.add(
                env_id=env_id,
                state=torch.randn(35),
                slot_action=0,
                blueprint_action=0,
                blend_action=0,
                op_action=0,
                # Fake log_probs that are very different from network's actual output
                slot_log_prob=-10.0,  # Network won't produce this
                blueprint_log_prob=-10.0,
                blend_log_prob=-10.0,
                op_log_prob=-10.0,
                value=1.0,
                reward=1.0,
                done=step == 4,
                truncated=False,
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(5, dtype=torch.bool),
                blend_mask=torch.ones(3, dtype=torch.bool),
                op_mask=torch.ones(4, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 128),
                hidden_c=torch.zeros(1, 1, 128),
                bootstrap_value=0.0,
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
    agent = PPOAgent(
        state_dim=35,
        clip_ratio=0.2,  # Policy clip
        clip_value=True,
        value_clip=DEFAULT_VALUE_CLIP,
        compile_network=False,
        device="cpu",
    )

    # Value clip should be much larger than policy clip
    assert agent.value_clip == DEFAULT_VALUE_CLIP, "Agent should have value_clip matching leyline default"
    assert agent.value_clip > agent.clip_ratio, "Value clip should be larger than policy clip"


def test_value_clipping_disabled_option():
    """Verify clip_value=False disables value clipping entirely."""
    agent = PPOAgent(
        state_dim=35,
        clip_value=False,
        compile_network=False,
        device="cpu",
    )
    assert agent.clip_value is False, "clip_value should be configurable to False"


def test_signals_to_features_with_multislot_params():
    """Test signals_to_features accepts total_seeds and max_seeds params."""
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
        active_seeds = []
        available_slots = 3
        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0
        seed_counterfactual = 0.0

    features = signals_to_features(
        signals=MockSignals(),
        slot_reports={},
        use_telemetry=False,
        slots=["r0c1"],
        total_seeds=1,  # NEW param
        max_seeds=3,    # NEW param
    )

    assert len(features) == MULTISLOT_FEATURE_SIZE


def test_signals_to_features_telemetry_slot_alignment() -> None:
    """Telemetry slices must align to [early][mid][late], not "first enabled slot"."""
    from esper.leyline import SeedMetrics, SeedStage, SeedStateReport, SeedTelemetry

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
        active_seeds = []
        available_slots = 3
        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0
        seed_counterfactual = 0.0

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
            stage=SeedStage.TRAINING,
            metrics=SeedMetrics(epochs_total=7),
            telemetry=mid_telemetry,
        )
    }

    features = signals_to_features(
        signals=MockSignals(),
        slot_reports=slot_reports,
        use_telemetry=True,
        slots=["r0c1"],
    )

    base = MULTISLOT_FEATURE_SIZE
    dim = SeedTelemetry.feature_dim()

    assert features[base:base + dim] == [0.0] * dim  # r0c0 (disabled)
    assert features[base + dim:base + 2 * dim] == pytest.approx(mid_telemetry.to_features())  # r0c1
    assert features[base + 2 * dim:base + 3 * dim] == [0.0] * dim  # r0c2 (disabled)


def test_ppo_agent_accepts_slot_config():
    """PPOAgent should accept slot_config and derive state_dim from it."""
    from esper.leyline.slot_config import SlotConfig
    from esper.simic.control import get_feature_size

    slot_config = SlotConfig.default()  # 3 slots
    agent = PPOAgent(
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
        compile_network=False,
    )

    expected_state_dim = get_feature_size(slot_config)  # 23 + 3*9 = 50
    assert agent.slot_config == slot_config
    assert agent._base_network.state_dim == expected_state_dim


def test_ppo_agent_with_3_slot_config():
    """PPOAgent with 3-slot config should have state_dim=50."""
    from esper.leyline.slot_config import SlotConfig
    from esper.simic.control import get_feature_size

    slot_config = SlotConfig.default()  # 3 slots (r0c0, r0c1, r0c2)
    agent = PPOAgent(
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
        compile_network=False,
    )

    expected_state_dim = get_feature_size(slot_config)  # 23 + 3*9 = 50
    assert agent._base_network.state_dim == expected_state_dim
    assert agent._base_network.num_slots == 3


def test_ppo_agent_with_5_slot_config():
    """PPOAgent with 5-slot config should have state_dim=68."""
    from esper.leyline.slot_config import SlotConfig
    from esper.simic.control import get_feature_size

    # Create a 5-slot config
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
    agent = PPOAgent(
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
        compile_network=False,
    )

    expected_state_dim = get_feature_size(slot_config)  # 23 + 5*9 = 68
    assert agent._base_network.state_dim == expected_state_dim
    assert agent._base_network.num_slots == 5


def test_ppo_agent_network_slot_head_matches_config():
    """Network's slot head size should match slot_config.num_slots."""
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0"))  # 4 slots
    agent = PPOAgent(
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
        compile_network=False,
    )

    # Verify network was initialized with correct num_slots
    assert agent._base_network.num_slots == 4


def test_ppo_agent_backwards_compatible_with_state_dim():
    """PPOAgent should still accept explicit state_dim for backwards compatibility."""
    agent = PPOAgent(
        state_dim=50,
        num_envs=2,
        max_steps_per_env=DEFAULT_EPISODE_LENGTH,
        device="cpu",
        compile_network=False,
    )

    assert agent._base_network.state_dim == 50


def test_ppo_agent_buffer_matches_slot_config():
    """PPOAgent buffer should match slot_config passed during initialization."""
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots

    agent = PPOAgent(
        slot_config=slot_config,
        device="cpu",
        num_envs=2,
        max_steps_per_env=10,
        compile_network=False,
    )

    # Buffer should have matching slot_config
    assert agent.buffer.num_slots == 5
    assert agent.buffer.slot_config == slot_config
    # slot_masks should have correct shape
    assert agent.buffer.slot_masks.shape == (2, 10, 5)


def test_ppo_agent_full_update_with_5_slots():
    """Full PPO update cycle with 5-slot config should work correctly."""
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots

    agent = PPOAgent(
        slot_config=slot_config,
        device="cpu",
        num_envs=1,
        max_steps_per_env=5,
        compile_network=False,
        target_kl=None,  # Disable KL early stopping - test uses fake log_probs
    )

    # Add some transitions to buffer
    agent.buffer.start_episode(env_id=0)
    for i in range(5):
        agent.buffer.add(
            env_id=0,
            state=torch.randn(agent._base_network.state_dim),
            slot_action=i % 5,  # Use all 5 slots
            blueprint_action=0,
            blend_action=0,
            op_action=0,
            slot_log_prob=-1.0,
            blueprint_log_prob=-1.0,
            blend_log_prob=-1.0,
            op_log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=(i == 4),
            slot_mask=torch.ones(5, dtype=torch.bool),  # 5 slots
            blueprint_mask=torch.ones(5, dtype=torch.bool),
            blend_mask=torch.ones(3, dtype=torch.bool),
            op_mask=torch.ones(4, dtype=torch.bool),
            hidden_h=torch.zeros(1, 1, 128),
            hidden_c=torch.zeros(1, 1, 128),
        )
    agent.buffer.end_episode(env_id=0)

    # Run PPO update - should not crash
    metrics = agent.update()

    # Should have produced metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
