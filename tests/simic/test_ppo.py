"""Tests for PPO module."""
import pytest
import torch

from esper.leyline import DEFAULT_EPISODE_LENGTH, DEFAULT_VALUE_CLIP
from esper.simic.ppo import signals_to_features, PPOAgent
from esper.simic.features import MULTISLOT_FEATURE_SIZE


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
