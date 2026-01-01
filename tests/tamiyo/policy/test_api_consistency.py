"""API consistency tests for (obs, blueprint_indices) tuple flow."""

import torch
from esper.tamiyo.policy.features import batch_obs_to_features, get_feature_size
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
from esper.leyline import (
    SlotConfig,
    TrainingSignals,
    TrainingMetrics,
    SeedStage,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    NUM_ALPHA_TARGETS,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_CURVES,
)
from esper.leyline.reports import SeedStateReport, SeedMetrics
from esper.leyline.alpha import AlphaMode, AlphaAlgorithm
from esper.leyline.telemetry import SeedTelemetry


def _make_test_signals():
    """Create mock TrainingSignals for testing."""
    metrics = TrainingMetrics(
        epoch=10,
        global_step=1000,
        train_loss=0.5,
        val_loss=0.5,
        loss_delta=-0.02,
        train_accuracy=70.0,
        val_accuracy=70.0,
        accuracy_delta=0.5,
        plateau_epochs=2,
        best_val_accuracy=70.0,
        best_val_loss=0.4,
    )

    return TrainingSignals(
        metrics=metrics,
        loss_history=[0.6, 0.55, 0.5, 0.52, 0.5],
        accuracy_history=[65.0, 66.0, 67.0, 68.0, 70.0],
    )


def _make_test_seed_report(slot_id: str, blueprint_index: int):
    """Create mock SeedStateReport for testing."""
    metrics = SeedMetrics(
        current_alpha=0.5,
        counterfactual_contribution=2.5,
        contribution_velocity=0.5,
        improvement_since_stage_start=2.5,
        interaction_sum=1.5,
        epochs_in_current_stage=3,
    )

    telemetry = SeedTelemetry(
        seed_id=f"seed_{slot_id}",
        blueprint_id="conv_light",
        gradient_norm=2.5,
        gradient_health=0.95,
        has_vanishing=False,
        has_exploding=False,
    )

    return SeedStateReport(
        seed_id=f"seed_{slot_id}",
        slot_id=slot_id,
        blueprint_id="conv_light",
        blueprint_index=blueprint_index,
        stage=SeedStage.TRAINING,
        alpha_mode=AlphaMode.UP.value,
        alpha_target=1.0,
        alpha_steps_total=10,
        alpha_steps_done=4,
        time_to_target=6,
        alpha_velocity=0.2,
        alpha_algorithm=AlphaAlgorithm.GATE.value,
        blend_tempo_epochs=8,
        metrics=metrics,
        telemetry=telemetry,
    )


def _make_test_env_state():
    """Create mock ParallelEnvState for testing."""
    from esper.simic.training.parallel_env_state import ParallelEnvState

    class MockModel:
        pass

    class MockOptimizer:
        pass

    class MockSignalTracker:
        def reset(self):
            pass

    class MockGovernor:
        def reset(self):
            pass

    return ParallelEnvState(
        model=MockModel(),
        host_optimizer=MockOptimizer(),
        signal_tracker=MockSignalTracker(),
        governor=MockGovernor(),
        last_action_success=True,
        last_action_op=0,
    )


def test_feature_extraction_returns_tuple():
    """batch_obs_to_features must return (obs, blueprint_indices) tuple."""
    config = SlotConfig.default()

    # Create batch of 4 environments
    batch_signals = [_make_test_signals() for _ in range(4)]
    batch_slot_reports = [
        {"r0c0": _make_test_seed_report("r0c0", 1)},
        {"r0c0": _make_test_seed_report("r0c0", 2)},
        {"r0c0": _make_test_seed_report("r0c0", 3)},
        {"r0c0": _make_test_seed_report("r0c0", 4)},
    ]
    batch_env_states = [_make_test_env_state() for _ in range(4)]
    device = torch.device("cpu")

    result = batch_obs_to_features(
        batch_signals,
        batch_slot_reports,
        batch_env_states,
        config,
        device,
        max_epochs=100,
    )

    # Must be a tuple
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"

    obs, blueprint_indices = result

    # Verify shapes
    assert obs.shape == (4, get_feature_size(config))
    assert blueprint_indices.shape == (4, config.num_slots)
    assert blueprint_indices.dtype == torch.int64


def test_network_forward_accepts_tuple():
    """Network.forward must accept (state, blueprint_indices) as separate args."""
    config = SlotConfig.default()
    network = FactoredRecurrentActorCritic(
        state_dim=get_feature_size(config),
        num_slots=config.num_slots,
    )

    obs = torch.randn(2, 10, get_feature_size(config))  # batch=2, seq=10
    blueprint_indices = torch.randint(0, 13, (2, 10, config.num_slots))

    # Should NOT raise
    output = network.forward(obs, blueprint_indices)

    # Output is a TypedDict
    assert output["value"].shape == (2, 10)
    assert output["sampled_op"].shape == (2, 10)


def test_network_evaluate_actions_accepts_tuple():
    """Network.evaluate_actions must accept blueprint_indices."""
    config = SlotConfig.default()
    network = FactoredRecurrentActorCritic(
        state_dim=get_feature_size(config),
        num_slots=config.num_slots,
    )

    obs = torch.randn(2, 10, get_feature_size(config))
    blueprint_indices = torch.randint(0, NUM_BLUEPRINTS, (2, 10, config.num_slots))
    actions = {
        "op": torch.randint(0, NUM_OPS, (2, 10)),
        "slot": torch.randint(0, config.num_slots, (2, 10)),
        "blueprint": torch.randint(0, NUM_BLUEPRINTS, (2, 10)),
        "style": torch.randint(0, NUM_STYLES, (2, 10)),
        "tempo": torch.randint(0, NUM_TEMPO, (2, 10)),
        "alpha_target": torch.randint(0, NUM_ALPHA_TARGETS, (2, 10)),
        "alpha_speed": torch.randint(0, NUM_ALPHA_SPEEDS, (2, 10)),
        "alpha_curve": torch.randint(0, NUM_ALPHA_CURVES, (2, 10)),
    }

    # Should NOT raise
    log_probs, values, entropy, hidden = network.evaluate_actions(obs, blueprint_indices, actions)

    # Check outputs
    assert values.shape == (2, 10)
    assert "op" in log_probs
    assert log_probs["op"].shape == (2, 10)


def test_buffer_stores_blueprint_indices():
    """TamiyoRolloutBuffer must store blueprint_indices field."""
    config = SlotConfig.default()
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=10,
        state_dim=get_feature_size(config),
        slot_config=config,
        device=torch.device("cpu"),
    )

    # Verify buffer has blueprint_indices tensor
    assert hasattr(buffer, "blueprint_indices")
    assert buffer.blueprint_indices.shape == (4, 10, config.num_slots)
    assert buffer.blueprint_indices.dtype == torch.int64


def test_ppo_update_uses_blueprint_indices_from_buffer():
    """PPO update must extract and use blueprint_indices from buffer batch."""
    config = SlotConfig.default()
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=10,
        state_dim=get_feature_size(config),
        slot_config=config,
        device=torch.device("cpu"),
    )

    # Populate buffer with fake data
    for env_id in range(4):
        for step in range(10):
            buffer.add(
                env_id=env_id,
                state=torch.randn(get_feature_size(config)),
                blueprint_indices=torch.randint(0, NUM_BLUEPRINTS, (config.num_slots,)),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
                alpha_target_action=0,
                alpha_speed_action=0,
                alpha_curve_action=0,
                op_action=0,
                slot_log_prob=0.0,
                blueprint_log_prob=0.0,
                style_log_prob=0.0,
                tempo_log_prob=0.0,
                alpha_target_log_prob=0.0,
                alpha_speed_log_prob=0.0,
                alpha_curve_log_prob=0.0,
                op_log_prob=0.0,
                value=0.0,
                reward=0.0,
                done=False,
                slot_mask=torch.ones(config.num_slots),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS),
                style_mask=torch.ones(NUM_STYLES),
                tempo_mask=torch.ones(NUM_TEMPO),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES),
                op_mask=torch.ones(NUM_OPS),
                hidden_h=torch.zeros(1, 512),  # [num_layers, hidden_dim]
                hidden_c=torch.zeros(1, 512),
            )

    # Get batched sequences (what PPO update receives)
    batch = buffer.get_batched_sequences(torch.device("cpu"))

    # Must include blueprint_indices
    assert "blueprint_indices" in batch, "Buffer batch missing blueprint_indices"
    assert batch["blueprint_indices"].shape == (4, 10, config.num_slots)
    assert batch["blueprint_indices"].dtype == torch.int64
