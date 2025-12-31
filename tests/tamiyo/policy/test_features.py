"""Tests for multi-slot observation feature extraction.

Note: Tests updated for schema v2 (one-hot stage encoding + scaffolding).
Per-slot layout: [is_active(1), stage_one_hot(10), state(15), blueprint(13)] = 39 dims
State features: alpha, improvement, contribution_velocity, tempo, 7 alpha controller params, 4 scaffolding params
"""

import pytest
import torch

# Slot feature layout constants for test clarity
_STAGE_ONE_HOT_DIMS = 10
_STATE_AFTER_STAGE_DIMS = 15  # alpha, improvement, velocity, tempo, 7 alpha controller, 4 scaffolding
_BLUEPRINT_ONE_HOT_DIMS = 13
_SLOT_FEATURE_SIZE = 39  # 1 + 10 + 15 + 13



# =============================================================================


def _make_mock_training_signals(epoch=10, val_loss=0.5, val_accuracy=70.0,
                                 loss_history=None, accuracy_history=None):
    """Create mock TrainingSignals for testing."""
    from esper.leyline.signals import TrainingSignals, TrainingMetrics

    if loss_history is None:
        loss_history = [0.6, 0.55, 0.5, 0.52, 0.5]
    if accuracy_history is None:
        accuracy_history = [65.0, 66.0, 67.0, 68.0, 70.0]

    metrics = TrainingMetrics(
        epoch=epoch,
        global_step=epoch * 100,
        train_loss=val_loss - 0.05,
        val_loss=val_loss,
        loss_delta=-0.02,
        train_accuracy=val_accuracy + 2.0,
        val_accuracy=val_accuracy,
        accuracy_delta=0.5,
        plateau_epochs=2,
        best_val_accuracy=val_accuracy,
        best_val_loss=val_loss - 0.1,
    )

    return TrainingSignals(
        metrics=metrics,
        loss_history=loss_history,
        accuracy_history=accuracy_history,
    )


def _make_mock_seed_state_report(slot_id="r0c0", stage=3, alpha=0.5,
                                  improvement=2.5, blueprint_id="conv_light",
                                  blueprint_index=1, gradient_health=0.95):
    """Create mock SeedStateReport for testing."""
    from esper.leyline.reports import SeedStateReport, SeedMetrics
    from esper.leyline.stages import SeedStage
    from esper.leyline.alpha import AlphaMode, AlphaAlgorithm
    from esper.leyline.telemetry import SeedTelemetry

    metrics = SeedMetrics(
        current_alpha=alpha,
        counterfactual_contribution=improvement,
        contribution_velocity=0.5,
        improvement_since_stage_start=improvement,
        interaction_sum=1.5,
        epochs_in_current_stage=3,
    )

    telemetry = SeedTelemetry(
        seed_id=f"seed_{slot_id}",
        blueprint_id=blueprint_id,
        gradient_norm=2.5,
        gradient_health=gradient_health,
        has_vanishing=False,
        has_exploding=False,
    )

    return SeedStateReport(
        seed_id=f"seed_{slot_id}",
        slot_id=slot_id,
        blueprint_id=blueprint_id,
        blueprint_index=blueprint_index,
        stage=SeedStage(stage),
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


def _make_mock_parallel_env_state(last_action_success=True, last_action_op=0,
                                   gradient_health_prev=None,
                                   epochs_since_counterfactual=None):
    """Create mock ParallelEnvState for testing."""
    from esper.simic.training.parallel_env_state import ParallelEnvState

    # Create minimal mock objects for required fields
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

    env_state = ParallelEnvState(
        model=MockModel(),
        host_optimizer=MockOptimizer(),
        signal_tracker=MockSignalTracker(),
        governor=MockGovernor(),
        last_action_success=last_action_success,
        last_action_op=last_action_op,
    )

    if gradient_health_prev is not None:
        env_state.gradient_health_prev = gradient_health_prev

    if epochs_since_counterfactual is not None:
        env_state.epochs_since_counterfactual = epochs_since_counterfactual

    return env_state


def test_batch_obs_to_features_basic():
    """batch_obs_to_features should return (obs, blueprint_indices) with correct shapes."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.default()  # 3 slots
    device = torch.device("cpu")

    # Create batch of 2 environments
    batch_signals = [
        _make_mock_training_signals(),
        _make_mock_training_signals(epoch=15, val_loss=0.4, val_accuracy=75.0),
    ]

    # First env: 2 active slots, second env: 1 active slot
    batch_slot_reports = [
        {
            "r0c0": _make_mock_seed_state_report("r0c0", stage=3, blueprint_index=1),
            "r0c1": _make_mock_seed_state_report("r0c1", stage=2, blueprint_index=2),
        },
        {
            "r0c0": _make_mock_seed_state_report("r0c0", stage=4, blueprint_index=3),
        },
    ]

    batch_env_states = [
        _make_mock_parallel_env_state(),
        _make_mock_parallel_env_state(),
    ]

    obs, blueprint_indices = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Obs V3: 23 base + 30 per slot × 3 slots = 113 dims
    assert obs.shape == (2, 113), f"Expected (2, 113), got {obs.shape}"
    assert blueprint_indices.shape == (2, 3), f"Expected (2, 3), got {blueprint_indices.shape}"

    # Check blueprint indices
    assert blueprint_indices[0, 0].item() == 1  # r0c0: conv_light
    assert blueprint_indices[0, 1].item() == 2  # r0c1: attention
    assert blueprint_indices[0, 2].item() == -1  # r0c2: inactive

    assert blueprint_indices[1, 0].item() == 3  # r0c0: norm
    assert blueprint_indices[1, 1].item() == -1  # r0c1: inactive
    assert blueprint_indices[1, 2].item() == -1  # r0c2: inactive


def test_batch_obs_to_features_base_features():
    """Base features should be 23 dims with proper normalization."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch
    import math

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Create single environment
    loss_history = [2.0, 1.8, 1.6, 1.5, 1.5]
    accuracy_history = [75.0, 78.0, 81.0, 83.0, 85.0]

    batch_signals = [
        _make_mock_training_signals(
            epoch=50,
            val_loss=1.5,
            val_accuracy=85.0,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
        )
    ]

    batch_slot_reports = [{}]  # No active slots
    batch_env_states = [_make_mock_parallel_env_state(last_action_success=True, last_action_op=2)]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Extract base features (first 23 dims)
    base = obs[0, :23]

    # Check epoch normalization (50 / 150 = 0.333...)
    assert abs(base[0].item() - (50.0 / 150.0)) < 1e-6

    # Check loss normalization: log(1 + 1.5) / log(16) ≈ 0.3296
    expected_loss_norm = math.log(1 + 1.5) / math.log(16)
    assert abs(base[1].item() - expected_loss_norm) < 1e-4

    # Check accuracy normalization (85 / 100 = 0.85)
    assert abs(base[2].item() - 0.85) < 1e-6

    # Check loss history (5 dims, indices 3-7)
    for i, loss in enumerate(loss_history):
        expected = math.log(1 + loss) / math.log(16)
        assert abs(base[3 + i].item() - expected) < 1e-4

    # Check accuracy history (5 dims, indices 8-12)
    for i, acc in enumerate(accuracy_history):
        expected = acc / 100.0
        assert abs(base[8 + i].item() - expected) < 1e-6

    # Check stage distribution (indices 13-15) - all zeros since no active slots
    assert base[13].item() == 0.0  # num_training_norm
    assert base[14].item() == 0.0  # num_blending_norm
    assert base[15].item() == 0.0  # num_holding_norm

    # Check action feedback (indices 16-22)
    assert base[16].item() == 1.0  # last_action_success = True
    # last_action_op one-hot for op=2 (indices 17-22)
    expected_one_hot = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for i, expected in enumerate(expected_one_hot):
        assert abs(base[17 + i].item() - expected) < 1e-6


def test_batch_obs_to_features_slot_features():
    """Slot features should be 30 dims per slot with proper normalization."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    from esper.leyline import DEFAULT_GAMMA
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Create environment with one active slot
    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [
        {"r0c0": _make_mock_seed_state_report("r0c0", stage=3, alpha=0.5, improvement=2.5)}
    ]
    batch_env_states = [
        _make_mock_parallel_env_state(
            gradient_health_prev={"r0c0": 0.85},
            epochs_since_counterfactual={"r0c0": 5},
        )
    ]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Extract first slot features (indices 23-52)
    slot = obs[0, 23:53]

    # is_active (index 0 of slot)
    assert slot[0].item() == 1.0

    # Stage one-hot (indices 1-10) - stage 3 (TRAINING)
    stage_one_hot = slot[1:11]
    assert stage_one_hot[3].item() == 1.0  # TRAINING at index 3
    assert stage_one_hot.sum().item() == 1.0

    # current_alpha (index 11)
    assert abs(slot[11].item() - 0.5) < 1e-6

    # contribution_norm (index 12) - 2.5 / 10.0 = 0.25
    assert abs(slot[12].item() - 0.25) < 1e-6

    # contribution_velocity (index 13) - 0.5 / 10.0 = 0.05
    assert abs(slot[13].item() - 0.05) < 1e-6

    # blend_tempo_norm (index 14) - 8 / 12.0 ≈ 0.6667
    assert abs(slot[14].item() - (8.0 / 12.0)) < 1e-4

    # Check gradient_health_prev (index 27)
    assert abs(slot[27].item() - 0.85) < 1e-6

    # Check counterfactual_fresh (index 29) - DEFAULT_GAMMA ** 5
    expected_fresh = DEFAULT_GAMMA ** 5
    assert abs(slot[29].item() - expected_fresh) < 1e-6


def test_batch_obs_to_features_normalization_ranges():
    """All feature values should be in reasonable ranges."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Create batch with diverse values
    batch_signals = [
        _make_mock_training_signals(epoch=100, val_loss=5.0, val_accuracy=95.0),
        _make_mock_training_signals(epoch=5, val_loss=0.1, val_accuracy=50.0),
    ]

    batch_slot_reports = [
        {"r0c0": _make_mock_seed_state_report("r0c0", stage=3, alpha=0.9, improvement=8.0)},
        {"r0c0": _make_mock_seed_state_report("r0c0", stage=1, alpha=0.1, improvement=-2.0)},
    ]

    batch_env_states = [
        _make_mock_parallel_env_state(),
        _make_mock_parallel_env_state(),
    ]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Check all values are finite
    assert torch.all(torch.isfinite(obs)), "All features should be finite (no NaN or Inf)"

    # Most features should be in [0, 1] range (with some exceptions for normalized deltas)
    # Check that no values are wildly out of range
    assert torch.all(obs > -10.0), "No features should be less than -10"
    assert torch.all(obs < 10.0), "No features should be greater than 10"


def test_batch_obs_to_features_action_feedback():
    """Action feedback should be correctly encoded in base features."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Test different action feedback scenarios
    batch_signals = [
        _make_mock_training_signals(),
        _make_mock_training_signals(),
    ]

    batch_slot_reports = [{}, {}]

    # First env: failed action with op=4 (PRUNE)
    # Second env: successful action with op=1 (GERMINATE)
    batch_env_states = [
        _make_mock_parallel_env_state(last_action_success=False, last_action_op=4),
        _make_mock_parallel_env_state(last_action_success=True, last_action_op=1),
    ]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # First env: action feedback (indices 16-22)
    assert obs[0, 16].item() == 0.0  # last_action_success = False
    # last_action_op one-hot for op=4
    expected_one_hot = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for i, expected in enumerate(expected_one_hot):
        assert abs(obs[0, 17 + i].item() - expected) < 1e-6

    # Second env: action feedback
    assert obs[1, 16].item() == 1.0  # last_action_success = True
    # last_action_op one-hot for op=1
    expected_one_hot = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    for i, expected in enumerate(expected_one_hot):
        assert abs(obs[1, 17 + i].item() - expected) < 1e-6


def test_batch_obs_to_features_gradient_health_tracking():
    """gradient_health_prev should track previous epoch's gradient health."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [
        {"r0c0": _make_mock_seed_state_report("r0c0", gradient_health=0.95)}
    ]

    # Test with tracked previous health
    batch_env_states = [
        _make_mock_parallel_env_state(gradient_health_prev={"r0c0": 0.75})
    ]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # gradient_health_prev is at slot feature index 27 within slot
    # Slot starts at 23, gradient_health_prev is at offset 27 within slot
    gradient_health_prev_idx = 23 + 27  # = 50
    assert abs(obs[0, gradient_health_prev_idx].item() - 0.75) < 1e-6

    # Test with no tracked health (should default to 1.0)
    batch_env_states = [_make_mock_parallel_env_state()]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    assert abs(obs[0, gradient_health_prev_idx].item() - 1.0) < 1e-6


def test_batch_obs_to_features_counterfactual_freshness():
    """counterfactual_fresh should decay with gamma-matched formula."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    from esper.leyline import DEFAULT_GAMMA
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [
        {"r0c0": _make_mock_seed_state_report("r0c0")}
    ]

    # Test different decay levels
    test_epochs = [0, 5, 10, 50, 138]  # 138 epochs should give ~0.5

    for epochs in test_epochs:
        batch_env_states = [
            _make_mock_parallel_env_state(epochs_since_counterfactual={"r0c0": epochs})
        ]

        obs, _ = batch_obs_to_features(
            batch_signals, batch_slot_reports, batch_env_states, slot_config, device
        )

        # counterfactual_fresh is at slot feature index 29
        cf_fresh_idx = 23 + 29  # = 52
        expected = DEFAULT_GAMMA ** epochs
        assert abs(obs[0, cf_fresh_idx].item() - expected) < 1e-4, \
            f"epochs={epochs}: expected {expected}, got {obs[0, cf_fresh_idx].item()}"


def test_batch_obs_to_features_dynamic_slots():
    """Feature extraction should work with different slot configurations."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    # Test with 5 slots
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
    device = torch.device("cpu")

    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [
        {
            "r0c0": _make_mock_seed_state_report("r0c0", blueprint_index=1),
            "r0c2": _make_mock_seed_state_report("r0c2", blueprint_index=2),
            "r1c1": _make_mock_seed_state_report("r1c1", blueprint_index=3),
        }
    ]
    batch_env_states = [_make_mock_parallel_env_state()]

    obs, blueprint_indices = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Obs V3 with 5 slots: 23 base + 30 × 5 = 173 dims
    assert obs.shape == (1, 173), f"Expected (1, 173) for 5 slots, got {obs.shape}"
    assert blueprint_indices.shape == (1, 5)

    # Check blueprint indices
    assert blueprint_indices[0, 0].item() == 1  # r0c0: active
    assert blueprint_indices[0, 1].item() == -1  # r0c1: inactive
    assert blueprint_indices[0, 2].item() == 2  # r0c2: active
    assert blueprint_indices[0, 3].item() == -1  # r1c0: inactive
    assert blueprint_indices[0, 4].item() == 3  # r1c1: active


def test_batch_obs_to_features_stage_distribution():
    """Base features should include stage distribution counts."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    from esper.leyline.stages import SeedStage
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    batch_signals = [_make_mock_training_signals()]

    # Create slots in different stages: 1 TRAINING, 1 BLENDING, 1 HOLDING
    batch_slot_reports = [
        {
            "r0c0": _make_mock_seed_state_report("r0c0", stage=SeedStage.TRAINING.value),
            "r0c1": _make_mock_seed_state_report("r0c1", stage=SeedStage.BLENDING.value),
            "r0c2": _make_mock_seed_state_report("r0c2", stage=SeedStage.HOLDING.value),
        }
    ]
    batch_env_states = [_make_mock_parallel_env_state()]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Stage distribution at indices 13-15 (base features)
    num_training_norm = obs[0, 13].item()
    num_blending_norm = obs[0, 14].item()
    num_holding_norm = obs[0, 15].item()

    # Each normalized by 3.0 (max slots)
    assert abs(num_training_norm - (1.0 / 3.0)) < 1e-6
    assert abs(num_blending_norm - (1.0 / 3.0)) < 1e-6
    assert abs(num_holding_norm - (1.0 / 3.0)) < 1e-6


def test_batch_obs_to_features_batch_processing():
    """Should correctly process multiple environments in batch."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Create batch of 4 environments with varying configurations
    batch_signals = [
        _make_mock_training_signals(epoch=10, val_loss=0.5, val_accuracy=70.0),
        _make_mock_training_signals(epoch=20, val_loss=0.4, val_accuracy=75.0),
        _make_mock_training_signals(epoch=30, val_loss=0.3, val_accuracy=80.0),
        _make_mock_training_signals(epoch=40, val_loss=0.2, val_accuracy=85.0),
    ]

    batch_slot_reports = [
        {},  # No active slots
        {"r0c0": _make_mock_seed_state_report("r0c0", blueprint_index=1)},  # 1 active
        {"r0c0": _make_mock_seed_state_report("r0c0", blueprint_index=1),
         "r0c1": _make_mock_seed_state_report("r0c1", blueprint_index=2)},  # 2 active
        {"r0c0": _make_mock_seed_state_report("r0c0", blueprint_index=1),
         "r0c1": _make_mock_seed_state_report("r0c1", blueprint_index=2),
         "r0c2": _make_mock_seed_state_report("r0c2", blueprint_index=3)},  # 3 active
    ]

    batch_env_states = [_make_mock_parallel_env_state() for _ in range(4)]

    obs, blueprint_indices = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Check shapes
    assert obs.shape == (4, 113)
    assert blueprint_indices.shape == (4, 3)

    # Check each environment has different epoch values
    epochs = [obs[i, 0].item() for i in range(4)]
    assert epochs[0] < epochs[1] < epochs[2] < epochs[3]

    # Check blueprint indices match expectations
    assert torch.all(blueprint_indices[0] == -1)  # All inactive
    assert blueprint_indices[1, 0].item() == 1 and torch.all(blueprint_indices[1, 1:] == -1)
    assert blueprint_indices[2, 0].item() == 1 and blueprint_indices[2, 1].item() == 2
    assert blueprint_indices[3, 0].item() == 1 and blueprint_indices[3, 1].item() == 2 and blueprint_indices[3, 2].item() == 3


def test_batch_obs_to_features_inactive_slots_are_zeros():
    """Inactive slots should have all zero features except is_active."""
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig
    import torch

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [
        {"r0c0": _make_mock_seed_state_report("r0c0", blueprint_index=1)}
        # r0c1 and r0c2 are inactive
    ]
    batch_env_states = [_make_mock_parallel_env_state()]

    obs, _ = batch_obs_to_features(
        batch_signals, batch_slot_reports, batch_env_states, slot_config, device
    )

    # Check r0c1 slot (indices 53-82) is all zeros
    r0c1_features = obs[0, 53:83]
    assert torch.all(r0c1_features == 0.0), "Inactive slot should be all zeros"

    # Check r0c2 slot (indices 83-112) is all zeros
    r0c2_features = obs[0, 83:113]
    assert torch.all(r0c2_features == 0.0), "Inactive slot should be all zeros"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_feature_extraction_performance_cuda_synchronized():
    """Verify V3 feature extraction is <1ms/batch with proper CUDA sync.

    Critical test from PyTorch specialist review: Performance tests on CUDA
    must use proper synchronization to get accurate timings. Without
    torch.cuda.synchronize(), async kernel launches can make code appear
    faster than it actually is.

    This test uses CUDA events for accurate timing and verifies the hot path
    feature extraction meets performance requirements.
    """
    import time
    from esper.tamiyo.policy.features import batch_obs_to_features
    from esper.leyline.slot_config import SlotConfig

    device = torch.device("cuda")
    slot_config = SlotConfig.default()

    # Prepare test data (4-environment batch)
    batch_signals = [_make_mock_training_signals() for _ in range(4)]
    batch_slot_reports = [
        {
            "r0c0": _make_mock_seed_state_report("r0c0", blueprint_index=1),
            "r0c1": _make_mock_seed_state_report("r0c1", blueprint_index=2),
        }
        for _ in range(4)
    ]
    batch_env_states = [_make_mock_parallel_env_state() for _ in range(4)]

    # Warmup - ensure CUDA context is initialized and kernels are compiled
    for _ in range(10):
        batch_obs_to_features(
            batch_signals, batch_slot_reports, batch_env_states, slot_config, device
        )

    # Timed measurement with proper CUDA synchronization
    torch.cuda.synchronize()
    start = time.perf_counter()

    num_iterations = 100
    for _ in range(num_iterations):
        batch_obs_to_features(
            batch_signals, batch_slot_reports, batch_env_states, slot_config, device
        )
        torch.cuda.synchronize()  # Force kernel completion before timing

    elapsed = (time.perf_counter() - start) / num_iterations

    # Target: <1ms per batch (conservative threshold for feature extraction)
    assert elapsed < 0.001, (
        f"Feature extraction too slow: {elapsed * 1000:.3f}ms/batch (target: <1ms). "
        "This may indicate regression in the hot path feature extraction."
    )
