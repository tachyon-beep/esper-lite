"""Tests for multi-slot observation feature extraction.

Note: Tests updated for schema v2 (one-hot stage encoding + scaffolding).
Per-slot layout: [is_active(1), stage_one_hot(10), state(15), blueprint(13)] = 39 dims
State features: alpha, improvement, contribution_velocity, tempo, 7 alpha controller params, 4 scaffolding params
"""

# Slot feature layout constants for test clarity
_STAGE_ONE_HOT_DIMS = 10
_STATE_AFTER_STAGE_DIMS = 15  # alpha, improvement, velocity, tempo, 7 alpha controller, 4 scaffolding
_BLUEPRINT_ONE_HOT_DIMS = 13
_SLOT_FEATURE_SIZE = 39  # 1 + 10 + 15 + 13


def test_multislot_features():
    """obs_to_multislot_features should include per-slot state with one-hot stage."""
    from esper.tamiyo.policy.features import obs_to_multislot_features

    obs = {
        # Base features
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],

        # Per-slot features
        'slots': {
            'r0c0': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c1': {'is_active': True, 'stage': 3, 'alpha': 0.5, 'improvement': 2.5},
            'r0c2': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    features = obs_to_multislot_features(obs)

    # Base features (23) + per-slot (3 slots * 39 features) = 140
    # Per-slot: 1 is_active + 10 stage one-hot + 15 state + 13 blueprint one-hot
    assert len(features) == 140

    # Check r0c0 slot (inactive, stage 0 = UNKNOWN)
    slot_start = 23
    # is_active = 0
    assert features[slot_start] == 0.0
    # stage one-hot: index 0 should be 1.0 (UNKNOWN maps to index 0)
    stage_one_hot = features[slot_start + 1:slot_start + 11]
    assert stage_one_hot[0] == 1.0  # UNKNOWN at index 0
    assert sum(stage_one_hot) == 1.0

    # Check r0c1 slot (active, stage 3 = TRAINING)
    r0c1_start = slot_start + 39
    assert features[r0c1_start] == 1.0  # is_active
    # stage one-hot: TRAINING (value 3) maps to index 3
    stage_one_hot_r0c1 = features[r0c1_start + 1:r0c1_start + 11]
    assert stage_one_hot_r0c1[3] == 1.0  # TRAINING at index 3
    assert sum(stage_one_hot_r0c1) == 1.0
    # alpha (offset 11)
    assert features[r0c1_start + 11] == 0.5
    # improvement (offset 12): 2.5 / 10.0 = 0.25
    assert features[r0c1_start + 12] == 0.25


def test_multislot_alpha_controller_features():
    """Alpha controller state should be present and normalized."""
    from esper.tamiyo.policy.features import obs_to_multislot_features
    from esper.leyline.alpha import AlphaMode, AlphaAlgorithm

    base_obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'max_epochs': 25,
    }

    obs = {
        **base_obs,
        'slots': {
            'r0c0': {
                'is_active': True,
                'stage': 3,
                'alpha': 0.5,
                'improvement': 5.0,  # -> 0.5 normalized
                'blend_tempo_epochs': 8,
                'alpha_target': 1.0,
                'alpha_mode': AlphaMode.UP.value,
                'alpha_steps_total': 10,
                'alpha_steps_done': 4,
                'time_to_target': 6,
                'alpha_velocity': 0.2,
                'alpha_algorithm': AlphaAlgorithm.GATE.value,
                'blueprint_id': 'norm',
            },
            'r0c1': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c2': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    features = obs_to_multislot_features(obs)

    slot_start = 23  # r0c0
    alpha_mode_max = max(mode.value for mode in AlphaMode)
    alpha_algo_min = min(algo.value for algo in AlphaAlgorithm)
    alpha_algo_max = max(algo.value for algo in AlphaAlgorithm)
    alpha_algo_range = max(alpha_algo_max - alpha_algo_min, 1)

    # New offsets: is_active(1) + stage_one_hot(10) = 11, then state features
    # alpha at offset 11, improvement at 12, velocity at 13, tempo at 14
    assert features[slot_start + 11] == 0.5  # alpha
    assert features[slot_start + 12] == 0.5  # improvement normalized (5.0/10)
    assert features[slot_start + 13] == 0.0  # contribution_velocity (not provided, defaults to 0)
    assert abs(features[slot_start + 14] - (8 / 12.0)) < 1e-6  # tempo normalized
    assert features[slot_start + 15] == 1.0  # alpha_target
    assert features[slot_start + 16] == AlphaMode.UP.value / max(alpha_mode_max, 1)  # alpha_mode normalized
    assert features[slot_start + 17] == 10 / 25.0  # alpha_steps_total normalized
    assert features[slot_start + 18] == 4 / 25.0  # alpha_steps_done normalized
    assert features[slot_start + 19] == 6 / 25.0  # time_to_target normalized
    assert features[slot_start + 20] == 0.2  # alpha_velocity (clamped)
    assert features[slot_start + 21] == (AlphaAlgorithm.GATE.value - alpha_algo_min) / alpha_algo_range


def test_multislot_features_normalized_values():
    """Feature values should be normalized to reasonable ranges."""
    from esper.tamiyo.policy.features import obs_to_multislot_features

    obs = {
        'epoch': 50,
        'global_step': 5000,
        'train_loss': 1.5,
        'val_loss': 1.8,
        'loss_delta': -0.2,
        'train_accuracy': 85.0,
        'val_accuracy': 83.0,
        'accuracy_delta': 1.5,
        'plateau_epochs': 5,
        'best_val_accuracy': 85.0,
        'best_val_loss': 1.4,
        'loss_history_5': [2.0, 1.8, 1.6, 1.5, 1.5],
        'accuracy_history_5': [75.0, 78.0, 81.0, 83.0, 83.0],
        'slots': {
            'r0c0': {'is_active': True, 'stage': 2, 'alpha': 0.3, 'improvement': 1.2},
            'r0c1': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c2': {'is_active': True, 'stage': 6, 'alpha': 0.9, 'improvement': -0.5},  # HOLDING = 6
        },
    }

    features = obs_to_multislot_features(obs)

    # All features should be reasonable values
    for i, f in enumerate(features):
        assert isinstance(f, float), f"Feature {i} should be float, got {type(f)}"
        # No NaN or inf
        assert not (f != f), f"Feature {i} is NaN"
        assert abs(f) < 1e6, f"Feature {i} has unreasonable magnitude: {f}"


def test_multislot_feature_size_constant():
    """MULTISLOT_FEATURE_SIZE constant should match actual size."""
    from esper.tamiyo.policy.features import MULTISLOT_FEATURE_SIZE, obs_to_multislot_features

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'slots': {
            'r0c0': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c1': {'is_active': True, 'stage': 3, 'alpha': 0.5, 'improvement': 2.5},
            'r0c2': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    features = obs_to_multislot_features(obs)

    # NOTE: This test uses obs_to_multislot_features() which is Obs V2 format (23 base + 39 per slot).
    # MULTISLOT_FEATURE_SIZE now reflects Obs V3 (24 base + 30 per slot = 114 for 3 slots).
    # For V2: 23 base + 117 slot features (3 slots × 39) = 140
    expected_v2_size = 23 + (3 * 39)  # V2 format
    assert len(features) == expected_v2_size
    assert expected_v2_size == 140, "Expected 23 base + 117 slot features (3 slots × 39)"


def test_seed_utilization_feature():
    """seed_utilization should track resource usage correctly."""
    from esper.tamiyo.policy.features import obs_to_multislot_features

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'total_params': 100_000,
        'slots': {
            'r0c0': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c1': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c2': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    # Test no seeds used (0 / 100 = 0.0)
    features = obs_to_multislot_features(obs, total_seeds=0, max_seeds=100)
    assert features[22] == 0.0, "Should be 0.0 when no seeds used"

    # Test half seeds used (50 / 100 = 0.5)
    features = obs_to_multislot_features(obs, total_seeds=50, max_seeds=100)
    assert features[22] == 0.5, "Should be 0.5 when half seeds used"

    # Test all seeds used (100 / 100 = 1.0)
    features = obs_to_multislot_features(obs, total_seeds=100, max_seeds=100)
    assert features[22] == 1.0, "Should be 1.0 when all seeds used"

    # Test default parameters (0 / 1 = 0.0)
    features = obs_to_multislot_features(obs)
    assert features[22] == 0.0, "Should default to 0.0"

    # Test edge case: max_seeds = 0 (should default to 0.0)
    features = obs_to_multislot_features(obs, total_seeds=10, max_seeds=0)
    assert features[22] == 0.0, "Should be 0.0 when max_seeds is 0"


def test_blueprint_one_hot_encoding():
    """Blueprint one-hot encoding should correctly represent blueprint type per slot."""
    from esper.tamiyo.policy.features import obs_to_multislot_features

    base_obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
    }

    # Test with conv_light in r0c0 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 1.0, 'stage': 2, 'alpha': 0.3, 'improvement': 1.5, 'blueprint_id': 'conv_light'},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    # r0c0 slot starts at index 23
    # Blueprint one-hot is at indices 49-61 (offset 26-38 within slot: after 1+10+15=26 state features)
    r0c0_blueprint = features[49:62]  # conv_light = index 1
    expected = [0.0, 1.0] + [0.0] * 11  # 13-element one-hot with index 1 set
    assert r0c0_blueprint == expected, f"conv_light should be {expected}, got {r0c0_blueprint}"

    # Test with attention in r0c1 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'attention'},
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    # r0c1 slot starts at index 23 + 39 = 62
    # Blueprint one-hot at offset 26 within slot = 62 + 26 = 88
    r0c1_blueprint = features[88:101]  # attention = index 2
    expected = [0.0, 0.0, 1.0] + [0.0] * 10  # 13-element one-hot with index 2 set
    assert r0c1_blueprint == expected, f"attention should be {expected}, got {r0c1_blueprint}"

    # Test with noop in r0c2 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c2': {'is_active': 1.0, 'stage': 1, 'alpha': 0.1, 'improvement': 0.5, 'blueprint_id': 'noop'},
    }}
    features = obs_to_multislot_features(obs)

    # r0c2 slot starts at index 23 + 78 = 101
    # Blueprint one-hot at offset 26 within slot = 101 + 26 = 127
    r0c2_blueprint = features[127:140]  # noop = index 0
    expected = [1.0] + [0.0] * 12  # 13-element one-hot with index 0 set
    assert r0c2_blueprint == expected, f"noop should be {expected}, got {r0c2_blueprint}"

    # Test with no blueprint (inactive slot) - should be all zeros
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},  # Missing blueprint_id key
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    mid_blueprint = features[88:101]  # r0c1 blueprint at offset 88
    assert mid_blueprint == [0.0] * 13, f"No blueprint should be all zeros, got {mid_blueprint}"


def test_dynamic_feature_size_3_slots():
    """Feature extraction with 3 slots should return 140 features."""
    from esper.tamiyo.policy.features import obs_to_multislot_features, get_feature_size
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig.default()  # 3 slots: r0c0, r0c1, r0c2

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'slots': {
            'r0c0': {'is_active': 1.0, 'stage': 2, 'alpha': 0.3, 'improvement': 1.5, 'blueprint_id': 'conv_light'},
            'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
            'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        },
    }

    features = obs_to_multislot_features(obs, slot_config=slot_config)

    # NOTE: obs_to_multislot_features() returns V2 format (23 base + 39 per slot).
    # get_feature_size() now returns V3 format (24 base + 30 per slot).
    # For V2 with 3 slots: 23 base + 3 slots * 39 features = 140
    expected_v2_size = 23 + (slot_config.num_slots * 39)
    assert expected_v2_size == 140, f"Expected feature size 140 for 3 slots (V2), got {expected_v2_size}"
    assert len(features) == expected_v2_size, f"Expected {expected_v2_size} features, got {len(features)}"


def test_dynamic_feature_size_5_slots():
    """Feature extraction with 5 slots should return 218 features."""
    from esper.tamiyo.policy.features import obs_to_multislot_features, get_feature_size
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))  # 5 slots

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'slots': {
            'r0c0': {'is_active': 1.0, 'stage': 2, 'alpha': 0.3, 'improvement': 1.5, 'blueprint_id': 'conv_light'},
            'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
            'r0c2': {'is_active': 1.0, 'stage': 1, 'alpha': 0.1, 'improvement': 0.5, 'blueprint_id': 'attention'},
            'r1c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
            'r1c1': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'norm'},
        },
    }

    features = obs_to_multislot_features(obs, slot_config=slot_config)

    # NOTE: obs_to_multislot_features() returns V2 format (23 base + 39 per slot).
    # get_feature_size() now returns V3 format (24 base + 30 per slot).
    # For V2 with 5 slots: 23 base + 5 slots * 39 features = 218
    expected_v2_size = 23 + (slot_config.num_slots * 39)
    assert expected_v2_size == 218, f"Expected feature size 218 for 5 slots (V2), got {expected_v2_size}"
    assert len(features) == expected_v2_size, f"Expected {expected_v2_size} features, got {len(features)}"


def test_dynamic_slot_iteration():
    """Feature extraction should iterate over slot_config.slot_ids, not hardcoded list."""
    from esper.tamiyo.policy.features import obs_to_multislot_features
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig(slot_ids=("r0c0", "r0c2"))  # Only 2 slots, skipping r0c1

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'slots': {
            'r0c0': {'is_active': 1.0, 'stage': 2, 'alpha': 0.3, 'improvement': 1.5, 'blueprint_id': 'conv_light'},
            'r0c2': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'attention'},
        },
    }

    features = obs_to_multislot_features(obs, slot_config=slot_config)

    # 23 base + 2 slots * 39 features = 101
    assert len(features) == 101, f"Expected 101 features for 2 slots, got {len(features)}"

    # Verify slot features are present with new layout
    # r0c0 slot at index 23: is_active at [23], stage one-hot at [24-33], alpha at [34]
    assert features[23] == 1.0, "r0c0 should be active"
    # Stage 2 (GERMINATED) maps to index 2 in one-hot
    assert features[23 + 1 + 2] == 1.0, "r0c0 stage one-hot[2] should be 1.0 for GERMINATED"
    assert features[23 + 11] == 0.3, "r0c0 alpha should be 0.3"
    assert features[23 + 12] == 0.15, "r0c0 improvement should be normalized (1.5 -> 0.15)"
    assert features[23 + 13] == 0.0, "r0c0 contribution_velocity (not provided, defaults to 0)"

    # r0c2 slot at index 23 + 39 = 62
    assert features[62] == 1.0, "r0c2 should be active"
    # Stage 3 (TRAINING) maps to index 3 in one-hot
    assert features[62 + 1 + 3] == 1.0, "r0c2 stage one-hot[3] should be 1.0 for TRAINING"
    assert features[62 + 11] == 0.7, "r0c2 alpha should be 0.7"
    assert features[62 + 12] == 0.2, "r0c2 improvement should be normalized (2.0 -> 0.2)"
    assert features[62 + 13] == 0.0, "r0c2 contribution_velocity (not provided, defaults to 0)"


def test_stage_one_hot_all_valid_stages():
    """Stage one-hot encoding should work for all valid SeedStage values."""
    from esper.tamiyo.policy.features import obs_to_multislot_features
    from esper.leyline.stages import SeedStage

    base_obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
    }

    # Test each valid stage value
    for stage in SeedStage:
        obs = {**base_obs, 'slots': {
            'r0c0': {'is_active': 1.0, 'stage': stage.value, 'alpha': 0.5, 'improvement': 1.0},
            'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        }}

        features = obs_to_multislot_features(obs)

        # Verify stage one-hot is valid (exactly one 1.0)
        stage_one_hot = features[24:34]  # offset 1-10 within first slot
        assert sum(stage_one_hot) == 1.0, f"Stage {stage.name} one-hot should sum to 1.0"
        assert max(stage_one_hot) == 1.0, f"Stage {stage.name} one-hot should have max 1.0"
        assert min(stage_one_hot) == 0.0, f"Stage {stage.name} one-hot should have min 0.0"


def test_contribution_velocity_raw_not_fossilize_value():
    """Verify contribution_velocity is passed through raw, not as fossilize_value."""
    from esper.tamiyo.policy.features import obs_to_multislot_features

    obs = {
        'epoch': 5,
        'global_step': 50,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'slots': {
            'r0c0': {
                'is_active': True,
                'stage': 3,  # TRAINING
                'alpha': 0.5,
                'improvement': 2.0,  # contribution
                'contribution_velocity': 0.8,  # raw velocity
                'blueprint_id': 'conv_light',
            },
            'r0c1': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'r0c2': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        }
    }

    features = obs_to_multislot_features(obs, total_seeds=1, max_seeds=3)

    # Slot 0 state features start at index 23 + 11 = 34 (after base + is_active + stage one-hot)
    # [0] is_active, [1-10] stage one-hot, [11] alpha, [12] contribution, [13] velocity
    velocity_idx = 23 + 13  # offset 36
    velocity_feature = features[velocity_idx]

    # Velocity should be 0.8 / 10.0 (normalized by _IMPROVEMENT_CLAMP_PCT_PTS)
    expected = 0.8 / 10.0  # 0.08
    assert abs(velocity_feature - expected) < 1e-6, (
        f"Expected raw velocity {expected}, got {velocity_feature}"
    )


def test_slot_features_include_interactions():
    """Verify slot features include interaction and topology fields."""
    from esper.tamiyo.policy.features import SLOT_FEATURE_SIZE

    # Obs V3 layout: 1 is_active + 10 stage + 19 state (no blueprint one-hot) = 30 dims
    # Blueprint one-hot (13 dims) was moved to embedding layer in the network.
    # State includes: alpha, contribution, velocity, tempo, alpha scaffolding (8),
    # telemetry (4), gradient_health_prev, epochs_in_stage_norm, counterfactual_fresh
    assert SLOT_FEATURE_SIZE == 30, f"Expected 30 dims/slot (Obs V3), got {SLOT_FEATURE_SIZE}"


def test_missing_slots_raises_keyerror():
    """Observations without 'slots' key should fail loudly."""
    import pytest
    from esper.tamiyo.policy.features import obs_to_multislot_features

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        # No 'slots' key - should fail
    }

    with pytest.raises(KeyError, match="slots"):
        obs_to_multislot_features(obs)


def test_missing_slot_id_raises_keyerror():
    """Missing slot_id in slots dict should fail loudly."""
    import pytest
    from esper.tamiyo.policy.features import obs_to_multislot_features

    obs = {
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],
        'slots': {},  # Empty - missing r0c0, r0c1, r0c2
    }

    with pytest.raises(KeyError):
        obs_to_multislot_features(obs)


def test_blueprint_index_sync_with_leyline():
    """Verify _BLUEPRINT_TO_INDEX matches BlueprintAction enum (B10-DRL-01).

    This test guards against silent feature corruption if someone adds a new
    blueprint to BlueprintAction without updating the hot-path lookup table.
    The dict duplication is intentional for performance, but must stay in sync.
    """
    from esper.tamiyo.policy.features import _BLUEPRINT_TO_INDEX, _NUM_BLUEPRINT_TYPES
    from esper.leyline import BlueprintAction

    # Verify count matches
    assert len(_BLUEPRINT_TO_INDEX) == len(BlueprintAction), (
        f"Blueprint count drift: features has {len(_BLUEPRINT_TO_INDEX)}, "
        f"leyline has {len(BlueprintAction)}. Update _BLUEPRINT_TO_INDEX!"
    )

    assert _NUM_BLUEPRINT_TYPES == len(BlueprintAction), (
        f"_NUM_BLUEPRINT_TYPES ({_NUM_BLUEPRINT_TYPES}) != len(BlueprintAction) "
        f"({len(BlueprintAction)}). Update both constants!"
    )

    # Verify each mapping matches
    for bp in BlueprintAction:
        bp_id = bp.to_blueprint_id()
        assert bp_id in _BLUEPRINT_TO_INDEX, (
            f"Missing blueprint '{bp_id}' in _BLUEPRINT_TO_INDEX. "
            f"Add: '{bp_id}': {bp.value}"
        )
        assert _BLUEPRINT_TO_INDEX[bp_id] == bp.value, (
            f"Index mismatch for '{bp_id}': features has {_BLUEPRINT_TO_INDEX[bp_id]}, "
            f"leyline has {bp.value}"
        )
