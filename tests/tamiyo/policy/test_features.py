"""Tests for multi-slot observation feature extraction.

Note: Tests updated for schema v1 (one-hot stage encoding).
Per-slot layout: [is_active(1), stage_one_hot(10), state(11), blueprint(13)] = 35 dims
"""

# Slot feature layout constants for test clarity
_STAGE_ONE_HOT_DIMS = 10
_STATE_AFTER_STAGE_DIMS = 11  # alpha, improvement, fossilize, tempo, 7 alpha controller
_BLUEPRINT_ONE_HOT_DIMS = 13
_SLOT_FEATURE_SIZE = 35  # 1 + 10 + 11 + 13


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

    # Base features (23) + per-slot (3 slots * 35 features) = 128
    # Per-slot: 1 is_active + 10 stage one-hot + 11 state + 13 blueprint one-hot
    assert len(features) == 128

    # Check r0c0 slot (inactive, stage 0 = UNKNOWN)
    slot_start = 23
    # is_active = 0
    assert features[slot_start] == 0.0
    # stage one-hot: index 0 should be 1.0 (UNKNOWN maps to index 0)
    stage_one_hot = features[slot_start + 1:slot_start + 11]
    assert stage_one_hot[0] == 1.0  # UNKNOWN at index 0
    assert sum(stage_one_hot) == 1.0

    # Check r0c1 slot (active, stage 3 = TRAINING)
    r0c1_start = slot_start + 35
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
    # alpha at offset 11, improvement at 12, fossilize at 13, tempo at 14
    assert features[slot_start + 11] == 0.5  # alpha
    assert features[slot_start + 12] == 0.5  # improvement normalized (5.0/10)
    assert features[slot_start + 13] == 0.5  # fossilize_value (same as improvement, no velocity)
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
            'early': {'is_active': True, 'stage': 2, 'alpha': 0.3, 'improvement': 1.2},
            'mid': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'late': {'is_active': True, 'stage': 6, 'alpha': 0.9, 'improvement': -0.5},  # HOLDING = 6
        },
    }

    features = obs_to_multislot_features(obs)

    # All features should be reasonable values
    for i, f in enumerate(features):
        assert isinstance(f, float), f"Feature {i} should be float, got {type(f)}"
        # No NaN or inf
        assert not (f != f), f"Feature {i} is NaN"
        assert abs(f) < 1e6, f"Feature {i} has unreasonable magnitude: {f}"


def test_multislot_features_missing_slots():
    """Should handle missing slot data gracefully."""
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
        # No 'slots' key
    }

    features = obs_to_multislot_features(obs)

    # Should still produce 128 features, with slot features defaulting to 0 or defaults
    assert len(features) == 128
    # Each slot has 35 features
    tempo_default = 5 / 12.0
    for i in range(3):  # 3 slots
        slot_offset = 23 + i * 35
        # is_active should be 0
        assert features[slot_offset] == 0.0
        # stage one-hot: stage 0 (UNKNOWN) -> index 0 set to 1.0
        stage_one_hot = features[slot_offset + 1:slot_offset + 11]
        assert stage_one_hot[0] == 1.0  # UNKNOWN at index 0
        assert sum(stage_one_hot) == 1.0
        # tempo at offset 14
        assert abs(features[slot_offset + 14] - tempo_default) < 1e-6
        # Blueprint zeros (last 13 dims)
        assert features[slot_offset + 22:slot_offset + 35] == [0.0] * 13


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
            'early': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'mid': {'is_active': True, 'stage': 3, 'alpha': 0.5, 'improvement': 2.5},
            'late': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    features = obs_to_multislot_features(obs)

    assert len(features) == MULTISLOT_FEATURE_SIZE
    assert MULTISLOT_FEATURE_SIZE == 128, "Expected 23 base + 105 slot features (3 slots × 35)"


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
            'early': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'mid': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'late': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
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
    # Blueprint one-hot is at indices 45-57 (offset 22-34 within slot: after 1+10+11=22 state features)
    r0c0_blueprint = features[45:58]  # conv_light = index 1
    expected = [0.0, 1.0] + [0.0] * 11  # 13-element one-hot with index 1 set
    assert r0c0_blueprint == expected, f"conv_light should be {expected}, got {r0c0_blueprint}"

    # Test with attention in r0c1 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'attention'},
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    # r0c1 slot starts at index 23 + 35 = 58
    # Blueprint one-hot at offset 22 within slot = 58 + 22 = 80
    r0c1_blueprint = features[80:93]  # attention = index 2
    expected = [0.0, 0.0, 1.0] + [0.0] * 10  # 13-element one-hot with index 2 set
    assert r0c1_blueprint == expected, f"attention should be {expected}, got {r0c1_blueprint}"

    # Test with noop in r0c2 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c2': {'is_active': 1.0, 'stage': 1, 'alpha': 0.1, 'improvement': 0.5, 'blueprint_id': 'noop'},
    }}
    features = obs_to_multislot_features(obs)

    # r0c2 slot starts at index 23 + 70 = 93
    # Blueprint one-hot at offset 22 within slot = 93 + 22 = 115
    r0c2_blueprint = features[115:128]  # noop = index 0
    expected = [1.0] + [0.0] * 12  # 13-element one-hot with index 0 set
    assert r0c2_blueprint == expected, f"noop should be {expected}, got {r0c2_blueprint}"

    # Test with no blueprint (inactive slot) - should be all zeros
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},  # Missing blueprint_id key
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    mid_blueprint = features[80:93]  # r0c1 blueprint at offset 80
    assert mid_blueprint == [0.0] * 13, f"No blueprint should be all zeros, got {mid_blueprint}"


def test_dynamic_feature_size_3_slots():
    """Feature extraction with 3 slots should return 128 features."""
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

    # 23 base + 3 slots * 35 features = 128
    expected_size = get_feature_size(slot_config)
    assert expected_size == 128, f"Expected feature size 128 for 3 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"


def test_dynamic_feature_size_5_slots():
    """Feature extraction with 5 slots should return 198 features."""
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

    # 23 base + 5 slots * 35 features = 198
    expected_size = get_feature_size(slot_config)
    assert expected_size == 198, f"Expected feature size 198 for 5 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"


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

    # 23 base + 2 slots * 35 features = 93
    assert len(features) == 93, f"Expected 93 features for 2 slots, got {len(features)}"

    # Verify slot features are present with new layout
    # r0c0 slot at index 23: is_active at [23], stage one-hot at [24-33], alpha at [34]
    assert features[23] == 1.0, "r0c0 should be active"
    # Stage 2 (GERMINATED) maps to index 2 in one-hot
    assert features[23 + 1 + 2] == 1.0, "r0c0 stage one-hot[2] should be 1.0 for GERMINATED"
    assert features[23 + 11] == 0.3, "r0c0 alpha should be 0.3"
    assert features[23 + 12] == 0.15, "r0c0 improvement should be normalized (1.5 -> 0.15)"
    assert features[23 + 13] == 0.15, "r0c0 fossilize_value should equal improvement (no velocity)"

    # r0c2 slot at index 23 + 35 = 58
    assert features[58] == 1.0, "r0c2 should be active"
    # Stage 3 (TRAINING) maps to index 3 in one-hot
    assert features[58 + 1 + 3] == 1.0, "r0c2 stage one-hot[3] should be 1.0 for TRAINING"
    assert features[58 + 11] == 0.7, "r0c2 alpha should be 0.7"
    assert features[58 + 12] == 0.2, "r0c2 improvement should be normalized (2.0 -> 0.2)"
    assert features[58 + 13] == 0.2, "r0c2 fossilize_value should equal improvement (no velocity)"


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
