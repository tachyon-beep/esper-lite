"""Tests for multi-slot observation feature extraction."""



def test_multislot_features():
    """obs_to_multislot_features should include per-slot state."""
    from esper.simic.features import obs_to_multislot_features

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

    # Base features (23) + per-slot (3 slots * 9 features) = 50
    # Per-slot: 4 state + 5 blueprint one-hot
    assert len(features) == 50

    # Check per-slot features are included
    # After base features, we have slot features (9 dims each)
    slot_start = 23
    # r0c0 slot: is_active=0, stage=0, alpha=0, improvement=0, blueprint=[0,0,0,0,0]
    assert features[slot_start:slot_start+4] == [0.0, 0.0, 0.0, 0.0]
    assert features[slot_start+4:slot_start+9] == [0.0, 0.0, 0.0, 0.0, 0.0]  # no blueprint
    # r0c1 slot: is_active=1, stage=3, alpha=0.5, improvement=2.5, blueprint=[0,0,0,0,0] (none specified)
    r0c1_start = slot_start + 9
    assert features[r0c1_start:r0c1_start+4] == [1.0, 3.0, 0.5, 2.5]


def test_multislot_features_normalized_values():
    """Feature values should be normalized to reasonable ranges."""
    from esper.simic.features import obs_to_multislot_features

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
            'late': {'is_active': True, 'stage': 5, 'alpha': 0.9, 'improvement': -0.5},
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
    from esper.simic.features import obs_to_multislot_features

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

    # Should still produce 50 features, with slot features defaulting to 0
    assert len(features) == 50
    # Last 27 features should be all zeros (3 slots * 9 features)
    assert features[23:] == [0.0] * 27


def test_multislot_feature_size_constant():
    """MULTISLOT_FEATURE_SIZE constant should match actual size."""
    from esper.simic.features import MULTISLOT_FEATURE_SIZE, obs_to_multislot_features

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
    assert MULTISLOT_FEATURE_SIZE == 50, "Expected 23 base + 27 slot features (3 slots × 9)"


def test_seed_utilization_feature():
    """seed_utilization should track resource usage correctly."""
    from esper.simic.features import obs_to_multislot_features

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
    from esper.simic.features import obs_to_multislot_features

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

    # Structure: 23 base + 3 slots * 9 features (4 state + 5 blueprint one-hot)
    # r0c0 slot starts at index 23
    # Blueprint one-hot is at indices 27-31 (after 4 state features)
    r0c0_blueprint = features[27:32]  # conv_light = index 1
    assert r0c0_blueprint == [0.0, 1.0, 0.0, 0.0, 0.0], f"conv_light should be [0,1,0,0,0], got {r0c0_blueprint}"

    # Test with attention in r0c1 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'attention'},
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    # r0c1 slot starts at index 23 + 9 = 32
    # Blueprint one-hot is at indices 36-40
    r0c1_blueprint = features[36:41]  # attention = index 2
    assert r0c1_blueprint == [0.0, 0.0, 1.0, 0.0, 0.0], f"attention should be [0,0,1,0,0], got {r0c1_blueprint}"

    # Test with noop in r0c2 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c2': {'is_active': 1.0, 'stage': 1, 'alpha': 0.1, 'improvement': 0.5, 'blueprint_id': 'noop'},
    }}
    features = obs_to_multislot_features(obs)

    # r0c2 slot starts at index 23 + 18 = 41
    # Blueprint one-hot is at indices 45-49
    r0c2_blueprint = features[45:50]  # noop = index 0
    assert r0c2_blueprint == [1.0, 0.0, 0.0, 0.0, 0.0], f"noop should be [1,0,0,0,0], got {r0c2_blueprint}"

    # Test with no blueprint (inactive slot) - should be all zeros
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},  # Missing blueprint_id key
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    mid_blueprint = features[36:41]
    assert mid_blueprint == [0.0, 0.0, 0.0, 0.0, 0.0], f"No blueprint should be all zeros, got {mid_blueprint}"


def test_dynamic_feature_size_3_slots():
    """Feature extraction with 3 slots should return 50 features."""
    from esper.simic.features import obs_to_multislot_features, get_feature_size
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

    # 23 base + 3 slots * 9 features = 50
    expected_size = get_feature_size(slot_config)
    assert expected_size == 50, f"Expected feature size 50 for 3 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"


def test_dynamic_feature_size_5_slots():
    """Feature extraction with 5 slots should return 68 features."""
    from esper.simic.features import obs_to_multislot_features, get_feature_size
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

    # 23 base + 5 slots * 9 features = 68
    expected_size = get_feature_size(slot_config)
    assert expected_size == 68, f"Expected feature size 68 for 5 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"


def test_dynamic_slot_iteration():
    """Feature extraction should iterate over slot_config.slot_ids, not hardcoded list."""
    from esper.simic.features import obs_to_multislot_features
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

    # 23 base + 2 slots * 9 features = 41
    assert len(features) == 41, f"Expected 41 features for 2 slots, got {len(features)}"

    # Verify slot features are present
    # r0c0 slot at index 23-31: is_active=1, stage=2, alpha=0.3, improvement=1.5
    assert features[23] == 1.0, "r0c0 should be active"
    assert features[24] == 2.0, "r0c0 stage should be 2"
    assert features[25] == 0.3, "r0c0 alpha should be 0.3"
    assert features[26] == 1.5, "r0c0 improvement should be 1.5"

    # r0c2 slot at index 32-40: is_active=1, stage=3, alpha=0.7, improvement=2.0
    assert features[32] == 1.0, "r0c2 should be active"
    assert features[33] == 3.0, "r0c2 stage should be 3"
    assert features[34] == 0.7, "r0c2 alpha should be 0.7"
    assert features[35] == 2.0, "r0c2 improvement should be 2.0"
