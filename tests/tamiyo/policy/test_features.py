"""Tests for multi-slot observation feature extraction."""



def test_multislot_features():
    """obs_to_multislot_features should include per-slot state."""
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

    # Base features (23) + per-slot (3 slots * 18 features) = 77
    # Per-slot: 5 state (is_active, stage, alpha, improvement, tempo) + 13 blueprint one-hot
    assert len(features) == 77

    # Check per-slot features are included
    # After base features, we have slot features (18 dims each)
    slot_start = 23
    # r0c0 slot: is_active=0, stage=0, alpha=0, improvement=0, tempo=5/12 (default), blueprint=[0]*13
    assert features[slot_start:slot_start+4] == [0.0, 0.0, 0.0, 0.0]
    assert abs(features[slot_start+4] - 5/12.0) < 1e-6  # tempo default is 5 epochs / 12
    assert features[slot_start+5:slot_start+18] == [0.0] * 13  # no blueprint
    # r0c1 slot: is_active=1, stage=3, alpha=0.5, improvement=2.5, tempo=5/12
    r0c1_start = slot_start + 18
    assert features[r0c1_start:r0c1_start+4] == [1.0, 3.0, 0.5, 2.5]


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

    # Should still produce 77 features, with slot features defaulting to 0 or defaults
    assert len(features) == 77
    # Last 54 features should be mostly zeros (3 slots * 18 features), except tempo defaults
    # Each slot has [0, 0, 0, 0, 5/12, 0*13] = 18 dims
    tempo_default = 5/12.0
    for i in range(3):  # 3 slots
        slot_offset = 23 + i * 18
        # First 4 should be 0
        assert features[slot_offset:slot_offset+4] == [0.0, 0.0, 0.0, 0.0]
        # 5th is tempo default
        assert abs(features[slot_offset+4] - tempo_default) < 1e-6
        # Rest are blueprint zeros
        assert features[slot_offset+5:slot_offset+18] == [0.0] * 13


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
    assert MULTISLOT_FEATURE_SIZE == 77, "Expected 23 base + 54 slot features (3 slots × 18)"


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

    # Structure: 23 base + 3 slots * 18 features (5 state + 13 blueprint one-hot)
    # r0c0 slot starts at index 23
    # Blueprint one-hot is at indices 28-40 (after 5 state features: is_active, stage, alpha, improvement, tempo)
    r0c0_blueprint = features[28:41]  # conv_light = index 1
    expected = [0.0, 1.0] + [0.0] * 11  # 13-element one-hot with index 1 set
    assert r0c0_blueprint == expected, f"conv_light should be {expected}, got {r0c0_blueprint}"

    # Test with attention in r0c1 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'attention'},
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    # r0c1 slot starts at index 23 + 18 = 41
    # Blueprint one-hot is at indices 46-58 (after 5 state features)
    r0c1_blueprint = features[46:59]  # attention = index 2
    expected = [0.0, 0.0, 1.0] + [0.0] * 10  # 13-element one-hot with index 2 set
    assert r0c1_blueprint == expected, f"attention should be {expected}, got {r0c1_blueprint}"

    # Test with noop in r0c2 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c2': {'is_active': 1.0, 'stage': 1, 'alpha': 0.1, 'improvement': 0.5, 'blueprint_id': 'noop'},
    }}
    features = obs_to_multislot_features(obs)

    # r0c2 slot starts at index 23 + 36 = 59
    # Blueprint one-hot is at indices 64-76 (after 5 state features)
    r0c2_blueprint = features[64:77]  # noop = index 0
    expected = [1.0] + [0.0] * 12  # 13-element one-hot with index 0 set
    assert r0c2_blueprint == expected, f"noop should be {expected}, got {r0c2_blueprint}"

    # Test with no blueprint (inactive slot) - should be all zeros
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},  # Missing blueprint_id key
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    mid_blueprint = features[46:59]  # r0c1 blueprint slice (41 + 5 state = 46)
    assert mid_blueprint == [0.0] * 13, f"No blueprint should be all zeros, got {mid_blueprint}"


def test_dynamic_feature_size_3_slots():
    """Feature extraction with 3 slots should return 74 features."""
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

    # 23 base + 3 slots * 18 features = 77
    expected_size = get_feature_size(slot_config)
    assert expected_size == 77, f"Expected feature size 77 for 3 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"


def test_dynamic_feature_size_5_slots():
    """Feature extraction with 5 slots should return 108 features."""
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

    # 23 base + 5 slots * 18 features = 113
    expected_size = get_feature_size(slot_config)
    assert expected_size == 113, f"Expected feature size 113 for 5 slots, got {expected_size}"
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

    # 23 base + 2 slots * 18 features = 59
    assert len(features) == 59, f"Expected 59 features for 2 slots, got {len(features)}"

    # Verify slot features are present
    # r0c0 slot at index 23-40: is_active=1, stage=2, alpha=0.3, improvement=1.5, tempo=5/12
    assert features[23] == 1.0, "r0c0 should be active"
    assert features[24] == 2.0, "r0c0 stage should be 2"
    assert features[25] == 0.3, "r0c0 alpha should be 0.3"
    assert features[26] == 1.5, "r0c0 improvement should be 1.5"

    # r0c2 slot at index 41-58: is_active=1, stage=3, alpha=0.7, improvement=2.0, tempo=5/12
    assert features[41] == 1.0, "r0c2 should be active"
    assert features[42] == 3.0, "r0c2 stage should be 3"
    assert features[43] == 0.7, "r0c2 alpha should be 0.7"
    assert features[44] == 2.0, "r0c2 improvement should be 2.0"
