# Blueprint Expansion Test Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update tests to match the expanded blueprint action space (5 → 13 blueprints) introduced in commits b57b995 and c2bc1bb.

**Architecture:** Test-only changes. No implementation changes needed - the implementation is correct.

**Tech Stack:** Python 3.11+, pytest, PyTorch, Hypothesis (property-based testing)

**Key Constants:**
- `NUM_BLUEPRINTS = 13` (was 5)
- `SLOT_FEATURE_SIZE = 17` (4 state + 13 blueprint one-hot, was 9)
- `MULTISLOT_FEATURE_SIZE = 74` (23 base + 3×17, was 50)

---

## Task 1: Fix Feature Size Tests

**Files:**
- Modify: `tests/tamiyo/policy/test_features.py`

**Step 1: Update test_multislot_features**

Change line 35-37 and 40-46:
```python
    # Base features (23) + per-slot (3 slots * 17 features) = 74
    # Per-slot: 4 state + 13 blueprint one-hot
    assert len(features) == 74

    # Check per-slot features are included
    # After base features, we have slot features (17 dims each)
    slot_start = 23
    # r0c0 slot: is_active=0, stage=0, alpha=0, improvement=0, blueprint=[0]*13
    assert features[slot_start:slot_start+4] == [0.0, 0.0, 0.0, 0.0]
    assert features[slot_start+4:slot_start+17] == [0.0] * 13  # no blueprint
    # r0c1 slot: is_active=1, stage=3, alpha=0.5, improvement=2.5
    r0c1_start = slot_start + 17
    assert features[r0c1_start:r0c1_start+4] == [1.0, 3.0, 0.5, 2.5]
```

**Step 2: Update test_multislot_features_missing_slots**

Change lines 108-111:
```python
    # Should still produce 74 features, with slot features defaulting to 0
    assert len(features) == 74
    # Last 51 features should be all zeros (3 slots * 17 features)
    assert features[23:] == [0.0] * 51
```

**Step 3: Update test_multislot_feature_size_constant**

Change line 142:
```python
    assert MULTISLOT_FEATURE_SIZE == 74, "Expected 23 base + 51 slot features (3 slots × 17)"
```

**Step 4: Run tests to verify**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_multislot_features tests/tamiyo/policy/test_features.py::test_multislot_features_missing_slots tests/tamiyo/policy/test_features.py::test_multislot_feature_size_constant -v`

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add tests/tamiyo/policy/test_features.py
git commit -m "test(tamiyo): update feature size tests for 13-blueprint action space

Update hardcoded feature sizes:
- 50 → 74 (3 slots)
- 27 → 51 (3 slots × 17 features per slot)

Blueprint action space expanded from 5 to 13 types (commit b57b995).

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Fix Blueprint One-Hot Encoding Test

**Files:**
- Modify: `tests/tamiyo/policy/test_features.py`

**Step 1: Update test_blueprint_one_hot_encoding**

The test needs updated slice indices and one-hot vector sizes. Change lines 220-261:

```python
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

    # Structure: 23 base + 3 slots * 17 features (4 state + 13 blueprint one-hot)
    # r0c0 slot starts at index 23
    # Blueprint one-hot is at indices 27-39 (after 4 state features)
    r0c0_blueprint = features[27:40]  # conv_light = index 1
    expected = [0.0, 1.0] + [0.0] * 11  # 13-element one-hot with index 1 set
    assert r0c0_blueprint == expected, f"conv_light should be {expected}, got {r0c0_blueprint}"

    # Test with attention in r0c1 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 1.0, 'stage': 3, 'alpha': 0.7, 'improvement': 2.0, 'blueprint_id': 'attention'},
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    # r0c1 slot starts at index 23 + 17 = 40
    # Blueprint one-hot is at indices 44-56
    r0c1_blueprint = features[44:57]  # attention = index 2
    expected = [0.0, 0.0, 1.0] + [0.0] * 10  # 13-element one-hot with index 2 set
    assert r0c1_blueprint == expected, f"attention should be {expected}, got {r0c1_blueprint}"

    # Test with noop in r0c2 slot
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c2': {'is_active': 1.0, 'stage': 1, 'alpha': 0.1, 'improvement': 0.5, 'blueprint_id': 'noop'},
    }}
    features = obs_to_multislot_features(obs)

    # r0c2 slot starts at index 23 + 34 = 57
    # Blueprint one-hot is at indices 61-73
    r0c2_blueprint = features[61:74]  # noop = index 0
    expected = [1.0] + [0.0] * 12  # 13-element one-hot with index 0 set
    assert r0c2_blueprint == expected, f"noop should be {expected}, got {r0c2_blueprint}"

    # Test with no blueprint (inactive slot) - should be all zeros
    obs = {**base_obs, 'slots': {
        'r0c0': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
        'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},  # Missing blueprint_id key
        'r0c2': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
    }}
    features = obs_to_multislot_features(obs)

    mid_blueprint = features[44:57]  # r0c1 blueprint slice
    assert mid_blueprint == [0.0] * 13, f"No blueprint should be all zeros, got {mid_blueprint}"
```

**Step 2: Run test to verify**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_blueprint_one_hot_encoding -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/tamiyo/policy/test_features.py
git commit -m "test(tamiyo): update blueprint one-hot test for 13-element vectors

Update slice indices and expected vectors:
- One-hot size: 5 → 13 elements
- Slot feature size: 9 → 17
- Slice offsets recalculated for new dimensions

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Fix Dynamic Feature Size Tests

**Files:**
- Modify: `tests/tamiyo/policy/test_features.py`

**Step 1: Update test_dynamic_feature_size_3_slots**

Change lines 294-297:
```python
    # 23 base + 3 slots * 17 features = 74
    expected_size = get_feature_size(slot_config)
    assert expected_size == 74, f"Expected feature size 74 for 3 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"
```

**Step 2: Update test_dynamic_feature_size_5_slots**

Change lines 332-335:
```python
    # 23 base + 5 slots * 17 features = 108
    expected_size = get_feature_size(slot_config)
    assert expected_size == 108, f"Expected feature size 108 for 5 slots, got {expected_size}"
    assert len(features) == expected_size, f"Expected {expected_size} features, got {len(features)}"
```

**Step 3: Update test_dynamic_slot_iteration**

Change lines 367-381:
```python
    # 23 base + 2 slots * 17 features = 57
    assert len(features) == 57, f"Expected 57 features for 2 slots, got {len(features)}"

    # Verify slot features are present
    # r0c0 slot at index 23-39: is_active=1, stage=2, alpha=0.3, improvement=1.5
    assert features[23] == 1.0, "r0c0 should be active"
    assert features[24] == 2.0, "r0c0 stage should be 2"
    assert features[25] == 0.3, "r0c0 alpha should be 0.3"
    assert features[26] == 1.5, "r0c0 improvement should be 1.5"

    # r0c2 slot at index 40-56: is_active=1, stage=3, alpha=0.7, improvement=2.0
    assert features[40] == 1.0, "r0c2 should be active"
    assert features[41] == 3.0, "r0c2 stage should be 3"
    assert features[42] == 0.7, "r0c2 alpha should be 0.7"
    assert features[43] == 2.0, "r0c2 improvement should be 2.0"
```

**Step 4: Run tests to verify**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_dynamic_feature_size_3_slots tests/tamiyo/policy/test_features.py::test_dynamic_feature_size_5_slots tests/tamiyo/policy/test_features.py::test_dynamic_slot_iteration -v`

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add tests/tamiyo/policy/test_features.py
git commit -m "test(tamiyo): update dynamic feature size tests for expanded blueprints

Update expected sizes:
- 3 slots: 50 → 74
- 5 slots: 68 → 108
- 2 slots: 41 → 57

Update slot feature indices (17 per slot instead of 9).

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Fix LSTM Bundle Tests

**Files:**
- Modify: `tests/tamiyo/policy/test_lstm_bundle.py`

**Step 1: Add imports for action space constants**

Add at line 9 (after existing imports):
```python
from esper.leyline.factored_actions import NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS
```

**Step 2: Update lstm_bundle fixture**

Change line 20 to use correct feature_dim:
```python
@pytest.fixture
def lstm_bundle(slot_config):
    return LSTMPolicyBundle(
        feature_dim=74,  # Updated: 23 base + 3 slots * 17 features
        hidden_dim=64,
        num_lstm_layers=1,
        slot_config=slot_config,
    )
```

**Step 3: Update test_lstm_bundle_get_action**

Change lines 53-59:
```python
def test_lstm_bundle_get_action(lstm_bundle, slot_config):
    """get_action should return ActionResult."""
    features = torch.randn(1, 74)  # Updated feature dim
    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
        "blend": torch.ones(1, NUM_BLENDS, dtype=torch.bool),
        "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
    }
```

**Step 4: Update test_lstm_bundle_evaluate_actions**

Change lines 72-78:
```python
def test_lstm_bundle_evaluate_actions(lstm_bundle, slot_config):
    """evaluate_actions should return EvalResult with gradients."""
    features = torch.randn(1, 10, 74)  # Updated feature dim
    masks = {
        "slot": torch.ones(1, 10, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, 10, NUM_BLUEPRINTS, dtype=torch.bool),
        "blend": torch.ones(1, 10, NUM_BLENDS, dtype=torch.bool),
        "op": torch.ones(1, 10, NUM_OPS, dtype=torch.bool),
    }
```

**Step 5: Update test_get_policy_lstm**

Change lines 126-130:
```python
def test_get_policy_lstm(slot_config):
    """get_policy('lstm', ...) should return LSTMPolicyBundle."""
    policy = get_policy("lstm", {
        "feature_dim": 74,  # Updated feature dim
        "hidden_dim": 64,
        "slot_config": slot_config,
    })
```

**Step 6: Update test_lstm_bundle_forward**

Change lines 136-142:
```python
def test_lstm_bundle_forward(lstm_bundle, slot_config):
    """forward() should return ForwardResult with logits."""
    features = torch.randn(1, 1, 74)  # Updated feature dim
    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
        "blend": torch.ones(1, NUM_BLENDS, dtype=torch.bool),
        "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
    }
```

**Step 7: Update test_lstm_bundle_get_value**

Change line 158:
```python
def test_lstm_bundle_get_value(lstm_bundle):
    """get_value() should return state value estimate."""
    features = torch.randn(1, 74)  # Updated feature dim
```

**Step 8: Update test_get_value_does_not_create_grad_graph**

Change line 172:
```python
def test_get_value_does_not_create_grad_graph(lstm_bundle):
    """get_value() should not create gradient computation graph."""
    features = torch.randn(1, 74).requires_grad_(True)  # Updated feature dim
```

**Step 9: Run tests to verify**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py -v`

Expected: All tests PASS

**Step 10: Commit**

```bash
git add tests/tamiyo/policy/test_lstm_bundle.py
git commit -m "test(tamiyo): update LSTM bundle tests for expanded action space

- Import NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS from leyline
- Update mask dimensions: 5 → NUM_BLUEPRINTS (13)
- Update feature_dim: 50 → 74
- Use constants instead of hardcoded values for future-proofing

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Fix Blueprint Mask Property Test

**Files:**
- Modify: `tests/tamiyo/properties/test_mask_properties.py`

**Step 1: Update test_non_noop_blueprints_enabled**

The test assumes ALL non-NOOP blueprints are enabled, but topology-aware masking now restricts blueprints by task type. Change lines 502-520:

```python
    @given(config=slot_configs())
    def test_non_noop_blueprints_enabled(self, config: SlotConfig):
        """Property: Topology-compatible non-NOOP blueprints are enabled."""
        from esper.leyline.factored_actions import BlueprintAction, CNN_BLUEPRINTS

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        # Default topology is "cnn", so only CNN-compatible blueprints should be enabled
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        for bp in BlueprintAction:
            if bp == BlueprintAction.NOOP:
                # NOOP always disabled
                assert masks["blueprint"][bp].item() is False, (
                    f"NOOP blueprint should always be disabled"
                )
            elif bp in CNN_BLUEPRINTS:
                # CNN-compatible blueprints should be enabled
                assert masks["blueprint"][bp].item() is True, (
                    f"CNN blueprint {bp.name} should be enabled for default topology"
                )
            else:
                # Non-CNN blueprints (LORA, MLP, etc.) should be disabled
                assert masks["blueprint"][bp].item() is False, (
                    f"Non-CNN blueprint {bp.name} should be disabled for default topology"
                )
```

**Step 2: Run test to verify**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/properties/test_mask_properties.py::TestBlueprintMask::test_non_noop_blueprints_enabled -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/tamiyo/properties/test_mask_properties.py
git commit -m "test(tamiyo): update blueprint mask test for topology-aware masking

Account for CNN vs Transformer blueprint restrictions:
- Only CNN_BLUEPRINTS enabled for default topology='cnn'
- LORA, MLP, FLEX_ATTENTION disabled for CNN tasks

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Final Verification

**Step 1: Run all previously failing tests**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py tests/tamiyo/policy/test_lstm_bundle.py tests/tamiyo/properties/test_mask_properties.py::TestBlueprintMask::test_non_noop_blueprints_enabled -v`

Expected: All tests PASS

**Step 2: Run full tamiyo test suite**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/ -v --tb=short -q`

Expected: All tests PASS (or only pre-existing unrelated failures)

---

## Summary

| Task | Tests Fixed | Root Cause |
|------|-------------|------------|
| 1 | Feature size tests | 50 → 74 (hardcoded sizes) |
| 2 | Blueprint one-hot test | 5-dim → 13-dim vectors |
| 3 | Dynamic size tests | Slot feature size 9 → 17 |
| 4 | LSTM bundle tests | Mask shapes 5 → 13 |
| 5 | Mask property test | Topology-aware masking |
| 6 | Final verification | - |

**Estimated time:** 20-25 minutes
