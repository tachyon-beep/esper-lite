# Task 4: DRL Review - Feature Extraction Optimization (commit deef9b6)

## Executive Summary

**Status: FLAGGED - 2 Critical Issues**

The feature extraction optimization introduces tensor-returning functions for performance. The new `obs_to_base_features_tensor()` implementation is well-designed, but there are **normalization bugs inherited from the existing list-based implementation** that affect state representation quality.

---

## Question 1: Feature Count (35 features for PPO state)

**Status: PASS ✓**

All 35 base features are correctly extracted and verified:

- **Timing** (2): epoch, global_step
- **Loss** (3): train_loss, val_loss, loss_delta
- **Accuracy** (3): train_accuracy, val_accuracy, accuracy_delta
- **Trends** (3): plateau_epochs, best_val_accuracy, best_val_loss
- **History** (10): loss_history_5 (5), accuracy_history_5 (5)
- **Seed state** (6): has_active_seed, seed_stage, seed_epochs_in_stage, seed_alpha, seed_improvement, available_slots
- **Counterfactual** (1): seed_counterfactual
- **Host state** (2): host_grad_norm, host_learning_phase
- **Blueprint** (5): one-hot encoding for categorical blueprint selection

**Test evidence:** All 11 tests pass, including:
- `TestBlueprintOneHotEncoding::test_base_features_includes_blueprint_one_hot`
- `TestTensorFeatureExtraction` (all 3 tests)
- `test_tensor_features_match_list_features` (element-wise matching verified)

---

## Question 2: Normalization Ranges

**Status: FLAG - 9 out of 35 features have normalization issues**

### Issues Identified

#### Issue 1: Loss Features Missing Clipping (Critical for NNs)

**Affected indices:** [2], [3], [10] + [11-15] (9 features total)

Loss features are normalized by dividing by 10.0 **without clamping**:

```python
# INCORRECT - existing bug in obs_to_base_features()
safe(obs['train_loss'], 10.0) / 10.0    # Line 112
safe(obs['val_loss'], 10.0) / 10.0      # Line 113
safe(obs['best_val_loss'], 10.0) / 10.0 # Line 122
safe(v, 10.0) for v in obs['loss_history_5']  # Line 124
```

The `safe()` function is called with only 2 arguments, so `max_val` defaults to 100.0:

```python
def safe(v, default: float = 0.0, max_val: float = 100.0) -> float:  # DEFAULT: 100.0!
```

**Impact:** When loss > 10.0, the normalized value exceeds 1.0:
- Loss = 50.0 → safe(50.0, 10.0) = 50.0 → 50.0/10.0 = **5.0** (should be ≤ 1.0)
- Loss = 100.0 → normalized to **10.0** (should be ≤ 1.0)

This violates the PPO state normalization contract and can cause:
- Exploding activation values in the policy network
- Poor gradient flow through early layers
- RunningMeanStd statistics dominated by outlier loss values
- Reduced learning efficiency during training instability

**Correct implementation (seen in tensor version history handling):**
```python
# CORRECT - tensor version uses vectorized clipping
loss_hist = torch.clamp(loss_hist, max=10.0) / 10.0  # Line 178
```

**Fix required:**
```python
safe(obs['train_loss'], 10.0, max_val=10.0) / 10.0    # Add max_val=10.0
safe(obs['val_loss'], 10.0, max_val=10.0) / 10.0      # Add max_val=10.0
safe(obs['best_val_loss'], 10.0, max_val=10.0) / 10.0 # Add max_val=10.0
safe(v, 10.0, max_val=10.0) for v in obs['loss_history_5']  # Add max_val=10.0
```

#### Issue 2: available_slots No Normalization (Inconsistent Scale)

**Affected index:** [26]

```python
float(obs['available_slots'])  # Line 132 - NO NORMALIZATION
```

While typically 0-2, this feature has **no clipping or scaling** and could theoretically be arbitrarily large. This is inconsistent with the normalized [0, 1] philosophy for other features.

**Recommendation:**
```python
float(obs['available_slots']) / 4.0  # Cap at [0, 0.25] for 0-4 slots, or [0, 1] for 0-4
```

### Normalization Matrix

| Feature Group | Indices | Status | Issue |
|---------------|---------|--------|-------|
| Timing | [0-1] | ✓ | Correct normalization |
| Loss | [2-3, 10] | ✗ | Missing max_val clipping |
| Accuracy | [5-7, 9] | ✓ | Correct [0, 1] range |
| Trends | [8] | ✓ | Correct [0, 1] range |
| History Loss | [11-15] | ✗ | Missing max_val clipping |
| History Accuracy | [16-20] | ✓ | Correct [0, 1] range |
| Seed State | [21-25] | ✓ | Correct ranges |
| Counterfactual | [27] | ✓ | Correct [0, 1] range |
| Host State | [28-29] | ✓ | Correct ranges |
| Blueprint | [30-34] | ✓ | Correct one-hot |
| **Available Slots** | **[26]** | **✗** | **No normalization** |

---

## Question 3: Blueprint One-Hot Encoding

**Status: PASS ✓**

The blueprint categorical feature uses correct one-hot encoding:

### Implementation Details (Both Versions)

```python
# List version (lines 101-105)
blueprint_id = obs.get('seed_blueprint_id', 0)
num_blueprints = obs.get('num_blueprints', 5)
blueprint_one_hot = [0.0] * num_blueprints
if blueprint_id > 0 and blueprint_id <= num_blueprints:
    blueprint_one_hot[blueprint_id - 1] = 1.0  # 1-indexed to 0-indexed
```

### Correctness Verification

| blueprint_id | num_blueprints | Result | Status |
|-------------|-----------------|--------|--------|
| 0 | 5 | [0,0,0,0,0] | ✓ No seed |
| 1 | 5 | [1,0,0,0,0] | ✓ feature[30] = 1.0 |
| 2 | 5 | [0,1,0,0,0] | ✓ feature[31] = 1.0 |
| 5 | 5 | [0,0,0,0,1] | ✓ feature[34] = 1.0 |
| 6 | 5 | [0,0,0,0,0] | ✓ Out of bounds → safe zeros |
| -1 | 5 | [0,0,0,0,0] | ✓ Invalid → safe zeros |

### Why One-Hot is Correct

One-hot encoding for the blueprint ID is the right choice for PPO because:

1. **No ordinal assumption**: Blueprint IDs are categorical identifiers, not ordered/ranked values. One-hot avoids implying that blueprint 5 is "better" than blueprint 1.

2. **DRL signal clarity**: Policy can learn distinct strategies per blueprint without interference from similarity metrics.

3. **Gradient flow**: Each blueprint gets its own feature, preventing crosstalk in gradient computation.

4. **Safe boundary handling**: Out-of-bounds or missing blueprint_id defaults to all-zeros (neutral embedding).

**Test coverage:** `TestBlueprintOneHotEncoding` validates all cases ✓

---

## Tensor vs List Implementation Comparison

### Tensor Version: obs_to_base_features_tensor()

**Strengths:**
- ✓ Vectorized loss history handling: `torch.clamp(loss_hist, max=10.0) / 10.0`
- ✓ Pre-allocated output tensor option (zero-alloc mode)
- ✓ No Python list allocation overhead
- ✓ Direct PyTorch tensor output for immediate use in neural networks

**Weakness:**
- ✗ Still uses `safe()` for scalar loss features without `max_val` parameter (line 184-185, 192)

### List Version: obs_to_base_features()

**Current state:**
- ✗ Uses `safe()` without `max_val` for all loss features
- ✗ Slower (Python list intermediate)
- ✓ Good for reference implementations and testing

### Critical Observation

The tensor version **correctly handles loss history** with vectorized clipping but **incorrectly handles individual losses** with `safe()`. This inconsistency suggests the code was partially optimized without comprehensive normalization review.

---

## DRL Impact Assessment

### Training Stability Risk: HIGH

Unnormalized loss features directly feed into:
1. Policy network input layer → poor activation distribution
2. Value network input layer → unstable baseline estimation
3. Advantage computation → inaccurate A_t = r_t + γV(s_{t+1}) - V(s_t)

In PPO with advantage normalization, oversized loss features will:
- Dominate the mean/variance statistics
- Cause inconsistent advantage scaling across batches
- Potentially trigger PPO's clip ratio warnings

### Blueprint Encoding Quality: GOOD

The one-hot encoding enables:
- ✓ Separate policy branches per blueprint type
- ✓ Emergent multi-task learning within single network
- ✓ Clear feature attribution for policy decisions

---

## Recommendations

### Fix 1 (Critical): Add max_val to Loss safe() Calls

```python
# obs_to_base_features() function
Line 112: safe(obs['train_loss'], 10.0, max_val=10.0) / 10.0
Line 113: safe(obs['val_loss'], 10.0, max_val=10.0) / 10.0
Line 122: safe(obs['best_val_loss'], 10.0, max_val=10.0) / 10.0
Line 124: safe(v, 10.0, max_val=10.0) for v in obs['loss_history_5']

# obs_to_base_features_tensor() function
Line 184: safe(obs['train_loss'], 10.0, max_val=10.0) / 10.0
Line 185: safe(obs['val_loss'], 10.0, max_val=10.0) / 10.0
Line 192: safe(obs['best_val_loss'], 10.0, max_val=10.0) / 10.0
```

**Impact:** Ensures loss features ∈ [0, 1], maintaining RunningMeanStd stability.

### Fix 2 (Recommended): Normalize available_slots

```python
# Current (line 132):
float(obs['available_slots']),  # No normalization!

# Recommended:
float(obs['available_slots']) / 4.0,  # Assumes max 4 slots, yields [0, 1]
```

**Impact:** Consistent feature scaling philosophy.

### Verification Steps

After fixes:
```bash
# Run tests to ensure backward compatibility
pytest tests/test_simic_features.py -v

# Verify normalization ranges on real training data
python -c "
from esper.simic.features import obs_to_base_features
# Test with extreme observations
assert all(f in [-1.0, 1.0] for f in features[2:35])  # All normalized
"
```

---

## Final Assessment

### Approval Recommendation: **CONDITIONAL PASS**

**Approval Status:**
- ✓ Feature extraction is complete (35 dimensions)
- ✓ Blueprint one-hot encoding is correct
- ✗ Loss feature normalization is broken (9/35 features)
- ✗ available_slots lacks normalization

**Decision:**
- **APPROVE** the tensor implementation architecture
- **FLAG** for normalization fixes before training

The commit introduces valuable performance optimization (tensor-returning functions, vectorized operations), but inherits normalization bugs from the list implementation that must be fixed before using these features in PPO training.

**Estimated fix effort:** < 2 minutes (4 one-line changes)

---

## Code Locations

### Key Files
- `/home/john/esper-lite/src/esper/simic/features.py` (lines 70-214)
- `/home/john/esper-lite/tests/test_simic_features.py` (test coverage)

### Bug Locations
| Line | Issue | Fix |
|------|-------|-----|
| 112 | `safe(obs['train_loss'], 10.0)` | Add `max_val=10.0` |
| 113 | `safe(obs['val_loss'], 10.0)` | Add `max_val=10.0` |
| 122 | `safe(obs['best_val_loss'], 10.0)` | Add `max_val=10.0` |
| 124 | `safe(v, 10.0) for v in loss_history_5` | Add `max_val=10.0` |
| 132 | `float(obs['available_slots'])` | Add `/ 4.0` |
| 184 | `safe(obs['train_loss'], 10.0)` | Add `max_val=10.0` |
| 185 | `safe(obs['val_loss'], 10.0)` | Add `max_val=10.0` |
| 192 | `safe(obs['best_val_loss'], 10.0)` | Add `max_val=10.0` |

---

## References

### DRL Principles Applied

1. **State Normalization**: Features should be normalized to ~[-1, 1] before feeding to neural networks (Lillicrap et al., 2016; Mnih et al., 2016)

2. **One-Hot Encoding for Categoricals**: Categorical features should use one-hot representation to avoid imposing ordinal structure (best practice in deep RL and supervised learning)

3. **RunningMeanStd Stability**: Outlier values in observation space can destabilize running statistics, reducing sample efficiency (PPO paper, Schulman et al., 2017)

4. **Pre-normalization Value**: Normalizing before the network reduces reliance on normalizing layers for warmup stability (noted in OpenAI Baselines, Dhariwal et al., 2017)

---

**Review Date:** 2025-12-06
**Reviewer:** DRL Expert
**Commit:** deef9b6
**Diff from:** b9a3758
