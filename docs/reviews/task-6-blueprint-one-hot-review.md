# Code Review: Task 6 - Blueprint ID to Observation Space
**Commit:** `2532a07`

**Reviewer Assessment:** NEEDS_FIXES

---

## Summary

Commit 2532a07 adds blueprint ID as a one-hot encoded feature to the observation space, increasing dimensions from 30 to 35. The implementation is **mostly sound** but contains **one critical bug** that prevents correct feature flow to the PPO agent. The one-hot encoding itself is well-designed and the numerical properties are sound.

---

## 1. One-Hot Implementation ✓ CORRECT

### Implementation Review

**Location:** `src/esper/simic/features.py` lines 95-102

```python
# Blueprint one-hot encoding (DRL Expert recommendation)
blueprint_id = obs.get('seed_blueprint_id', 0)
num_blueprints = obs.get('num_blueprints', 5)
blueprint_one_hot = [0.0] * num_blueprints
if blueprint_id > 0 and blueprint_id <= num_blueprints:
    blueprint_one_hot[blueprint_id - 1] = 1.0  # 1-indexed to 0-indexed
```

**Assessment: CORRECT**

✓ **Proper one-hot encoding:** Creates a dense categorical representation without imposing artificial ordinal relationships
✓ **Boundary handling:** Correctly converts 1-indexed blueprint IDs (1-5) to 0-indexed array positions (0-4)
✓ **Default behavior:** All zeros when `blueprint_id=0` (no active blueprint), which is semantically correct
✓ **Safe access:** Uses `.get()` with sensible defaults (`seed_blueprint_id=0`, `num_blueprints=5`)
✓ **Robust validation:** Includes bounds check `blueprint_id > 0 and blueprint_id <= num_blueprints`

### Why This Approach is Sound

One-hot encoding is the **correct choice** for categorical features in neural networks because:

1. **No false ordinality:** Unlike passing blueprint_id directly (0-5), one-hot prevents the network from learning misleading relationships (blueprint 3 is not "between" blueprints 2 and 4)
2. **Standard DRL practice:** This matches recommendations from DRL literature for discrete categorical state features
3. **Easy interpretability:** Each dimension has clear meaning for reward shaping and policy analysis

---

## 2. Numerical Concerns with Sparse One-Hot Features ⚠ MOSTLY SOUND

### Feature Distribution Analysis

**Location:** `src/esper/simic/features.py` lines 104-136

The one-hot features are **appended directly** to the feature list without normalization:

```python
return [
    # ... 30 normalized features (all in [0, 1] range) ...
    *blueprint_one_hot,  # 5 features, exactly one is 1.0 or all zeros
]
```

**Assessment: ACCEPTABLE, with caveats**

#### Variance Properties

The one-hot encoding has extreme variance properties:
- **Per-dimension:** Each blueprint bit has variance = `p(1-p)` where p ≈ 0.2 (assuming uniform blueprint distribution)
  - Expected variance per dimension: 0.2 × 0.8 = **0.16**
- **Comparison:** Other features like `epoch/max_epochs` have variance ≈ **0.083** (uniform [0,1])
- **Imbalance:** One-hot dims are **~2x higher variance** than normalized continuous features

#### Why This Still Works

1. **RunningMeanStd handles variance imbalance:** The normalization layer (momentum=0.99 EMA) independently computes per-dimension statistics. High-variance dimensions will have correspondingly larger std estimates, normalizing them correctly.

2. **Sparse structure preserved:** The one-hot pattern (sum=1.0) is actually **advantageous**:
   - Creates clear state distinctions with minimal ambiguity
   - Easy for policy network to learn conditional behavior
   - No multicollinearity among blueprint dimensions

3. **Pre-normalization elsewhere:** Most features are already pre-normalized to ~[0,1], establishing a consistent scale. The one-hot features fit naturally into this distribution.

#### Minor Risk: Early Training

The **real risk** is **early in training** when `RunningMeanStd` statistics are poorly estimated:

- Iteration 1-100: RunningMeanStd count=100, statistics ~50% converged
- Blueprint features with variance ~0.16 might be over-weighted before stats stabilize
- However: **This is mitigated** by the comment in the docstring: _"Pre-normalizes features to ~[0, 1] range for early training stability. This reduces the burden on RunningMeanStd during the initial warmup phase"_

**Recommendation:** Monitor early training curves (first 5-10% of episodes) for unexpected policy instability, but no code changes needed.

---

## 3. RunningMeanStd Compatibility ✓ SOUND

### Normalization Flow

**Location:** `src/esper/simic/normalization.py` lines 48-106

The RunningMeanStd class is **fully compatible** with the new 35-dim feature space:

```python
# Construction (vectorized.py:275)
obs_normalizer = RunningMeanStd((state_dim,), device=device, momentum=0.99)

# Update (vectorized.py, batch processing)
obs_normalizer.update(state_batch)  # Welford or EMA per dimension

# Normalize
normalized_obs = obs_normalizer.normalize(obs_tensor)  # z-norm per dimension
```

**Assessment: CORRECT**

✓ **Per-dimension statistics:** Each of the 35 dimensions gets independent mean/variance estimates
✓ **EMA-based:** Using momentum=0.99 for slow adaptation prevents distribution shift
✓ **Clipping:** Outputs are clipped to [-10, 10] to prevent value function saturation

### Expected Behavior

When the normalizer processes 35-dim vectors:

| Phase | Mean Estimate | Variance Estimate | Expected Output |
|-------|---------------|-------------------|-----------------|
| Iteration 1 | All ~0.5 | Mixed convergence | Rough but improving |
| Iteration 100 | Converged | Converged | Stable z-norm |
| Blueprint dims | Blueprint avg ~0.2 | ~0.16 (higher) | Clipped to ~[-10,10] |

The normalizer will automatically learn that blueprint dimensions have higher variance and normalize accordingly.

---

## 🚨 CRITICAL BUG: Constant Not Updated

### Location
**File:** `src/esper/simic/vectorized.py` **Line:** 271

```python
BASE_FEATURE_DIM = 30  # ❌ OUTDATED - Should be 35
state_dim = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
```

### Problem

The constant was **not updated** when observation space changed 30 → 35:

- **Without telemetry:** Agent expects 30-dim input, receives 35-dim → **Shape mismatch**
- **With telemetry:** Agent expects 40-dim (30+10), receives 45-dim (35+10) → **Shape mismatch**
- **Impact:** Training will crash immediately with "input mismatch" error on first forward pass

### Test Coverage Gap

Tests were updated (e.g., `test_simic_ppo.py` line 50):
```python
# Comment updated: "Feature vector without telemetry must be exactly 35 dimensions."
# But the implementation constant was never updated
```

The test **comments** were updated but the **underlying constant** was not.

### Fix Required

```python
# BEFORE (BROKEN):
BASE_FEATURE_DIM = 30

# AFTER (CORRECT):
BASE_FEATURE_DIM = 35
```

This single line change will fix both paths:
- `state_dim = 35` (without telemetry)
- `state_dim = 45` (35 + 10 telemetry)

### Verification Checklist

After fix, verify:
1. [ ] `python -m pytest tests/test_simic_ppo.py::TestPPOFeatureDimensions -v` passes
2. [ ] `python -m pytest tests/test_simic_features.py::TestBlueprintOneHotEncoding -v` passes
3. [ ] Check `src/esper/simic/vectorized.py` line 271 is `BASE_FEATURE_DIM = 35`
4. [ ] Verify line 272 computes correctly: `state_dim = 35 (or 45 with telemetry)`

---

## Summary Table

| Aspect | Status | Evidence | Risk Level |
|--------|--------|----------|-----------|
| One-hot correctness | ✓ PASS | Proper bounds handling, semantics correct | None |
| Sparse feature handling | ✓ PASS | RunningMeanStd handles variance imbalance | Low |
| Normalization compatibility | ✓ PASS | Per-dimension statistics work correctly | None |
| **Constant update** | ❌ FAIL | BASE_FEATURE_DIM still 30, should be 35 | **CRITICAL** |
| Test coverage | ✓ PASS | Blueprint tests comprehensive | None |

---

## Recommendations

### Must Fix (Blocking)

1. **Update `BASE_FEATURE_DIM` in `src/esper/simic/vectorized.py` line 271:**
   ```python
   BASE_FEATURE_DIM = 35  # Updated for V3.1 (35 base + blueprint one-hot)
   ```

### Should Monitor (Post-Fix)

2. **Early training stability:** After fixing the constant, run a small training job and monitor:
   - First epoch policy entropy (should be stable, not collapse)
   - First epoch loss curve (should not spike unexpectedly)
   - If observing instability: May need feature scaling or warmup adjustments

3. **Blueprint utilization:** Add logging to verify blueprints are being encoded:
   ```python
   # In training loop, periodically log:
   print(f"Blueprint distribution: {blueprint_one_hot_features.mean(dim=0)}")
   ```

### Optional Enhancements (Non-Blocking)

4. **Document variance property:** Add comment explaining why one-hot has higher variance:
   ```python
   # Blueprint one-hot: higher variance (~0.16) than normalized features (~0.08),
   # but RunningMeanStd handles this correctly via per-dimension statistics.
   ```

---

## Detailed Analysis: Why Sparse One-Hot Works

### Distribution Assumptions

Assuming uniform blueprint distribution during training:
- P(blueprint=i) = 1/5 = 0.2 for each of 5 blueprints
- P(blueprint=0, none) = 0.2

For a single one-hot dimension:
- Value if active: 1.0 (probability 0.2)
- Value if inactive: 0.0 (probability 0.8)
- Mean: 0.2
- Variance: 0.2 × (1-0.2)² + 0.8 × (0-0.2)² = 0.16

### Normalization Impact

When RunningMeanStd processes the feature vector:

```
Raw features: [feat_0, ..., feat_29, blueprint_0, ..., blueprint_4]
Means:        [μ_0,   ..., μ_29,   ~0.2,        ..., ~0.2       ]
Stds:         [σ_0,   ..., σ_29,   ~0.4,        ..., ~0.4       ]
Normalized:   [(x_0-μ_0)/σ_0, ..., (bp_i-0.2)/0.4, ...]
             = [z_0, ..., z_29, ±0.5 or ±0.25, ...]
```

**Result:** One-hot features normalize to values in [-0.5, 1.25] range (due to integer nature), which is within the [-10, 10] clipping range and compatible with network weight initialization.

### Why Network Learns Correctly

The policy network sees normalized features:
- Most continuous features: N(0, 1) distribution (standard Gaussian after clipping)
- Blueprint features: Discrete {±0.5, ±0.25, 0} values (very discrete signal)

The **contrast helps learning:**
- Network can easily detect blueprint ID from discrete pattern
- Makes blueprint feature highly informative for conditional policy learning
- Avoids the feature drowning among continuous noise

---

## Assessment Conclusion

**Overall Grade: B+ (Needs one line fix)**

**One-hot encoding:** A+ (well-designed, semantically correct)
**Numerical properties:** A (sparse features handled correctly by normalizer)
**Integration:** F (constant not updated, will crash at runtime)

The implementation demonstrates strong understanding of DRL feature engineering, but contains a simple but critical oversight in constant maintenance. This is a **one-line fix** to make production-ready.

