# Fossilize Lookahead & Observation Enrichment Plan

> **Status:** Planning
> **Date:** 2025-12-23
> **Goal:** Enhance Tamiyo's decision-making with fossilize lookahead and verify counterfactual observations

---

## Executive Summary

Three options were evaluated:

| Option | Description | Status |
|--------|-------------|--------|
| **1. Fossilize Lookahead** | Add counterfactual evaluation before G5 fossilization | To implement |
| **2. Counterfactual in Observations** | Add contribution to policy obs | **Already done!** |
| **3. "Parallel Blended Upstream"** | Explore architectural concept | Discussion needed |

---

## Option 1: Fossilize Lookahead at G5 Gate

### Current Behavior

G5 gate at `slot.py:736-778` checks:
```python
contribution = state.metrics.counterfactual_contribution
if contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION:
    # Allow fossilization
```

This is a **one-shot snapshot** - it only considers the current counterfactual contribution at the moment of the fossilization request.

### Proposed Enhancement

Add a **fossilize value estimate** that answers: "What is the expected long-term value of keeping this seed vs removing it?"

Two approaches:

#### Approach A: Trend-Based Lookahead (Lightweight)

Use historical contribution trend to predict future value:
```python
# In _check_g5():
contribution = state.metrics.counterfactual_contribution
contribution_trend = state.metrics.contribution_velocity  # NEW: Δcontribution over last N epochs

# Lookahead estimate: current + projected future
projected_contribution = contribution + contribution_trend * LOOKAHEAD_EPOCHS
fossilize_value = (contribution + projected_contribution) / 2  # Avg of now + projected
```

**Pros:** No additional forward passes, uses existing data
**Cons:** Assumes linear trend, may not capture nonlinear dynamics

#### Approach B: Counterfactual Future Simulation (Heavyweight)

Actually run forward passes with/without the seed on held-out validation data:
```python
# Before G5 decision:
acc_with_seed = model.fused_forward(val_batch, alpha_overrides={slot: 1.0})
acc_without_seed = model.fused_forward(val_batch, alpha_overrides={slot: 0.0})
fossilize_value = acc_with_seed - acc_without_seed
```

**Pros:** Ground-truth counterfactual
**Cons:** Requires additional forward passes, compute cost

### Recommended: Hybrid Approach

1. **Always compute:** Trend-based lookahead (cheap)
2. **Add to observations:** `fossilize_value_estimate` for policy to consider
3. **Optional:** For high-stakes fossilization, run actual counterfactual simulation

### Implementation Tasks

#### Task 1: Add contribution velocity tracking to SeedMetrics ✅
- **File:** `src/esper/leyline/reports.py`
- **Change:** Added `contribution_velocity: float = 0.0` and `_prev_contribution: float | None = None` fields
- **Status:** COMPLETED

#### Task 2: Compute contribution velocity in training loop ✅
- **File:** `src/esper/simic/training/vectorized.py`
- **Change:** Added EMA velocity computation with decay 0.7 in counterfactual validation section
- **Status:** COMPLETED

#### Task 3: Add fossilize_value to observations ✅
- **File:** `src/esper/tamiyo/policy/features.py`
- **Change:**
  - Added `_FOSSILIZE_LOOKAHEAD_EPOCHS = 3.0` constant
  - Increased `SLOT_FEATURE_SIZE` from 25 to 26
  - Updated `MULTISLOT_FEATURE_SIZE` from 98 to 101
  - Added `fossilize_value` feature at per-slot index 4 (after `improvement`)
  - Formula: `fossilize_value = contribution + velocity * 3 epochs`
- **Status:** COMPLETED

#### Task 4: Use fossilize_value in G5 gate (optional enhancement)
- **File:** `src/esper/kasmina/slot.py`
- **Change:** Consider trend in fossilization decision
- **Status:** DEFERRED (policy can use the new observation to learn this)

---

## Option 2: Counterfactual in Observations (ALREADY IMPLEMENTED)

### Verification

The counterfactual contribution IS already in observations:

**Source:** `src/esper/simic/agent/ppo.py:125-127`
```python
contribution = (
    report.metrics.counterfactual_contribution
    if report.metrics.counterfactual_contribution is not None
    else report.metrics.improvement_since_stage_start
)
```

**Feature extraction:** `src/esper/tamiyo/policy/features.py:266`
```python
safe(slot.get("improvement", 0.0), 0.0, max_val=_IMPROVEMENT_CLAMP_PCT_PTS) / _IMPROVEMENT_CLAMP_PCT_PTS
```

**Position in observation vector:** Per-slot index 3 (`improvement`)

### Verification Task

- Run a training episode and log observation vectors
- Confirm `improvement` field correlates with actual counterfactual contribution
- Verify the normalization to [-1, 1] is appropriate for the observed ranges

---

## Option 3: "Parallel Blended Seed Upstream" Exploration

This is a conceptual discussion with the user to understand what they envision.

### Possible Interpretations

1. **Multi-alpha policy exploration:** Policy evaluates multiple alpha configurations before committing
2. **Ensemble blending:** Multiple seeds at different alphas combined upstream
3. **Hierarchical blending:** Seeds that modulate other seeds before reaching host
4. **Speculative execution:** Forward passes with candidate actions before action selection

### Questions for User

1. What does "upstream" mean in your mental model? Before which layer/segment?
2. Is this about Tamiyo having more information, or about changing the forward pass architecture?
3. Should this be during training (learning signal) or inference (decision-making)?

---

## Implementation Sequence

1. **Verify Option 2** - Confirm counterfactual is flowing to observations correctly
2. **Implement Option 1 (lightweight)** - Add contribution velocity + fossilize_value_estimate
3. **Discuss Option 3** - Interactive exploration of "upstream blending" concept

---

## Test Plan

### Option 1 Tests
- [ ] Unit test: contribution_velocity computation is correct
- [ ] Unit test: fossilize_value_estimate combines current + projected
- [ ] Integration test: New observation fields are present and normalized

### Option 2 Verification
- [ ] Log observations during training, confirm improvement field populated
- [ ] Compare logged improvement vs actual counterfactual_contribution
