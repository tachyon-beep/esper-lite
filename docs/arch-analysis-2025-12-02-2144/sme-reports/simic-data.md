# SME Report: Simic Data Files

**Files:** episodes.py, sanity.py
**Location:** `src/esper/simic/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert

---

## 1. episodes.py - Episode Data Structures

### Purpose
Complete data pipeline for RL trajectory collection: observations, actions, outcomes, episode management.

### Key Components

| Component | Description |
|-----------|-------------|
| `TrainingSnapshot` | 27-dim observation dataclass |
| `ActionTaken` | Action record with metadata |
| `StepOutcome` | Reward signal container |
| `DecisionPoint` | (obs, action, outcome) triplet |
| `Episode` | Complete trajectory |
| `EpisodeCollector` | Stateful builder |
| `DatasetManager` | Persistent storage |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Observation Space | GOOD | 27-dim fixed structure |
| Action Recording | GOOD | Includes blueprint, reason, confidence |
| Episode Boundaries | CORRECT | Proper terminal handling |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Tensor Creation | GOOD | batch_to_tensor pre-allocates |
| Serialization | GOOD | JSON with proper conversion |
| NaN Handling | GOOD | inf/NaN handled in vectorization |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| **CRITICAL** | compute_reward() is placeholder (acc_change × 10) | episodes.py:301 |
| HIGH | action: object type bypasses type safety | episodes.py:115 |
| HIGH | load_all() loads entire dataset into memory | episodes.py:450 |

### Critical Issue Detail
```python
# episodes.py:301
def compute_reward(self) -> float:
    return self.accuracy_change * 10.0  # Placeholder!
```
**Impact:** No meaningful learning signal for PPO

---

## 2. sanity.py - Runtime Checks

### Purpose
Lightweight sanity checks for debugging: reward magnitude, param ratios, tensor shapes.

### Key Components

| Component | Description |
|-----------|-------------|
| `check_reward_magnitude()` | Warn on large rewards |
| `log_params_ratio()` | Debug compute rent |
| `assert_slot_shape()` | Validate tensor dims |
| `SANITY_CHECKS_ENABLED` | Environment flag |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Reward Monitoring | GOOD | Catches outliers |
| Threshold | QUESTIONABLE | 10.0 may conflict with intrinsic rewards |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Overhead | MINIMAL | Guards only |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| HIGH | SANITY_CHECKS_ENABLED never used | sanity.py:12 |
| MEDIUM | check_reward_magnitude only warns, doesn't clip | sanity.py:25 |

---

## Recommendations Summary

| Priority | Recommendation | File |
|----------|----------------|------|
| **P0** | Replace placeholder compute_reward with PBRS | episodes.py |
| P1 | Add streaming DatasetManager | episodes.py |
| P1 | Wire SANITY_CHECKS_ENABLED to checks | sanity.py |
| P2 | Add proper action typing | episodes.py |
| P3 | Add reward clipping option | sanity.py |

---

**Quality Score:** 6/10 - Critical placeholder in reward computation
**Confidence:** HIGH
