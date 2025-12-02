# SME Report: Simic Rewards & Features

**Files:** rewards.py, features.py, normalization.py
**Location:** `src/esper/simic/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert

---

## 1. rewards.py - Reward Shaping

### Purpose
Comprehensive reward shaping for seed lifecycle control with PBRS (Potential-Based Reward Shaping) and action-specific bonuses.

### Key Components

| Component | Description |
|-----------|-------------|
| `RewardConfig` | Phase 1 (accuracy-primary) parameters |
| `LossRewardConfig` | Phase 2 (loss-primary) parameters |
| `SeedInfo` | Lightweight seed state NamedTuple |
| `compute_shaped_reward()` | Main reward computation |
| `compute_loss_reward()` | Loss-primary variant |
| `compute_seed_potential()` | PBRS potential function |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| PBRS Correctness | CORRECT | Follows Ng et al. (1999) |
| Stage Potentials | GOOD | Flattened to prevent farming |
| Action Shaping | INTENTIONAL | Not PBRS (domain-specific) |
| Compute Rent | GOOD | Quadratic penalty on bloat |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Hot Path Design | EXCELLENT | Pure Python, no tensors |
| Memory | GOOD | NamedTuple for SeedInfo |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| MEDIUM | Unused compute_potential() function | rewards.py:180 |
| MEDIUM | String matching for action names | rewards.py:290 |
| LOW | stage_potentials dict recreated | rewards.py:220 |

---

## 2. features.py - Feature Extraction

### Purpose
Hot-path feature extraction producing 27-dimensional observation vector for PPO.

### Key Components

| Component | Description |
|-----------|-------------|
| `safe()` | Value sanitization (None, inf, nan) |
| `obs_to_base_features()` | 27-dim feature extraction |
| `TaskConfig` | Task-specific parameters |
| `normalize_observation()` | Per-field normalization |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Feature Coverage | GOOD | Temporal, performance, trends, history, seed |
| Missing Features | MEDIUM | No gradient health in base features |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Import Discipline | EXCELLENT | Only leyline imports |
| Vectorization | NEEDS WORK | List operations don't vectorize |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| HIGH | List unpacking creates allocations | features.py:85 |
| MEDIUM | normalize_observation() returns 9, expects 27 | features.py:140 |
| MEDIUM | Unused TensorSchema import | features.py:12 |

---

## 3. normalization.py - Running Stats

### Purpose
GPU-native running mean/std using Welford's numerically stable algorithm.

### Key Components

| Component | Description |
|-----------|-------------|
| `RunningMeanStd` | GPU-native statistics tracker |
| `update()` | Welford's algorithm with auto-migration |
| `normalize()` | Z-score with clipping |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Welford Algorithm | CORRECT | Numerically stable |
| Clipping | STANDARD | ±10.0 range |
| Missing | Windowed/decay variants | For non-stationary envs |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| GPU-Native | EXCELLENT | Tensor-based count |
| Auto-Migration | GOOD | First-call device migration |
| No-Grad | GOOD | @torch.no_grad on update |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| HIGH | Reduction ops cause GPU sync | normalization.py:45 |
| MEDIUM | normalize() lacks @no_grad | normalization.py:55 |
| LOW | Missing state_dict methods | normalization.py |

---

## Recommendations Summary

| Priority | Recommendation | File |
|----------|----------------|------|
| P0 | Delete unused compute_potential() | rewards.py |
| P1 | Add @no_grad to normalize() | normalization.py |
| P1 | Fix normalize_observation() return count | features.py |
| P2 | Use enum for action matching | rewards.py |
| P2 | Add state_dict/load_state_dict | normalization.py |

---

**Quality Score:** 7.5/10 - Good PBRS implementation, hot path needs optimization
**Confidence:** HIGH
