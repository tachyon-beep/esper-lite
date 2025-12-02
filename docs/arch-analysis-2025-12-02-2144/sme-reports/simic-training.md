# SME Report: Simic Training Files

**Files:** training.py, vectorized.py, gradient_collector.py
**Location:** `src/esper/simic/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert

---

## 1. training.py - Single-GPU Training

### Purpose
Single-environment PPO training loop for Tamiyo seed lifecycle controller.

### Key Components

| Component | Description |
|-----------|-------------|
| `run_ppo_episode()` | Main episode runner (300+ lines) |
| `train_heuristic()` | Heuristic baseline training |
| `train_ppo()` | PPO training entry point |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| PPO Integration | CORRECT | Proper buffer usage |
| Seed Lifecycle | CORRECT | Stage-appropriate training |
| Reward Shaping | CORRECT | PBRS integration |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Device Handling | GOOD | Consistent device usage |
| Optimizer Management | GOOD | Dual optimizers |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| HIGH | 300+ line function with 6 elif branches | training.py:50-352 |
| MEDIUM | Training loop duplicated 5× | training.py:120-234 |

---

## 2. vectorized.py - Multi-GPU Training

### Purpose
High-performance vectorized PPO with CUDA streams and inverted control flow.

### Key Components

| Component | Description |
|-----------|-------------|
| `ParallelEnvState` | Per-environment state container |
| `train_ppo_vectorized()` | Multi-GPU entry point |
| CUDA Streams | Async execution per environment |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Vectorization | GOOD | Independent environments |
| Buffer Updates | CORRECT | Proper episode collection |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| CUDA Streams | NEEDS WORK | Race conditions possible |
| Memory | GOOD | GPU-side accumulation |
| Non-blocking | GOOD | Async data transfers |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| **CRITICAL** | CUDA stream race condition | vectorized.py:518-522 |
| HIGH | Stream context discontinuity | vectorized.py:507-522 |
| HIGH | Normalized vs unnormalized state mismatch | vectorized.py:725-732 |

### Critical Issue Detail
```python
# vectorized.py:518-522
# Accumulator tensors may have concurrent writes from
# multiple environments on same device without stream sync
```

---

## 3. gradient_collector.py - Gradient Statistics

### Purpose
Lightweight vectorized gradient statistics for seed telemetry.

### Key Components

| Component | Description |
|-----------|-------------|
| `SeedGradientCollector` | Stateless collector |
| `collect_async()` | CUDA-stream safe collection |
| `materialize_grad_stats()` | Deferred tensor→Python |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Gradient Health | GOOD | Vanishing/exploding detection |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Async Design | EXCELLENT | No blocking in streams |
| Vectorization | GOOD | Uses torch._foreach_norm |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| MEDIUM | Private API torch._foreach_norm | gradient_collector.py:45 |

---

## Recommendations Summary

| Priority | Recommendation | File |
|----------|----------------|------|
| **P0** | Fix CUDA stream race condition | vectorized.py |
| **P0** | Store normalized states in buffer | vectorized.py |
| P1 | Refactor run_ppo_episode() | training.py |
| P1 | Add stream synchronization | vectorized.py |
| P2 | Replace _foreach_norm with public API | gradient_collector.py |

---

**Quality Score:** 6.5/10 - Critical issues in vectorized training
**Confidence:** HIGH
