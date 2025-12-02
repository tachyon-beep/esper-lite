# SME Report: Simic Core Files

**Files:** ppo.py, networks.py, buffers.py
**Location:** `src/esper/simic/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert

---

## 1. ppo.py - PPO Agent

### Purpose
PPO agent for seed lifecycle control with entropy annealing, gradient clipping, and checkpoint support.

### Key Components

| Component | Description |
|-----------|-------------|
| `PPOAgent` | Main agent class with network, optimizer, buffer |
| `get_action()` | Action sampling (deterministic/stochastic) |
| `update()` | PPO-Clip update with GAE |
| `signals_to_features()` | Observation conversion |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| PPO-Clip | CORRECT | Proper surrogate objective |
| GAE | CORRECT | Standard λ=0.95 |
| Advantage Normalization | CORRECT | Per-minibatch |
| Entropy Annealing | GOOD | Configurable schedule |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Device Handling | NEEDS WORK | Redundant transfers |
| Checkpointing | NEEDS WORK | Use weights_only=True |
| Inference Mode | GOOD | Proper context managers |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| MEDIUM | Redundant device transfers | ppo.py:145 |
| MEDIUM | Unsafe torch.load() | ppo.py:198 |
| LOW | Unused tracker parameter | ppo.py:50 |

---

## 2. networks.py - Neural Architectures

### Purpose
Neural network architectures for policy learning: ActorCritic (RL), PolicyNetwork (imitation), Q/V networks (offline RL).

### Key Components

| Component | Description |
|-----------|-------------|
| `ActorCritic` | Shared features + actor/critic heads |
| `PolicyNetwork` | Supervised imitation learning |
| `QNetwork` | State-action value function |
| `VNetwork` | State value function |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Weight Init | EXCELLENT | Orthogonal with proper gains |
| Distribution | CORRECT | Categorical for discrete actions |
| Architecture | GOOD | Shared features efficient |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Inference Mode | EXCELLENT | Proper usage in get_action |
| torch.compile | COMPATIBLE | No dynamic shapes |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| HIGH | None stub classes cause confusing errors | networks.py:463 |
| MEDIUM | Duplicated torch availability checks | networks.py:15,21 |

---

## 3. buffers.py - Trajectory Storage

### Purpose
PPO rollout buffer with GAE computation for trajectory storage and minibatch generation.

### Key Components

| Component | Description |
|-----------|-------------|
| `RolloutStep` | NamedTuple for single step |
| `RolloutBuffer` | Trajectory storage + GAE |
| `compute_returns_and_advantages()` | Standard GAE algorithm |
| `get_batches()` | Shuffled minibatch generator |

### DRL Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| GAE Implementation | CORRECT | Proper episode boundary handling |
| Advantage Normalization | CORRECT | Done in agent update |

### PyTorch Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Memory | GOOD | NamedTuple is lightweight |
| Batching | NEEDS WORK | Could pre-allocate tensors |

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| MEDIUM | Batch construction allocates per call | buffers.py:65 |

---

## Recommendations Summary

| Priority | Recommendation | File |
|----------|----------------|------|
| P0 | Fix None stub classes - raise ImportError | networks.py |
| P1 | Use weights_only=True in torch.load | ppo.py |
| P1 | Remove unused tracker parameter | ppo.py |
| P2 | Pre-allocate batch tensors | buffers.py |

---

**Quality Score:** 8/10 - Solid PPO implementation
**Confidence:** HIGH
