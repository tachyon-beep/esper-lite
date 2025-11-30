# Architecture Report: Esper Morphogenetic Neural Networks

**Version**: 1.0
**Date**: 2025-11-30
**Confidence Level**: HIGH

---

## Executive Summary

Esper is a framework for **Morphogenetic AI** - neural networks that dynamically grow and adapt their topology during training. The system implements a novel "trust escalation" model where new neural modules (seeds) must prove their worth through a multi-stage lifecycle before permanent integration.

### Key Findings

| Aspect | Assessment |
|--------|------------|
| Architecture Quality | **Excellent** - Clean domain separation, well-defined contracts |
| Code Organization | **Excellent** - 6 cohesive subsystems with minimal coupling |
| Design Patterns | **Strong** - Factory, State Machine, Strategy, Observer patterns |
| Performance Design | **Strong** - Hot path optimizations, GPU-native operations |
| Documentation | **Good** - README, docstrings, inline comments present |
| Test Coverage | **Present** - Property-based testing with Hypothesis |
| Behavioral Maturity | **Partial** - Heuristic validated; PPO under active training |

### Core Innovation

The system's unique contribution is the **seed lifecycle state machine** with quality gates:

```
DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZED
                          ↓           ↓
                        CULLED → EMBARGOED → RESETTING → (recycle)
```

This prevents catastrophic forgetting by isolating new modules during training and gradually integrating them only after validation.

---

## System Overview

### Purpose

Train neural networks that can:
1. Dynamically grow capacity through "seed" modules
2. Learn optimal growth strategies via reinforcement learning
3. Prevent catastrophic forgetting through gradient isolation
4. Outperform static architectures on continual learning tasks

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| RL Algorithms | PPO (on-policy), IQL (offline) |
| GPU Support | CUDA streams, multi-GPU |
| Testing | pytest, Hypothesis |

---

## Architecture

### Domain Model

Esper uses a **biological metaphor** for its architecture:

| Domain | Metaphor | Responsibility |
|--------|----------|----------------|
| **Kasmina** | Body | Neural network mechanics, seed lifecycle |
| **Leyline** | Nervous System | Contracts, signals, schemas |
| **Tamiyo** | Brain | Strategic decision-making |
| **Tolaria** | Hands | Training loop execution |
| **Simic** | Gym | RL infrastructure for learning |
| **Nissa** | Senses | Telemetry and diagnostics |

### Dependency Graph

```
                    Scripts (Entry Points)
                         │
                         ▼
                       Simic ─────────────────┐
                         │                    │
            ┌────────────┼────────────┐       │
            ▼            ▼            ▼       │
         Tamiyo       Tolaria      Utils     │
            │            │                    │
            └────────┬───┘                    │
                     ▼                        │
                  Kasmina ◄───────────────────┘
                     │
                     ▼
                  Leyline (Foundation - No Dependencies)
```

### Key Design Decisions

#### 1. Two-Tier Signal System

**Problem**: High-frequency RL training creates GC pressure
**Solution**: FastTrainingSignals (NamedTuple) for hot path, full TrainingSignals for rich context

```python
# Hot path - zero allocations
class FastTrainingSignals(NamedTuple):
    epoch: int
    val_accuracy: float
    # ... 18 more fields as primitives

# Rich context - full dataclass
@dataclass
class TrainingSignals:
    metrics: TrainingMetrics
    active_seeds: list[str]
    # ... with methods and history
```

#### 2. Quality Gate Architecture

**Problem**: Premature integration causes catastrophic forgetting
**Solution**: Six quality gates (G0-G5) validate each stage transition

| Gate | Transition | Validation |
|------|------------|------------|
| G0 | → GERMINATED | seed_id, blueprint_id present |
| G1 | → TRAINING | Germination complete |
| G2 | → BLENDING | improvement > threshold |
| G3 | → SHADOWING | alpha >= 0.95 |
| G4 | → PROBATIONARY | Shadowing complete |
| G5 | → FOSSILIZED | total_improvement > 0 |

#### 3. Vectorized Training with CUDA Streams

**Problem**: Sequential environment execution underutilizes GPU
**Solution**: Inverted control flow - iterate batches first, parallelize environments

```
Traditional:  for env in envs: for batch in data: process()
Esper:        for batch in data: for env in envs: process_async()
```

Benefits:
- 4x throughput with 4 environments
- True parallel execution via CUDA streams
- Independent DataLoaders avoid GIL contention

#### 4. Potential-Based Reward Shaping (PBRS)

**Problem**: Sparse rewards make RL training unstable
**Solution**: PBRS provides dense feedback while preserving optimal policy

```python
reward = base_reward + gamma * potential(s') - potential(s)
```

Stage-based potentials guide exploration:
- TRAINING: 15.0
- BLENDING: 25.0
- FOSSILIZED: 10.0 (value realized)

---

## Subsystem Details

### Leyline (Foundation)

**Files**: 7
**LOC**: ~500
**Dependencies**: None

Core contracts:
- `SeedStage`: IntEnum defining lifecycle stages
- `Action`: Enum for controller actions
- `TrainingSignals`: Observation space
- `TensorSchema`: Feature indices for vectorized access

**Key Pattern**: Protocol-first design enables loose coupling

### Kasmina (Model Mechanics)

**Files**: 4
**LOC**: ~700
**Dependencies**: Leyline

Core components:
- `MorphogeneticModel`: Host network with injection points
- `SeedSlot`: Lifecycle manager with quality gates
- `BlueprintCatalog`: Factory for seed modules

**Key Pattern**: State machine with validated transitions

### Tamiyo (Decision Making)

**Files**: 3
**LOC**: ~350
**Dependencies**: Leyline, Kasmina (types only)

Core components:
- `HeuristicTamiyo`: Rule-based baseline policy
- `SignalTracker`: Training signal aggregation
- `TamiyoDecision`: Action with confidence

**Key Pattern**: Strategy pattern for interchangeable policies

### Tolaria (Training Execution)

**Files**: 2
**LOC**: ~200
**Dependencies**: Leyline, Kasmina

Core functions:
- `train_epoch_normal`: Standard training
- `train_epoch_seed_isolated`: Seed-only updates
- `train_epoch_blended`: Joint host+seed updates

**Key Pattern**: Function-based API, no classes

### Simic (RL Infrastructure)

**Files**: 11
**LOC**: ~1500
**Dependencies**: All others

Core components:
- `PPOAgent`: On-policy RL
- `train_ppo_vectorized`: Multi-GPU training
- `compute_shaped_reward`: PBRS implementation
- `obs_to_base_features`: Hot path feature extraction

**Key Pattern**: Inverted control flow for parallelization

### Nissa (Telemetry)

**Files**: 3
**LOC**: ~300
**Dependencies**: Leyline (loose)

Core components:
- `NissaHub`: Event router
- `DiagnosticTracker`: Rich training diagnostics
- `OutputBackend`: Pluggable output destinations

**Key Pattern**: Hub-and-spoke for event routing

---

## Performance Characteristics

### Hot Path Optimizations

| Optimization | Location | Benefit |
|-------------|----------|---------|
| NamedTuple signals | leyline/signals.py | Zero GC pressure |
| Slots dataclasses | Multiple | 40% faster attribute access |
| GPU-native normalization | simic/normalization.py | No CPU sync |
| CUDA streams | simic/vectorized.py | Parallel execution |
| Singleton configs | simic/rewards.py | No allocations |

### Scalability

| Dimension | Support |
|-----------|---------|
| Multi-environment | 4+ parallel (tested) |
| Multi-GPU | Round-robin device mapping |
| Episode length | Up to 75 epochs |
| Feature dimensions | 27 base + 10 telemetry |

---

## Preliminary Results

| Approach | CIFAR-10 Accuracy |
|----------|-------------------|
| Static Baseline | 69.31% |
| From-Scratch (larger) | 65.97% |
| **Esper (Heuristic)** | **82.16%** |
| Esper (PPO) | Training in progress |

The heuristic controller demonstrates 12.85% absolute improvement over static training.

---

## Risks and Considerations

### Technical Risks

1. **Complexity**: 6 subsystems require understanding of contracts
2. **GPU Memory**: Vectorized training scales memory with environments
3. **Hyperparameter Sensitivity**: Reward shaping weights need tuning

### Mitigation Strategies

1. Clear documentation and diagrams
2. Configurable n_envs parameter
3. RewardConfig dataclass with defaults

---

## Recommendations

### For Onboarding

1. Start with `01-discovery-findings.md` for high-level understanding
2. Read Leyline first (contracts define everything)
3. Trace a single training episode through the system
4. Use diagrams (`03-diagrams.md`) for navigation

### For Development

1. Follow existing patterns in each subsystem
2. Use TYPE_CHECKING imports to avoid circular dependencies
3. Add hot path optimizations for any new signal types
4. Write property-based tests for new state transitions

### For Operations

1. Monitor GPU memory usage with n_envs > 4
2. Use telemetry profiles (nissa) for debugging
3. Save checkpoints frequently (--save flag)
4. Use --resume for crash recovery

---

## Document Index

| Document | Purpose |
|----------|---------|
| `00-coordination.md` | Analysis methodology and decisions |
| `01-discovery-findings.md` | Initial codebase assessment |
| `02-subsystem-catalog.md` | Detailed subsystem documentation |
| `03-diagrams.md` | C4 architecture diagrams |
| `04-final-report.md` | This synthesized report |
| `temp/validation-*.md` | Validation evidence |

---

## Conclusion

Esper demonstrates a well-architected approach to morphogenetic neural networks. The clear domain separation, rigorous state machine design, and performance-conscious implementation position it well for both research experimentation and production use.

**Overall Assessment**:

> **Esper v1.0** - Production-ready *engine architecture*, validated on CIFAR-10 with heuristic Tamiyo (82.16%). RL-based Tamiyo (PPO) shows structural signal but remains under active training. Architecture is solid; behavioral validation ongoing.

**Note on stages**: The canonical `SeedStage` enum in `leyline/stages.py` is the source of truth. Future stages mentioned in `ROADMAP.md` (e.g., WITHERING for Phase 3 "Eagerness Repair") are not yet implemented.
