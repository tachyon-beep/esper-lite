# Discovery Findings - Esper Morphogenetic Neural Networks

## Executive Summary

**Esper** is a framework for **Morphogenetic AI** - neural networks that dynamically grow, prune, and adapt their own topology during training. Instead of static architectures, Esper uses a lifecycle-driven approach where "seed" modules are germinated in isolation, trained on residuals, and carefully grafted into a stable "host" model only when they prove their worth.

**Core Innovation**: A trust escalation model where new neural modules must pass through stages (TRAINING → BLENDING → FOSSILIZED) with quality gates at each transition, preventing catastrophic forgetting.

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| RL Algorithms | PPO (on-policy), IQL (offline) |
| Parallelization | CUDA streams, multi-GPU vectorized training |
| Data Structures | Dataclasses with slots for performance |
| Dataset | CIFAR-10 (proof of concept) |
| Package Management | uv (pyproject.toml) |
| Testing | pytest with Hypothesis (property-based testing) |

## Directory Structure

```
src/esper/
├── leyline/      # Nervous System: Shared contracts, schemas, enums
├── kasmina/      # Body: Neural network model, slot management, grafting
├── tamiyo/       # Brain: Strategic decision-making (heuristic or RL)
├── tolaria/      # Hands: PyTorch training loops and execution
├── simic/        # Gym: RL infrastructure (PPO, IQL, features, rewards)
├── nissa/        # Senses: Telemetry, diagnostics, observability
├── utils/        # Shared utilities (data loading)
└── scripts/      # CLI entry points (train.py, evaluate.py)
```

## Subsystem Identification (6 Core + 2 Support)

### 1. Leyline (Nervous System)
**Role**: Shared data contracts that flow between all components.
**Location**: `src/esper/leyline/`
**Key Components**:
- `stages.py`: SeedStage enum (DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZED)
- `actions.py`: Action enum (WAIT, GERMINATE_*, ADVANCE, CULL)
- `schemas.py`: AdaptationCommand, GateLevel, GateResult
- `signals.py`: TrainingSignals, TensorSchema
- `telemetry.py`: TelemetryEvent, SeedTelemetry
**Dependencies**: None (foundational)

### 2. Kasmina (Body)
**Role**: Neural network model mechanics and seed lifecycle management.
**Location**: `src/esper/kasmina/`
**Key Components**:
- `host.py`: HostCNN, MorphogeneticModel (model with injection points)
- `slot.py`: SeedSlot, SeedState, SeedMetrics, QualityGates
- `blueprints.py`: Seed module blueprints (conv_enhance, attention, norm, depthwise)
- `isolation.py`: Gradient isolation, alpha blending
**Dependencies**: Leyline

### 3. Tamiyo (Brain)
**Role**: Strategic decision-making for seed lifecycle management.
**Location**: `src/esper/tamiyo/`
**Key Components**:
- `heuristic.py`: HeuristicTamiyo (rule-based policy)
- `decisions.py`: TamiyoDecision
- `tracker.py`: SignalTracker
**Dependencies**: Leyline, Kasmina (for SeedState)

### 4. Tolaria (Hands)
**Role**: PyTorch training loop execution engine.
**Location**: `src/esper/tolaria/`
**Key Components**:
- `trainer.py`: train_epoch_normal, train_epoch_seed_isolated, train_epoch_blended
- `environment.py`: create_model factory
**Dependencies**: Leyline, Kasmina

### 5. Simic (Gym)
**Role**: Reinforcement learning infrastructure for training Tamiyo.
**Location**: `src/esper/simic/`
**Key Components**:
- `ppo.py`: PPOAgent (on-policy RL)
- `iql.py`: IQL (offline RL)
- `vectorized.py`: Multi-GPU parallel training with CUDA streams
- `networks.py`: ActorCritic, PolicyNetwork
- `rewards.py`: Potential-based reward shaping (PBRS)
- `features.py`: Feature extraction (hot path, 27 base + 10 telemetry)
- `buffers.py`: RolloutBuffer, ReplayBuffer
- `normalization.py`: RunningMeanStd for observation normalization
**Dependencies**: Leyline, Kasmina, Tamiyo, Tolaria

### 6. Nissa (Senses)
**Role**: Telemetry hub for observability and diagnostics.
**Location**: `src/esper/nissa/`
**Key Components**:
- `config.py`: TelemetryConfig with profile management
- `tracker.py`: DiagnosticTracker, GradientStats
- `output.py`: NissaHub, OutputBackend (console, file)
**Dependencies**: Leyline (loosely coupled)

### 7. Utils (Support)
**Location**: `src/esper/utils/`
**Role**: Shared utilities (CIFAR-10 loading)

### 8. Scripts (Entry Points)
**Location**: `src/esper/scripts/`
**Key Components**:
- `train.py`: CLI for PPO/IQL training
- `evaluate.py`: Evaluation scripts

## Core Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                │
│  ┌───────────┐                                                      │
│  │  Tolaria  │ ─── train_epoch() ──→ [PyTorch backward/step]        │
│  │  (Hands)  │                                                      │
│  └─────┬─────┘                                                      │
│        │                                                            │
│        ▼                                                            │
│  ┌───────────┐    TrainingSignals     ┌───────────┐                │
│  │  Kasmina  │ ◄──────────────────── │  Leyline   │                │
│  │  (Body)   │                        │ (Contracts)│                │
│  │  - Host   │ ── SeedStateReport ──► │            │                │
│  │  - Slots  │                        └─────┬─────┘                │
│  └─────┬─────┘                              │                       │
│        │                                    │                       │
│        ▼                                    ▼                       │
│  ┌───────────┐    TamiyoDecision    ┌───────────┐                  │
│  │  Tamiyo   │ ◄─────────────────── │   Simic   │ (RL learns to    │
│  │  (Brain)  │                      │   (Gym)   │  improve Tamiyo) │
│  │ Heuristic │ ─── Action ────────► │ PPO/IQL   │                  │
│  └───────────┘                      └───────────┘                  │
│        │                                                            │
│        ▼                                                            │
│  ┌───────────┐                                                      │
│  │   Nissa   │ ◄── TelemetryEvent (carbon copies from all domains)│
│  │  (Senses) │                                                      │
│  └───────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## The Seed Lifecycle (Trust Escalation Model)

```
DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► FOSSILIZED
    │            │             │            │            │
    │            │             │            │            └── Terminal success
    │            │             │            └── Alpha blending 0→1
    │            │             └── Isolated training (host frozen)
    │            └── Sanity checks passed
    └── Empty slot

    ──► CULLED ──► EMBARGOED ──► RESETTING ──► (back to DORMANT)
         │              │
         │              └── Cooldown (anti-thrashing)
         └── Performance regression
```

**Quality Gates**: G0 (basic sanity) → G1 (training readiness) → G2 (blending readiness) → ... → G5 (fossilization readiness)

## Key Architectural Patterns

### 1. Gradient Isolation
Seeds train on host errors without destabilizing existing knowledge. Host weights are frozen during seed TRAINING stage, preventing catastrophic forgetting.

### 2. Alpha Blending
During BLENDING stage, seed outputs are gradually integrated: `output = (1 - alpha) * host + alpha * seed`, where alpha ramps from 0 to 1.

### 3. Vectorized Training (CUDA Streams)
Simic uses **inverted control flow** - iterate dataloader batches FIRST, then run all environments in parallel using CUDA streams. This maximizes GPU utilization.

### 4. Potential-Based Reward Shaping (PBRS)
Reward function uses potential-based shaping to guide RL without changing optimal policy:
- Accuracy improvement rewards
- Stage progression rewards
- Intervention costs (germinate/cull penalties)

### 5. Observation Normalization
RunningMeanStd normalizes observations for stable RL training, with GPU-native implementation.

## Codebase Statistics

- **Total Python Files**: 43
- **Estimated LOC**: ~10,000
- **Test Files**: Located in `tests/`
- **Documentation**: AGENTS.md, README.md, CLAUDE.md

## Entry Points

1. **PPO Training (Vectorized)**:
   ```bash
   PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 4 --device cuda:0
   ```

2. **IQL Training (Offline)**:
   ```bash
   PYTHONPATH=src python -m esper.scripts.train iql --pack data/pack.json --epochs 100
   ```

## Preliminary Performance Results

| Approach | Final Accuracy |
|----------|----------------|
| Static Baseline | 69.31% |
| From-Scratch | 65.97% |
| Esper (Heuristic) | **82.16%** |
| Esper (PPO) | Training... |

## Confidence Level

**HIGH** - The codebase is well-documented with clear module boundaries, docstrings, and a comprehensive README. The architecture follows a consistent domain-driven design pattern.
