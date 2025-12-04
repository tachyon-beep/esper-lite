# Discovery Findings

**Project:** esper-lite
**Analysis Date:** 2025-12-02
**Total LOC:** ~11,095 lines Python

## Executive Summary

Esper-lite is a **morphogenetic neural network training system** that uses reinforcement learning to control a "seed lifecycle" for dynamically adapting neural networks. The system implements meta-learning: an RL agent (Tamiyo) learns when and how to modify a base neural network (Model Alpha) through a lifecycle of germination → training → blending → fossilization.

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| PyTorch | 2.8.0+ | Neural networks, RL training |
| HuggingFace datasets | 4.4.1+ | Data loading |
| HuggingFace transformers | 4.57.3+ | Transformer architectures |
| Hypothesis | 6.148.3+ | Property-based testing |
| NumPy | 1.24.0+ | Numerical operations |

## Directory Structure

```
src/esper/
├── __init__.py          # Package root, re-exports key types
├── leyline/             # Data contracts and schemas (6 files)
│   ├── actions.py       # Action types (SimicAction)
│   ├── stages.py        # Seed lifecycle stages
│   ├── signals.py       # Training signals (hot path)
│   ├── schemas.py       # Specifications and protocols
│   ├── reports.py       # Metric reporting structures
│   └── telemetry.py     # Telemetry event contracts
├── kasmina/             # Seed mechanics (6 files + blueprints/)
│   ├── slot.py          # SeedSlot state management
│   ├── isolation.py     # Gradient isolation for seeds
│   ├── protocol.py      # HostProtocol interface
│   ├── host.py          # CNNHost, TransformerHost implementations
│   └── blueprints/      # Neural architecture blueprints (3 files)
│       ├── cnn.py       # CNN seed blueprints
│       ├── transformer.py # Transformer seed blueprints
│       └── registry.py  # Blueprint registration
├── simic/               # RL training infrastructure (12 files)
│   ├── buffers.py       # Rollout buffers for PPO
│   ├── normalization.py # Running mean/std for observations
│   ├── networks.py      # Policy networks (ActorCritic)
│   ├── rewards.py       # Reward shaping (PBRS)
│   ├── features.py      # Feature extraction (hot path)
│   ├── episodes.py      # Episode data structures
│   ├── ppo.py           # PPO agent implementation
│   ├── training.py      # Training loops
│   ├── vectorized.py    # Multi-GPU vectorized training
│   ├── gradient_collector.py # Gradient statistics
│   └── sanity.py        # Sanity checks
├── tamiyo/              # Strategic decision-making (3 files)
│   ├── decisions.py     # TamiyoDecision type
│   ├── tracker.py       # SignalTracker for observations
│   └── heuristic.py     # Heuristic baseline policy
├── nissa/               # Telemetry hub (4 files)
│   ├── config.py        # TelemetryConfig (Pydantic)
│   ├── tracker.py       # DiagnosticTracker
│   ├── output.py        # Output backends (console, file)
│   └── analytics.py     # Blueprint analytics
├── tolaria/             # Model training infrastructure (3 files)
│   ├── environment.py   # Model factory (create_model)
│   ├── trainer.py       # Epoch training functions
│   └── governor.py      # Fail-safe watchdog
├── runtime/             # Task presets (1 file)
│   └── tasks.py         # TaskSpec definitions
├── utils/               # Shared utilities (1 file)
│   └── data.py          # Dataset loading (CIFAR-10)
└── scripts/             # Entry points (2 files)
    ├── train.py         # Training CLI
    └── evaluate.py      # Evaluation CLI
```

## Subsystem Identification

### 1. Leyline (Data Contracts Layer)
**Location:** `esper/leyline/`
**Files:** 6
**Responsibility:** Defines all data contracts that flow between components. The "invisible substrate" of the system.

**Key Types:**
- `SimicAction` / `Action` - Discrete actions for seed lifecycle
- `SeedStage` - Lifecycle stages (DORMANT, GERMINATED, TRAINING, etc.)
- `TrainingSignals` / `FastTrainingSignals` - Training metrics (hot path)
- `SeedMetrics`, `SeedStateReport` - Reporting structures

**Coupling:** Low (dependency source, not sink)

---

### 2. Kasmina (Seed Mechanics)
**Location:** `esper/kasmina/`
**Files:** 6 + 3 (blueprints)
**Responsibility:** Manages seed lifecycle mechanics - germination, training, blending, fossilization.

**Key Types:**
- `SeedSlot` - Stateful container for a seed
- `MorphogeneticModel` - Model with seed capabilities
- `CNNHost`, `TransformerHost` - Host model implementations
- `BlueprintRegistry` - Neural architecture templates
- `GradientIsolationMonitor` - Prevents gradient contamination

**Coupling:** Depends on leyline; depended on by simic, tolaria

---

### 3. Simic (RL Training Infrastructure)
**Location:** `esper/simic/`
**Files:** 12
**Responsibility:** Reinforcement learning infrastructure for training the Tamiyo controller using PPO.

**Key Types:**
- `PPOAgent` - Proximal Policy Optimization implementation
- `ActorCritic` - Policy/value network
- `RolloutBuffer` - Experience buffer
- `RewardConfig` - Reward shaping parameters
- `RunningMeanStd` - Observation normalization

**Coupling:** High (depends on leyline, kasmina, tolaria, tamiyo)

**Complexity:** HIGHEST - Core RL implementation

---

### 4. Tamiyo (Strategic Decision-Making)
**Location:** `esper/tamiyo/`
**Files:** 3
**Responsibility:** The RL agent that observes training signals and makes strategic decisions about seed lifecycle.

**Key Types:**
- `TamiyoDecision` - Decision output type
- `SignalTracker` - Observation aggregation
- `HeuristicTamiyo` - Baseline heuristic policy

**Coupling:** Depends on leyline; used by simic

---

### 5. Nissa (Telemetry Hub)
**Location:** `esper/nissa/`
**Files:** 4
**Responsibility:** System-wide telemetry collection, routing to output backends.

**Key Types:**
- `NissaHub` - Central telemetry router
- `DiagnosticTracker` - Training diagnostics
- `TelemetryConfig` - Configuration (Pydantic)
- `ConsoleOutput`, `FileOutput` - Output backends

**Coupling:** Low (receives events from all domains)

---

### 6. Tolaria (Model Training)
**Location:** `esper/tolaria/`
**Files:** 3
**Responsibility:** Training loop for Model Alpha (the neural network being enhanced). Provides epoch trainers for different seed states.

**Key Types:**
- `create_model` - Model factory
- `train_epoch_normal`, `train_epoch_incubator_mode`, `train_epoch_blended` - Training modes
- `TolariaGovernor` - Fail-safe watchdog for catastrophic failures

**Coupling:** Depends on kasmina, leyline; used by simic

---

### 7. Runtime (Task Presets)
**Location:** `esper/runtime/`
**Files:** 1
**Responsibility:** Task specification utilities and presets.

**Key Types:**
- `TaskSpec` - Task configuration
- `get_task_spec` - Factory function

**Coupling:** Minimal

---

### 8. Utils (Shared Utilities)
**Location:** `esper/utils/`
**Files:** 1
**Responsibility:** Shared utilities (data loading).

**Key Types:**
- `load_cifar10` - Dataset loader

**Coupling:** Minimal

---

### 9. Scripts (Entry Points)
**Location:** `esper/scripts/`
**Files:** 2
**Responsibility:** CLI entry points for training and evaluation.

**Entry Points:**
- `python -m esper.scripts.train ppo` - Train PPO agent
- `python -m esper.scripts.train heuristic` - Train with heuristic
- `python -m esper.scripts.evaluate` - Evaluate models

---

## Architectural Patterns Observed

### 1. Meta-Learning Architecture
Two-level training:
- **Outer loop:** Simic trains Tamiyo (RL agent) to make seed lifecycle decisions
- **Inner loop:** Tolaria trains Model Alpha with seeds controlled by Tamiyo

### 2. Domain-Driven Design
Clear bounded contexts:
- Leyline = shared kernel (data contracts)
- Kasmina = seed domain
- Simic = RL domain
- Tolaria = model training domain
- Nissa = telemetry domain

### 3. State Machine (Seed Lifecycle)
Defined transitions in leyline.stages:
```
DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZED
                    ↘ FAILED (terminal)
```

### 4. Event-Driven Telemetry
Nissa hub receives carbon copies of events from all domains, routes to configured backends.

### 5. Gradient Isolation
kasmina.isolation prevents gradient contamination between seeds and host during blending phase.

### 6. Potential-Based Reward Shaping (PBRS)
simic.rewards implements PBRS for shaping the RL reward signal without affecting optimal policy.

## Entry Points

| Entry Point | Command | Purpose |
|-------------|---------|---------|
| train.py | `python -m esper.scripts.train ppo --episodes 100` | Train PPO agent |
| train.py | `python -m esper.scripts.train heuristic` | Train with heuristic baseline |
| train.py | `python -m esper.scripts.train ppo --vectorized` | Multi-GPU training |
| evaluate.py | `python -m esper.scripts.evaluate` | Model evaluation |

## Dependencies Between Subsystems

```
┌─────────────────────────────────────────────────────────┐
│                      scripts (entry)                     │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                     simic (RL core)                      │
│           depends on: leyline, kasmina, tolaria          │
└─────┬──────────────────┬──────────────────────┬─────────┘
      │                  │                      │
      ▼                  ▼                      ▼
┌──────────┐     ┌──────────────┐        ┌──────────┐
│  tamiyo  │     │   tolaria    │        │ kasmina  │
│(decisions)│    │(model train) │        │  (seeds) │
└─────┬────┘     └──────┬───────┘        └────┬─────┘
      │                  │                     │
      └────────┬─────────┴─────────────────────┘
               │
               ▼
      ┌─────────────────┐
      │     leyline     │
      │ (data contracts)│
      └─────────────────┘
               ▲
               │
      ┌────────┴────────┐
      │      nissa      │
      │   (telemetry)   │
      └─────────────────┘
```

## SME Report Granularity Decision

Based on complexity analysis:

| Package | Granularity | Reason |
|---------|-------------|--------|
| simic | **File-level** | 12 files, core RL implementation, highest complexity |
| kasmina | Package-level | Cohesive seed mechanics, moderate complexity |
| leyline | Package-level | Data contracts, moderate complexity |
| tamiyo | Package-level | 3 cohesive files, moderate complexity |
| nissa | Package-level | Telemetry hub, moderate complexity |
| tolaria | Package-level | Training infrastructure, moderate complexity |
| runtime | Package-level | Simple task presets |
| utils | Package-level | Simple utilities |
| scripts | Package-level | CLI entry points |

**Total SME Reports:** 8 package-level + 12 file-level (simic) = **20 SME reports**

## Confidence Level

**HIGH** - All packages have docstrings, clear public APIs, and well-defined responsibilities. The architecture is intentionally domain-driven with explicit data contracts.

## Open Questions for Deep Analysis

1. How does gradient isolation actually prevent contamination during blending?
2. What are the specific PBRS potential functions used?
3. How does vectorized training coordinate across GPUs?
4. What quality gates exist in the seed lifecycle state machine?
5. How does the governor detect and respond to catastrophic failures?
