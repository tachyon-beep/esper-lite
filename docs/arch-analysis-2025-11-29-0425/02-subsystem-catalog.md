# Subsystem Catalog - Esper-Lite

## Overview

This catalog provides detailed documentation of all 7 subsystems in the Esper-lite morphogenetic neural network training framework. Each subsystem is analyzed for its purpose, public API, dependencies, and patterns.

---

## Leyline

### Overview
Leyline is the foundation layer of Esper that defines all data contracts and schemas flowing between subsystems. It provides immutable dataclass specifications for actions, commands, signals, lifecycle states, and telemetry events that establish the contract-based communication protocol across the entire system. Acting as the "invisible substrate," Leyline enforces type safety and semantic consistency without imposing operational logic.

### Location
`src/esper/leyline/`

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| SeedStage | stages.py | Seed lifecycle state machine with valid transitions |
| FastTrainingSignals | signals.py | NamedTuple for hot path observation (27 features) |
| TrainingSignals | signals.py | Rich dataclass with full training state |
| AdaptationCommand | schemas.py | Tamiyo→Kasmina command contract |
| SimicAction | actions.py | Discrete action space enum (7 actions) |
| SeedStateReport | reports.py | Kasmina→Tamiyo status report |

### Public API

**Lifecycle & Stages:**
- `SeedStage` - IntEnum: DORMANT, GERMINATED, TRAINING, BLENDING, SHADOWING, PROBATIONARY, FOSSILIZED, CULLED, EMBARGOED, RESETTING
- `CommandType` - Enum for command types
- `RiskLevel` - IntEnum (GREEN, YELLOW, ORANGE, RED, CRITICAL)
- `VALID_TRANSITIONS`, `is_valid_transition()`, `is_terminal_stage()`, `is_active_stage()`, `is_failure_stage()`

**Actions:**
- `SimicAction` - Enum: WAIT, GERMINATE_CONV, GERMINATE_ATTENTION, GERMINATE_NORM, GERMINATE_DEPTHWISE, ADVANCE, CULL

**Signals:**
- `TensorSchema` - IntEnum mapping 27 feature names to indices
- `FastTrainingSignals` - NamedTuple with `to_vector()` method
- `TrainingMetrics`, `TrainingSignals`

**Commands & Blueprints:**
- `AdaptationCommand`, `SeedOperation`, `GateLevel`, `GateResult`
- `BlueprintProtocol`, `BlueprintSpec`

**Reports:**
- `SeedMetrics`, `SeedStateReport`, `FieldReport`

**Telemetry:**
- `TelemetryEventType`, `TelemetryEvent`, `PerformanceBudgets`

### Dependencies
- **Inbound:** Kasmina, Tamiyo, Simic, Nissa, Tolaria, Scripts (all subsystems)
- **Outbound:** Python standard library only (no external packages)

### Patterns Observed
- Contract-First Design with frozen dataclasses
- Hot-Path Optimization (FastTrainingSignals vs TrainingSignals)
- State Machine Formalization with explicit validation
- Enum-Based Classification (IntEnum for performance)
- `__slots__` optimization for high-frequency types
- Protocol-Based Extensibility (BlueprintProtocol)

### Confidence: High
Complete analysis of all 7 Python files with explicit exports.

---

## Kasmina

### Overview
Kasmina manages the complete lifecycle of injectable seed modules that enhance the host neural network. It implements the seed mechanics pipeline—from germination through training, blending, and eventual fossilization—while providing gradient isolation guarantees that prevent seeds from destabilizing the host during training.

### Location
`src/esper/kasmina/`

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| SeedSlot | slot.py | Lifecycle manager for a single seed |
| MorphogeneticModel | host.py | Host CNN + seed slot container |
| QualityGates | slot.py | G0-G5 gate validation system |
| BlueprintCatalog | blueprints.py | Registry for seed architectures |
| GradientIsolationMonitor | isolation.py | Ensures host protection during seed training |

### Public API

**Slot Management:**
- `SeedSlot`, `SeedState`, `SeedMetrics`, `QualityGates`

**Blueprints:**
- `ConvBlock`, `ConvEnhanceSeed`, `AttentionSeed`, `NormSeed`, `DepthwiseSeed`
- `BlueprintCatalog`

**Isolation:**
- `AlphaSchedule`, `blend_with_isolation`, `GradientIsolationMonitor`

**Host:**
- `HostCNN`, `MorphogeneticModel`

### Dependencies
- **Inbound:** Tolaria (environment), Tamiyo (seed state inspection), main esper package
- **Outbound:** Leyline (stages, commands, reports), PyTorch

### Seed Lifecycle State Machine

```
DORMANT ──G0──> GERMINATED ──G1──> TRAINING ──G2──> BLENDING ──G3──> SHADOWING ──G4──> PROBATIONARY ──G5──> FOSSILIZED
                                      │                │                 │
                                      └───────> CULLED <─────────────────┘
```

**Gate Requirements:**
- G0: seed_id and blueprint_id present
- G1: germination complete
- G2: improvement ≥0.5%, isolation_violations ≤10
- G3: ≥3 epochs in blending, alpha ≥0.95
- G4: shadowing complete
- G5: positive total improvement, healthy status

### Patterns Observed
- Lifecycle State Machine with strict transition validation
- Quality Gates (G0-G5) for stage progression
- Gradient Isolation via explicit detachment
- Blueprint Factory Pattern
- Alpha Blending Schedule (sigmoid-based)
- Dataclass-Based State with `__slots__`

### Confidence: High
Clear contracts and comprehensive gate validation.

---

## Tamiyo

### Overview
Tamiyo is the strategic decision-making component (heuristic controller) that observes training signals and makes strategic decisions about seed lifecycle management. It implements a rule-based policy for managing seed germination, training, blending, and culling based on training metrics and performance indicators.

### Location
`src/esper/tamiyo/`

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| HeuristicTamiyo | heuristic.py | Rule-based strategic controller |
| TamiyoDecision | decisions.py | Decision data structure |
| SignalTracker | tracker.py | Running statistics observer |
| HeuristicPolicyConfig | heuristic.py | Configuration for thresholds |

### Public API

- `TamiyoAction` - Enum: WAIT, GERMINATE, ADVANCE_TRAINING, ADVANCE_BLENDING, ADVANCE_FOSSILIZE, CULL, CHANGE_BLUEPRINT
- `TamiyoDecision` - Decision record with action, target, reason, confidence
- `SignalTracker` - Metric accumulator with plateau detection
- `TamiyoPolicy` - Protocol for policy implementations
- `HeuristicPolicyConfig`, `HeuristicTamiyo`

### Dependencies
- **Inbound:** Simic (IQL/PPO), Scripts, main esper package
- **Outbound:** Leyline (stages, signals, commands), Kasmina (type-only SeedState)

### Decision Logic

**No Active Seeds:**
1. Reject if too early (epoch < min_epochs_before_germinate)
2. Germinate if plateau detected
3. Otherwise wait

**Active Seeds:**
- GERMINATED → Advance to TRAINING
- TRAINING → If improvement ≥ threshold → BLENDING; if degradation → CULL
- BLENDING → If positive improvement → FOSSILIZE; otherwise CULL

### Patterns Observed
- Protocol-based abstraction (TamiyoPolicy)
- Configuration-driven behavior
- Round-robin blueprint selection
- Stateful tracking with deques
- Rule-based stage transitions

### Confidence: High
Well-structured with clear module boundaries.

---

## Simic

### Overview
Simic is the reinforcement learning infrastructure for training Tamiyo policies to control the seed lifecycle. It implements two complementary RL approaches: PPO for online learning from live training episodes, and IQL/CQL for offline learning from collected data. The subsystem handles feature extraction, reward design, episode collection, and policy training.

### Location
`src/esper/simic/`

### Key Components

| Component | File | LOC | Description |
|-----------|------|-----|-------------|
| PPOAgent | ppo.py | 1591 | Online actor-critic with GAE |
| IQL | iql.py | 1326 | Offline Q-learning with expectile regression |
| Episode/Collector | episodes.py | 719 | Trajectory data structures |
| Rewards | rewards.py | 376 | Multi-component reward shaping |
| Features | features.py | 161 | 27-dim feature extraction (hot path) |
| PolicyNetwork | networks.py | 342 | MLP for imitation learning |

### Public API

**Observations & Episodes:**
- `TrainingSnapshot`, `ActionTaken`, `StepOutcome`, `DecisionPoint`
- `Episode`, `EpisodeCollector`, `DatasetManager`
- `snapshot_from_signals`, `action_from_decision`

**Rewards:**
- `compute_shaped_reward()`, `compute_potential()`, `compute_pbrs_bonus()`
- `RewardConfig`, `SeedInfo`
- `INTERVENTION_COSTS`, `STAGE_TRAINING`, `STAGE_BLENDING`, `STAGE_FOSSILIZED`

**Features:**
- `obs_to_base_features()`, `telemetry_to_features()`, `safe()`

**Networks:**
- `PolicyNetwork`, `print_confusion_matrix`

### Dependencies
- **Inbound:** Scripts (evaluate.py)
- **Outbound:** Leyline, Tamiyo, Kasmina, Tolaria, PyTorch

### RL Algorithms

**PPO:**
- Actor-Critic with shared feature extractor (2x256)
- GAE with λ=0.95, γ=0.99
- PPO-Clip (ε=0.2) with entropy bonus (0.01)
- Vectorized multi-GPU training support

**IQL:**
- Q-network and V-network (expectile τ=0.7)
- Soft target updates (rate=0.005)
- Optional CQL regularization
- Policy extraction via softmax (β=3.0)

### Reward Design

| Component | Values |
|-----------|--------|
| Base Signal | accuracy_improvement × 0.5 |
| Lifecycle Bonus | TRAINING +0.2, BLENDING +0.3, FOSSILIZED +0.5 |
| GERMINATE | no_seed +0.3, early +0.2, existing -0.3 |
| ADVANCE | improving +0.5, blending +0.4, premature -0.2 |
| CULL | failing +0.3, acceptable +0.1 |
| WAIT | plateau -0.1, stagnation -0.1 |
| Intervention Costs | GERMINATE -0.02, ADVANCE -0.01, CULL -0.005 |

### Patterns Observed
- Lazy module loading (PPO/IQL on demand)
- Hot path feature extraction
- Shaped rewards with PBRS
- Vectorized environment architecture
- Running observation normalization

### Confidence: High
Comprehensive, well-documented with clear algorithm implementations.

---

## Nissa

### Overview
Nissa is the system telemetry hub, receiving carbon copies of events from all training domains and routing them to configured output backends. It provides rich diagnostic tracking with configurable profiles (minimal, standard, diagnostic, research) that collect gradient statistics, loss landscape analysis, per-class metrics, and training narratives.

### Location
`src/esper/nissa/`

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| TelemetryConfig | config.py | Pydantic configuration with profile management |
| DiagnosticTracker | tracker.py | Rich telemetry collector with gradient hooks |
| NissaHub | output.py | Central event router to backends |
| ConsoleOutput/FileOutput | output.py | Output backends |

### Public API

**Config:**
- `TelemetryConfig`, `GradientConfig`, `LossLandscapeConfig`, `PerClassConfig`

**Tracker:**
- `DiagnosticTracker`, `GradientStats`, `GradientHealth`, `EpochSnapshot`

**Output:**
- `OutputBackend`, `ConsoleOutput`, `FileOutput`
- `NissaHub`, `get_hub()`, `emit()`

### Dependencies
- **Inbound:** Scripts, Simic
- **Outbound:** Leyline (TelemetryEvent), PyTorch, NumPy, Pydantic, YAML

### Telemetry Profiles

| Profile | Gradient | Loss Landscape | Per-Class | Impact |
|---------|----------|----------------|-----------|--------|
| minimal | Off | Off | Off | Negligible |
| standard | Basic | Off | Off | Low |
| diagnostic | Full | On | On | Medium |
| research | Full | Full | Full | High |

### Patterns Observed
- Pydantic-based validation with field_validator
- Profile-based configuration with deep_merge
- Singleton pattern for hub (get_hub())
- Backend failure isolation
- Deque-based bounded history

### Confidence: High
Complete package with clear responsibilities.

---

## Tolaria

### Overview
Tolaria owns the training loop infrastructure for Model Alpha (the morphogenetic neural network). It provides generic epoch training functions for different seed lifecycle stages (normal, seed-isolated, blended) and a model factory. Tolaria is decoupled from dataset loading but tightly coupled with Kasmina.

### Location
`src/esper/tolaria/`

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| create_model | environment.py | Factory for MorphogeneticModel |
| train_epoch_normal | trainer.py | Standard training without seed |
| train_epoch_seed_isolated | trainer.py | Seed-only training (host frozen) |
| train_epoch_blended | trainer.py | Joint host+seed training |
| validate_and_get_metrics | trainer.py | Validation with per-class tracking |

### Public API

- `create_model()` - Factory function
- `train_epoch_normal()` - Standard epoch
- `train_epoch_seed_isolated()` - Seed exclusive
- `train_epoch_blended()` - Dual optimizer
- `validate_and_get_metrics()` - Validation

### Dependencies
- **Inbound:** Simic, Scripts
- **Outbound:** Kasmina (HostCNN, MorphogeneticModel), PyTorch

### Patterns Observed
- Generic trainer decoupled from datasets
- Three training modes for lifecycle stages
- Optimizer isolation in seed_isolated mode
- Device abstraction (cuda/cpu)

### Confidence: High
Simple, focused module.

---

## Utils

### Overview
Utils is a shared utilities package containing dataset loading utilities that don't belong to specific domain subsystems. Currently focused on CIFAR-10 dataset loading with extensibility for future datasets.

### Location
`src/esper/utils/`

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| load_cifar10 | data.py | CIFAR-10 DataLoader factory |

### Public API

- `load_cifar10(batch_size, data_root, generator)` → `(trainloader, testloader)`

### Dependencies
- **Inbound:** Tolaria (implicit), Simic, Scripts
- **Outbound:** PyTorch, Torchvision

### Patterns Observed
- Standard CIFAR-10 preprocessing (0.5 mean/std normalization)
- Generator parameter for reproducible shuffling
- Symmetric train/test batch sizes

### Confidence: High
Minimal, single-responsibility module.

---

## Dependency Graph

```
                    ┌─────────────┐
                    │   Leyline   │  (Foundation - No Dependencies)
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌──────────┐      ┌─────────┐
    │ Kasmina │      │  Tamiyo  │      │  Nissa  │
    └────┬────┘      └────┬─────┘      └─────────┘
         │                │
         │    ┌───────────┤
         ▼    ▼           │
    ┌──────────────┐      │
    │   Tolaria    │      │
    └──────┬───────┘      │
           │              │
           ▼              ▼
    ┌─────────────────────────┐
    │         Simic           │
    └─────────────────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │        Scripts          │
    └─────────────────────────┘

    Utils ─── (used by Tolaria, Simic, Scripts for data loading)
```

---

## Summary Statistics

| Subsystem | Files | LOC | Public Types | Confidence |
|-----------|-------|-----|--------------|------------|
| Leyline | 7 | ~600 | 25+ | High |
| Kasmina | 4 | ~1,100 | 15 | High |
| Tamiyo | 4 | ~500 | 6 | High |
| Simic | 7 | ~4,600 | 20+ | High |
| Nissa | 4 | ~1,000 | 12 | High |
| Tolaria | 2 | ~270 | 5 | High |
| Utils | 2 | ~70 | 1 | High |
| **Total** | **30** | **~9,200** | **84+** | **High** |
