# Subsystem Catalog

**Project:** esper-lite
**Analysis Date:** 2025-12-02
**Packages Analyzed:** 8 (leyline, kasmina, simic, tamiyo, nissa, tolaria, runtime, utils) + scripts

---

## 1. Leyline (Data Contracts Layer)

**Location:** `esper/leyline/`
**Files:** 7
**Confidence:** HIGH

### Package Responsibility
The leyline package defines the **data contracts and communication protocols** that bind all Esper subsystems together. It provides the shared language for: (1) stage transitions and seed lifecycle (stages.py), (2) actions available to controllers (actions.py), (3) observations flowing from training/seeds (signals.py, reports.py), (4) commands flowing from controllers to subsystems (schemas.py), and (5) events for system observability (telemetry.py). Leyline is the invisible substrate enabling all cross-subsystem communication.

### Key Public API

| Export | Purpose |
|--------|---------|
| `SeedStage` | Lifecycle stages enum (DORMANT → FOSSILIZED) |
| `Action` / `SimicAction` | Discrete action space for controllers |
| `build_action_enum()` | Dynamic action enum per topology |
| `TrainingSignals` | Rich observation contract |
| `FastTrainingSignals` | GC-free hot-path signals |
| `TensorSchema` | Feature indices for 27-dim observation |
| `AdaptationCommand` | Controller → subsystem commands |
| `SeedStateReport` | Seed snapshot from Kasmina |
| `FieldReport` | Complete seed lifecycle record |
| `TelemetryEvent` | Structured observability events |
| `BlueprintProtocol` | Contract for seed architectures |

### File Breakdown

| File | Responsibility |
|------|----------------|
| `actions.py` | Action space with dynamic enum building |
| `stages.py` | Seed lifecycle state machine (10 stages) |
| `signals.py` | Two-tier signal representation (fast + rich) |
| `schemas.py` | Commands, blueprints, quality gates |
| `reports.py` | SeedMetrics, SeedStateReport, FieldReport |
| `telemetry.py` | Event types, performance budgets, SeedTelemetry |

### Dependencies
- **Inbound:** simic, kasmina, tamiyo, nissa, scripts, runtime (most heavily imported package)
- **Outbound:** kasmina.blueprints (lazy import in actions.py to avoid circular dependency)

### Key Patterns
- Two-tier representation (FastTrainingSignals vs TrainingSignals)
- Protocol-based contracts (BlueprintProtocol)
- IntEnum for tensor indices (TensorSchema)
- Immutable command dataclass (frozen=True)
- State machine with explicit transition graph
- Slots optimization for memory efficiency

---

## 2. Kasmina (Seed Mechanics)

**Location:** `esper/kasmina/`
**Files:** 9 (6 main + 3 blueprints)
**Confidence:** HIGH

### Package Responsibility
Kasmina manages the **lifecycle and integration of seed modules into host neural networks**. It provides the mechanical framework for seed germination from registered blueprints, staged training with gradient isolation, smooth blending into the host via alpha schedules, and eventual fossilization. The package bridges seed lifecycle (managed by Leyline) and training mechanics (managed by Simic), exposing quality gates and injection points for external coordination.

### Key Public API

| Export | Purpose |
|--------|---------|
| `SeedSlot` | Lifecycle container for a single seed |
| `MorphogeneticModel` | Host + seed orchestrator |
| `HostProtocol` | Structural typing for graftable hosts |
| `CNNHost` / `TransformerHost` | Concrete host implementations |
| `BlueprintRegistry` | Dynamic seed architecture registry |
| `AlphaSchedule` | Sigmoid-based blending schedule |
| `blend_with_isolation()` | Gradient-safe feature blending |
| `QualityGates` | Multi-level stage transition validation (G0-G5) |
| `GradientIsolationMonitor` | Prevents gradient contamination |

### File Breakdown

| File | Responsibility |
|------|----------------|
| `slot.py` | SeedSlot, SeedState, SeedMetrics, QualityGates |
| `isolation.py` | AlphaSchedule, blend_with_isolation, GradientIsolationMonitor |
| `protocol.py` | HostProtocol (structural typing) |
| `host.py` | CNNHost, TransformerHost, MorphogeneticModel |
| `blueprints/registry.py` | BlueprintRegistry with decorator registration |
| `blueprints/cnn.py` | CNN seed blueprints (norm, attention, depthwise, conv) |
| `blueprints/transformer.py` | Transformer seeds (norm, LoRA, attention, MLP) |

### Dependencies
- **Inbound:** tolaria, runtime.tasks, tamiyo (TYPE_CHECKING), simic (via tolaria)
- **Outbound:** leyline (stages, gates, telemetry)

### Key Patterns
- State Machine with Quality Gates (G0-G5)
- Womb Mode (Straight-Through Estimator for gradient isolation)
- Plugin Registry (decorator-based blueprint registration)
- Injection Point Abstraction (HostProtocol)
- Smooth Blending (sigmoid-scheduled alpha ramp)
- Device Propagation (to() override for GPU management)

---

## 3. Simic (RL Training Infrastructure)

**Location:** `esper/simic/`
**Files:** 12
**Confidence:** HIGH
**Complexity:** HIGHEST

### Package Responsibility
The **esper.simic** package is the core reinforcement learning infrastructure for Tamiyo's seed lifecycle controller. It implements:
1. **Episode Management**: TrainingSnapshot, ActionTaken, Episode
2. **Policy Learning**: PPO with ActorCritic networks
3. **Reward Shaping**: Multi-component PBRS-based rewards
4. **Vectorized Training**: Multi-GPU CUDA streams
5. **Feature Extraction**: Hot-path observation processing
6. **Gradient Telemetry**: Per-epoch gradient statistics

### Key Public API

| Export | Purpose |
|--------|---------|
| `TrainingSnapshot` | 27-dim observation state |
| `Episode` / `EpisodeCollector` | Training trajectory management |
| `RolloutBuffer` | PPO experience buffer with GAE |
| `PPOAgent` | Online policy gradient agent |
| `ActorCritic` | RL network (actor + critic heads) |
| `RunningMeanStd` | GPU-native observation normalization |
| `RewardConfig` / `LossRewardConfig` | Reward hyperparameters |
| `compute_shaped_reward()` | Core reward computation |
| `obs_to_base_features()` | Hot-path feature extraction |
| `train_ppo_vectorized()` | Multi-GPU training entry point |

### File Breakdown

| File | Responsibility | Hot Path? |
|------|----------------|-----------|
| `episodes.py` | Episode data structures, serialization | No |
| `buffers.py` | RolloutBuffer with GAE computation | Yes (per-episode) |
| `normalization.py` | GPU-native RunningMeanStd | Yes |
| `features.py` | 27-dim feature extraction | **CRITICAL** |
| `rewards.py` | PBRS reward shaping | Yes |
| `networks.py` | ActorCritic, PolicyNetwork, Q/V networks | Yes |
| `ppo.py` | PPOAgent with entropy annealing | Yes |
| `training.py` | Single-GPU training loop | Yes |
| `vectorized.py` | Multi-GPU CUDA streams | **CRITICAL** |
| `gradient_collector.py` | Async gradient statistics | No |
| `sanity.py` | Optional runtime checks | No |

### Dependencies
- **Inbound:** runtime, scripts
- **Outbound:** leyline, tolaria, tamiyo, nissa, kasmina (indirect)

### Key Patterns
- PPO with clip ratio and entropy regularization
- PBRS (Potential-Based Reward Shaping)
- CUDA Streams for async multi-GPU execution
- Inverted Control Flow (batch-first iteration)
- Configuration Objects (RewardConfig, TaskConfig)
- Lazy Imports for heavy modules

### Performance-Critical Paths
1. `features.py:obs_to_base_features()` - once per step × n_envs
2. `rewards.py:compute_shaped_reward()` - once per step × n_envs
3. `vectorized.py` CUDA stream processing - main bottleneck
4. `buffers.py:compute_returns_and_advantages()` - once per episode

---

## 4. Tamiyo (Strategic Decision-Making)

**Location:** `esper/tamiyo/`
**Files:** 4
**Confidence:** HIGH

### Package Responsibility
Tamiyo is the strategic decision-making subsystem for Esper's adaptive learning architecture. It observes training progress through signal tracking and makes decisions about seed lifecycle management (germination, fossilization, culling) based on configurable heuristics. Tamiyo bridges the training signals flowing from the network and the commands that control seed operations in Kasmina.

### Key Public API

| Export | Purpose |
|--------|---------|
| `TamiyoDecision` | Strategic decision output |
| `SignalTracker` | Training signal observation engine |
| `TamiyoPolicy` | Protocol for policy implementations |
| `HeuristicPolicyConfig` | Configuration for heuristic policy |
| `HeuristicTamiyo` | Rule-based baseline policy |

### File Breakdown

| File | Responsibility |
|------|----------------|
| `decisions.py` | TamiyoDecision dataclass with command conversion |
| `tracker.py` | SignalTracker with plateau/delta computation |
| `heuristic.py` | TamiyoPolicy protocol, HeuristicTamiyo implementation |

### Dependencies
- **Inbound:** simic.training, simic.vectorized, simic.episodes, scripts.evaluate
- **Outbound:** leyline (signals, stages, commands), kasmina (TYPE_CHECKING)

### Key Patterns
- Policy Pattern (TamiyoPolicy protocol with HeuristicTamiyo implementation)
- Signal Aggregation (deltas, plateaus, bests)
- Blueprint Penalty System (decay-based rotation)
- Embargo System (cooldown after culling)
- Data Transfer Object (TamiyoDecision → AdaptationCommand)

---

## 5. Nissa (Telemetry Hub)

**Location:** `esper/nissa/`
**Files:** 5
**Confidence:** HIGH

### Package Responsibility
**Nissa** is the system telemetry hub and diagnostics layer for Esper. It receives carbon copies of lifecycle and training events from all domains (Kasmina seed lifecycle, Simic training progress, Tamiyo decisions) and routes them to configurable output backends (console, files, metrics systems). Additionally, it provides the `DiagnosticTracker` for rich in-training telemetry collection (gradient health, loss landscape analysis, per-class metrics, training narratives) and the `BlueprintAnalytics` aggregator for strategic dashboards.

### Key Public API

| Export | Purpose |
|--------|---------|
| `TelemetryConfig` | Configuration with profiles |
| `DiagnosticTracker` | Rich per-epoch diagnostics |
| `GradientStats` / `GradientHealth` | Gradient monitoring |
| `EpochSnapshot` | Complete training state |
| `NissaHub` | Central event router |
| `ConsoleOutput` / `FileOutput` | Output backends |
| `BlueprintAnalytics` | Blueprint performance aggregation |
| `get_hub()` / `emit()` | Global singleton access |

### File Breakdown

| File | Responsibility |
|------|----------------|
| `config.py` | TelemetryConfig with profile system |
| `tracker.py` | DiagnosticTracker, gradient hooks, narratives |
| `output.py` | OutputBackend, ConsoleOutput, FileOutput, NissaHub |
| `analytics.py` | BlueprintStats, SeedScoreboard, BlueprintAnalytics |

### Dependencies
- **Inbound:** scripts.train, simic.vectorized, simic.training
- **Outbound:** leyline (TelemetryEvent, TelemetryEventType)

### Key Patterns
- Hub-and-Spoke (NissaHub central router)
- Observer Pattern (gradient hooks)
- Strategy Pattern (pluggable output backends)
- Profile Factory Pattern (from_profile())
- Aggregation Pattern (raw events → statistics)
- Singleton Pattern (global hub)

---

## 6. Tolaria (Model Training)

**Location:** `esper/tolaria/`
**Files:** 4
**Confidence:** HIGH

### Package Responsibility
Tolaria owns the **training infrastructure for Model Alpha** - the neural network enhanced with morphogenetic seeds. It provides generic epoch-based training loops for three seed lifecycle stages (normal/womb-mode/blended), model instantiation via task specifications, and a fail-safe watchdog for catastrophic failure detection with automatic rollback and RL punishment signaling.

### Key Public API

| Export | Purpose |
|--------|---------|
| `create_model()` | Model factory via TaskSpec |
| `train_epoch_normal()` | Standard training (no seeds) |
| `train_epoch_womb_mode()` | STE isolation + dual optimization |
| `train_epoch_blended()` | Joint host+seed training |
| `validate_and_get_metrics()` | Comprehensive evaluation |
| `TolariaGovernor` | Catastrophic failure watchdog |
| `GovernorReport` | Rollback event report |

### File Breakdown

| File | Responsibility |
|------|----------------|
| `environment.py` | create_model() factory |
| `trainer.py` | Three train_epoch_* functions + validation |
| `governor.py` | TolariaGovernor with 6σ anomaly detection |

### Dependencies
- **Inbound:** simic.vectorized, scripts.evaluate
- **Outbound:** runtime (TaskSpec)

### Key Patterns
- Facade Pattern (__init__.py presents unified interface)
- Factory Pattern (create_model via TaskSpec)
- Strategy Pattern (train_epoch_* for different stages)
- Watchdog/Monitor Pattern (TolariaGovernor)
- Checkpoint/Rollback Pattern (LKG snapshot)
- 6-sigma Anomaly Detection + Lobotomy Detection

---

## 7. Runtime (Task Presets)

**Location:** `esper/runtime/`
**Files:** 2
**Confidence:** HIGH

### Package Responsibility
Provides task-specific wiring presets and factories for training environments. Acts as a configuration bridge between domain subsystems by bundling task-specific setup into reusable `TaskSpec` objects.

### Key Public API

| Export | Purpose |
|--------|---------|
| `TaskSpec` | Complete task specification |
| `get_task_spec()` | Factory for preset tasks |

### Dependencies
- **Inbound:** tolaria.environment, simic.training, simic.vectorized, scripts.evaluate
- **Outbound:** kasmina.host, leyline.actions, simic.features, simic.rewards, utils.data

---

## 8. Utils (Shared Utilities)

**Location:** `esper/utils/`
**Files:** 2
**Confidence:** HIGH

### Package Responsibility
Provides shared dataset loading utilities decoupled from domain logic. Supports CIFAR-10 and TinyStories with configurable DataLoaders and mock fallbacks for testing.

### Key Public API

| Export | Purpose |
|--------|---------|
| `load_cifar10()` | CIFAR-10 dataloader factory |
| `load_tinystories()` | TinyStories LM dataloader factory |

### Dependencies
- **Inbound:** runtime.tasks, simic.training
- **Outbound:** PyTorch, torchvision, transformers (optional), datasets (optional)

---

## 9. Scripts (Entry Points)

**Location:** `esper/scripts/`
**Files:** 3
**Confidence:** MEDIUM-HIGH

### Package Responsibility
Provides CLI entry points for training and evaluating RL agents. Offers two main training modes (heuristic and PPO with optional vectorization) and a diagnostic evaluation tool.

### Key Entry Points

| Script | Purpose |
|--------|---------|
| `train.py heuristic` | Train with heuristic policy |
| `train.py ppo` | Train PPO agent |
| `train.py ppo --vectorized` | Multi-GPU PPO training |
| `evaluate.py` | Diagnostic policy evaluation |

### Dependencies
- **Inbound:** User CLI only
- **Outbound:** nissa, simic.training, simic.vectorized, runtime, tolaria, tamiyo

---

## Dependency Graph Summary

```
┌─────────────────────────────────────────────────────────┐
│                      scripts (CLI)                       │
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
      ┌────────┴────────┐     ┌──────────┐
      │      nissa      │     │  runtime │
      │   (telemetry)   │     │  utils   │
      └─────────────────┘     └──────────┘
```

---

## Confidence Summary

| Package | Confidence | Reasoning |
|---------|------------|-----------|
| leyline | HIGH | Complete analysis, minimal dependencies, clear contracts |
| kasmina | HIGH | Well-documented lifecycle mechanics, explicit gates |
| simic | HIGH | Comprehensive RL infrastructure, clear hot paths |
| tamiyo | HIGH | Clean policy pattern, straightforward signal flow |
| nissa | HIGH | Standard observability patterns, well-separated |
| tolaria | HIGH | Clear training strategies, well-defined watchdog |
| runtime | HIGH | Simple task wiring, explicit factory pattern |
| utils | HIGH | Focused data loading utilities |
| scripts | MEDIUM-HIGH | CLI interface, some domain logic in evaluate.py |
