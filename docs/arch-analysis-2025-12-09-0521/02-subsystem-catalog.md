# Subsystem Catalog: esper-lite

## Overview

This catalog documents all 9 subsystems in the esper-lite codebase with their responsibilities, key components, dependencies, and architectural patterns.

---

## 1. Kasmina (Body/Model)

### Location
`/home/john/esper-lite/src/esper/kasmina/`

### Files
| File | Size | Description |
|------|------|-------------|
| `__init__.py` | 0.7 KB | Module exports |
| `protocol.py` | 1.1 KB | HostProtocol structural type |
| `host.py` | 13.9 KB | CNNHost, TransformerHost, MorphogeneticModel |
| `slot.py` | 45 KB | SeedSlot, SeedState, QualityGates |
| `isolation.py` | 5.0 KB | Gradient isolation, alpha blending |
| `blueprints/registry.py` | 3.8 KB | Plugin system for seed blueprints |
| `blueprints/cnn.py` | 6.1 KB | CNN seed modules |
| `blueprints/transformer.py` | 7.2 KB | Transformer seed modules |

### Responsibility
Implements the complete seed lifecycle (germination → training → blending → fossilization) with quality gates, gradient isolation, and blending mechanisms. Seeds are pluggable neural modules that attach to host networks at defined injection points.

### Key Components

| Component | Role |
|-----------|------|
| **SeedSlot** | Core lifecycle manager for a single seed module |
| **SeedState** | Complete state container (id, stage, metrics, alpha) |
| **QualityGates** | Six gate levels (G0-G5) controlling stage transitions |
| **MorphogeneticModel** | Wrapper around host network with SeedSlot |
| **HostProtocol** | Structural typing interface for graftable hosts |
| **CNNHost** | Convolutional host with injection points |
| **TransformerHost** | GPT-style decoder with injection points |
| **BlueprintRegistry** | Plugin system for extensible seed implementations |
| **AlphaSchedule** | Sigmoid-based blending schedule |

### Dependencies

**Inbound:**
- esper/__init__.py (re-exports)
- esper.runtime.tasks
- esper.tamiyo (TYPE_CHECKING)
- esper.leyline.actions (lazy import)

**Outbound:**
- esper.leyline (SeedStage, transitions, gates, telemetry)
- esper.simic.features (TYPE_CHECKING)
- torch, torch.nn

### Patterns
- Plugin Registry (BlueprintRegistry)
- Structural Typing (HostProtocol)
- State Machine (SeedState)
- Quality Gates/Staged Rollout
- Lazy Imports (circular dependency avoidance)

### Confidence: HIGH

---

## 2. Leyline (Nervous System/Signals)

### Location
`/home/john/esper-lite/src/esper/leyline/`

### Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API re-exports |
| `schemas.py` | Command and blueprint specifications |
| `actions.py` | Action space definitions |
| `signals.py` | Training state observations |
| `stages.py` | Seed lifecycle stages and transitions |
| `reports.py` | Structured reporting contracts |
| `telemetry.py` | Telemetry event definitions |

### Responsibility
Data contract and nervous system layer defining all shared types, enums, and data structures that flow between subsystems. Contains NO business logic - only data structures and helper functions.

### Key Components

| Component | Role |
|-----------|------|
| **SeedStage** | Lifecycle enum with 10 stages |
| **AdaptationCommand** | Tamiyo → Kasmina communication contract |
| **FastTrainingSignals** | Lightweight NamedTuple for PPO hot path (35 features) |
| **TrainingSignals** | Rich observation dataclass |
| **TelemetryEvent** | Observability contract with event types |
| **TensorSchema** | IntEnum for feature indexing |

### Dependencies

**Inbound:**
- ALL other esper modules import from leyline

**Outbound:**
- esper.kasmina.blueprints (LAZY IMPORT only)
- Standard library only

### Patterns
- Lazy Import Pattern (BlueprintRegistry)
- Immutable Data Contracts (frozen dataclasses)
- Dual-Representation (FastTrainingSignals vs TrainingSignals)
- Enum-Based Indexing (TensorSchema)
- State Machine Documentation

### Confidence: HIGH

---

## 3. Tamiyo (Brain/Gardener)

### Location
`/home/john/esper-lite/src/esper/tamiyo/`

### Files
| File | Size | Description |
|------|------|-------------|
| `__init__.py` | - | Module exports |
| `decisions.py` | 2.7 KB | TamiyoDecision dataclass |
| `heuristic.py` | 11.6 KB | HeuristicTamiyo policy |
| `tracker.py` | 8.9 KB | SignalTracker for metrics |

### Responsibility
Observes training signals and makes strategic, heuristic-based decisions about seed lifecycle management. Determines when to germinate, fossilize, cull, or wait.

### Key Components

| Component | Role |
|-----------|------|
| **SignalTracker** | Running statistics, plateau detection, stabilization gating |
| **HeuristicTamiyo** | Rule-based policy with blueprint rotation |
| **TamiyoDecision** | Output type bridging to AdaptationCommand |
| **HeuristicPolicyConfig** | Configuration for all thresholds |
| **TamiyoPolicy** | Protocol for pluggable policies |

### Dependencies

**Inbound:**
- esper.simic.training
- esper.simic.vectorized
- esper.scripts.evaluate

**Outbound:**
- esper.leyline (contracts, actions)
- esper.nissa (telemetry hub)
- esper.kasmina (TYPE_CHECKING)

### Patterns
- Protocol Pattern (TamiyoPolicy)
- Dataclass Configuration
- State Machine Pattern
- Exponential Decay (blueprint penalties)
- Embargo/Cooldown Pattern

### Confidence: HIGH

---

## 4. Tolaria (Hands/Tools)

### Location
`/home/john/esper-lite/src/esper/tolaria/`

### Files
| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 45 | Public API exports |
| `environment.py` | 27 | Model factory |
| `trainer.py` | 352 | Core training loops |
| `governor.py` | 239 | Fail-safe watchdog |

**Total:** 663 lines

### Responsibility
Execution engine that owns PyTorch training loops for morphogenetic models. Provides training step implementations for different seed state modes and validation mechanisms.

### Key Components

| Component | Role |
|-----------|------|
| **train_epoch_normal()** | Standard training without seed |
| **train_epoch_incubator_mode()** | STE training (seed isolation) |
| **train_epoch_blended()** | Joint host+seed training |
| **TolariaGovernor** | Multi-layered fail-safe watchdog |
| **validate_with_attribution()** | Counterfactual validation |

### Dependencies

**Inbound:**
- esper.simic.vectorized
- esper.simic.training
- esper.scripts.evaluate

**Outbound:**
- esper.runtime (TaskSpec)
- torch, torch.nn
- Implicit: esper.kasmina (seed_slot attribute)

### Patterns
- Straight-Through Estimator (STE)
- Fail-Safe Watchdog
- Counterfactual Validation
- Task Abstraction
- GPU Memory Optimization

### Confidence: HIGH

---

## 5. Simic (Gym/Simulator)

### Location
`/home/john/esper-lite/src/esper/simic/`

### Files (21 modules)

**Core RL Training:**
- `ppo.py` - PPO agent with masked actions, LSTM support
- `training.py` - Single-environment training loops
- `vectorized.py` - Multi-environment parallel PPO

**Neural Networks:**
- `networks.py` - Actor-Critic, Q/V networks
- `config.py` - Hyperparameter configuration

**Data Structures:**
- `episodes.py` - Episode containers
- `buffers.py` - RolloutBuffer with GAE
- `prioritized_buffer.py` - PER buffer

**Features & Processing:**
- `features.py` - Feature extraction (HOT PATH)
- `normalization.py` - Running mean/std normalization

**Rewards:**
- `rewards.py` - Shaped reward computation
- `curriculum.py` - UCB1-based blueprint curriculum

**Telemetry (7 modules):**
- `telemetry_config.py`, `ppo_telemetry.py`, `reward_telemetry.py`
- `memory_telemetry.py`, `gradient_collector.py`, `debug_telemetry.py`
- `anomaly_detector.py`

### Responsibility
Complete RL infrastructure implementing PPO agents that learn to control seed lifecycle management by observing training signals and optimizing accuracy while managing computational cost.

### Key Components

| Component | Role |
|-----------|------|
| **PPOAgent** | Actor-Critic with masked actions, entropy annealing |
| **train_ppo_vectorized()** | Multi-env parallel training with CUDA streams |
| **obs_to_base_features()** | 35-dim feature extraction (HOT PATH) |
| **compute_shaped_reward()** | PBRS reward with lifecycle bonuses |
| **ActorCritic** | Shared feature extractor + actor/critic heads |
| **RecurrentActorCritic** | LSTM-based policy with LayerNorm |

### Dependencies

**Inbound:**
- esper.kasmina.slot
- esper.runtime.tasks
- esper.scripts.*

**Outbound:**
- esper.leyline (heavy)
- esper.nissa (light)
- esper.tolaria (light)
- esper.runtime (light)

### Patterns
- HOT PATH Discipline (features.py)
- Telemetry-First Design
- RL Best Practices (orthogonal init, GAE, KL stopping)
- Multi-GPU Infrastructure (CUDA streams)
- Configuration-Driven

### Confidence: HIGH

---

## 6. Nissa (Senses/Sensors)

### Location
`/home/john/esper-lite/src/esper/nissa/`

### Files (5 Python modules + 1 config)
| File | Description |
|------|-------------|
| `__init__.py` | Public API |
| `config.py` | TelemetryConfig with Pydantic |
| `tracker.py` | DiagnosticTracker |
| `output.py` | Output backends and NissaHub |
| `analytics.py` | Blueprint performance tracking |
| `profiles.yaml` | (YAML config) Predefined telemetry profiles |

### Responsibility
Centralized telemetry hub receiving events from all domains and routing to configured backends. Provides rich diagnostic tracking during training including gradient health, loss landscape analysis, and training pathology detection.

### Key Components

| Component | Role |
|-----------|------|
| **NissaHub** | Central singleton router |
| **DiagnosticTracker** | Rich telemetry collection |
| **BlueprintAnalytics** | Per-blueprint statistics |
| **TelemetryConfig** | Pydantic-validated configuration |
| **OutputBackend** | Console, File, Directory backends |

### Dependencies

**Inbound:**
- esper.scripts.train
- esper.simic.ppo
- esper.simic.training
- esper.simic.vectorized
- esper.tamiyo.tracker

**Outbound:**
- esper.leyline (TelemetryEvent contracts only)

### Patterns
- Singleton Hub Pattern
- Backend Strategy Pattern
- Event-Driven Architecture
- Pydantic Configuration
- Narrative Generation

### Confidence: HIGH

---

## 7. Scripts (CLI)

### Location
`/home/john/esper-lite/src/esper/scripts/`

### Files
| File | Size | Description |
|------|------|-------------|
| `__init__.py` | - | Empty |
| `train.py` | 8 KB | Training CLI |
| `evaluate.py` | 30 KB | Diagnostic evaluation |

### Responsibility
Command-line interface entry points for training and evaluating morphogenetic neural network agents.

### Key Components

| Component | Role |
|-----------|------|
| **train.main()** | CLI for ppo/heuristic training |
| **DiagnosticReport** | 5-dimension diagnostic analysis |
| **run_diagnostic_episode()** | Full episode simulation |

### Dependencies

**Inbound:** None (entry points)

**Outbound:**
- esper.nissa, esper.simic, esper.runtime
- esper.tolaria, esper.tamiyo, esper.leyline

### Confidence: HIGH

---

## 8. Runtime (Infrastructure)

### Location
`/home/john/esper-lite/src/esper/runtime/`

### Files
| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `tasks.py` | Task presets and factories |

### Responsibility
Task-specific configuration registry managing presets (CIFAR-10, TinyStories) with bundled model factories, dataloaders, action enums, and reward configurations.

### Key Components

| Component | Role |
|-----------|------|
| **TaskSpec** | Task configuration container |
| **get_task_spec()** | Registry lookup |
| **_cifar10_spec()** | CIFAR-10 preset |
| **_tinystories_spec()** | TinyStories preset |

### Dependencies

**Inbound:**
- esper.scripts.*, esper.simic.*, esper.tolaria

**Outbound:**
- esper.kasmina, esper.leyline, esper.simic, esper.utils

### Confidence: HIGH

---

## 9. Utils (Utilities)

### Location
`/home/john/esper-lite/src/esper/utils/`

### Files
| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `data.py` | Dataset loading utilities |

### Responsibility
Shared utility functions for dataset loading with support for mock/synthetic data for testing.

### Key Components

| Component | Role |
|-----------|------|
| **load_cifar10()** | CIFAR-10 loader with augmentation |
| **TinyStoriesDataset** | Custom dataset for causal LM |
| **load_tinystories()** | TinyStories loader |

### Dependencies

**Inbound:**
- esper.runtime.tasks
- esper.simic.training

**Outbound:**
- torch, torchvision
- HuggingFace datasets/transformers (optional)

### Confidence: MEDIUM

---

## Dependency Matrix

| Module | Leyline | Kasmina | Tamiyo | Tolaria | Simic | Nissa | Runtime | Utils |
|--------|---------|---------|--------|---------|-------|-------|---------|-------|
| **Kasmina** | imports | - | - | - | TYPE | - | - | - |
| **Tamiyo** | imports | TYPE | - | - | - | imports | - | - |
| **Tolaria** | implicit | implicit | - | - | - | - | imports | - |
| **Simic** | imports | TYPE | TYPE | imports | - | imports | imports | - |
| **Nissa** | imports | - | - | - | - | - | - | - |
| **Runtime** | imports | imports | - | - | imports | - | - | imports |
| **Scripts** | imports | - | imports | imports | imports | imports | imports | - |

Legend:
- `imports` = runtime import
- `TYPE` = TYPE_CHECKING import only
- `implicit` = structural coupling (no explicit import)
