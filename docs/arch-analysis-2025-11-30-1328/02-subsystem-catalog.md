# Subsystem Catalog - Esper Morphogenetic Neural Networks

## Overview

This catalog documents the 6 core subsystems plus 2 support modules of Esper. Each subsystem follows a consistent structure: location, responsibility, key components, dependencies, patterns, and confidence level.

---

## 1. Leyline (Nervous System)

### Location
`src/esper/leyline/`

### Responsibility
Defines shared data contracts that flow between all Esper components. Acts as the "nervous system" providing type-safe communication protocols.

### Key Components

| File | Purpose |
|------|---------|
| `stages.py` | SeedStage IntEnum (DORMANT→FOSSILIZED), VALID_TRANSITIONS, CommandType, RiskLevel |
| `actions.py` | Action enum (WAIT, GERMINATE_*, ADVANCE, CULL) - shared action space |
| `schemas.py` | AdaptationCommand, GateLevel, GateResult, BlueprintSpec |
| `signals.py` | TrainingSignals, FastTrainingSignals (NamedTuple for hot path), TensorSchema |
| `telemetry.py` | TelemetryEventType, TelemetryEvent, SeedTelemetry |
| `reports.py` | SeedMetrics, SeedStateReport, FieldReport |
| `blueprints.py` | Blueprint constants (BLUEPRINT_CONV_ENHANCE, etc.) |

### Dependencies
- **Inbound**: All other subsystems import from Leyline
- **Outbound**: None (foundational, no dependencies)

### Patterns Observed
- **Data Transfer Objects**: Dataclasses and NamedTuples for contracts
- **Protocol-first design**: Types defined before implementations
- **Two-tier signals**: Full TrainingSignals + lightweight FastTrainingSignals
- **IntEnum for stages**: Enables efficient integer comparisons in hot path

### Confidence: HIGH

---

## 2. Kasmina (Body)

### Location
`src/esper/kasmina/`

### Responsibility
Neural network model mechanics: seed creation, lifecycle management, gradient isolation, and grafting. The "body" that physically implements morphogenetic growth.

### Key Components

| File | Purpose |
|------|---------|
| `host.py` | HostCNN (base network), MorphogeneticModel (injectable model) |
| `slot.py` | SeedSlot (lifecycle manager), SeedState, SeedMetrics, QualityGates |
| `blueprints.py` | ConvEnhanceSeed, AttentionSeed, NormSeed, DepthwiseSeed, BlueprintCatalog |
| `isolation.py` | AlphaSchedule, blend_with_isolation, GradientIsolationMonitor |

### Dependencies
- **Inbound**: Tamiyo, Tolaria, Simic
- **Outbound**: Leyline (stages, schemas, telemetry)

### Patterns Observed
- **Slot pattern**: SeedSlot manages full lifecycle in isolation
- **Quality gates**: G0-G5 gates validate stage transitions
- **Factory pattern**: BlueprintCatalog.create_seed()
- **State machine**: SeedState tracks transitions with history
- **Gradient isolation**: detach() for isolated training

### Key Abstractions

```python
class SeedSlot:
    """Single injection point where seeds can be attached."""
    def germinate(blueprint_id, seed_id) -> SeedState
    def advance_stage() -> GateResult
    def cull(reason) -> None
    def forward(host_features) -> Tensor  # Alpha blending

class MorphogeneticModel(nn.Module):
    """Host network with seed slot for morphogenetic growth."""
    def germinate_seed(blueprint_id, seed_id) -> None
    def cull_seed() -> None
    def forward(x) -> Tensor
```

### Confidence: HIGH

---

## 3. Tamiyo (Brain)

### Location
`src/esper/tamiyo/`

### Responsibility
Strategic decision-making for seed lifecycle. The "brain" that decides when to germinate, advance, or cull seeds based on training signals.

### Key Components

| File | Purpose |
|------|---------|
| `heuristic.py` | HeuristicTamiyo (rule-based baseline), HeuristicPolicyConfig |
| `decisions.py` | TamiyoDecision (action + reason + confidence) |
| `tracker.py` | SignalTracker (maintains training signal state) |

### Dependencies
- **Inbound**: Simic (RL learns to improve Tamiyo), Tolaria
- **Outbound**: Leyline (Action, SeedStage, TrainingSignals), Kasmina (SeedState - TYPE_CHECKING only)

### Patterns Observed
- **Policy pattern**: TamiyoPolicy Protocol for interchangeable policies
- **Strategy pattern**: HeuristicTamiyo implements rule-based strategy
- **Anti-thrashing**: embargo_epochs_after_cull prevents rapid germinate/cull cycles
- **Blueprint penalty**: Tracks failed blueprints to avoid retry

### Decision Flow

```
TrainingSignals → SignalTracker → HeuristicTamiyo.decide() → TamiyoDecision
                                         │
                                         ├── No seed? → Germinate (plateau detection)
                                         ├── Training? → Evaluate improvement
                                         ├── Blending? → Check total improvement
                                         └── Return WAIT/ADVANCE/CULL
```

### Confidence: HIGH

---

## 4. Tolaria (Hands)

### Location
`src/esper/tolaria/`

### Responsibility
PyTorch training loop execution. The "hands" that perform the actual gradient updates on models.

### Key Components

| File | Purpose |
|------|---------|
| `trainer.py` | train_epoch_normal, train_epoch_seed_isolated, train_epoch_blended, validate_and_get_metrics |
| `environment.py` | create_model factory function |

### Dependencies
- **Inbound**: Simic (uses for RL environment), Scripts
- **Outbound**: Kasmina (MorphogeneticModel), Leyline (SeedStage)

### Patterns Observed
- **Function-based API**: No classes, just pure training functions
- **Three training modes**: Normal, seed-isolated, blended
- **Quick validation**: Samples first 10 batches for train metrics

### Training Modes

| Mode | Description | Host Params | Seed Params |
|------|-------------|-------------|-------------|
| Normal | Standard training | Updated | N/A |
| Seed-Isolated | TRAINING stage | Frozen (grad flows, no step) | Updated |
| Blended | BLENDING stage | Updated | Updated |

### Confidence: HIGH

---

## 5. Simic (Gym)

### Location
`src/esper/simic/`

### Responsibility
RL infrastructure for training Tamiyo. The "gym" where the strategic controller learns to optimize seed lifecycle decisions.

### Key Components

| File | Purpose |
|------|---------|
| `ppo.py` | PPOAgent (on-policy RL), ActorCritic network integration |
| `iql.py` | IQL agent (offline RL from logged data) |
| `vectorized.py` | train_ppo_vectorized (multi-GPU, CUDA streams) |
| `networks.py` | ActorCritic, PolicyNetwork, QNetwork, VNetwork |
| `rewards.py` | RewardConfig, compute_shaped_reward, PBRS functions |
| `features.py` | obs_to_base_features (HOT PATH - 27 dims) |
| `buffers.py` | RolloutBuffer, ReplayBuffer |
| `normalization.py` | RunningMeanStd (GPU-native observation normalization) |
| `episodes.py` | TrainingSnapshot, Episode, EpisodeCollector |
| `training.py` | train_ppo, train_iql (non-vectorized) |
| `comparison.py` | live_comparison, head_to_head_comparison |
| `gradient_collector.py` | collect_seed_gradients (telemetry) |

### Dependencies
- **Inbound**: Scripts (entry points)
- **Outbound**: Leyline (Action, SeedStage, signals), Kasmina (for model), Tamiyo (SignalTracker), Tolaria (create_model), Utils

### Patterns Observed
- **Vectorized training**: Inverted control flow (batch-first iteration)
- **CUDA streams**: Async GPU execution for parallel environments
- **Observation normalization**: RunningMeanStd with GPU-native stats
- **Potential-based reward shaping**: PBRS preserves optimal policy
- **Two-tier features**: 27 base + 10 telemetry (optional)

### Vectorized Architecture

```
DataLoader Batch ────────────────────────────────────────────────
     │
     ├──► Env 0 (CUDA Stream 0) ─┬─► [train batch] ──┐
     ├──► Env 1 (CUDA Stream 1) ─┤                    │
     ├──► Env 2 (CUDA Stream 2) ─┤      ASYNC         │
     └──► Env 3 (CUDA Stream 3) ─┘                    │
                                                      ▼
                                          [Sync All Streams]
                                                      │
                                          [Batch Policy Inference]
                                                      │
                                          [PPO Update (single)]
```

### Reward Function

```python
reward = acc_delta * weight  # Base: accuracy improvement
       + stage_bonus         # TRAINING/BLENDING/FOSSILIZED bonuses
       + action_shaping      # Action-specific guidance
       + terminal_bonus      # Final accuracy at episode end
```

### Confidence: HIGH

---

## 6. Nissa (Senses)

### Location
`src/esper/nissa/`

### Responsibility
Telemetry hub for observability. Receives "carbon copies" of events from all domains and routes to output backends.

### Key Components

| File | Purpose |
|------|---------|
| `config.py` | TelemetryConfig with profile management (diagnostic, minimal, etc.) |
| `tracker.py` | DiagnosticTracker, GradientStats, GradientHealth, EpochSnapshot |
| `output.py` | NissaHub, OutputBackend, ConsoleOutput, FileOutput |

### Dependencies
- **Inbound**: Simic (optional telemetry), Tolaria (optional)
- **Outbound**: Leyline (TelemetryEvent - loose coupling)

### Patterns Observed
- **Hub and spoke**: NissaHub routes events to multiple backends
- **Profile-based config**: Diagnostic vs minimal profiles
- **Pluggable backends**: OutputBackend protocol for extensibility
- **Loose coupling**: Can be completely bypassed (fast_mode in SeedSlot)

### Telemetry Flow

```
All Domains ──[TelemetryEvent]──► NissaHub ───┬──► ConsoleOutput
                                              ├──► FileOutput (JSONL)
                                              └──► Custom backends
```

### Confidence: HIGH

---

## 7. Utils (Support)

### Location
`src/esper/utils/`

### Responsibility
Shared utilities, primarily CIFAR-10 data loading.

### Key Components

| File | Purpose |
|------|---------|
| `data.py` | load_cifar10 function |
| `__init__.py` | Re-exports |

### Dependencies
- **Inbound**: Simic, Tolaria
- **Outbound**: None (leaf module)

### Confidence: HIGH

---

## 8. Scripts (Entry Points)

### Location
`src/esper/scripts/`

### Responsibility
CLI entry points for training and evaluation.

### Key Components

| File | Purpose |
|------|---------|
| `train.py` | CLI for PPO/IQL training with argparse |
| `evaluate.py` | Evaluation scripts |

### Dependencies
- **Inbound**: None (top-level)
- **Outbound**: Simic (training functions), all others indirectly

### CLI Examples

```bash
# Vectorized PPO
PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 4 --device cuda:0

# IQL from offline data
PYTHONPATH=src python -m esper.scripts.train iql --pack data/pack.json --epochs 100

# Policy comparison
PYTHONPATH=src python -m esper.scripts.train compare --model model.pt --mode head-to-head
```

### Confidence: HIGH

---

## Dependency Matrix

| Subsystem | Depends On | Depended By |
|-----------|------------|-------------|
| Leyline | - | Kasmina, Tamiyo, Tolaria, Simic, Nissa |
| Kasmina | Leyline | Tamiyo, Tolaria, Simic |
| Tamiyo | Leyline, Kasmina* | Simic |
| Tolaria | Leyline, Kasmina | Simic, Scripts |
| Simic | Leyline, Kasmina, Tamiyo, Tolaria, Utils | Scripts |
| Nissa | Leyline | Simic*, Tolaria* |
| Utils | - | Simic |
| Scripts | Simic | - |

*TYPE_CHECKING only or optional

---

## Cross-Cutting Concerns

### Hot Path Optimization
- `FastTrainingSignals` (NamedTuple) in Leyline
- `obs_to_base_features` in Simic/features.py (no heavy imports)
- `RunningMeanStd` GPU-native operations
- CUDA streams in vectorized training

### Type Safety
- Protocol classes (TamiyoPolicy, BlueprintProtocol, OutputBackend)
- TYPE_CHECKING imports to avoid circular dependencies
- Strict IntEnum for stages and actions

### Performance Patterns
- `@dataclass(slots=True)` for hot path classes
- Singleton config (`_DEFAULT_CONFIG` in rewards.py)
- Lazy imports for heavy modules in Simic/__init__.py
