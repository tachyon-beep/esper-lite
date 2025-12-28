# Subsystem Catalog

**Analysis Date:** 2025-12-28
**Total LOC:** ~47,346 Python + 8,722 Vue/TS

## Overview

Esper comprises 7 active domains ("organs") plus support modules. Each subsystem has been analyzed in depth by parallel exploration agents.

---

## 1. Leyline (DNA/Genome)

**Location:** `src/esper/leyline/` | **LOC:** 3,735

**Responsibility:** Foundational contracts layer defining all shared data contracts, enums, constants, schemas, and telemetry types that flow between the seven domain organs.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 750 | Central constants registry (PPO, lifecycle, gates, rewards, governor) |
| `telemetry.py` | 1,414 | 30+ event types, 18 typed payload classes, SeedTelemetry feature vector |
| `factored_actions.py` | 495 | 8-head factored action space (slot, blueprint, style, tempo, alpha_*) |
| `stages.py` | 97 | SeedStage enum, VALID_TRANSITIONS state machine |
| `slot_config.py` | 130 | SlotConfig for dynamic action spaces |

### Public APIs

- **Lifecycle:** `SeedStage`, `VALID_TRANSITIONS`, `is_valid_transition()`, gate enums
- **Actions:** `FactoredAction`, `BlueprintAction`, `LifecycleOp`, ACTION_HEAD_SPECS
- **Telemetry:** `TelemetryEvent`, 18 typed payload classes, `SeedTelemetry`
- **Constants:** All DEFAULT_* training parameters, STAGE_COLORS, thresholds

### Dependencies

- **Inbound:** ALL other domains import from Leyline
- **Outbound:** None (foundation layer)

### Confidence: HIGH (92%)

---

## 2. Kasmina (Stem Cells)

**Location:** `src/esper/kasmina/` | **LOC:** 5,174

**Responsibility:** Manages the lifecycle of modular seed components through germination, training, blending, and fossilization stages.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `slot.py` | 2,610 | SeedSlot lifecycle engine, quality gates (G0-G5), alpha scheduling |
| `host.py` | 769 | CNNHost, TransformerHost, MorphogeneticModel |
| `blending.py` | 241 | BlendAlgorithm (GatedBlend), alpha composition |
| `alpha_controller.py` | 187 | Time-based alpha scheduling (LINEAR, COSINE, SIGMOID) |
| `blueprints/registry.py` | 128 | Blueprint decorator pattern and factory |
| `blueprints/cnn.py` | 295 | CNN seed templates (norm, attention, bottleneck) |
| `blueprints/transformer.py` | 345 | Transformer seeds (lora, attention, mlp) |

### Public APIs

- **SeedSlot:** `germinate()`, `advance_stage()`, `prune()`, `forward()`
- **MorphogeneticModel:** Multi-slot host wrapper
- **HostProtocol:** `injection_specs`, `forward_to_segment()`
- **Blend operators:** `blend_add()`, `blend_multiply()`, `blend_gate()`

### Dependencies

- **Inbound:** Simic, Tamiyo, Runtime
- **Outbound:** Leyline (contracts)

### Patterns

1. **State Machine with Quality Gates:** G0-G5 progression checks
2. **Composition Operators:** ADD/MULTIPLY/GATE blend strategies
3. **Protocol-Oriented Hosts:** HostProtocol for pluggable architectures
4. **DDP Symmetry:** `_sync_gate_decision()` for distributed training

### Confidence: HIGH (85%)

---

## 3. Simic (Evolution)

**Location:** `src/esper/simic/` | **LOC:** 13,352

**Responsibility:** RL infrastructure (PPO) enabling adaptation through selection pressure on seed lifecycle decisions.

### Key Components

| Subdirectory | LOC | Purpose |
|--------------|-----|---------|
| `agent/` | 2,695 | PPOAgent, TamiyoRolloutBuffer, GAE computation |
| `training/` | 2,535 | Vectorized PPO training loop, config, helpers |
| `rewards/` | 1,895 | Contribution/loss rewards, PBRS, reward engineering |
| `telemetry/` | 2,227 | Gradient collectors, anomaly detection, emitters |
| `attribution/` | 716 | Counterfactual engine (Shapley values, factorial) |
| `control/` | 244 | Running mean normalization |

### Public APIs

- **PPOAgent:** `update()`, `save()`, `load()`
- **Training:** `train_ppo_vectorized()`, `TrainingConfig`
- **Rewards:** `compute_contribution_reward()`, `RewardMode` enum
- **Telemetry:** `AnomalyDetector`, gradient emitters

### Dependencies

- **Inbound:** Scripts/CLI
- **Outbound:** Leyline, Tamiyo, Kasmina (via protocols), Tolaria, Nissa, Karn

### Patterns

1. **Factored Policy:** 8 independent action heads with per-head credit
2. **Per-Env GAE:** Prevents cross-environment contamination (P0 bug fix)
3. **Causal Masking:** Only active heads receive advantage signals
4. **PBRS:** Stage potentials for shaped rewards (Ng et al., 1999)
5. **Multi-GPU Vectorization:** SharedBatchIterator for efficiency

### Complexity Hotspots

- `vectorized.py` (3,404 LOC): Large training orchestration function
- LSTM hidden state management: Fragile permutation logic
- Reward mode combinations: Multiple overlapping flags

### Confidence: HIGH (Core), MEDIUM (Vectorized Training)

---

## 4. Tamiyo (Brain/Cortex)

**Location:** `src/esper/tamiyo/` | **LOC:** 3,811

**Responsibility:** Strategic decision-making for seed lifecycle via heuristic rules or neural policies.

### Key Components

| File/Dir | LOC | Purpose |
|----------|-----|---------|
| `heuristic.py` | 357 | Rule-based HeuristicTamiyo controller |
| `tracker.py` | 347 | SignalTracker for training metric aggregation |
| `decisions.py` | 53 | TamiyoDecision dataclass |
| `policy/` | ~2,000 | PolicyBundle protocol, LSTM/heuristic bundles |
| `networks/` | 635 | FactoredRecurrentActorCritic neural architecture |

### Public APIs

- **HeuristicTamiyo:** `decide(signals, active_seeds) → TamiyoDecision`
- **PolicyBundle Protocol:** `get_action()`, `evaluate_actions()`, `forward()`
- **SignalTracker:** `update() → TrainingSignals`
- **Features:** `obs_to_multislot_features()`, `compute_action_masks()`

### Dependencies

- **Inbound:** Simic (policy training), Kasmina (execution)
- **Outbound:** Leyline (contracts), Nissa (telemetry)

### Patterns

1. **Strategy Pattern:** PolicyBundle for swappable heuristic vs neural
2. **Factory + Registry:** Hot-swappable policy implementations
3. **Hot Path Design:** Feature extraction only imports from Leyline
4. **Stabilization Latch:** Gates germination during explosive growth
5. **Ransomware Detection:** Prunes high-counterfactual, negative-improvement seeds

### Confidence: HIGH

---

## 5. Karn (Memory)

**Location:** `src/esper/karn/` | **LOC:** 8,341 Python + 8,722 Vue/TS

**Responsibility:** Research telemetry system with TUI (Sanctum), web dashboard (Overwatch), and SQL analytics (MCP server).

### Key Components

| Subdirectory | LOC | Purpose |
|--------------|-----|---------|
| Root | 2,841 | TelemetryStore, collector, health, triggers, serialization |
| `mcp/` | 294 | DuckDB SQL interface for Claude Code |
| `sanctum/` | 1,591 | Textual TUI backend and aggregator |
| `sanctum/widgets/` | 2,000+ | 16 TUI widget components |
| `overwatch/` | 361 | Vue 3 web dashboard backend |
| `overwatch/web/` | 8,722 | 12 Vue components + composables |

### Public APIs

- **KarnCollector:** Central event ingestion
- **TelemetryStore:** 3-tier model (Episode, Epoch, DenseTrace)
- **SanctumBackend:** TUI telemetry consumer
- **OverwatchBackend:** WebSocket dashboard server
- **MCP Server:** SQL queries over telemetry

### Dependencies

- **Inbound:** Simic, Tolaria, Kasmina, Tamiyo (event producers)
- **Outbound:** Leyline (event types), DuckDB, Textual, Vue 3

### Patterns

1. **Three-Tier Fidelity:** Context → Epochs → Dense Traces
2. **Aggregator Pattern:** Raw events → rich SanctumSnapshot
3. **Protocol Inversion:** TelemetryEventLike decouples from Leyline
4. **MCP Bridge:** JSONL → DuckDB → SQL for Claude Code

### Confidence: HIGH

---

## 6. Nissa (Sensory Organs)

**Location:** `src/esper/nissa/` | **LOC:** 1,969

**Responsibility:** Central telemetry hub routing events to configurable backends.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `output.py` | 679 | NissaHub, ConsoleOutput, FileOutput, DirectoryOutput |
| `tracker.py` | 547 | DiagnosticTracker, EpochSnapshot, GradientStats |
| `analytics.py` | 424 | BlueprintAnalytics, SeedScoreboard |
| `config.py` | 238 | TelemetryConfig profiles |

### Public APIs

- **NissaHub:** `add_backend()`, `emit()`, `flush()`, `close()`
- **Global:** `get_hub()`, `reset_hub()`, `emit()`
- **DiagnosticTracker:** Per-model gradient/loss telemetry

### Dependencies

- **Inbound:** All domains emit via Nissa
- **Outbound:** Leyline (event contracts)

### Patterns

1. **Pub-Sub:** Multiple backends receive all events
2. **Async Worker:** Queue-based event processing
3. **Typed Payloads:** Discriminated union for polymorphic dispatch

### Concerns

- Dead event types (ISOLATION_VIOLATION, GOVERNOR_PANIC unused)
- Queue overflow drops events under extreme load

### Confidence: HIGH

---

## 7. Tolaria (Metabolism)

**Location:** `src/esper/tolaria/` | **LOC:** 462

**Responsibility:** Training fail-safe watchdog (Governor) providing anomaly detection and emergency rollback.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `governor.py` | 371 | TolariaGovernor anomaly detector and rollback executor |
| `environment.py` | 73 | Model factory with device validation |

### Public APIs

- **TolariaGovernor:** `check_vital_signs()`, `execute_rollback()`, `snapshot()`
- **create_model():** MorphogeneticModel factory

### Dependencies

- **Inbound:** Simic (training loop calls Governor)
- **Outbound:** Leyline, Nissa (rollback events), Runtime

### Patterns

1. **Guardian/Super-Ego:** Safety net for catastrophic failures
2. **RAM Checkpointing:** LKG state in CPU memory for instant rollback
3. **Filtering Experimental State:** Only fossilized seeds in snapshot
4. **Hysteresis:** Consecutive panics threshold before action

### Critical Contract

**Optimizer state must be cleared after rollback** (momentum buffers survive load_state_dict)

### Confidence: HIGH

---

## 8. Runtime (Task Configuration)

**Location:** `src/esper/runtime/` | **LOC:** 298

**Responsibility:** Task preset factories for CIFAR-10, TinyStories configurations.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `tasks.py` | 294 | TaskSpec dataclass, preset factories |

### Public APIs

- **TaskSpec:** Model factory, dataloader factory, task config
- **get_task_spec(name):** Load preset by name

### Presets

- `cifar10`: Weak CNN (8 channels) for seed demonstration
- `cifar10_deep`: 5-block CNN for GPU saturation
- `cifar10_blind`: 1x1 convolutions only (spatial blindness)
- `tinystories`: Transformer for language modeling

### Confidence: HIGH

---

## 9. Scripts (CLI)

**Location:** `src/esper/scripts/` | **LOC:** 680

**Responsibility:** CLI entry points for training.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `train.py` | 680 | argparse CLI, telemetry wiring, TUI/dashboard setup |

### Commands

```bash
# PPO training
python -m esper.scripts.train ppo --preset cifar10

# Heuristic baseline
python -m esper.scripts.train heuristic --episodes 1
```

### CLI Flags

- `--preset`: Hyperparameter preset selection
- `--sanctum`: Textual TUI mode
- `--overwatch`: Vue dashboard mode
- `--telemetry-*`: Telemetry configuration

### Confidence: HIGH

---

## 10. Utils (Shared Utilities)

**Location:** `src/esper/utils/` | **LOC:** 802

**Responsibility:** Data loading and loss computation utilities.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `data.py` | 661 | SharedBatchIterator, GPU preloading, TinyStories |
| `loss.py` | 126 | Task-agnostic loss computation |

### Public APIs

- **SharedBatchIterator:** Single DataLoader for N environments
- **SharedGPUBatchIterator:** GPU-resident data (8x faster for CIFAR-10)
- **compute_task_loss_with_metrics():** Loss + accuracy

### Confidence: HIGH

---

## Dependency Matrix

```
             Leyline  Kasmina  Tamiyo  Simic  Nissa  Tolaria  Karn  Runtime  Scripts  Utils
Leyline         -       -        -       -      -       -       -      -        -       -
Kasmina         ✓       -        -       -      -       -       -      -        -       -
Tamiyo          ✓       -        -       -      ✓       -       -      -        -       -
Simic           ✓       ✓*       ✓       -      ✓       ✓       ✓      -        -       ✓
Nissa           ✓       -        -       -      -       -       -      -        -       -
Tolaria         ✓       -        -       -      ✓       -       -      ✓        -       -
Karn            ✓       -        -       -      -       -       -      -        -       -
Runtime         ✓       ✓        ✓       -      -       -       -      -        -       ✓
Scripts         ✓       -        -       ✓      ✓       -       ✓      -        -       -
Utils           -       -        -       -      -       -       -      -        -       -

* Simic depends on Kasmina via protocol contracts (contracts.py), not direct import
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python LOC | ~47,346 |
| Total Vue/TS LOC | ~8,722 |
| Total Test LOC | ~59,411 |
| Active Domains | 7 |
| Support Modules | 3 (Runtime, Scripts, Utils) |
| High Confidence Subsystems | 10/10 |
