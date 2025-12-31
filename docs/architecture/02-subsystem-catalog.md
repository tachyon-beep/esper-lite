# Esper Subsystem Catalog

**Analysis Date:** 2025-12-30
**Schema Version:** 1.0

---

## 1. Kasmina (Stem Cells)

**Location:** `src/esper/kasmina/`

**Responsibility:** Manages pluripotent seed slots that differentiate into neural modules. Implements the grafting mechanics where seeds train on host residuals and eventually integrate (fossilize) into the host network.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `protocol.py` | Host interface contract | `HostProtocol` |
| `host.py` | Backbone implementations | `CNNHost`, `TransformerHost`, `MorphogeneticModel` |
| `slot.py` | Seed lifecycle manager | `SeedSlot`, `SeedState`, `SeedMetrics`, `QualityGates` |
| `isolation.py` | Gradient safety | `GradientHealthMonitor`, `blend_with_isolation` |
| `alpha_controller.py` | Temporal scheduling | `AlphaController` |
| `blending.py` | Per-sample gating | `BlendAlgorithm`, `GatedBlend` |
| `blend_ops.py` | Composition operators | `blend_add`, `blend_multiply`, `blend_gate` |
| `blueprints/registry.py` | Plugin system | `BlueprintRegistry`, `BlueprintSpec` |
| `blueprints/cnn.py` | CNN seed modules | 8 blueprints (norm, attention, conv_*, etc.) |
| `blueprints/transformer.py` | Transformer seeds | 8 blueprints (lora, attention, mlp, etc.) |

### Dependencies

- **Inbound:** Simic (creates models), Tolaria (executes training)
- **Outbound:** Leyline (contracts, enums), nothing else

### Patterns Observed

- **Segment Routing:** Host fragments into segments; seeds attach at boundaries
- **Gradient Isolation:** Structural `detach()` at seed input prevents host→seed gradient flow
- **Quality Gates:** G0-G5 validators for stage transitions (permissive vs strict modes)
- **torch.compile Compatible:** ~6-8 graphs per slot (acceptable, warmup once/epoch)

### Concerns

- None observed. Clean separation of concerns.

**Confidence:** High - Comprehensive analysis of all files with clear contracts.

---

## 2. Leyline (DNA/Genome)

**Location:** `src/esper/leyline/`

**Responsibility:** Single source of truth for all shared type contracts, constants, enums, and telemetry schemas. Enforces separation of concerns by centralizing training behavior constants.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `stages.py` | Seed lifecycle states | `SeedStage`, transition validators |
| `factored_actions.py` | 8-dim action space | `FactoredAction`, 6 action enums |
| `actions.py` | Flat action builder | `build_action_enum()` (heuristic only) |
| `schemas.py` | Gate specifications | `GateLevel`, `GateResult`, `SeedOperation` |
| `slot_config.py` | Dynamic slot config | `SlotConfig` |
| `slot_id.py` | Canonical slot IDs | `format_slot_id()`, `parse_slot_id()` |
| `signals.py` | Training observations | `TrainingMetrics`, `TrainingSignals` |
| `types.py` | Observation contracts | `SeedObservationFields`, `SlotObservationFields` |
| `alpha.py` | Blending contracts | `AlphaMode`, `AlphaCurve`, `AlphaAlgorithm` |
| `stage_schema.py` | Stage encoding | `stage_to_one_hot()`, `NUM_STAGES` |
| `telemetry.py` | Event definitions | `TelemetryEvent`, 17 payload types, `SeedTelemetry` |
| `reports.py` | Structured reports | `SeedMetrics`, `SeedStateReport`, `FieldReport` |
| `causal_masks.py` | Action head masks | `compute_causal_masks()` |
| `__init__.py` | Public API + constants | 107 training behavior constants |

### Dependencies

- **Inbound:** ALL domains import from Leyline
- **Outbound:** stdlib only, `torch` only in `causal_masks.py`

### Patterns Observed

- **Single Source of Truth:** Constants defined once, imported everywhere
- **TYPE_CHECKING Guards:** Prevents circular imports for late-bound types
- **Schema Versioning:** `STAGE_SCHEMA_VERSION`, `LEYLINE_VERSION` for migrations
- **Hot-Path Optimization:** Lookup tables (`OP_NAMES`, `BLUEPRINT_IDS`) for RL speed

### Concerns

- **Dead Code Detected:** Several telemetry event types defined but never emitted:
  - `ISOLATION_VIOLATION` - defined, never emitted
  - `GOVERNOR_PANIC` - defined, only `GOVERNOR_ROLLBACK` used
  - `GOVERNOR_SNAPSHOT` - defined, never emitted
  - `CHECKPOINT_SAVED` - defined, never emitted
  - `PerformanceBudgets` / `DEFAULT_BUDGETS` - defined, never used

**Confidence:** High - Exhaustive analysis with clear ownership boundaries documented.

---

## 3. Simic (Evolution)

**Location:** `src/esper/simic/`

**Responsibility:** RL infrastructure implementing PPO for Tamiyo seed lifecycle control. Includes reward computation, counterfactual attribution, gradient telemetry, and vectorized multi-GPU training.

### Key Components

| Subdirectory | Purpose | Key Files |
|--------------|---------|-----------|
| `agent/` | PPO implementation | `ppo.py`, `rollout_buffer.py`, `advantages.py`, `types.py` |
| `rewards/` | Reward computation | `rewards.py`, `reward_telemetry.py` |
| `training/` | Training orchestration | `vectorized.py`, `helpers.py`, `config.py`, `parallel_env_state.py` |
| `telemetry/` | Diagnostics | `emitters.py`, `gradient_collector.py`, `anomaly_detector.py`, `lstm_health.py` |
| `attribution/` | Causal credit | `counterfactual.py`, `counterfactual_helper.py` |
| `control/` | Preprocessing | `normalization.py` |

### Dependencies

- **Inbound:** Scripts (CLI), Tolaria (governor callbacks)
- **Outbound:** Leyline (contracts), Tamiyo (policy), Kasmina (models), Nissa (telemetry), Karn (health), utils (data)

### Patterns Observed

- **Inverted Control Flow:** Batch-first iteration with SharedBatchIterator
- **Per-Environment Rollout Storage:** Fixes GAE interleaving bug (P0)
- **LSTM Recurrence Safety:** `recurrent_n_epochs=1` prevents hidden state staleness
- **Causal Masking:** Zero advantages for causally-irrelevant heads
- **Weight Decay on Critic Only:** Preserves exploration (DRL best practice)

### Concerns

- **Entropy Floor:** Uses `torch.clamp` (zero gradient when only one action valid)
- **Multi-Epoch Recurrence:** Disabled by default; could be improved with careful value caching

**Confidence:** High - Comprehensive PPO analysis with algorithm-level understanding.

---

## 4. Tamiyo (Brain/Cortex)

**Location:** `src/esper/tamiyo/`

**Responsibility:** Strategic decision-making for seed lifecycle management. Implements dual-mode control: heuristic (rule-based baseline) and neural (learned via PPO).

### Key Components

| File/Dir | Purpose | Key Exports |
|----------|---------|-------------|
| `tracker.py` | Signal observation | `SignalTracker` |
| `decisions.py` | Decision structures | `TamiyoDecision` |
| `heuristic.py` | Rule-based policy | `HeuristicTamiyo`, `HeuristicPolicyConfig` |
| `networks/factored_lstm.py` | Actor-critic network | `FactoredRecurrentActorCritic` |
| `policy/protocol.py` | Policy interface | `PolicyBundle` (runtime_checkable Protocol) |
| `policy/features.py` | Feature extraction | `obs_to_multislot_features()` |
| `policy/action_masks.py` | Action masking | `MaskedCategorical`, `compute_action_masks()` |
| `policy/factory.py` | Policy creation | `create_policy()` |
| `policy/lstm_bundle.py` | LSTM wrapper | `LSTMPolicyBundle` |

### Dependencies

- **Inbound:** Simic (policy training), Scripts (heuristic baseline)
- **Outbound:** Leyline (contracts), Nissa (telemetry hub only)

### Patterns Observed

- **Dual-Mode Control:** Heuristic for baseline, neural for optimization
- **Stabilization Gating:** Blocks germination during explosive host growth
- **Blueprint Penalty Tracking:** Penalizes pruned blueprints with decay
- **Ransomware Detection:** High counterfactual + negative improvement = prune
- **Optimistic Action Masking:** Only blocks physically impossible actions

### Concerns

- None observed. Clean decoupling from Simic via Protocol.

**Confidence:** High - Full analysis including network architecture and decision logic.

---

## 5. Karn (Memory)

**Location:** `src/esper/karn/`

**Responsibility:** Research telemetry platform with three-tiered adaptive fidelity. Provides TUI (Sanctum), web dashboard (Overwatch), and SQL query interface (MCP).

### Key Components

| File/Dir | Purpose | Key Exports |
|----------|---------|-------------|
| `collector.py` | Central event hub | `KarnCollector` |
| `store.py` | In-memory database | `TelemetryStore`, data models |
| `ingest.py` | Data coercion | `coerce_*` helpers |
| `health.py` | System monitoring | `HealthMonitor`, `VitalSignsMonitor` |
| `triggers.py` | Anomaly detection | `AnomalyDetector`, `PolicyAnomalyDetector` |
| `constants.py` | Display thresholds | `AnomalyThresholds`, `TUIThresholds` |
| `sanctum/` | Textual TUI | `SanctumApp`, `SanctumBackend`, 20+ widgets |
| `overwatch/` | Vue 3 dashboard | `OverwatchBackend`, Vue components |
| `mcp/` | SQL interface | `KarnMCPServer`, DuckDB views |

### Dependencies

- **Inbound:** Nissa (events via OutputBackend protocol)
- **Outbound:** Leyline (event types), Textual (TUI), Vue/Vite (web)

### Patterns Observed

- **Three-Tier Fidelity:** Episode context (minimal) → Epoch snapshots (standard) → Dense traces (on anomaly)
- **Thread-Safe Event Processing:** Training emits non-blocking, backends consume async
- **Dependency Inversion:** `TelemetryEventLike` protocol decouples from Leyline
- **Lazy Imports:** Textual/FastAPI optional (prevents install bloat)

### Concerns

- **Large Domain:** 17,800 LOC is significant; could consider splitting Sanctum/Overwatch into separate packages

**Confidence:** High - Comprehensive analysis including widget inventory.

---

## 6. Tolaria (Metabolism)

**Location:** `src/esper/tolaria/`

**Responsibility:** Training execution infrastructure including model factory and fail-safe governor for catastrophic failure detection/rollback.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `environment.py` | Model factory | `create_model()` |
| `governor.py` | Fail-safe watchdog | `TolariaGovernor`, `GovernorReport` |

### Dependencies

- **Inbound:** Simic (training loop), Scripts (model creation)
- **Outbound:** Leyline (events), Nissa (telemetry), Runtime (TaskSpec - lazy)

### Patterns Observed

- **Lazy Import:** TaskSpec imported at runtime to avoid circular dependency
- **LKG Checkpointing:** Last Known Good state on CPU (reduces GPU memory)
- **Anomaly Filtering:** Anomalous losses NOT added to statistics (prevents contamination)
- **Seed Filtering:** Non-fossilized seeds excluded from snapshots

### Concerns

- None observed. Small, focused module.

**Confidence:** High - Complete analysis of both files.

---

## 7. Nissa (Sensory Organs)

**Location:** `src/esper/nissa/`

**Responsibility:** Telemetry hub that routes events to multiple output backends. Provides diagnostic tracking with profile-based configuration.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `analytics.py` | Blueprint metrics | `BlueprintAnalytics`, `BlueprintStats`, `SeedScoreboard` |
| `config.py` | Telemetry config | `TelemetryConfig`, `GradientConfig` |
| `output.py` | Event routing | `NissaHub`, `ConsoleOutput`, `FileOutput`, `DirectoryOutput` |
| `tracker.py` | Diagnostic collection | `DiagnosticTracker`, `EpochSnapshot`, `GradientHealth` |
| `profiles.yaml` | Collection presets | minimal, standard, diagnostic, research |

### Dependencies

- **Inbound:** Simic (emit events), Tolaria (emit events), Karn (consume events)
- **Outbound:** Leyline (event types), stdout/filesystem

### Patterns Observed

- **Async Event Processing:** Background worker thread prevents I/O blocking training
- **Profile-Based Config:** minimal (+0%) → research (+60% overhead)
- **Compute Cost Multipliers:** Blueprint-specific overhead estimates
- **Narrative Generation:** Human-readable training summaries

### Concerns

- None observed. Clean hub architecture.

**Confidence:** High - Full profile and backend analysis.

---

## 8. Runtime (Task Specifications)

**Location:** `src/esper/runtime/`

**Responsibility:** Task specifications and factories for wiring Tolaria/Simic. Defines model, dataloader, and configuration factories per task.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `tasks.py` | Task presets | `TaskSpec`, `get_task_spec()` |

### Dependencies

- **Inbound:** Scripts (CLI), Tolaria (model creation)
- **Outbound:** Kasmina (hosts), Tamiyo (task config), Simic (reward config), Utils (data)

### Patterns Observed

- **Task Presets:** cifar10, cifar10_deep, cifar10_blind, tinystories
- **Factory Pattern:** Model + dataloader factories per task
- **Deliberate Weakness:** Host networks intentionally weak to leave headroom for seeds

### Concerns

- None observed. Clear factory pattern.

**Confidence:** High - Complete preset inventory.

---

## 9. Utils (Shared Utilities)

**Location:** `src/esper/utils/`

**Responsibility:** Shared utilities for data loading and loss computation, optimized for multi-environment training.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `data.py` | Data loading | `SharedBatchIterator`, `SharedGPUBatchIterator`, `load_cifar10()` |
| `loss.py` | Loss computation | `compute_task_loss()`, `compute_task_loss_with_metrics()` |

### Dependencies

- **Inbound:** Simic (training), Tolaria (validation), Runtime (dataloaders)
- **Outbound:** torch, torchvision, transformers/datasets (optional)

### Patterns Observed

- **SharedBatchIterator:** Single DataLoader for N envs (massive IPC reduction)
- **GPU Preloading:** 8x faster for small datasets (CIFAR-10)
- **Critical Cloning:** After `tensor_split()` to prevent CUDA stream races
- **Fused Metrics:** `argmax().eq().sum()` stays on GPU (no per-batch sync)

### Concerns

- None observed. Performance-critical code well-optimized.

**Confidence:** High - Complete analysis with performance rationale.

---

## 10. Scripts (CLI Entry Points)

**Location:** `src/esper/scripts/`

**Responsibility:** CLI entry points for training Simic RL agents.

### Key Components

| File | Purpose | Key Exports |
|------|---------|-------------|
| `train.py` | Training CLI | `main()`, `build_parser()` |

### Dependencies

- **Inbound:** User (CLI invocation)
- **Outbound:** All domains (orchestration)

### Patterns Observed

- **Subcommand Structure:** `heuristic` vs `ppo` algorithms
- **Multiprocessing Safety:** spawn method set before main()
- **Graceful Shutdown:** threading.Event signaled by Sanctum TUI

### Concerns

- None observed. Clean orchestration.

**Confidence:** High - Complete CLI analysis.

---

## Dependency Matrix

| Domain | Imports From | Exports To |
|--------|--------------|------------|
| **Leyline** | stdlib | ALL |
| **Kasmina** | Leyline | Simic, Tolaria |
| **Tamiyo** | Leyline, Nissa(hub) | Simic |
| **Simic** | Leyline, Tamiyo, Kasmina, Nissa, Karn, Utils | Tolaria, Scripts |
| **Tolaria** | Leyline, Nissa, Runtime(lazy) | Simic, Scripts |
| **Nissa** | Leyline | Karn, Simic, Tolaria |
| **Karn** | Leyline, Nissa(protocol) | Scripts |
| **Runtime** | Kasmina, Tamiyo, Simic, Utils | Scripts, Tolaria |
| **Utils** | torch, torchvision | Simic, Tolaria, Runtime |
| **Scripts** | ALL | User |

---

## Schema Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-30 | Initial catalog |
