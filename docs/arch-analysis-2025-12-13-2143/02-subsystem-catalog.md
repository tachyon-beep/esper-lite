# Subsystem Catalog - esper-lite

**Analysis Date:** 2025-12-13
**Total Subsystems:** 9
**Total Source LOC:** ~17,100
**Analysis Method:** Parallel subagent exploration with structured output contracts

---

## Catalog Summary

| # | Subsystem | Location | LOC | Responsibility | Confidence |
|---|-----------|----------|-----|----------------|------------|
| 1 | Leyline | `src/esper/leyline/` | 1,177 | Data contracts, signal schemas, stage enums | HIGH |
| 2 | Kasmina | `src/esper/kasmina/` | 2,935 | Seed lifecycle, slot management, gradient isolation | HIGH |
| 3 | Tamiyo | `src/esper/tamiyo/` | 631 | Decision policy (heuristic controller), signal tracking | HIGH |
| 4 | Tolaria | `src/esper/tolaria/` | 701 | Training execution engine, failure monitoring | HIGH |
| 5 | Simic | `src/esper/simic/` | 8,290 | RL infrastructure (PPO, vectorized training, rewards) | HIGH |
| 6 | Nissa | `src/esper/nissa/` | 1,558 | Telemetry hub, output backends, analytics | HIGH |
| 7 | Runtime | `src/esper/runtime/` | 229 | Task presets and factories | HIGH |
| 8 | Utils | `src/esper/utils/` | 571 | Data loading, loss computation, shared utilities | HIGH |
| 9 | Scripts | `src/esper/scripts/` | 1,021 | CLI entry points (training, evaluation) | HIGH |

---

## 1. Leyline (Data Contracts)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/leyline/`
- **LOC:** 1,177 across 8 files
- **Role:** Foundation layer defining immutable data contracts for inter-subsystem communication

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `stages.py` | SeedStage enum, valid transitions, risk levels | 123 |
| `signals.py` | FastTrainingSignals (NamedTuple), TrainingSignals (dataclass) | 291 |
| `schemas.py` | AdaptationCommand, GateResult, BlueprintProtocol | 147 |
| `factored_actions.py` | FactoredAction (4-head action space for PPO) | 137 |
| `reports.py` | SeedMetrics, SeedStateReport, FieldReport | 127 |
| `telemetry.py` | TelemetryEvent, TelemetryEventType enum | 175 |

### Design Patterns
- **Zero external dependencies** (stdlib only)
- **Immutability principle** (frozen dataclasses, NamedTuples)
- **Two-tier signals** (FastTrainingSignals for hot path, TrainingSignals for context)
- **Enum-heavy** for type safety

### Dependencies
- **Inbound:** All subsystems depend on Leyline contracts
- **Outbound:** None (pure stdlib)

---

## 2. Kasmina (Seed Mechanics)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/kasmina/`
- **LOC:** 2,935 across 10 files
- **Role:** Manages seed module lifecycle from creation through fossilization

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `slot.py` | SeedSlot (main Module), SeedState, QualityGates (G0-G5) | 1,319 |
| `host.py` | MorphogeneticModel, CNNHost, TransformerHost | 635 |
| `isolation.py` | AlphaSchedule, blend_with_isolation, GradientIsolationMonitor | 221 |
| `blueprints/registry.py` | BlueprintRegistry (plugin system) | 125 |
| `blueprints/cnn.py` | 7 CNN seed blueprints (noop, norm, attention, etc.) | 206 |
| `blueprints/transformer.py` | 5 transformer seed blueprints (lora, flex_attention, etc.) | 197 |

### Design Patterns
- **Lifecycle state machine** (11 stages with VALID_TRANSITIONS)
- **Quality gates** (deterministic G0-G5 for stage advancement)
- **Blueprint registry** (decorator-based plugin architecture)
- **Gradient isolation** (STE for TRAINING, topology-aware for BLENDING)
- **DDP consensus** (unanimous gate decisions across ranks)

### Dependencies
- **Inbound:** Simic, Tamiyo, Tolaria, Runtime, Scripts
- **Outbound:** Leyline, PyTorch

---

## 3. Tamiyo (Decision Policy)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/tamiyo/`
- **LOC:** 631 across 4 files
- **Role:** Strategic decision-making for seed lifecycle (germinate/fossilize/cull)

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `heuristic.py` | HeuristicTamiyo (rule-based policy), HeuristicPolicyConfig | 291 |
| `tracker.py` | SignalTracker (observes training metrics) | 220 |
| `decisions.py` | TamiyoDecision (action + reason + confidence) | 99 |

### Design Patterns
- **Strategy pattern** (TamiyoPolicy Protocol enables multiple implementations)
- **Signal tracking** (running statistics, plateau detection)
- **Penalty tracking with decay** (anti-thrashing for blueprints)
- **Hierarchical decision logic** (germination vs. seed management branches)

### Dependencies
- **Inbound:** Simic (uses SignalTracker, HeuristicTamiyo)
- **Outbound:** Leyline, Kasmina (TYPE_CHECKING), Nissa

---

## 4. Tolaria (Training Execution)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/tolaria/`
- **LOC:** 701 across 4 files
- **Role:** Generic training/validation loops with failure detection

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `trainer.py` | train_epoch_normal/incubator/blended, validate_with_attribution | 379 |
| `governor.py` | TolariaGovernor (fail-safe watchdog), rollback semantics | 241 |
| `environment.py` | create_model() factory | 36 |

### Design Patterns
- **Three training modes** (normal, STE incubator, blended)
- **Counterfactual validation** (causal seed attribution via alpha=0 baseline)
- **Fail-safe watchdog** (NaN/Inf detection, statistical anomaly, LKG rollback)
- **Async GPU transfers** (non_blocking for efficiency)

### Dependencies
- **Inbound:** Simic, Scripts
- **Outbound:** Kasmina, Runtime, Utils, PyTorch

---

## 5. Simic (RL Infrastructure)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/simic/`
- **LOC:** 8,290 across 22 files
- **Role:** PPO-based RL training infrastructure for learned seed management

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `vectorized.py` | train_ppo_vectorized (multi-GPU parallel training) | 1,496 |
| `rewards.py` | Counterfactual validation, PBRS, compute rent | 988 |
| `ppo.py` | PPOAgent (factored recurrent actor-critic) | 577 |
| `training.py` | compiled_train_step, single-episode training | 519 |
| `networks.py` | PolicyNetwork, FactoredRecurrentActorCritic | 482 |
| `tamiyo_buffer.py` | TamiyoRolloutBuffer (per-env pre-allocated) | 395 |
| `features.py` | obs_to_base_features (HOT PATH, 35-dim) | 387 |
| `tamiyo_network.py` | LSTM with LayerNorm, 4-head outputs | 334 |

### Design Patterns
- **PPO with factored actions** (4 heads: op, slot, blueprint, blend)
- **Recurrent actor-critic** (LSTM for temporal seed dynamics)
- **Vectorized training** (inverted control flow, SharedBatchIterator)
- **PBRS reward shaping** (policy-invariant stage bonuses)
- **Counterfactual validation** (ransomware-resistant reward signal)
- **Tiered telemetry** (MINIMAL/NORMAL/DEBUG with auto-escalation)

### Dependencies
- **Inbound:** Scripts
- **Outbound:** Leyline, Kasmina, Tamiyo, Tolaria, Nissa, Runtime, Utils

---

## 6. Nissa (Telemetry Hub)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/nissa/`
- **LOC:** 1,558 across 5 files
- **Role:** Central telemetry collection, routing, and analytics

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `tracker.py` | DiagnosticTracker (gradient/loss/per-class metrics) | 542 |
| `output.py` | NissaHub, ConsoleOutput, FileOutput, DirectoryOutput | 405 |
| `analytics.py` | BlueprintAnalytics, BlueprintStats, SeedScoreboard | 294 |
| `config.py` | TelemetryConfig, profiles (minimal/standard/diagnostic/research) | 237 |

### Design Patterns
- **Observer/Pub-Sub** (NissaHub broadcasts to all backends)
- **Profile-based configuration** (YAML profiles with deep merge)
- **Singleton hub** (global get_hub() for cross-subsystem access)
- **Hook-based instrumentation** (gradient tracking via PyTorch hooks)
- **Narrative generation** (human/LLM-readable training summaries)

### Dependencies
- **Inbound:** All subsystems emit telemetry to Nissa
- **Outbound:** Leyline, PyTorch, Pydantic, PyYAML

---

## 7. Runtime (Task Presets)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/runtime/`
- **LOC:** 229 across 2 files
- **Role:** Task-specific wiring and configuration factories

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `tasks.py` | TaskSpec dataclass, get_task_spec() factory | 224 |

### Task Presets
| Task | Topology | Description |
|------|----------|-------------|
| `cifar10` | CNN | Weak baseline CNN (3 blocks, ~40% acc) |
| `cifar10_deep` | CNN | Deeper CNN (5 blocks) for GPU saturation |
| `tinystories` | Transformer | GPT-style LM (6 layers, 384 dim) |

### Design Patterns
- **Factory pattern** (get_task_spec dispatches to private factories)
- **Configuration composition** (TaskConfig + LossRewardConfig held as fields)
- **Adapter pattern** (unified interface across CNN/Transformer)

### Dependencies
- **Inbound:** Simic, Tolaria, Scripts
- **Outbound:** Kasmina, Leyline, Simic (configs), Utils

---

## 8. Utils (Shared Utilities)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/utils/`
- **LOC:** 571 across 3 files
- **Role:** Cross-cutting infrastructure for data and loss computation

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `data.py` | SharedBatchIterator, load_cifar10, load_tinystories | 427 |
| `loss.py` | compute_task_loss, compute_task_loss_with_metrics | 127 |

### Design Patterns
- **SharedBatchIterator** (eliminates IPC overhead for multi-env training)
- **Lazy imports** (optional transformers/datasets loaded in function bodies)
- **Tensor metric returns** (deferred CUDA sync for hot path)
- **Mock support** (synthetic data for offline testing)

### Dependencies
- **Inbound:** Simic, Tolaria, Scripts, Runtime
- **Outbound:** PyTorch, torchvision, transformers (optional), datasets (optional)

---

## 9. Scripts (CLI Entry Points)

### Overview
- **Location:** `/home/john/esper-lite/src/esper/scripts/`
- **LOC:** 1,021 across 3 files
- **Role:** User-facing CLI for training and evaluation

### Key Components
| File | Responsibility | LOC |
|------|----------------|-----|
| `train.py` | Training CLI (heuristic, ppo subcommands) | 185 |
| `evaluate.py` | Evaluation CLI, diagnostic analysis | 836 |

### Commands
```bash
python -m esper.scripts.train heuristic --task cifar10
python -m esper.scripts.train ppo --task cifar10 --n-envs 4
python -m esper.scripts.evaluate --model ppo.pt --task cifar10
```

### Design Patterns
- **Command pattern** (argparse with subcommands)
- **Facade pattern** (hides subsystem complexity)
- **Telemetry wiring** (configures NissaHub before training)

### Dependencies
- **Inbound:** None (top-level entry point)
- **Outbound:** Simic, Nissa, Runtime, Tolaria, Tamiyo, Leyline, Utils

---

## Dependency Matrix

```
           Leyline  Kasmina  Tamiyo  Tolaria  Simic  Nissa  Runtime  Utils  Scripts
Leyline      -        -        -        -       -      -       -       -       -
Kasmina      ✓        -        -        -       -      -       -       -       -
Tamiyo       ✓        ✓*       -        -       -      ✓       -       -       -
Tolaria      -        ✓        -        -       -      -       ✓       ✓       -
Simic        ✓        ✓        ✓        ✓       -      ✓       ✓       ✓       -
Nissa        ✓        -        -        -       -      -       -       -       -
Runtime      ✓        ✓        -        -       ✓*     -       -       ✓       -
Utils        -        -        -        -       -      -       -       -       -
Scripts      ✓        -        ✓        ✓       ✓      ✓       ✓       ✓       -

Legend: ✓ = direct dependency, ✓* = TYPE_CHECKING only, - = no dependency
```

---

## Architecture Highlights

### Acyclic Dependency Graph
The dependency graph is strictly acyclic:
- **Leyline** is the foundation (no outbound deps)
- **Kasmina** depends only on Leyline
- **Simic** is the most dependent (integrates all subsystems)
- **Scripts** is the top-level orchestrator

### Performance Optimizations
1. **FastTrainingSignals** (NamedTuple for zero GC pressure)
2. **SharedBatchIterator** (eliminates N×M DataLoader worker overhead)
3. **torch.compile()** on networks and training steps
4. **CUDA streams** for async environment execution
5. **Tensor metric returns** (deferred sync for hot path)

### Quality Controls
1. **Quality gates** (G0-G5 for seed advancement)
2. **DDP consensus** (unanimous gate decisions)
3. **TolariaGovernor** (NaN/Inf detection, rollback)
4. **Counterfactual validation** (causal attribution)
5. **Tiered telemetry** (auto-escalation on anomalies)

---

## Analysis Confidence

All 9 subsystems analyzed with **HIGH** confidence:
- Complete code access
- Clear responsibility boundaries
- Explicit dependencies
- Well-documented patterns
- Production-ready indicators (torch.compile, DDP, checkpointing)
