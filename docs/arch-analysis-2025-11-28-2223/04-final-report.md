# Esper V1.0 Architecture Report

## Executive Summary

Esper is a sophisticated adaptive neural architecture system that explores dynamic model enhancement through controlled seed lifecycle management. The system implements a morphogenetic approach where new components (seeds) are trained in isolation, then carefully grafted into a host model through multi-stage lifecycle transitions using gradient isolation and alpha-blending. This prevents catastrophic forgetting while strategically expanding model capacity.

The architecture demonstrates **strong design maturity** with clear separation of concerns across 6 well-defined subsystems (9,146 LOC in Python). The system combines domain-driven design, comprehensive type safety, and performance-conscious patterns (hot path isolation, lazy imports, named tuples) to create a maintainable foundation for advanced neural architecture research.

Overall quality assessment: **Grade B+** - Excellent architectural design with operational gaps that are addressable through focused implementation work. The codebase is ready for research use but requires attention to error handling, test coverage, and orchestration decomposition before production deployment.

---

## System Overview

### Purpose

Esper solves the problem of **efficient model expansion without catastrophic forgetting**. Rather than retraining entire models from scratch when adding capacity, Esper:

1. Generates seed modules (neural network components) with specific architectures
2. Trains seeds in isolated environments on the main task
3. Gradually blends successful seeds into a host model using alpha-scheduling
4. Monitors for regressions with multi-stage lifecycle gates
5. Uses reinforcement learning to learn optimal seed lifecycle decisions

This approach enables:
- **Efficient capacity expansion** - Add only what improves performance
- **Risk management** - Quality gates prevent catastrophic forgetting
- **Learned automation** - RL policies optimize when/how to integrate seeds
- **Observable process** - Rich telemetry of gradient health and per-class metrics

### Key Capabilities

- **Morphogenetic seed lifecycle** - Germination → Training → Blending → Shadowing → Probationary → Fossilized (success) or Culled (failure)
- **Gradient isolation** - Hook-based interception prevents host gradient corruption during seed blending
- **Alpha-blending** - Gradual integration coefficient (0→1) schedules smooth seed adoption
- **RL-based decision making** - PPO/IQL agents learn when to advance, cull, or change seed architectures
- **Multiple seed blueprints** - Convolutional, Attention, Normalization, Depthwise architectures
- **Heuristic baseline** - Rule-based policy for comparison and research validation
- **Comprehensive telemetry** - Gradient health metrics, loss landscape analysis, per-class accuracy tracking
- **Episode collection** - Full training data capture for offline RL analysis

### Technology Stack

**Languages & Runtime**:
- Python 3.11+ (modern syntax, type hints)
- PyTorch 2.0+ (deep learning framework)
- NumPy 1.24+ (numerical computation)

**Key Dependencies**:
- `torch` - Neural network training, gradient manipulation, nn.Module inheritance
- `numpy` - Statistical operations for telemetry and rewards
- `torchvision` - CIFAR-10 dataset for benchmarking
- `pydantic` - Configuration management (Nissa telemetry profiles)

**Development Tools**:
- `pytest` - Testing infrastructure
- Standard Python: `dataclasses`, `enum`, `typing`, `pathlib`

---

## Architecture

### Design Philosophy

Esper embodies three core principles:

1. **Domain-Driven Design**: Each subsystem has a single, well-defined responsibility using MTG Planeswalker character names for memorability:
   - Leyline (Data contracts) → Kasmina (Mechanics) → Tamiyo (Decisions) → Simic (Learning) → Nissa (Telemetry)

2. **Type-Safe Communication**: Shared contracts (enums, dataclasses, protocols) enable loose coupling across subsystems without sacrificing static analysis.

3. **Performance-First Architecture**: Hot path isolation (features.py), lazy imports, and memory optimization (slots, named tuples) ensure research-grade efficiency without sacrificing readability.

### Package Structure

```
src/esper/ (9,146 LOC total)
├── leyline/          (1,057 LOC) - Data contracts & protocols [Foundation]
│   ├── actions.py    - SimicAction enum (seed lifecycle actions)
│   ├── stages.py     - SeedStage FSM with 11 states and validation
│   ├── signals.py    - TrainingSignals & FastTrainingSignals (27 features)
│   ├── schemas.py    - Domain types: GateLevel, AdaptationCommand, BlueprintProtocol
│   ├── reports.py    - SeedMetrics, FieldReport (observation types)
│   └── telemetry.py  - TelemetryEvent contracts for cross-cutting concerns
│
├── kasmina/          (1,210 LOC) - Seed mechanics & lifecycle [Mechanics Tier]
│   ├── slot.py       - SeedSlot lifecycle management (607 LOC)
│   ├── blueprints.py - 5 seed architectures (Conv, Attention, Norm, Depthwise)
│   ├── host.py       - HostCNN & MorphogeneticModel (seed composition)
│   └── isolation.py  - Gradient isolation & alpha-blending
│
├── tamiyo/           (501 LOC) - Strategic decision-making [Intelligence Tier]
│   ├── decisions.py  - TamiyoAction enum & decision structures
│   ├── tracker.py    - Signal observation & decision history
│   └── heuristic.py  - Rule-based baseline policy (251 LOC)
│
├── simic/            (4,615 LOC) - RL training infrastructure [Learning Tier]
│   ├── episodes.py   - Episode data structures & collection (719 LOC)
│   ├── features.py   - Feature extraction HOT PATH (27-dim, O(1))
│   ├── rewards.py    - Multi-component reward shaping (376 LOC)
│   ├── networks.py   - PolicyNetwork (actor-critic architecture)
│   ├── ppo.py        - PPO online RL algorithm (1,590 LOC)
│   └── iql.py        - IQL/CQL offline RL (1,326 LOC)
│
├── nissa/            (358 LOC) - Telemetry hub [Cross-Cutting]
│   ├── config.py     - TelemetryConfig with profiles (diagnostic/minimal/production)
│   ├── tracker.py    - DiagnosticTracker & gradient health metrics
│   └── output.py     - Output backends (Console, File, NissaHub router)
│
├── simic_overnight.py (859 LOC) - Orchestrator (legacy monolithic)
└── scripts/          (Entry points)
    ├── train.py      - PPO training CLI
    ├── generate.py   - Data generation (TODO stub)
    └── evaluate.py   - Policy evaluation (TODO stub)
```

### Key Architectural Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| **Contract-first design (Leyline)** | Loose coupling enables subsystem evolution independently. Enum/dataclass/Protocol contracts prevent coupling on implementation details. | Requires discipline to use contracts consistently; initial overhead in contract definition. |
| **Finite State Machine for seeds** | 11-state FSM with validation ensures reproducible lifecycle. VALID_TRANSITIONS dict prevents invalid state transitions. | FSM inherently finite; complex transitions require new states rather than conditions. |
| **Gradient isolation via hooks** | PyTorch hook mechanism intercepts backward passes without modifying core training loop. Prevents host gradient contamination during blending. | Hooks must be registered/cleaned carefully; exception-safety not verified. |
| **Dual RL algorithms (PPO + IQL)** | PPO for online learning (fast iteration), IQL for offline analysis. Supports different data regimes and validation approaches. | Code duplication in networks, buffers; hyperparameter tuning required per algorithm. |
| **Hot path isolation (features.py)** | Feature extraction isolated with zero cross-package imports. Enables future JIT compilation, vectorization without subsystem overhead. | Requires conscious dependency management; refactoring may break this invariant. |
| **Alpha-blending over direct integration** | Gradual integration (α: 0→1) prevents catastrophic forgetting, allows rollback if issues detected. More controllable than all-or-nothing approaches. | Adds computational cost during blending phase; schedule requires tuning. |
| **Named tuples for episodes** | Stack-allocated, zero-copy, immutable training data. Efficient for dataset storage and vectorized processing. | Less flexible than dataclasses; schema changes require new types. |
| **Lazy imports for PPO/IQL** | Avoids loading 2,900+ LOC of heavy algorithms until needed. Fast CLI startup, independent feature extraction usage. | Runtime error if lazy import called before main script loads; not caught at import time. |

---

## Subsystem Summary

### Leyline (Contracts) - Foundation Layer

**Responsibility**: Define shared data contracts and protocols enabling type-safe communication across all subsystems.

Leyline implements the "lingua franca" of Esper through:
- **11-state FSM** (SeedStage): DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED (or failure paths)
- **Action space** (SimicAction): 8 seed lifecycle actions (GERMINATE, ADVANCE_STAGE, CULL, CHANGE_BLUEPRINT, etc.)
- **Observation space** (TensorSchema): 27 features (epoch, loss, accuracy, gradient norms, per-class metrics)
- **Protocols** (BlueprintProtocol, TamiyoPolicy, OutputBackend): Abstract interfaces for extensible implementations

**Strengths**:
- Zero external dependencies (stdlib only)
- Comprehensive FSM with validation functions (`is_valid_transition()`, `is_terminal_stage()`)
- Well-organized into 7 focused modules
- Extensive docstrings explaining lifecycle and contracts

**Concerns**:
- Two parallel signal types (FastTrainingSignals vs TrainingSignals) creating confusion
- TensorSchema hardcoded to 27 features (scaling requires extension)
- No error messages for invalid transitions (could help debugging)

---

### Kasmina (Seed Mechanics) - Mechanics Tier

**Responsibility**: Manage seed module lifecycle through germination, training, blending, and fossilization with safe gradient isolation.

Kasmina implements the "morphogenetic" approach:
- **SeedSlot** (607 LOC): Lifecycle state machine managing seed training, quality gates, and stage transitions
- **Blueprints** (154 LOC): 5 configurable architectures (ConvEnhanceSeed, AttentionSeed, NormSeed, DepthwiseSeed)
- **MorphogeneticModel**: Host model composition integrating seeds via alpha-blending
- **GradientIsolationMonitor**: Hook-based interception preventing host gradient corruption

**Strengths**:
- Comprehensive lifecycle with clear quality gates at each transition
- Multiple blueprint options for architecture exploration
- Safe gradient isolation via hook-based interception
- Alpha-blending prevents catastrophic forgetting

**Concerns**:
- SeedSlot.py at 607 LOC could be split (state, metrics, lifecycle)
- Gradient isolation relies on hooks—cleanup on exception not verified
- Alpha schedule hardcoded (could be parameterized)
- Limited edge case handling (seed already at terminal stage)

---

### Tamiyo (Decision Engine) - Intelligence Tier

**Responsibility**: Observe training signals and make strategic decisions about seed lifecycle management.

Tamiyo bridges observation and action:
- **SignalTracker**: Aggregates training signals and tracks decision history
- **TamiyoAction enum**: 7 strategic actions (WAIT, GERMINATE, ADVANCE_STAGE, CULL, CHANGE_BLUEPRINT)
- **HeuristicTamiyo**: Rule-based baseline policy (~10 tunable parameters)
- **TamiyoPolicy Protocol**: Interface for heuristic and learned implementations

**Strengths**:
- Clean abstraction: observes signals, makes decisions, remains agnostic to implementation
- Protocol-based design enables side-by-side comparison (heuristic vs learned)
- Risk level assignment provides accountability trace
- Configurable heuristic with sensible defaults

**Concerns**:
- Many tunable parameters (plateau_epochs, improvement_threshold, etc.) with minimal guidance
- SignalTracker implementation not fully verified for memory efficiency
- No explicit tie-breaking logic when multiple conditions trigger
- Confidence always hardcoded to 1.0 (no uncertainty model)

---

### Simic (RL Training) - Learning Tier

**Responsibility**: Train neural network policies to improve Tamiyo's decisions via PPO (online) and IQL/CQL (offline) algorithms.

Simic provides the learning infrastructure:
- **PPO (1,590 LOC)**: Online RL with vectorized environments, GAE, entropy regularization
- **IQL (1,326 LOC)**: Offline RL with conservative Q-learning, expectile regression
- **PolicyNetwork**: Actor-critic architecture (policy logits + value estimate)
- **EpisodeCollector**: Structured episode collection (snapshot→decision→outcome→reward)
- **Feature extraction** (161 LOC, HOT PATH): O(1) 27-dimensional observation space with zero cross-package dependencies

**Strengths**:
- Well-organized into logical components (episodes, features, rewards, networks, algorithms)
- Hot path properly isolated (features.py with only Leyline imports)
- Comprehensive reward shaping (stage progress, convergence, intervention cost)
- Vectorized PPO enables sample-efficient exploration
- Dataset management supports offline learning workflows

**Concerns**:
- PPO and IQL are large (1,590 and 1,326 LOC) - could be split into agent/trainer/buffer
- Algorithm documentation missing (GAE, PPO-Clip, expectile regression not explained)
- ReplayBuffer pre-loads full dataset to GPU (OOM risk on large datasets)
- Observation normalization (RunningMeanStd) partially verified
- PPO hyperparameters not reviewed for sensitivity

---

### Nissa (Telemetry) - Cross-Cutting Concerns

**Responsibility**: Collect and route gradient health, loss landscape, and per-class metrics without polluting core logic.

Nissa provides observability:
- **DiagnosticTracker**: Gradient health metrics (norm, std, percentiles, vanishing/exploding %)
- **TelemetryConfig**: Pydantic-based profiles (diagnostic, minimal, production)
- **NissaHub**: Observer pattern multiplexing to multiple backends (Console, File)
- **Narrative generation**: Human-readable training health summaries

**Strengths**:
- Clean separation: config → collection → output
- Multiple output backends support different deployment scenarios
- Rich gradient metrics enable debugging
- Pydantic validation prevents invalid configurations
- Profile support simplifies tuning for different scenarios

**Concerns**:
- Computational cost of DiagnosticTracker not measured (could be 5-10% overhead)
- Hook registration on parameters may have cleanup issues
- Loss landscape computation (Hessian) potentially expensive
- Per-class metrics require dataset knowledge (may not generalize)
- No rate limiting on file output (could generate massive files)

---

## Quality Assessment Summary

### Strengths

1. **Clear separation of concerns**: 6 well-defined subsystems each with single responsibility
2. **Type safety**: 78% of functions have return type hints; heavy Protocol usage enables extensibility
3. **Performance awareness**: Hot path isolation (features.py), lazy imports, dataclass slots optimization
4. **Comprehensive contracts**: Leyline enables loose coupling without sacrificing static analysis
5. **Sophisticated mechanisms**: FSM validation, gradient isolation, alpha-blending, reward shaping all implemented thoughtfully

### Areas for Improvement

1. **Limited error handling**: Only 10 files have try-except blocks; training crash on invalid data/state without recovery
2. **Large files create maintenance friction**: simic/ppo.py (1,590 LOC), simic/iql.py (1,326 LOC), simic_overnight.py (859 LOC)
3. **Incomplete test coverage**: ~30% of codebase tested (leyline + simic); gaps in kasmina, tamiyo, nissa
4. **Missing algorithm documentation**: PPO/IQL training loops lack inline comments explaining GAE, expectile regression
5. **Monolithic orchestrator**: simic_overnight.py integrates all subsystems in single file; hard to test, reuse individual components

### Quality Scores

| Dimension | Score | Grade | Notes |
|-----------|-------|-------|-------|
| **Code Complexity** | 6/10 | B | 5 files >600 LOC, acceptable for ML code but watch growth |
| **Maintainability** | 7/10 | B | Clear structure but monolithic orchestrator hurts |
| **Type Safety** | 8/10 | B+ | Strong 78% coverage; Protocols well-used |
| **Documentation** | 6/10 | B | Module/class docs good; algorithms lack explanation |
| **Testing** | 5/10 | C+ | 30% coverage; major subsystems untested |
| **Performance** | 7/10 | B | Hot path isolated; telemetry cost not measured |
| **Error Handling** | 4/10 | C | Minimal recovery; relies on exception propagation |
| **Extensibility** | 8/10 | B+ | Protocol-based design, good separation of concerns |
| **OVERALL** | **6.5/10** | **B+** | Excellent architecture, operational gaps addressable |

---

## Key Diagrams

The architecture employs a **layered design** with clear dependency flow:

**System Context** (Level 1):
- Developer interacts with Esper CLI
- System trains on CIFAR-10 dataset
- Computational work leverages GPU hardware
- System reports metrics and model improvements

**Container Architecture** (Level 2):
```
                Entry Points (Scripts + simic_overnight.py)
                           |
        ┌──────────────────┼──────────────────┐
        |                  |                  |
     Tamiyo            Kasmina            Simic
   (Decisions)       (Mechanics)         (Learning)
        |                  |                  |
        └──────────────────┼──────────────────┘
                      Leyline (Contracts)
```

**Data Flow Pipeline**:
```
TrainingSignals → Feature Extraction → Policy Decision → Seed Command → 
Model Execution → Metrics & Telemetry → Next Epoch
```

**Seed Lifecycle FSM**:
```
DORMANT → GERMINATED → TRAINING ──┬→ BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED
                            └───→ CULLED (failure)
                      EMBARGOED ←─ (regression)
```

Refer to `03-diagrams.md` for detailed C4 component diagrams, sequence diagrams, and architectural patterns.

---

## Recommendations

### Immediate (This Sprint)

1. **Complete script stubs** (2 days)
   - Implement `scripts/generate.py` for data generation CLI
   - Implement `scripts/evaluate.py` for policy evaluation
   - Remove TODO comments blocking user workflows

2. **Add error handling** (3 days)
   - Wrap training loops with try-except
   - Handle invalid stage transitions, feature dimension mismatches, NaN rewards
   - Log errors with context (epoch, seed, action)

3. **Extract orchestrator functions** (2 days)
   - Move `generate_episodes()`, `train_policy()`, `evaluate_policy()` to separate modules
   - Create `simic/generation.py`, `simic/training.py`, `simic/evaluation.py`
   - Add unit tests for each function in isolation

### Short-term (2-4 Weeks)

4. **Consolidate signal types** (2 days)
   - Choose FastTrainingSignals or TrainingSignals as canonical
   - Migrate all 27-feature observations to single type
   - Remove duplicate conversion logic

5. **Add algorithm documentation** (2 days)
   - Document PPO: GAE computation, PPO-Clip objective, entropy bonus
   - Document IQL: V-network, expectile regression, conservative Q-learning
   - Explain reward shaping weights and sensitivity

6. **Implement missing subsystem tests** (4 days)
   - `test_kasmina.py`: Lifecycle transitions, quality gates, alpha blending
   - `test_tamiyo.py`: HeuristicTamiyo edge cases, signal tracking
   - `test_nissa.py`: Telemetry config, gradient tracking
   - Target: >80% function coverage per subsystem

### Long-term (Roadmap)

7. **Benchmark performance hotspots** (2 days)
   - Measure feature extraction (claim O(1))
   - Characterize telemetry overhead
   - Test replay buffer on 100k+ episodes
   - Document performance profile

8. **Parameterize hardcoded values** (1 day)
   - Extract CIFAR-10 dataset to CLI argument
   - Move hardcoded thresholds to config (where possible)
   - Expose PPO/IQL hyperparameters via argparse

9. **Multi-dataset support** (3 days)
   - Generalize from CIFAR-10 to arbitrary datasets
   - Support ImageNet, custom datasets
   - Enables broader research applicability

10. **Production hardening** (ongoing)
    - Setup CI/CD pipeline for regression testing
    - Add performance regression tests
    - Implement graceful shutdown/checkpointing for long training runs

---

## Appendix

### Document References

- `01-discovery-findings.md` - Initial codebase scan, subsystem identification, dependency analysis
- `02-subsystem-catalog.md` - Detailed subsystem documentation with APIs, patterns, confidence levels
- `03-diagrams.md` - C4 architectural diagrams (context, container, component), data flow, FSM
- `05-quality-assessment.md` - Code quality analysis, technical debt inventory, recommendations

### Analysis Metadata

- **Date**: 2025-11-28 to 2025-11-29
- **Scope**: Full source tree (34 Python files, 9,146 LOC)
- **Analyzer**: Claude Code (Haiku 4.5)
- **Confidence**: High (85%) - systematic exploration with focused deep dives
- **Limitations**: 
  - Algorithm details (PPO/IQL) partially examined
  - Performance characteristics measured via code review, not benchmarks
  - Runtime error handling paths not exhaustively traced

### Next Steps

1. **Implement High Priority recommendations** to improve test coverage from 30% to >70%
2. **Review findings with team** for architectural concerns or disagreements
3. **Plan Phase 2** development cycle around medium/long-term recommendations
4. **Monitor performance** once improvements implemented

---

**Report Generated**: 2025-11-29  
**Analysis Basis**: 4 discovery documents (01-05) synthesized into executive summary  
**Recommended Audience**: Technical leads, architects, development team

