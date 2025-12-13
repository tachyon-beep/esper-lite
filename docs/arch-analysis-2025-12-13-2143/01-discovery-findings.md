# Discovery Findings - esper-lite

**Analysis Date:** 2025-12-13
**Analyst:** System Archaeologist (automated analysis)
**Confidence Level:** HIGH (comprehensive file access, clear codebase structure)

## Executive Summary

esper-lite is a **morphogenetic neural network training system** that uses reinforcement learning to optimize model adaptation strategies. The system manages "seeds" (neural network configurations) through lifecycle stages, using either heuristic rules or a learned PPO policy to make adaptation decisions.

**Key Statistics:**
- **Total Source LOC:** ~17,100 (across 57 files in 11 modules)
- **Test LOC:** ~7,500 (across 88 test files)
- **Primary Language:** Python 3.11+
- **ML Framework:** PyTorch 2.8+
- **Architecture Pattern:** Domain-Driven Design with acyclic dependencies

---

## 1. Directory Structure Analysis

### Project Layout

```
esper-lite/
├── src/esper/                    # Main package (17.1K LOC)
│   ├── leyline/                  # Data contracts (844 LOC)
│   ├── kasmina/                  # Seed mechanics (1.4K LOC)
│   ├── tamiyo/                   # Decision logic (411 LOC)
│   ├── tolaria/                  # Training execution (746 LOC)
│   ├── simic/                    # RL infrastructure (7.8K LOC)
│   ├── nissa/                    # Telemetry hub (1.7K LOC)
│   ├── runtime/                  # Task presets (224 LOC)
│   ├── utils/                    # Utilities (571 LOC)
│   └── scripts/                  # CLI entry points (1.0K LOC)
├── tests/                        # Test suite (7.5K LOC, 88 files)
├── docs/                         # Documentation & analysis
├── data/                         # Datasets & checkpoints
├── _archive/                     # Historical prototypes (DO NOT USE)
└── telemetry/                    # Output artifacts
```

### Organizational Pattern: Domain-Driven Design

Each module encapsulates a distinct domain responsibility:
- **Leyline** = Contracts layer (upstream dependency for all)
- **Kasmina** = Topology operations (models, slots, grafting)
- **Tamiyo** = Decision-making policy layer
- **Tolaria** = PyTorch training execution engine
- **Simic** = RL training infrastructure for Tamiyo
- **Nissa** = Observability hub (receives telemetry from all domains)

### Notable Characteristics
- Clean, acyclic dependencies (no circular imports)
- Consistent public API exports via `__init__.py`
- Explicit separation of concerns
- No monolithic modules

---

## 2. Entry Points

### Main Training Script
**Location:** `src/esper/scripts/train.py` (185 LOC)

```bash
# Heuristic-based training
python -m esper.scripts.train heuristic --task cifar10

# PPO-based training
python -m esper.scripts.train ppo --task cifar10 --episodes 1000
```

**Features:**
- Subcommands: `heuristic` (rule-based) and `ppo` (RL-based)
- Telemetry output to console/file/directory
- GPU preload option for datasets

### Evaluation Script
**Location:** `src/esper/scripts/evaluate.py` (836 LOC)

Multi-task evaluation with:
- CIFAR-10, transformer variants, LM validation
- Blueprint cost analysis
- Checkpoint loading and metrics computation

### Task Presets
**Location:** `src/esper/runtime/tasks.py` (224 LOC)

Factory functions for:
- `cifar10_task()` - CNN classification
- `cifar10_deep_task()` - Transformer classification
- `tinystories_task()` - Language modeling

---

## 3. Technology Stack

### Core Dependencies

| Category | Technology | Version |
|----------|------------|---------|
| Language | Python | 3.11+ |
| ML Framework | PyTorch | >= 2.8.0 |
| Transformers | HuggingFace | >= 4.57.3 |
| Data Loading | Datasets | >= 4.4.1 |
| Numerics | NumPy | >= 1.24.0 |
| Testing | pytest | >= 7.0.0 |
| Property Testing | Hypothesis | >= 6.148.3 |

### RL Infrastructure (Custom Implementation)
- **PPO** - Proximal Policy Optimization from scratch
- **GAE** - Generalized Advantage Estimation
- **Prioritized Replay** - Sum-tree based experience buffer
- **Vectorized Environments** - Parallel training with CUDA streams

### Observability Stack
- Custom telemetry hub (Nissa)
- Pluggable backends: Console, File (JSONL), Directory
- Per-layer gradient statistics
- Memory profiling capabilities

---

## 4. Subsystem Identification

### 9 Major Subsystems Identified

| # | Subsystem | Location | Responsibility | LOC | Complexity |
|---|-----------|----------|----------------|-----|------------|
| 1 | **Leyline** | `src/esper/leyline/` | Data contracts, signal schemas, stage enums, action spaces | 844 | Low |
| 2 | **Kasmina** | `src/esper/kasmina/` | Seed lifecycle, slot management, gradient isolation, host models | 1,400 | Medium |
| 3 | **Tamiyo** | `src/esper/tamiyo/` | Decision policy (heuristic controller), signal tracking | 411 | Low |
| 4 | **Tolaria** | `src/esper/tolaria/` | Training execution (epoch loops, validation, failure monitoring) | 746 | Medium |
| 5 | **Simic** | `src/esper/simic/` | RL infrastructure (PPO, vectorized envs, rewards, networks) | 7,800 | High |
| 6 | **Nissa** | `src/esper/nissa/` | Telemetry hub (tracker, output backends, analytics) | 1,700 | Medium |
| 7 | **Runtime** | `src/esper/runtime/` | Task presets and factories | 224 | Low |
| 8 | **Utils** | `src/esper/utils/` | Dataset loading, loss computation | 571 | Low |
| 9 | **Scripts** | `src/esper/scripts/` | CLI entry points | 1,021 | Low |

### Dependency Graph (Acyclic)

```
                    ┌─────────────┐
                    │   Leyline   │  (contracts - foundation)
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Kasmina  │    │  Nissa   │    │  Utils   │
    │ (models) │    │(telemetry)│   │ (data)   │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         ▼               │               │
    ┌──────────┐         │               │
    │  Tamiyo  │◄────────┼───────────────┘
    │(decisions)│        │
    └────┬─────┘         │
         │               │
         ▼               ▼
    ┌──────────┐    ┌──────────┐
    │ Tolaria  │◄───│  Simic   │
    │(training)│    │   (RL)   │
    └────┬─────┘    └────┬─────┘
         │               │
         └───────┬───────┘
                 ▼
          ┌──────────┐
          │ Runtime  │
          │ (tasks)  │
          └────┬─────┘
               ▼
          ┌──────────┐
          │ Scripts  │
          │  (CLI)   │
          └──────────┘
```

---

## 5. Observed Design Patterns

### Architectural Patterns
1. **Domain-Driven Design** - Each domain owns contracts and implementations
2. **Acyclic Dependency Graph** - Clear upstream/downstream relationships
3. **Config-Driven Architecture** - Extensive dataclass configuration
4. **Telemetry-First Design** - Observability baked into core loops

### Implementation Patterns
1. **Dataclass-Heavy** - `@dataclass(slots=True)` for performance
2. **Protocol/ABC Abstraction** - `OutputBackend` ABC, `HostProtocol`
3. **Factory Pattern** - TaskSpec, model factories, blueprint registry
4. **Strategy Pattern** - Heuristic vs learned policy
5. **Observer Pattern** - Telemetry with pluggable backends
6. **State Machine** - SeedStage transitions with quality gates

### Performance Patterns
1. **Hot Path Optimization** - Separated fast feature extraction
2. **Vectorized RL** - Parallel environments with CUDA stream coordination
3. **Non-blocking Transfers** - Async GPU memory operations
4. **Slots-based Dataclasses** - Memory-efficient records

---

## 6. Initial Quality Observations

### Strengths
- **Strong modularity** - 9 focused subsystems, no monoliths
- **Excellent type coverage** - Comprehensive annotations throughout
- **Active maintenance** - Recent aggressive dead code removal
- **Comprehensive testing** - 88 test files, property-based testing
- **Performance conscious** - CUDA streams, vectorization, slots

### Areas of Concern

| Issue | Location | Severity | Impact |
|-------|----------|----------|--------|
| Large core module | `simic/vectorized.py` (1.5K LOC) | Medium | Testability, maintenance |
| Complex reward system | `simic/rewards.py` (988 LOC) | Medium | Understandability |
| Action space fragmentation | Multiple action representations | Low | Cognitive load |
| Sparse inline comments | Core subsystems | Medium | Onboarding difficulty |
| No formal API docs | Entire codebase | Medium | External adoption |

### Documentation Status
- **README.md** - Good project overview
- **CLAUDE.md** - Development constraints (no legacy code)
- **Inline comments** - Sparse in complex logic
- **API documentation** - Not present
- **Architecture docs** - Multiple review notes exist

---

## 7. Recommended Analysis Strategy

**Decision: PARALLEL ANALYSIS**

**Reasoning:**
- 9 independent subsystems identified
- Subsystems are loosely coupled (acyclic dependencies)
- Large codebase (17K+ LOC)
- Clear domain boundaries enable parallel work

**Execution Plan:**
1. Spawn parallel subagents for subsystem catalog entries
2. For each subsystem: spawn DRL specialist + PyTorch specialist reviews
3. Validate catalog completeness
4. Generate C4 diagrams from validated catalog
5. Synthesize final report with expert findings

**Estimated parallelization benefit:** 9 sequential analyses → ~3 parallel batches

---

## Appendix: File Inventory

### Largest Files (Complexity Indicators)
1. `simic/vectorized.py` - 1,500 LOC (parallel RL environments)
2. `kasmina/slot.py` - 1,300 LOC (seed slot state machine)
3. `simic/rewards.py` - 988 LOC (reward computation)
4. `scripts/evaluate.py` - 836 LOC (evaluation CLI)
5. `kasmina/host.py` - 635 LOC (host model management)

### Test Coverage by Subsystem
- `tests/simic/` - 25+ test files (RL infrastructure)
- `tests/kasmina/` - 15+ test files (seed mechanics)
- `tests/leyline/` - 10+ test files (contracts)
- `tests/integration/` - E2E tests
- `tests/properties/` - Hypothesis-based tests
