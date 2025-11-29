# Discovery Findings - Esper-Lite

## Executive Summary

Esper-lite is a **morphogenetic neural network training framework** that implements staged growth for neural networks. The core innovation is training "seed" modules in isolation, blending them into a host model, and fossilizing them once proven—aiming to add capabilities without destabilizing prior knowledge.

**Key Finding**: Well-architected research codebase with clear subsystem boundaries, consistent naming conventions, and separation of concerns. The system demonstrates a mature understanding of domain-driven design applied to ML research.

## Project Overview

| Attribute | Value |
|-----------|-------|
| Name | esper-lite |
| Version | 1.0.0 (package), 0.1.0 (pyproject) |
| Language | Python 3.11+ |
| Framework | PyTorch >= 2.0.0 |
| Lines of Code | ~9,200 (active src/esper/) |
| Subsystems | 7 identified |
| Test Coverage | Unit tests present for core modules |

## Directory Structure

```
esper-lite/
├── src/
│   └── esper/                    # Main package
│       ├── leyline/              # Data contracts (shared substrate)
│       ├── kasmina/              # Seed mechanics & host models
│       ├── tamiyo/               # Strategic decision-making
│       ├── simic/                # RL training (PPO/IQL)
│       ├── nissa/                # Telemetry hub
│       ├── tolaria/              # Training loop infrastructure
│       ├── utils/                # Shared utilities
│       └── scripts/              # CLI tools (train, evaluate)
├── tests/                        # Test suite
├── scripts/                      # Shell scripts (train_ppo.sh)
├── docs/
│   ├── plans/                    # Design documents
│   └── results/                  # POC results
├── data/                         # Generated artifacts (gitignored)
├── notebooks/                    # Research notebooks
└── _archive/                     # Deprecated code (preserved)
```

## Technology Stack

### Core Dependencies
- **PyTorch** (>= 2.0.0): Neural network framework
- **NumPy** (>= 1.24.0): Numerical computing

### Development Dependencies
- **pytest** (>= 7.0.0): Testing
- **IPython** (>= 8.0.0): Interactive development
- **Jupyter** (>= 1.0.0): Research notebooks

### Build System
- setuptools (>= 61.0)
- Package structure: `src/` layout

## Architectural Patterns

### 1. Layered Domain Architecture
The system follows a clear layered approach:
```
Leyline (Contracts) ← Foundation layer, defines all shared types
    ↑
Kasmina (Seeds) ← Core domain, implements seed lifecycle
    ↑
Tamiyo (Decisions) ← Heuristic controller
    ↑
Simic (Learning) ← RL training to improve Tamiyo
    ↑
Tolaria (Training) ← Training loop orchestration
    ↑
Nissa (Telemetry) ← Cross-cutting observability
```

### 2. Data Flow Pattern
```
Training Signals → Tamiyo/Simic Action → Kasmina Seed Update → Leyline Reports → Back to Signals
```

### 3. Seed Lifecycle State Machine
```
DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED
                ↓           ↓          ↓            ↓
              CULLED ←── EMBARGOED ← RESETTING ← (failure paths)
```

### 4. Naming Convention
Subsystems are named after Magic: The Gathering Planeswalkers:
- **Kasmina**: "Master of hidden knowledge" → Seed mechanics
- **Tamiyo**: "Observer who records without interference" → Decision-making
- **Leyline**: "Invisible substrate" → Data contracts
- **Simic**: (Guild, not Planeswalker) → Growth/adaptation → RL training
- **Nissa**: (Nature planeswalker) → System health → Telemetry
- **Tolaria**: (Academy/location) → Training infrastructure

## Subsystem Summary

| Subsystem | Location | LOC | Responsibility | Dependencies |
|-----------|----------|-----|----------------|--------------|
| Leyline | `src/esper/leyline/` | ~600 | Data contracts, schemas, enums | None (foundation) |
| Kasmina | `src/esper/kasmina/` | ~1,100 | Seed lifecycle, host models | Leyline |
| Tamiyo | `src/esper/tamiyo/` | ~500 | Heuristic decision controller | Leyline |
| Simic | `src/esper/simic/` | ~4,600 | RL training (PPO/IQL) | Leyline, Tamiyo |
| Nissa | `src/esper/nissa/` | ~1,000 | Telemetry configuration & output | Leyline |
| Tolaria | `src/esper/tolaria/` | ~270 | Training loop orchestration | Leyline, Kasmina, Tamiyo |
| Utils | `src/esper/utils/` | ~70 | Dataset loading | None |
| Scripts | `src/esper/scripts/` | ~700 | CLI tools | All subsystems |

## Entry Points

### Training Scripts
1. **POC (Fixed Schedule)**: `src/esper/poc.py` - Basic morphogenetic training
2. **POC (Tamiyo-Controlled)**: `src/esper/poc_tamiyo.py` - Heuristic-driven training
3. **PPO Training**: `scripts/train_ppo.sh` - Policy learning script
4. **Simic Overnight**: `src/esper/simic_overnight.py` - Batch policy training

### CLI Tools
- `src/esper/scripts/train.py` - Training entry point
- `src/esper/scripts/evaluate.py` - Policy evaluation & diagnostics

## Key Findings

### Strengths
1. **Clear Separation of Concerns**: Each subsystem has distinct responsibility
2. **Contract-First Design**: Leyline defines shapes/enums before implementation
3. **Well-Documented**: README, AGENTS.md, and design docs provide context
4. **Testable Architecture**: Unit tests present, deterministic fixtures
5. **Type Hints**: Consistent use throughout codebase
6. **Hot Path Optimization**: `@dataclass(slots=True)` for performance-critical types

### Areas for Investigation
1. **Version Mismatch**: `__version__ = "1.0.0"` vs `pyproject.toml version = "0.1.0"`
2. **Archive Presence**: Large `_archive/` suggests significant refactoring history
3. **Script Organization**: Some scripts in `src/esper/scripts/`, others in `scripts/`
4. **Simic Size**: Largest subsystem (~4,600 LOC) - may benefit from further decomposition

## External Dependencies

### Direct Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0.0 | Neural network operations |
| numpy | >= 1.24.0 | Array operations |

### Indirect/Bundled
- torchvision (for CIFAR-10 dataset, installed separately)

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies |
| `.gitignore` | VCS exclusions |
| `src/esper/nissa/profiles.yaml` | Telemetry configuration profiles |

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Project structure | High | Clear directory layout |
| Technology stack | High | Explicit in pyproject.toml |
| Subsystem boundaries | High | Well-defined in __init__.py files |
| Data flow | High | Documented in README |
| Internal implementation | Medium | Requires deeper file analysis |
| Test coverage | Medium | Tests exist but coverage unknown |
