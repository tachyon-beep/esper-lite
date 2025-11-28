# Esper V1.0 Discovery - Executive Summary

## What is Esper?

Esper is a **morphogenetic neural network system** that explores dynamic model enhancement through controlled seed lifecycle management. Seeds are new neural components trained in isolation, then carefully grafted into a host model through multi-stage transitions (germinate → train → blend → fossilize).

## Key Innovation

**Gradient Isolation + Alpha Blending**: Seeds train independently via hook-based gradient interception, then gradually merge using alpha schedules (0→1) to prevent catastrophic forgetting.

## Core Subsystems (5 Packages)

| Package | Purpose | Size |
|---------|---------|------|
| **Leyline** | Data contracts (FSM, signals, schemas) | 1.1 KLOC |
| **Kasmina** | Seed mechanics (lifecycle, isolation, blending) | 1.2 KLOC |
| **Tamiyo** | Strategic decisions (heuristic + learned policies) | 501 LOC |
| **Simic** | RL training (PPO/IQL episode collection) | 4.6 KLOC |
| **Nissa** | Telemetry hub (gradient health, metrics) | 358 LOC |

**Total**: 9,146 LOC across 34 Python files

## Architecture Pattern

**Domain-Driven Design** with strict tier separation:
- Data Tier (Leyline): Contracts only
- Mechanics Tier (Kasmina): Lifecycle implementation
- Intelligence Tier (Tamiyo): Decision-making
- Learning Tier (Simic): RL algorithms
- Telemetry Tier (Nissa): Cross-cutting observation

## Hot Path Optimization

**simic/features.py** is compartmentalized for performance:
- Extracts 27 features in O(1) time
- Only imports leyline (no cross-package dependencies)
- Uses NamedTuple for zero GC pressure
- Sits in tight vectorized training loop

## Entry Points

1. **simic_overnight.py** - Full orchestrator (episodes → training → evaluation)
2. **scripts/train.py** - PPO CLI with vectorized environments
3. **scripts/generate.py** - Offline RL data generation
4. **Module APIs**: MorphogeneticModel, HeuristicTamiyo, PolicyNetwork

## Key Design Patterns

1. **FSM with validation**: SeedStage transitions via VALID_TRANSITIONS dict
2. **Dataclass + Slots**: Memory-efficient metrics (~40% reduction)
3. **Named tuples**: FastTrainingSignals for zero-GC hot paths
4. **Protocol-based contracts**: Pluggable policies (heuristic vs learned)
5. **Lazy imports**: PPO/IQL (2.9 KLOC) deferred until needed
6. **Observer pattern**: NissaHub routes telemetry to backends

## Strengths

- Clear separation of concerns (each package has one job)
- Type-safe FSM prevents invalid state transitions
- Performance-conscious (hot path isolation, slots, named tuples)
- Extensible protocol design (swappable policies)
- Rich telemetry (gradient health, per-class metrics)

## Concerns

- Boundary complexity (simic_overnight.py integrates 5 subsystems)
- Incomplete error handling in core paths
- Sparse documentation (many algorithms lack docstrings)
- Test coverage gaps (2 files vs 9 KLOC)
- Loose PyTorch version pinning (>=2.0.0)

## Confidence: 85% High

**Gaps**: Error paths, datagen integration, algorithmic details, performance characteristics

---

**Analysis Date**: 2025-11-28 22:23 UTC
**Files Scanned**: 34 Python files + README/pyproject.toml
**Time Investment**: ~30 min systematic exploration
