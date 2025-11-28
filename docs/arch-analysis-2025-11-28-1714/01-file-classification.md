# src/esper File Classification Report (v2 - Deep Trace)

**Date:** 2025-11-28
**Method:** Runtime import tracing + static analysis

## Executive Summary

After tracing actual runtime imports from both training systems:

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| **PPO System** | 10 files | ~8,500 | Active |
| **IQL System** | 6 files | ~3,500 | Active |
| **Orphaned** | 3 files | ~1,450 | Archive candidates |
| **Broken Imports** | 3 refs | - | Need fixing |

## Runtime Dependency Analysis

### simic_ppo.py System (10 files)

```
simic_ppo.py (2053 lines)
├── esper.simic           (SimicAction)
│   └── esper.leyline     (contracts)
└── [lazy imports at runtime]:
    ├── esper.simic_overnight  (create_model, load_cifar10)
    │   ├── esper.kasmina      (SeedSlot, SeedStage, BlueprintCatalog)
    │   │   └── esper.leyline
    │   ├── esper.poc_tamiyo   (ConvBlock, HostCNN, MorphogeneticModel)
    │   ├── esper.tamiyo       (HeuristicTamiyo, SignalTracker)
    │   │   └── esper.leyline
    │   ├── esper.telemetry    (DiagnosticTracker)
    │   │   └── esper.telemetry_config
    │   └── esper.simic
    └── [BROKEN: esper.simic_cifar10 - does not exist]
```

**Files in PPO dependency chain:**
1. `simic_ppo.py` - PPO training entry point
2. `simic.py` - SimicAction, data structures
3. `leyline.py` - Contracts/interfaces
4. `simic_overnight.py` - Model creation (used as utility)
5. `poc_tamiyo.py` - CNN model classes (ConvBlock, HostCNN, MorphogeneticModel)
6. `kasmina.py` - Seed management
7. `tamiyo.py` - Heuristic controller
8. `telemetry.py` - Diagnostic tracker
9. `telemetry_config.py` - Telemetry config

### simic_iql.py System (6 files)

```
simic_iql.py (1325 lines)
├── esper.simic           (SimicAction, TrainingSnapshot)
│   └── esper.leyline     (contracts)
├── esper.simic_train     (obs_to_base_features, telemetry_to_features)
│   └── esper.simic
└── esper.tamiyo          (HeuristicTamiyo, SignalTracker)
    └── esper.leyline
```

**Files in IQL dependency chain:**
1. `simic_iql.py` - IQL training entry point
2. `simic.py` - SimicAction, data structures
3. `leyline.py` - Contracts
4. `simic_train.py` - Feature extraction utilities
5. `tamiyo.py` - Heuristic controller

## File Classification

### ORPHANED - Safe to Archive

| File | Lines | Reason |
|------|-------|--------|
| `poc.py` | 841 | Standalone demo, zero imports |
| `simic_test_collection.py` | 250 | Test script, zero imports |
| `rewards.py` | 377 | Recently created but NOT imported anywhere |

**Total orphaned:** 1,468 lines (14% of codebase)

### ACTIVE - Core PPO System

| File | Lines | Role |
|------|-------|------|
| `simic_ppo.py` | 2,053 | Entry point - online PPO training |
| `simic_overnight.py` | 857 | **Utility for PPO** - provides `create_model()`, `load_cifar10()` |
| `poc_tamiyo.py` | 595 | **Model definitions** - `ConvBlock`, `HostCNN`, `MorphogeneticModel` |
| `kasmina.py` | 860 | Seed lifecycle management |
| `tamiyo.py` | 634 | Strategic controller (heuristic) |
| `telemetry.py` | 501 | Training diagnostics |
| `telemetry_config.py` | 237 | Config for telemetry |

### ACTIVE - Core IQL System

| File | Lines | Role |
|------|-------|------|
| `simic_iql.py` | 1,325 | Entry point - offline IQL/CQL training |
| `simic_train.py` | 414 | Feature extraction (`obs_to_base_features`) |

### SHARED - Used by Both

| File | Lines | Role |
|------|-------|------|
| `simic.py` | 1,024 | SimicAction enum, data structures |
| `leyline.py` | 791 | Contracts (SeedStage, etc.) |
| `tamiyo.py` | 634 | Heuristic (used by both for comparison) |

## Broken Imports in simic_ppo.py

| Line | Import | Status |
|------|--------|--------|
| 1678 | `from esper.simic_cifar10 import SimicCIFAR10, get_cifar10_loaders` | **MISSING FILE** |
| 1804 | `from esper.simic_cifar10 import SimicCIFAR10, get_cifar10_loaders` | **MISSING FILE** |
| 1680 | `from esper.telemetry import TelemetryTracker` | **WRONG NAME** - should be `DiagnosticTracker` |
| 1806 | `from esper.telemetry import TelemetryTracker` | **WRONG NAME** - should be `DiagnosticTracker` |
| 1920 | `from esper.tamiyo import TamiyoSignals` | **WRONG NAME** - should be `TrainingSignals` |

These broken imports are in functions that run with `--compare` flag, which explains why normal training still works.

## Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           TRAINING ENTRY POINTS          │
                    ├─────────────────┬───────────────────────┤
                    │  simic_ppo.py   │     simic_iql.py      │
                    │  (online PPO)   │    (offline IQL)      │
                    └────────┬────────┴──────────┬────────────┘
                             │                   │
              ┌──────────────┼───────────────────┼──────────────┐
              │              │                   │              │
              ▼              ▼                   ▼              │
    ┌─────────────────┐ ┌─────────┐      ┌─────────────┐       │
    │simic_overnight.py│ │ simic.py│      │simic_train.py│       │
    │(model factory)  │ │(actions)│      │(features)   │       │
    └────────┬────────┘ └────┬────┘      └──────┬──────┘       │
             │               │                  │              │
    ┌────────┴────────┐      │                  │              │
    │                 │      │                  │              │
    ▼                 ▼      ▼                  ▼              │
┌──────────┐   ┌──────────┐ ┌──────────┐                       │
│poc_tamiyo│   │ kasmina  │ │ leyline  │◄──────────────────────┘
│(CNN defs)│   │(seeds)   │ │(contracts)│
└──────────┘   └──────────┘ └──────────┘
                    │
                    ▼
              ┌──────────┐
              │ tamiyo   │
              │(heuristic)│
              └──────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ORPHANED (no imports)                     │
├────────────┬───────────────────────┬────────────────────────┤
│  poc.py    │ simic_test_collection │      rewards.py        │
│  (demo)    │       (test)          │    (unused module)     │
└────────────┴───────────────────────┴────────────────────────┘
```

## Naming Issues

### Misnamed Files
1. **`poc_tamiyo.py`** - Name suggests POC but contains **production model classes**
   - `ConvBlock`, `HostCNN`, `MorphogeneticModel` are used by `simic_overnight.py`
   - Should be renamed to `models.py` or similar

2. **`simic_overnight.py`** - Name suggests "overnight runner" but actually provides:
   - `create_model()` factory function used by `simic_ppo.py`
   - `load_cifar10()` data loading used by `simic_ppo.py`
   - Should be renamed to `simic_env.py` or `simic_utils.py`

### Redundant Definitions
- `TrainingSignals` defined in both `tamiyo.py:41` and `leyline.py:399`

## Recommended Actions

### Phase 1: Archive Orphaned Files (Safe - No Breaking Changes)

```bash
mkdir -p _archive/poc
git mv src/esper/poc.py _archive/poc/
git mv src/esper/simic_test_collection.py _archive/poc/
git mv src/esper/rewards.py _archive/poc/  # Recently created but never integrated
```

### Phase 2: Fix Broken Imports in simic_ppo.py

```python
# Line 1678, 1804: simic_cifar10 doesn't exist
# Option A: Create simic_cifar10.py by extracting from simic_overnight.py
# Option B: Change import to: from esper.simic_overnight import load_cifar10

# Line 1680, 1806: Wrong class name
from esper.telemetry import DiagnosticTracker  # not TelemetryTracker

# Line 1920: Wrong class name
from esper.tamiyo import TrainingSignals  # not TamiyoSignals
```

### Phase 3: Rename Confusing Files (Requires Import Updates)

| Current | Suggested | Reason |
|---------|-----------|--------|
| `poc_tamiyo.py` | `models.py` | Contains production CNN classes |
| `simic_overnight.py` | `simic_env.py` | Provides environment/model utilities |

This would require updating imports in `simic_ppo.py` lines 43, 798, 1041, 1226.

## Summary

**Your instinct was correct:**
- `poc.py` is indeed orphaned and can be archived
- `poc_tamiyo.py` is **NOT orphaned** - it's a misnomer, contains production code
- `simic_overnight.py` is **NOT obsolete** - `simic_ppo.py` depends on it for model creation
- `simic_train.py` is **NOT obsolete** - `simic_iql.py` depends on it for feature extraction
- `rewards.py` was recently created but never actually integrated (orphaned)
