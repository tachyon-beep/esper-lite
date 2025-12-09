# Discovery Findings: esper-lite

## Project Overview

**Name:** esper-lite
**Description:** Morphogenetic Neural Networks framework - neural networks that dynamically grow, prune, and adapt their topology during training
**Runtime:** Python 3.13, PyTorch 2.9

## Core Concept

Esper implements a "seed grafting" paradigm where neural modules ("seeds") are:
1. **Germinated** in isolation from the host network
2. **Trained** on host errors without destabilizing existing knowledge
3. **Blended** gradually into the host via alpha-scheduled integration
4. **Fossilized** permanently when they prove their worth

This addresses catastrophic forgetting by treating neural growth as a lifecycle-managed process.

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Source files | 59 Python files |
| Lines of code | ~16,500 LOC |
| Test files | 73 files |
| Subsystems | 9 (7 core + 2 support) |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| Deep Learning | PyTorch 2.9 |
| RL Algorithms | PPO (custom implementation) |
| Testing | pytest, hypothesis |
| Data | torchvision, HuggingFace datasets/transformers |

## Architectural Organization

The system follows a **domain-driven design** with 7 core subsystems organized by responsibility:

```
                    ┌─────────────┐
                    │   Scripts   │  CLI Entry Points
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Runtime │  │  Simic  │  │  Nissa  │
        │ (Tasks) │  │  (RL)   │  │(Telemetry)
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                    ┌─────────┐
                    │ Tamiyo  │  Strategic Decisions
                    │ (Brain) │
                    └────┬────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌─────────┐           ┌─────────┐
        │ Tolaria │           │ Kasmina │
        │(Trainer)│           │ (Model) │
        └────┬────┘           └────┬────┘
             │                     │
             └──────────┬──────────┘
                        ▼
                  ┌─────────┐
                  │ Leyline │  Shared Contracts
                  │(Schemas)│
                  └─────────┘
```

## Subsystem Summary

| Subsystem | Role | Analogy | LOC | Files |
|-----------|------|---------|-----|-------|
| **Kasmina** | Body/Model | The Plant | ~5,000 | 9 |
| **Leyline** | Nervous System | Signals | ~1,500 | 7 |
| **Tamiyo** | Brain | The Gardener | ~1,000 | 4 |
| **Tolaria** | Hands | Tools | ~700 | 4 |
| **Simic** | Gym | Simulator | ~6,000 | 21 |
| **Nissa** | Senses | Sensors | ~1,500 | 5 |
| **Scripts** | CLI | Entry Points | ~500 | 3 |
| **Runtime** | Infrastructure | Config | ~300 | 2 |
| **Utils** | Utilities | Helpers | ~500 | 2 |

## Key Architectural Patterns

### 1. Lifecycle State Machine
Seeds progress through defined stages: DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED (with failure paths to CULLED/EMBARGOED)

### 2. Quality Gates
Six gate levels (G0-G5) enforce stage transitions:
- G0: Germination sanity
- G1: Training readiness
- G2: Blending readiness (improvement thresholds)
- G3: Shadowing readiness
- G4: Probation readiness
- G5: Fossilization (counterfactual validation)

### 3. Gradient Isolation
- **STE (Straight-Through Estimator)** for training stage
- **Alpha blending** for gradual integration
- **Gradient monitoring** to detect isolation violations

### 4. Plugin Architecture
- **BlueprintRegistry** for extensible seed implementations
- CNN blueprints: norm, attention, depthwise, conv_light, conv_heavy
- Transformer blueprints: norm, LoRA, attention, MLP, FlexAttention

### 5. Telemetry-First Design
- Central NissaHub routes events to multiple backends
- Adaptive telemetry levels (OFF/MINIMAL/NORMAL/DEBUG)
- Anomaly detection triggers automatic escalation

### 6. Configuration-Driven Tasks
- TaskSpec bundles model factories, dataloaders, action enums
- Presets: CIFAR-10 (CNN), TinyStories (Transformer)

## Dependency Structure

### Inward Dependencies (Most Depended Upon)
1. **Leyline** - Foundation layer, imported by ALL other modules
2. **Kasmina** - Core model layer, imported by runtime, simic, tamiyo
3. **Nissa** - Telemetry hub, imported by simic, tamiyo, scripts

### Outward Dependencies (Most Dependent)
1. **Scripts** - Entry points, imports most modules
2. **Simic** - RL infrastructure, heavy imports from leyline, nissa
3. **Runtime** - Task config, imports kasmina, leyline, simic, utils

### Circular Dependency Avoidance
- TYPE_CHECKING imports used for type hints only
- Lazy imports in leyline.actions for BlueprintRegistry
- Local imports in runtime.tasks to break cycles

## Entry Points

| Command | Description |
|---------|-------------|
| `python -m esper.scripts.train ppo` | Train PPO agent |
| `python -m esper.scripts.train heuristic` | Run heuristic baseline |
| `python -m esper.scripts.evaluate` | Diagnostic evaluation |

## Notable Technical Details

1. **torch.compile aware** - SeedSlot.forward() disabled from compilation due to control flow
2. **CUDA streams** - Vectorized training uses async execution for multi-GPU
3. **Zero-GC feature path** - FastTrainingSignals uses NamedTuple for hot-path PPO
4. **Counterfactual validation** - Alpha=0 baseline for true causal attribution
5. **Potential-based reward shaping** - PBRS for policy-invariant rewards

## Confidence Assessment

| Subsystem | Confidence | Reasoning |
|-----------|------------|-----------|
| Kasmina | HIGH | Comprehensive analysis, well-documented |
| Leyline | HIGH | Clear contracts, explicit exports |
| Tamiyo | HIGH | Clean separation, documented patterns |
| Tolaria | HIGH | Small footprint, clear purpose |
| Simic | HIGH | Extensive but well-organized (21 modules) |
| Nissa | HIGH | Event-driven, clear boundaries |
| Scripts | HIGH | Entry points only, minimal logic |
| Runtime | HIGH | Configuration registry, straightforward |
| Utils | MEDIUM | May grow, currently data-focused |

## Identified Risks/Concerns

1. **Simic complexity** - 21 modules may benefit from internal documentation
2. **Utils growth** - Currently a "bit bucket", may need structure
3. **Implicit coupling** - Tolaria expects `seed_slot` attribute on models (structural coupling)

## Analysis Metadata

- **Analysis date:** 2025-12-09
- **Workspace:** docs/arch-analysis-2025-12-09-0521/
- **Method:** Parallel subagent exploration (7 agents)
- **Total files analyzed:** 59 source files
