# Architect Handover Briefing - esper-lite

**Analysis Date:** 2025-12-13
**Purpose:** Actionable briefing for architects taking ownership of the codebase
**Prerequisite Reading:** 04-final-report.md (10 min), 07-expert-review-findings.md (15 min)

---

## Quick Orientation (5 minutes)

### What is esper-lite?

A **morphogenetic neural network training system** that:
1. Grafts modular "seeds" (network configurations) onto a host model
2. Uses RL (PPO) or heuristics to decide when to germinate, train, blend, and fossilize seeds
3. Implements quality gates (G0-G5) for seed lifecycle progression

### Why does it exist?

To enable **automated neural architecture adaptation** where an RL agent learns optimal strategies for integrating modular components into a base model during training.

### Where does it run?

- **Current:** Single NVIDIA GPU with PyTorch 2.8+
- **Target:** Multi-GPU clusters (DDP support needed)

---

## Architecture at a Glance

```
Scripts (CLI) ─────────────────────────────────────────────────┐
    │                                                          │
    ├─► Simic (RL Infrastructure, 8.3K LOC) ◄──────────────────┤
    │       │                                                  │
    │       ├─► Tolaria (Training Engine, 700 LOC)             │
    │       ├─► Tamiyo (Decision Policy, 630 LOC)              │
    │       ├─► Kasmina (Seed Lifecycle, 2.9K LOC)             │
    │       ├─► Nissa (Telemetry, 1.6K LOC)                    │
    │       ├─► Runtime (Task Presets, 230 LOC)                │
    │       └─► Utils (Data/Loss, 570 LOC)                     │
    │                                                          │
    └─► Leyline (Data Contracts, 1.2K LOC) ◄───────────────────┘
```

**Key insight:** Dependencies flow DOWN. Leyline has none. Simic has most.

---

## Your First Day Tasks

### 1. Run the system (15 min)

```bash
# Heuristic training (no RL)
python -m esper.scripts.train heuristic --task cifar10 --epochs 5

# PPO training (RL-based decisions)
python -m esper.scripts.train ppo --task cifar10 --episodes 100 --n-envs 4
```

### 2. Read the critical files (30 min)

| Priority | File | What to understand |
|----------|------|-------------------|
| 1 | `leyline/stages.py` | Seed lifecycle states (11 stages) |
| 2 | `leyline/factored_actions.py` | How the RL agent acts (4 heads) |
| 3 | `kasmina/slot.py:1-100` | SeedSlot class overview |
| 4 | `simic/rewards.py:1-150` | Reward computation overview |
| 5 | `simic/ppo.py:400-500` | PPO loss computation |

### 3. Understand the seed lifecycle (20 min)

```
DORMANT → GERMINATED → TRAINING → BLENDING → PROBATIONARY → FOSSILIZED
   │           │           │          │            │
   └──────────────────────────────────────────────────► CULLED (failure)
```

Quality gates (G0-G5) control advancement. See `kasmina/slot.py` for gate logic.

---

## Critical Issues to Address

### MUST FIX (Before any production use)

#### Issue 1: Missing Imports
**Location:** `src/esper/simic/training.py:193-197`
**Problem:** BLUEPRINT_IDS, BLEND_IDS, SLOT_IDS referenced but not imported
**Impact:** Runtime NameError
**Fix:**
```python
from esper.leyline.factored_actions import BLUEPRINT_IDS, BLEND_IDS, SLOT_IDS
```

#### Issue 2: Unbounded Observation
**Location:** `src/esper/leyline/signals.py:152`
**Problem:** seed_counterfactual not clamped
**Impact:** Observation space violation, potential training instability
**Fix:**
```python
seed_counterfactual=max(-10.0, min(10.0, seed_counterfactual)) / 10.0
```

### SHOULD FIX (Before scaling)

#### Issue 3: Global Mutable State
**Location:** `src/esper/simic/training.py:30-31`
**Problem:** USE_COMPILED_TRAIN_STEP is global, breaks DDP
**Impact:** Cannot scale to multi-GPU
**Fix:** Convert to per-instance configuration on PPOAgent or TrainingConfig

#### Issue 4: No AMP Support
**Location:** `src/esper/simic/ppo.py`, `vectorized.py`
**Problem:** FP32 only
**Impact:** Missing 30-50% speedup
**Fix:** Wrap training in `torch.amp.autocast('cuda')`

---

## Key Design Decisions (Why things are this way)

### Q: Why factored actions instead of flat?
**A:** Combinatorial explosion. With 4 slots × 7 blueprints × 3 blend modes × 5 ops, flat action space = 420 actions. Factored = 19 outputs with independent heads.

### Q: Why PBRS reward shaping?
**A:** Policy invariance. Ng et al. (1999) showed potential-based shaping doesn't change optimal policy but speeds learning.

### Q: Why counterfactual validation?
**A:** Ransomware prevention. Seeds could learn to temporarily harm the host, then "improve" it - appearing beneficial. Counterfactual comparison (alpha=0 baseline) catches this.

### Q: Why quality gates instead of learned advancement?
**A:** Determinism. Gates ensure seeds meet measurable criteria. RL decides *when* to attempt advancement, gates decide *if* it succeeds.

### Q: Why LSTM in the policy?
**A:** Temporal credit assignment. Seed decisions have delayed effects. LSTM enables remembering earlier states that caused current outcomes.

---

## Code Navigation Guide

### "Where do I find..."

| Feature | Location |
|---------|----------|
| PPO algorithm | `simic/ppo.py:PPOAgent` |
| Reward computation | `simic/rewards.py:RewardComputer` |
| Seed state machine | `kasmina/slot.py:SeedSlot` |
| Quality gates | `kasmina/slot.py:_check_gate_*` methods |
| Feature extraction | `simic/features.py:obs_to_base_features` |
| Training loop | `simic/vectorized.py:train_ppo_vectorized` |
| Heuristic policy | `tamiyo/heuristic.py:HeuristicTamiyo` |
| Telemetry config | `nissa/config.py:TelemetryConfig` |
| Task definitions | `runtime/tasks.py:get_task_spec` |

### "Where do I add..."

| New feature | Modify these files |
|-------------|-------------------|
| New seed blueprint | `kasmina/blueprints/cnn.py` or `transformer.py` + registry |
| New reward component | `simic/rewards.py:RewardComputer._compute_*` |
| New quality gate | `kasmina/slot.py:_check_gate_*` + stages.py |
| New observation feature | `leyline/signals.py` + `simic/features.py` |
| New telemetry event | `leyline/telemetry.py` + emit in appropriate subsystem |
| New task preset | `runtime/tasks.py:_*_task()` factory |

---

## Testing Strategy

### Run tests
```bash
pytest tests/ -v                      # All tests
pytest tests/simic/ -v                # RL infrastructure
pytest tests/properties/ -v           # Property-based tests
pytest tests/integration/ -v          # End-to-end
```

### Test categories

| Category | What it tests | Run when |
|----------|--------------|----------|
| Unit (`tests/<subsystem>/`) | Individual components | Every change |
| Property (`tests/properties/`) | Invariants via Hypothesis | Before merge |
| Integration (`tests/integration/`) | Multi-subsystem flows | Before release |

### Coverage gaps
- DDP scenarios (no multi-GPU test infrastructure)
- Some reward edge cases
- Memory pressure scenarios

---

## Performance Characteristics

### Current throughput (single A100)
- ~1000 RL steps/second with 4 environments
- ~50 training epochs/hour for CIFAR-10

### Bottlenecks
1. **Data loading** - SharedBatchIterator helps, but GPU cache is better
2. **Policy inference** - CUDA graphs help via torch.compile
3. **Reward computation** - Counterfactual validation adds overhead

### Optimization opportunities
| Optimization | Expected gain | Effort |
|--------------|---------------|--------|
| AMP (FP16) | 30-50% | Low |
| More CUDA graphs | 10-20% | Medium |
| DDP | Linear scaling | High |

---

## Telemetry & Debugging

### Telemetry levels
```python
TelemetryLevel.MINIMAL   # Epoch summaries only
TelemetryLevel.NORMAL    # Standard metrics (default)
TelemetryLevel.DEBUG     # Everything including per-step
```

### Auto-escalation
Nissa automatically escalates to DEBUG when anomalies detected:
- NaN/Inf gradients
- Loss spikes
- Statistical outliers

### Key telemetry events
- `EPISODE_START/END` - RL episode boundaries
- `SEED_STAGE_CHANGE` - Lifecycle transitions
- `GATE_RESULT` - Quality gate outcomes
- `REWARD_COMPUTED` - Detailed reward breakdown

---

## Dependency Management

### External dependencies
```
pytorch >= 2.8.0        # Core framework
transformers >= 4.57    # Transformer architectures (optional)
datasets >= 4.4         # Data loading (optional)
numpy >= 1.24           # Numerics
pydantic >= 2.0         # Config validation
pyyaml >= 6.0           # Config files
hypothesis >= 6.148     # Property testing
pytest >= 7.0           # Testing
```

### Upgrade considerations
- **PyTorch:** torch.compile API stability, check for breaking changes
- **Transformers:** GPT architecture compatibility
- **Private APIs:** `torch._foreach_norm` in isolation.py may break

---

## Common Pitfalls

### Pitfall 1: Modifying Leyline contracts
**Problem:** All subsystems depend on Leyline
**Solution:** Changes require coordinated updates across entire codebase

### Pitfall 2: Adding global state
**Problem:** Blocks DDP
**Solution:** Use config dataclasses or per-instance attributes

### Pitfall 3: Forgetting non_blocking
**Problem:** CUDA sync overhead
**Solution:** Always use `tensor.to(device, non_blocking=True)` in hot paths

### Pitfall 4: Breaking acyclic dependencies
**Problem:** Circular imports, harder testing
**Solution:** Use TYPE_CHECKING for type hints, events for communication

### Pitfall 5: Large files
**Problem:** slot.py and vectorized.py are already too large
**Solution:** Extract focused modules when adding features

---

## Recommended Reading Order

1. **This document** (you are here)
2. **07-expert-review-findings.md** - Detailed issue analysis
3. **02-subsystem-catalog.md** - Deep dive into each subsystem
4. **03-diagrams.md** - Visual architecture reference
5. **05-quality-assessment.md** - Code quality metrics

---

## Quick Reference Card

### Commands
```bash
# Training
python -m esper.scripts.train heuristic --task cifar10
python -m esper.scripts.train ppo --task cifar10 --n-envs 4

# Testing
pytest tests/ -v
pytest tests/simic/ -v -k "test_ppo"

# Evaluation
python -m esper.scripts.evaluate --model checkpoints/ppo.pt --task cifar10
```

### Key locations
```
src/esper/
├── leyline/     # Contracts (start here for data structures)
├── kasmina/     # Seeds (slot.py is the core)
├── simic/       # RL (ppo.py, rewards.py, vectorized.py)
├── tamiyo/      # Decisions (heuristic.py)
├── tolaria/     # Training (trainer.py)
├── nissa/       # Telemetry (tracker.py, output.py)
├── runtime/     # Tasks (tasks.py)
├── utils/       # Data (data.py)
└── scripts/     # CLI (train.py, evaluate.py)
```

### Contact points
- **Algorithm questions:** simic/ppo.py, simic/rewards.py
- **Lifecycle questions:** kasmina/slot.py, leyline/stages.py
- **Performance questions:** simic/vectorized.py, utils/data.py
- **Telemetry questions:** nissa/tracker.py, nissa/config.py

---

## Handover Checklist

- [ ] Run both training modes (heuristic, ppo)
- [ ] Read critical files listed above
- [ ] Understand seed lifecycle state machine
- [ ] Review critical issues (missing imports, global state)
- [ ] Run test suite
- [ ] Review telemetry output formats
- [ ] Understand dependency graph

**Estimated onboarding time:** 4-6 hours for comfortable navigation
