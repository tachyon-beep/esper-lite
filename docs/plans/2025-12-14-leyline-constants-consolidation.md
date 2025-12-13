# Plan: Leyline Constants Consolidation

**Status:** COMPLETED
**Created:** 2025-12-14
**Completed:** 2025-12-14
**Constants Moved:** 40 constants across 6 phases

## Problem Statement

The codebase has ~100 hardcoded numeric constants scattered across modules. Many are:
1. **Duplicated** (e.g., `max_epochs=25` in 5 places with sync requirements)
2. **Not CLI-accessible** for experimentation
3. **Inconsistent** (e.g., `gamma` was 0.99 vs 0.995 until today's fix)

## Solution

Consolidate hyperparameters into `leyline/__init__.py` following the existing pattern (`MIN_CULL_AGE`, `DEFAULT_GAMMA`), then wire to CLI.

---

## Phase 1: Critical Synchronization Constants

**Goal:** Eliminate multi-location sync bugs like the gamma mismatch.

### Constants to Add to leyline/__init__.py

```python
# =============================================================================
# Episode & Architecture Constants
# =============================================================================

# Episode length - MUST sync with chunk_length and max_steps_per_env
# Used by: config.py, vectorized.py, ppo.py (x2), tamiyo_buffer.py
DEFAULT_EPISODE_LENGTH = 25

# LSTM hidden dimension - architecture constant
# Used by: config.py, vectorized.py, ppo.py, tamiyo_buffer.py, tamiyo_network.py
DEFAULT_LSTM_HIDDEN_DIM = 128

# Parallel environments for vectorized training
# Used by: config.py, vectorized.py, ppo.py
DEFAULT_N_ENVS = 4
```

### Files to Update

| File | Change |
|------|--------|
| `leyline/__init__.py` | Add constants + `__all__` exports |
| `simic/config.py` | Import & use `DEFAULT_EPISODE_LENGTH`, `DEFAULT_LSTM_HIDDEN_DIM`, `DEFAULT_N_ENVS` |
| `simic/vectorized.py` | Import & use all three |
| `simic/ppo.py` | Import & use `DEFAULT_LSTM_HIDDEN_DIM`, add assertion `chunk_length == max_epochs` |
| `simic/tamiyo_buffer.py` | Import & use `DEFAULT_LSTM_HIDDEN_DIM` |
| `scripts/train.py` | Import for CLI defaults |

### Validation

- [ ] `PYTHONPATH=src python -c "from esper.leyline import DEFAULT_EPISODE_LENGTH, DEFAULT_LSTM_HIDDEN_DIM, DEFAULT_N_ENVS; print('OK')"`
- [ ] Run `pytest tests/simic/ -x` to verify no regressions

---

## Phase 2: PPO Hyperparameters

**Goal:** CLI-accessible PPO tuning knobs.

### Constants to Add

```python
# =============================================================================
# PPO Hyperparameters
# =============================================================================

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_CLIP_RATIO = 0.2
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_VALUE_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_N_PPO_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_ENTROPY_COEF = 0.05
DEFAULT_ENTROPY_COEF_MIN = 0.01
```

### Files to Update

| File | Change |
|------|--------|
| `leyline/__init__.py` | Add constants |
| `simic/ppo.py` | Import all, update constructor defaults |
| `simic/config.py` | Import all, update dataclass defaults |
| `simic/vectorized.py` | Import entropy constants |
| `scripts/train.py` | Wire to argparse (already has some, ensure consistency) |

### CLI Additions

```bash
# New/updated flags
--lr              (already exists, ensure uses DEFAULT_LEARNING_RATE)
--clip-ratio      (already exists)
--gae-lambda      (ADD)
--value-coef      (ADD)
--max-grad-norm   (ADD)
--n-ppo-epochs    (ADD)
--batch-size      (ADD)
```

---

## Phase 3: Reward & Gate Constants

**Goal:** Tuneable reward shaping and lifecycle gates.

### Constants to Add

```python
# =============================================================================
# Reward Shaping Constants
# =============================================================================

DEFAULT_CONTRIBUTION_WEIGHT = 1.0
DEFAULT_PBRS_WEIGHT = 0.3
DEFAULT_RENT_WEIGHT = 0.5
DEFAULT_FOSSILIZE_TERMINAL_SCALE = 3.0

# =============================================================================
# Lifecycle Gate Thresholds
# =============================================================================

DEFAULT_MIN_FOSSILIZE_CONTRIBUTION = 1.0  # 1% causal contribution for G5
DEFAULT_GRADIENT_RATIO_THRESHOLD = 0.05   # G2 seed activity
DEFAULT_MIN_PROBATION_STABILITY = 0.95    # G3 stability
DEFAULT_GRADIENT_EMA_DECAY = 0.9
```

### Files to Update

| File | Change |
|------|--------|
| `leyline/__init__.py` | Add constants |
| `simic/rewards.py` | Import from leyline |
| `kasmina/slot.py` | Import from leyline (removes 6 module-level constants) |
| `scripts/train.py` | Add `--contribution-weight`, `--fossilize-terminal-scale` |

---

## Phase 4: Stabilization & Governor

**Goal:** Task-adaptable stabilization and safety thresholds.

### Constants to Add

```python
# =============================================================================
# Host Stabilization
# =============================================================================

DEFAULT_STABILIZATION_THRESHOLD = 0.03    # 3% relative improvement
DEFAULT_STABILIZATION_EPOCHS = 3          # Consecutive stable epochs

# =============================================================================
# Governor (Tolaria)
# =============================================================================

DEFAULT_GOVERNOR_SENSITIVITY = 6.0
DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD = 12.0  # Task-dependent
DEFAULT_GOVERNOR_DEATH_PENALTY = 10.0
DEFAULT_GOVERNOR_HISTORY_WINDOW = 20
DEFAULT_MIN_PANICS_BEFORE_ROLLBACK = 2
```

### Files to Update

| File | Change |
|------|--------|
| `leyline/__init__.py` | Add constants |
| `simic/config.py` | Import stabilization & governor |
| `tolaria/governor.py` | Import governor constants |
| `tamiyo/tracker.py` | Import stabilization constants |
| `scripts/train.py` | Add `--stabilization-threshold`, `--governor-sensitivity` |

---

## Phase 5: Heuristic Policy

**Goal:** Tuneable heuristic for baseline comparisons.

### Constants to Add

```python
# =============================================================================
# Heuristic Policy (Tamiyo)
# =============================================================================

DEFAULT_PLATEAU_EPOCHS_TO_GERMINATE = 3
DEFAULT_MIN_EPOCHS_BEFORE_GERMINATE = 5
DEFAULT_CULL_AFTER_EPOCHS_WITHOUT_IMPROVEMENT = 5
DEFAULT_CULL_IF_ACCURACY_DROPS_BY = 2.0
DEFAULT_EMBARGO_EPOCHS_AFTER_CULL = 5
DEFAULT_BLENDING_TOTAL_STEPS = 5
```

### Files to Update

| File | Change |
|------|--------|
| `leyline/__init__.py` | Add constants |
| `tamiyo/heuristic.py` | Import for `HeuristicPolicyConfig` defaults |
| `kasmina/blending.py` | Import `DEFAULT_BLENDING_TOTAL_STEPS` |
| `scripts/train.py` | Add heuristic flags for `heuristic` subcommand |

---

## Phase 6: Task-Specific Learning Rates

**Goal:** Separate host/seed learning rates with CLI control.

### Constants to Add

```python
# =============================================================================
# Task Training Defaults
# =============================================================================

DEFAULT_HOST_LR = 0.01
DEFAULT_SEED_LR = 0.01
```

### Files to Update

| File | Change |
|------|--------|
| `leyline/__init__.py` | Add constants |
| `runtime/tasks.py` | Import for `TaskSpec` defaults |
| `scripts/train.py` | Add `--host-lr`, `--seed-lr` |

---

## Implementation Order

1. **Phase 1** first - prevents future sync bugs
2. **Phase 2** next - most frequently tuned
3. **Phases 3-6** can be done in parallel or incrementally

## Validation Strategy

After each phase:
1. Run `pytest tests/` to catch import/default changes
2. Run a quick training smoke test: `PYTHONPATH=src python -m esper.scripts.train ppo --episodes 2 --n-envs 2`
3. Verify CLI `--help` shows new flags

## NOT Moving (Documented Decision)

The following categories are **intentionally NOT moved** to leyline:

| Category | Examples | Rationale |
|----------|----------|-----------|
| Domain constants | `vocab_size=50257`, `num_classes=10` | Task-defined, not hyperparameters |
| Numerical stability | `eps=1e-8`, `clip=10.0` | Implementation details |
| UI/Display | TUI heights, bar widths | Not behavioral |
| Algorithm internals | Sigmoid scaling, SE reduction | Algorithm design choices |
| WebSocket/Server | Timeouts, ping intervals | Infrastructure tuning |

---

## Success Criteria

- [ ] All 38 constants defined in `leyline/__init__.py`
- [ ] All importing modules use leyline constants (no hardcoded duplicates)
- [ ] CLI `--help` shows new flags for Tier 1-2 constants
- [ ] No test regressions
- [ ] Bible updated to reflect new constant locations
