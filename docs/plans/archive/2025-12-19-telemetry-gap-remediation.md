# Telemetry Gap Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the 67-field gap between telemetry emission and aggregator capture in Sanctum.

**Discovery:** Systematic analysis revealed 47% of emitted telemetry fields are not being captured by the Sanctum aggregator, resulting in significant observability loss.

**Architecture:** The Sanctum aggregator (`src/esper/karn/sanctum/aggregator.py`) receives telemetry events from the Nissa hub but only extracts a subset of fields. This plan prioritizes gaps by impact on operator visibility.

---

## Executive Summary

| Event Type | Fields Emitted | Fields Captured | Gap | Priority |
|------------|----------------|-----------------|-----|----------|
| TRAINING_STARTED | 27 | 10 | 17 | P2 |
| EPOCH_COMPLETED (per-env) | 7 | 7 | 0 | Done |
| EPOCH_COMPLETED (aggregate) | 9 | 0 | 9 | P3 |
| REWARD_COMPUTED | 21 | 20 | 1 | P3 |
| PPO_UPDATE_COMPLETED | 27 | 14 | 13 | P1 |
| BATCH_EPOCH_COMPLETED | 9 | 1 | 8 | P1 |
| SEED_GERMINATED | 9 | 9 | 0 | Done |
| SEED_STAGE_CHANGED | 12 | 10 | 2 | P3 |
| SEED_FOSSILIZED | 10 | 2 | 8 | P2 |
| SEED_CULLED | 11 | 2 | 9 | P2 |
| **TOTAL** | **142** | **75** | **67** | - |

---

## Priority Definitions

- **P1 (Critical):** Directly impacts real-time training visibility. Fix immediately.
- **P2 (High):** Important for diagnostics and understanding training behavior. Fix soon.
- **P3 (Low):** Nice-to-have context. Fix when convenient.

---

## Detailed Gap Analysis

### BATCH_EPOCH_COMPLETED (P1 - 89% uncaptured)

**Current capture:** Only `episodes_completed`

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | batch_idx | ✓ | ✗ | P1 | Batch progress tracking |
| 2 | total_episodes | ✓ | ✗ | P1 | Progress denominator |
| 3 | start_episode | ✓ | ✗ | P3 | Batch context |
| 4 | requested_episodes | ✓ | ✗ | P3 | Config context |
| 5 | env_accuracies | ✓ | ✗ | P1 | Per-env batch summary |
| 6 | avg_accuracy | ✓ | ✗ | P1 | Aggregate batch accuracy |
| 7 | rolling_accuracy | ✓ | ✗ | P1 | Trend indicator |
| 8 | avg_reward | ✓ | ✗ | P1 | Aggregate batch reward |

**Recommended action:** Capture batch_idx, avg_accuracy, rolling_accuracy, avg_reward for throughput display and trend tracking.

---

### PPO_UPDATE_COMPLETED (P1 - 48% uncaptured)

**Current capture:** Policy/value loss, entropy, KL, clip fraction, ratio stats, gradient health flags, LR

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | inner_epoch | ✓ | ✗ | P2 | Training progress context |
| 2 | batch | ✓ | ✗ | P2 | Batch context |
| 3 | train_steps | ✓ | ✗ | P3 | Debugging |
| 4 | early_stop_epoch | ✓ | ✗ | P2 | KL early stopping indicator |
| 5 | avg_accuracy | ✓ | ✗ | P2 | Batch average (redundant with BATCH_EPOCH_COMPLETED) |
| 6 | avg_reward | ✓ | ✗ | P2 | Batch average (redundant with BATCH_EPOCH_COMPLETED) |
| 7 | rolling_avg_accuracy | ✓ | ✗ | P2 | Trend (redundant with BATCH_EPOCH_COMPLETED) |
| 8 | grad_norm | ✓ | ✗ | **P1** | **Critical gradient health metric** |
| 9 | update_time_ms | ✓ | ✗ | P1 | Training speed monitoring |
| 10 | head_slot_entropy | ✓ | ✗ | P2 | Per-head entropy (slot action head) |
| 11 | head_slot_grad_norm | ✓ | ✗ | P2 | Per-head gradient health |
| 12 | head_blueprint_entropy | ✓ | ✗ | P2 | Per-head entropy (blueprint head) |
| 13 | head_blueprint_grad_norm | ✓ | ✗ | P2 | Per-head gradient health |

**Recommended action:** Capture grad_norm (critical), update_time_ms (throughput), and early_stop_epoch (KL monitoring).

---

### TRAINING_STARTED (P2 - 63% uncaptured)

**Current capture:** run_id, task, max_epochs, n_envs, policy_device, env_devices, host_params, slot_ids

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | seed | ✓ | ✗ | P2 | Reproducibility tracking |
| 2 | topology | ✓ | ✗ | P3 | Task architecture |
| 3 | task_type | ✓ | ✗ | P3 | Classification vs other |
| 4 | reward_mode | ✓ | ✗ | **P1** | **A/B test cohort identification** |
| 5 | n_episodes | ✓ | ✗ | P2 | Training duration |
| 6 | use_telemetry | ✓ | ✗ | P3 | Debug flag |
| 7 | state_dim | ✓ | ✗ | P3 | Observation space size |
| 8 | env_device_counts | ✓ | ✗ | P3 | Multi-GPU distribution |
| 9 | env_device_map_strategy | ✓ | ✗ | P3 | Distribution strategy |
| 10 | resume_path | ✓ | ✗ | P2 | Checkpoint resume tracking |
| 11 | lr | ✓ | ✗ | P2 | Initial learning rate |
| 12 | clip_ratio | ✓ | ✗ | P2 | PPO hyperparameter |
| 13 | entropy_coef | ✓ | ✗ | P2 | Initial entropy coefficient |
| 14 | entropy_anneal | ✓ | ✗ | P2 | Entropy schedule config |
| 15 | gpu_preload | ✓ | ✗ | P3 | Memory optimization flag |
| 16 | dataloader | ✓ | ✗ | P3 | Dataloader config |
| 17 | param_budget | ✓ | ✗ | P2 | Seed parameter budget |

**Recommended action:** Capture reward_mode (required for A/B cohort display), seed (reproducibility), n_episodes (progress tracking).

---

### SEED_FOSSILIZED (P2 - 80% uncaptured)

**Current capture:** slot_id, env_id, params_added

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | blueprint_id | ✓ | ✗ | P2 | Which blueprint succeeded |
| 2 | seed_id | ✓ | ✗ | P3 | Unique seed identifier |
| 3 | improvement | ✓ | ✗ | **P1** | **Accuracy improvement from seed** |
| 4 | blending_delta | ✓ | ✗ | P2 | Contribution during blending |
| 5 | counterfactual | ✓ | ✗ | P2 | Causal attribution score |
| 6 | epochs_total | ✓ | ✗ | P2 | Time to fossilize |
| 7 | epochs_in_stage | ✓ | ✗ | P3 | Time in final stage |

**Recommended action:** Capture improvement (explains fossilization value), blueprint_id (which blueprints work), epochs_total (fossilization speed).

---

### SEED_CULLED (P2 - 82% uncaptured)

**Current capture:** slot_id, env_id

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | reason | ✓ | ✗ | **P1** | **Why was seed culled** |
| 2 | auto_culled | ✓ | ✗ | P2 | Policy vs automatic cull |
| 3 | blueprint_id | ✓ | ✗ | P2 | Which blueprint failed |
| 4 | seed_id | ✓ | ✗ | P3 | Unique seed identifier |
| 5 | improvement | ✓ | ✗ | P2 | Accuracy delta at cull |
| 6 | blending_delta | ✓ | ✗ | P3 | Contribution during blending |
| 7 | counterfactual | ✓ | ✗ | P2 | Causal attribution score |
| 8 | epochs_total | ✓ | ✗ | P2 | Time before cull |
| 9 | epochs_in_stage | ✓ | ✗ | P3 | Time in final stage |

**Recommended action:** Capture reason (critical for understanding failures), auto_culled (policy vs system decision), blueprint_id (which blueprints fail).

---

### SEED_STAGE_CHANGED (P3 - 17% uncaptured)

**Current capture:** Most fields captured

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | epochs_total | ✓ | ✗ | P3 | Total seed lifetime |
| 2 | counterfactual | ✓ | ✗ | P3 | Causal attribution |

**Recommended action:** Low priority - existing capture is sufficient for display.

---

### REWARD_COMPUTED (P3 - 5% uncaptured)

**Current capture:** Nearly complete

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | episode | ✓ | ✗ | P3 | Episode context |

**Recommended action:** Very low priority - 95% complete.

---

### EPOCH_COMPLETED Aggregate (P3 - 100% uncaptured)

This is an aggregate variant emitted at batch boundaries. Per-env EPOCH_COMPLETED is fully captured.

| # | Field | Emitted | Captured | Priority | Impact |
|---|-------|---------|----------|----------|--------|
| 1 | inner_epoch | ✓ | ✗ | P3 | Redundant with per-env |
| 2 | batch | ✓ | ✗ | P3 | Available in BATCH_EPOCH_COMPLETED |
| 3 | train_loss | ✓ | ✗ | P3 | Aggregate loss |
| 4 | train_accuracy | ✓ | ✗ | P3 | Aggregate accuracy |
| 5 | val_loss | ✓ | ✗ | P3 | Aggregate val loss |
| 6 | val_accuracy | ✓ | ✗ | P3 | Aggregate val accuracy |
| 7 | n_envs | ✓ | ✗ | P3 | Env count |
| 8 | skipped_update | ✓ | ✗ | P3 | Skip indicator |
| 9 | plateau_detected | ✓ | ✗ | P3 | Plateau flag |

**Recommended action:** Skip - redundant with per-env events and BATCH_EPOCH_COMPLETED.

---

## Implementation Tasks

### Phase 1: Critical Gaps (P1)

**Task 1: Fix host_params capture** (DONE)
- Already fixed in aggregator - captures from TRAINING_STARTED and sets on EnvState

**Task 2: Add grad_norm to PPO_UPDATE_COMPLETED handler**
- File: `src/esper/karn/sanctum/aggregator.py`
- Add to TamiyoState schema if not present
- Capture `data.get("grad_norm", 0.0)`

**Task 3: Add update_time_ms to PPO_UPDATE_COMPLETED handler**
- File: `src/esper/karn/sanctum/aggregator.py`
- Add to TamiyoState or SystemVitals
- Capture `data.get("update_time_ms", 0.0)`

**Task 4: Enhance BATCH_EPOCH_COMPLETED handler**
- File: `src/esper/karn/sanctum/aggregator.py`
- Capture: batch_idx, avg_accuracy, rolling_accuracy, avg_reward
- Update snapshot with batch-level aggregates

**Task 5: Add cull reason to SEED_CULLED handler**
- File: `src/esper/karn/sanctum/aggregator.py`
- Capture: reason, auto_culled, blueprint_id
- Display in event log with reason

**Task 6: Add fossilization metrics to SEED_FOSSILIZED handler**
- File: `src/esper/karn/sanctum/aggregator.py`
- Capture: improvement, blueprint_id, epochs_total
- Display improvement in event log

**Task 7: Add reward_mode to TRAINING_STARTED handler**
- File: `src/esper/karn/sanctum/aggregator.py`
- Capture: reward_mode
- Store at aggregator level for default A/B cohort assignment

### Phase 2: High Priority Gaps (P2)

**Task 8: Add hyperparameter capture to TRAINING_STARTED**
- Capture: seed, n_episodes, lr, clip_ratio, entropy_coef, param_budget
- Store in new RunConfig dataclass or snapshot field

**Task 9: Add early_stop_epoch to PPO_UPDATE_COMPLETED**
- Capture KL early stopping indicator
- Display in TamiyoBrain if triggered

**Task 10: Add per-head entropy/grad_norm to PPO_UPDATE_COMPLETED**
- Capture head_slot_entropy, head_blueprint_entropy
- Capture head_slot_grad_norm, head_blueprint_grad_norm
- Display in expanded TamiyoBrain view

### Phase 3: Low Priority Gaps (P3)

Defer to future work - existing capture provides sufficient visibility.

---

## Verification

After each task:
1. Run `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`
2. Verify no regressions
3. Test with live training to confirm capture

---

## Appendix: Field Reference

### TamiyoState Fields (Current)

```python
@dataclass
class TamiyoState:
    entropy: float = 0.0
    clip_fraction: float = 0.0
    kl_divergence: float = 0.0
    explained_variance: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    learning_rate: float = 0.0
    entropy_coef: float = 0.0
    ratio_mean: float = 0.0
    ratio_min: float = 0.0
    ratio_max: float = 0.0
    ratio_std: float = 0.0
    dead_layers: int = 0
    exploding_layers: int = 0
    nan_grad_count: int = 0
    layer_gradient_health: float = 0.0
    entropy_collapsed: bool = False
    # MISSING: grad_norm, update_time_ms, early_stop_epoch, per-head metrics
```

### Recommended TamiyoState Additions

```python
    # P1 additions
    grad_norm: float = 0.0
    update_time_ms: float = 0.0

    # P2 additions
    early_stop_epoch: int | None = None
    head_slot_entropy: float = 0.0
    head_slot_grad_norm: float = 0.0
    head_blueprint_entropy: float = 0.0
    head_blueprint_grad_norm: float = 0.0
```
