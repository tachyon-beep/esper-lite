# P2 Telemetry Gap Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close remaining 16 P2 telemetry gaps to reach ~75% capture rate.

**Architecture:** Add fields to schema dataclasses, then capture in aggregator handlers. All changes are additive and backwards-compatible.

**Tech Stack:** Python dataclasses, existing aggregator pattern

---

## Summary of Remaining Gaps

| Event | Field | Status |
|-------|-------|--------|
| TRAINING_STARTED | seed, n_episodes, lr, clip_ratio, entropy_coef, param_budget, resume_path, entropy_anneal | Not captured |
| PPO_UPDATE_COMPLETED | head_slot_entropy, head_slot_grad_norm, head_blueprint_entropy, head_blueprint_grad_norm, inner_epoch, batch | Not captured |
| BATCH_EPOCH_COMPLETED | avg_reward, total_episodes | Not captured |
| SEED_CULLED | blueprint_id | Not captured |

---

## Task 1: Add RunConfig to Schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`

**Step 1: Add RunConfig dataclass**

Add after `DecisionSnapshot` dataclass (~line 422):

```python
@dataclass
class RunConfig:
    """Training run configuration captured at TRAINING_STARTED.

    Stores hyperparameters and config for display in run header.
    """
    seed: int | None = None  # Random seed for reproducibility
    n_episodes: int = 0  # Total episodes to train
    lr: float = 0.0  # Initial learning rate
    clip_ratio: float = 0.2  # PPO clip ratio
    entropy_coef: float = 0.01  # Initial entropy coefficient
    param_budget: int = 0  # Seed parameter budget
    resume_path: str = ""  # Checkpoint resume path (empty if fresh run)
    entropy_anneal: dict = field(default_factory=dict)  # Entropy schedule config
```

**Step 2: Add run_config field to SanctumSnapshot**

In `SanctumSnapshot` class, add after `task_name`:

```python
    run_config: RunConfig = field(default_factory=RunConfig)
```

**Step 3: Verify no syntax errors**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.schema import SanctumSnapshot, RunConfig"`

---

## Task 2: Add Per-Head PPO Fields to TamiyoState

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`

**Step 1: Add per-head fields to TamiyoState**

In `TamiyoState` class, add after `early_stop_epoch` (~line 276):

```python
    # Per-head entropy and gradient norms (for multi-head policy)
    head_slot_entropy: float = 0.0  # Entropy for slot action head
    head_slot_grad_norm: float = 0.0  # Gradient norm for slot head
    head_blueprint_entropy: float = 0.0  # Entropy for blueprint action head
    head_blueprint_grad_norm: float = 0.0  # Gradient norm for blueprint head

    # PPO inner loop context
    inner_epoch: int = 0  # Current inner optimization epoch
    batch: int = 0  # Current batch within PPO update
```

**Step 2: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.schema import TamiyoState"`

---

## Task 3: Add Batch Aggregate Fields to SanctumSnapshot

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`

**Step 1: Add batch aggregate fields to SanctumSnapshot**

In `SanctumSnapshot` class, add after `aggregate_mean_reward` (~line 497):

```python
    # Batch-level aggregates (from BATCH_EPOCH_COMPLETED)
    batch_avg_reward: float = 0.0  # Average reward for last batch
    batch_total_episodes: int = 0  # Total episodes in training run
```

**Step 2: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.schema import SanctumSnapshot"`

---

## Task 4: Add blueprint_id to SEED_CULLED Capture

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Capture blueprint_id in SEED_CULLED handler**

In `_handle_seed_event`, in the `elif event_type == "SEED_CULLED":` block (~line 549), add after `seed.counterfactual`:

```python
            seed.blueprint_id = data.get("blueprint_id") or seed.blueprint_id
```

**Step 2: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.aggregator import SanctumAggregator"`

---

## Task 5: Capture RunConfig in TRAINING_STARTED Handler

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Add RunConfig import**

At top of file, add to imports:

```python
from esper.karn.sanctum.schema import (
    ...
    RunConfig,  # Add this
)
```

**Step 2: Add _run_config field to SanctumAggregator**

In the dataclass fields section (~line 110), add:

```python
    _run_config: RunConfig = field(default_factory=RunConfig)
```

**Step 3: Capture hyperparameters in _handle_training_started**

In `_handle_training_started`, after `self._reward_mode = ...` (~line 293), add:

```python
        # Capture hyperparameters for run header display
        self._run_config = RunConfig(
            seed=data.get("seed"),
            n_episodes=data.get("n_episodes", 0),
            lr=data.get("lr", 0.0),
            clip_ratio=data.get("clip_ratio", 0.2),
            entropy_coef=data.get("entropy_coef", 0.01),
            param_budget=data.get("param_budget", 0),
            resume_path=data.get("resume_path", ""),
            entropy_anneal=data.get("entropy_anneal", {}),
        )
```

**Step 4: Add run_config to _get_snapshot_unlocked**

In `_get_snapshot_unlocked`, add to the SanctumSnapshot constructor:

```python
            run_config=self._run_config,
```

---

## Task 6: Capture Per-Head PPO Metrics

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Capture per-head metrics in _handle_ppo_update**

In `_handle_ppo_update`, after `self._tamiyo.early_stop_epoch = ...` (~line 429), add:

```python
        # Per-head entropy and gradient norms
        self._tamiyo.head_slot_entropy = data.get("head_slot_entropy", 0.0)
        self._tamiyo.head_slot_grad_norm = data.get("head_slot_grad_norm", 0.0)
        self._tamiyo.head_blueprint_entropy = data.get("head_blueprint_entropy", 0.0)
        self._tamiyo.head_blueprint_grad_norm = data.get("head_blueprint_grad_norm", 0.0)

        # PPO inner loop context
        self._tamiyo.inner_epoch = data.get("inner_epoch", 0)
        self._tamiyo.batch = data.get("batch", 0)
```

**Step 2: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.aggregator import SanctumAggregator"`

---

## Task 7: Capture Batch Aggregates in BATCH_EPOCH_COMPLETED

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Add batch aggregate fields to aggregator**

In the dataclass fields section (~line 113), add:

```python
    _batch_avg_reward: float = 0.0  # Batch average reward
    _batch_total_episodes: int = 0  # Total episodes in run
```

**Step 2: Capture in _handle_batch_completed**

In `_handle_batch_completed`, after `self._batch_rolling_accuracy = ...` (~line 586), add:

```python
        self._batch_avg_reward = data.get("avg_reward", 0.0)
        self._batch_total_episodes = data.get("total_episodes", 0)
```

**Step 3: Add to _get_snapshot_unlocked**

In the SanctumSnapshot constructor in `_get_snapshot_unlocked`, add:

```python
            batch_avg_reward=self._batch_avg_reward,
            batch_total_episodes=self._batch_total_episodes,
```

---

## Task 8: Run Tests

**Step 1: Run all Sanctum tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests pass

**Step 2: Verify import chain**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.aggregator import SanctumAggregator; a = SanctumAggregator(); print('OK')"`

---

## Task 9: Commit

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): complete P2 telemetry gap closure

Add remaining P2 telemetry captures:

Schema additions:
- RunConfig dataclass for hyperparameters (seed, n_episodes, lr, etc.)
- TamiyoState per-head metrics (head_*_entropy, head_*_grad_norm)
- SanctumSnapshot batch aggregates (batch_avg_reward, batch_total_episodes)

Aggregator captures:
- TRAINING_STARTED: seed, n_episodes, lr, clip_ratio, entropy_coef, param_budget, resume_path, entropy_anneal
- PPO_UPDATE_COMPLETED: head_slot_entropy, head_slot_grad_norm, head_blueprint_entropy, head_blueprint_grad_norm, inner_epoch, batch
- BATCH_EPOCH_COMPLETED: avg_reward, total_episodes
- SEED_CULLED: blueprint_id

Telemetry capture improved from ~65% to ~75%.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After implementation, verify these fields are captured:

| Event | Field | Where to Check |
|-------|-------|----------------|
| TRAINING_STARTED | seed | snapshot.run_config.seed |
| TRAINING_STARTED | n_episodes | snapshot.run_config.n_episodes |
| TRAINING_STARTED | lr | snapshot.run_config.lr |
| PPO_UPDATE_COMPLETED | head_slot_entropy | snapshot.tamiyo.head_slot_entropy |
| PPO_UPDATE_COMPLETED | head_blueprint_entropy | snapshot.tamiyo.head_blueprint_entropy |
| BATCH_EPOCH_COMPLETED | avg_reward | snapshot.batch_avg_reward |
| BATCH_EPOCH_COMPLETED | total_episodes | snapshot.batch_total_episodes |
| SEED_CULLED | blueprint_id | env.seeds[slot_id].blueprint_id |

---

## Updated Gap Status (Post-Implementation)

| Category | Before | After |
|----------|--------|-------|
| P1 Gaps | 0 | 0 |
| P2 Gaps | 16 | 0 |
| P3 Gaps | 18 | 18 (deferred) |
| Capture Rate | ~65% | ~75% |
