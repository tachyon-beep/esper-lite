# Rename seed_contribution to blending_delta Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix misleading metric naming by renaming `SeedMetrics.seed_contribution` to `blending_delta` and updating log labels for clarity.

**Architecture:** The `SeedMetrics.seed_contribution` property measures `current_val_accuracy - accuracy_at_blending_start`, which is a temporal delta that conflates host drift with seed impact. The TRUE counterfactual (`real_acc - baseline_acc`) is computed separately in `vectorized.py` and used for rewards. This rename clarifies the distinction.

**Tech Stack:** Python, dataclasses

---

## Background

Two different metrics are both called "seed_contribution":

| Location | Definition | What It Actually Measures |
|----------|------------|---------------------------|
| `SeedMetrics.seed_contribution` | `current - blending_start` | Temporal delta (includes host drift) |
| `vectorized.py` counterfactual | `real_acc - baseline_acc(alpha=0)` | True causal attribution |

The rename fixes the misleading name in `SeedMetrics` without touching the correct counterfactual usage in rewards.

---

### Task 1: Rename Property in SeedMetrics

**Files:**
- Modify: `src/esper/kasmina/slot.py:103-112`

**Step 1: Rename property and fix docstring**

Change from:
```python
@property
def seed_contribution(self) -> float:
    """Accuracy change attributable to the seed (from BLENDING onward).

    This is the true causal attribution - measures impact only during
    stages where the seed actually affects network output (alpha > 0).
    Returns 0 if seed never reached BLENDING (no measurable impact).
    """
    if self.accuracy_at_blending_start == 0.0:
        return 0.0
    return self.current_val_accuracy - self.accuracy_at_blending_start
```

To:
```python
@property
def blending_delta(self) -> float:
    """Accuracy change since blending started (includes host drift).

    This is NOT causal attribution - it measures the total accuracy change
    during BLENDING stages, which conflates host training gains with seed
    impact. For true causal attribution, use counterfactual validation
    (real_acc - baseline_acc with alpha=0).

    Returns 0 if seed never reached BLENDING.
    """
    if self.accuracy_at_blending_start == 0.0:
        return 0.0
    return self.current_val_accuracy - self.accuracy_at_blending_start
```

**Step 2: Verify change compiles**

Run: `python3 -m py_compile src/esper/kasmina/slot.py`
Expected: No output (success)

---

### Task 2: Update References in slot.py

**Files:**
- Modify: `src/esper/kasmina/slot.py:707,740,777,799`

**Step 1: Update advance_stage() method (line ~707)**

Change from:
```python
seed_contribution = metrics.seed_contribution
```

To:
```python
blending_delta = metrics.blending_delta
```

**Step 2: Update fossilize telemetry data (line ~740)**

Change from:
```python
"seed_contribution": seed_contribution,  # True causal attribution
```

To:
```python
"blending_delta": blending_delta,
```

**Step 3: Update cull() method (line ~777)**

Change from:
```python
seed_contribution = self.state.metrics.seed_contribution
```

To:
```python
blending_delta = self.state.metrics.blending_delta
```

**Step 4: Update cull telemetry data (line ~799)**

Change from:
```python
"seed_contribution": seed_contribution,  # True causal attribution
```

To:
```python
"blending_delta": blending_delta,
```

**Step 5: Verify change compiles**

Run: `python3 -m py_compile src/esper/kasmina/slot.py`
Expected: No output (success)

---

### Task 3: Update Analytics Event Handling

**Files:**
- Modify: `src/esper/nissa/analytics.py:47,60-62,155,161,172,179,185,195,218,252`

**Step 1: Rename BlueprintStats field (line ~47)**

Change from:
```python
seed_contributions: list[float] = field(default_factory=list)  # True causal attribution
```

To:
```python
blending_deltas: list[float] = field(default_factory=list)  # Accuracy change during blending
```

**Step 2: Rename property (lines ~60-62)**

Change from:
```python
@property
def mean_seed_contribution(self) -> float:
    """Mean seed contribution (from BLENDING onward) - true causal attribution."""
    return sum(self.seed_contributions) / len(self.seed_contributions) if self.seed_contributions else 0.0
```

To:
```python
@property
def mean_blending_delta(self) -> float:
    """Mean accuracy change during blending stages."""
    return sum(self.blending_deltas) / len(self.blending_deltas) if self.blending_deltas else 0.0
```

**Step 3: Update SEED_FOSSILIZED handler (lines ~155,161,172)**

Change from:
```python
seed_contribution = event.data.get("seed_contribution", 0.0)
...
self.stats[bp_id].seed_contributions.append(seed_contribution)
...
print(f"    [env{env_id}] Fossilized '{seed_id}' ({bp_id}, "
      f"total Δacc {improvement:+.2f}%, seed contrib {seed_contribution:+.2f}%)")
```

To:
```python
blending_delta = event.data.get("blending_delta", 0.0)
...
self.stats[bp_id].blending_deltas.append(blending_delta)
...
print(f"    [env{env_id}] Fossilized '{seed_id}' ({bp_id}, "
      f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%)")
```

**Step 4: Update SEED_CULLED handler (lines ~179,185,195)**

Change from:
```python
seed_contribution = event.data.get("seed_contribution", 0.0)
...
self.stats[bp_id].seed_contributions.append(seed_contribution)
...
print(f"    [env{env_id}] Culled '{seed_id}' ({bp_id}, "
      f"total Δacc {improvement:+.2f}%, seed contrib {seed_contribution:+.2f}%){reason_str}")
```

To:
```python
blending_delta = event.data.get("blending_delta", 0.0)
...
self.stats[bp_id].blending_deltas.append(blending_delta)
...
print(f"    [env{env_id}] Culled '{seed_id}' ({bp_id}, "
      f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%){reason_str}")
```

**Step 5: Update summary_table output (line ~218)**

Change from:
```python
f"{s.mean_acc_delta:>+7.2f}% {s.mean_seed_contribution:>+7.2f}% "
```

To:
```python
f"{s.mean_acc_delta:>+7.2f}% {s.mean_blending_delta:>+7.2f}% "
```

**Step 6: Update to_dict output (line ~252)**

Change from:
```python
"mean_seed_contribution": s.mean_seed_contribution,
```

To:
```python
"mean_blending_delta": s.mean_blending_delta,
```

**Step 7: Verify change compiles**

Run: `python3 -m py_compile src/esper/nissa/analytics.py`
Expected: No output (success)

---

### Task 4: Run Tests

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run specific analytics tests if they exist**

Run: `uv run pytest tests/ -v -k "analytics or nissa" --tb=short`
Expected: Pass or no tests found

---

### Task 5: Commit

**Step 1: Stage changes**

```bash
git add src/esper/kasmina/slot.py src/esper/nissa/analytics.py
```

**Step 2: Commit with descriptive message**

```bash
git commit -m "refactor: rename seed_contribution to blending_delta for clarity

The SeedMetrics.seed_contribution property was misleadingly named - it
measures accuracy change since blending started (current - blending_start),
which conflates host drift with seed impact. This is NOT causal attribution.

Renamed to blending_delta to distinguish from the TRUE counterfactual
seed_contribution computed in vectorized.py (real_acc - baseline_acc).

Log output now shows 'blending Δ' instead of 'seed contrib' to avoid
confusion with the causal metric used in reward computation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Files NOT Modified (Intentionally)

These files use `seed_contribution` correctly for the TRUE counterfactual:

- `src/esper/tolaria/trainer.py` - `AttributionResult.seed_contribution` is the counterfactual
- `src/esper/simic/vectorized.py` - computes counterfactual as `val_acc - baseline_acc`
- `src/esper/simic/rewards.py` - uses counterfactual for reward shaping

These are correctly named and should NOT be changed.

---

## Expected Log Output Change

**Before:**
```
[env2] Fossilized 'env2_seed_0' (conv_heavy, total Δacc +1.95%, seed contrib -1.81%)
```

**After:**
```
[env2] Fossilized 'env2_seed_0' (conv_heavy, total Δacc +1.95%, blending Δ -1.81%)
```

The new label makes it clear this is a temporal delta during blending, not causal attribution.
