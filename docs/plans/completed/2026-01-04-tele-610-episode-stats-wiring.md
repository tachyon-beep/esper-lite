# TELE-610 Episode Stats Wiring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the EpisodeStats fields (TELE-610) from training loop through telemetry to the EpisodeMetricsPanel.

**Architecture:** Extend EpisodeOutcomePayload with episode length and outcome classification, track action counts during episodes, aggregate in the Aggregator, and populate EpisodeStats.

**Key Insight:** In Esper, episodes are fixed-length (max_epochs), but we still want:
- Episode count tracking (already wired)
- Success/timeout classification based on final_accuracy thresholds
- Action efficiency metrics (germinate/prune/fossilize frequency)
- Trend detection from rolling success rate

---

## Overview

**Fields to Wire:**

| Field | Data Source | Computation |
|-------|-------------|-------------|
| `length_mean/std/min/max` | Episode length from vectorized.py | Rolling stats over recent episodes |
| `timeout_rate` | final_accuracy < SUCCESS_THRESHOLD | count / total |
| `success_rate` | final_accuracy >= SUCCESS_THRESHOLD | count / total |
| `early_termination_rate` | Unused (episodes are fixed-length) | Always 0.0 |
| `steps_per_germinate` | Action counts from training loop | total_steps / germinate_count (avg steps between actions) |
| `steps_per_prune` | Action counts from training loop | total_steps / prune_count |
| `steps_per_fossilize` | Action counts from training loop | total_steps / fossilize_count |
| `completion_trend` | Rolling window comparison | Compare recent N vs older N episodes |

**DRL Expert Review Incorporated:**
- Keep `steps_per_action` formulation (semantically "average steps between actions")
- Trend detection uses rolling window comparison (last 10 vs previous 10 episodes)
- SUCCESS_THRESHOLD should be documented as tunable

---

## Task 1: Extend EpisodeOutcomePayload with Length and Outcome Type

**Files:**
- Modify: `src/esper/leyline/telemetry.py` (EpisodeOutcomePayload)

**Step 1: Add fields to EpisodeOutcomePayload**

After `reward_mode: str`, add:

```python
    # Episode diagnostics (TELE-610)
    episode_length: int = 0  # Steps in this episode (usually max_epochs)
    outcome_type: str = "unknown"  # "success", "timeout", "early_termination"
    germinate_count: int = 0  # GERMINATE actions this episode
    prune_count: int = 0  # PRUNE actions this episode
    fossilize_count: int = 0  # FOSSILIZE actions this episode
```

**Step 2: Update from_dict() to parse new fields**

Add to the return statement:
```python
            episode_length=data.get("episode_length", 0),
            outcome_type=data.get("outcome_type", "unknown"),
            germinate_count=data.get("germinate_count", 0),
            prune_count=data.get("prune_count", 0),
            fossilize_count=data.get("fossilize_count", 0),
```

**Step 3: Run tests**
```bash
PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py -v --tb=short
```

**Step 4: Commit**
```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(telemetry): extend EpisodeOutcomePayload with diagnostics

TELE-610: Add episode_length, outcome_type, and action counts
(germinate/prune/fossilize) to enable EpisodeStats wiring."
```

---

## Task 2: Track Action Counts in Vectorized Training Loop

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`
- Modify: `src/esper/simic/training/parallel_env_state.py` (add counters)

**Step 1: Add action counters to ParallelEnvState**

In `ParallelEnvState`, add fields:

```python
    # Action counters for episode diagnostics (TELE-610)
    germinate_count: int = 0
    prune_count: int = 0
    fossilize_count: int = 0
```

**Step 2: Increment counters when actions are taken**

In vectorized.py, find where LifecycleOp actions are processed. After each action:

```python
if action == LifecycleOp.GERMINATE:
    env_state.germinate_count += 1
elif action == LifecycleOp.PRUNE:
    env_state.prune_count += 1
elif action == LifecycleOp.FOSSILIZE:
    env_state.fossilize_count += 1
```

**Step 3: Reset counters at episode start**

When a new episode starts, reset:
```python
env_state.germinate_count = 0
env_state.prune_count = 0
env_state.fossilize_count = 0
```

**Step 4: Commit**
```bash
git add src/esper/simic/training/parallel_env_state.py src/esper/simic/training/vectorized.py
git commit -m "feat(simic): track action counts per episode for TELE-610

Count GERMINATE, PRUNE, FOSSILIZE actions per episode for action
efficiency metrics (steps_per_germinate, etc.)."
```

---

## Task 3: Emit Extended EpisodeOutcomePayload

**Files:**
- Modify: `src/esper/simic/training/vectorized.py` (~line 3347-3364)

**Step 1: Determine outcome type**

Before creating EpisodeOutcomePayload, classify outcome:

```python
# TELE-610: Classify episode outcome
SUCCESS_THRESHOLD = 0.8  # final_accuracy >= 80% is "success"
if env_state.val_acc >= SUCCESS_THRESHOLD:
    outcome_type = "success"
else:
    outcome_type = "timeout"  # Fixed-length episodes that don't hit goal
```

**Step 2: Include new fields in payload**

Update the EpisodeOutcomePayload creation:

```python
data=EpisodeOutcomePayload(
    # ... existing fields ...
    reward_mode=episode_outcome.reward_mode,
    # TELE-610: Episode diagnostics
    episode_length=epoch,  # max_epochs for this episode
    outcome_type=outcome_type,
    germinate_count=env_state.germinate_count,
    prune_count=env_state.prune_count,
    fossilize_count=env_state.fossilize_count,
),
```

**Step 3: Commit**
```bash
git add src/esper/simic/training/vectorized.py
git commit -m "feat(simic): emit episode diagnostics in EPISODE_OUTCOME

TELE-610: Include episode_length, outcome_type, and action counts
in EPISODE_OUTCOME events for EpisodeStats aggregation."
```

---

## Task 4: Wire Aggregator to Populate EpisodeStats

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Add episode stats tracking state**

Add instance variables to SanctumAggregator.__init__():

```python
# Episode diagnostics (TELE-610)
self._episode_lengths: deque[int] = deque(maxlen=100)  # Rolling window
self._success_count: int = 0
self._timeout_count: int = 0
self._total_germinate: int = 0
self._total_prune: int = 0
self._total_fossilize: int = 0
self._recent_outcomes: deque[bool] = deque(maxlen=20)  # For trend detection (True=success)
```

**Step 2: Update _handle_episode_outcome to track stats**

In `_handle_episode_outcome()`, after creating EpisodeOutcome:

```python
        # TELE-610: Track episode diagnostics
        self._episode_lengths.append(data.episode_length)
        is_success = data.outcome_type == "success"
        self._recent_outcomes.append(is_success)
        if is_success:
            self._success_count += 1
        elif data.outcome_type == "timeout":
            self._timeout_count += 1
        self._total_germinate += data.germinate_count
        self._total_prune += data.prune_count
        self._total_fossilize += data.fossilize_count
```

**Step 3: Compute EpisodeStats in _get_snapshot_unlocked**

Replace the stub with computed stats:

```python
        # Episode stats (TELE-610)
        total = self._current_episode
        if total > 0 and self._episode_lengths:
            lengths = list(self._episode_lengths)
            length_mean = sum(lengths) / len(lengths)
            length_std = (sum((x - length_mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
            length_min = min(lengths)
            length_max = max(lengths)

            success_rate = self._success_count / total
            timeout_rate = self._timeout_count / total

            # Trend detection (DRL expert: compare rolling windows, not single samples)
            outcomes = list(self._recent_outcomes)
            if len(outcomes) >= 10:
                recent = outcomes[-10:]  # Last 10 episodes
                older = outcomes[-20:-10] if len(outcomes) >= 20 else outcomes[:len(outcomes)//2]
                recent_rate = sum(recent) / len(recent)
                older_rate = sum(older) / len(older) if older else recent_rate
                if recent_rate > older_rate + 0.1:
                    trend = "improving"
                elif recent_rate < older_rate - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"  # Not enough data for trend

            # Action efficiency
            total_steps = sum(lengths)
            steps_per_germinate = total_steps / max(1, self._total_germinate)
            steps_per_prune = total_steps / max(1, self._total_prune)
            steps_per_fossilize = total_steps / max(1, self._total_fossilize)

            episode_stats = EpisodeStats(
                total_episodes=total,
                length_mean=length_mean,
                length_std=length_std,
                length_min=length_min,
                length_max=length_max,
                success_count=self._success_count,
                timeout_count=self._timeout_count,
                success_rate=success_rate,
                timeout_rate=timeout_rate,
                early_termination_rate=0.0,  # Not applicable
                steps_per_germinate=steps_per_germinate,
                steps_per_prune=steps_per_prune,
                steps_per_fossilize=steps_per_fossilize,
                completion_trend=trend,
            )
        else:
            episode_stats = EpisodeStats(total_episodes=total)
```

**Step 4: Run tests**
```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py -v --tb=short
```

**Step 5: Commit**
```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): wire EpisodeStats in aggregator

TELE-610: Compute episode length stats, success/timeout rates,
action efficiency, and completion trend from EPISODE_OUTCOME events.
EpisodeMetricsPanel now displays real data."
```

---

## Task 5: Remove xfails from Environment Tests

**Files:**
- Modify: `tests/telemetry/test_environment_metrics.py`

**Step 1: Find and remove xfail decorators**

Remove `@pytest.mark.xfail` from:
- TestTELE610EpisodeStats class (3 tests)

**Step 2: Run tests to verify they pass**
```bash
PYTHONPATH=src uv run pytest tests/telemetry/test_environment_metrics.py -v -k TELE610
```

**Step 3: Commit**
```bash
git add tests/telemetry/test_environment_metrics.py
git commit -m "test(telemetry): enable TELE-610 episode stats tests

Remove xfail markers now that EpisodeStats wiring is complete."
```

---

## Task 6: Update WIRING_GAPS.md

**Files:**
- Modify: `docs/telemetry/WIRING_GAPS.md`

**Step 1: Mark TELE-610 as fixed**

Update the TELE-610 section to show it's now wired.

**Step 2: Commit**
```bash
git add docs/telemetry/WIRING_GAPS.md
git commit -m "docs(telemetry): mark TELE-610 as wired

Episode stats (length, outcome rates, action efficiency, trend)
now fully wired from training loop through aggregator to widget."
```

---

## Task 7: Run Full Test Suite

**Command:**
```bash
PYTHONPATH=src uv run pytest tests/telemetry/ tests/simic/ tests/karn/ -v --tb=short
```

**Expected:** All tests pass, no new xfails.

---

## Summary

After implementation, the EpisodeMetricsPanel will display:
- **Length**: μ:150 σ:0.0 [150-150] (fixed-length episodes)
- **Outcomes**: ✗15% ✓75% ⊗0% (timeout, success, early-term rates)
- **Steps/Act**: germ:45 prune:120 foss:200 (action efficiency)
- **Trend**: improving ↗ (from rolling success rate)

The episode count is already shown in the panel border title.
