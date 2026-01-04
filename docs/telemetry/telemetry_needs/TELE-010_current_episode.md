# Telemetry Record: [TELE-010] Current Episode

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-010` |
| **Name** | Current Episode |
| **Category** | `training` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Which episode are we on in the training run? How far through the training loop are we?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `int` |
| **Units** | episode count (1-indexed) |
| **Range** | `[start_episode, start_episode + n_episodes]` |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Episode counter representing progress through the training run. Updated at each batch completion.
>
> Starts at `start_episode` (for resume support) and increments by `n_envs` per batch.
> Formula: `current_episode = start_episode + (batch_idx * n_envs)`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0 < current_episode < total_episodes` | Normal training progress |
| **Info** | `current_episode == 0` | Pre-training (waiting for first batch) |
| **Complete** | `current_episode >= total_episodes` | Training finished |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Training main loop episode counter (vectorized environment batch completion) |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | `train()` — main training loop |
| **Line(s)** | ~1800 (initialization), ~3657 (aggregation for event) |

```python
# Initialization (line 1800)
episodes_completed = start_episode

# Increment per batch (line 1816, 3657)
batch_epoch_id = episodes_completed + envs_this_batch
# ... pass to batch emitter
episodes_completed=episodes_completed + envs_this_batch
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.on_batch_completed()` with field `episodes_completed` | `simic/telemetry/emitters.py:3657` |
| **2. Collection** | `BatchEpochCompletedPayload.episodes_completed` field | `leyline/telemetry.py:549` |
| **3. Aggregation** | `SanctumAggregator.handle_batch_epoch_completed()` reads payload | `karn/sanctum/aggregator.py:1193` |
| **4. Delivery** | Written to `snapshot.current_episode` field | `karn/sanctum/schema.py:1319` |

```
[VectorizedEnv] --episodes_completed--> [TelemetryEmitter] --BATCH_EPOCH_COMPLETED event--> [Aggregator] --> [SanctumSnapshot.current_episode]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `current_episode` |
| **Path from SanctumSnapshot** | `snapshot.current_episode` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~1319 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Displayed as "Ep 47" in segment 4 (line 252) |

Display format: `"Ep " + format(snapshot.current_episode, width=3, align=right)` → "Ep  47"

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `vectorized.py` main loop increments `episodes_completed`
- [x] **Transport works** — `BatchEpochCompletedPayload` carries `episodes_completed` field (required)
- [x] **Schema field exists** — `SanctumSnapshot.current_episode: int = 0` (line 1319)
- [x] **Default is correct** — `0` appropriate before first batch
- [x] **Consumer reads it** — `RunHeader._render()` accesses `snapshot.current_episode` (line 252)
- [x] **Display is correct** — Formatted as "Ep NNN" with 3-digit right-alignment
- [x] **Thresholds applied** — No thresholds needed (pure counter display)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | Not found | — | `[ ]` |
| Unit (aggregator) | Not found | — | `[ ]` |
| Integration (end-to-end) | Not found | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens)
3. Observe RunHeader segment 4 showing "Ep NNN"
4. Verify episode counter increments after each batch completes
5. For resumed runs: Verify counter continues from `start_episode` offset

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `start_episode` parameter | config | Resume offset (default 0) |
| `n_episodes` parameter | config | Total episodes requested (determines total_episodes) |
| `n_envs` parameter | config | Environments per batch (increment amount) |
| Training loop main thread | execution | Only updates when training thread is active |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `SeedLifecycleStats` germination/prune/fossilize rates | telemetry | Normalizes counts by `current_episode` (line 1208-1211) |
| Progress bar computation | display | Used in RunHeader progress calculation |
| Training completion detection | logic | Aggregator checks against `total_episodes` |
| Episode-level metrics (EpisodeStats) | telemetry | May use for episode indexing |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation, full wiring verified |

---

## 8. Notes

> **Design Pattern:** Pure counter tracking, not a computed metric. Increments by `n_envs` per batch for vectorized execution (N parallel environments complete per step). This matches the "batch = N episodes" model.
>
> **Resume Support:** Correctly handles resumed runs via `start_episode` offset. Counter starts at `start_episode` and increments, ensuring monotonic progress across session boundaries.
>
> **Vectorization:** Increments by `n_envs` (not 1), reflecting that N parallel environments complete simultaneously per batch. This aligns with the terminology: "episode" = one environment run, "batch" = N parallel episodes.
>
> **UI Presentation:** RunHeader shows only the episode counter without visual progress bar distinction. Progress tracking happens via `current_epoch` for the epoch progress bar (separate telemetry).
>
> **No Computed Dependency Chain:** This is a pure value pass-through. No intermediate transformations or correlations. Just carries the counter from training loop → schema field → display.
