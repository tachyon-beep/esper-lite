# Telemetry Record: [TELE-021] Episode Return History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-021` |
| **Name** | Episode Return History |
| **Category** | `training` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "What is the random policy baseline performance, and is the training return improving over episodes?"

This metric serves two distinct purposes:
1. **During warmup** (before PPO updates): Establishes the random policy baseline that trained policy must beat
2. **During training**: Tracks episode return trend for training progress monitoring

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `deque[float]` |
| **Max Length** | 20 entries |
| **Units** | Cumulative episode reward (dimensionless) |
| **Range** | `(-inf, inf)` — typically negative early, improving toward positive |
| **Precision** | 1 decimal place for display |
| **Default** | Empty deque (`deque(maxlen=20)`) |

### Semantic Meaning

> Rolling window of the last 20 episode returns (average reward per batch).
>
> Each value represents `avg_reward = sum(env_total_rewards) / n_envs` where:
> - `env_total_rewards[i] = sum(env_states[i].episode_rewards)` for each environment
> - Episode rewards are accumulated per-step throughout the episode
>
> **Warmup phase**: Shows random policy performance. The mean and std establish the baseline
> that PPO must improve upon.
>
> **Training phase**: Used for sparkline visualization and trend detection. Rising values
> indicate policy improvement; stagnation or decline suggests training issues.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Mean increasing over window | Policy is learning |
| **Warning** | Mean stagnant for >5 episodes | Policy may be stuck |
| **Critical** | Mean declining consistently | Policy regression occurring |

**Note:** Thresholds are relative (trend-based) rather than absolute, as raw return values depend on reward function design.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Vectorized training loop, end of each episode batch |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | `train_ppo_vectorized()` |
| **Line(s)** | ~3614, 3637 (avg_reward computation and emission) |

```python
# vectorized.py ~3614
avg_reward = sum(env_total_rewards) / len(env_total_rewards)

# vectorized.py ~3637 (passed to emitter)
avg_reward=avg_reward,
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_batch_completed()` | `simic/telemetry/emitters.py:424-438` |
| **2. Collection** | `BATCH_EPOCH_COMPLETED` event with `BatchEpochCompletedPayload` | `leyline/telemetry.py:545-580` |
| **3. Aggregation** | `SanctumAggregator._handle_batch_epoch_completed()` | `karn/sanctum/aggregator.py:1183-1205` |
| **4. Delivery** | Appended to `TamiyoState.episode_return_history` | `karn/sanctum/schema.py:938-940` |

```
[VectorizedTraining]
       │ avg_reward = sum(env_total_rewards) / n_envs
       ▼
[TelemetryEmitter.emit_batch_completed()]
       │ TelemetryEvent(BATCH_EPOCH_COMPLETED, BatchEpochCompletedPayload(avg_reward=...))
       ▼
[TelemetryHub]
       │ event dispatch
       ▼
[SanctumAggregator._handle_batch_epoch_completed()]
       │ self._tamiyo.episode_return_history.append(payload.avg_reward)
       ▼
[TamiyoState.episode_return_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `episode_return_history` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.episode_return_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 938-940 |

```python
# schema.py:938-940
# Episode return tracking (PRIMARY RL METRIC - per DRL review)
episode_return_history: deque[float] = field(
    default_factory=lambda: deque(maxlen=20)
)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EpisodeMetricsPanel | `widgets/tamiyo_brain/episode_metrics_panel.py:81` | Warmup baseline stats (mean, std) |
| ActionDistributionPanel | `widgets/tamiyo_brain/action_distribution.py:386` | Returns section with percentiles |
| SparklineUtils | `widgets/tamiyo_brain/sparkline_utils.py:150-157` | Ep.Return sparkline row |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `emit_batch_completed()` emits `BATCH_EPOCH_COMPLETED` with `avg_reward`
- [x] **Transport works** — Event payload reaches aggregator via telemetry hub
- [x] **Schema field exists** — `TamiyoState.episode_return_history: deque[float]`
- [x] **Default is correct** — Empty deque appropriate before first episode completes
- [x] **Consumer reads it** — Multiple widgets access `snapshot.tamiyo.episode_return_history`
- [x] **Display is correct** — Sparkline, stats, and percentiles render correctly
- [x] **Thresholds applied** — Trend detection uses `detect_trend()` for color coding

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | Not covered | `[ ]` |
| Unit (aggregator) | — | Not covered | `[ ]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | Snapshot construction | `[x]` |
| Integration (end-to-end) | — | Not covered | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |
| Web (Overwatch) | `web/e2e/smoke.spec.ts:136` | Fixture includes field | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens)
3. Observe **WARMUP** panel before PPO updates:
   - Should show "Collecting warmup data..."
   - Baseline row shows mean and std of returns
   - Episode count increments as batches complete
4. After PPO updates begin, observe **Episode Return** sparkline:
   - Sparkline should populate with trend visualization
   - Current value and trend indicator should appear
5. Verify returns in **Actions** panel "Returns" section:
   - Last 5 returns with color coding
   - Percentile breakdown (p10/p50/p90)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `BATCH_EPOCH_COMPLETED` event | telemetry | Triggered at end of each episode batch |
| `env_total_rewards` | computation | Sum of per-step rewards in each env |
| Episode completion | lifecycle | Only populated after first batch completes |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `current_episode_return` (TELE-022) | telemetry | Set to same value as latest append |
| Warmup baseline statistics | display | Mean/std computed in EpisodeMetricsPanel |
| Return percentiles | display | p10/p50/p90 computed in ActionDistributionPanel |
| Sparkline trend indicator | display | Uses `detect_trend()` for direction |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Created telemetry record during audit |

---

## 8. Notes

> **Design Decision:** Uses 20-element rolling window (maxlen=20) to balance:
> - Enough samples for stable statistics (percentiles, trend detection)
> - Recent enough to reflect current policy behavior
> - Memory-efficient via deque with automatic overflow
>
> **Warmup Phase Semantics:** The same field serves dual purposes:
> 1. Before `ppo_data_received=True`: Collects random policy baseline
> 2. After PPO starts: Continues tracking for trend monitoring
>
> The EpisodeMetricsPanel switches display mode based on `ppo_data_received` flag.
>
> **Related Fields:**
> - `current_episode_return` (scalar): Most recent value from this history
> - `current_episode` (int): Episode counter for display context
>
> **TypeScript Mirror:** Defined in `web/src/types/sanctum.ts:176` as `number[]` for Overwatch.
