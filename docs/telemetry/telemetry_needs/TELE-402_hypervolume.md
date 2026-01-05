# Telemetry Record: [TELE-402] Pareto Hypervolume

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-402` |
| **Name** | Pareto Hypervolume |
| **Category** | `reward` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the multi-objective optimization making progress? Are we finding better trade-offs between accuracy and parameter efficiency?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | Area units (accuracy * param_ratio) |
| **Range** | `[0.0, infinity)` - should increase monotonically |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Hypervolume (HV) measures the size of the objective space dominated by the Pareto frontier.
>
> For 2D (accuracy vs param_ratio):
> - Reference point: (min_accuracy=0.0, max_param_ratio=1.0)
> - Uses sweep-line algorithm: sort by accuracy descending, accumulate rectangular areas
> - Higher HV = better trade-off frontier found
>
> Formula per contributing point:
> `HV += (acc - ref_acc) * (current_param - point_param)`
>
> Only points with `accuracy > ref_acc` AND `param_ratio < ref_param` contribute.
>
> Monotonic increase indicates the policy is discovering better Pareto-optimal outcomes.
> Stagnation suggests the optimization has converged or the policy is stuck.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `HV > 0 AND increasing` | Finding better trade-offs |
| **Warning** | `HV stagnant for 10+ batches` | Potential convergence or stuck policy |
| **Critical** | `HV = 0` | No valid Pareto outcomes yet |

**Note:** No explicit threshold constants defined. Displayed with cyan styling in RewardHealthPanel.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Episode completion in vectorized training |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | Batch completion section (~line 3337-3362) |
| **Line(s)** | ~3337-3362, 3520-3545 (rollback correction) |

```python
# Episode outcome created at episode boundary
episode_outcome = EpisodeOutcome(
    env_id=env_idx,
    episode_idx=episodes_completed + env_idx,
    final_accuracy=env_state.val_acc,
    param_ratio=(model.total_params - host_params_baseline) / max(1, host_params_baseline),
    num_fossilized=env_state.seeds_fossilized,
    num_contributing_fossilized=env_state.contributing_fossilized,
    episode_reward=env_total_rewards[env_idx],
    stability_score=stability,
    reward_mode=env_reward_configs[env_idx].reward_mode.value,
)

# Emit EPISODE_OUTCOME telemetry for Pareto analysis
env_state.telemetry_cb(TelemetryEvent(
    event_type=TelemetryEventType.EPISODE_OUTCOME,
    epoch=episodes_completed + env_idx,
    data=EpisodeOutcomePayload(...),
))
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEvent(EPISODE_OUTCOME)` via callback | `simic/training/vectorized.py` |
| **2. Collection** | Event routed through backend | `karn/sanctum/backend.py` |
| **3. Aggregation** | `SanctumAggregator._handle_episode_outcome()` | `karn/sanctum/aggregator.py` |
| **4. Computation** | `_compute_hypervolume()` -> `compute_hypervolume_2d()` | `karn/pareto.py` |
| **5. Delivery** | `compute_reward_health()` -> `RewardHealthData.hypervolume` | `karn/sanctum/aggregator.py` |

```
[VectorizedTrainer] --EPISODE_OUTCOME--> [Backend] --event--> [Aggregator]
                                                              |
                                                              v
[EpisodeOutcome list] --extract_pareto_frontier()--> [Frontier]
                                                              |
                                                              v
[Frontier] --compute_hypervolume_2d(ref=(0.0, 1.0))--> [HV float]
                                                              |
                                                              v
[compute_reward_health()] --> [RewardHealthData.hypervolume]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardHealthData` |
| **Field** | `hypervolume` |
| **Path from SanctumSnapshot** | Via `aggregator.compute_reward_health().hypervolume` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/reward_health.py` |
| **Schema Line** | ~26 |

```python
@dataclass
class RewardHealthData:
    """Aggregated reward health metrics."""
    pbrs_fraction: float = 0.0
    anti_gaming_trigger_rate: float = 0.0
    ev_explained: float = 0.0
    hypervolume: float = 0.0  # <-- This field
```

### Intermediate Storage

The aggregator maintains episode outcomes in `_episode_outcomes` list:

| Property | Value |
|----------|-------|
| **Storage** | `SanctumAggregator._episode_outcomes: list[EpisodeOutcome]` |
| **Max Size** | 100 outcomes (bounded) |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Line** | ~299, ~1598-1602 |

```python
# In __post_init__:
self._episode_outcomes: list[EpisodeOutcome] = []

# In _handle_episode_outcome:
self._episode_outcomes.append(outcome)
if len(self._episode_outcomes) > 100:
    self._episode_outcomes = self._episode_outcomes[-100:]
```

### Pareto Computation

| Property | Value |
|----------|-------|
| **File** | `/home/john/esper-lite/src/esper/karn/pareto.py` |
| **Functions** | `extract_pareto_frontier()`, `compute_hypervolume_2d()` |

```python
def _compute_hypervolume(self) -> float:
    """Compute hypervolume indicator from recent episode outcomes."""
    if not self._episode_outcomes:
        return 0.0

    frontier = extract_pareto_frontier(self._episode_outcomes)
    ref_point = (0.0, 1.0)  # (min_accuracy, max_param_ratio)
    return compute_hypervolume_2d(frontier, ref_point)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RewardHealthPanel | `widgets/reward_health.py` | Displayed as "HV:{value}" with cyan styling |
| ActionContext | `widgets/tamiyo_brain/action_distribution.py` | Section 2 "Reward Signal" shows HV |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `EpisodeOutcomePayload` emitted at episode boundaries in vectorized trainer
- [x] **Transport works** — Event routed through `_handle_episode_outcome()` to `_episode_outcomes` list
- [x] **Schema field exists** — `RewardHealthData.hypervolume: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before any episodes complete
- [x] **Consumer reads it** — RewardHealthPanel and ActionContext display the value
- [x] **Display is correct** — Value renders as "HV:X.X" with cyan styling
- [ ] **Thresholds applied** — No thresholds defined; always cyan

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (pareto) | `tests/karn/test_pareto.py` | `test_hypervolume_2d_*` (10 tests) | `[x]` |
| Unit (pareto) | `tests/karn/test_pareto.py` | `test_extract_pareto_frontier_*` (4 tests) | `[x]` |
| Unit (widget) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_*` (9 tests) | `[x]` |
| Property-based | `tests/karn/test_pareto.py` | `test_hypervolume_is_non_negative` | `[x]` |
| Property-based | `tests/karn/test_pareto.py` | `test_pareto_frontier_*` (2 tests) | `[x]` |
| Integration (end-to-end) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 20`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Wait for first batch to complete (episodes complete)
4. Observe ActionContext panel "Reward Signal" section
5. Verify "HV:X.X" appears with cyan styling
6. Verify HV value increases as more episodes complete
7. Check RewardHealthPanel also shows HV value

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPISODE_OUTCOME events | telemetry | One per episode completion |
| `EpisodeOutcome.final_accuracy` | data | Y-axis of Pareto space |
| `EpisodeOutcome.param_ratio` | data | X-axis of Pareto space |
| `EpisodeOutcome.stability_score` | data | Third objective (not used in 2D HV) |
| Batch completion | event | Triggers episode finalization |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RewardHealthPanel | display | Shows HV in reward health summary |
| ActionContext | display | Shows HV in Reward Signal section |
| (Future) Multi-objective alerts | system | Could trigger on HV stagnation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - verified end-to-end wiring |

---

## 8. Notes

> **Design Decision:** Hypervolume uses a 2D projection (accuracy vs param_ratio) rather than full 3D (including stability_score). This simplifies computation and visualization while capturing the primary trade-off of interest.
>
> **Reference Point:** Fixed at (0.0, 1.0) meaning:
> - min_accuracy = 0.0 (any positive accuracy contributes)
> - max_param_ratio = 1.0 (param growth up to 100% of host)
>
> This is a conservative reference that includes most reasonable outcomes. Future work could make this configurable.
>
> **Memory Bound:** Only the last 100 episode outcomes are kept in `_episode_outcomes`, so HV reflects recent training behavior rather than all-time best. This prevents memory growth and focuses on current policy quality.
>
> **Algorithm:** Uses sweep-line algorithm for 2D hypervolume - O(n log n) complexity where n = frontier size.
>
> **Monotonicity:** In ideal training, HV should increase monotonically as the policy discovers better Pareto-optimal outcomes. Decreases can occur due to:
> 1. Memory bound dropping older good outcomes
> 2. Policy regression (exploration noise)
> 3. Task difficulty changes
