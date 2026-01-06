# Telemetry Record: [TELE-600] Observation NaN Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-600` |
| **Name** | Observation NaN Count |
| **Category** | `environment` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Are there NaN values in observation tensors? Any NaN in observations will propagate through the network and corrupt gradients."

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
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
| **Units** | count of NaN values |
| **Range** | `[0, max_observations_in_batch]` — non-negative |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> NaN count represents the number of NaN (Not-a-Number) values detected in observation tensors during a forward pass. Observations are the input feature vectors to the policy network. NaN values in observations are upstream errors that will:
> 1. Corrupt the policy forward pass (NaN propagates through all downstream operations)
> 2. Make advantage estimates invalid
> 3. Eventually corrupt gradients and cause training divergence
>
> This metric catches the ROOT CAUSE before downstream metrics (NaN gradients) appear.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `nan_count == 0` | No NaN in observations (expected state) |
| **Warning** | `0 < nan_count < batch_size` | Some observations corrupted, others valid |
| **Critical** | `nan_count > 0` | ANY NaN in observations is a critical failure |

**Threshold Source:** Any value > 0 → critical. NaN in observations cannot be tolerated (unlike gradient noise which can be transient).

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Observation tensor health check during vectorized PPO training |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized_trainer.py` |
| **Function/Method** | `compute_observation_stats()` (invoked after `obs_normalizer.normalize`) |
| **Line(s)** | Near the per-step observation normalization block |

**Emission Path:** `compute_observation_stats()` (in
`/home/john/esper-lite/src/esper/simic/telemetry/observation_stats.py`)
produces `ObservationStatsTelemetry`, which is attached to
`EpochCompletedPayload` in `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py`.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `EpochCompletedPayload.observation_stats` attached per env | `simic/telemetry/emitters.py` |
| **2. Collection** | `EpochCompletedPayload` carries `observation_stats` | `leyline/telemetry.py` |
| **3. Aggregation** | `_handle_epoch_completed` updates `ObservationStats` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Populated to `snapshot.observation_stats.nan_count` | `karn/sanctum/schema.py` |

```
[compute_observation_stats] --> [EpochCompletedPayload.observation_stats] --> [SanctumAggregator] --> [ObservationStats.nan_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ObservationStats` |
| **Field** | `nan_count` |
| **Path from SanctumSnapshot** | `snapshot.observation_stats.nan_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~219 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Displayed in "Obs Health" row with status coloring |
| (None other identified) | | Only health panel currently displays obs stats |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_observation_stats()` in `simic/telemetry/observation_stats.py`
- [x] **Transport works** — `EpochCompletedPayload.observation_stats` carries the stats
- [x] **Schema field exists** — `ObservationStats.nan_count: int = 0` defined in schema
- [x] **Default is correct** — Default 0 is correct; NaN is exceptional
- [x] **Consumer reads it** — `HealthStatusPanel._render_obs_health()` reads and displays it
- [x] **Display is correct** — Shows "NaN:X" with red bold styling when > 0
- [x] **Thresholds applied** — Critical styling when `nan_count > 0`

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | — | `[ ]` |
| Unit (aggregator) | — | — | `[ ]` |
| Integration (end-to-end) | `tests/telemetry/test_environment_metrics.py` | `TestTELE600ObsNanCount::test_nan_count_populated_from_epoch_completed` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe HealthStatusPanel "Obs Health" row
4. In a healthy run, row should show no NaN/Inf values (or very brief spikes)
5. If NaN appears, row should display red "NaN:X Inf:Y" and status "critical"
6. Trigger observation corruption (if possible) to verify color coding

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Observation normalization | computation | Requires running normalizer with valid statistics |
| Batch observation generation | event | Only populated when observations collected from environment |
| Environment health | event | Bad observations indicate upstream environment/wrapper issues |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| [NOT YET] Auto-rollback | system | When NaN observed, may trigger governor intervention |
| [NOT YET] Training pause | system | May auto-pause training to prevent corruption spread |
| StatusBanner status | display | Could contribute to FAIL/WARN status if integrated |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude Code | Initial creation; identified wiring gap |
| 2026-01-07 | Claude Code | Wired via EpochCompletedPayload observation_stats and added integration test |
| | | |

---

## 8. Notes

> **WIRING STATUS:** Observation NaN counts are emitted via
> `EpochCompletedPayload.observation_stats`, aggregated in
> `SanctumAggregator._handle_epoch_completed`, and surfaced in the
> HealthStatusPanel.
>
> **RELATED TELEMETRY:** This is the environment-side counterpart to TELE-300 (nan_grad_count in gradients). Unlike gradient NaN which can be transient, observation NaN indicates a systematic environment/normalization failure that requires immediate investigation.
>
> **DESIGN DECISION:** NaN in observations is treated as P0-critical because it's:
> - An upstream error (before policy network)
> - Deterministic (not transient noise)
> - Will corrupt ALL downstream computation
> - Requires stopping training to investigate
>
> Unlike gradient noise which can fluctuate naturally, observation NaN should never occur in a healthy training run.
