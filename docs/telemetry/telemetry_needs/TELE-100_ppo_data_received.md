# Telemetry Record: [TELE-100] PPO Data Received

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-100` |
| **Name** | PPO Data Received |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Has the PPO policy agent begun training? Should we display policy metrics and start gating display panels?"

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
| **Type** | `bool` |
| **Units** | N/A (binary state flag) |
| **Range** | `True` or `False` |
| **Precision** | N/A |
| **Default** | `False` |

### Semantic Meaning

> Boolean gate flag that transitions from `False` (waiting for first PPO update) to `True` (first PPO_UPDATE_COMPLETED event received).
>
> This metric controls the visibility of all PPO-dependent UI panels and metrics. Before PPO data arrives, the system displays "Waiting for PPO data..." and warmup diagnostics. Once True, full training metrics become visible.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `True` after first PPO update | Training has begun, metrics are flowing |
| **Warning** | `False` after warmup_batches=50 | Training may be stalled or not started |
| **Critical** | `False` after episode ends | PPO is not running despite training loop active |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | First PPO update completion, after gradient computation |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py` |
| **Function/Method** | `TelemetryEmitter.emit_ppo_update_completed()` |
| **Line(s)** | ~780-820 |

```python
# PPO update event emission
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
    epoch=episodes_completed,
    data=PPOUpdatePayload(
        policy_loss=metrics["policy_loss"],
        value_loss=metrics["value_loss"],
        entropy=metrics["entropy"],
        ...
    )
))
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `PPO_UPDATE_COMPLETED` telemetry event with payload | `simic/telemetry/emitters.py` |
| **2. Collection** | Telemetry hub queues event with timestamp | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` processes event | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Sets `TamiyoState.ppo_data_received = True` | `karn/sanctum/schema.py` |

```
[PPOAgent] --emit_ppo_update_completed()--> [TelemetryHub] --event--> [SanctumAggregator.handle_ppo_update()] --> [TamiyoState.ppo_data_received]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `ppo_data_received` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.ppo_data_received` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 965 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Gate for metrics display; shows "Waiting for PPO data..." until True (line 134-137) |
| EpisodeMetricsPanel | `widgets/tamiyo_brain/episode_metrics_panel.py` | Gate for switching from warmup mode to training mode (lines 49, 65) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `emit_ppo_update_completed()` in emitters.py sends PPO_UPDATE_COMPLETED event
- [x] **Transport works** — Event reaches aggregator via telemetry hub
- [x] **Schema field exists** — `TamiyoState.ppo_data_received: bool = False` defined
- [x] **Default is correct** — `False` is appropriate before first PPO update
- [x] **Consumer reads it** — Both StatusBanner and EpisodeMetricsPanel access the field
- [x] **Display is correct** — Correctly gates metrics display and shows waiting message
- [x] **Thresholds applied** — No thresholds (boolean gate), used for conditional rendering

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` or integration tests | Event emission confirmed | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_ppo_data_received` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_ppo_data_gate_controls_display` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification of warmup → training transition | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe StatusBanner initially shows `[?] WAITING` with "Waiting for PPO data..." message
4. After first PPO batch completes, banner updates to show metrics (entropy, KL, etc.)
5. Verify EpisodeMetricsPanel transitions from "WARMUP" to "EPISODE HEALTH" mode
6. Confirm StatusBanner now displays full metrics (KL, Batch count, Memory) instead of waiting message

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only becomes True after first PPO_UPDATE_COMPLETED event |
| Training loop execution | system | Requires active Simic/PPO training thread |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner metrics display | widget | Gates whether full metrics are shown vs "Waiting for PPO data..." |
| EpisodeMetricsPanel mode | widget | Controls switching from warmup diagnostics to training metrics |
| TamiyoBrain visibility | system | All policy metrics in TamiyoBrain depend on this gate |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Claude Code | Initial creation - TELE-100 telemetry audit |

---

## 8. Notes

> **Design Decision:** Boolean gate (not timestamp or counter) because the question is "has training started?" not "when did it start?". Once True, it never reverts to False during a training session, making it a stable one-way state transition.
>
> **Behavioral Note:** The metric is set in `SanctumAggregator.handle_ppo_update()` at line 782, immediately upon receiving the first PPO_UPDATE_COMPLETED event with a non-skipped payload. This happens during the `process_event()` call from the training thread.
>
> **Consumer Pattern:** Both widgets check this flag to determine rendering mode:
> - StatusBanner (line 134): `if tamiyo.ppo_data_received:` gates whether metrics are appended
> - EpisodeMetricsPanel (lines 49, 65): Switches border title and render method based on flag
>
> **Warmup Context:** During warmup (first 50 batches with no PPO updates), the system collects rollout data. Once PPO begins, this flag transitions and widgets shift display modes. This is a critical control point for the user experience.
>
> **Integration Point:** The event path is:
> 1. PPO training loop completes gradient update
> 2. `emit_ppo_update_completed()` fires PPO_UPDATE_COMPLETED event
> 3. Telemetry hub queues the event
> 4. Aggregator's `process_event()` method receives it
> 5. `handle_ppo_update()` checks payload.skipped and sets flag
> 6. Next UI poll reads the flag and updates display

