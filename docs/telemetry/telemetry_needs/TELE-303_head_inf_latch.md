# Telemetry Record: [TELE-303] Head Inf Latch

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-303` |
| **Name** | Head Inf Latch |
| **Category** | `gradient` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Which action heads have ever produced Inf gradients during this training run?"

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
| **Type** | `dict[str, bool]` |
| **Units** | Boolean per action head |
| **Range** | Keys: `slot`, `blueprint`, `style`, `tempo`, `alpha_target`, `alpha_speed`, `alpha_curve`, `op`. Values: `True` or `False` |
| **Precision** | N/A (boolean) |
| **Default** | `{head: False for head in HEAD_NAMES}` |

### Semantic Meaning

> This is a latched indicator dictionary tracking which action heads have ever produced Inf (infinity) values in their log probabilities during the PPO update. Once a head's latch is set to `True`, it never resets for the duration of the run.
>
> Inf gradients in action heads indicate numerical overflow, typically caused by:
> - Log probability explosion from very confident but incorrect predictions
> - Ratio explosion in PPO clipping (action probability changed too dramatically)
> - Loss scale overflow in AMP (automatic mixed precision) training
>
> Unlike transient NaN/Inf counts, the latch provides "indicator light" behavior: once lit, it stays lit, making it easy to see which heads have ever had issues.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | All values `False` | No Inf detected in any head |
| **Warning** | N/A | (No warning state - binary) |
| **Critical** | Any value `True` | Inf detected in that head at some point |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, during log probability computation per action head |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._update()` |
| **Line(s)** | 529 (initialization), 604 (detection), 996 (emission) |

```python
# Line 529: Initialize per-head Inf tracking
head_inf_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}

# Line 604: Detect Inf in log probabilities per head
if torch.isinf(lp).any():
    head_inf_detected[key] = True
    nonfinite_sources.append(f"log_probs[{key}]: Inf detected")

# Line 996: Add to aggregated result for emission
aggregated_result["head_inf_detected"] = head_inf_detected
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update_event()` with `head_inf_detected` in metrics | `simic/telemetry/emitters.py` (line 862) |
| **2. Collection** | `PPOUpdatePayload.head_inf_detected` field | `leyline/telemetry.py` (line 671) |
| **3. Aggregation** | OR-latch in `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py` (lines 873-876) |
| **4. Delivery** | Written to `TamiyoState.head_inf_latch` | `karn/sanctum/schema.py` (line 884) |

```
[PPOAgent._update()]
    --head_inf_detected-->
[emit_ppo_update_event()]
    --PPOUpdatePayload-->
[SanctumAggregator._handle_ppo_update()]
    --OR-latch-->
[TamiyoState.head_inf_latch]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `head_inf_latch` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.head_inf_latch` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 884 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` (line 452) | Displayed as indicator row with filled/empty circles per head |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent._update()` computes and sets `head_inf_detected`
- [x] **Transport works** — `emit_ppo_update_event()` includes `head_inf_detected` in payload
- [x] **Schema field exists** — `TamiyoState.head_inf_latch: dict[str, bool]`
- [x] **Default is correct** — All heads default to `False` via `HEAD_NAMES` factory
- [x] **Consumer reads it** — ActionHeadsPanel reads `tamiyo.head_inf_latch[leyline_key]`
- [x] **Display is correct** — Shows filled circle (red) if `True`, empty circle (dim) if `False`
- [x] **Thresholds applied** — Red bold style for latched (`True`), dim style for unlatched (`False`)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/agent/test_ppo_nan_detection.py` | `test_head_inf_detection_logic` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_aggregator_latches_both_nan_and_inf_same_head` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_tamiyo_state_has_nan_inf_latch_fields` | `[x]` |
| Unit (payload) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_with_per_head_fields` | `[x]` |
| Widget (display) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py` | (sets `head_inf_latch["slot"] = True`) | `[x]` |
| Integration (end-to-end) | | | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe ActionHeadsPanel "Inf" indicator row
4. Verify all indicators show empty circles (dim) initially
5. To test: artificially inject Inf in an action head's log probabilities
6. Verify the corresponding indicator changes to filled red circle and stays latched

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Per-head log probability computation | computation | Requires factored policy forward pass |
| `HEAD_NAMES` constant | leyline | Defines the action head keys |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ActionHeadsPanel Inf row | display | Visual indicator of Inf status per head |
| Training health assessment | diagnostic | Latched Inf indicates serious numerical issues |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** The latch behavior (once True, stays True) is intentional. Unlike transient counts that reset each batch, the latch provides persistent visibility into numerical health issues. If a head ever produced Inf, the operator knows even if subsequent updates appear healthy.
>
> **Relationship to head_nan_latch:** These are companion fields. A head can have both NaN and Inf latches set simultaneously (tested in `test_aggregator_latches_both_nan_and_inf_same_head`).
>
> **OR-Latch Logic:** The aggregator implements OR-latch semantics:
> ```python
> if payload.head_inf_detected:
>     for head, detected in payload.head_inf_detected.items():
>         if detected:
>             self._tamiyo.head_inf_latch[head] = True
> ```
> This means multiple PPO updates can only turn latches ON, never OFF.
>
> **Display Mapping:** The widget uses `DISPLAY_TO_LEYLINE_KEY` to map display names (e.g., "Slot", "Blueprint") to leyline keys (e.g., "slot", "blueprint") for latch lookup.
