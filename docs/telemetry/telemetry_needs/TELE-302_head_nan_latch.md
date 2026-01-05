# Telemetry Record: [TELE-302] Head NaN Latch

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-302` |
| **Name** | Head NaN Latch |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which action heads have ever produced NaN gradients during this training run?"

This is a latched indicator - once True for a head, it stays True for the entire run. This provides persistent visibility into gradient health issues that may have occurred transiently but could indicate underlying numerical stability problems.

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
| **Type** | `dict[str, bool]` |
| **Units** | boolean (True = NaN detected at some point) |
| **Range** | `{head: True/False for head in HEAD_NAMES}` |
| **Precision** | N/A (boolean) |
| **Default** | `{head: False for head in HEAD_NAMES}` |

### HEAD_NAMES Keys

The dictionary keys correspond to Leyline's `HEAD_NAMES`:
- `op` - Operation head (GERMINATE, WAIT, etc.)
- `slot` - Slot selection head
- `blueprint` - Blueprint selection head (conv_light, attention, etc.)
- `style` - Blending style head (LINEAR_ADD, GATED_GATE, etc.)
- `tempo` - Integration tempo head (FAST, STANDARD, SLOW)
- `alpha_target` - Alpha target head (HALF, SEVENTY, FULL)
- `alpha_speed` - Alpha ramp speed head (INSTANT, FAST, MEDIUM, SLOW)
- `alpha_curve` - Alpha curve shape head (LINEAR, COSINE, SIGMOID, etc.)

### Semantic Meaning

> Per-head NaN detection with OR-latch behavior. During each PPO update, log probabilities
> are checked for non-finite values. If NaN is detected in a head's log_probs tensor, that
> head's latch is set to True and remains True for the entire training run.
>
> This provides a "permanent record" of numerical instability - even if NaN occurs once
> and is subsequently masked or recovered from, the operator knows it happened.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | All values `False` | No NaN ever detected in any head |
| **Warning** | Any value `True` | NaN was detected at some point - investigate |
| **Critical** | Multiple values `True` | Widespread numerical instability |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO training loop, non-finite value check |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._ppo_update()` |
| **Line(s)** | ~526-605 |

```python
# Per-head NaN/Inf tracking (for indicator lights)
# OR across all epochs - once detected, stays detected for this update
head_nan_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}
head_inf_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}

# ... in epoch loop ...

# Check new log_probs - separate NaN from Inf per head
# Fast path: only drill down when isfinite fails (preserves 0 syncs in happy path)
for key in HEAD_NAMES:
    lp = log_probs[key]
    if not torch.isfinite(lp).all():
        # Slow path: distinguish NaN from Inf
        if torch.isnan(lp).any():
            head_nan_detected[key] = True
            nonfinite_sources.append(f"log_probs[{key}]: NaN detected")
        if torch.isinf(lp).any():
            head_inf_detected[key] = True
            nonfinite_sources.append(f"log_probs[{key}]: Inf detected")
        nonfinite_found = True
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Added to `aggregated_result` dict | `simic/agent/ppo.py:995` |
| **2. Collection** | `emit_ppo_update_event()` extracts from metrics | `simic/telemetry/emitters.py:861` |
| **3. Aggregation** | `_handle_ppo_update()` OR-latches to TamiyoState | `karn/sanctum/aggregator.py:868-872` |
| **4. Delivery** | `get_snapshot()` returns TamiyoState with latches | `karn/sanctum/aggregator.py:398` |

```
[PPOAgent._ppo_update()]
    --aggregated_result["head_nan_detected"]-->
[TelemetryEmitter.emit_ppo_update_event()]
    --PPOUpdatePayload.head_nan_detected-->
[SanctumAggregator._handle_ppo_update()]
    --OR-latch-->
[TamiyoState.head_nan_latch]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `head_nan_latch` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.head_nan_latch` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~881-883 |

```python
# Per-head NaN/Inf latch (indicator lights - once True, stays True for entire run)
# Pre-populated with all HEAD_NAMES keys to enable direct access without .get()
# Keys: slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op
head_nan_latch: dict[str, bool] = field(
    default_factory=lambda: {head: False for head in HEAD_NAMES}
)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py` | NaN indicator row (lines 431-445) |

Display rendering:
```python
# Row 8: NaN indicator row
result.append("NaN   ", style="dim")
result.append(" " * self._PRE_OP_GUTTER, style="dim")
for col_idx, (head_key, _, _, width, _) in enumerate(HEAD_CONFIG):
    leyline_key = DISPLAY_TO_LEYLINE_KEY[head_key]
    latched = tamiyo.head_nan_latch[leyline_key]
    indicator = "●" if latched else "○"
    style = "red bold" if latched else "dim"
    # ... render indicator ...
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPOAgent checks `torch.isnan()` per head and populates dict
- [x] **Transport works** — `head_nan_detected` flows through PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.head_nan_latch: dict[str, bool]`
- [x] **Default is correct** — Pre-populated with `{head: False for head in HEAD_NAMES}`
- [x] **Consumer reads it** — ActionHeadsPanel accesses `snapshot.tamiyo.head_nan_latch[key]`
- [x] **Display is correct** — Shows ○ (dim) when False, ● (red bold) when True
- [x] **Thresholds applied** — Red styling when latched

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/agent/test_ppo_nan_detection.py` | `test_detect_nan_per_head` | `[x]` |
| Unit (emitter) | `tests/simic/agent/test_ppo_nan_detection.py` | `test_detect_nan_and_inf_same_tensor` | `[x]` |
| Unit (emitter) | `tests/simic/agent/test_ppo_nan_detection.py` | `test_clean_tensor_fast_path` | `[x]` |
| Unit (payload) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_has_per_head_nan_inf_flags` | `[x]` |
| Unit (payload) | `tests/leyline/test_telemetry.py` | `test_ppo_update_payload_per_head_nan_inf_defaults_to_none` | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_tamiyo_state_has_nan_inf_latch_fields` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_aggregator_latches_per_head_nan_inf` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_aggregator_latches_both_nan_and_inf_same_head` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py` | `test_heads_panel_shows_nan_inf_indicator_rows` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py` | `test_heads_panel_shows_all_clear_when_no_nan_inf` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe ActionHeadsPanel - look for NaN/Inf indicator rows below State row
4. Verify all indicators show ○ (empty circle, dim) in healthy training
5. To verify red latch behavior, inject NaN into training (requires code modification)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after PPO update with non-finite check |
| `HEAD_NAMES` | constant | Defines the set of action heads to track |
| `torch.isnan()` | PyTorch | Used for NaN detection in log_probs tensors |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-303` head_inf_latch | telemetry | Companion field for Inf detection (same structure) |
| ActionHeadsPanel NaN row | display | Visual indicator in TUI |
| Gradient health diagnostics | analysis | Part of overall gradient health assessment |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Claude | Created per-head NaN/Inf indicator implementation |
| 2026-01-03 | Audit | Created telemetry record during audit |

---

## 8. Notes

> **Design Decision: OR-Latch Behavior**
> The latch never resets during a training run. This is intentional - NaN occurrences may be
> transient (masked by subsequent operations), but the operator should know they happened.
> A training run that had any NaN should be investigated, even if it appeared to recover.

> **Design Decision: Pre-populated Dict**
> The `head_nan_latch` dict is pre-populated with all `HEAD_NAMES` keys set to False at
> TamiyoState construction. This allows display code to access keys directly without `.get()`
> fallbacks, per CLAUDE.md prohibition on defensive programming patterns.

> **Performance: Fast-Path Detection**
> The PPO loop uses a fast-path pattern: `torch.isfinite()` is checked first (single GPU sync),
> and only if non-finite values are found does it drill down with `torch.isnan()` and
> `torch.isinf()` (additional syncs). In healthy training (no NaN/Inf), there are 0 extra syncs.

> **Related Field: TELE-303 head_inf_latch**
> The companion field `head_inf_latch` has identical structure and behavior but tracks
> Inf (infinity) values instead of NaN. Both are displayed in the ActionHeadsPanel.

> **Display Format:**
> ```
>              Op       Slot     Blueprint  Style    Tempo    αTarget  αSpeed   Curve
> ...
> State          ●          ●           ●        ○        ●         ●        ●       ●
> NaN            ○          ○           ○        ●        ○         ○        ○       ○
> Inf            ○          ○           ○        ○        ○         ○        ○       ○
> ```
> - `○` = Clear (dim gray) - no NaN ever detected for this head
> - `●` = Latched (bright red bold) - NaN was detected at some point during this run
