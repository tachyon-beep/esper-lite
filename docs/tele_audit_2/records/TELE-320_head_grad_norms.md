# Telemetry Record: [TELE-320] Per-Head Gradient Norms

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-320` |
| **Name** | Per-Head Gradient Norms |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are individual action heads receiving healthy gradient signal, or are some heads experiencing vanishing/exploding gradients while others appear normal?"

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
| **Type** | 8 `float` fields |
| **Units** | L2 norm (dimensionless) |
| **Range** | `[0.0, inf)` — typically `[0.0, 10.0]` |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` each |

### Fields

| Field Name | Description |
|------------|-------------|
| `head_op_grad_norm` | Gradient norm for operation (lifecycle) head |
| `head_slot_grad_norm` | Gradient norm for slot selection head |
| `head_blueprint_grad_norm` | Gradient norm for blueprint selection head |
| `head_style_grad_norm` | Gradient norm for blending style head |
| `head_tempo_grad_norm` | Gradient norm for training tempo head |
| `head_alpha_target_grad_norm` | Gradient norm for alpha target head |
| `head_alpha_speed_grad_norm` | Gradient norm for alpha speed head |
| `head_alpha_curve_grad_norm` | Gradient norm for alpha curve (easing) head |

### Semantic Meaning

> Per-head gradient L2 norm measures the magnitude of gradients flowing into each action head after backward pass but before gradient clipping. Computed as:
>
> `||grad||_2 = sqrt(sum(grad_i^2))` for all parameters in the head
>
> For multi-parameter heads, the individual parameter norms are stacked and a vector norm is computed over them.
>
> Low norms (< 0.01) indicate vanishing gradients — the head is not receiving learning signal.
> High norms (> 5.0, 10x clip norm) indicate exploding gradients — significant clipping will occur.
> NaN values indicate missing gradient data (no params had .grad set).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.1 <= grad_norm <= 2.0` | Normal operating range |
| **Warning (vanishing)** | `0.01 < grad_norm < 0.1` | Weak gradient signal |
| **Warning (exploding)** | `2.0 < grad_norm <= 5.0` | Elevated, will be moderately clipped |
| **Critical (dead)** | `grad_norm < 0.01` | Head receiving no learning signal |
| **Critical (exploding)** | `grad_norm > 5.0` | Severe clipping, gradient direction preserved but magnitude heavily scaled |

**Threshold Source:** `DEFAULT_EXPLODING_THRESHOLD = 10.0 * DEFAULT_MAX_GRAD_NORM = 5.0` from `src/esper/simic/telemetry/gradient_collector.py`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after `loss.backward()` but before `clip_grad_norm_()` |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` inner epoch loop |
| **Line(s)** | ~835-870 |

```python
# Collect all head norms as tensors (no .item() yet)
head_norm_tensors: list[torch.Tensor] = []
for head_module in head_modules:
    params_with_grad = [p for p in head_module.parameters() if p.grad is not None]
    if params_with_grad:
        norm_t = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad) for p in params_with_grad])
        )
    else:
        # BUG FIX: Use NaN to signal "no gradient data" instead of 0.0
        norm_t = torch.tensor(float("nan"), device=self.device)
    head_norm_tensors.append(norm_t)

# Single GPU->CPU sync: stack all norms, then .tolist()
all_norms = torch.stack(head_norm_tensors).cpu().tolist()
for head_name, grad_norm in zip(head_names, all_norms):
    head_grad_norm_history[head_name].append(grad_norm)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` with `head_*_grad_norm` fields | `simic/telemetry/emitters.py:774-843` |
| **2. Collection** | `PPOUpdatePayload` dataclass with optional fields | `leyline/telemetry.py:686-693` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py:885-910` |
| **4. Delivery** | Written to `TamiyoState.head_*_grad_norm` fields | `karn/sanctum/schema.py:905-912` |

```
                    head_grad_norms dict
                           ↓
[PPOAgent.update()] → [emit_ppo_update()] → [TelemetryEvent] → [Aggregator] → [TamiyoState.head_*_grad_norm]
                           ↓
                    Averages across epochs
```

**Averaging:** In `emit_ppo_update()`, per-epoch values are averaged:
```python
if "head_grad_norms" in metrics:
    for head, values in metrics["head_grad_norms"].items():
        avg_grad_norm = sum(values) / len(values)  # Fail on empty list
        head_grad_norms_avg[f"head_{head}_grad_norm"] = avg_grad_norm
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Fields** | `head_slot_grad_norm`, `head_blueprint_grad_norm`, `head_style_grad_norm`, `head_tempo_grad_norm`, `head_alpha_target_grad_norm`, `head_alpha_speed_grad_norm`, `head_alpha_curve_grad_norm`, `head_op_grad_norm` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.head_*_grad_norm` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 905-912 |

```python
# Per-head gradient norms (for multi-head gradient health)
head_slot_grad_norm: float = 0.0
head_blueprint_grad_norm: float = 0.0
head_style_grad_norm: float = 0.0
head_tempo_grad_norm: float = 0.0
head_alpha_target_grad_norm: float = 0.0
head_alpha_speed_grad_norm: float = 0.0
head_alpha_curve_grad_norm: float = 0.0
head_op_grad_norm: float = 0.0
```

### Previous Value Tracking (Trend Detection)

The schema also includes `*_prev` fields for each head gradient norm (lines 916-923):
```python
head_slot_grad_norm_prev: float = 0.0
head_blueprint_grad_norm_prev: float = 0.0
# ... etc
```

These are populated by the aggregator before updating the current value:
```python
if payload.head_slot_grad_norm is not None:
    self._tamiyo.head_slot_grad_norm_prev = self._tamiyo.head_slot_grad_norm
    self._tamiyo.head_slot_grad_norm = payload.head_slot_grad_norm
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | Displays gradient values with trend arrows and mini-bars |
| PolicyDiagnostics.vue | `overwatch/web/src/components/PolicyDiagnostics.vue` | Web dashboard grid display |

**TUI Display (ActionHeadsPanel):**
- Row 3: Gradient values with trend arrows (e.g., `0.131->` stable, `0.135↗` increasing, `0.275↘` decreasing)
- Row 4: Mini-bars showing gradient magnitude on log scale
- Row 7: State indicators synthesizing entropy + gradient health (e.g., `○` dead = collapsed entropy + vanishing grads)

**Web Display (PolicyDiagnostics.vue):**
- Grid of 8 head gradient norms with formatting to 2 decimal places

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPO agent computes per-head gradient norms after backward
- [x] **Transport works** — Values reach aggregator via `emit_ppo_update()`
- [x] **Schema field exists** — 8 fields defined in `TamiyoState`
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — ActionHeadsPanel and PolicyDiagnostics.vue access fields
- [x] **Display is correct** — Values render with appropriate formatting and trend arrows
- [x] **Thresholds applied** — ActionHeadsPanel uses gradient thresholds for coloring and state indicators

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | Various PPO tests verify gradient computation | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_aggregator_tracks_per_head_gradient_norms_with_prev` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_sanctum_head_gradients.py` | `test_head_gradients_finite_cuda_*` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |
| Web (Vue component) | `tests/karn/overwatch/web/src/components/__tests__/PolicyDiagnostics.spec.ts` | Grid display tests | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe ActionHeadsPanel "Grad" row (row 3)
4. Verify gradient values update after each PPO batch with trend arrows
5. Observe gradient mini-bars (row 4) showing magnitude
6. Check State row (row 7) for synthesized health indicators
7. Trigger low gradients to verify `○` dead indicator (red)
8. Trigger high gradients (>5.0) to verify `▲` exploding indicator (red bold)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after backward pass completes |
| `loss.backward()` | computation | Gradients must be computed first |
| Head module structure | architecture | Requires `FactoredRecurrentActorCritic` with named heads |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `head_*_grad_norm_prev` | telemetry | Previous values for trend detection |
| ActionHeadsPanel state indicator | display | Synthesizes entropy + gradient health |
| Gradient CV computation | metric | Uses all head norms for coefficient of variation |
| `ppo_updates` SQL view | analytics | Exposes fields for Karn MCP queries |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-MM-DD | Initial | Created with Policy V2 implementation |
| 2024-MM-DD | B11-PT-01 | Fixed NaN gradient norms under AMP/torch.compile |
| 2025-01-03 | Audit | Verified wiring in telemetry audit |

---

## 8. Notes

> **Design Decision:** Gradient norms are averaged across PPO inner epochs (typically 3) rather than reported per-epoch. This reduces noise while still providing timely feedback.
>
> **NaN Handling:** When no parameters have gradients (`.grad is None` for all), NaN is emitted rather than 0.0 to distinguish "no data" from "vanishing gradients."
>
> **Trend Detection:** The `*_prev` fields enable trend arrows in the TUI:
> - `→` stable (delta < 0.01)
> - `↗` increasing (delta > 0.01)
> - `↘` decreasing (delta < -0.01)
>
> **Performance:** All 9 head norms (8 action + 1 value) are collected in a single GPU->CPU sync using `torch.stack(...).cpu().tolist()` to minimize synchronization overhead.
>
> **Related Metrics:**
> - `TELE-310` (grad_norm): Overall gradient norm across all parameters
> - `TELE-300` (nan_grad_count): Count of NaN gradients (global)
> - `TELE-301` (inf_grad_count): Count of Inf gradients (global)
