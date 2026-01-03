# Telemetry Record: [TELE-170] Per-Head Entropies

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-170` |
| **Name** | Per-Head Entropies |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which specific action heads are collapsing to deterministic behavior, while overall policy entropy may still appear healthy?"

The factored action policy has 8 independent heads (op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve). Individual heads can collapse while the aggregate entropy remains acceptable. Per-head entropy enables targeted diagnosis of which head is problematic.

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
| **Units** | nats (natural log base) |
| **Range** | `[0.0, HEAD_MAX_ENTROPIES[head]]` — varies per head (see table below) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` each |

### Fields

| Field | Schema Field | Max Entropy | Max Entropy Formula |
|-------|-------------|-------------|---------------------|
| Op | `head_op_entropy` | `log(6)` = 1.79 | 6 lifecycle ops |
| Slot | `head_slot_entropy` | `log(3)` = 1.10 | 3 default slots |
| Blueprint | `head_blueprint_entropy` | `log(13)` = 2.56 | 13 blueprints |
| Style | `head_style_entropy` | `log(4)` = 1.39 | 4 blending styles |
| Tempo | `head_tempo_entropy` | `log(3)` = 1.10 | 3 tempo options |
| Alpha Target | `head_alpha_target_entropy` | `log(3)` = 1.10 | 3 targets (50%, 70%, 100%) |
| Alpha Speed | `head_alpha_speed_entropy` | `log(4)` = 1.39 | 4 speeds |
| Alpha Curve | `head_alpha_curve_entropy` | `log(5)` = 1.61 | 5 easing curves |

### Semantic Meaning

> Per-head entropy measures the "spread" of each action head's output distribution. Computed as:
>
> H(head) = -Sum over a: pi(a|s) log pi(a|s)
>
> For each head, the distribution is over its action space (e.g., 6 ops, 13 blueprints).
> High entropy = exploring many actions equally. Low entropy = converging on specific actions.
> Maximum entropy = log(n_actions) for each head, stored in `HEAD_MAX_ENTROPIES` in leyline.

### Health Thresholds

Thresholds are applied to **normalized entropy** (entropy / max_entropy):

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `normalized > 0.5` | Head maintaining exploration |
| **Warning** | `0.25 < normalized <= 0.5` | Exploration declining, monitor closely |
| **Critical** | `normalized <= 0.25` | Head collapse occurring |

**Threshold Source:** `ActionHeadsPanel._entropy_color()` in `widgets/tamiyo_brain/action_heads_panel.py`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after action probability computation |
| **File** | `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py` |
| **Function/Method** | `FactoredRecurrentActorCritic.evaluate_actions()` |
| **Line(s)** | ~907 |

```python
# Entropy computed per head from MaskedCategorical distribution
dist = MaskedCategorical(logits=logits_flat, mask=mask_flat)
log_probs[key] = dist.log_prob(action_flat).reshape(batch, seq)
entropy[key] = dist.entropy().reshape(batch, seq)  # <-- Per-head entropy
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | PPO `_update_network()` averages per-head entropy across epochs | `simic/agent/ppo.py` ~648-651 |
| **2. Aggregation** | `head_entropy_history` dict collects values during PPO update | `simic/agent/ppo.py` ~512 |
| **3. Averaging** | Per-head entropies averaged in `emit_ppo_update()` | `simic/telemetry/emitters.py` ~765-769 |
| **4. Transport** | `PPOUpdatePayload` dataclass carries 8 optional fields | `leyline/telemetry.py` ~684-699 |
| **5. Aggregation** | `SanctumAggregator.handle_ppo_update()` writes to TamiyoState | `karn/sanctum/aggregator.py` ~883-924 |

```
[FactoredLSTM.evaluate_actions()]
        |
        v (entropy dict per timestep)
[PPO._update_network() inner loop]
        |
        v (head_entropy_history dict: {head: [values across epochs]})
[emit_ppo_update() averages to head_entropies_avg]
        |
        v (PPOUpdatePayload with 8 optional float fields)
[SanctumAggregator.handle_ppo_update()]
        |
        v
[TamiyoState.head_{name}_entropy fields]
```

### Key Code Locations

**PPO Inner Loop (accumulation):**
```python
# simic/agent/ppo.py ~648-651
head_entropy_tensors = [entropy[key].mean() for key in HEAD_NAMES]
head_entropy_values = torch.stack(head_entropy_tensors).cpu().tolist()
for key, val in zip(HEAD_NAMES, head_entropy_values):
    head_entropy_history[key].append(val)
```

**Emitter (averaging):**
```python
# simic/telemetry/emitters.py ~765-769
head_entropies_avg = {}
if "head_entropies" in metrics:
    for head, values in metrics["head_entropies"].items():
        avg_entropy = sum(values) / len(values)
        head_entropies_avg[f"head_{head}_entropy"] = avg_entropy
```

**Aggregator (assignment):**
```python
# karn/sanctum/aggregator.py ~883-924
if payload.head_slot_entropy is not None:
    self._tamiyo.head_slot_entropy = payload.head_slot_entropy
# ... (same pattern for all 8 heads)
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Fields** | `head_op_entropy`, `head_slot_entropy`, `head_blueprint_entropy`, `head_style_entropy`, `head_tempo_entropy`, `head_alpha_target_entropy`, `head_alpha_speed_entropy`, `head_alpha_curve_entropy` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.head_{name}_entropy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~895-902 |

```python
# karn/sanctum/schema.py ~894-902
# Per-head entropy (for multi-head policy collapse detection)
head_slot_entropy: float = 0.0
head_blueprint_entropy: float = 0.0
head_style_entropy: float = 0.0
head_tempo_entropy: float = 0.0
head_alpha_target_entropy: float = 0.0
head_alpha_speed_entropy: float = 0.0
head_alpha_curve_entropy: float = 0.0
head_op_entropy: float = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | Entropy row with normalized bar display |
| HeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | Head state classification (healthy/dead/confused/deterministic) |
| PolicyDiagnostics.vue | `overwatch/web/src/components/PolicyDiagnostics.vue` | Web dashboard display |

**TUI Display Details:**
- Entropy values shown with 3 decimal places
- Mini-bar visualization normalized to HEAD_MAX_ENTROPIES
- Color coding: green (>50%), yellow (25-50%), red (<25%)
- Sparse heads (blueprint, style, tempo, alpha) display coefficient prefix (e.g., "1.3x0.456")
- State indicator row synthesizes entropy + gradient health into symbols

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `FactoredRecurrentActorCritic.evaluate_actions()` computes entropy per head
- [x] **Transport works** — PPO accumulates in `head_entropy_history`, emitter averages to payload
- [x] **Schema field exists** — All 8 `head_{name}_entropy: float = 0.0` fields in TamiyoState
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — ActionHeadsPanel accesses via `getattr(tamiyo, ent_field)`
- [x] **Display is correct** — Values render with formatting, bar, and color
- [x] **Thresholds applied** — `_entropy_color()` applies normalized thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_per_head_entropies` | `[x]` |
| Unit (head state) | `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py` | `test_head_state_health_classification` | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe ACTION HEADS panel - "Entr" row
4. Verify all 8 head entropy values update after each PPO batch
5. Check entropy bars fill proportionally to max entropy
6. Verify color coding transitions (green/yellow/red) match normalized thresholds
7. Observe state indicators (Row 4) synthesize entropy + gradient into health symbols

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| MaskedCategorical distribution | computation | Entropy computed from masked logits |
| `HEAD_MAX_ENTROPIES` | constant | Normalization reference from leyline |
| Per-head action counts | config | Max entropy derived from enum sizes |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Head state indicators | display | Combines entropy + gradient for health classification |
| Per-head collapse detection | analysis | Individual heads can collapse while aggregate healthy |
| Differential entropy coefficients | training | Sparse heads (blueprint, tempo) use 1.3x entropy bonus |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation during telemetry audit |
| 2024-12 | Implementation | Added per-head entropy tracking (P3-1) in PPO update |
| 2024-12 | Implementation | ActionHeadsPanel unified display with entropy bars |

---

## 8. Notes

> **Design Decision:** Per-head entropies are averaged across PPO epochs within a batch update. This reduces noise while still capturing per-head trends.
>
> **Normalization:** The `HEAD_MAX_ENTROPIES` dict in leyline defines max entropy per head as log(n_actions). This is computed at module load time from enum sizes to prevent drift between action space changes and display thresholds.
>
> **Sparse Head Bonus:** Heads with fewer actions (blueprint=13, style=4, tempo=3) receive differential entropy coefficients (1.0-1.3x) displayed in the TUI. These coefficients are for training, not display thresholds.
>
> **State Synthesis:** The ActionHeadsPanel synthesizes per-head entropy with gradient norms to produce state indicators:
> - Healthy (green dot): Moderate entropy + normal gradients
> - Dead (red circle): Collapsed entropy + vanishing gradients
> - Confused (yellow half-moon): Very high entropy
> - Deterministic (diamond): Low entropy - yellow if concerning, dim if expected (single slot)
> - Exploding (red triangle): Gradient explosion regardless of entropy
>
> **Relationship to TELE-001:** TELE-001 tracks aggregate policy entropy (sum across heads). TELE-170 provides the per-head breakdown that reveals which specific head is problematic when aggregate entropy looks healthy.
