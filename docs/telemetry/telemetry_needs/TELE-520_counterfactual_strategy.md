# Telemetry Record: [TELE-520] Counterfactual Strategy

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-520` |
| **Name** | Counterfactual Strategy |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What counterfactual analysis method was used for seed attribution in this environment?"

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
| **Type** | `str` |
| **Units** | enum value |
| **Range** | `"unavailable"`, `"full_factorial"`, `"ablation_only"` |
| **Precision** | N/A (categorical) |
| **Default** | `"unavailable"` |

### Semantic Meaning

> Indicates which counterfactual evaluation strategy was used:
>
> - **"unavailable":** No counterfactual data available (no active seeds or not yet computed)
> - **"full_factorial":** Complete 2^n evaluation of all seed combinations (computed at episode end)
> - **"ablation_only":** Live ablation estimates based on cached baselines (approximate, available mid-episode)
>
> Full factorial provides exact attribution but is expensive (O(2^n) evaluations). Ablation-only provides estimates during training but lacks pair interaction data until episode end.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == "full_factorial"` | Complete attribution data available |
| **Good** | `value == "ablation_only"` | Approximate data available (mid-episode) |
| **Warning** | `value == "unavailable"` | No attribution data (no active seeds or pre-warmup) |
| **Critical** | N/A | Not applicable for this field |

**Threshold Source:** `src/esper/karn/sanctum/widgets/counterfactual_panel.py` — strategy check at line 46

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Counterfactual engine (not yet implemented) |
| **File** | TBD |
| **Function/Method** | TBD |
| **Line(s)** | TBD |

```python
# PLANNED: Counterfactual engine will emit this field
# Example emission structure:
CounterfactualSnapshot(
    strategy="full_factorial",  # or "ablation_only", "unavailable"
    ...
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TBD - Counterfactual engine event | TBD |
| **2. Collection** | TBD - Payload with strategy field | TBD |
| **3. Aggregation** | TBD - Aggregator handler | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `env_state.counterfactual_matrix.strategy` | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --CounterfactualSnapshot.strategy-->
  [TBD Event Emission]
  --TBD Payload-->
  [SanctumAggregator.handle_counterfactual()]
  --env_state.counterfactual_matrix-->
  [EnvState.counterfactual_matrix.strategy]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `CounterfactualSnapshot` |
| **Field** | `strategy` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].counterfactual_matrix.strategy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 324 |
| **Default Value** | `"unavailable"` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 46) | Determines whether to render unavailable state or waterfall visualization |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (lines 100-106) | Shows "Live Ablation Analysis" indicator for ablation_only mode |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Not yet implemented (counterfactual engine TBD)
- [ ] **Transport works** — No event emission path exists
- [x] **Schema field exists** — `CounterfactualSnapshot.strategy: str = "unavailable"` at line 324
- [x] **Default is correct** — `"unavailable"` is appropriate before data is available
- [x] **Consumer reads it** — CounterfactualPanel directly accesses `self._matrix.strategy`
- [x] **Display is correct** — Panel shows appropriate visualization based on strategy
- [x] **Thresholds applied** — Panel checks for "unavailable" and "ablation_only" states

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | TBD | TBD | `[ ]` |
| Unit (aggregator) | TBD | TBD | `[ ]` |
| Integration (end-to-end) | TBD | TBD | `[ ]` |
| Widget (CounterfactualPanel) | `tests/karn/sanctum/widgets/test_counterfactual_panel.py` | Strategy rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe CounterfactualPanel — should show "unavailable" state initially
4. After seeds germinate and episode ends, strategy should change to "full_factorial"
5. Mid-episode with active seeds, strategy may show "ablation_only"

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Active seeds | state | Strategy is "unavailable" without germinated seeds |
| Counterfactual engine | computation | Engine must compute factorial or ablation analysis |
| Episode boundary | event | Full factorial computed at episode end |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| CounterfactualPanel render mode | display | Determines waterfall vs unavailable view |
| Ablation indicator | display | Shows "(estimates based on cached baselines)" for ablation_only |
| Pair contribution display | display | Pairs only available with full_factorial |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** This field exists in the schema but has NO emitter implemented. The counterfactual engine that would populate this field is not yet built.
>
> **Design Intent:** The strategy field enables the UI to provide appropriate context:
> - "unavailable" → Show placeholder text, prevent misleading zero displays
> - "ablation_only" → Show approximation disclaimer, hide pair interactions
> - "full_factorial" → Show complete analysis with pair synergies
>
> **Implementation Priority:** This is a P1 field because counterfactual analysis is a key debugging tool for understanding seed ensemble behavior.
