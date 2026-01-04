# Telemetry Record: [TELE-523] Counterfactual Total Synergy

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-523` |
| **Name** | Counterfactual Total Synergy |
| **Category** | `attribution` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are the seeds working together (synergy) or interfering with each other (negative synergy)?"

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
| **Type** | `float` |
| **Units** | percentage points |
| **Range** | Unbounded (typically -10.0 to +10.0) |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before first computation) |

### Semantic Meaning

> Total synergy measures the interaction effect between seeds in the ensemble.
>
> **Formula:**
> ```
> synergy = combined_accuracy - baseline_accuracy - sum(individual_contributions)
> ```
>
> Where `individual_contribution[i] = accuracy_with_seed_i_only - baseline`
>
> **Interpretation:**
> - **Synergy > 0:** Seeds amplify each other's effects (the whole is greater than the sum of parts)
> - **Synergy = 0:** Seeds are independent (no interaction)
> - **Synergy < 0:** Seeds interfere with each other (the whole is LESS than the sum of parts)
>
> Negative synergy (interference) is a critical diagnostic signal — it indicates seeds are fighting
> each other, often due to conflicting learned representations or gradient interference.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `synergy > 0.5` | Strong positive synergy (green) |
| **Neutral** | `-0.5 <= synergy <= 0.5` | Independent or weak interaction (dim) |
| **Warning** | `synergy < -0.5` | Interference detected (red) |
| **Critical** | `synergy < -2.0` | Severe interference (red reverse) |

**Threshold Source:** `src/esper/karn/sanctum/widgets/counterfactual_panel.py` — lines 175-185

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
# PLANNED: Synergy computed from factorial results
synergy = combined - baseline - sum(individual_contributions)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TBD - Counterfactual engine event | TBD |
| **2. Collection** | TBD - All CounterfactualConfigs needed | TBD |
| **3. Aggregation** | TBD - Aggregator handler | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Derived via `total_synergy()` method | `karn/sanctum/schema.py` |

```
[Counterfactual Engine]
  --CounterfactualConfigs-->
  [SanctumAggregator]
  --env_state.counterfactual_matrix-->
  [CounterfactualSnapshot.total_synergy() (method)]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `CounterfactualSnapshot` |
| **Field** | `total_synergy()` (method) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].counterfactual_matrix.total_synergy()` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 376-382 |
| **Default Value** | `0.0` (when insufficient configs) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| CounterfactualPanel | `widgets/counterfactual_panel.py` (line 98) | Computed for synergy display |
| CounterfactualPanel | `widgets/counterfactual_panel.py` (lines 175-185) | Synergy/interference indicator with color coding |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Not yet implemented (counterfactual engine TBD)
- [ ] **Transport works** — No event emission path exists
- [x] **Schema field exists** — `CounterfactualSnapshot.total_synergy()` method at lines 376-382
- [x] **Default is correct** — Returns `0.0` when insufficient data
- [x] **Consumer reads it** — CounterfactualPanel calls `self._matrix.total_synergy()`
- [x] **Display is correct** — Shows synergy indicator with color coding
- [x] **Thresholds applied** — `>0.5` (green), `<-0.5` (red), else dim

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | TBD | TBD | `[ ]` |
| Unit (aggregator) | TBD | TBD | `[ ]` |
| Unit (schema method) | `tests/karn/sanctum/test_schema.py` | `test_counterfactual_total_synergy` | `[ ]` |
| Widget (CounterfactualPanel) | `tests/karn/sanctum/widgets/test_counterfactual_panel.py` | Synergy rendering | `[ ]` |

### Manual Verification Steps

1. Start training with multiple seeds
2. Launch Sanctum TUI
3. After first episode completes, observe CounterfactualPanel
4. Verify synergy line shows:
   - Green checkmark + positive value for synergy
   - Red "INTERFERENCE DETECTED" for negative synergy
   - Dim neutral text for near-zero

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| baseline_accuracy (TELE-521) | computation | Needed for synergy formula |
| combined_accuracy (TELE-522) | computation | Needed for synergy formula |
| individual_contributions (TELE-524) | computation | Sum needed for synergy formula |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Interference detection UI | display | Red banner for negative synergy |
| Synergy indicator | display | Green/dim/red coloring |
| Automated pruning decisions | system | Future: prune interfering seeds |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** This method exists in the schema but relies on configs list which has NO emitter implemented.
>
> **Single-Seed Edge Case:** When only one seed is active (n_seeds=1), synergy is mathematically zero because there are no pair interactions. The widget correctly suppresses interference warnings for this case (line 175: `n_seeds >= 2`).
>
> **Diagnostic Priority:** Interference (negative synergy) is MORE critical to surface than positive synergy. The UI uses bold red reverse styling for interference because it indicates seeds are actively hurting each other — this needs immediate operator attention.
>
> **Formula Derivation:**
> ```
> expected_combined = baseline + sum(individual_contributions)
> actual_combined = combined_accuracy
> synergy = actual_combined - expected_combined
>         = combined - baseline - sum(individuals)
> ```
