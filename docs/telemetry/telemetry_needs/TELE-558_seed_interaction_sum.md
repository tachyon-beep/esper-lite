# Telemetry Record: [TELE-558] Seed Interaction Sum

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-558` |
| **Name** | Seed Interaction Sum |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How much does this seed synergize with other seeds in the ensemble?"

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
| **Units** | interaction score (dimensionless) |
| **Range** | `(-∞, +∞)` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (no interaction data) |

### Semantic Meaning

> Interaction sum (Σ I_ij for all j ≠ i) measures the total synergy this seed receives from inter-slot interactions with other seeds:
>
> - **interaction_sum > 0:** Net positive synergy — this seed benefits from other seeds
> - **interaction_sum = 0:** No interaction effects (seed operates independently)
> - **interaction_sum < 0:** Net negative synergy — this seed conflicts with other seeds
>
> This is computed by the counterfactual engine using pairwise interaction analysis.

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| BLENDING/HOLDING | interaction_sum > INTERACTION_SYNERGY_THRESHOLD (0.5) | Positive synergy (green) |
| BLENDING/HOLDING | interaction_sum < -INTERACTION_SYNERGY_THRESHOLD (-0.5) | Negative synergy (red) |
| BLENDING/HOLDING | |interaction_sum| ≤ 0.5 | Neutral (dim) |
| TRAINING/GERMINATED | N/A | Not displayed (alpha=0) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Counterfactual engine pairwise interaction analysis |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `interaction_sum: float` (schema line 419) |

```python
@dataclass
class SeedState:
    """State of a single seed slot."""
    # Inter-slot interaction metrics (from counterfactual engine)
    interaction_sum: float = 0.0  # Σ I_ij for all j ≠ i (total synergy from interactions)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | Counterfactual pairwise analysis | `tamiyo/counterfactual.py` |
| **2. Storage** | SeedState.interaction_sum updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.interaction_sum field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.interaction_sum in Sanctum | `karn/sanctum/schema.py` (line 419) |
| **5. Display** | EnvDetailScreen SeedCard | `karn/sanctum/widgets/env_detail_screen.py` (lines 168-178) |

```
[Counterfactual pairwise analysis]
  --sums-->
  [SeedState.interaction_sum]
  --EPOCH_COMPLETED event-->
  [SeedTelemetry.interaction_sum]
  --SanctumAggregator-->
  [schema.SeedState.interaction_sum]
  --SeedCard-->
  ["Synergy: +1.5"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `interaction_sum` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].interaction_sum` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 419 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen SeedCard | `widgets/env_detail_screen.py` (lines 168-178) | "Synergy: +1.5" with color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Counterfactual engine computes pairwise interactions
- [x] **Transport works** — SeedState.interaction_sum updated; telemetry carries value
- [x] **Schema field exists** — `SeedState.interaction_sum: float = 0.0` at line 419
- [x] **Default is correct** — `0.0` is correct for no interaction data
- [x] **Consumer reads it** — SeedCard._render_active() displays `seed.interaction_sum`
- [x] **Display is correct** — Format: "+1.5" green, "-0.8" red, "0.3" dim

### Display Code

```python
# From env_detail_screen.py lines 167-184
# Inter-slot interaction metrics (always visible, greyed out when not applicable)
interaction_text = Text("Synergy: ")
if seed.stage in ("BLENDING", "HOLDING") and (
    seed.interaction_sum != 0
    or seed.boost_received > DisplayThresholds.BOOST_RECEIVED_THRESHOLD
):
    if seed.interaction_sum > DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD:
        interaction_text.append(f"+{seed.interaction_sum:.1f}", style="green")
    elif seed.interaction_sum < -DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD:
        interaction_text.append(f"{seed.interaction_sum:.1f}", style="red")
    else:
        interaction_text.append(f"{seed.interaction_sum:.1f}", style="dim")
    # Add boost indicator if significant
    if seed.boost_received > DisplayThresholds.BOOST_RECEIVED_THRESHOLD:
        interaction_text.append(f" (↗{seed.boost_received:.1f})", style="cyan")
else:
    interaction_text.append("--", style="dim")
lines.append(interaction_text)
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Counterfactual engine | computation | Computes pairwise I_ij interactions |
| Other seeds' alpha | parameter | Interactions only exist with active seeds |
| Seed alpha | parameter | This seed must be active to have interactions |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SeedCard display | display | Shows synergy with color coding |
| Ensemble health | analytics | Negative synergy indicates conflicts |
| Prune decisions | training | Conflicting seeds may be pruned |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvDetailScreen SeedCard |

---

## 8. Notes

> **Mathematical Definition:** interaction_sum = Σ I_ij for all j ≠ i, where I_ij represents the interaction effect between seed i and seed j computed via counterfactual analysis.
>
> **Threshold:** DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD = 0.5
> - Values above 0.5: Displayed in green (positive synergy)
> - Values below -0.5: Displayed in red (negative synergy)
> - Values in [-0.5, 0.5]: Displayed dim (neutral)
>
> **Stage-Aware:** Only displayed for BLENDING and HOLDING stages. TRAINING/GERMINATED seeds have alpha=0 and cannot participate in interactions.
>
> **Related Field:** boost_received (TELE-559) shows the strongest individual interaction partner.
