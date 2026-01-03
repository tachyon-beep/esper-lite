# Telemetry Record: [TELE-559] Seed Boost Received

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-559` |
| **Name** | Seed Boost Received |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which other seed is helping this seed the most?"

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
| **Type** | `float` |
| **Units** | interaction score (dimensionless) |
| **Range** | `[0.0, ∞)` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (no boost) |

### Semantic Meaning

> Boost received (max(I_ij) for j ≠ i) measures the strongest positive interaction this seed receives from any other seed:
>
> - **boost_received > 0:** Another seed is boosting this one's performance
> - **boost_received = 0:** No significant positive interactions from other seeds
>
> This identifies beneficial seed pairings in the ensemble.

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| BLENDING/HOLDING | boost_received > BOOST_RECEIVED_THRESHOLD (0.1) | Significant boost (cyan indicator) |
| BLENDING/HOLDING | boost_received ≤ 0.1 | No significant boost (not displayed) |
| TRAINING/GERMINATED | N/A | Not displayed (alpha=0) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Counterfactual engine max pairwise interaction |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `boost_received: float` (schema line 420) |

```python
@dataclass
class SeedState:
    """State of a single seed slot."""
    # Inter-slot interaction metrics (from counterfactual engine)
    boost_received: float = 0.0  # max(I_ij) for j ≠ i (strongest interaction partner)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | Counterfactual pairwise max extraction | `tamiyo/counterfactual.py` |
| **2. Storage** | SeedState.boost_received updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.boost_received field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.boost_received in Sanctum | `karn/sanctum/schema.py` (line 420) |
| **5. Display** | EnvDetailScreen SeedCard | `karn/sanctum/widgets/env_detail_screen.py` (lines 171, 180-181) |

```
[Counterfactual pairwise max]
  --extracts-->
  [SeedState.boost_received]
  --EPOCH_COMPLETED event-->
  [SeedTelemetry.boost_received]
  --SanctumAggregator-->
  [schema.SeedState.boost_received]
  --SeedCard-->
  ["(↗0.8)"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `boost_received` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].boost_received` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 420 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen SeedCard | `widgets/env_detail_screen.py` (lines 171, 180-181) | "(↗0.8)" in cyan appended to synergy |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Counterfactual engine extracts max pairwise interaction
- [x] **Transport works** — SeedState.boost_received updated; telemetry carries value
- [x] **Schema field exists** — `SeedState.boost_received: float = 0.0` at line 420
- [x] **Default is correct** — `0.0` is correct for no boost
- [x] **Consumer reads it** — SeedCard._render_active() displays `seed.boost_received`
- [x] **Display is correct** — Format: "(↗0.8)" cyan, only when above threshold

### Display Code

```python
# From env_detail_screen.py lines 169-181
if seed.stage in ("BLENDING", "HOLDING") and (
    seed.interaction_sum != 0
    or seed.boost_received > DisplayThresholds.BOOST_RECEIVED_THRESHOLD
):
    # ... synergy display ...
    # Add boost indicator if significant
    if seed.boost_received > DisplayThresholds.BOOST_RECEIVED_THRESHOLD:
        interaction_text.append(f" (↗{seed.boost_received:.1f})", style="cyan")
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Counterfactual engine | computation | Computes pairwise I_ij interactions |
| Other seeds' alpha | parameter | Boost comes from active seeds |
| Seed alpha | parameter | This seed must be active |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SeedCard display | display | Shows boost indicator alongside synergy |
| Ensemble analysis | analytics | Identifies beneficial seed pairings |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvDetailScreen SeedCard |

---

## 8. Notes

> **Mathematical Definition:** boost_received = max(I_ij) for all j ≠ i, where I_ij > 0. This extracts the strongest positive pairwise interaction.
>
> **Threshold:** DisplayThresholds.BOOST_RECEIVED_THRESHOLD = 0.1
> - Values above 0.1: Displayed as "(↗0.8)" in cyan
> - Values at or below 0.1: Not displayed
>
> **Display Context:** Boost indicator is appended to the synergy line when above threshold. It shows alongside interaction_sum to provide both total synergy and strongest individual partner.
>
> **Related Field:** interaction_sum (TELE-558) shows the total synergy from all interactions.
