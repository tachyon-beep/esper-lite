# Telemetry Record: [TELE-553] Seed Alpha Curve

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-553` |
| **Name** | Seed Alpha Curve |
| **Category** | `seed` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "What shape of alpha ramp is being used to blend this seed into the host?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
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
| **Units** | curve type identifier |
| **Values** | `"LINEAR"`, `"COSINE"`, `"SIGMOID_GENTLE"`, `"SIGMOID"`, `"SIGMOID_SHARP"` |
| **Precision** | N/A (string) |
| **Default** | `"LINEAR"` |

### Semantic Meaning

> Alpha curve determines the shape of the alpha ramp during BLENDING stage. Different curves affect how quickly the seed's contribution increases:
>
> - **LINEAR:** Constant rate increase (α increases uniformly per epoch)
> - **COSINE:** Ease-in/ease-out (slow start, fast middle, slow end)
> - **SIGMOID_GENTLE:** Gradual S-curve with slow start/end (steepness=6)
> - **SIGMOID:** Standard S-curve (steepness=12, default)
> - **SIGMOID_SHARP:** Near-step function with rapid transition (steepness=24)
>
> The curve is selected at GERMINATE time (via Tamiyo's alpha_curve head) and remains fixed for the seed's lifetime.

### Display Glyphs

Curves are displayed as single-character glyphs in EnvOverview slot columns:

| Curve | Glyph | Rationale |
|-------|-------|-----------|
| LINEAR | `╱` | Diagonal line = constant rate |
| COSINE | `∿` | Wave = oscillation/ease |
| SIGMOID_GENTLE | `⌒` | Wide top arc = slow ends |
| SIGMOID | `⌢` | Narrow bottom arc = moderate S |
| SIGMOID_SHARP | `⊐` | Squared bracket = near-step |

**Source:** `leyline/factored_actions.py` → `ALPHA_CURVE_GLYPHS`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | AlphaController.alpha_curve, set at BLENDING start |
| **File** | `/home/john/esper-lite/src/esper/kasmina/alpha_controller.py` |
| **Field** | `alpha_curve: AlphaCurve` |

The curve is stored as a string in Sanctum schema for display purposes:

```python
# From sanctum/schema.py line 414
alpha_curve: str = "LINEAR"  # Maps to ALPHA_CURVE_GLYPHS
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Selection** | Tamiyo alpha_curve action head selects curve | `tamiyo/policy/network.py` |
| **2. Storage** | AlphaController stores curve for ramp computation | `kasmina/alpha_controller.py` |
| **3. Telemetry** | BLENDING stage change event carries curve | `leyline/telemetry.py` |
| **4. Schema** | SeedState.alpha_curve in Sanctum | `karn/sanctum/schema.py` (line 414) |
| **5. Display** | EnvOverview._format_slot_cell() maps to glyph | `karn/sanctum/widgets/env_overview.py` (line 748) |

```
[Tamiyo alpha_curve action]
  --GERMINATE-->
  [AlphaController.alpha_curve]
  --BLENDING stage change-->
  [SeedStageChangedPayload]
  --SanctumAggregator-->
  [schema.SeedState.alpha_curve]
  --ALPHA_CURVE_GLYPHS-->
  [slot column: "⌢"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `alpha_curve` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].alpha_curve` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 414 |
| **Default Value** | `"LINEAR"` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (line 748) | Slot column: glyph shown for BLENDING/HOLDING/FOSSILIZED |
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py` | Full curve name in detailed view |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — AlphaController.alpha_curve stores curve type
- [x] **Transport works** — Stage change events carry curve; aggregator populates schema
- [x] **Schema field exists** — `SeedState.alpha_curve: str = "LINEAR"` at line 414
- [x] **Default is correct** — `"LINEAR"` is reasonable default
- [x] **Consumer reads it** — EnvOverview._format_slot_cell() accesses `seed.alpha_curve`
- [x] **Display is correct** — ALPHA_CURVE_GLYPHS maps to single-char glyphs

### Display Code

```python
# From env_overview.py line 748
# Curve glyph: shown for BLENDING/HOLDING/FOSSILIZED, dim "-" for other stages
curve_glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "−") if seed.stage in ("BLENDING", "HOLDING", "FOSSILIZED") else "−"
```

### Glyph Source

```python
# From leyline/factored_actions.py lines 200-206
ALPHA_CURVE_GLYPHS: dict[str, str] = {
    "LINEAR": "╱",
    "COSINE": "∿",
    "SIGMOID_GENTLE": "⌒",
    "SIGMOID": "⌢",
    "SIGMOID_SHARP": "⊐",
}
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Tamiyo alpha_curve head | action | Selects curve at germination time |
| AlphaController | computation | Stores curve for ramp calculation |
| ALPHA_CURVE_GLYPHS | constants | Maps curve names to display glyphs |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Slot column display | display | Shows curve glyph |
| Alpha progression | computation | Curve shapes the α(t) function |
| Blend stability | training | Sharper curves = higher risk of instability |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvOverview slot columns |

---

## 8. Notes

> **Curve Visibility:** The curve glyph is only shown for stages where it's causally relevant:
> - BLENDING: Curve actively shaping alpha progression
> - HOLDING: Curve was used, shows blend history
> - FOSSILIZED: Historical record of how seed was blended
> - Other stages: Dim `−` glyph (curve not active)
>
> **Design Rationale:** Glyphs are chosen to visually represent the curve shape:
> - LINEAR (`╱`): Straight diagonal = constant slope
> - COSINE (`∿`): Wave = smooth oscillation
> - SIGMOID variants: Arc/bracket shapes show transition sharpness
>
> **Steepness Parameter:** For SIGMOID curves, the steepness value (6/12/24) is embedded in the curve variant name. The ALPHA_CURVE_TO_STEEPNESS mapping provides the actual values used by the AlphaController.
>
> **Immutability:** Like blueprint_id, alpha_curve is set at germination (specifically when entering BLENDING) and remains fixed. This ensures consistent blend behavior throughout the seed's integration period.
