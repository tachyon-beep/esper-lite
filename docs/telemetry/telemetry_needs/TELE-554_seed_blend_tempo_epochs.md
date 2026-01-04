# Telemetry Record: [TELE-554] Seed Blend Tempo Epochs

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-554` |
| **Name** | Seed Blend Tempo Epochs |
| **Category** | `seed` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How quickly is this seed being blended into the host model?"

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
| **Type** | `int | None` |
| **Units** | epochs |
| **Values** | `3` (FAST), `5` (STANDARD), `8` (SLOW), or `None` |
| **Precision** | integer |
| **Default** | `5` (STANDARD) |

### Semantic Meaning

> Blend tempo defines how many epochs the alpha ramp takes to complete during BLENDING stage. This controls the integration speed:
>
> - **FAST (3 epochs):** Rapid integration, quick signal, higher instability risk
> - **STANDARD (5 epochs):** Balanced approach (default)
> - **SLOW (8 epochs):** Gradual integration, better stability assessment, longer investment
>
> Tempo is selected at GERMINATE time (via Tamiyo's tempo action head) and remains fixed for the seed's lifetime.

### Display Format (Tempo Arrows)

Tempo is displayed as arrow glyphs indicating blend speed:

| Tempo | Epochs | Glyph | Meaning |
|-------|--------|-------|---------|
| FAST | ≤3 | `▸▸▸` | Triple arrows = rapid |
| STANDARD | ≤5 | `▸▸` | Double arrows = normal |
| SLOW | >5 | `▸` | Single arrow = gradual |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Kasmina SeedState.blend_tempo_epochs, set at germination |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `blend_tempo_epochs: int = 5` (line 398) |

```python
@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""
    blend_tempo_epochs: int = 5  # Default to STANDARD (5 epochs)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Selection** | Tamiyo tempo action head selects TempoAction | `tamiyo/policy/network.py` |
| **2. Mapping** | TEMPO_TO_EPOCHS maps action to epoch count | `leyline/factored_actions.py` (lines 98-101) |
| **3. Storage** | SeedState.blend_tempo_epochs stores value | `kasmina/slot.py` |
| **4. Telemetry** | SeedGerminatedPayload carries tempo | `leyline/telemetry.py` |
| **5. Schema** | SeedState.blend_tempo_epochs in Sanctum | `karn/sanctum/schema.py` (line 411) |
| **6. Display** | EnvOverview._format_slot_cell() computes arrows | `karn/sanctum/widgets/env_overview.py` (lines 752-755) |

```
[Tamiyo tempo action: TempoAction.FAST]
  --TEMPO_TO_EPOCHS-->
  [3 epochs]
  --GERMINATE-->
  [SeedState.blend_tempo_epochs = 3]
  --SEED_GERMINATED event-->
  [SeedGerminatedPayload]
  --SanctumAggregator-->
  [schema.SeedState.blend_tempo_epochs]
  --_tempo_arrows()-->
  [slot column: "▸▸▸"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `blend_tempo_epochs` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].blend_tempo_epochs` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 411 |
| **Default Value** | `5` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 752-755, 759-760) | Slot column: tempo arrows for BLENDING/HOLDING/FOSSILIZED |
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py` | Detailed tempo display |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — SeedState.blend_tempo_epochs set at germination
- [x] **Transport works** — SeedGerminatedPayload carries tempo; aggregator populates schema
- [x] **Schema field exists** — `SeedState.blend_tempo_epochs: int = 5` at line 411
- [x] **Default is correct** — `5` (STANDARD) is reasonable default
- [x] **Consumer reads it** — EnvOverview._format_slot_cell() accesses `seed.blend_tempo_epochs`
- [x] **Display is correct** — Tempo arrows via `_tempo_arrows()` helper

### Display Code

```python
# From env_overview.py lines 752-755
def _tempo_arrows(tempo: int | None) -> str:
    if tempo is None:
        return ""
    return "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
```

### Usage in Slot Cell

```python
# From env_overview.py lines 758-760 (BLENDING stage)
if seed.stage == "BLENDING" and seed.alpha > 0:
    tempo_arrows = _tempo_arrows(seed.blend_tempo_epochs)
    base = f"[{style}]{stage_short}:{blueprint} {curve_glyph} {tempo_arrows} {seed.alpha:.1f}[/{style}]"
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Tamiyo tempo action head | action | Selects tempo at germination |
| TEMPO_TO_EPOCHS | constants | Maps TempoAction enum to epoch count |
| Germination event | lifecycle | Tempo fixed at seed creation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Slot column display | display | Shows tempo arrows |
| Alpha progression | computation | Tempo determines steps_total for ramp |
| Blend stability | training | Faster tempo = higher instability risk |
| Credit assignment | training | Tempo affects when blend rewards arrive |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvOverview slot columns |

---

## 8. Notes

> **Tempo vs Alpha Relationship:** Blend tempo determines `alpha_steps_total` for the AlphaController. Combined with alpha_curve, tempo controls both:
> - **Duration:** How many epochs from α=0 to α=target
> - **Speed:** Fast tempo = steeper α(t) slope
>
> **Display Contexts:** Tempo arrows are shown for stages where blend history is relevant:
> - BLENDING: Active blend, shows current tempo
> - HOLDING: Blend complete, shows how fast it integrated
> - FOSSILIZED: Historical record of integration speed
>
> **Strategic Implications:**
> - **FAST (▸▸▸):** Quick feedback for agent, but risky if seed destabilizes host
> - **STANDARD (▸▸):** Balanced approach, default choice
> - **SLOW (▸):** Conservative, good for high-value seeds where stability matters
>
> **TEMPO_TO_EPOCHS Mapping:**
> ```python
> TEMPO_TO_EPOCHS: dict[TempoAction, int] = {
>     TempoAction.FAST: 3,
>     TempoAction.STANDARD: 5,
>     TempoAction.SLOW: 8,
> }
> ```
>
> **Immutability:** Blend tempo is set at germination and cannot be changed. This simplifies credit assignment - Tamiyo commits to a tempo upfront and receives reward signals based on that choice.
