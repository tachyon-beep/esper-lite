# Telemetry Record: [TELE-551] Seed Alpha

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-551` |
| **Name** | Seed Alpha |
| **Category** | `seed` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "How much is this seed currently contributing to the host model's output?"

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
| **Units** | blend weight (dimensionless) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (seed not blending) |

### Semantic Meaning

> Alpha (α) is the blend weight controlling how much a seed module contributes to the host model's forward pass. It represents the "volume dial" for seed influence:
>
> - **α = 0.0:** Seed has zero output contribution (TRAINING stage - learning but isolated)
> - **α = 0.5:** Seed contributes equally with host baseline
> - **α = 1.0:** Seed is fully integrated (FOSSILIZED or max blend)
>
> Alpha progression:
> - **DORMANT/GERMINATED:** α = 0.0 (seed exists but not contributing)
> - **TRAINING:** α = 0.0 (seed trains in isolation, output discarded)
> - **BLENDING:** α ramps from 0.0 → target (controlled by alpha curve + tempo)
> - **HOLDING:** α held at target (evaluation period)
> - **FOSSILIZED:** α = 1.0 (permanent integration)

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| BLENDING | α increasing | Normal ramp-up, seed integrating |
| BLENDING | α static | Blend paused or tempo=SLOW |
| HOLDING | 0.5 ≤ α ≤ 1.0 | Normal evaluation range |
| FOSSILIZED | α = 1.0 | Expected (permanent integration) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | AlphaController state, updated per epoch |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `alpha: float` (line 389) |

```python
@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""
    alpha: float = 0.0
    alpha_controller: AlphaController = field(default_factory=AlphaController)
```

The alpha value is synchronized from `AlphaController.alpha` after each epoch step:

```python
# From slot.py __post_init__
self.alpha_controller.alpha = self.alpha
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | AlphaController.step() computes new alpha | `kasmina/alpha_controller.py` |
| **2. Storage** | SeedState.alpha updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.alpha field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.alpha in Sanctum | `karn/sanctum/schema.py` (line 395) |
| **5. Display** | EnvOverview._format_slot_cell() | `karn/sanctum/widgets/env_overview.py` (lines 758-760) |

```
[AlphaController.step()]
  --updates-->
  [SeedState.alpha]
  --EPOCH_COMPLETED event-->
  [SeedTelemetry.alpha]
  --SanctumAggregator-->
  [schema.SeedState.alpha]
  --EnvOverview-->
  [slot column: "0.3"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `alpha` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].alpha` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 395 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 758-760, 764-766) | Slot column: alpha shown for BLENDING/HOLDING stages |
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py` | Detailed alpha display with progress bar |
| Scoreboard | `widgets/scoreboard.py` | Alpha summary across all slots |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — AlphaController computes alpha per epoch
- [x] **Transport works** — SeedState.alpha synced from controller; telemetry carries value
- [x] **Schema field exists** — `SeedState.alpha: float = 0.0` at line 395
- [x] **Default is correct** — `0.0` is correct for non-blending stages
- [x] **Consumer reads it** — EnvOverview._format_slot_cell() displays `seed.alpha`
- [x] **Display is correct** — Format: `0.3` (1 decimal place)

### Display Code

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
| AlphaController.step() | computation | Computes alpha from curve + progress |
| Alpha curve | parameter | LINEAR/COSINE/SIGMOID shapes the ramp |
| Blend tempo | parameter | Epochs for full 0→1 transition |
| Seed stage | lifecycle | Alpha only changes during BLENDING |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Slot column display | display | Shows current blend weight |
| Counterfactual engine | analytics | Uses alpha for contribution isolation |
| Reward computation | training | Alpha affects seed_contribution calculation |
| Gate decisions | lifecycle | Alpha thresholds gate BLENDING→HOLDING |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvOverview slot columns |

---

## 8. Notes

> **Alpha Semantics:** Alpha represents "blending progress" in the context of the alpha controller, not necessarily the actual per-sample blend value. For GatedBlend (GATE algorithm), the actual per-sample alpha is learned and input-dependent - the AlphaController's alpha represents the timeline/progress of the blend, not the emergent gate values.
>
> **Display Contexts:**
> - BLENDING: `"Blend:conv_l ⌢ ▸▸ 0.3"` — alpha shown with curve glyph and tempo arrows
> - HOLDING: `"Hold:conv_l ⌢ ▸▸ 0.7"` — alpha shown (blend complete, evaluating)
> - FOSSILIZED: Alpha not shown (implied 1.0)
> - TRAINING/GERMINATED: Alpha not shown (implied 0.0)
>
> **Update Frequency:** Alpha changes once per epoch during BLENDING stage. In other stages, it remains constant (0.0 for pre-blend, held value for HOLDING, 1.0 for FOSSILIZED).
