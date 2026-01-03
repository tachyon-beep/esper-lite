# Telemetry Record: [TELE-552] Seed Epochs In Stage

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-552` |
| **Name** | Seed Epochs In Stage |
| **Category** | `seed` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "How long has this seed been in its current lifecycle stage?"

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
| **Type** | `int` |
| **Units** | epochs |
| **Range** | `[0, MAX_EPOCHS_IN_STAGE]` (typically 0-150) |
| **Precision** | integer (no decimals) |
| **Default** | `0` (just entered stage) |

### Semantic Meaning

> Epochs in stage tracks how many training epochs a seed has spent in its current lifecycle stage. This counter:
>
> - **Resets to 0** when stage transitions occur
> - **Increments by 1** each epoch the seed remains in the same stage
>
> Use cases:
> - **TRAINING:** "e5" means seed has been training for 5 epochs
> - **BLENDING:** Progress through blend tempo (e.g., 3/5 epochs done)
> - **HOLDING:** Evaluation period duration
>
> Stage-specific semantics:
> - **TRAINING:** Learning progress; higher = more developed seed
> - **BLENDING:** Blend progress; related to alpha via tempo
> - **HOLDING:** Probation duration; affects fossilize decision
> - **EMBARGOED:** Cooldown remaining; higher = longer since prune

### Display Format in EnvOverview

| Stage | Display | Example |
|-------|---------|---------|
| TRAINING | `e{N}` suffix | `Train:conv_l e5` |
| GERMINATED | `e{N}` suffix | `Germi:attent e2` |
| BLENDING | Not shown (alpha is more relevant) | `Blend:conv_l 0.3` |
| HOLDING | Not shown (alpha is more relevant) | `Hold:conv_l 0.7` |
| FOSSILIZED | Not shown (seed is permanent) | `Foss:conv_l` |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | SeedMetrics.epochs_in_current_stage, incremented per epoch |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedMetrics` |
| **Field** | `epochs_in_current_stage: int` (line 135) |

```python
@dataclass(slots=True)
class SeedMetrics:
    """Metrics tracked for a seed throughout its lifecycle."""
    epochs_total: int = 0
    epochs_in_current_stage: int = 0  # Reset on stage transition
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Increment** | SeedSlot.step_epoch() increments counter | `kasmina/slot.py` |
| **2. Reset** | SeedSlot.advance_stage() resets to 0 | `kasmina/slot.py` |
| **3. Telemetry** | SeedState.sync_telemetry() copies to telemetry | `kasmina/slot.py` (line 457) |
| **4. Schema** | SeedState.epochs_in_stage in Sanctum | `karn/sanctum/schema.py` (line 403) |
| **5. Display** | EnvOverview._format_slot_cell() | `karn/sanctum/widgets/env_overview.py` (lines 776-777) |

```
[SeedSlot.step_epoch()]
  --increments-->
  [SeedMetrics.epochs_in_current_stage]
  --sync_telemetry-->
  [SeedTelemetry.epochs_in_stage]
  --SanctumAggregator-->
  [schema.SeedState.epochs_in_stage]
  --EnvOverview-->
  [slot column: "e5"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `epochs_in_stage` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].epochs_in_stage` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 403 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 776-777) | Slot column: `e{N}` suffix for TRAINING/GERMINATED stages |
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py` | Detailed stage duration display |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — SeedMetrics.epochs_in_current_stage tracked by SeedSlot
- [x] **Transport works** — sync_telemetry() copies value; aggregator populates schema
- [x] **Schema field exists** — `SeedState.epochs_in_stage: int = 0` at line 403
- [x] **Default is correct** — `0` represents just-entered stage
- [x] **Consumer reads it** — EnvOverview._format_slot_cell() accesses `seed.epochs_in_stage`
- [x] **Display is correct** — Format: `e5` for 5 epochs

### Display Code

```python
# From env_overview.py lines 776-777
# Other stages show epochs in stage with dim "-" for curve
epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""
base = f"[{style}]{stage_short}:{blueprint} [dim]{curve_glyph}[/dim]{epochs_str}[/{style}]"
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| SeedSlot.step_epoch() | lifecycle | Increments counter each epoch |
| Stage transitions | lifecycle | Counter resets to 0 on transition |
| Training loop | execution | Drives epoch advancement |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Slot column display | display | Shows training progress |
| PBRS rewards | training | Stage duration affects potential |
| Gate evaluations | lifecycle | Some gates require minimum epochs |
| Auto-prune logic | lifecycle | Timeout after MAX epochs in stage |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvOverview slot columns |

---

## 8. Notes

> **Reset Behavior:** The counter resets to 0 whenever the seed transitions to a new stage. This is handled in `SeedSlot.advance_stage()` which resets `metrics.epochs_in_current_stage` before entering the new stage.
>
> **Display Logic:** Epochs are only shown for stages where time-in-stage is the primary progress indicator:
> - TRAINING/GERMINATED: Show `e{N}` because epochs represent learning progress
> - BLENDING/HOLDING: Don't show epochs; alpha is the relevant progress metric
> - FOSSILIZED: Don't show epochs; seed is permanent, duration is historical
>
> **Normalization:** For observation features, epochs_in_stage is normalized by `MAX_EPOCHS_IN_STAGE` (from leyline) to produce `epochs_in_stage_norm` in [0.0, 1.0] range.
>
> **MAX_EPOCHS_IN_STAGE:** Defined in leyline as equal to `DEFAULT_EPISODE_LENGTH` (150), representing the theoretical maximum a seed could stay in one stage (entire episode).
