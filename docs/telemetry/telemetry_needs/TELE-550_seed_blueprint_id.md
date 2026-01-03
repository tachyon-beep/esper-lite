# Telemetry Record: [TELE-550] Seed Blueprint ID

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-550` |
| **Name** | Seed Blueprint ID |
| **Category** | `seed` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What type of neural module blueprint is planted in this slot?"

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
| **Type** | `str | None` |
| **Units** | identifier string |
| **Values** | Blueprint names: `"conv_light"`, `"attention"`, `"norm"`, `"depthwise"`, `"bottleneck"`, `"conv_small"`, `"conv_heavy"`, `"lora"`, `"lora_large"`, `"mlp_small"`, `"mlp"`, `"flex_attention"`, `"noop"`, or `None` |
| **Precision** | N/A (string) |
| **Default** | `None` (no seed in slot) |

### Semantic Meaning

> Blueprint ID identifies the type of neural module template used when a seed is germinated. Each blueprint defines:
>
> - **Architecture**: Convolution, attention, normalization, etc.
> - **Size class**: light/small/heavy variants with different parameter counts
> - **Topology**: CNN-compatible or Transformer-compatible
>
> Examples:
> - `"conv_light"`: Lightweight 3x3 convolution seed (CNN hosts)
> - `"attention"`: Multi-head attention seed (Transformer hosts)
> - `"lora"`: Low-rank adaptation seed (parameter-efficient)
> - `None` or `"?"`: Slot is DORMANT (no active seed)

### Display Format in EnvOverview

| Stage | Display | Example |
|-------|---------|---------|
| DORMANT | `─` | No seed |
| Active | Truncated to 6 chars | `conv_l`, `attent`, `depthw` |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Kasmina SeedState, set at germination time |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `blueprint_id: str` (line 381) |

```python
@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""
    seed_id: str
    blueprint_id: str  # Set at germination, never changes
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Storage** | SeedState dataclass field | `kasmina/slot.py` |
| **2. Telemetry** | SeedTelemetry.blueprint_id | `leyline/telemetry.py` |
| **3. Schema** | SeedState.blueprint_id in Sanctum | `karn/sanctum/schema.py` (line 394) |
| **4. Display** | EnvOverview._format_slot_cell() | `karn/sanctum/widgets/env_overview.py` (line 731-733) |

```
[Kasmina SeedState.blueprint_id]
  --SEED_GERMINATED event-->
  [SeedGerminatedPayload.blueprint_id]
  --SanctumAggregator-->
  [schema.SeedState.blueprint_id]
  --EnvOverview-->
  [slot column display: "conv_l"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `blueprint_id` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].blueprint_id` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 394 |
| **Default Value** | `None` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 731-733) | Slot column display: truncated to 6 chars after stage abbreviation |
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py` | Full blueprint name in detailed slot view |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Kasmina SeedState stores blueprint_id at germination
- [x] **Transport works** — SeedGerminatedPayload carries blueprint_id; aggregator populates schema
- [x] **Schema field exists** — `SeedState.blueprint_id: str | None = None` at line 394
- [x] **Default is correct** — `None` represents DORMANT slot with no seed
- [x] **Consumer reads it** — EnvOverview._format_slot_cell() accesses `seed.blueprint_id`
- [x] **Display is correct** — Truncated to 6 chars: `"conv_light"` → `"conv_l"`

### Display Code

```python
# From env_overview.py lines 731-733
blueprint = seed.blueprint_id or "?"
if len(blueprint) > 6:
    blueprint = blueprint[:6]
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed germination | lifecycle event | Blueprint assigned at GERMINATE action |
| Blueprint registry | configuration | Valid blueprint IDs from `leyline.BLUEPRINT_IDS` |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Slot column display | display | Shows blueprint type for quick identification |
| SlotsPanel detail | display | Full blueprint name in expanded view |
| Blueprint penalty system | analytics | Tracks blueprint success/failure rates |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvOverview slot columns |

---

## 8. Notes

> **Truncation Rationale:** 6-character limit balances readability with column width. Most blueprints have recognizable prefixes:
> - `conv_l` → conv_light
> - `attent` → attention
> - `depthw` → depthwise
> - `lora_l` → lora_large
>
> **Blueprint Source:** Valid blueprint IDs are defined in `leyline/factored_actions.py` via the `BlueprintAction` enum. The `BLUEPRINT_IDS` tuple provides the canonical string names.
>
> **Immutability:** Blueprint ID is set once at germination and never changes for the seed's lifetime. This is enforced by the kasmina lifecycle - there is no "morph" operation.
