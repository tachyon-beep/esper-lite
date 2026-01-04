# Telemetry Record: [TELE-555] Seed Params

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-555` |
| **Name** | Seed Params |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many parameters does this seed module have?"

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
| **Units** | parameter count (dimensionless) |
| **Range** | `[0, ∞)` |
| **Precision** | Formatted with K/M suffix for display |
| **Default** | `0` (no parameters) |

### Semantic Meaning

> Parameter count represents the total number of trainable parameters in the seed module. This is a static property determined when the seed is instantiated from a blueprint:
>
> - **seed_params = 0:** Seed has no parameters (invalid or dormant placeholder)
> - **seed_params > 0:** Actual parameter count (e.g., 12,500 → "12.5K")
>
> Display format uses human-readable suffixes:
> - < 1,000: raw number (e.g., "256")
> - 1,000 - 999,999: K suffix (e.g., "12.5K")
> - ≥ 1,000,000: M suffix (e.g., "1.2M")

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| Any stage | seed_params > 0 | Normal - seed has parameters |
| Any stage | seed_params = 0 | Placeholder or uninitialized |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Module parameter count, computed at instantiation |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `seed_params: int` (schema line 397) |

```python
@dataclass
class SeedState:
    """State of a single seed slot."""
    seed_params: int = 0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | Module.numel() at seed instantiation | `kasmina/blueprints.py` |
| **2. Storage** | SeedState.seed_params updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.seed_params field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.seed_params in Sanctum | `karn/sanctum/schema.py` (line 397) |
| **5. Display** | EnvDetailScreen SeedCard | `karn/sanctum/widgets/env_detail_screen.py` (lines 111-114) |

```
[Module instantiation]
  --counts-->
  [SeedState.seed_params]
  --SEED_GERMINATED event-->
  [SeedTelemetry.seed_params]
  --SanctumAggregator-->
  [schema.SeedState.seed_params]
  --SeedCard-->
  ["Params: 12.5K"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `seed_params` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].seed_params` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 397 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen SeedCard | `widgets/env_detail_screen.py` (lines 111-114) | "Params: 12.5K" with format_params() |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Module parameter count computed at instantiation
- [x] **Transport works** — SeedState.seed_params populated; telemetry carries value
- [x] **Schema field exists** — `SeedState.seed_params: int = 0` at line 397
- [x] **Default is correct** — `0` is correct for placeholder/dormant slots
- [x] **Consumer reads it** — SeedCard._render_active() displays `seed.seed_params`
- [x] **Display is correct** — Format: "Params: 12.5K" (formatted with suffix)

### Display Code

```python
# From env_detail_screen.py lines 111-114
if seed.seed_params and seed.seed_params > 0:
    lines.append(Text(f"Params: {format_params(seed.seed_params)}", style="dim"))
else:
    lines.append(Text("Params: --", style="dim"))
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Blueprint instantiation | computation | Module created from blueprint spec |
| Module architecture | parameter | Layer sizes determine param count |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SeedCard display | display | Shows parameter count |
| Growth ratio calculation | analytics | Tracks total parameter overhead |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvDetailScreen SeedCard |

---

## 8. Notes

> **Static Property:** Unlike alpha or accuracy_delta, seed_params is computed once at module instantiation and never changes. It represents the architectural capacity of the seed, not its current state.
>
> **Display Context:** Always shown in SeedCard for any active (non-DORMANT) seed. Uses format_params() helper to show human-readable values (e.g., 12500 → "12.5K").
