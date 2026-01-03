# Telemetry Records

This folder contains **telemetry records** — structured documentation for each telemetry requirement in the Sanctum TUI system.

## Purpose

While the audit documents in `docs/tele_audit_2/` describe what widgets **consume**, telemetry records describe what telemetry **should exist** and how it flows from source to display.

| Document Type | Perspective | Question Answered |
|--------------|-------------|-------------------|
| Widget Audit | Consumer | "What does this widget need?" |
| Telemetry Record | Requirement | "How does this data flow end-to-end?" |

## File Naming Convention

```
TELE-XXX_short_name.md
```

Examples:
- `TELE-001_entropy.md`
- `TELE-015_grad_norm.md`
- `TELE-042_seed_interaction_sum.md`

## Categories

| Category | ID Range | Description |
|----------|----------|-------------|
| `training` | 001-099 | Core training loop metrics |
| `policy` | 100-199 | Policy network and action distribution |
| `value` | 200-299 | Value function and advantage estimation |
| `gradient` | 300-399 | Gradient health and flow |
| `reward` | 400-499 | Reward signals and shaping |
| `seed` | 500-599 | Seed lifecycle and morphogenesis |
| `environment` | 600-699 | Per-environment state |
| `infrastructure` | 700-799 | System resources (GPU, memory, etc.) |
| `decision` | 800-899 | Tamiyo decision snapshots |

## Status Workflow

```
[ ] Planned     → Requirement documented, not yet implemented
[ ] In Progress → Implementation started
[ ] Wired       → Data flows from source to schema
[ ] Tested      → Automated tests verify wiring
[ ] Verified    → Manual verification in TUI complete
```

## Template

Use `TELEMETRY_RECORD_TEMPLATE.md` as the starting point for new records.

See `EXAMPLE_TELE-001_entropy.md` for a fully filled-in example.

## Creating a New Record

1. Copy the template:
   ```bash
   cp TELEMETRY_RECORD_TEMPLATE.md TELE-XXX_name.md
   ```

2. Fill in sections 1-3 (Identity, Purpose, Data Specification)

3. Implement the telemetry if not already present

4. Fill in section 4 (Data Flow) with actual file paths and code

5. Complete section 5 (Wiring Verification) checklist

6. Document dependencies in section 6

7. Update status checkboxes as you progress

## Cross-References

- **Widget audits:** `docs/tele_audit_2/*.md`
- **Master telemetry reference:** `docs/tele_audit_2/MASTER_TELEMETRY.md`
- **Schema source:** `src/esper/karn/sanctum/schema.py`
- **Threshold constants:** `src/esper/karn/constants.py`
