# Telemetry Record Creation Task

## Your Mission

Create a telemetry record for the assigned metric by tracing its data flow from source to display.

## Instructions

1. **Read the template:** `docs/tele_audit_2/templates/TELEMETRY_RECORD_TEMPLATE.md`
2. **Read the example:** `docs/tele_audit_2/templates/EXAMPLE_TELE-001_entropy.md`
3. **Trace the metric end-to-end:**
   - Find where the value is **computed/emitted** (usually in `src/esper/simic/` or `src/esper/tolaria/`)
   - Find how it **transports** to the schema (via `src/esper/karn/sanctum/aggregator.py`)
   - Find the **schema field** in `src/esper/karn/sanctum/schema.py`
   - Find which **widgets consume** it (check `src/esper/karn/sanctum/widgets/`)
4. **Create the record file:** `docs/tele_audit_2/records/TELE-{ID}_{name}.md`
5. **Fill in all sections** with what you find
6. **Mark status honestly:**
   - If wiring is incomplete or broken, mark the relevant checkboxes as unchecked
   - Document what's missing in the Notes section

## What You Can Edit

- **ONLY** files in `docs/tele_audit_2/records/`
- Do NOT modify source code, schema, or widgets

## What To Do If Wiring Is Broken

This is expected! Many metrics may be:
- Computed but not emitted
- Emitted but not aggregated
- In schema but not consumed by any widget
- Partially wired

**Document what you find.** The goal is to expose gaps, not fix them.

## Output Format

Create ONE file: `docs/tele_audit_2/records/TELE-{ID}_{snake_case_name}.md`

Use the template structure exactly. Fill in what you can find, mark unknowns with `[UNKNOWN]` or `[NOT FOUND]`.

## Category ID Ranges

| Category | ID Range |
|----------|----------|
| training | 001-099 |
| policy | 100-199 |
| value | 200-299 |
| gradient | 300-399 |
| reward | 400-499 |
| seed | 500-599 |
| environment | 600-699 |
| infrastructure | 700-799 |
| decision | 800-899 |

## Search Tips

- Schema fields: `grep -r "field_name" src/esper/karn/sanctum/schema.py`
- Emitters: `grep -r "field_name" src/esper/simic/telemetry/`
- Aggregator: `grep -r "field_name" src/esper/karn/sanctum/aggregator.py`
- Widget consumers: `grep -r "field_name" src/esper/karn/sanctum/widgets/`
