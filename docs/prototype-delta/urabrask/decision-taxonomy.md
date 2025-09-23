# Urabrask → Tamiyo Decision Taxonomy (Prototype)

This document defines how BSDS‑Lite fields map to Tamiyo action adjustments and telemetry in the prototype.

## Inputs
- Canonical: Leyline `BSDS` message (Day 1 contracts), carrying `risk_score`, `hazard_band`, `handling_class`, etc.
- Prototype transport: Urza extras JSON `extras["bsds"]` mirroring Leyline fields.

## Action Mapping
- CRITICAL hazard
  - Action: `COMMAND_PAUSE`
  - Telemetry: `bsds_hazard_critical` (CRITICAL), set conservative mode
  - Annotations: `bsds_hazard_band=CRITICAL`, `bsds_risk=…`
- HIGH hazard
  - Baseline SEED → downgrade to `COMMAND_OPTIMIZER` (conservative optimizer adjustment)
  - Telemetry: `bsds_hazard_high` (WARNING)
- MEDIUM/LOW hazard
  - No direct override; rely on policy and descriptor risk
  - Telemetry: `bsds_present` (INFO)

## Risk Precedence
- When present, `risk_score` supersedes descriptor `risk` for Tamiyo’s blueprint risk evaluation.
- `handling_class=quarantine` is treated as CRITICAL hazard for the prototype.

## Rationale
- Keeps Tamiyo within tight step budgets while honoring high‑risk signals.
- Uses existing telemetry routes and annotations for observability; aligns with `URABRASK_COMPLETION_PACKAGES.md`.
