# ADR-004: Kasmina Lifecycle Canonicalisation (Esper‑Lite)

- Status: Accepted
- Date: 2025-09-21
- Driver: Esper‑Lite Working Group
- Deciders: Kasmina Owner, Leyline Owner, Tech Lead

## Context

Multiple documents and historical drafts describe differing Kasmina lifecycles and gate names (e.g., alternative 11‑state lists and gate labels). The prototype implementation, tests, and generated Leyline bindings already use a specific, consistent set of lifecycle states and gate enums.

To remove ambiguity during the prototype phase, we must pin a single authoritative set of lifecycle states and gate names that all code, tests, and documentation reference.

## Decision

The lifecycle and gate definitions in Leyline are the single source of truth for Kasmina in Esper‑Lite. The canonical names are:

- Lifecycle states (Leyline `SeedLifecycleStage`):
  - `SEED_STAGE_DORMANT`
  - `SEED_STAGE_GERMINATED`
  - `SEED_STAGE_TRAINING`
  - `SEED_STAGE_BLENDING`
  - `SEED_STAGE_SHADOWING`
  - `SEED_STAGE_PROBATIONARY`
  - `SEED_STAGE_FOSSILIZED`
  - `SEED_STAGE_CULLED`
  - `SEED_STAGE_EMBARGOED`
  - `SEED_STAGE_RESETTING`
  - `SEED_STAGE_TERMINATED`
  - (Bootstrap: `SEED_STAGE_UNKNOWN` is internal only.)

- Lifecycle gates (Leyline `SeedLifecycleGate`):
  - `SEED_GATE_G0_SANITY`
  - `SEED_GATE_G1_GRADIENT_HEALTH`
  - `SEED_GATE_G2_STABILITY`
  - `SEED_GATE_G3_INTERFACE`
  - `SEED_GATE_G4_SYSTEM_IMPACT`
  - `SEED_GATE_G5_RESET`

All implementation and documentation must use these exact names. Any alternative naming appearing in older documents is superseded.

## Consequences

- Documentation must be updated to reference these enums. A canonical section is inserted at the head of the Kasmina design as a binding reference.
- Code must not introduce local overlays/mappings; it must import and use Leyline enums directly.
- Tests should assert transitions and gate usage against these enum values.
- Papers and narratives may use simplified lifecycle explanations, but must include a note that the operational lifecycle and gate names are defined in Leyline and link back to the canonical design section.

## References

- Canonical design section: `docs/design/detailed_design/02-kasmina-unified-design.md`
- Schema/bindings: `src/esper/leyline/_generated/leyline_pb2.py`
- Implementation: `src/esper/kasmina/lifecycle.py`, `src/esper/kasmina/gates.py`
- Tests: `tests/kasmina/test_lifecycle.py`
