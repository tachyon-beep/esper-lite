# Tamiyo Next Implementation Guide

## How to Use This Directory

This directory contains the **Obs V3 + Policy V2 implementation plan**, split into manageable phases.

### Authoritative Sources

**Execute from the individual phase documents.** They are authoritative over the parent 6000-line file:

| File | Phase | Status |
|------|-------|--------|
| `00-preamble.md` | Prerequisites & setup | Reference |
| `01-phase1-leyline-constants.md` | Leyline constants | ✅ COMPLETE |
| `02-phase2-obs-v3-features.md` | Obs V3 feature extraction | Execute |
| `03-phase3-blueprint-embedding.md` | Blueprint embedding module | Execute |
| `04-phase4-policy-network-v2.md` | Policy network V2 | Execute |
| `05-phase5-training-config.md` | Training configuration | Execute |
| `06-phase6-vectorized-training.md` | Vectorized training integration | Execute |
| `07-phase7-validation-testing.md` | Validation & testing | Execute |
| `08-phase8-cleanup-appendix.md` | Cleanup & reference | Execute |

### When to Consult the Parent Document

If there is **ambiguity** in a phase document, consult `tamiyo_next.md` for:

- Cross-phase dependencies and ordering
- Rationale behind design decisions
- Edge cases that span multiple phases
- The "why" when the phase doc only gives the "what"

### Supporting Design Documents

- `2025-12-30-obs-v3-design.md` - Observation space V3 design rationale
- `2025-12-30-policy-v2-design.md` - Policy network V2 architecture design

### Implementation Order

Follow the phase numbers in sequence (1 → 2 → 3 → ...). Phase 1 is already complete.
