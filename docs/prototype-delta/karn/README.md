# Karn — Prototype Delta (Phase‑1 Templates)

Executive summary: the prototype implements Phase‑1 features: a Leyline‑backed in‑memory catalogue (`KarnCatalog`) with 50+ static templates from `templates.py`, parameter‑bounds validation, tier filtering, and template selection with conservative fallback options. The full Phase‑1 design also calls for request handling via Leyline `BlueprintQuery`, circuit‑breaker‑driven conservative mode, TTL cleanup for cached metadata, telemetry on selections, and approval/quarantine flags. Neural/generative Phase‑2 features are explicitly out of scope.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps (Phase‑1 only)
- `pytorch-2.8-upgrades.md` — 2.8 review (no changes needed in Karn; compile stays in Tezzeret)

Design sources:
- `docs/design/detailed_design/05-karn-unified-design.md`
- `docs/design/detailed_design/05.1-karn-template-system.md`

Implementation evidence (primary):
- `src/esper/karn/catalog.py`, `src/esper/karn/templates.py`
- Tests: `tests/karn/test_catalog.py`, `tests/integration/test_blueprint_pipeline_integration.py`
