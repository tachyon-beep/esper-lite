# Tamiyo — Prototype Delta (Strategic Controller)

Executive summary: the prototype implements a light Tamiyo service with a stub feed‑forward policy, risk gating based on simple loss deltas and blueprint metadata, telemetry emission, a WAL‑backed field‑report store with retention, Oona publish/consume for telemetry and policy updates, and a basic inference latency guard in tests. The restored design specifies a 4‑layer hetero GNN policy, PPO/IMPALA learning stack with graph experience replay, a comprehensive multi‑signal risk engine with circuit breakers and conservative mode management, strict deadlines/timeouts, and a full field‑report lifecycle with ack/retry. Leyline is the single source of truth for all contracts.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — ordered plan to close gaps without tech debt

Design sources:
- `docs/design/detailed_design/03-tamiyo-unified-design.md`
- `docs/design/detailed_design/03.2-tamiyo-policy-training.md`
- `docs/design/detailed_design/03.3-tamiyo-risk-modeling.md`
- `docs/design/detailed_design/03.4-tamiyo-integration-contracts.md`

Implementation evidence (primary):
- `src/esper/tamiyo/service.py`, `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/persistence.py`
- Tests: `tests/tamiyo/test_service.py`, `tests/integration/test_blueprint_pipeline_integration.py`
