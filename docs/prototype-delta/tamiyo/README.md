# Tamiyo — Prototype Delta (Strategic Controller)

Executive summary: the prototype implements a light Tamiyo service with a stub feed‑forward policy, risk gating based on simple loss deltas and blueprint metadata, telemetry emission, a WAL‑backed field‑report store with retention, Oona publish/consume for telemetry and policy updates, and a basic inference latency guard in tests. The restored design specifies a 4‑layer hetero GNN policy, PPO/IMPALA learning stack with graph experience replay, a comprehensive multi‑signal risk engine with circuit breakers and conservative mode management, strict deadlines/timeouts, and a full field‑report lifecycle with ack/retry. Leyline is the single source of truth for all contracts.

Outstanding Items (for coders)

- Strategic policy upgrade (GNN)
  - Replace FFN stub with 4‑layer hetero GNN (GraphSAGE→GAT) and maintain <45 ms inference.
  - Add optional `torch.compile(..., mode='reduce-overhead')` path with eager fallback and unit perf tests.
  - Pointers: `src/esper/tamiyo/policy.py` (architecture), `docs/prototype-delta/tamiyo/pytorch-2.8-upgrades.md`.

- Risk engine (multi‑signal) and gating
  - Incorporate stability/latency/memory/lifecycle signals; thresholds + categories; emit detailed telemetry.
  - Pointers: `src/esper/tamiyo/service.py::evaluate_epoch`, add a `RiskEngine` helper.

- Circuit breakers + conservative mode
  - Wrap inference, Urza lookups, and Oona IO with breakers; auto‑enter/exit conservative mode on repeated faults.
  - Pointers: `src/esper/tamiyo/service.py` (risk_config), align breaker types with Oona/Kasmina.

- Deadlines & timeouts
  - Enforce strict timeouts (e.g., inference 45 ms, Oona 2 s, Urza metadata 200 ms); degrade on breach; add telemetry.
  - Pointers: `src/esper/tamiyo/service.py` (timers + guards).

- Policy update validation + rollback
  - Validate checkpoint payloads (shape/hash/version); hot‑reload safely; rollback on failure; emit telemetry.
  - Pointers: `src/esper/tamiyo/service.py::ingest_policy_update`.

- Field report lifecycle (ack/retry + observation windows)
  - Add ack/retry semantics for publishes, observation windows (≥3 epochs) before synthesis, WAL index, and bounded retries.
  - Pointers: `src/esper/tamiyo/persistence.py` (WAL), `src/esper/tamiyo/service.py::generate_field_report/publish_history`.

- Security envelope
  - Sign emitted AdaptationCommands (HMAC/nonce/freshness) and verify signed PolicyUpdate payloads.
  - Pointers: `esper.security.signing`, mirror Kasmina verifier pattern.

- Telemetry enrichment
  - Emit `tamiyo.validation_loss`, `tamiyo.loss_delta`, risk scores, blueprint risk metrics, and breaker states; escalate severity appropriately.
  - Pointers: `src/esper/tamiyo/service.py` (telemetry builder).

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — ordered plan to close gaps without tech debt
- `pytorch-2.8-upgrades.md` — mandatory hetero‑GNN inference upgrades (compile, inference_mode, TF32, data transfer)

Design sources:
- `docs/design/detailed_design/03-tamiyo-unified-design.md`
- `docs/design/detailed_design/03.2-tamiyo-policy-training.md`
- `docs/design/detailed_design/03.3-tamiyo-risk-modeling.md`
- `docs/design/detailed_design/03.4-tamiyo-integration-contracts.md`

Implementation evidence (primary):
- `src/esper/tamiyo/service.py`, `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/persistence.py`
- Tests: `tests/tamiyo/test_service.py`, `tests/integration/test_blueprint_pipeline_integration.py`
