# Tamiyo — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Strategic inference (policy) | `03-tamiyo.md` (Neural policy), `03.2` | 4‑layer hetero GNN; <45 ms inference; Leyline I/O | `src/esper/tamiyo/policy.py` (FFN stub), `service.evaluate_epoch()` timing | Partially Implemented | Must‑have | Budget measured and asserted; architecture is a stub, not GNN. |
| Risk governance | `03.3-tamiyo-risk-modeling.md` | Multi‑signal risk engine (grad, memory, latency, stability); circuit breakers; conservative mode management | `src/esper/tamiyo/service.py::_apply_risk_engine`, `_set_conservative_mode`; tests in `tests/tamiyo/test_service.py::test_loss_spike_triggers_pause` | Partially Implemented | Must‑have | Loss/blueprint/time budget signals mapped to actions; reason codes emitted. Grad/memory signals via Tolaria metrics still minimal. |
| Learning loop | `03.2-tamiyo-policy-training.md` | PPO with GAE; optional IMPALA/V‑trace; LR via UnifiedLRController | — | Missing | Should‑have | No training algorithms in prototype; policy is inference‑only stub. |
| Experience replay | `03.2` | Graph trajectory buffer (100 K), GC, prioritised sampling | — | Missing | Nice‑to‑have | Not present. |
| Field report lifecycle | `03-tamiyo.md` | Capture→Synthesis→Publish→Ack/Retry; WAL retention | `persistence.py` WAL with fsync + retention; `service.generate_field_report()` and publish | Partially Implemented | Should‑have | WAL/retention implemented; no ack/retry semantics; no observation windows. |
| Telemetry aggregation hub | `03-tamiyo.md` | Ingest Tamiyo/Tolaria/Kasmina telemetry; normalise; forward | `service.evaluate_epoch()` emits telemetry | Missing | Should‑have | Emits own telemetry only; no ingestion/aggregation. |
| Policy update interface | `03-tamiyo.md` | Validate and hot‑reload policy checkpoints; rollback on failure | `service.ingest_policy_update()` applies state dict | Partially Implemented | Should‑have | No validation/rollback gate; best‑effort load only. |
| Pause quotas/security | `03-tamiyo.md` | Server‑side pause quota with audit trail | — | Missing | Must‑have | Not implemented. |
| Circuit breakers & conservative mode | `03-tamiyo.md` | Breakers wrap inference/training/IO; conservative mode managed by breakers | `src/esper/tamiyo/service.py::TamiyoCircuitBreaker`, `_set_conservative_mode`; `tests/tamiyo/test_service.py::test_inference_breaker_enters_conservative_mode` | Implemented | Must‑have | Inference & metadata breakers auto-enter/exit conservative mode with telemetry; training/IO breakers remain future work. |
| Deadlines & timeouts | `03.4` | Strict timeout matrix (e.g., 45 ms inference, 2 s handshakes) | `src/esper/tamiyo/service.py::_run_policy`, `_resolve_blueprint_with_timeout`; `tests/tamiyo/test_service.py::test_evaluate_step_timeout_inference` | Partially Implemented | Should‑have | Inference/metadata deadlines enforced with fail-open paths; Kasmina apply timeout still pending in Tolaria (T‑A4). |
| Security envelope | `03.4` | HMAC/nonce/freshness on emitted commands; auth on updates | — | Missing | Must‑have | Use shared signing util; verify policy updates if signed. |
| Leyline as canonical | `03.4` | Use Leyline messages/enums directly | `leyline_pb2.*` used everywhere | Implemented | Must‑have | No local mappings present. |
| Performance budgets | `03-tamiyo.md` | VRAM/latency budgets, telemetry of breakdowns | Latency metric present | Partially Implemented | Nice‑to‑have | No VRAM budget checks or breakdown. |
