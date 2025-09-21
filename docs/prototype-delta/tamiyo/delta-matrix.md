# Tamiyo — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Strategic inference (policy) | `03-tamiyo.md` (Neural policy), `03.2` | 4‑layer hetero GNN; <45 ms inference; Leyline I/O | `src/esper/tamiyo/policy.py` (FFN stub), `service.evaluate_epoch()` timing | Partially Implemented | Must‑have | Budget measured and asserted; architecture is a stub, not GNN. |
| Risk governance | `03.3-tamiyo-risk-modeling.md` | Multi‑signal risk engine (grad, memory, latency, stability); circuit breakers; conservative mode management | `service.evaluate_epoch()` loss‑delta and blueprint risk checks; `RiskConfig.conservative_mode` | Partially Implemented | Must‑have | Only basic loss/blueprint checks; no breakers or multi‑channel thresholds. |
| Learning loop | `03.2-tamiyo-policy-training.md` | PPO with GAE; optional IMPALA/V‑trace; LR via UnifiedLRController | — | Missing | Should‑have | No training algorithms in prototype; policy is inference‑only stub. |
| Experience replay | `03.2` | Graph trajectory buffer (100 K), GC, prioritised sampling | — | Missing | Nice‑to‑have | Not present. |
| Field report lifecycle | `03-tamiyo.md` | Capture→Synthesis→Publish→Ack/Retry; WAL retention | `persistence.py` WAL with fsync + retention; `service.generate_field_report()` and publish | Partially Implemented | Should‑have | WAL/retention implemented; no ack/retry semantics; no observation windows. |
| Telemetry aggregation hub | `03-tamiyo.md` | Ingest Tamiyo/Tolaria/Kasmina telemetry; normalise; forward | `service.evaluate_epoch()` emits telemetry | Missing | Should‑have | Emits own telemetry only; no ingestion/aggregation. |
| Policy update interface | `03-tamiyo.md` | Validate and hot‑reload policy checkpoints; rollback on failure | `service.ingest_policy_update()` applies state dict | Partially Implemented | Should‑have | No validation/rollback gate; best‑effort load only. |
| Pause quotas/security | `03-tamiyo.md` | Server‑side pause quota with audit trail | — | Missing | Must‑have | Not implemented. |
| Circuit breakers & conservative mode | `03-tamiyo.md` | Breakers wrap inference/training/IO; conservative mode managed by breakers | `RiskConfig.conservative_mode` flag only | Missing | Must‑have | No breaker framework or automatic transitions. |
| Deadlines & timeouts | `03.4` | Strict timeout matrix (e.g., 45 ms inference, 2 s handshakes) | Inference measured; no enforced deadline | Partially Implemented | Should‑have | Enforce and degrade on breach. |
| Security envelope | `03.4` | HMAC/nonce/freshness on emitted commands; auth on updates | — | Missing | Must‑have | Use shared signing util; verify policy updates if signed. |
| Leyline as canonical | `03.4` | Use Leyline messages/enums directly | `leyline_pb2.*` used everywhere | Implemented | Must‑have | No local mappings present. |
| Performance budgets | `03-tamiyo.md` | VRAM/latency budgets, telemetry of breakdowns | Latency metric present | Partially Implemented | Nice‑to‑have | No VRAM budget checks or breakdown. |

