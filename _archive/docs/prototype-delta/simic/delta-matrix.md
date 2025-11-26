# Simic — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| FieldReport ingestion | `04-simic.md` | Consume `FieldReport` from Oona with ack/retry and TTL persistence | `FieldReportReplayBuffer.ingest_from_oona()` | Partially Implemented | Should‑have | Ingestion exists; no ack/retry semantics or duplicate handling. |
| Replay buffer | `04.2` | Graph‑aware experiences, TTL, memory budget, prioritised sampling | `FieldReportReplayBuffer` (FIFO, TTL, random sample) | Partially Implemented | Must‑have | No memory budgeting or prioritisation; experiences are tensors, not PyG graphs. |
| Reward shaping | `04-simic.md` | Outcome/metric‑driven rewards | `_compute_reward()` (loss_delta/outcome) | Implemented | Should‑have | Basic shaping present. |
| Trainer algorithm | `04.1` | IMPALA (V‑trace) default; PPO optional; LR controller | `SimicTrainer` PPO‑style only | Partially Implemented | Must‑have | No IMPALA/V‑trace; no LR controller integration. |
| Policy network | `04.1` | Graph‑aware; supports EWC/LoRA | `_PolicyNetwork` with LoRA; no graph ops | Partially Implemented | Should‑have | LoRA present; no GraphSAGE/GAT; no EWC. |
| Validation & gating | `04-simic.md` | Validation suite gates publishing; chaos/property tests | `PolicyValidator`, gating in `create_policy_update()` | Partially Implemented | Should‑have | Threshold checks present; no chaos/property/security tests. |
| Versioning & publishing | `04-simic.md` | Versioned checkpoints + metadata; publish via Oona | `create_policy_update()`, `publish_policy_updates()` | Partially Implemented | Should‑have | Metadata minimal; no version history management. |
| Circuit breakers & conservative mode | `04-simic.md` | Breakers around training/storage; conservative mode reduces batch, LR | — | Missing | Must‑have | Not implemented. |
| Telemetry | `04-simic.md` | `simic.training.*`, breaker states | `build_metrics_packet()` emits training metrics | Partially Implemented | Should‑have | No breaker metrics; limited set present. |
| Leyline as canonical | `00-leyline` | Use Leyline messages for reports/updates | `leyline_pb2.FieldReport`, `PolicyUpdate` used directly | Implemented | Must‑have | Canonical use. |

