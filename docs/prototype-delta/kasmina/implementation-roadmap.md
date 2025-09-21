# Kasmina — Implementation Roadmap (Closing the Delta)

Prioritised steps to reach parity with the design. This is documentation only; no source edits are performed here.

| Order | Theme | Key Tasks | Dependencies | Outcome |
| --- | --- | --- | --- | --- |
| 1 | Safety stack (Must‑have) | Introduce circuit breaker + monotonic timer utilities; wrap kernel fetch/attach; add pause/identity semantics | None | Observable, recoverable failure modes |
| 2 | Gradient isolation (Must‑have) | Backward hooks on host/seed; blending with `.detach()` during grafting; violation counters → breaker; minimal alpha schedule | 1 | Enforce invariant `∇L_host ∩ ∇L_seed = ∅` |
| 3 | Parameter registration (Must‑have) | Per‑seed registry, LR‑group mapping, pre‑update validation; protect teacher params (future KD) | 1 | Guard optimiser updates and audit trail |
| 4 | Telemetry priorities (Should‑have) | Elevate CRITICAL events (violations/breakers) and add emergency path | 1 | Faster operator awareness |
| 5 | Security envelope (Should‑have) | Verify HMAC + nonce + freshness on incoming commands before action | None | Defend against replay and forgery |
| 6 | Memory governance (Should‑have) | TTL caches, epoch GC plumbing; metrics; KD memory checks (stub until KD) | None | Predictable long‑run memory |
| 7 | Performance validation (Nice‑to‑have) | Add micro‑benchmarks for load latency and isolation overhead; report via telemetry | 1–2 | Track regressions |
| 8 | KD (Optional) | Teacher load + checkpointed memory budgeting; KD loss side‑channel | 3,6 | C‑024 capability |

Notes:
- Where the design references optional features (e.g., KD), stub interfaces can land without enabling runtime paths until data/experiments require them.
- Align lifecycle semantics: if 11‑state model (with G0–G5) remains authoritative, extend the Leyline enum alignment or model the additional states as internal substates with explicit gate checks.

