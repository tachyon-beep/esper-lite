# Urabrask — Delta Matrix (Planning Baseline)

Prototype baseline: no Urabrask implementation exists; this matrix records the intended capabilities and what success looks like.

| Area | Expected Behaviour | Prototype Status | What We’ll Look For |
| --- | --- | --- | --- |
| Evaluation Harness (Crucible) | Deterministic hazard tests (grad/mem/numerical/fail) with reproducible inputs | Not present | Stable runs with fixed seeds; bounded duration; artefact logs; repeatability across machines |
| BSDS Generator | Authoritative BSDS artefact with hazards, handling, provenance, signature | Leyline enums/messages added (Day 1) | BSDS stored in Urza; queryable by bands/hazards; signatures verified |
| Performance Benchmarks | Latency/throughput on reference batch profiles; resource curves | Not present | Benchmark artefacts stored; regressions detected with tolerances |
| Integration: Urza | Store BSDS + benchmarks; expose filters; attach to descriptors | Not present | Urza entries with BSDS links; `bsds_provenance=URABRASK` |
| Integration: Tamiyo/Karn | Consume BSDS for gating/selection; annotate decisions | Not present | Decisions include risk/mitigation annotations; policy follows recommendations |
| Oona Topics | Publish BSDS issuance/failure; benchmark reports | Not present | Topics active; counts visible in Nissa |
| Nissa Dashboards | Hazard distributions; benchmark trends; CRITICAL alerts | Not present | Dashboards live; alerts wired to routes |
| Signing/Immutability | Signed BSDS; immutable records; WAL recovery | Not present | Signature checks pass; recovery tested |
| SLOs & Telemetry | Crucible duration p95 targets; recovery ≤12s; exports | Not present | Telemetry present in Nissa; SLO snapshots available |
