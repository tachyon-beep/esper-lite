# Speculative Deltas

Purpose: Curate research concepts that could materially improve the Esper‑Lite prototype and present them as design‑style deltas. Each item is explicitly speculative (not committed to the prototype scope) but framed as if it were a planned design addition, with a minimal adoption path.

How to read: Each subfolder contains a short rationale, proposed behaviour, cross‑subsystem impact, a status/matrix, and a low‑risk adoption sketch for the prototype if desired.

Contents

- `scaffolds/` — Temporary, foldable blueprint deployments (policy overlay; no lifecycle change).
- `tamiyo-narset-split/` — Strategic vs tactical controller split with protected zones and inverse throttle.
- `mycosynth-config-fabric/` — Centralised, real‑time configuration fabric distinct from Leyline.
- `future-pruning/` — Hardware‑aware, structured pruning and torch.ao integration.
- `bsds-lite/` — Blueprint Safety Data Sheet (Urabrask) fields adopted in a lightweight form.

Notes

- These are research‑forward; keep prototype risk low by adopting metadata/contracts first, behaviours later.
- Leyline remains canonical; any contract changes are proposed there first.

MCDA — Utility vs Complexity (Prioritisation)

- Scoring (1–5): higher is better. Overall score = Σ(weight × score).
- Criteria and weights:
  - Safety/Robustness (w=0.25) — improves guard‑rails/operability
  - Results/Performance (w=0.20) — likely impact on measurable outcomes
  - Prototype Fit / Adoption Ease (w=0.20) — low‑risk path, incremental adoption
  - Complexity (inverted; 5 = low complexity) (w=0.20)
  - Blast Radius (inverted; 5 = low cross‑subsystem churn) (w=0.15)

MCDA Table

- BSDS‑Lite: Safety 4, Results 2, Fit 5, Complexity 5, Blast 4 → Overall ≈ 4.00
- Scaffolds (metadata/telemetry‑only): Safety 3, Results 3, Fit 4, Complexity 4, Blast 4 → Overall ≈ 3.55
- Future Pruning (descriptors only): Safety 2, Results 3, Fit 4, Complexity 4, Blast 5 → Overall ≈ 3.45
- Tamiyo–Narset Split: Safety 5, Results 3, Fit 1, Complexity 1, Blast 1 → Overall ≈ 2.40
- Mycosynth (shim): Safety 2, Results 1, Fit 3, Complexity 3, Blast 3 → Overall ≈ 2.35

Priority Order (recommended)

1) BSDS‑Lite (low complexity, immediate safety utility, metadata only)
2) Scaffolds (metadata + telemetry overlay; no lifecycle change)
3) Future Pruning descriptors (prepare Tezzeret/Urza/Nissa for 2:4 / torch.ao)
4) Tamiyo–Narset split (defer; schema + orchestration heavy)
5) Mycosynth shim (nice‑to‑have; consider if config churn increases)

Suggested Execution Plan (2–3 short sprints)

- Sprint A: BSDS‑Lite descriptors (Karn/Urza), Tamiyo gating hints, Nissa dashboards.
- Sprint B: Scaffolds metadata + telemetry overlay (Karn/Urza/Kasmina/Tamiyo/Nissa), no behavioural gates.
- Sprint C: Future‑pruning descriptors + Tezzeret no‑op optimisation stage; add Nissa compile/opt dashboards.
