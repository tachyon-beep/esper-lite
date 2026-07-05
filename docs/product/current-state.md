# Current State — Esper        Checkpoint: 2026-07-05 ~23:30 (checkpoint #19 — EV Stage-0 instrumentation DELIVERED + gate PASSES; PDR-0032)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet). **Stage-0 is DELIVERED and the
advance gate PASSED (PDR-0032):** the value-free `Cov(R_cf,R)/Var(R)` reads **median ~1.02,
12/13 cf-active updates > 0.40** on a Stage-2-OFF control run — the counterfactual stream
dominates value-target variance, so **Stage-2 de-shaping is justified and the epic proceeds
to Stage-2** (it is NOT re-scoped to Stage-1-only). Delivery lives on
`feat/ev-stab-stage2-hra`.

## What just happened (this session)
Built the Stage-0 instrument to a rigorous bar and read the gate. **5 TDD-green commits on
`feat/ev-stab-stage2-hra`** (`562cb5bf`→`70753923`, ~2400 tests green, no regressions):
decompose_additends + completeness keystone (falsified/has-teeth); per-component buffer SoA
+ `return_variance_telemetry` flag (byte-identity proven DIRECTLY — not inferred);
value-free gate computed in `ppo_agent.update()` + the ON-leg λ-metric DEMOTED to a
V_cf-contaminated diagnostic (PDR-0028); 6-file telemetry contract; full-loop reducer smoke.
No checkpoint/schema bump (telemetry-only). Then read the gate on a canonical baseline
Stage-2-OFF control run (2×4060 Ti, gpu_preload) → decisive PASS.

## In flight
- **Control run still accumulating** the full gate distribution (backgrounded; reading is
  PRELIMINARY at n≈14 updates but decisive — 12/13 ≫ 0.40, residual ~2e-17).
- **Stage-0 task esper-lite-3d67b09687** (in_progress): advance-gate MET; comment added with
  the reading. **Closes on control-run completion** with the final median/IQR.
- **Stage-2 now unblocked** by the gate pass: esper-lite-2a4b56e719 (Stage-2 acceptance,
  incl. MAJOR-1 `ev_sum` hard floor + MAJOR-3 provenance) and esper-lite-e6382020d2.

## Facts the next session must not relitigate
- **The gate PASSED** (~1.02 ≫ 0.40) — do NOT re-read the premise as unproven; Stage-2 is
  justified. The reading is value-free (PDR-0028), not the contaminated ON-leg λ-metric.
- The instrument is DONE (5 commits). Do not rebuild it. `return_variance_telemetry=True` +
  `reward_mode=SHAPED` + `hra_value_decomposition=False` = the control posture.
- **UNIFICATION delta:** ev-stab decomposes `synergy_bonus`; the Shapley/main line renamed it
  `interaction_bonus` (`5910c5e6`). The two `ADDITEND_SIGN_MAP` copies diverge on that key.
- Honor MAJOR-3 sequencing: the value-free gate on the OFF control run comes BEFORE any HRA
  ON run (done — this reading is the OFF control).

## Open questions / blocked-on-owner
- **Full-distribution median/IQR** — pending the control run finishing (reversal trigger:
  final median < 0.40 reopens; remote given 12/13 ≫ 0.40).
- **ev-stab→main unification** — still OWNER-GATED / deferred (PDR-0031); with Stage-0 now
  complete it is closer to ready (gated on OFF-leg byte-identity + drl/pytorch review).
- **Housekeeping (surface, not mine):** the ev-stab working tree has hook-modified
  `AGENTS.md`, `CLAUDE.md`, `.claude|.agents/skills/loomweave-workflow/*` (loomweave
  session-start refresh) — uncommitted, untouched by this checkpoint. Owner decides.
- Standing placeholders: north-star / rent TARGETs (metrics.md) still owner-set.

## Last checkpoint did (checkpoint #19)
- PDR-0032 (Stage-0 instrument delivered; gate PASSES ~1.02 ≫ 0.40 → Stage-2 justified,
  epic proceeds; launch-locally = canonical-host call).
- metrics.md EV Stage-0 gate row: NO VALID READING → PASSED (median 1.02, preliminary).
- roadmap.md Now bullet: first leg (Stage-0) → delivered + gate passed; next leg = Stage-2.
- Tracker: comment on esper-lite-3d67b09687 with the reading (leave in_progress; closes on
  run completion). No horizon change (EV-stab stays Now).

## Next session, start here
**Read the final gate distribution** from the completed control run (telemetry in scratchpad;
or re-run if the process was reaped) — confirm median ≥ 0.40, close esper-lite-3d67b09687.
Then **proceed to Stage-2 acceptance** (esper-lite-2a4b56e719): Stage-2 HRA is already built
on `feat/ev-stab-stage2-hra`; the work is its acceptance criteria (MAJOR-1 `ev_sum` hard
floor + MAJOR-3 provenance) and the paired fresh-init A/B on EV_main liftoff. Memory pointers:
`ev-stab-stage2-impl-state` (updated this session), `ev-variance-research-verdict`.
