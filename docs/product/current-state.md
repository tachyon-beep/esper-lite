# Current State — Esper        Checkpoint: 2026-07-05 ~22:15 (checkpoint #20 — EV Stage-0 control run reaped at 43%; leak RULED OUT; gate PASS confirmed on the fuller partial read; PDR-0033)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet). **Stage-0 is DELIVERED and the
advance gate PASSED (PDR-0032, confirmed on the fuller read PDR-0033):** the value-free
`Cov(R_cf,R)/Var(R)` reads **median 1.017, IQR [1.006, 1.033], 41/42 cf-active updates
> 0.40**, **flat at ~1.0 across the whole observed window** (random-init → ~50% acc) —
the counterfactual stream dominates value-target variance, so **Stage-2 de-shaping is
justified and the epic proceeds to Stage-2** (NOT re-scoped to Stage-1-only). Delivery
lives on `feat/ev-stab-stage2-hra`.

## What just happened (this session)
Built the Stage-0 instrument to a rigorous bar (**5 TDD-green commits on
`feat/ev-stab-stage2-hra`**, `562cb5bf`→`70753923`, ~2400 tests green), read the gate on
a canonical Stage-2-OFF control run, then **diagnosed the run's mid-flight death**:
- The control run **terminated externally at batch 43 / episode 172 of 400 (~43%)** —
  the "process was reaped" contingency the prior checkpoint flagged.
- **Advisor-flagged risk investigated:** did the new `return_variance_telemetry` code
  (first real run with it on) leak and OOM the run? **RULED OUT** on four independent
  lines (pre-allocated SoA — code-proven; flat 72–83 s batch cadence — no swap thrash;
  41 GiB free / no OOM; no CUDA-OOM traceback). The instrumentation is clean; a rerun
  would not repeat the death. (PDR-0033.)
- **Gate confirmed on the 42-update partial read** — stronger than PDR-0032's n≈14, and
  flat across random→50% acc. residual ~1e-17 = partition reconciles on LIVE data.

## In flight
- **Stage-0 task esper-lite-3d67b09687** (in_progress): advance-gate MET and confirmed
  on the partial read. **Closes on the owner's accept-vs-rerun call** (below) — accept
  the partial PASS, or run a formal full-400 first.
- **Stage-2 now unblocked** by the gate pass: esper-lite-2a4b56e719 (Stage-2 acceptance,
  incl. MAJOR-1 `ev_sum` hard floor + MAJOR-3 provenance) and esper-lite-e6382020d2.

## Facts the next session must not relitigate
- **The gate PASSED** (median 1.017, 41/42 ≫ 0.40, flat across the observed window) — do
  NOT re-read the premise as unproven; Stage-2 is justified. Value-free (PDR-0028), not
  the contaminated ON-leg λ-metric.
- **The reaped run was NOT a code fault** (PDR-0033). Do not "fix" the Stage-0 telemetry
  for a memory leak — there is none. `component_additends` is a pre-allocated SoA.
- The instrument is DONE (5 commits). Do not rebuild it. `return_variance_telemetry=True`
  + `reward_mode=SHAPED` + `hra_value_decomposition=False` = the control posture.
- **UNIFICATION delta:** ev-stab decomposes `synergy_bonus`; the Shapley/main line renamed
  it `interaction_bonus` (`5910c5e6`). The two `ADDITEND_SIGN_MAP` copies diverge on that key.
- Honor MAJOR-3 sequencing: the value-free gate on the OFF control run comes BEFORE any HRA
  ON run (done — this reading is the OFF control).

## Open questions / blocked-on-owner
- **Accept the partial-run PASS, or run a formal full-400 first?** (compute-spend + record
  call — the run is your GPU). Recommendation: ACCEPT — the partial read is decisive (2.5×
  the threshold, flat, IQR ~0.03) and a rerun is low-information. A full-400 median < 0.40
  would reopen (remote). Either way the gate PASS is not in doubt.
- **ev-stab→main unification** — still OWNER-GATED / deferred (PDR-0031); with Stage-0 now
  complete it is closer to ready (gated on OFF-leg byte-identity + drl/pytorch review).
- **Housekeeping (surface, not mine):** the ev-stab working tree has hook-modified
  `AGENTS.md`, `CLAUDE.md`, `.claude|.agents/skills/loomweave-workflow/*` (loomweave
  session-start refresh) — uncommitted, untouched. Owner decides.
- Standing placeholders: north-star / rent TARGETs (metrics.md) still owner-set.

## Last checkpoint did (checkpoint #20)
- PDR-0033 (control run reaped at 43%; leak investigated + RULED OUT; gate PASS stands on
  the partial read; formal full-400 rerun = optional owner call). Refines PDR-0032.
- metrics.md EV Stage-0 gate row: preliminary (n≈14) → confirmed on 42-update partial read,
  with reaped-at-172/400 + leak-ruled-out provenance.
- No horizon change (EV-stab stays Now). No tracker close yet (awaits accept-vs-rerun call).

## Next session, start here
**Resolve the accept-vs-rerun call** (Open questions, above). If ACCEPT: close
esper-lite-3d67b09687 on the partial PASS. If RERUN: relaunch the canonical Stage-2-OFF
control (100 ep × 4 env × 150 epoch, `return_variance_telemetry=True`,
`hra_value_decomposition=False`, gpu_preload) to full 400, confirm median ≥ 0.40, then
close. **Then proceed to Stage-2 acceptance** (esper-lite-2a4b56e719): Stage-2 HRA is
already built on `feat/ev-stab-stage2-hra`; the work is its acceptance criteria (MAJOR-1
`ev_sum` hard floor + MAJOR-3 provenance) and the paired fresh-init A/B on EV_main liftoff.
Memory pointers: `ev-stab-stage2-impl-state`, `ev-variance-research-verdict`.
