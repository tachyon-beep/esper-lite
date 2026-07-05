# Current State — Esper        Checkpoint: 2026-07-05 ~22:25 (checkpoint #20 — EV Stage-0 gate PASSED + ACCEPTED on the partial read; task CLOSED; leak ruled out; PDR-0033)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet). **Stage-0 is DONE: the advance gate
PASSED and the owner ACCEPTED the reading (PDR-0033)** — value-free `Cov(R_cf,R)/Var(R)`
median **1.017**, IQR [1.006, 1.033], **41/42 cf-active updates > 0.40**, **flat at ~1.0
across the whole observed window** (random-init → ~50% acc). The counterfactual stream
dominates value-target variance ⇒ **Stage-2 de-shaping is justified; the epic proceeds to
Stage-2** (NOT re-scoped to Stage-1-only). Delivery lives on `feat/ev-stab-stage2-hra`.

## What just happened (this session)
Built the Stage-0 instrument to a rigorous bar (**5 TDD-green commits on
`feat/ev-stab-stage2-hra`**, `562cb5bf`→`70753923`, ~2400 tests green), read the gate on a
canonical Stage-2-OFF control run, diagnosed the run's mid-flight death, and closed Stage-0:
- The control run **terminated externally at episode 172/400 (~43%)** — the "process reaped"
  contingency the prior checkpoint flagged.
- **Advisor-flagged risk investigated:** did the new `return_variance_telemetry` code (first
  real run with it on) leak/OOM the run? **RULED OUT** on four independent lines (pre-allocated
  SoA — code-proven; flat 72–83 s batch cadence — no swap thrash; 41 GiB free / no OOM; no
  CUDA-OOM traceback). The instrumentation is clean; a rerun would not repeat the death.
- **Gate confirmed on the 42-update partial read** (median 1.017, 41/42 > 0.40, flat across
  random→50% acc, residual ~1e-17). **Owner ACCEPTED the partial read** as the gate result
  (rerun-for-record declined — the read is 2.5× the threshold, flat, IQR ~0.03).
- **esper-lite-3d67b09687 CLOSED** on the accepted gate; Stage-2 tasks unblocked.

## In flight
- **Nothing active right now.** Stage-2 is unblocked and ready to pick up (below).

## Facts the next session must not relitigate
- **The gate PASSED and is ACCEPTED** (median 1.017, 41/42 ≫ 0.40, flat across the observed
  window; PDR-0033). Stage-2 is justified; do NOT re-read the premise as unproven or re-run
  the OFF control. Value-free (PDR-0028), not the contaminated ON-leg λ-metric.
- **The reaped run was NOT a code fault** (PDR-0033). Do not "fix" the Stage-0 telemetry for a
  memory leak — there is none. `component_additends` is a pre-allocated SoA.
- The instrument is DONE (5 commits). Do not rebuild it. `return_variance_telemetry=True` +
  `reward_mode=SHAPED` + `hra_value_decomposition=False` = the control posture.
- **UNIFICATION delta:** ev-stab decomposes `synergy_bonus`; the Shapley/main line renamed it
  `interaction_bonus` (`5910c5e6`). The two `ADDITEND_SIGN_MAP` copies diverge on that key.
- Honor MAJOR-3 sequencing: the value-free gate on the OFF control run comes BEFORE any HRA
  ON run (done — this reading is the OFF control).

## Open questions / blocked-on-owner
- **ev-stab→main unification** — still OWNER-GATED / deferred (PDR-0031); with Stage-0 now
  complete it is closer to ready (gated on OFF-leg byte-identity + drl/pytorch review).
- **Housekeeping (surface, not mine):** the ev-stab working tree has hook-modified
  `AGENTS.md`, `CLAUDE.md`, `.claude|.agents/skills/loomweave-workflow/*` (loomweave
  session-start refresh) — uncommitted, untouched. Owner decides.
- Standing placeholders: north-star / rent TARGETs (metrics.md) still owner-set.

## Last checkpoint did (checkpoint #20)
- PDR-0033 (control run reaped at 43%; leak investigated + RULED OUT; gate PASS stands on the
  partial read). **Owner ACCEPTED** the partial read; **esper-lite-3d67b09687 CLOSED**.
- metrics.md EV Stage-0 gate row: preliminary (n≈14) → confirmed on the 42-update partial read,
  with reaped-at-172/400 + leak-ruled-out provenance.
- No horizon change (EV-stab stays Now).

## Next session, start here
**Proceed to Stage-2 acceptance** (esper-lite-2a4b56e719). Stage-2 HRA is already built on
`feat/ev-stab-stage2-hra`; the remaining work is its acceptance criteria — **MAJOR-1 `ev_sum`
hard floor + MAJOR-3 provenance** — and the **paired fresh-init A/B on EV_main liftoff**
(the noisy cf head must not mask the de-shaping win in the aggregate). Memory pointers:
`ev-stab-stage2-impl-state`, `ev-variance-research-verdict`. (esper-lite-e6382020d2 also
unblocked.)
