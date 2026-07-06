# Current State — Esper        Checkpoint: 2026-07-05 ~22:45 (checkpoint #20 — EV Stage-0 gate PASSED + ACCEPTED, task CLOSED [PDR-0033]; Stage-0 telemetry DRAINED to main [PDR-0034], PUSHED to origin 2026-07-06 @ 50af6643)

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
- **UNIFICATION acted (PDR-0034):** owner authorized "do it now" → the 5 value-free Stage-0
  telemetry commits were **drained onto local `main`** (`e5bab048`→`17b5f634`, +965/−16). Not
  "merge the branch": main already had the squashed Stage-2, so the drain was the narrow unit.
  Reconciled `synergy_bonus`→`interaction_bonus`, unioned `reward_variance.py`, added only
  `COMPONENT_TERMS`. **Byte-identity golden PASSED** + 410 changed-domain tests green +
  **pytorch-expert GO**. `feat/phase-minus1-scale-falsifier` (redundant) deleted.

## In flight
- **Nothing active in code.** Stage-2 is unblocked and ready to pick up (below).
- **PUSH DONE (2026-07-06):** owner authorized "merge to main" → pushed `2f95aa04..50af6643`;
  `origin/main` now at `50af6643` (ahead 0 / behind 0, tachyon-beep). The belt-and-braces gate
  was **reframed**: the full-suite run is un-reachable (it wedges on the pre-existing GPU
  `test_data_opt.py` / `test_dual_ab.py` hangs — provably NOT drain importers), so it was
  replaced by a **complete drained-module importer gate = 862 passed / 0 failed** on the pushed
  HEAD, atop byte-identity golden 2/2 + pytorch-expert GO.

## Facts the next session must not relitigate
- **The gate PASSED and is ACCEPTED** (median 1.017, 41/42 ≫ 0.40, flat across the observed
  window; PDR-0033). Stage-2 is justified; do NOT re-read the premise as unproven or re-run
  the OFF control. Value-free (PDR-0028), not the contaminated ON-leg λ-metric.
- **The reaped run was NOT a code fault** (PDR-0033). Do not "fix" the Stage-0 telemetry for a
  memory leak — there is none. `component_additends` is a pre-allocated SoA.
- The instrument is DONE (5 commits). Do not rebuild it. `return_variance_telemetry=True` +
  `reward_mode=SHAPED` + `hra_value_decomposition=False` = the control posture.
- **UNIFICATION delta RESOLVED (PDR-0034):** `interaction_bonus` is canonical on main (matches
  the `RewardComponentsTelemetry` field); `synergy_bonus` is gone. Do not reintroduce it.
- Honor MAJOR-3 sequencing: the value-free gate on the OFF control run comes BEFORE any HRA
  ON run (done — this reading is the OFF control).

## Open questions / blocked-on-owner
- **PUSH — DONE (2026-07-06).** `origin/main` @ `50af6643` (ahead 0 / behind 0). Resolved; see
  "In flight" for the reframed gate and evidence.
- **Retire `feat/ev-stab-stage2-hra`** — the remaining unification step, AFTER Stage-2 acceptance.
  Its granular Stage-2 ≡ main's squash (byte-identity confirmed); what still lives ONLY there =
  the oracle-sandbox WIP commits → finish-or-abandon separately (PDR-0030) before deleting.
  **TOPOLOGY MOVED (2026-07-06 11:41, concurrent session — NOT this session's push):**
  `feat/ev-stab-stage2-hra` was fast-forwarded to the `feat/sanctum-layout` tip `6120ac9a`
  ("merge: sync feat/ev-stab-stage2-hra into feat/sanctum-layout") — the two branches now
  coincide. Retire-ev-stab and the sanctum-layout disposition are therefore now ONE entangled
  call; reconcile the merged tip before deleting either. (Main is unaffected: the drain landed
  independently.)
- Standing placeholders: north-star / rent TARGETs (metrics.md) still owner-set.

## Last checkpoint did (checkpoint #20)
- PDR-0033 (control run reaped at 43%; leak RULED OUT; gate PASS on the partial read). **Owner
  ACCEPTED**; **esper-lite-3d67b09687 CLOSED**.
- PDR-0034 (unification recommendation + execution): drained the 5 Stage-0 commits to local main
  (byte-identity golden + 410 tests + pytorch-expert GO); deleted redundant `feat/phase-minus1`.
- metrics.md EV Stage-0 gate row: confirmed on the 42-update partial read; owner accepted.
- No horizon change (EV-stab stays Now).

## Next session, start here
**Proceed to Stage-2 acceptance** (esper-lite-2a4b56e719). Stage-2 HRA is already built on
`feat/ev-stab-stage2-hra`; the remaining work is its acceptance criteria — **MAJOR-1 `ev_sum`
hard floor + MAJOR-3 provenance** — and the **paired fresh-init A/B on EV_main liftoff**
(the noisy cf head must not mask the de-shaping win in the aggregate). Memory pointers:
`ev-stab-stage2-impl-state`, `ev-variance-research-verdict`. (esper-lite-e6382020d2 also
unblocked.)
