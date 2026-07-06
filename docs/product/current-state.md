# Current State — Esper        Checkpoint: 2026-07-06 (checkpoint #21 — EV Stage-0 DRAINED + PUSHED to origin [PDR-0035]; branch topology consolidated 12→7 [PDR-0036]; EV research corpus salvaged to main)

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
- **BRANCH TOPOLOGY CONSOLIDATED (2026-07-06):** 12→7 branches, 8→3 worktrees; EV research corpus
  salvaged to main (`127f25cb`); codex fork-base archived to `origin/archive/oracle-sandbox-wip-2026-06`.
  Only the live stack remains. Details under Open questions → "Retire `feat/ev-stab-stage2-hra`".

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
- **Retire `feat/ev-stab-stage2-hra`** — the LAST remaining branch disposition, AFTER Stage-2
  acceptance. **TOPOLOGY CONSOLIDATED this session (2026-07-06): 12→7 branches, 8→3 worktrees.**
  Retired 6 fully-absorbed dead branches + 5 stale/experiment worktrees (`codex/*`, `post-p01-*`,
  `wf_*`, both `ab-*`) — each proven lossless by ancestor-of-main or patch-id equivalence. The
  codex fork-base was preserved as **`archive/oracle-sandbox-wip-2026-06`** on origin (holds the
  35-file oracle-sandbox WIP); its unique EV research corpus was **SALVAGED to main (`127f25cb`)** —
  the 4 `docs/research/` variance-reduction reports + reward-redesign + ppo-learning-gate plans.
  What remains is ONLY the **LIVE STACK**: `feat/ev-stab-stage2-hra` (active — owner landing TUI)
  with `feat/sanctum-layout` + `feat/entropy-thermometer-fix` folded in as ancestors. These three
  are ONE unit; they retire together when the stack merges to main after Stage-2. Granular Stage-2
  ≡ main's squash (byte-identity confirmed). Kept refs: `0.3.0` (release), `backup/0.1.1-pre-p01`.
- Standing placeholders: north-star / rent TARGETs (metrics.md) still owner-set.

## Last checkpoint did (checkpoint #21)
- **PDR-0035** — PUSHED the Stage-0 drain to `origin/main` (owner "ok lets merge to main").
  Pre-push gate REFRAMED: the full-suite run is unreachable (wedges on env GPU hangs
  `test_data_opt`/`test_dual_ab` — not drain importers) → replaced by a complete drained-module
  importer gate = **862 passed / 0 failed**. origin advanced `2f95aa04 → 127f25cb → bb9ae6af`.
- **PDR-0036** — BRANCH TOPOLOGY CONSOLIDATED **12→7 branches, 8→3 worktrees** (PDR-0030 discipline).
  Retired 6 absorbed branches + 5 stale/experiment worktrees, all lossless (ancestor-of-main /
  patch-id). Codex fork-base ARCHIVED → `origin/archive/oracle-sandbox-wip-2026-06`; its unique EV
  research corpus SALVAGED to main (`127f25cb`).
- No horizon change (EV-stab stays Now). No new metric readings (housekeeping). No tracker status
  changes (esper-lite-3d67b09687 already CLOSED at #20; Stage-2 esper-lite-2a4b56e719 still ready).
- **Checkpoint #21 commit is on LOCAL main (NOT pushed — /product-checkpoint does not push).**

## Next session, start here
**Proceed to Stage-2 acceptance** (esper-lite-2a4b56e719). Stage-2 HRA is already built on
`feat/ev-stab-stage2-hra`; the remaining work is its acceptance criteria — **MAJOR-1 `ev_sum`
hard floor + MAJOR-3 provenance** — and the **paired fresh-init A/B on EV_main liftoff**
(the noisy cf head must not mask the de-shaping win in the aggregate). Memory pointers:
`ev-stab-stage2-impl-state`, `ev-variance-research-verdict`. (esper-lite-e6382020d2 also
unblocked.)
