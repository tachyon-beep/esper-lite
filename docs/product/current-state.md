# Current State — Esper        Checkpoint: 2026-07-12 (~16:15) · post-#40 dispatch (PDR-0062/0063/0064; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet). The 600-round OFF diagnostic is
DONE and READ; the epic direction is DECIDED (PDR-0064): **read the ON-leg V_cf-bootstrap
endogeneity — the dominant unknown — before committing to a redesigned-HRA (A′/B)
experiment.** The n=2 ON-leg Gram diagnostic is now in flight; its held-out
calibration-rescue read is the A′/B discriminator.

## In flight
- **ON-leg Gram diagnostic, n=2 (seeds 41 cuda:1, 42 cuda:0)** launched 2026-07-12 16:12
  @ `92d998bd` (`telemetry/stage2_on_longdiag/run_on_longdiag.sh`, setsid — survives
  reboot). Both training clean (rounds ticking, 0 crashes); ON vg Gram family confirmed
  emitting (`vg_*_cf` present). 600 rounds each, ~20h → done ~midday-ish 2026-07-13. Logs:
  `telemetry/stage2_on_longdiag/logs/on-longdiag-s{41,42}.log` (done = `exited rc=`).
  **QUIET BOX while they write** (PDR-0056). Monitor armed for completion/crash.
- **Branch-survivor consolidation** (esper-lite-1f1e55f58f, in_progress): audit DONE,
  direction decided (PDR-0062, main-as-trunk merge), all 4 owner sign-offs RESOLVED
  (HRA drop; harness KEEP; tag-anchored retire; anti-recurrence deferred). EXECUTION still
  gated on the ON-leg read + the eventual merge window.

## Facts the next session must not relitigate
- **PDR-0064 (epic DECIDE):** ON-leg Gram diagnostic FIRST; NOT a jump to A′/B. Pre-committed
  reading = 3 plateaus (cf target-process Var(G_cf) / cf learner Var(e_cf),ev_cf /
  normalizer) + held-out affine calibration-rescue (fit 250–350, test 400–600): rescues
  total EV ⇒ A′ credible; cannot ⇒ B favoured. Diagnostic, not a gate; re-uses rejected
  HRA as substrate only.
- **OFF longdiag read (`docs/analysis/2026-07-12-longdiag-s41-read.md`, `92d998bd`,
  advisor + external-peer reviewed):** operational verdict SCHEDULE-TRANSIENT / post-anneal
  plateau (structural runaway ruled out); scientific claim softened (cause not causally
  isolated — schedule/penalty/host-maturation confounded at n=1). Objective A stays
  rejected. A′ modestly up, B still favoured pending this ON read. Two metric-label fixes
  banked (vg mean-bias ≠ normalizer-lag; value_target_scale = running normalizer).
- **PDR-0062 (branch-survivor):** main-as-trunk via history-preserving merge; harness KEEP;
  execution gated. **PDR-0063 (anti-recurrence):** proposed, deferred.
- Owner grant confirmed 2026-07-10, re-confirmed 2026-07-11. Owner pushes to origin
  themselves; agent never pushes.

## Open questions / blocked-on-owner
- **A′/B DECIDE** — after the ON-leg read, on the calibration-rescue + 3-plateau evidence.
  (Owner reserved; serial focus.)
- **Branch-survivor execution timing** — the merge/retirement runs after the ON read; all
  sign-offs are in, only sequencing remains.
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last did (post-#40 dispatch)
- OFF diagnostic completed 600/600 rc=0; read scored (PDR-0055), advisor-corrected, then
  external-peer-reviewed and revised (2 metric-label fixes + design notes; `92d998bd`).
- Owner epic DECIDE = ON-leg Gram diagnostic (PDR-0064); harness disposition resolved KEEP.
- Launched n=2 ON-leg diagnostic; verified startup + ON vg family emission; monitor armed.
- Filed a monitoring-scar observation (loose pgrep liveness check).

## Next session, start here
Check the ON diagnostic: `grep "exited rc=" telemetry/stage2_on_longdiag/logs/on-longdiag-s*.log`.
Both running ⇒ quiet box, light read-only work only. Both done rc=0 ⇒ score the PDR-0064
reading (3 plateaus + held-out calibration-rescue A′/B discriminator) over rounds 250–600
per seed, then bring the owner the A′/B DECIDE. Any rc≠0 ⇒ RCA first (ON leg builds the
cf_value_head + per-stream normalizers — a distinct code path). After the A′/B DECIDE,
the branch-survivor merge execution is unblocked (all sign-offs in).
