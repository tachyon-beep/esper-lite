# PDR-0033 — EV Stage-0 control run reaped at 43%; leak ruled out; gate PASS stands on the partial read

Date: 2026-07-05   Status: accepted (autonomous within grant: "accept against
criteria"; the disposition below records a settled measurement, and the one
compute-spend question — a formal full-400 rerun — is surfaced to the owner, not
executed)
Author: Claude (agent)   Supersedes: none. Refines PDR-0032 (which read the same
gate PRELIMINARY at n≈14 while the run was in flight). Related: PDR-0028 (the
value-free gate IS the gate; the ON-leg λ-metric is contaminated). Tracker:
esper-lite-3d67b09687 (Stage-0 task), esper-lite-f25b71c165 (epic).

## Context

PDR-0032 recorded the Stage-0 advance gate as a PRELIMINARY PASS (median ~1.02 at
n≈14 updates) while the canonical Stage-2-OFF control run was still in flight, and
armed a reversal trigger on the FULL-distribution median at run completion. The run
did **not** complete: it terminated externally at **batch 43 / episode 172 of 400
(~43%)** at ~21:59 machine-time, and the process was gone by the next check (~4 min
later). This is the "process was reaped" contingency current-state.md had already
flagged.

Two questions had to be answered before touching the gate verdict:
1. **Did the NEW Stage-0 telemetry cause the death?** This was the first real run
   with `return_variance_telemetry=True` — the first to allocate the
   `component_additends` SoA and run `collect_component_rewards()`. A memory leak
   there would be a bug in code PDR-0032 called "delivered clean," AND would make any
   rerun die at ~172 again.
2. **Is a 43% read a valid gate reading, or is it contaminated / too thin?**

## Options

- **(a) Investigate the death cause; if the code is clean, accept the partial read as
  the gate result and offer a formal full-400 rerun as OPTIONAL closure.**
- **(b) Treat the run as failed, discard the read, relaunch immediately** for a full
  400 before recording anything. Rejected as premature: relaunching an
  un-diagnosed failure may just repeat it, and it presumes the partial read is
  invalid without checking.
- **(c) Silently keep PDR-0032's "preliminary, still accumulating" framing.**
  Rejected: dishonest — the run is dead, not accumulating.

## Call

**(a), executed.**

**Death cause — leak RULED OUT; termination was EXTERNAL.** Four independent lines
of evidence converge:
1. **Code-proven non-leak.** `component_additends` is a pre-allocated fixed tensor
   `torch.zeros(n, m, len(COMPONENT_TERMS))` (~31 KB at n=4, m=150), written by direct
   index in `add()`, and `reset()` restores `step_counts=[0]*n` each rollout so the
   valid range is reused, never grown. There is no per-run accumulator.
2. **Flat batch cadence.** All 43 batches ran 71–83 s with no upward drift, then an
   abrupt stop. A leak → swap-thrash → OOM death produces a *rising* batch-time curve;
   metronomic cadence ending instantly is the signature of an external SIGKILL/SIGTERM.
3. **Memory state.** 41 GiB RAM free after death; no OOM string in any readable kernel
   log; top RAM users are unrelated long-lived processes (session-start
   `loomweave analyze`, VS Code, node).
4. **No `CUDA out of memory`.** A GPU-OOM would have left a RuntimeError traceback in
   the merged log; the log cuts off clean mid-epoch with none.

Most plausible trigger: the training process group was reaped as the session compacted
/ the tailing monitor timed out (~22:00). **The Stage-0 instrumentation did not cause
the death, and a rerun would not repeat it.**

**Gate reading on the partial run (42 cf-active updates, ep 4→172):**
median `Cov(R_cf,R)/Var(R)` = **1.017**, IQR **[1.006, 1.033]**, **41/42 (98%) > 0.40**
(min 0.10, max 1.16), residual_share **~1e-17**. The share is **FLAT at ~1.0 across the
entire observed window** (random-init through epoch 21 / ~50% acc) — it does not decay
as the host matures. The single sub-0.40 reading is one early transient (ep 20) that
snaps back by ep 24. `cf_share` slightly > 1.0 is expected, not a bug: cost components
(rent/penalties, negative rewards) carry net-negative covariance share, which R_cf more
than fills. residual ~0 confirms the signed-additend partition reconciles on LIVE data
(the §7 completeness keystone holding in production).

**Disposition: the gate PASS STANDS on the partial read.** A partial run honestly
labeled "172/400" is not *contaminated* — that was PDR-0028's sin (a V_cf-biased metric
presented as valid). It is merely smaller-n, and the discipline is satisfied by honest
provenance, not by a rerun. A formal full-400 rerun is **optional closure** for the
record, surfaced to the owner as a compute-spend choice — **not** a validity gate on
Stage-2. Recommendation: accept now; the trajectory (flat 1.0 across random→50% acc,
2.5× the threshold, IQR width ~0.03) leaves no plausible path to a <0.40 full-run median.

## Rationale

The reversal trigger PDR-0032 armed against "full-run median" was a rigor hedge, and the
partial read discharges its intent: the worry a full run would guard against is that the
cf share *collapses* toward 0.40 in steady state. The observed window already spans
random-init → ~50% acc with the share pinned at ~1.0 and no downward slope — the collapse
did not begin. Note the honest limit: epoch 21/150 is NOT converged, so this is "held
across the observed window," not "steady state at convergence." Even so, 42 updates at a
tight ~1.0 is a far stronger read than PDR-0032's n≈14, and re-running merely to watch
1.02 → 1.02 is low-information.

## Reversal triggers

- If a formal full-400 rerun is later run and its median `Cov(R_cf,R)/Var(R) < 0.40`
  → reopen (the de-shaping premise falsifies; epic re-scopes to Stage-1-only). Remote:
  41/42 partial updates ≫ 0.40, flat across random→50% acc.
- If the Stage-0 instrumentation is later shown to have caused the reap after all
  (e.g. a rerun dies at ~172 again) → reopen the leak investigation; this PDR's
  "code is clean" call is falsified. (Evidence to date makes this remote.)
- PDR-0032's Stage-2 guardrail triggers (host-accuracy contribution regression on
  de-shape) still stand unchanged.
