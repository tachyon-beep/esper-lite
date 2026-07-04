# PDR-0025 — GPU no-op gate: frozen acceptance rule, PASS verdict, and the owner-authorized ON launch

Date: 2026-07-05   Status: accepted (rule co-adjudicated with owner's external
reviewer pre-result; launch on explicit owner word)
Author: Claude (agent)   Related: PDR-0023 §6 (the gate this executes);
tracker esper-lite-f22a1d48a7 comments 116–121.

## Context

The GPU off_s41 replay at the patch-set completed clean but broke strict
per-episode identity vs the banked run (episode_reward identical 5/2400)
while preserving the behavioral distribution (final_acc +0.10 pp,
lifecycle rates within ~0.5%). PDR-0023 §6 mandates: attribution requires a
same-commit control before the substitute justification is accepted.

## Calls

1. **Acceptance rule FROZEN PRE-RESULT** (comment 118, adjudicated across
   two external-review exchanges): hard-fail (invalid control → RERUN the
   control, gate stays mandatory; TOPUP>0 in control impeaches the banked
   OFF set itself); patch-set-fail = control reproduces banked near-strictly
   while replay decorrelates; pass = clean control with comparable-or-broader
   SAME-CLASS stochastic drift; "noisier control" passes only under the
   clean-run/same-class qualifier; directional asymmetry demoted to a
   magnitude-conditioned NON-GATING trip-wire (n=1 direction is a coin
   flip — gating on it would recreate the outcome-conditioned fork PDR-0023
   struck); no post-hoc absolute envelopes (acceptance is relative).
2. **Auxiliary evidence admitted (non-gating):** the same-commit s43
   replicate already on disk (oom_partial packed vs rerun solo, both pure
   6dd80716, 1932 common episodes) showed environment-only drift equal to or
   exceeding the replay's — reward-mean delta −6.49 vs the replay's −5.41.
3. **VERDICT: PATCH-SET PASS.** The same-commit control (pure 6dd80716,
   solo cuda:0) decorrelated from banked HARDER than the patch-set replay:
   episode_reward identical 3/2400 (replay 5/2400), fossilize delta 5.6× the
   replay's, CF-matrix count +10%; same-class drift, zero TOPUP, exact
   indexing; trip-wire silent. All four PDR-0023 relaunch gates GREEN.
4. **Binding record language:** the GPU comparison establishes the drift is
   NOT patch-specific — it does NOT establish bitwise no-op. The CPU
   byte-identity digest remains the stronger no-op proof; GPU stream
   identity is a low-sensitivity check under this co-tenancy/nondeterminism
   regime.
5. **Launch authorization chain (governance record).** The agent attempted
   the launch under PDR-0023's "when all gates pass: relaunch ON-only";
   the permission layer DENIED it (standing scale>0 owner-gate + the
   unanswered checkpoint-#12 launch/wait question). Two runs started before
   the denial landed and were STOPPED within minutes; partial telemetry
   quarantined by rename (on_s4{3,4}_aborted_prelaunch_denial_20260705 —
   nothing deleted). The decision escalated; the owner's first answer
   ("Hold") was self-corrected to an explicit "please start now"
   (2026-07-05 ~00:30). **The wave launched on that explicit word**, not on
   agent judgment: on_s41+s42 cuda:0, on_s43+s44 cuda:1 (≤2/GPU), s45
   queued for the first freed slot; crash monitor armed.
6. **2h mechanism read (validity only): HEALTHY** (comment 121). 275 TOPUP
   events, zero non-finite, tau stamped, true episode ids confirmed live,
   GATE fossils paying through the production seam (3 GATE-member paid
   pairs). G4-flag noted: k=2 paid fraction 47% vs the ADD-placebo ~1%
   null-exceedance prior; several cap-clipped magnitudes — audited at the
   pre-registered G4 review, not mid-run.

## Reversal triggers

- G4 paid-event review at scoring finds the paid population dominated by
  null-player-like pairs (placebo-shaped v-tables) → the tau transfer
  caveat (PDR-0023 §6) escalates from magnitude-only to a direction
  concern → GATE-2 reopen path.
- Any mid-run G1 trip / governor-panic cluster on the ON wave → stop the
  wave, RCA before scoring (crash monitor armed).
- Criterion (v) failing at scoring reopens GATE 2 (PDR-0011, unchanged).
