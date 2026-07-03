# PDR-0017 — tau delivered (+0.28 pp, provisional lower bound); epsilon-plateau trigger FIRED; disposition PROPOSED for owner

Date: 2026-07-03   Status: accepted for the delivery + task closure (within grant);
the trigger DISPOSITION is **proposed — owner decision at the gate**   Author: Claude (agent)
Related: PDR-0014, PDR-0015 (whose sharpened trigger fired), esper-lite-94869250f1 (closed
@ 05569309), esper-lite-f22a1d48a7 (memo comment #98),
docs/analysis/2026-07-03-pin-e-placebo-noise-floor.md

## Context
The full PIN-E runs completed (2 arms × 3 seeds, 3240 terminal samples/arm). Results:
non-degeneracy PASSED both arms (PDR-0014 trigger NOT fired; efficiency identity exact on
all 2160 terminals; Rec3 masking equivalence bit-identical). D1 GATE-0 line: the credit
system is UNBIASED at the freeloader threshold (|bias| sub-grid) with per-step noise std
0.06–0.18 pp. D2: tau(1e-3) = +0.2401, CI [+0.2398, +0.2721]; tau(1e-4) = +0.1201,
CI [+0.1000, +0.1401].

**The epsilon-plateau condition (PDR-0015) FIRED**: the arm CIs are disjoint — the floor
tracks the null player's own perturbation magnitude (sub-linearly, with a ~one-grid-step
basement), it is not ε-independent measurement noise.

## Options
1. Mechanically execute the drafted consequence (re-scope to methodology §5 option (a),
   K resampled val minibatches). Rejected as the sole response: option (a) measures
   val-resampling noise at fixed weights and is equally magnitude-blind — it cannot
   repair the finding it would be invoked for.
2. Deliver tau as a CONSERVATIVE-PROVISIONAL **LOWER BOUND** (+0.28 = 1e-3 upper CI
   rounded up), with first-ON-run recalibration upgraded from advisory to MANDATORY, and
   put the accept/supplement/re-scope choice to the owner at the gate (chosen — as a
   PROPOSAL, not a unilateral disposition).

## The call (proposed disposition)
Recommend the gate accept tau = +0.28 pp as a lower bound and proceed to F2, because:
(a) the direction is conservative-compatible — real committed seeds perturb more than
ε=1e-3, so their zero-synergy excess likely exceeds +0.24, meaning the placebo tau
under-estimates and the term's other guards (cap, normalized_cap, efficiency clamp G)
bound the damage of an under-sized deadband until recalibration; (b) the finding
empirically confirms the heteroscedasticity caveat pre-recorded in the task spec.
The harness task was closed with all three deliverables met; the disposition question
lives at the gate issue where the owner decides.

## Interaction flag (recorded on the gate, owner-raised)
The top-up pays ONLY at FOSSILIZE; audit findings F1/F5 (PDR-0016) are confirmed
commitment suppressors on top of coalition blindness (fossilize 0.207/ep vs germinate
12.4). If the OFF-arm policy rarely fossilizes, the OFF/ON A/B risks a FALSE NEGATIVE —
the term never fires often enough to move committed-J. The prioritization review must
decide whether F1/F5 remediation (or at minimum the occurrence probes) precede the A/B.

## Reversal trigger
If the first ON calibration run measures real-magnitude null-player excess with
P99 > 2× the placebo tau (i.e. the lower bound is not merely conservative but
uselessly loose), the deadband method reopens: revisit option (a) supplementation or a
magnitude-scaled tau(‖seed‖) model before the A/B is read.
