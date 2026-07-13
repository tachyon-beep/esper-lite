# PDR-0065 — ON-leg Gram read: the pre-committed A′/B discriminator proved invalid; A′-vs-B underdetermined

Date: 2026-07-13
Status: accepted (finding banked within grant; the A′/B/C DECIDE it reframes is owner-gated)
Supersedes: the pre-committed A′/B reading in PDR-0064 (on validity grounds — same pattern
as PDR-0052 demoting LEG-B when its validity gate proved unsatisfiable on real data).
Read: `docs/analysis/2026-07-13-on-leg-gram-read.md` (`a87ba291`, advisor-reviewed).

## Context

The n=2 ON-leg Gram diagnostic (PDR-0064) completed clean (both seeds 600/600, rc=0) and
emitted the full ON vg family. Scoring the pre-committed held-out affine calibration-rescue
discriminator surfaced that the instrument itself does not discriminate the A′ on the table.

## The finding

1. **The discriminator is near-tautological.** Under Objective A, `V_main`→`E[G_main|s]`,
   `V_cf`→`E[G_cf|s]`, and `returns_total = returns_main + returns_cf` by construction, so
   `V_main + V_cf ≈ E[G_total|s]` — the L2-optimal linear predictor. `(1,1,0)` is optimal
   by construction; affine recalibration can only catch gross miscalibration; "no rescue"
   is EXPECTED under both A′ and B. The observed +0.04 lift is the ~15–20% normalizer
   over-estimate (plateau-3), not an A′/B signal.
2. **It only refutes light-A′** (a post-hoc calibration layer on existing heads). PDR-0060's
   A′ is heavy-A′ (primary `V_total` *directly trained*), which learns representations the
   frozen Objective-A heads cannot express — the frozen-head projection is structurally
   blind to it.

## What the diagnostic DID establish (robust, n=2)

- **Objective A stays rejected** (unchanged, PDR-0059).
- **The total-fit deficit is real and cf-driven** — `ev_sum` ~0.55 ≪ `ev_main` ~0.84;
  `ev_cf` ~0.55–0.60 (up from PDR-0060's 0.44–0.54).
- **It is NOT a gross-calibration artifact** — heads decently calibrated; **light-A′ is
  refuted.**

## What stays UNDETERMINED

- **B vs heavy-A′.** Underdetermined — low `ev_cf` is equally consistent with "cf
  intrinsically hard (B)" and "cf under-trained by the endogenous bootstrap, fixable by
  directly training `V_total` (heavy-A′)"; `ev_cf` still rising on seed 42 leans mildly
  against a hard ceiling. Plateau-1 (cf-target non-stationarity) is split 41-vs-42 — the
  disagreement-⇒-extend-n trigger fired, but that leg speaks to non-stationarity, not the
  ceiling.

## Reframed DECIDE (owner-gated) — what actually moves A′/B

The frozen-head Gram data cannot separate heavy-A′ from B; more of the same read won't
help. The discriminating next steps:
- **Bounded heavy-A′ pilot** — train a direct `V_total` head and see whether total EV
  breaks the ~0.61 affine ceiling. The only *direct* test ("to know if retraining helps,
  retrain"); cheapest thing that genuinely moves the decision.
- **Path-C state-conditional information probe** — measure the true `Var(G_total|s)`
  ceiling (offline; vg scalars do not contain it).
- **Extend n on plateau-1** — firms the non-stationarity leg only; does NOT resolve the
  ceiling. Lowest value for the A′/B question.

Recommendation to the owner DECIDE: the A′ pilot (it directly tests the live hypothesis);
Path-C if the pilot also stalls. Not another frozen-head diagnostic.

## Reversal trigger

If a heavy-A′ pilot's directly-trained `V_total` breaks materially above the ~0.61 affine
ceiling, heavy-A′ is live and B is not the answer; if it cannot beat the ceiling, the
deficit is representation/information-bound → B or C.

## Process note (recorded as memory)

Third instance this session of a directional verdict over-reading an instrument. Design-time
guard adopted: for any diagnostic read, ask "what can this metric STRUCTURALLY not tell me?"
BEFORE running it (see [[instrument-validity-before-interpretation]]).
