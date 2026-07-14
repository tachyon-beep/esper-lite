# PDR-0080 — Round-14 calibration: the shelter half of READ 4 is under verification (propensity not a known constant); the coupling rationale is "either-alone-is-degenerate"; the first experiment is a TRIANGULAR attributable design

Date: 2026-07-14   Status: accepted (both primes converged; ratifying their calibration + the experiment shape)
Calibrates PDR-0079 (does NOT reverse it — the coupled direction stands, the emphasis and the banked confidence change). Related: PDR-0078 (READ 1/3), round-14 primes (gpt-prime + claude-prime), advisor.

## What does this buy?  (REQUIRED — PDR-0068)
It prevents banking a propensity-artifact (the shelter half) as a load-bearing reason to re-prioritize the floor, and it structures the first GPU experiment so the settlement and floor effects are *attributable* rather than confounded. Measured: the shelter estimate becomes conditional on 3 verification reads; the experiment yields `B−A` and `C−B` as clean contrasts.

## Calibration of READ 4 (PDR-0079)
1. **The two halves are NOT equal-confidence.** The **residual (permanence) half is SOLID** — Support-1 (val@1, 89–95% residual) is a DESIGN-based result: the READ-3 exact randomization (op head's own action, propensity exactly 0.15) evaluated at a horizon where forced-PRUNE structurally cannot yet have fired. No IPCW, no prune propensity. The **shelter half is IPCW-model-dependent and UNDER VERIFICATION:** it censors "this seed gets pruned," which is NOT the op head's own action — it needs the **SLOT head** (which selects the seat; `slot_entropy≈0` in 97–98% of HOLDING → deterministic/learned, a SECOND non-constant propensity) AND PRUNE is **masked below `MIN_PRUNE_AGE=5`** (so the per-decision hazard is `0.15 | legal`, not a flat 0.15). The flat-0.15 IPCW weight is therefore mis-specified, and the arithmetic confirms it: `0.85^13 ≈ 12%` survival ≫ the reported 55%. So the shelter magnitude and the ~50/50 split are **provisional pending verification.**
2. **RENAME the components (gpt-prime):** "controlled forced-PRUNE **shelter** effect" (was intrinsic-vs-shelter framing) + "**residual FOSSILIZE effect beyond the no-future-PRUNE regime**." Do NOT call the residual "intrinsic permanence value" until the val@1 event ordering is verified (gpt-prime: p10 first-prune delay = 1 epoch, so a PRUNE opportunity *might* exist before val@1 in a minority — verify the ordering directly; if some can occur, it is "residual beyond controlled no-PRUNE," not "intrinsic").
3. **Emphasis INVERTED (claude-prime §2):** the **settlement is the well-evidenced leg** (rests on the solid residual half); the **floor-repair is strongly-motivated but rests on the softer half** (under verification). PDR-0079's "stronger case for the floor repair than the settlement" is retracted.

## The REAL coupling rationale (claude-prime §3 — stronger than 50/50, independent of the shelter verification)
Couple the bet because **either fix alone yields a DEGENERATE policy**, not because "each fixes half":
- **Settlement alone (floor intact):** commits are still 93–94% floor-forced and good seeds are still force-destroyed → the settlement can't act on a policy that can't choose.
- **Floor-repair alone (reward intact):** the op head's only learnable region (|H|≥2) was shaped under a reward that forfeits −113 at commit and pays ~3/epoch to hold → the learned preference is **PARK**. Un-forcing the floor yields a policy that holds good seeds forever and **never commits → zero fossils** (round-10 §3: the floor IS the commitment).
This holds whether or not the shelter magnitude survives §1.

## The first experiment: TRIANGULAR attributable design (gpt-prime) — NOT both-at-once
| Arm | Obs | Settlement | Op-floor primitive |
|---|---|---|---|
| **A control** | obs-v4/L1 | current SHAPED (no settlement) | shipping hard floor |
| **B settlement-only** | obs-v4/L1 | non-selectable settlement | shipping hard floor |
| **C coupled** | obs-v4/L1 | same settlement as B | forced-PRUNE-safe differentiable primitive |
All: K=4, same critic/entropy/optimiser, same reward except the settlement, paired fresh inits, identical telemetry. `B−A` = settlement effect under the existing floor; `C−B` = incremental floor/PRUNE-repair effect with a valid ledger; `C−A` = combined. No full floor-repair/no-settlement arm (unsafe under the discontinuous reward; optional frozen-abort smoke only). r9 is an archival reference, NOT the concurrent control (different regime).

## Design requirements
- **Floor-repair:** remove **structurally-guaranteed PRUNE probability** — NOT merely make the hard floor differentiable while keeping a per-action PRUNE minimum. Candidates: (a) a smooth constraint on total non-WAIT probability preserving learned relative op-probs; (b) a safe-action reserve where anti-WAIT mass is never forcibly allocated to destructive PRUNE. Invariants: no positive lower bound on PRUNE just because it is legal; exact sampled+evaluated `q`; differentiable; no STE; hard transform behind a flag as control; per-head + per-op KL logged; explicit rate of PRUNE at structural minimum; no global λ that changes every head differently.
- **Settlement:** non-cancellable scheduled confirmation (`REQUEST_FOSSILIZE` → uncancellable → seed measurable to the next settlement boundary/audit window → α frozen → quote from pre-request-fixed timing → integration + ledger at the boundary). Close the finite-horizon/discount, negative-value, stale/never-measured eligibility, one-shot-bonus netting, double-payment, terminal/truncation, multi-seed, quote-vs-disbursement, and window-delay-cost holes.

## The verification reads (last things r9 owes — decide whether the shelter half survives)
1. PRUNE-legality rate at pinned WAIT-arm decisions (age≥5 unmasked) — the mask correction to the propensity.
2. Empirical per-decision destruction hazard for a focal HOLDING seed (MEASURE, don't derive from 0.15; must reconcile with 55%/13-epochs).
3. Slot-head propensity for the focal slot at PRUNE (is it policy-dependent? if yes, shelter is not identified by a known constant → **retract the 50/50, keep Support-1**).
4. val@1 event-ordering verification (does a PRUNE opportunity exist before val@1? → "intrinsic" vs "residual beyond no-PRUNE").
5. pinned-only vs all-later-PRUNE sensitivity (quantify the ~5% deliberate-prune approximation).
(Note: the α-target "random dominant action" concern is ALREADY REFUTED by A3 — alpha_target enum n=3, realized 3-way spread 0.7:43%/0.5:33%/1.0:24%.)

## Reversal trigger
- If verification-read 3 fires (slot selection policy-dependent) → the shelter estimand is unidentified → retract the ~50/50; the residual (settlement) half carries the finding; the floor-repair remains motivated by the "either-alone-degenerate" rationale, not by a shelter magnitude.
- If the val@1 ordering shows pre-measurement PRUNE opportunities are common → downgrade "intrinsic" to "residual beyond no-PRUNE."
- Hold ALL GPU work until the 5 verification items + the triangular design + both spec docs are in a single pre-registration.

## OUTCOME — READ 5 verification (2026-07-14): the shelter half is RETRACTED; the finding is the intrinsic/settlement half
The 5 checks came back decisive (PR20 fired). **The READ-4 shelter half is NOT identified offline — RETRACTED**, three ways:
1. **Mis-specified in degree (Check 2):** empirical per-decision hazard 0.109/0.127 (not 0.15) over k̄≈1.8 pinned decisions → the IPCW predicts 79–82% survival but 55% are observed pruned → the `(1/0.85)^k` weight (mean ≈1.3) under-corrects by a wide margin; most prunes fall OUTSIDE the modeled pinned-HOLDING hazard points. (Check 1: the age-mask is NOT the cause — HOLDING seeds are all ≥5.)
2. **Mis-specified in kind (Checks 3 + 3b, decisive):** the SLOT head is DETERMINISTIC (slot_confidence median 1.0, 98% ≥0.9), so "this seed gets pruned" is a state-dependent point mass, NOT a known 0.15; and pruning is quality-SELECTIVE — it removes LOWER-LOO seeds (FOSS-destined lifetime LOO +6.9/+6.6 vs PRUNE-destined +6.1/+5.9), so the "shelter" was confounded UPWARD by selection-on-badness.
3. **Under the clean estimand (Check 5): shelter → +0.23–0.31 (~16–20% of TE), residual → ~+1.26 (~80–84%).** Only 46–48% of WAIT-arm prunes are floored/pinned (the rest deliberate, no known propensity). Check 4: val@1 is pre-mediator for ~90% → the residual is ~90% genuinely intrinsic.

**What survives / the corrected plan:**
- **The total effect (+1.5 pp, READ 3) STANDS; its decomposition collapses to the INTRINSIC-PERMANENCE / residual half** (identified, assumption-light). → **the non-selectable SETTLEMENT (arm B) is the robust, well-evidenced bet.**
- **The "floor/PRUNE-repair co-required via ~50% shelter" claim (PDR-0079 item 3) is WITHDRAWN.** The floor's forced-PRUNE is NOT an indiscriminate destroyer — the target selection is a sensible LEARNED quality-pruning (slot head); only the timing/whether is floor-forced. So arm C is **RE-SCOPED**: its op-primitive targets the FORCES-COMMITS / gradient-dead-zone facet (round-8 per-num_valid differentiable floor, gradient-restoring, exact q, no STE, hard-floor control), NOT a "forced-PRUNE-safe" removal (which could stop sensible quality-pruning). Whether C is in the FIRST experiment (vs a clean A/B settlement experiment with the gradient-fix as a pre-registered 2nd arm) is a design call handed to the drl-expert.
- **The coupling rationale is now ONLY "either-alone-degenerate"** (settlement-alone: commits stay 94% floor-forced; gradient-fix-alone: the reward still prefers park) — it never depended on the shelter magnitude.
- The **55% cumulative prune incidence** remains a valid DESCRIPTIVE fact; it licenses NO causal shelter conclusion.
This is the 7th confident conclusion retracted before it cost a GPU-run; the direction (pursue the non-selectable settlement) is unchanged and now rests on the identified half.
