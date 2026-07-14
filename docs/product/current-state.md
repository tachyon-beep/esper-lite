# Current State — Esper        Checkpoint: 2026-07-14 (#44+) · commitment defect RESOLVED to a fossil measurement gap; "make permanence visible" refocus APPROVED (PDR-0074 accepted, PDR-0076 approved-with-conditions; on `feat/ev-stab-stage2-hra`)

## The bet right now
**The commitment defect is a permanent-value MEASUREMENT gap — ONE gap, THREE coupled symptoms + a separate inert channel
(PDR-0074 + round-11/H6 + drl deliverable).** The counterfactual is undefined at both ends of a seed's life (α=0 birth;
permanence — fossils excluded from ablation by design, `vectorized_trainer.py:956-975`), and every consumer of the `None`
inherits it: (1) reward `bounded_attribution` stops firing → −113; (2) OBSERVATION contribution feature = 0
(`features.py:830` `None→0`, H6-CONFIRMED); (3) the critic "overstates" the commit penalty because it is fit to that
lying observation (a SYMPTOM, not independent). The reward AND the observation AND the value function all value a fossil
at exactly 0 — while the freshness channel that could say "STALE not zero" (`γ^epochs_since_cf`) sits built-but-bypassed.
**`hindsight_credit`-inert is a SEPARATE defect (H3): a scaffold channel keyed on `total_improvement`, cap 0.2 — it can
NEVER settle the −113, so L2 is a NEW channel, not a reactivation** (corrects PDR-0074/0075). APPROVED refocus
(PDR-0076, owner-approved with binding conditions): **MAKE PERMANENCE VISIBLE** (L1 freeze-don't-zero + L2
settle-at-fossilize on existing data; H7 GATES L1). Conditions: design against the FOSSIL-FARM adversary (naive L1+L2 ≈
800:1 revenue:cost → commit-everything; sell-at-spike / decay-lie / double-count), L2 revenue ≈ the forfeited stream not
above it, OFFLINE-REPLAY before any run, DESIGN-don't-land, floor + critic held CONSTANT, premise UNMEASURED not wrong.
· metric: does a genuinely-contributing seed retain its measured value across FOSSILIZE; guardrail = the fix does NOT
farm fossils.

## In flight
- **The three gating reads are RESOLVED (PDR-0078).** READ 1: every policy-timed settlement is farmable OOS (lower bound)
  → **settle on a NON-SELECTABLE instant** (fixed-time / terminal / confirmation-window); A[ewma] BLOCKED. READ 2: no
  spike-selection, but uninformative (dead-zone, not safety). **READ 3 (exact |H|=1 ID + IPW/MSM, PR15-validated,
  balance-checked, well-powered 2–5× MDE, both seeds): committing a floor-pinned good HOLDING seed NOW causally BEATS
  WAITING — terminal +1.4–1.6 pp, val-acc all horizons, AUC +1.5–1.7, no added destruction.** Permanence carries real,
  proxy-free product value the reward zeroes out → the fix is worth building. **Calibration (advisor): do NOT bank
  "under-fossilization is a defect, full stop"** — it's commit-EARLIER vs commit-more (WAIT≠never), the contrast excludes
  SET_ALPHA (the real alternative), and part of the gain may be permanence SHIELDING from the floor's own later PRUNE
  (s42 prune↓). One free r9 read (WAIT-arm decomposed by later-pruned) settles it.
- **L1 (obs-v4) + K>1 code diff DONE + tested (uncommitted, under my review).** v3 byte-identical (tested); +9 state_dim
  behind a default-OFF flag; shrink-toward-UNKNOWN + explicit status dims; K=4 (drl-justified, guard activates). 2 test
  issues: one pre-existing (EV-liftoff threshold, re-baseline); one is MY ESCROW-guard regression (proof_packet builds
  ESCROW config → hits the raise) — I own the fix.
- **ESCROW guard LANDED + tested** (checkpoint #46).
- **Harness enablers** (esper-lite-7fe21bd091, decision ACCEPTED, task OPEN, no GPU): default-on checkpointing +
  per-decision advantage/pre-floor/per-head-KL logging + **K>1** (Read B, P8=0). Correct regardless of the epic.
- **EV-stabilization epic** (esper-lite-f25b71c165): reframed by PDR-0074; comment #181.
- **Branch-survivor** (esper-lite-1f1e55f58f): main-as-trunk (PDR-0062); gated on merge window — untouched.

## Facts the next session must not relitigate
- **Measurement gap (PDR-0074), fire-rate not magnitude:** fossil `bounded_attribution` collapses 88%→11% FIRE-RATE
  (magnitude barely moves). Realized FOSS−SET_ALPHA return penalty −113/−102 = ~99% forfeited cf-credit, ~0.8% accuracy,
  ~0.3% rent. Bonus:forfeit ≈ 75:1.
- **RETRACTED:** PDR-0069 "the gate passes"/"reward already correct"; PDR-0071 §4 "signal survives the 9-pt spread"; the
  round-9 "reward-specification / vindicates the epic" over-read; "ESCROW as fix." Un-retracted: PDR-0071's
  settlement/transfer co-requisite (it is the missing revenue line).
- **The premise is UNMEASURED, not refuted:** τ_main ≈ 0 once the cf phantom is stripped; G3's "no positive region" is
  bounded to THIS floored policy's seeds. Read A was PROXY-tier; the "measurement gap CAUSES avoidance" causal claim is a
  strong hypothesis (policy gradient-frozen on 94% of commits), not intervention-confirmed.
- **Floor + reward are a PACKAGE:** FOSSILIZE is floor-bound 93-94%; the ~17% rate is the FLOOR's, not the policy's.
  Fixing the gradient alone → fossilization ~0. Floor ρ-sweep CONTRAINDICATED + deferred (not rejected).
- **Critic = SYMPTOM, not an independent defect (round-11/H6):** V_main "overstates" the FOSSILIZE main-stream penalty
  4-6× because the OBSERVATION zeroes a fossil's contribution (`features.py:830`) → the critic is fit to a lying state
  vector, correctly predicting accuracy damage that never materializes. Retire the critic as a separate line; PDR-0066
  ("wrong lever") was right for a new reason. Do NOT redesign the critic. (Subject to MC/value-target parity.)
- **No reward transform fixes a missing INPUT:** ESCROW / telescoping / de-shape all take the missing `attributed` as
  input. **`RewardMode.ESCROW` is CONTRAINDICATED** (it reads the fossilize measurement-collapse as "transient" → claws
  back accrued credit) — flag it in code before a future session flips it. The only fixes SUPPLY a fossil's value (L1/L2/L3).
- **The exclusion is a JUDGMENT, not a fact:** the "ablating a fossil measures host damage" comment is unmeasured; the fix
  for a biased estimator is a bias correction, not substituting 0. H7 (drift probe: valid fossilize-instant LOO vs the
  "invalid" ablation at t+1/+5/+20) decides whether a hard measurement programme (L3) is even needed. Don't scope hard before H7.
- **Read B (K=1):** the configured target-KL early-stop supplies no operative pre-emptive guard under the K=1 loop (P8=0)
  — a competing, reward-independent structural fact; the λ=0 necessity smoke would be uninterpretable. Fix K before any
  floor smoke.
- Docs: eval `2026-07-13-round8-findings-evaluation.md`; the Read-A→round-10 arc `2026-07-14-advantage-loo-read-preregistration.md`.

## Open questions / blocked-on-owner  (Step-2 escalations — flagged, NOT enacted)
- **REFOCUS APPROVED (PDR-0076)** — enacted: roadmap Now bet moved, `vision.md` reinforced (SNR + Goodhart anti-goals).
  ESCROW guard APPROVED (fail-closed, folded into the drl deliverable). drl-expert L1/L2 design + reads DISPATCHED this session.
- **Still owner-gated (AFTER the drl deliverable + offline replay):** LANDING L1/L2 (reward/observation code change) and any
  GPU rerun. Do NOT land until the design + offline replay show the fix removes the −113 discontinuity WITHOUT farming
  fossils (guardrail).
- Owner-gated + deferred: the ρ-sweep floor primitive; the K>1 / trust-region fix (P0 harness, independent of the epic).
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: git identity tachyon-beep; no push/tag/release/branch-deletion/telemetry-deletion/remote action without an explicit ask.

## Last did (this checkpoint, #44)
- Ran the pre-registered reads (PDR-0073): Read A = INVERTED (PROXY); Read B = K=1 no-trust-region (P8=0). Round-9 G1/G2/G3
  + round-10 fire-rate decomposition + a code read (H2) RESOLVED the commitment defect to a **fossil measurement gap**.
- Banked the facts + retractions (PDR-0074, accepted); proposed the instrument-first refocus (PDR-0075, owner-gated);
  reconciled the epic (comments #181/#182). Multiple over-reads caught and retracted (advisor + primes) before banking.
- Round-11 + **H6**: confirmed the OBSERVATION also zeroes a fossil's contribution (`features.py:830`) → ONE gap, FOUR
  symptoms; the **critic RETIRES as a symptom** (not independent); ESCROW contraindicated; refocus reworded to "make
  permanence visible."
- Round-12: owner APPROVED all three (refocus / ESCROW guard / drl design) subject to the primes' roadblocks. Enacted:
  roadmap refocused, `vision.md` reinforced, PDR-0076 (binding conditions), drl-expert dispatched.
- Round-13: drl-expert deliverable DONE — L1 freeze-value+rising-uncertainty (H7: no decay-γ), L2 settle-then-annuitize
  EWMA-priced (offline farm verdict: A[ewma] VIABLE, A[spot]/B FARM), de-shape not survivable (H8), **ESCROW guard LANDED
  + tested**, L3 still required. **Primes review pack written** + record corrected (hindsight = separate scaffold channel,
  not symptom 2 → THREE symptoms not four).

## Next session, start here
1. **MINE r9 FIRST — it is a one-way door (round-13.5 §2): K>1/obs-v4 make future runs a different regime, so r9 (the whole
   13-round evidence base) becomes non-extendable once a new-regime run happens.** The load-bearing free read: **WAIT-arm
   decomposition** (split the READ-3 WAIT arm by later-force-pruned vs not) — settles commit-EARLIER-vs-more and
   protect-from-forced-PRUNE vs intrinsic-benefit (PDR-0078 OPEN). Also mine while free: per-head floor-binding rates,
   |H|=1 rate vs training round, α-target/blueprint histograms. Do these before authorizing any K>1/v4 GPU run.
2. **Review + commit the L1(obs-v4) + K>1 diff** (uncommitted, done+tested): read the actual diff (esp. `features.py`
   `_encode_contribution_v4` + the schema plumbing), confirm v3 byte-identical, then commit. **Fix the ESCROW-guard
   regression** (`scripts/proof_packet.py` builds an ESCROW config → hits the fail-closed raise; set the opt-in). Re-baseline
   the pre-existing EV-liftoff-k4 threshold separately (not this diff's regression).
3. **L2 REDESIGN around a NON-SELECTABLE settlement instant** (READ 1 verdict): fixed-time / terminal / non-cancellable
   REQUEST_FOSSILIZE confirmation window — the only construction that removes the timing max-operator. The EWMA quote is a
   valid VALUE proxy (READ 3) but must not be evaluated at a policy-chosen instant. Close the 4 L2 spec holes (gpt-prime §3).
   Owner-gated to land; L1 lands first (don't run L1 alone).
4. **Then the first GPU arm** (owner-gated, after r9 mined): SHAPED control vs L1+non-selectable-L2, floor+critic+K held
   constant for attribution. Success = the −113 discontinuity gone + contribution-sensitive commitment + NO timing farm +
   stable/better terminal outcomes. Floor is P3 (packaged with the reward, never alone).
