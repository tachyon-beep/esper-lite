# Current State — Esper        Checkpoint: 2026-07-14 (#49) · make-permanence-visible: mechanism RESOLVED → COUPLED floor-forced-PRUNE-repair + non-selectable-settlement; premise causally supported; L1+K>1 landed (PDR-0074→0079; on `feat/ev-stab-stage2-hra`)

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
  proxy-free product value the reward zeroes out. Bank ONLY the positive TOTAL effect (not "under-fossilization is a
  defect full stop"; comparator is inaction not SET_ALPHA).
- **READ 4 (WAIT-arm decomposition) split the effect ~50/50 (residual FOSSILIZE + forced-PRUNE shelter) — but ROUND-14
  RECALIBRATED it (PDR-0080).** The **residual/permanence half is SOLID** (Support-1: val@1, assumption-light, design-based
  — the READ-3 exact randomization where forced-PRUNE can't yet have fired). The **shelter half is UNDER VERIFICATION** —
  its flat-0.15 IPCW propensity is likely mis-specified (PRUNE masked below age-5; "this seed pruned" needs the SLOT head
  too; 0.85^13≈12% ≫ the reported 55%). So the settlement is the WELL-EVIDENCED leg; the floor-repair is
  strongly-motivated-but-softer. **The coupling rationale is "EITHER-ALONE-IS-DEGENERATE"** (settlement-alone: commits stay
  floor-forced + seeds destroyed; floor-repair-alone: the learned preference is PARK → zero fossils) — NOT the 50/50. The
  floor is on the critical path regardless.
- **READ 5 (verification) RESOLVED it (PDR-0080 outcome): the shelter half is RETRACTED** (3 ways: IPCW mis-specified in
  degree — predicts 79-82% survival vs observed 45%; in kind — the SLOT head is DETERMINISTIC not a known 0.15, and
  pruning is quality-SELECTIVE, removing LOWER-LOO seeds; under the clean estimand shelter→16-20%, residual→80-84%).
  **The intrinsic-permanence / SETTLEMENT half is the identified, robust bet.** The floor's forced-PRUNE is NOT an
  indiscriminate destroyer — target selection is sensible learned quality-pruning; only the timing is forced. So the
  "floor co-required via shelter" claim is WITHDRAWN; arm C's floor intervention RE-SCOPED to the gradient-dead-zone fix
  (round-8 per-num_valid differentiable floor), NOT forced-PRUNE-removal. Coupling rests ONLY on "either-alone-degenerate."
- **The PRE-REGISTRATION is DONE (`docs/analysis/2026-07-14-permanence-visible-preregistration.md`) — the GPU-decision
  artifact.** Triangular attributable design (A control / B settlement-only / C = gradient-restoring differentiable floor),
  RECOMMENDED STAGED (Exp-1 A/B settlement, Exp-2 add C); verdict rests only on the shelter-independent B−A / C−A. Full
  non-selectable-settlement spec (annuity, quote from pre-request measurements only, one-shot-bonus netting to kill the
  spot channel, all holes closed) + the re-scoped gradient-floor spec (per-num_valid differentiable floor, op-head-only,
  keeps the PRUNE minimum). Metrics P1(terminal acc)/P2(quote-graded commitment) + guardrails G-FARM/G-QUALPRUNE/G-STAB.
  **Sequencing DECIDED = STAGED (owner).** The "3 remaining reads" are ALREADY SATISFIED by READ 5 (PRUNE-legality 100%,
  hazard 0.109/0.127, val@1 ~90% pre-mediator, G-QUALPRUNE baseline FOSS +6.9/+6.6 vs PRUNE +6.1/+5.9) → frozen into the
  guardrail-baseline table; **no reads outstanding.** **The ONLY remaining pre-freeze gate is owner sign-off on the other
  flags** (HRA posture, P1-vs-P2 primary, τ_acc + MDEs + guardrail materiality, n/seeds/budget/device; settlement:
  re-interpret-FOSSILIZE vs new-op, boundary policy, EWMA-vs-min-window, flat-premium; floor: form + f + op-head-only).
  No-peek freeze; GPU held. **FREEZE ON HOLD (rounds 15–19, PDR-0081/0082/0083).** Premium SETTLED: **1.464538 × legitimacy_request**
  (matched-control consolidation of BOTH shipping flat terms — the "tanh bonus" is a CONSTANT 3·tanh(1/3)≈0.9645, not c-graded), paid
  at the settlement BOUNDARY, discount-neutral (÷γ^d, unit-tested), qualified on q_settle≥1.0, never retuned. Rounds 18–19 REJECTED
  the audit's own q_decision re-key (available at the request instant → invariant violation; also would bake quote→request into P2
  via the mask) → request-instant sterilization: NO contribution-conditioned payment of any sign at request; only −0.01 cost;
  legitimacy frozen. F5 PBRS reclassified (should telescope; truncation residual γ^150≈0.47×Φ_FOSS=6.0 possibly PRO-fossil; PENDING
  potential to define; never a numeric refund). holding_warning window read ANSWERED (fires only when targeting the pending slot →
  suspend for committed=True). Gates: GATE-DOM (per-run paired slope of pre-floor p_FOSS on q_decision) + GATE-INFL, one-sided-in-B,
  verdict-changing. Pre-reg BODY RECONCILED to Groups-A–D altitude (no stale clauses). Freeze blocked on THREE: **executable G-LEDGER
  replay** (9-dim case matrix, per-component PV, 2 invariants @1e-6, event-order assertion — the new load-bearing blocker);
  paired-run power calc (P2+both gates, blinded reassessment, frozen max-n); one final adversarial pass. No GPU.
- **Archival r9 reads DONE + r9 hash-SEALED** (`docs/analysis/2026-07-14-r9-archival-record.md`). The one-way-door gate is
  satisfied. Findings: the floor pathology is THREE distinct things — OP floor-binding (~43%, and its pinned rate GROWS
  over training → round-11 confidence-trap CONFIRMED), `alpha_speed`+`alpha_curve` fully COLLAPSED (dead heads), blueprint
  NEAR-UNIFORM (weak preference). Enum ranges expressive, selection under-differentiated (N1 partially refuted).
- **L1 (obs-v4) + K>1 LANDED as two reviewed commits** (`b5839021` L1 default-OFF v3-byte-identical; `7715a62e` K=4
  trust-region restore) + the ESCROW-guard proof_packet regression fixed (`b35dec21`, opt-in). All verified green (L1 9 +
  features 55 + config 45 + k-test 4). Obs-v4 = default-OFF (+9 state_dim, re-warm/retrain when enabled); K=4 restores the
  trust-region guard. **LOG (do not silently re-baseline):** the pre-existing `test_ev_liftoff_k4` threshold drifted
  (2.9× vs 3.0×, proven not-this-diff via a clean worktree) — a finding with an unread cause.
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
1. **Fold in the ARCHIVAL r9 reads** (running: per-head floor-binding, |H|=1-over-time confidence-trap, α/blueprint
   histograms, data-provenance hashes) — the last free evidence; record them, then r9 is frozen.
2. **Design the COUPLED bet (PDR-0079) — neither ships alone:** (a) **non-selectable settlement** (fixed-time / terminal /
   non-cancellable REQUEST_FOSSILIZE confirmation window; EWMA a valid QUOTE not cashable; close the 4 L2 spec holes,
   gpt-prime §3); (b) **floor forced-PRUNE repair** — the floor destroys 55% of good held seeds within ~13 epochs (a
   distinct workstream from the ρ-sweep gradient fix). drl-expert design; owner-gated to land.
3. **First GPU arm (owner-gated, after r9 mined + the coupled design):** SHAPED control vs L1(obs-v4)+non-selectable-L2
   (+/− the forced-PRUNE repair, to attribute the two halves), critic + K held constant. Success = −113 discontinuity gone
   + contribution-sensitive commitment + NO timing farm + stable/better terminal outcomes + reduced forced-destruction of
   good seeds. Do NOT run L1 alone (repairs obs while the reward still forfeits).
4. **LOG, don't paper over:** the `test_ev_liftoff_k4` threshold drift (unread cause); freeze r9's schema in the read
   tooling (assert obs-v3/K=1) so archival reads can't be silently re-run on the new regime.
