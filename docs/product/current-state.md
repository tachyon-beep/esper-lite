# Current State — Esper        Checkpoint: 2026-07-14 (#44) · commitment defect RESOLVED to a fossil measurement gap; instrument-first refocus proposed (PDR-0074 accepted, PDR-0075 proposed; on `feat/ev-stab-stage2-hra`)

## The bet right now
**The commitment defect is a permanent-value MEASUREMENT gap — ONE gap, FOUR symptoms (PDR-0074 + round-11/H6).** The
counterfactual is undefined at both ends of a seed's life (α=0 birth; permanence — fossils excluded from ablation by
design, `vectorized_trainer.py:956-975`), and every consumer of a `None` inherits it: (1) reward `bounded_attribution`
stops firing → −113; (2) settlement `hindsight_credit` inert [H3, predicted]; (3) OBSERVATION contribution feature = 0
(`features.py:830` `None→0`, H6-CONFIRMED); (4) the critic "overstates" the commit penalty because it is fit to that
lying observation. So the reward AND the observation AND the value function all value a fossil at exactly 0 — while the
freshness channel that could say "STALE not zero" (`γ^epochs_since_cf`) sits built-but-bypassed. PROPOSED refocus
(PDR-0075, owner-gated): **MAKE PERMANENCE VISIBLE** (L1 freeze-don't-zero + L2 settle-at-fossilize, both on existing
data; H7 drift-probe before any hard measurement programme) — the premise "under-fossilization is a defect" has never
been tested with a working instrument. · metric: can a genuinely-contributing seed be committed WITHOUT its measured
value going to 0.

## In flight
- **Instrument-first fix design (next):** drl-expert to design the valid permanent-seed contribution measure — likely
  **settle-at-fossilize** (carry the last valid HOLDING LOO forward), NOT fossil-ablation (host-damage artifact) and NOT
  ESCROW (would claw back on the measurement-zero). Then telemetry-first, then a coupled floor-fix + rerun. Owner offered
  the measurement change + rerun; drl-design + a spec for owner review come first.
- **Harness enablers** (esper-lite-7fe21bd091, decision ACCEPTED, task OPEN, no GPU): default-on checkpointing +
  per-decision advantage/pre-floor/per-head-KL logging. Still the enabler for any instrumented run.
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
- **Confirm the "MAKE PERMANENCE VISIBLE" REFOCUS (PDR-0075, reworded)?** A strategy re-scope of the epic Now bet
  (vision-level). You've signalled the direction (change+rerun offer); `vision.md`/roadmap band NOT rewritten pending your
  explicit go. Note the reword: L1/L2 on existing data + H7 before any hard L3 — NOT a committed hard-measurement programme.
- **Approve L1 (freeze-don't-zero, `features.py:830`) + L2 (settle-at-fossilize)** — after the drl-expert design + a spec
  for your review. Do NOT just flip the fossilized-exclusion flag (host-damage artifact); L1 is restoring intended
  semantics (the freshness channel already exists). Owner-gated code change (+ eventual rerun).
- **Small safety flag (approve?):** mark `RewardMode.ESCROW` CONTRAINDICATED in code (a comment/guard) so a future
  session doesn't flip it — it would claw back accrued credit on the fossilize measurement-collapse.
- Owner-gated + deferred: the ρ-sweep floor primitive; the trust-region fix (K≥2 / post-step KL, Read B).
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: git identity tachyon-beep; no push/tag/release/branch-deletion/telemetry-deletion/remote action without an explicit ask.

## Last did (this checkpoint, #44)
- Ran the pre-registered reads (PDR-0073): Read A = INVERTED (PROXY); Read B = K=1 no-trust-region (P8=0). Round-9 G1/G2/G3
  + round-10 fire-rate decomposition + a code read (H2) RESOLVED the commitment defect to a **fossil measurement gap**.
- Banked the facts + retractions (PDR-0074, accepted); proposed the instrument-first refocus (PDR-0075, owner-gated);
  reconciled the epic (comments #181/#182). Multiple over-reads caught and retracted (advisor + primes) before banking.
- Round-11 + **H6**: confirmed the OBSERVATION also zeroes a fossil's contribution (`features.py:830`) → ONE gap, FOUR
  symptoms; the **critic RETIRES as a symptom** (not independent); the fix is **L1/L2 on existing data** (H7 decides if a
  hard L3 is needed); ESCROW contraindicated; refocus reworded to "make permanence visible." (Checkpoint #44 addendum.)

## Next session, start here
1. **Zero-GPU reads that decide the epic's cost/shape:** H7 (ablation-drift probe — is a hard L3 measurement programme
   even needed?), H3 (`hindsight_credit` — dead for the same measurement reason?), H8 (shaping as % of episode return —
   is de-shape survivable / is the product objective trainable?). H6 DONE (observation zeroes a fossil's contribution).
2. **drl-expert: design L1 (freeze-don't-zero, one line at `features.py:830`) + L2 (settle-at-fossilize) — both on
   EXISTING data** + a telemetry-first spec. Bring the owner the reviewed change + the explicit **refocus confirmation**
   (PDR-0075, reworded to "make permanence visible") BEFORE modifying code or spending a rerun.
3. **P0 harness regardless:** K>1 (restore the trust-region guard — Read B), default-on checkpointing, per-decision
   telemetry. Then P1 (L1+L2) → P2 re-ask "is commitment good?" with a working instrument (never done) → P3 floor
   (packaged with the reward, never alone). Do NOT run the λ=0 floor smoke until K is fixed.
