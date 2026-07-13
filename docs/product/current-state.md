# Current State — Esper        Checkpoint: 2026-07-14 (#44) · commitment defect RESOLVED to a fossil measurement gap; instrument-first refocus proposed (PDR-0074 accepted, PDR-0075 proposed; on `feat/ev-stab-stage2-hra`)

## The bet right now
**The commitment defect is a MEASUREMENT gap, not a policy/optimizer/floor/critic defect (PDR-0074).** Fossilized seeds
are excluded from the counterfactual ablation BY DESIGN (`vectorized_trainer.py:956-975` — ablating a baked-in fossil
measures host damage, not contribution), so a fossil's `seed_contribution` is structurally `None` → the recurring
`bounded_attribution` credit can't fire → **the reward pays a seed for its contribution ONLY while provisional.** The
policy would never commit (SET_ALPHA 55%). PROPOSED refocus (PDR-0075, owner-gated): move the epic Now bet to **causal
permanent-seed instrumentation** (measure/settle a fossil's contribution), floor-fix deferred, critic secondary.
· metric: can a genuinely-contributing seed be committed WITHOUT forfeiting its credit.

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
- **Critic:** V_main overstates the FOSSILIZE main-stream penalty 4-6× (subject to MC/value-target parity) — real,
  SECONDARY (not "the critic is broken"); qualifies PDR-0066. Hold critic + floor CONSTANT in the first reward experiment.
- **Read B (K=1):** the configured target-KL early-stop supplies no operative pre-emptive guard under the K=1 loop (P8=0)
  — a competing, reward-independent structural fact; the λ=0 necessity smoke would be uninterpretable. Fix K before any
  floor smoke.
- Docs: eval `2026-07-13-round8-findings-evaluation.md`; the Read-A→round-10 arc `2026-07-14-advantage-loo-read-preregistration.md`.

## Open questions / blocked-on-owner  (Step-2 escalations — flagged, NOT enacted)
- **Confirm the instrument-first REFOCUS (PDR-0075)?** A strategy re-scope of the epic Now bet (vision-level). You've
  signalled the direction (change+rerun offer); `vision.md`/roadmap band NOT rewritten pending your explicit go.
- **Approve the measurement change + rerun** — after the drl-expert design + a spec for your review (do NOT just flip the
  fossilized-exclusion flag: it reintroduces the host-damage artifact). Owner-gated core-loop change + GPU.
- Owner-gated + deferred: the ρ-sweep floor primitive; the trust-region fix (K≥2 / post-step KL, Read B).
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: git identity tachyon-beep; no push/tag/release/branch-deletion/telemetry-deletion/remote action without an explicit ask.

## Last did (this checkpoint, #44)
- Ran the pre-registered reads (PDR-0073): Read A = INVERTED (PROXY); Read B = K=1 no-trust-region (P8=0). Round-9 G1/G2/G3
  + round-10 fire-rate decomposition + a code read (H2) RESOLVED the commitment defect to a **fossil measurement gap**.
- Banked the facts + retractions (PDR-0074, accepted); proposed the instrument-first refocus (PDR-0075, owner-gated);
  reconciled the epic (comment #181). Multiple over-reads caught and retracted (advisor + claude-prime) before banking.

## Next session, start here
1. **drl-expert: design the valid permanent-seed contribution measure** (settle-at-fossilize vs retrain-without vs
   influence) + **H3** (`hindsight_credit` numerator/denominator/trigger — is it dead for the same measurement reason?).
2. Bring the owner a concrete, drl-reviewed **measurement change + telemetry-first spec** for approval BEFORE modifying
   training code or spending the rerun; and the explicit **refocus confirmation** (PDR-0075).
3. Then: telemetry-first change → confirm fossils retain contribution → reward fold-in → coupled floor-fix → GPU rerun
   (hard floor + critic held constant for attribution). Do NOT run the λ=0 floor smoke until K is fixed.
