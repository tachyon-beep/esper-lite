# Current State — Esper        Checkpoint: 2026-07-14 (#43) · round-8 evaluation calibrates the floor reframe + read-before-build (PDR-0072/0073; on `feat/ev-stab-stage2-hra`)

## The bet right now
**Fix the commitment defect at its root — the floor gradient dead-zone — via the exploration primitive
(PDR-0069/0071; CALIBRATED by the round-8 evaluation `docs/analysis/2026-07-13-round8-findings-evaluation.md`,
PDR-0072).** MECHANISM ESTABLISHED (autograd-forced; dual-expert numeric-confirmed on the shipping symbols):
the **op-policy loss cannot DIRECTLY update the op logits on these decisions** — the PPO log-prob is locally
independent of the underweight raw logits, and at |H|=1 the entire post-floor op distribution is locally
constant → zero gradient to ALL op logits (bit-exact). Shared-trunk + other-state gradients still move them
(NOT "permanently frozen"). Leading PROXIMAL mechanism for flat commitment. **CAUSAL SUFFICIENCY is
HYPOTHESIZED, NOT established** — the live alternative ("SET_ALPHA is the new WAIT; the policy does not want
to commit and the floor merely masks it") is not yet ruled out. The only unproven bridge is **advantage-vs-LOO**,
which is UNMEASURED and directionally ambiguous — **now under test (Read A, in flight).** Reframe status:
sound as a REPRIORITIZATION, overstated as a supersession — hold to PDR-0071's calibration.

## In flight
- **Read A — the RCT (§5, PDR-0073):** offline advantage-vs-LOO reconstruction from existing `value_estimate`
  telemetry; pre-registered GRADED/FLAT/**INVERTED** rule in `docs/analysis/2026-07-14-advantage-loo-read-preregistration.md`.
  Zero-GPU, launched this session (drl-expert). **The premise test** — INVERTED would collapse the epic's founding
  assumption.
- **Harness enablers** (esper-lite-7fe21bd091, decision ACCEPTED, task OPEN, no GPU): default-on checkpointing +
  per-decision advantage/pre-floor logits/per-head KL. Lands before any instrumented run.
- **EV-stabilization epic** (esper-lite-f25b71c165): commitment-premise reframed; critic (PDR-0064/0065) and
  cf-earns-keep/SNR (PDR-0067/0068) demoted to secondary.
- **Branch-survivor** (esper-lite-1f1e55f58f): main-as-trunk decided (PDR-0062); gated on the merge window — untouched.

## Facts the next session must not relitigate
- **Dead-zone mechanism (PDR-0069/0071/0072):** floor-bound ⇒ op-policy loss can't directly update those op logits;
  |H|=1 ⇒ whole op head constant (bit-exact zero), corroborated at **63% (s41)/72% (s42)** of HOLDING decisions
  (telemetry; reconstructed 67/80 was slightly high). SET_ALPHA 0.55 = analytic cap, not learned. Policy-wide (all 8
  heads, differentiable update path). **Single-pass floor CONFIRMED** — it does NOT strictly enforce its floor/cap at
  the margin (so an "entropy ≥ H_min or STOP" gate would false-fire; do not use it).
- **`hindsight_credit` is OFF the shipping gate (PDR-0072, supersedes the PDR-0071 co-requisite framing):** it is keyed
  on the non-causal `total_improvement`, and its 0.005% is a rare-trigger artifact (functional, not broken). Restate the
  co-requisite in ADVANTAGE terms (per-decision FOSSILIZE advantage graded by transferred value, not current LOO).
- **Experiment redesign (PDR-0073, supersedes the PDR-0071 global λ-sweep):** change ONLY the op head first; ρ∈{0,0.5,1}
  endpoint-matched per-num_valid (`f+(1−nf)p_i`); fresh hard-floor control arm. A global λ=0.6 is confounded (moves both
  axes on 7 of 8 heads, N5). Score on mechanism metrics, NOT "floor-bound rate→0" (a tautology).
- **Gate caveat:** −0.7→+8.2 is IMMEDIATE reward; advantage-vs-LOO is directionally ambiguous ("V(s) op-independent ⇒
  advantage rises" is a non-sequitur — V(s) absorbs the LOO-driven part of the return). **Scope:** the floor does NOT
  erase the prior HRA fit-failure or coverage-bound Shapley null.
- **Round-8 NEW findings (verified, in the eval doc):** N1 wide-head expressiveness clamp — mechanism real but severity
  REVISED DOWN (realized num_valid=7, ceiling ~2× uniform, realized 1.3-1.4× → mild, NOT a new epic item); N2 `approx_kl`
  dilution → optimizer step size — **Read B DONE, reframed:** the trust region IS vacuous (P8=0.000%, both seeds) but the
  cause is **K=1** (`recurrent_n_epochs=1` → epoch-0 KL = KL(θ‖θ) = 0 by construction; ratio≡1.0, ratio-clip also inert),
  NOT head-dilution; N2's dilution claim is untestable on K=1 (real only at K≥2). N3 entropy/collapse monitoring STRUCTURALLY
  BLIND to the dead-zone (a TIP scar); N4 floor-bound samples contaminate global advantage standardization.
- Docs: eval `2026-07-13-round8-findings-evaluation.md`; pre-reg `2026-07-14-advantage-loo-read-preregistration.md`;
  round-1..7 `2026-07-13-decision-point-diagnosis.md`. Memory `floor-gradient-dead-zone`.

## Open questions / blocked-on-owner  (Step-2 escalations — flagged, NOT enacted)
- **Owner-gated core-training-loop changes (drafted/surfaced, awaiting sign-off):** the op-isolated **ρ-sweep primitive**
  (action dist / PPO ratio; needs drl-expert review); and a **trust-region fix** — Read B showed the K=1 runs have NO
  trust region (KL early-stop AND ratio-clip both inert), so the real levers are **K≥2 epochs OR post-step KL gating**,
  NOT the aggregation fix (moot at K=1). Neither built.
- **AUTHORIZE the ρ-sweep GPU arm** — only after Read A returns GRADED (VALID tier) and Read B clears sequencing. GPU +
  owner acceptance; pre-registered gates owner-gated.
- **Potential vision-level escalation:** if Read A returns **INVERTED**, the epic's founding premise ("under-fossilization
  is a defect") is the over-read → a strategy re-examination, owner-gated.
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: git identity tachyon-beep; no push/tag/release/branch-deletion/telemetry-deletion/remote action without an explicit ask.

## Last did (this checkpoint, #43)
- Evaluated the round-7 findings on owner request: independent drl-expert + pytorch-expert (both numeric on the shipping
  symbols) + telemetry recompute (~1.08M decisions/seed) + two relayed prime reviews. Verdict: mechanism SOUND,
  reframe = reprioritization-not-supersession; localized the one overclaim (claim 5).
- Fixed the causal-sufficiency DRIFT across current-state/metrics/roadmap + a preserve-and-annotate banner on PDR-0069
  (owner-directed wording). Confirmed the floor is single-pass; took `hindsight_credit` off the shipping gate; revised
  N1 down empirically. PDR-0072 (calibration), PDR-0073 (read-before-build).
- Launched two pre-registered zero-GPU reads. **Read B (KL gate) LANDED: P8 = 0.000% both seeds — the trust region is
  vacuous, but via K=1 (`recurrent_n_epochs=1` → self-vs-self KL = 0, ratio-clip also inert), NOT head-dilution. A
  competing, reward-independent structural fact for the owner DECIDE; the aggregation fix is misdirected at K=1.** Read A
  (advantage RCT) still in flight.

## Next session, start here
**Fold in Read A + Read B when they land** (they were in flight at checkpoint). Read A's classification is the pivot:
GRADED (VALID) → license the ρ-sweep design for owner GPU authorization; FLAT → reopen the reward/critic line (N2 first);
INVERTED → escalate a premise re-examination (owner-gated). Read B is DONE (P8=0.000%, K=1 no-trust-region) — a
trust-region decision (K≥2 or post-step KL gating, NOT the aggregation fix) joins the DECIDE. Then bring the owner the
DECIDE: ρ-sweep primitive build + trust-region fix (both owner-gated) + GPU authorization. Remaining
zero-GPU reads if useful: hindsight numerator/denominator/eligibility; adaptive-LR/ent-coef coupling of `approx_kl`. Do
NOT start a reward/critic arm unless Read A is FLAT.
