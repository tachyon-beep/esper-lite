# PDR-0068 — cf is credit-or-noise (SNR) BEFORE the ablation; governance schema + framing fixes

Date: 2026-07-13
Status: accepted for the corrections + governance (within grant); the experiment ORDERING and
the epic framing are PROPOSED — owner-gated. Synthesizes GPT-set-2 + claudemain (out-of-frame)
reviews. Refines PDR-0066/0067.

## Scientific corrections banked (this round)

- **"light-A′ refuted" STANDS** — on an INFORMATION argument (two frozen scalars carry
  `E[G_main|s]`,`E[G_cf|s]`; no function of them adds info), stronger than the retracted +0.04
  empirical claim. The prior blanket retraction was a sign-flipped over-correction.
- **"anti-shaping" RETRACTED** — `ev_cf ≈ 0.57` is what a WORKING credit signal looks like
  (a state-value critic predicts pre-action `E[G|s]`; action-dependent variance SHOULD be
  unexplained). Low `ev_cf` is not evidence against cf.
- **heavy-A′-ceiling-from-OFF is PROVISIONAL** — assumption (a) OFF==ON reward CONFIRMED
  (`partition.py`); (b) state-distribution and (c) aux-head representation UNCHECKED (aux
  heads are a real representation-learning lever). "near-perfect main" qualified (scale-dep).
- **"reward-intrinsic" → "current-model residual"**; Objective A stays rejected.

## Reframed ordering (owner-gated) — SNR before the ablation

**(1) The kill-criterion question is cheaper than the ablation: is cf's unexplained variance
CREDIT or NOISE?** `SNR(cf) = Var_a(E[G_cf|s,a]) / E[Var(G_cf|s,a)]` — between-action credit
vs within-(s,a) noise. Credit dominates ⇒ the critic is fine, EV-stabilization was mis-specified
at PDR-0001, KEEP cf and close the epic. Noise dominates ⇒ cf pumps gradient variance ⇒
delete/re-estimate. The critic's EV appears nowhere in SNR — the tightest statement of why the
A/A′/B critic tree could never answer this. Needs per-sample `(s,a,G_cf)`: the rollout buffer
computes it but does NOT dump it → **one short logging run** (add a per-sample dump/emitter),
then a variance decomposition. Report action entropy alongside (on-policy under-counts
between-action variance under a narrow action distribution; instrument exists from the
thermometer work).

**(0) First, governance archaeology (no GPU):** find the decision that INTRODUCED
`bounded_attribution` and state what it was supposed to buy and how measured. Initial search:
it is the dense counterfactual credit-assignment / anti-freeloading steering term; NO clean
"measured how" benefit statement was found — provisionally the governance gap below. The
ablation must measure the thing cf was introduced to deliver, or a null proves nothing.

**(2) Then the sized cf-off ablation (PDR-0067 design), corrected:** NOT a small screen — cf
carries ~500 of ~590 return variance, so ablation trains on a near-flat signal (legitimate
null model; not coverage-bound like PDR-0026). Direct-`V_total` critic both arms; downstream
accuracy/efficiency-frontier primary (NOT EV/J). heavy-A′ deferred until cf is shown load-bearing.

## Framing (owner question 1): a SCREEN, not the epic's new Now bet

Do not promote an experiment to headline status when its purpose is to determine whether the
epic should exist — that is how frame-lock happens (what the pre-registration did the first
time). Title: "dense counterfactual reward earn-its-keep screen"; it gates whether further
critic work is worth doing, without becoming a multi-phase programme.

## Metric (owner question 2): don't pick one up front — measure what cf was FOR

Primary = the system-level accuracy–efficiency frontier (capability `C = acc_all-on −
acc_all-off`, under a parameter/compute non-inferiority rule), NOT committed-J (cf-derived →
circular; misses synergy per Phase-0) and NOT EV or a hand-built controller-quality composite.
Secondary (explain the answer): compact-motif frequency, destructive-intervention frequency,
credit diagnostics, critic diagnostics. But the endpoint MUST include the thing cf was
introduced to deliver (from step 0).

## Governance changes adopted

- **New REQUIRED PDR field** (`decisions/PDR-TEMPLATE.md`): *"What does this buy, and how is
  that measured?"* — any decision introducing an optimization target / reward term / metric
  epic must state the downstream product outcome and its measurement, or declare that
  establishing it IS the experiment. Makes the fractal "no empirical baseline" finding
  uninstantiable.
- **Standing rule:** before proposing a GPU run, state what ALREADY-COLLECTED data bears on
  the question and why it is insufficient. (The heavy-A′ pilot's answer was on disk in the OFF
  run.)
- **Out-of-frame reviewer:** keep a reviewer who has NOT been in the session — same-frame
  review (the in-session advisor) catches errors but stays inside the epic and over-read too;
  out-of-frame review catches mis-framings (it caught the "does it cost anything" gap).
- **Gate, not memory** (PDR-0067 G1) — memory notes fire on recall; the lesson was written
  after over-reads #1/#2 and #3 happened anyway. Watch whether the gate holds on an exciting
  result. **Pre-registration without a validity derivation makes things worse** (it lent a
  broken instrument the authority of a commitment — nearly shipped "B favoured").

## Reversal trigger
If SNR(cf) shows between-action credit dominates, cf is a working credit signal → close the
critic epic, keep cf. If within-(s,a) noise dominates, proceed to the sized cf-off ablation to
quantify the downstream cost before deleting/re-estimating.
