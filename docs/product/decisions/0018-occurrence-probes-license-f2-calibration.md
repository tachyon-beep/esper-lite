# PDR-0018 — tau ACCEPT-PROVISIONAL; occurrence probes license F2 calibration (coverage YELLOW); large-effect A/B targets set

Date: 2026-07-03   Status: accepted (owner-ruled; ratifies the comment-#99 pre-registration)
   — §3's Δfossilize_payable_J overlay and §6's scope framing SUPERSEDED IN PART by PDR-0019
Author: Claude (agent), owner-ratified
Related: PDR-0015, PDR-0016, PDR-0017; gate esper-lite-f22a1d48a7 (comments #99–#101);
docs/analysis/2026-07-03-occurrence-probes-f1-f2-f5-s1-coverage.md;
docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md §falsifiable criteria (i)–(v)

## Context
PDR-0017 delivered tau = +0.28 pp with the PDR-0015 epsilon-plateau trigger FIRED (the noise
floor is magnitude-dependent) and put the disposition to the owner. The zero-GPU occurrence
probes (pre-registered on the gate at comment #99 BEFORE data was read, executed same day)
returned: F1 not exploited (0.17% of positive-ba mass, no farming correlation), F5 not
material (all-off config in 100% of CF matrices), S1 mild (fossilize timing near-uniform),
F2 occurring at full scale but value-aligned in aggregate (92.4% of steps / 97.8% of |ba|
mass on r0c0 vs 91–97% of value mass), coverage 66.7% in J currency / 19.4% in raw
cf-points, bootstrap floor ~1 fossilize-firing episode in 5, k≥2 in 1.1% of episodes, and
66% of all positive cf mass parked in HOLDING-never-fossilized seeds.

## The rulings (owner, 2026-07-03)

1. **tau: ACCEPT-PROVISIONAL / LOWER-BOUND ONLY.** +0.28 pp is sufficient to open F2
   calibration. It is NOT the final ON-run operating threshold. ON-run recalibration is
   MANDATORY before any n=10 magnitude claim. Option-a-lite (small K-resampled-minibatch
   calibration sample) fires only if the ON term pays frequently at non-zero magnitudes,
   the A/B lands near the decision boundary, or the tau-adjusted effect depends on the
   exact threshold.
2. **Coverage currency: J (param-normalized).** 66.7% → **YELLOW**: A/B licensed only with
   a large effect-size target. Raw cf-point coverage (19.4%) is an interpretation caveat,
   not the gate: the A/B tests whether small-param/high-efficiency fossilized seeds can be
   better credited — most raw positive contribution mass is out of this experiment's reach.
3. **A/B effect-size targets (the YELLOW-band overlay).** Primary gates remain the design
   doc's pre-registered falsifiable criteria (i)–(v); these thresholds layer on top, and
   any conflict at scoring time surfaces to the owner rather than being resolved silently.
   - **n=5 direction gate:** median paired Δ`fossilize_payable_J` ≥ **2 × tau_provisional**
     (= +0.56 pp at the current tau; written as the formula so ON-recalibration updates it);
     sign consistency ≥ 4/5 seeds; Δcorr(reward, J) ≥ **+0.10 absolute, measured at
     EPISODE level** (paired, same-seed: corr of episode-summed reward vs episode J — the
     level at which the banked 0.212 baseline reproduces; at step level the F2 sampling
     ceiling makes +0.10 structurally unfair for a ~0.2-firings/episode term). Safety:
     final-accuracy median paired Δ ≥ −0.3 pp; system efficiency not degraded >10%;
     committed params/compute not materially up unless accuracy/J justifies; fossilize
     opportunity not further suppressed.
   - **n=10 magnitude/bank gate:** point estimate ≥ 2 × tau_ON with paired-bootstrap lower
     bound > 0 (stronger bank: lower bound ≥ tau_ON); Δcorr lower bound > 0, preferred
     point estimate ≥ +0.10.
4. **No F1/F5 remediation before the A/B.** F1: not exploited; re-probe on Phase −1 arm /
   ON-run telemetry when it exists, BEFORE any clip-arm verdict is read. F5: latent bug,
   parked on the audit register (esper-lite-a7ef375203) for post-A/B hardening. S1:
   monitor only. Rationale: fixing unexpressed mechanisms now would mutate the experiment
   ("term + fossilization-path fixes") without addressing the binding question.
5. **F2: calibrate, do not correct.** Proceed to scale/cap/normalized_cap/std_floor
   calibration (gate criterion 2). NO slot-specific r0c0 correction before the A/B — the
   r0c0 concentration is aggregate value-aligned, and the n=5 suppress read (PDR-0009,
   5/5 seeds) banked r0c0 as an efficiency-enabling stem; whether the reward bias *created*
   that concentration is a paired-run question, deliberately left open.
6. **A/B scope, named to prevent overclaim.** This is a **terminal commitment-credit**
   experiment: the term fires in ~1 in 5 episodes and its synergy component (k≥2) is
   sampled in ~1.1% of episodes. It will NOT cleanly test synergy credit, developmental
   r0c0 enabling, the HOLDING never-fossilize reservoir, or raw counterfactual mass.
7. **Never-fossilize reservoir: follow-on, not precondition.** 66% of measured positive
   value parks in HOLDING and never commits. If the A/B underperforms, the FIRST
   explanation to test is "term semantically right but too rarely payable" → a
   pre-fossil/HOLDING-stage credit unlocker (adjacent to the design doc's pre-identified
   salvage candidate B, developmental/temporal credit).
8. **Metric naming:** the A/B primary metric is **`fossilize_payable_J`** (the repaired,
   retro-written committed-Shapley top-up quantity). Bare "committed-J" is banned in A/B
   docs unless explicitly disambiguated from the RETIRED Phase-0 committed-J bucket
   (structurally zero for any policy; fossils excluded from LOO ablation).

## Options considered
Re-scope the deadband method now (strongest response to the plateau trigger, largest
delay) and full option-(a) supplementation (measures val-resampling noise at fixed weights;
equally magnitude-blind) were both declined in favor of proceed-with-mandatory-ON-
recalibration — the direction is conservative-compatible (smaller placebo ⇒ smaller floor)
and the caveat was pre-recorded in the Phase-0 packet. Overruling coverage to BLOCKER on
the raw-cf reading was declined because the north-star and A/B metric are param-normalized.

## Reversal triggers
- PDR-0017's trigger stands: first ON calibration run measuring real-magnitude null-player
  excess with P99 > 2× placebo tau reopens the deadband method (option-a supplementation
  or magnitude-scaled tau(‖seed‖)) before the A/B is read.
- Design-doc criterion (v) failing reopens GATE 2 (PDR-0011 trigger, unchanged).
- If the n=5 direction gate fails on the Δ`fossilize_payable_J` primary while safety holds,
  ruling §7's follow-on (pre-fossil credit) is evaluated BEFORE any scale increase.
