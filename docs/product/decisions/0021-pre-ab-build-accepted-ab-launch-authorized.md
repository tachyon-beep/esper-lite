# PDR-0021 — pre-A/B build ACCEPTED (triple-verified); paired n=5 OFF/ON A/B launch AUTHORIZED by owner

Date: 2026-07-03   Status: accepted (build acceptance within grant; the A/B launch —
including the ON arm's shapley_synergy_scale=1.0 as an experiment run — is
OWNER-AUTHORIZED this date: "make the checkpoint and you can roll straight into the
test")   Author: Claude (agent)
Related: PDR-0019, PDR-0020; esper-lite-3f3e9b9b4f (closed); gate esper-lite-f22a1d48a7
comment #107; commits bb8d3c90, 459d01cf;
docs/analysis/2026-07-03-shapley-ab-scoring-preregistration.md;
docs/plans/completed/2026-07-03-pre-ab-build.md

## Context
The build between the F2 freeze and the A/B: payload provenance extension
(running_std_raw, bound flags, raw 2^k v(S) table via coalition_accs plumbing), the G1
FossilizeRateGuard wired behind the scale>0 gate (emit-then-raise; flush via the
shutdown finally), normalized_cap ≤ REWARD_NORMALIZER_CLIP validation in both configs,
the validated ON-arm config, and the scoring pre-registration (criteria i–v
operationalized, Δcorr floor, G2/G3/G4 rules, asymmetric null, G1-abort pairing rule,
std<1.0 confound gate, ON tau recalibration).

## The call — acceptance evidence
Verification chain: (1) drl-expert plan review (verdict ADJUST; all adjustments
incorporated, including the G4 review-trigger correction and never-a-second-event);
(2) TDD build, full suite green (5,126 passed; both pre-existing tracked reds cleared —
esper-lite-obs-52b8d18afc, esper-lite-obs-7240279be3, their deferral reason having
lapsed); (3) 4-lens adversarial review with double refutation (28 agents): the
OFF-arm bitwise-identity attacker lens produced NOTHING that survived; the single
surviving MINOR (unpinned ON-config knob values — a pre-registration-integrity
tripwire gap, not a code defect) fixed same cycle via pin tests
(tests/scripts/test_shapley_on_config.py: frozen values, arms-differ-only-in-treatment,
OFF-untreated).

## Launch terms (owner-authorized)
Paired seeds 41–45, BOTH arms at one commit (PDR-0015 same-commit constraint — the
existing causal_r1_n5 control runs are at an older commit and are NOT the OFF arm; the
OFF arm re-runs). OFF = config-3slot-3seed-baseline-shaped.json; ON =
config-3slot-3seed-baseline-shaped-shapley-on.json (pinned). Telemetry →
telemetry/shapley_ab_n5/. Scoring per PDR-0019 + the pre-registration doc; launch
commit hash recorded there at launch. Gate criteria 4–6 (entrenchment monitor,
dormancy recheck, first-ON-run residuals incl. MANDATORY tau recalibration) execute
during/after the ON runs.

## Reversal trigger
Inherited, not new: G1 abort (drop pair, investigate); G2/G3 at scoring; G4 forces
adjudication; asymmetric null (a null n=5 fires nothing); PDR-0017 tau trigger.
Acceptance itself reverses only if the adversarial-review verdict is later shown wrong
on the OFF-arm invariant — i.e., any scale=0 behavioral diff traced to bb8d3c90 —
which invalidates the A/B pairing and stops the runs.
