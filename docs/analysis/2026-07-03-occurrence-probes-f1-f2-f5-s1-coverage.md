# Zero-GPU occurrence probes — F1 / F2 / F5 / S1 + A/B coverage (2026-07-03)

**Pre-registration:** gate `esper-lite-f22a1d48a7` comment #99 (2026-07-03T09:02Z), recorded
BEFORE any probe data was read. Thresholds: coverage <50% = BLOCKER, 50–75% = YELLOW,
>75% = LICENSED; bootstrap-opportunity floor reported alongside.
**Data:** the 5 control runs `telemetry/causal_r1_n5/control_s41..45` (n=5, seeds 41–45,
200 episodes × 12 envs × 150 epochs each; 1,799,955 reward steps; 2,475 fossilize events;
148,479 residency rows; 249,158 CF matrices). Read-only; scripted DuckDB
(`occurrence_probes.py` / `probes_fix.py`, session scratchpad).
**Sanity:** fossilized `seed_id` → residency join **2475/2475 (100%)**; fossilize/ep
0.191–0.236 matches banked 0.207±0.006; per-episode max epoch = 150 everywhere.
**Method errata (disclosed):** the first S1 run mixed lifecycle-event `epoch` semantics with
reward-row semantics (fractions >1) and the first F5 regex demanded a space DuckDB's compact
JSON does not emit (returning a spurious 0%). Both probes were re-run corrected; only the
corrected numbers below are banked. The occurrence claims are otherwise as pre-specified.

---

## Verdicts (audit finding → occurrence)

| Finding | Mechanism (audit) | Occurrence (this probe) | Verdict |
|---|---|---|---|
| **F1** prune-flip pays harm as reward, clip-exempt | HIGH, confirmed | PRUNE carries **0.17%** of all positive-ba mass (973.3 of 556.6k); only 1,055/139,310 successful PRUNEs have ba>0; per-episode corr(prune count, prune pos-ba) = −0.057…+0.013 across all 5 seeds (no farming signature) | **NOT EXPLOITED — tail-only** (same class as the 48× ransomware events) |
| **F2** reward keys on targeted slot; WAIT→r0c0 | HIGH, confirmed | **92.4%** of steps target r0c0 (1.47M of them WAIT), carrying **97.8%** of \|ba\| mass. BUT the value mass is also r0c0-concentrated: **91.0%** of positive cf-integral mass and **96.8%** of positive J (per-param) mass sit in r0c0-germinated seeds | **OCCURRING AT FULL SCALE — but value-aligned in aggregate**; causality (did the bias *create* the concentration?) is NOT disambiguated by occurrence data |
| **F5** min-LOO fallback for missing all-off | MED, confirmed-in-code | **100%** of CF matrices at every arity include the all-off config (k=1: 3 configs; k=2: full 2²=4; k=3: full 2³=8). All 988 multi-slot env-episodes have all-off matrices. Fallback can fire only in refresh-cadence gaps (~21 matrices/env-episode → small) | **NOT MATERIAL** on these runs — downgrade to latent-bug |
| **S1** late-episode fossilize dominated | Plausible sink | Fossilize position near-uniform: p50 = 0.513, deciles 1–9 hold 202–356 events each; final decile 221 (8.9% vs ~11% uniform) | **MILD at most** — slight late shortfall, no collapse |

Two corollary observations (not pre-registered; flagged for follow-up, not banked as verdicts):

1. **The prune-flip's realized effect is the OPPOSITE of the hack:** PRUNE-step negative-ba
   mass is −23,838 vs +973 positive — the flip predominantly *charges* the policy for pruning
   measured *helpers* (PRUNE-step seed_contribution: p50 = **+5.16**, p90 = +19.2; only
   1,055/7,855 measured prunes had negative LOO). 94% of prunes (131k) fire on unmeasured
   TRAINING-stage seeds and carry no ba either way — the churn is mostly ba-silent.
2. **SET_ALPHA_TARGET is the highest-yield op per step:** 19.9% of all positive-ba mass in
   just 35k steps (mean pos-ba 3.16/step vs WAIT's 0.30) — consistent with audit F2-c3's
   "aim a cheap op at the best seed" surface actually being used (though partially benign:
   the op legitimately targets high-contribution BLENDING seeds). Worth a targeted look
   before any dense-credit redesign.

## Coverage — the pre-registered A/B licensing read

| Quantity | Value | Note |
|---|---|---|
| Env-episodes with ≥1 fossilize (bootstrap floor) | **18.5–22.0%** per run | term fires ~1 in 5 episodes |
| k distribution among firing episodes | k=1: 2,208 · k=2: 126 · k=3: 5 | **k≥2 = 1.1% of all episodes** |
| Positive value mass on fossilized seeds — **J currency (per-param)** | **66.7%** | **YELLOW** (50–75% band) |
| Positive value mass on fossilized seeds — raw cf-points | **19.4%** | would read BLOCKER |
| HOLDING-reached seeds that fossilize | 2,475 / 7,373 (33.6%) | |
| Positive cf mass parked in HOLDING-never-fossilized | **66% of ALL positive cf mass** (816k / 1,228k) | the mass the top-up exists to unlock |

**Currency ambiguity, called honestly:** the pre-registration said "positive pre-fossil value
mass" without fixing the unit. The two currencies diverge because fossilized seeds are
small-param/high-efficiency (per-param they carry most J). Since the north-star J and the
A/B primary metric (committed-J) are param-normalized, **J currency is the aligned reading
→ coverage = YELLOW: proceed to the OFF/ON A/B only with a large effect-size target**
(per the pre-registered rule). The raw-mass reading is reported so the owner can overrule.

**Synergy-sampling scarcity (design-relevant new fact):** with k≥2 in only 1.1% of episodes,
the *Shapley/synergy* component of the top-up is sampled in ~6% of its own firings
(131/2,339); in the rest it degenerates to a solo marginal-vs-empty credit. The A/B will
mostly test "terminal commitment credit," not "synergy credit." Interpret accordingly.

## Consequences for the standing decisions

1. **Phase −1 clip-arm verdict (audit risk #1):** the F1 confound is bounded at 0.17% of
   positive-ba mass *on the control policy*. No arm runs exist on disk yet; the F1 probe
   must still be re-run on the arm telemetry itself when it exists (the clip could shift
   mass toward the exempt channel), but the prior from control is that it will not.
2. **F1/F5 fixes before the A/B:** the pre-registered rule says fixes only if coverage is
   BLOCKER-band. Coverage is YELLOW (J currency) and F1/F5 occurrence is negligible →
   **no code fixes before the A/B**; F1's clip-exemption and F5's fallback stay open as
   audit findings (mechanism real, occurrence bounded) for post-A/B hardening.
3. **Never-fossilize pathology:** now quantified in mass terms — two-thirds of measured
   positive value parks in HOLDING and never commits. This is the term's actual target.
4. **corr decomposition:** per run, corr(J, Σba) ≡ corr(J, Σreward) to 3 decimals
   (0.179/0.207/0.239/0.261/0.224) — bounded_attribution is the reward's *entire*
   J-relevant content; every other additend is J-orthogonal on these runs.

## Caveats

- Occurrence ≠ impossibility: these bounds hold for the CURRENT trained policies (which are
  the degenerate-entropy population noted in PDR-0005/esper-lite-425dcc4ca2). A healthier
  policy could exploit F1/F2 surfaces the current one does not.
- Coverage is measured on the OFF-arm policy, which is precisely what the term would change;
  it is a *bootstrap-opportunity* bound, not a payment-fairness bound.
- F5's residual: fallback firing inside refresh gaps was bounded, not measured to zero.
- The F2 value-alignment reading is aggregate-level; it does not test the counterfactual
  ("would r0c1/r0c2 seeds have developed under symmetric credit"), which is a paired-run
  question, not a telemetry question.
