# R1 Pairing Pilot — Result (n=3)

**Date:** 2026-06-30 · **Scope:** the dispositive R1 leg of the causal-contribution run (v2 §6). · **Verdict
(two parts):** (1) the RNG-split + SUPPRESS-SLOT **harness is VALIDATED at full scale** for the offset-bias channel
— policy-independent, banked. (2) The run delivers **NO read on (a)/(b)**: it was measured on **collapsed,
accuracy-saturated policies** — the wrong population for the causal question. The relocation/structure signal below
is **recorded but NOT interpreted**; the causal read is **blocked until the controller is healthy**.

## What ran
`control` (RNG-split ON, no mask) + `suppress_slot_on` (r0c0 un-committable), seeds 41–43, full config
(200 ep × 150 epoch × 12 env), `gpu_preload`, eager. First attempt crashed at ep~161 on a transient non-finite
logit reaching `multinomial`; a NaN-safe sampling guard (drl-expert; `scratchpad/cc_r1_stage2/CRASH_DIAGNOSIS.md`)
fixed it. Re-run: **all 6 runs completed full-length, no crash, `finiteness-gate trips = 0` in every run** ⇒ the
guard never fired ⇒ no asymmetric-guard contamination.

## PART 1 — Harness validation (banked; policy-independent)
Gates PASS, all 3 seeds, FULL 30,000-step runs:
- **offset-free** (`control == on` controller draw-count parity) — the dispositive proof that suppressing r0c0 does
  NOT offset the controller sampling stream ⇒ **the systematic RNG-bias channel into Δ_struct is closed**. This is
  the real, policy-independent R1 deliverable.
- **zero-r0c0** germination under suppression; **within-run draw-count stride constant**.
This validates the RNG split, the content-addressed blueprint-init, the SUPPRESS-SLOT mask, and the crash patch at
production scale. **Caveat:** the *pairing-strength* read (the realized paired-Δ SD) was measured on the collapsed
regime (Part 2); the SD on a healthy policy will differ and **must be re-measured** — so "CIs trustworthy" applies
to the *no-bias* property, not yet to the variance structure the causal run will use.

## PART 2 — Why this run gives NO causal read (the population is wrong)
Both arms suffer sustained **slot-head entropy collapse from ~update 3** (first anomaly: slot entropy 0.082 < 0.1 at
3 consecutive updates; 594 anomalies/run). Two discriminators show the data cannot speak to (a)/(b):

1. **All structure commits post-collapse.** Fossils/decile (control_s41): `[30,27,29,24,33,33,71,82,68,75]` —
   committed throughout, i.e. **after** the collapse onset (update 3). So the entire committed structure is built by
   a degenerate controller.
2. **Accuracy is pinned, independent of structure.** final-acc/decile is flat ~52% in BOTH arms across the whole run
   (control `[51.9,51.7,51.9,51.9,52.0,52.3,52.5,53.0,52.8,52.7]`; on `[51.8,…,52.4]`), regardless of committing
   472 vs 540 fossils. ⇒ **Δacc ≈ 0 is accuracy-INSENSITIVITY (acc-saturated regime), not "accuracy preserved."**

Therefore the raw paired numbers below are **observations, not a verdict-table classification:**

| seed | Δ_struct (downstream params, on−control) | Δacc |
|------|------------------------------------------|------|
| 41 | +908,263 | −0.43 |
| 42 | +707,325 | +0.08 |
| 43 | +575,335 | −0.33 |
| median | +707,325 (mean +730k, SD 137k) | −0.33 (mean −0.23, SD 0.22) |

The Δ_struct>0 ("blocking r0c0 → more downstream") is **near-tautological** here: block one of three germination
slots under a churn-happy collapsed policy and the ~400-fossil r0c0 germination pressure lands on r0c1/r0c2 by
construction — that is **germination pressure redirected, not function relocated.** Combined with acc-insensitivity,
the data is uninterpretable for whether r0c0 is a freeloader (a), an enabling stem (b), or fungible. **No causal
direction is banked.**

## Next (sequencing is strict)
1. **Entropy-collapse track is STRICTLY PRIOR to any causal read** — n=5 on this config would just buy more
   uninterpretable seeds. Levers (drl-expert): fix `entropy_anneal_episodes=3000` vs 200-episode mismatch; activate
   the dormant std-floor / strengthen the entropy-floor penalty on the raw (pre-floor) distribution; keep policy
   logits fp32 (also shrinks the amp-overflow surface that caused the crash). Re-run on a healthy policy.
2. **Re-measure the realized paired-Δ SD on the healthy-policy re-run** (Part 1 caveat), THEN extend to n=5 → n=10
   (the morphogenesis seed floor, zero margin).
3. **Check whether GATE-1's runs collapsed similarly** (Karn). If collapse is a general feature of this config, it
   is a higher-priority finding than (a)/(b) and may color other reads (GATE-1's SURVIVE verdict still stands — it
   is a comparison across arms that all collapse alike — but the collapse itself wants explaining).
4. **Owner estimand decision (still pending, deferred):** unchanged by this run.

Durable: `scratchpad/cc_r1_stage2_run2/` (telemetry), `cc_r1_stage2_analyze.py` (analyzer+gates),
`cc_r1_stage2/CRASH_DIAGNOSIS.md`. Design: `docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md`.

---

## Collapse investigation (2026-06-30) — refines Part 2; decides the fix-vs-reprioritize fork

Owner-chosen cheap discriminator (from existing telemetry, no new GPU). Three findings:

1. **The morphogenesis is REAL and robust — NOT hollow.** From the counterfactual matrices (all-on vs
   all-off = host-alone), committed structure adds **+7.69pp mean across all 6 runs** (host-alone ~38% →
   with-structure ~46%; per-run +6.67…+8.27, tight). This CONFIRMS and exceeds the n=1 control:41 +6.1pp.
   ⇒ the "accuracy decoupled from structure / morphogenesis hollow" story is **REFUTED**.
2. **Refines Part 2:** structure is NOT decoupled from accuracy (it adds +7.7pp). The between-arm Δacc≈0 is that
   **both arms reach the same +7.7pp** — i.e. **r0c0 is accuracy-replaceable** (suppress it, the controller hits
   the same contribution via more downstream), not that accuracy ignores structure. Whether a HEALTHY policy also
   finds r0c0 replaceable (vs using it efficiently/enabling) is the still-open (a)/(b) question — the collapse
   caveat applies to the structural-CHOICE quality, not to whether structure matters.
3. **The entropy collapse is GENERAL and pre-existing — not caused by the causal harness.** GATE-1's control s41
   run (`telemetry/telemetry_2026-06-25_202151`) shows **411 entropy_collapse anomalies**, comparable to R1's
   ~450. So the collapse has been present throughout the reward-redesign work (the +6.1/+7.7pp were always measured
   on collapsed policies; the NaN crash was a latent transient GATE-1 happened not to trip). GATE-1 SURVIVE still
   stands (all arms collapse alike).

**Decision:** the morphogenesis works (+7.7pp, robust) and the collapse is a FIXABLE, pervasive policy-stability
issue — not fundamental brokenness. ⇒ **fix the collapse → resume the causal track is justified; re-prioritizing /
abandoning is not.** The collapse fix is now more than a causal-run prerequisite — it is a foundational improvement
(may raise the contribution; required for trustworthy (a)/(b) structural decisions; affects the whole line of work).

---

## UPDATE 2026-07-01 — the "collapse" was a MEASUREMENT ARTIFACT; the fork discriminator is J, not accuracy

**SUPERSEDES Part 2's "wrong population" block AND the Collapse-investigation's "fix collapse first."** A diagnose-first
ultracode workflow (4 agents + adversarial synthesis, very-high confidence) + independent verification reversed the
entropy-collapse premise.

**1. There was NO training collapse — it is a telemetry/analyser artifact.** The anomaly detector
(`ppo_coordinator.py:619`) reads `head_X_entropy` = per-head entropy averaged over ALL steps; on the ~60% of steps where
a head does not decide (single valid action) normalized entropy is EXACTLY 0, so `head_X_entropy = learnable_fraction ×
conditional_entropy` — a decision-DENSITY proxy. Verified identity: `head_slot_entropy == head_slot_learnable_fraction
= 0.0872` at birth (ratio 1.0000). The slot threshold (0.10) sits BELOW that diluted baseline from update 1, so the
~411 anomalies are FALSE and structurally can never clear. Reconstructed DECISION-STEP slot entropy is HEALTHY:
control 1.0→~0.70 (min 0.48), suppress 1.0→~0.60 (min 0.30) — never <0.1; op (undiluted) ~0.88. The entropy bonus is
alive (`entropy_loss=0.0` is a hardcoded emitter STUB; the real term rides the `entropy` field 14.16→12.99); the floor
is anti-collapse; cardinality + anneal ruled out; GATE-1 reproduces the identity → structural, run-independent.
⇒ **the policy ran on a HEALTHY-EXPLORATION population. The "wrong population" block on the R1 causal read is
RETRACTED.** (Caveat: healthy exploration ≠ fully-converged policy — the ordinary n=3→n=5/n=10 caveat stands.)

**2. The fork discriminator is J (committed counterfactual gain per param), NOT accuracy.** On accuracy both arms tie
(~+7.7pp) — which is exactly why "replaceable" kept oscillating. On acc-per-param they do NOT tie (R1 re-analysis,
existing data, no GPU):

| seed | control pp/Mparam | suppress pp/Mparam | Δ | control totParams | suppress totParams |
|------|-------------------|--------------------|-----|-------------------|--------------------|
| 41 | 19.96 | 7.34 | −12.6 | 411k | 1,126k |
| 42 | 11.92 | 7.51 | −4.4 | 560k | 1,079k |
| 43 | 16.34 | 9.30 | −7.0 | 438k | 830k |
| median | | | **−7.0** | | |

Suppressing r0c0 roughly HALVES system parameter-efficiency (same accuracy, ~2–2.7× the total params; cf/param
0.008–0.011 → 0.004–0.005). **Dropping r0c0 makes the system LESS efficient — the OPPOSITE of a free-droppable
freeloader.**

**Corrected verdict (PILOT, n=3 — bank the METHOD, not yet the direction):**
- "R1 was on a collapsed / wrong population" → FALSE, retracted.
- The (a)/(b)/(c) fork's discriminator is **ΔJ / Δ(acc-per-param)**; the "+707k params for ~0 accuracy" gap is the SIGNAL.
- At n=3 the pilot leans AWAY from (a) freeloader and TOWARD **(b) an efficiency-enabling stem the LOO reward
  UNDERVALUES** — penalizing r0c0 would be the wrong fix; crediting its enabling/efficiency contribution is indicated.
  NOT banked (n=3; design needs n=5 to begin causal evidence, n=10 floor; healthy-exploration ≠ converged).

**Sequencing change:** "fix the collapse first, then re-run R1" is MOOTED — no collapse to fix. The telemetry read-path
swap (wire `conditional_head_entropies` to the detector; fix the `entropy_loss` stub; emit conditional entropy; relabel
the raw series a density proxy) is HYGIENE (so the analyser stops lying), shipped bit-identical-guarded. The next
decision-relevant GPU step is the **n=5 J-read**, NOT a collapse-fix rerun.
