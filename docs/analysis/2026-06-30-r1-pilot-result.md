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
