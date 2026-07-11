# PDR-0061 — PDR-0055/0060 landing ACCEPTED (drl-approved); 600-round diagnostic launched on fixed code

Date: 2026-07-11
Status: accepted (execution + acceptance within grant; run launch under the standing
run authorization with PDR-0055's pre-committed reading)

## Context

Freeze lifted after the Stage-2 REJECT (PDR-0059). PDR-0055 licensed the
penalty-schedule fix (drl-expert review REQUIRED, 0–199 coefficient-identity
regression) followed by the clean 600-round seed-41 OFF diagnostic; PDR-0060 adopted
the vg Gram-matrix telemetry contract riding the same observability landing.

## What was done and accepted (landed `7ea84dc6`, full suite 5341 green)

1. **Schedule fix (PDR-0055):** `_get_penalty_schedule` breakpoints are ABSOLUTE
   update rounds pinned to the 200-round shape (leyline `ENTROPY_PENALTY_*`);
   `total_train_steps` retired end-to-end; pre-fix checkpoints fail with a
   break-naming RuntimeError. Implementation detail that mattered: pinned-fraction
   arithmetic, because naive round-based decay differs in the last ulp on 16/50
   decay rounds and would have broken the pre-registered bit-identity property.
2. **vg sufficient-statistic telemetry (PDR-0060):** `compute_vg_sufficient_stats`
   (leyline.value_metrics) emits count + means + upper-triangle Gram (raw scale)
   once per update; disjoint OFF (v, g) / ON (v_main, v_cf, g_main, g_cf) key
   families; wired agent → reducer whitelist → PPOUpdatePayload (leg-absent fields
   dropped at serialization) → karn `ppo_updates` view (20 columns).
3. **drl-expert review of record: APPROVE-WITH-NITS** (high confidence; bit-identity
   independently reproduced; Gram sufficiency derived from first principles;
   training-path read-only invariance traced). Both nits closed pre-landing:
   checkpoint break-guard + end-to-end leg-family payload tests (real update →
   reducer → emitter, non-None per own leg / None per other leg).
4. **Acceptance criteria met:** pre-registered coefficient-identity regression
   (float ==, rounds 0–199) green; drl gate passed; both tracker tasks closed
   (esper-lite-00ff305247, esper-lite-657fec0019).
5. **600-round seed-41 OFF diagnostic LAUNCHED** 2026-07-11 19:29 AEST at
   `7ea84dc6` via `telemetry/stage2_off_longdiag/run_longdiag_v2.sh` (session-proof
   setsid, cuda:1, gpu-preload). Log:
   `telemetry/stage2_off_longdiag/logs/longdiag-s41.log`; telemetry:
   `telemetry/stage2_off_longdiag/seed41/telemetry_2026-07-11_192943/`. Verified
   live (90% GPU util, morphogenesis active, epoch 41 by 19:33). First run emitting
   vg telemetry ((v, g) family). Quiet-box rule standing while it writes.

## Offline-consumer caveats banked from the review

- **Bit-identity is scoped to EXACTLY 200-round runs.** Any other length changes the
  realized schedule by design (a 50-round smoke sits in the 1.5× boost throughout).
  Non-200-round arms are not schedule-comparable with the completed A/B arms.
- **vg moments are population (correction=0); the EV family is correction=1.** Apply
  the N/(N−1) factor before cross-comparing a vg-reconstructed variance against an
  EV-denominator variance.
- **Mean-of-Grams pooling across updates** is exact only for equal per-update counts
  (moot at ppo_updates_per_batch=1: one update per event).

## Reversal trigger

- If the diagnostic's rounds 0–200 diverge materially from the completed seed-41 OFF
  arm beyond GPU-nondeterminism tolerance (PDR-0035 replay lessons — corroboration,
  not bitwise), the true-continuation claim is void: investigate provenance before
  reading rounds 250–600; fallback remains PDR-0054's salvaged window on the old
  code's data.
- PDR-0060's reversal trigger stands unchanged: if the exact (Gram) decomposition on
  this instrumented run contradicts the approximate pre-DECIDE reads, the
  provisional findings are discarded wholesale and the path lean resets to neutral.
