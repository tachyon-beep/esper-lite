# PDR-0020 — F2 calibration FROZEN: scale=1.0, cap=5.0pp, std_floor=0.25, normalized_cap=3.0; guard architecture G1–G4

Date: 2026-07-03   Status: accepted (within grant; owner directed "proceed to F2" via
PDR-0018/0019; drl-expert review completed pre-freeze)   Author: Claude (agent)
Related: PDR-0018, PDR-0019; gate esper-lite-f22a1d48a7 comments #103–#105;
docs/analysis/2026-07-03-f2-scale-cap-calibration.md (data + full review outcome);
scripts scratchpad/f2_calibration/

## Context
Gate criterion 2 required calibrating the four knobs against the divisor the credit
meets. Offline Welford reconstruction (order-insensitive to 0.000%) measured the
terminal running std at 0.345→1.86 across a run (5.4× drift); the 13,448 logged k=2
factorials gave the payment prior (P(pay)=45.4% at tau=0.28, positive gaps p50 1.58pp /
p99 13.4). During calibration, code verification established that k=1 pays structurally
zero (φ ≡ c_paid), sharpening the payment floor to k≥2 episodes (PDR-0019 §3).

## The call
- **scale=1.0** (natural pp currency; bounding is the guards' job).
- **cap=5.0 pp** (≈p85–p90 of the gap prior; the p99 tail is the CF-noise/entrenchment
  population and must not pay in full).
- **std_floor=0.25** (safety-only, below the observed minimum 0.345; unit-consistency
  upheld by review: the credit shares BOTH the reward-std divisor and the batch
  advantage z-score with every neighboring reward — a private stationarity floor would
  desynchronize it. 0.5 recorded as risk-averse fallback; 1.0 as the ON-recalibration
  lever).
- **normalized_cap=3.0** — the SOLE operative bound (the ±10 clip is not in this
  channel's path); ≈2× the largest existing terminal package. Plus the previously
  missing validation: normalized_cap ≤ REWARD_NORMALIZER_CLIP (10.0).
- **Guard architecture:** G1 in-run runaway breaker (≥13 fossils/batch × 2 consecutive
  → abort); G2 end-of-run gross-spike backstop (3× paired rate); G3 HARD efficiency
  floor (10% provisional pending a banked paired noise floor); **G4 paid-event REVIEW
  TRIGGER** (>36 paid events/run forces the G3+Δcorr adjudication; never auto-reject —
  the honest success case also raises paid count). G3 + Δcorr are the correctness
  adjudicators; G1/G2 are breakers. (drl MAJOR-2 correction adopted.)

## Options considered
Stationarity floor ~1.0 (rejected: desynchronizes shared normalization; blunts the
signal the direction A/B is powered for); hard paid-count gate (rejected: Goodharts the
guard); carrying no v(S) table (rejected: one-shot chance to retire the transient-
factorial proxy caveat).

## Reversal triggers
- First ON run: per-event telemetry shows the effect carried by std_used<1.0 events →
  re-confirm at std_floor ≥ 1.0 before banking (confound gate, pre-registered).
- ON tau recalibration P99 > 2× placebo tau (PDR-0017 trigger, unchanged) → deadband
  method reopens before the A/B is read.
- normalized_cap bind rate dominant on early events → magnitude gradation flattened;
  revisit ncap before n=10.
