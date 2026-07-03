# Current State — Esper        Checkpoint: 2026-07-03 (checkpoint #10 closed — A/B LAUNCHED + liftoff CONFIRMED; PDR-0020, PDR-0021)

## The bet right now
**Reward credit-assignment redesign — the enablement A/B is RUNNING.** Committed-Shapley
top-up: F2 knobs FROZEN (PDR-0020: scale=1.0, tau=0.28, cap=5.0, std_floor=0.25,
ncap=3.0 — pinned by tests/scripts/test_shapley_on_config.py); pre-A/B build accepted
after triple verification (PDR-0021; commits bb8d3c90+459d01cf; OFF-arm bitwise identity
survived a dedicated adversarial attacker lens). Metric: PDR-0019 — design criteria
(i)–(v) primary; episode-level paired Δcorr(reward,J) ≥ +0.10 floor; asymmetric null.

## In flight
- **Paired n=5 OFF/ON A/B RUNNING on both GPUs** (owner GPU-go 2026-07-03 "roll straight
  into the test"; liftoff verified same day — 5 OFF runs training, both GPUs ~100%):
  seeds 41–45, both arms at launch commit **6dd80716** (prereg-stamped at 1c5046ff,
  docs-only ahead — source-identical), telemetry → telemetry/shapley_ab_n5/{off,on}_s*.
  Per-GPU launcher (session scratchpad launch_gpu.sh; commands recorded in the prereg
  doc) runs the OFF wave then chains the ON wave automatically; cuda:0={41,43,45},
  cuda:1={42,44}. G1 guard arms on ON runs only. A persistent in-session monitor tails
  all ten logs for crashes/OOM/guard-trips/completions. NOTE: the old causal_r1_n5
  controls are at an older commit and are NOT the OFF arm (PDR-0015 same-commit rule).
- Gate criteria 4–6 (entrenchment monitor, dormancy recheck, first-ON-run residuals
  incl. MANDATORY tau recalibration) execute during/after the ON runs per
  docs/analysis/2026-07-03-shapley-ab-scoring-preregistration.md.

## Scoring rules (pre-registered — do not relitigate at scoring time)
Primary (i)–(v); Δcorr floor; G2 backstop (3× paired fossilize rate), G3 HARD (10%
efficiency, provisional pending a banked paired noise floor — bank it before scoring or
state its absence), G4 paid-event review trigger (>36/run forces adjudication, never
auto-rejects); std<1.0 confound gate; G1-abort pairing rule (aborted seed = pair
dropped); a NULL n=5 fires NOTHING (asymmetric null, PDR-0019).

## Open questions / blocked-on-owner
- None for the runs. Standing: north-star/rent TARGET placeholders (metrics.md);
  enabling scale>0 BEYOND this experiment stays owner-gated (gate criteria 4–7);
  G3's paired noise floor for the efficiency read not yet banked (drl MINOR-3).

## Last checkpoint did (checkpoint #10 + close-out)
- PDR-0020 (F2 freeze + G1–G4 guard architecture), PDR-0021 (build accepted on the
  drl-review → TDD → 28-agent adversarial-review chain; A/B launch owner-authorized).
- Build shipped (bb8d3c90+459d01cf): payload provenance + v(S) table, G1 guard,
  ncap ≤ clip validation, ON config + pin tests, scoring pre-registration; cleared both
  pre-existing tracked red tests; suite green 5,126+; esper-lite-3f3e9b9b4f closed;
  plan → completed/.
- Launched the A/B (owner-directed), stamped the launch commit into the prereg doc
  (1c5046ff), VERIFIED liftoff (5 OFF runs training, GPUs saturated), armed the
  failure/completion monitor. No new decisions in the close-out — no new PDR.

## Next session, start here
**Monitor the A/B runs** (Karn: run_dir filter telemetry/shapley_ab_n5/). On completion:
score STRICTLY per the pre-registration doc (record the commit hash check first), run
the ON tau recalibration (P99 null-player excess vs 0.56pp trigger), then bring the
verdict + the G4/confound-gate reads to the owner. If any ON run G1-aborts: drop the
pair, investigate before any re-run. Queued behind the A/B decision: advantage-pathology
/ EV track (owner sequence, PDR-0018-era ruling).
