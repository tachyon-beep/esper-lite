# Current State — Esper        Checkpoint: 2026-07-05 ~03:40 (checkpoint #14 — ON WAVE RUNNING; all relaunch gates passed; PDR-0025)

## The bet right now
**Reward credit-assignment redesign — the ON wave is IN THE AIR.** All four
PDR-0023 relaunch gates passed (frozen-rule adjudication, PDR-0025); the owner
gave the explicit launch word 2026-07-05 ~00:30. Metric: PDR-0019 criteria
(i)–(v) primary; episode-level paired Δcorr(reward,J) ≥ +0.10; asymmetric null.

## In flight
- **ON wave RUNNING (4/5):** on_s41+on_s42 on cuda:0, on_s43+on_s44 on cuda:1,
  all from `.worktrees/ab-on-relaunch` (6dd80716 + f98b6b3e + d6514416),
  scale=1.0, frozen F2 knobs. At 03:34: ~70–74/200 batch-episodes each, zero
  tracebacks. ETA first completions ~09:00. **on_s45 QUEUED — launch it on the
  first completion notification (freed slot, ≤2/GPU), done ~15:00.**
- **Crash monitor armed** (persistent task bxel34ks0: tracebacks/OOM/governor/
  completion across all five stdout logs). 2h mechanism read HEALTHY
  (comment 121; metrics.md guardrail row): GATE fossils paying through the
  production seam; ⚠ G4-flag = k=2 paid fraction 47% vs ~1% null prior +
  cap-clipped magnitudes — audit at scoring, not mid-run.
- **OFF set banked 5/5.** Aborted prelaunch fragments quarantined at
  on_s4{3,4}_aborted_prelaunch_denial_20260705 (rename only, nothing deleted);
  crashed 6dd80716 attempts remain in gate_crashed_on_wave_6dd80716/.

## Scoring rules (prereg + addendum + PDR-0025 — do not relitigate)
All PDR-0019 rules stand. episode_idx decode RETIRED for these runs (true ids
CONFIRMED live in production TOPUP); gate/non-gate stratification via TOPUP
`alpha_algorithms` + `tau_used`; stratified P99 tau recalibration; GATE-credit
MAGNITUDE claims unlicensed without a gated null-player read; G4 paid-event
review is where the 47%-paid flag gets adjudicated; population-validity gates
carried. n=5 = direction, n=10 = magnitude/banking. NO outcome-conditioned
forks. Binding record language: the GPU gate proved drift-not-patch-specific,
NOT bitwise no-op (CPU digest is the stronger proof).

## Open questions / blocked-on-owner
- None new — the launch/wait question was RESOLVED by explicit owner word
  (PDR-0025 §5 records the full authorization chain incl. the classifier
  denial and the self-corrected Hold).
- Standing: n=10 magnitude/banking call comes only AFTER the n=5 direction
  read; scale>0 beyond this experiment stays owner-gated; north-star/rent
  TARGET placeholders (metrics.md); G3 paired noise floor unbanked.

## Last checkpoint did (checkpoint #14)
- **PDR-0025:** frozen acceptance rule (pre-result) + PASS verdict (control
  decorrelated harder than replay; trip-wire silent) + low-sensitivity record
  language + the launch authorization chain + 2h mechanism read.
- metrics.md: dated ON-wave mechanism-validity guardrail row (healthy; G4 flag).
- Tracker comments 118–121 (rule freeze, verdict, launch, mechanism read).

## Next session, start here
**(1) On any completion notification: run integrity checks (2400/2400, zero
tracebacks, TOPUP present+finite), then LAUNCH on_s45 into the freed GPU slot**
(same command shape, ≤2/GPU). (2) When all five are banked: score STRICTLY per
prereg + addendum — criteria (i)–(v), paired Δcorr, asymmetric null, stratified
tau recalibration, G4 paid-event review (the 47% flag), population-validity
gates. (3) Checkpoint the scoring verdict before acting on it. Queued behind
the A/B decision: advantage-pathology / EV track (esper-lite-f25b71c165).
