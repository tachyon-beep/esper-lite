# Current State — Esper        Checkpoint: 2026-07-04 evening (checkpoint #13 — relaunch gates 3/4 green; guardrail batch fixed; PDR-0024)

## The bet right now
**Reward credit-assignment redesign — A/B in controlled recovery, one gate from
ON relaunch.** GATE-crash fix shipped and adjudicated (PDR-0023); relaunch gates
3 of 4 PASSED. Metric: unchanged — PDR-0019 criteria (i)–(v) primary;
episode-level paired Δcorr(reward,J) ≥ +0.10; asymmetric null.

## In flight
- **OFF set COMPLETE (5/5):** s41/s42/s44/s45 banked at 6dd80716 + s43 solo
  rerun finished clean 2026-07-04 ~16:58 (2400/2400 EPISODE_OUTCOME, zero
  tracebacks, zero TOPUP — arm integrity verified). OOM predecessor stays
  quarantined (off_s43_oom_partial).
- **Relaunch gates (PDR-0023 §6): 3/4 green.** CPU OFF no-op ✅ (byte-identical
  digest d468d3e7…); ON smoke ✅ (96/96, 8 finite TOPUP, organic
  {ADD,GATE,MULTIPLY} coverage, 0 paid as expected); off_s43 completion ✅.
  **Remaining: GPU off_s41 replay** at the patch-set (pid 4115042, started
  13:24, ETA ~18:45, telemetry/relaunch_gates/off_replay_s41/). On completion:
  compare decision/reward/lifecycle streams + final acc vs banked off_s41;
  CUDA/co-tenancy drift ⇒ same-commit control BEFORE attribution (reversal
  trigger).
- **ON wave armed:** 5 seeds from `.worktrees/ab-on-relaunch` (6dd80716 +
  f98b6b3e + d6514416), ≤2 runs/GPU, telemetry to
  telemetry/shapley_ab_n5/on_s4*, monitor pattern ready. Fires on gates-green +
  the owner's launch/wait answer.
- **CI guardrails now GREEN on the branch** (e0b2e84d, PDR-0024): mypy 0,
  leyline 0, gpu-sync 130/130. Latent allow_batch_shrink TypeError deleted.
  HEAD-only; frozen worktrees untouched.

## Scoring rules (prereg + 2026-07-04 gate-fossil ADDENDUM — do not relitigate)
All PDR-0019 rules stand. The stored×12+env_id episode_idx decode is RETIRED
for relaunched ON TOPUP events (quarantined crashed attempts only);
gate/non-gate stratification reads from TOPUP `alpha_algorithms` + `tau_used`;
tau=0.28 is ADD-placebo provenance — GATE-credit MAGNITUDE claims unlicensed
without a gated null-player read (n=5 direction unaffected); stratified P99
recalibration; population-validity gates carried forward. NO
outcome-conditioned rerun clause: n=5 = direction, n=10 = magnitude/banking.

## Open questions / blocked-on-owner
- **STANDING (checkpoint #12): may the ON wave fire on CPU byte-identity +
  passing smoke, or must it wait for the GPU off_s41 replay comparison?**
  Default = wait. Now nearly moot — the replay lands ~18:45; if it compares
  clean the question answers itself (gates all green → launch).
- Standing: north-star/rent TARGET placeholders (metrics.md); scale>0 beyond
  this experiment stays owner-gated; G3 paired noise floor unbanked (bank or
  state absence at scoring).

## Last checkpoint did (checkpoint #13)
- Recorded relaunch gates 3/4: ON smoke PASSED + off_s43 rerun complete/clean
  (OFF set 5/5) — PDR-0023 checklist current (40fcb765); tracker comments
  114–115 on esper-lite-f22a1d48a7.
- **PDR-0024:** external-review guardrail batch re-adjudicated — violations are
  branch-introduced vs main (not "pre-existing"; amends PDR-0022's hygiene
  disposition) — and FIXED (e0b2e84d): explicit returns, severed
  allow_batch_shrink feature deleted, 8 leyline + 7 gpu-sync justified
  whitelist entries; 57 tests green.

## Next session, start here
**The GPU off_s41 replay comparison** (task notification will fire; PDR-0023
§6 mandates the stream comparison; drift ⇒ same-commit control first). If it
compares clean: all gates green → launch the ON wave (5 seeds, patch-set
worktree, ≤2 runs/GPU), monitor armed, then score STRICTLY per prereg +
addendum. Queued behind the A/B decision: advantage-pathology / EV track
(esper-lite-f25b71c165).
