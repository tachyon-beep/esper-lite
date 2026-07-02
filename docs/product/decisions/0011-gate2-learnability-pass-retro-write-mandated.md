# PDR-0011 — GATE 2 (learnability) PASSED; retro-write delivery MANDATED; terminal-flush killed

Date: 2026-07-02   Status: accepted (within grant — run analysis, dispatch delivery, launch/kill GPU runs within the active bet)   Author: Claude (agent)
Supersedes: —   Related: PDR-0010, docs/analysis/2026-07-02-gate2-learnability-probe-preregistration.md, docs/analysis/2026-07-02-gate2-learnability-result.md, esper-lite-254175df90

## Context
PDR-0010 left GATE 2 (learnability) as the next gate on the reward-credit term. A drl-expert consult restructured
the gate into a four-link chain (reward→advantage; advantage SNR; gradient→logit; persistence vs critic catch-up)
with a three-tier probe, pre-registered BEFORE any run. Debugging the probe against the real regime banked three
corrections: the regime is the RUN CONFIG (gae_lambda=0.95, auto-forward, entropy 0.15→0.08 + per-head
multipliers), NOT leyline defaults; the design doc's "scaffold-credit rail" routing was wrong as written
(pending_hindsight_credit next-step-flushes and zeroes at reset — a terminal credit would be silently dropped);
and use_telemetry=False silently disabled the whole lifecycle (bug esper-lite-4fe98055f7, fixed this session).

## Options
(a) Terminal-flush delivery (credit into the last step's reward); (b) retro-write delivery (buffer write at the
FOSSILIZE timestep pre-GAE); (c) both fail → pivot to candidate B (temporal/hindsight credit).

## The call
**GATE 2 PASSES with retro-write as a MANDATED build requirement; terminal-flush is DEAD.** Tier-2 paired
campaign (5 same-seed pairs 51–55, control vs retro-write injection, K=30, regime config verbatim, local GPUs):
paired ΔP(FOSSILIZE|eligible) positive 5/5 seeds (median +0.081 on ~0.13 baseline), exact sign-flip p=0.0312,
persistent (no critic-erasure collapse), 3.1× the control-drift magnitude floor; guardrails all clean; CRN
pairing verified ±0.00007 pre-injection. Terminal-flush: pooled realized Δadv/σ_A median 0.0397 < the
pre-registered 0.05 FAIL bar (analytic decay at γλ=0.94525: median factor 0.0139). The PASS transfers to
candidate B (retro-write is mechanically B's delivery point), so learnability is settled for BOTH candidates.

## Rationale
Pre-registered thresholds locked before the campaign; all three PASS criteria met independently; the kill of
terminal-flush was analytic before it was empirical. One Tier-2 test answers both candidates — gate money spent once.

## Reversal trigger
Reopen GATE 2 if, at build time, the retro-write unit test (hand-computed GAE) fails, or if the built term's
calibrated credit (mature-regime scale, Shapley-gated) fails success criterion (v) of the design — the scaffold
rail demonstrably failing to raise the credited FOSSILIZE action's advantage on a paired ≥5-seed A/B.
