# Current State — Esper        Checkpoint: 2026-07-03 (checkpoint #9 — owner ruled: tau ACCEPT-PROVISIONAL, probes ran, coverage YELLOW, F2 calibration LICENSED; PDR-0018)

## The bet right now
**Reward credit-assignment redesign.** Term BUILT default-OFF (PDR-0013). Enablement gate
(esper-lite-f22a1d48a7) OPEN and advancing: tau = +0.28 pp **ACCEPT-PROVISIONAL / LOWER-BOUND
ONLY** (owner-ratified, PDR-0018); zero-GPU occurrence probes DONE (pre-registered comment
#99, results #100/#101 + docs/analysis/2026-07-03-occurrence-probes-f1-f2-f5-s1-coverage.md);
coverage **YELLOW** (66.7% J-currency). `shapley_synergy_scale=0.0` everywhere. A/B metric
renamed **`fossilize_payable_J`** (never bare "committed-J").

## In flight
- Nothing on GPU. **Next dispatch: F2 scale/cap calibration** (gate criterion 2 —
  scale/cap/normalized_cap/std_floor against the minimum running std the term will meet;
  normalized_cap currently has NO upper sanity bound). The A/B launches only after F2 is
  frozen, under the PDR-0018 large-effect targets.
- **Telemetry hygiene** (esper-lite-425dcc4ca2): unchanged; still the prerequisite for any
  healthy-policy causal read, and the probe caveat leans on it (occurrence bounds hold for
  the current entropy-degenerate policy population only).

## Probe verdicts (banked 2026-07-03)
F1 prune-flip NOT exploited (0.17% of pos-ba mass; flip mostly CHARGES helper-prunes);
F5 downgraded to latent-bug (all-off config in 100% of CF matrices); S1 mild (fossilize
timing near-uniform); F2 occurring at full scale but value-aligned in aggregate (92.4% of
steps / 97.8% |ba| on r0c0 vs 91–97% of value mass) — calibrate, don't correct; NO
slot-specific r0c0 correction pre-A/B. 66% of positive cf mass parks HOLDING-never-
fossilized (the follow-on target if the A/B underperforms: pre-fossil credit unlocker,
salvage-B-adjacent). k≥2 in 1.1% of episodes ⇒ the A/B is a terminal COMMITMENT-credit
test, not a synergy test. SET_ALPHA_TARGET harvest signal (19.9% of pos-ba mass, 10× per-
step yield) flagged for a targeted probe before any dense-credit redesign.

## Standing constraints on the A/B (PDR-0018)
n=5 = direction gate: median paired Δfossilize_payable_J ≥ 2×tau_provisional (+0.56 pp now),
≥4/5 seeds, EPISODE-level Δcorr ≥ +0.10, safety floors (acc ≥ −0.3 pp, efficiency ≥ −10%,
fossilize opportunity not suppressed). n=10 = magnitude/bank: ≥2×tau_ON with bootstrap
LB > 0; ON-run tau recalibration MANDATORY first. Design-doc criteria (i)–(v) remain the
PRIMARY gates. F1 probe must re-run on Phase −1 arm telemetry BEFORE any clip-arm verdict
(no arm runs on disk). Advantage-pathology/EV track stays queued BEHIND the A/B decision.

## Open questions / blocked-on-owner
- None blocking F2 calibration. Standing: n=10 magnitude call (now wired into the PDR-0018
  n=10 gate); north-star/rent TARGET placeholders in metrics.md still owner-unset;
  NUM_BLUEPRINTS 13→14 paired A/Bs same-commit (PDR-0015).

## Last checkpoint did (checkpoint #9)
- Pre-registered licensing rule BEFORE probe data (gate comment #99); ran F1/F2/F5/S1 +
  coverage probes (two probe bugs caught, corrected, disclosed); banked analysis doc +
  gate/audit comments (#100/#101).
- Owner ruled (via external-analysis review): PDR-0018 recorded — tau accept-provisional,
  J-currency coverage YELLOW, large-effect targets, no F1/F5 pre-A/B fixes, F2 calibrate-
  not-correct, scope naming, never-fossilize as follow-on.
- metrics.md: tau row ratified; coverage row added; corr row got the episode-level Δ target.
- Committed the owner's two-checkpoint-old Phase-0 analysis-doc edits (data-loss risk closed).

## Next session, start here
**F2 scale/cap calibration** (design doc + gate criterion 2; scoping banked at gate comment
#103). RESOLVED: running std is NOT logged in telemetry — primary path is offline Welford
reconstruction of the std trajectory from the logged total_reward stream (order-sensitivity
caveat + validation check recorded; fallback: cheap instrumentation run — `current_std()`
exists at simic/control/normalization.py:259). Read the std distribution at episode-terminal
steps (early-run minimum matters most), then propose scale/cap/normalized_cap/std_floor
(incl. the missing normalized_cap upper bound) with drl-expert review before freezing.
After freeze: n=5 OFF/ON A/B per PDR-0018. Housekeeping: 3 pending P3 test-hygiene
observations (seed_residency round-trip rule, optimizer-lifecycle mock drift, seed_residency
view expected-set) — mechanical fixes, now unblocked since the Phase-0 doc edits are
committed (93bc930e).
