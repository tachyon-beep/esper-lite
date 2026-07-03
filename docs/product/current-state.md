# Current State — Esper        Checkpoint: 2026-07-03 (checkpoint #9+F2 — F2 knobs FROZEN (drl-reviewed); scoring re-based per PDR-0019; next = pre-A/B build)

## The bet right now
**Reward credit-assignment redesign.** Term BUILT default-OFF (PDR-0013). Enablement gate
(esper-lite-f22a1d48a7) OPEN and advancing: tau = +0.28 pp **ACCEPT-PROVISIONAL / LOWER-BOUND
ONLY** (owner-ratified, PDR-0018); zero-GPU occurrence probes DONE (pre-registered comment
#99, results #100/#101 + docs/analysis/2026-07-03-occurrence-probes-f1-f2-f5-s1-coverage.md);
coverage **YELLOW** (66.7% J-currency). `shapley_synergy_scale=0.0` everywhere. A/B metric
renamed **`fossilize_payable_J`** (never bare "committed-J").

## In flight
- Nothing on GPU. **F2 calibration FROZEN 2026-07-03** (criterion 2 SATISFIED; drl-expert
  review: all knobs ACCEPTED): scale=1.0, cap=5.0pp, std_floor=0.25 (0.5 fallback / 1.0
  ON-recalibration lever), normalized_cap=3.0 (**sole operative bound** — ±10 clip not in
  this channel). Memo + review outcome: docs/analysis/2026-07-03-f2-scale-cap-calibration.md;
  scripts scratchpad/f2_calibration/.
- **Scoring re-based (PDR-0019, owner-ruled):** primary = design criteria (i)–(v);
  effect-size floor = episode-level paired Δcorr(reward,J) ≥ +0.10 (n=5) / LB>0 (n=10);
  Δfossilize_payable_J overlay SUPERSEDED (structurally-zero median); per-run paid mass
  REJECTED as Goodhart-aligned; **asymmetric null recorded** (null n=5 fires nothing).
- **Telemetry hygiene** (esper-lite-425dcc4ca2): open — NOTE the entropy-degeneracy premise
  is stale per PDR-0006 (decision-step entropy HEALTHY); remaining scope is read-path
  wiring + alarm recalibration, no longer an A/B prerequisite.

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
**The pre-A/B build** (tracker: see gate comment #105 conditions 3+4): (a) per-paid-event
`(credit_buf, met_std)` + normalized_cap bind-rate telemetry at the delivery seam
(telemetry-only); (b) the criterion-3 fossilize-count guard + HARD §3 safety gates — the
drl-mandated enablement precondition (knobs bound magnitude, not payment frequency; guard
semantics need a small design pass first); (c) ON-arm A/B config with the frozen knobs +
tau=0.28 (NUM_BLUEPRINTS same-commit constraint, PDR-0015); (d) the scoring-time confound
gate in the analysis plan (effect carried by std<1.0 events → re-confirm at floor ≥1.0).
Then launch the paired n=5 OFF/ON A/B scored per PDR-0019. Housekeeping: 3 pending P3
test-hygiene observations (seed_residency round-trip rule, optimizer-lifecycle mock drift,
seed_residency view expected-set) — mechanical, unblocked since 93bc930e.
