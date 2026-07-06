# Current State — Esper        Checkpoint: 2026-07-06 (checkpoint #22 — Stage-2 MAJOR-1 acceptance harness designed + scorer built + review-hardened [PDR-0037]; NOT-yet-freezable pending the telemetry wrapper)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), at **Stage-2 acceptance**
(esper-lite-2a4b56e719, in_progress). Stage-2 HRA de-shaping is **code-complete for a
valid ON run** (verified vs current code — the ON-leg cf value + reward streams are wired
end-to-end with a loud guard against "cf trained on zeros"; the 12-day-stale memory's fear
is RESOLVED). The gate now needs a falsifiable, pre-registered acceptance test before any
A/B — that harness is what this session built. Metric it moves: the **Stage-2 MAJOR-1
acceptance gate** (metrics.md) → ultimately host-accuracy contribution (guardrail).

## What just happened (this session)
Designed, built, and review-hardened the **MAJOR-1 acceptance harness** (analysis code only,
no behaviour change, per owner ruling — no ON run until reviewed + frozen):
- **Design:** rubric via `drl-expert`, then merged an external (ChatGPT) critique under a
  take-good/dump-bad filter — adopted 4 (actor-objective A-vs-B declaration, validity-gate
  first layer, per-stream target-scale diagnostics, churn/guard-channel safety), rejected 2
  on code/algebra grounds (advantage-std *level* signal; paired-ΔEV_main primary — no
  `EV_main_OFF` exists, ON-leg-only telemetry).
- **Built:** frozen-candidate gate doc `docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`
  + pure scorer `src/esper/simic/telemetry/stage2_acceptance.py` (42 TDD tests, ruff + mypy
  clean). Committed `feat/ev-stab-stage2-hra@2fad1fc4`.
- **Reviewed (2 adversarial passes):** python-code-reviewer — 3 critical + 4 warnings, ALL
  FIXED. drl-expert — **VERIFIED the core** (exact Wilcoxon vs an independent subset-sum-DP
  reference; the §4 identity; §8a/b comparability; canonical hollow win caught); found a
  **compound ACCEPT route** (H2 tier-blindness + H3 mech absolute-count + M1
  differential-flooring). H2/H3/M3 fixed in the scorer; H1/M1/M2/L1 + a citation synced into
  the doc.

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress):** the **pure scorer is DONE**.
  The **telemetry wrapper is the remaining half** and is REQUIRED before freeze (drl-expert
  H1): §1 validity gates, §0 provenance block, burn-in `W` discard, §8B floored-update
  exclusion (a SCORED precondition — the M1 route), §8E `Var(returns)`-volatility covariate,
  G3/G4 computation, the tier-gated packet CLI, and the §9 diagnostic telemetry scalars
  (`value_main_target_scale`, `cf_value_target_scale` — touches `ppo_agent.py`, ON-leg-only,
  needs the byte-identity check).
- **Nothing else active in code.**

## Facts the next session must not relitigate
- The Stage-2 HRA ON-leg is **code-complete** (cf value + reward wired; per-stream GAE;
  per-stream EV) — do NOT rebuild it or re-fear "cf trained on zeros" (guard exists).
- The MAJOR-1 rubric's **core is drl-verified**: the exact Wilcoxon (vs independent DP ref),
  the `pre_norm_advantage_std² = (1−ev_sum)·Var(returns)` identity, and §8a/b comparability.
- The **two rejected external recs are code-grounded**, not preference: advantage-std *level*
  is algebraically entangled with the ev_sum floor; `EV_main_OFF` does not exist (ON-only).
- The gate is **pure-scorer-only so far**; it is **NOT freezable** until the wrapper exists +
  is tested. Do not run the A/B before freeze (owner ruling).
- MAJOR-3 provenance: the Stage-0 value-free gate LICENSES the A/B; it is NOT HRA evidence.

## Open questions / blocked-on-owner
- **Nothing blocked-on-owner.** Authority grant CONFIRMED this session (holds as written,
  reviewed 2026-07-05).
- Standing placeholders: north-star / rent / host-acc-floor TARGETs (metrics.md) still
  owner-set; `Δparam_max` (G2 ceiling) to be frozen before the ON run.
- Nothing escalated this session (all work was reversible, inward-facing analysis code).

## Last checkpoint did (checkpoint #22)
- **PDR-0037** — Stage-2 MAJOR-1 acceptance harness designed + scorer built + review-hardened;
  recorded the NOT-yet-freezable boundary and the reversal triggers (ev_sum-below-δ FAILS
  Stage-2; M1 differential-flooring becomes load-bearing if `ev_return_variance` sits near 1.0).
- metrics.md: added the **Stage-2 MAJOR-1 acceptance gate** as a pre-registered (not-yet-read)
  input metric. No new empirical readings (no run this session). No horizon change (EV-stab
  stays Now). Stage-2 task stays in_progress (claimed).
- Delivery committed on `feat/ev-stab-stage2-hra@2fad1fc4`; workspace checkpoint #22 on
  LOCAL main (NOT pushed — /product-checkpoint never pushes; origin still at `e2753d8b`,
  1 behind local main incl. #21).

## Next session, start here
**Build the telemetry wrapper** (the remaining half of the MAJOR-1 gate), TDD throughout:
§1 validity + §0 provenance + burn-in `W` + §8B floored-exclusion + §8E covariate + G3/G4 +
packet CLI (mirror `scripts/proof_packet.py` — duckdb over Karn `ppo_updates`) + the §9
diagnostic scalars (byte-identity check on `ppo_agent.py`). Then **freeze the gate doc** →
then the paired fresh-init HRA ON/OFF A/B on the GPU host (owner launches). Pointers:
gate doc `docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`; scorer
`src/esper/simic/telemetry/stage2_acceptance.py`; memory `ev-stab-stage2-impl-state`.
