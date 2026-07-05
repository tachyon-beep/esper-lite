# Run sheet — n=5 J-read (causal contribution: r0c0 freeloader vs enabling stem)

**Status:** PRE-REGISTERED · Phase 0 COMPLETE · **Phase 1 LAUNCHED 2026-07-01 (in flight, ~13h — control bg `bq3r29zsq` / suppress bg `bmbp8eg0w`)** · **Date:** 2026-07-01
**Decision record:** PDR-0006 (J-reframe), PDR-0007 (commission n=5) · **Result doc:** TBD
**Reviewed_by:** drl-expert (diagnosis workflow wrpmqysf6), advisor (J-reframe + sequencing)

## Objective
Extend the causal-contribution read from n=3 to **n=5** (seeds 41–45) and measure the fork
discriminator **ΔJ / Δ(acc-per-param)** at the seed level, to BEGIN banking the direction the
n=3 pilot leans: **r0c0 is an efficiency-enabling stem (b), not a freeloader (a)**.

## Why J, not accuracy (PDR-0006)
On accuracy both arms tie (~+7.7pp) — that is why "replaceable" kept oscillating. The
discriminator is committed counterfactual gain per param. Pilot (n=3): control 12–20 vs
suppress 7–9 pp/M-param; suppressing r0c0 ~halves efficiency (median Δeff −7.0; all 3 negative).

## Estimand (PRIMARY — pre-registered)
`paired Δeff = efficiency_suppress − efficiency_control`, per seed, where
`efficiency = (all-on − all-off late-run counterfactual accuracy) / (Σ committed params / 1e6)`.
**Inference unit = SEED** (resample the ≥5 seeds, NOT the 12 vec-envs — the pre-registered
invalidator). Seed-level bootstrap 95% CI (B=10000).

## Arms & seeds
`control` (RNG-split ON, no mask) + `suppress_slot_on` (r0c0 un-committable), seeds 41–45.
- **41–43: ALREADY RUN**, gates passed — preserved at `telemetry/causal_r1_n5/`. REUSE.
- **44–45: TO RUN** (4 runs). Config `config-3slot-3seed-suppress-slot-r0c0.json`, `gpu_preload`,
  committed harness + NaN-safe patch (68fca06d/a84f9a78). ≤2 concurrent (1/card), ~6.6h/run ⇒ **~13h wall**.

## GATES (ALL must pass before banking — `scripts/causal_contribution_j_analyze.py`)
per seed: constant controller stride (control, on); offset-free parity (control==on draw-count
+ state hash); `on` germinates ZERO r0c0; **finiteness-gate trips == 0**; **decision-step
slot-entropy health** (min `head_slot_entropy / head_slot_learnable_fraction` > 0.1 — the
choice-conditioned signal, PDR-0006).

## DECISION RULE (pre-registered)
| Δeff seed-level bootstrap CI | Reading | Reward implication |
|------------------------------|---------|--------------------|
| excludes 0, **NEGATIVE** | **(b) efficiency-enabling stem** | credit enabling; the LOO/freeloader penalty is the WRONG fix |
| **includes 0** | (c) fungible | no slot-specific penalty; reward the efficiency frontier |
| excludes 0, **POSITIVE** | (a) freeloader | penalize (n=3 makes this unlikely) |
| too wide to separate | **escalate to n=10** | morphogenesis floor for a low-margin claim |

n=5 BEGINS causal evidence; **n=10 is the floor** (zero margin). Given the pilot's consistency
n=5 may suffice; the n=10 escalation is pre-registered either way.

## Commands
**Phase 1 (GPU — owner go).** Two processes, one per card, each looping seeds 44,45:
```
uv run python scripts/causal_contribution_r1_pilot.py --arm control \
  --device cuda:0 --telemetry-dir telemetry/causal_r1_n5 --seeds 44,45 --gpu-preload
uv run python scripts/causal_contribution_r1_pilot.py --arm suppress_slot_on \
  --device cuda:1 --telemetry-dir telemetry/causal_r1_n5 --seeds 44,45 --gpu-preload
```
**Phase 2 (analysis — no GPU):**
```
uv run python scripts/causal_contribution_j_analyze.py telemetry/causal_r1_n5 41,42,43,44,45
```

## Caveats (held honestly)
1. Population is healthy **exploration** but a 200-ep policy — not fully converged. The read is
   on this regime; "the well-trained controller's verdict" is a later question.
2. The earlier "collapse" was a MEASUREMENT artifact (PDR-0006); decision-step entropy is healthy.
3. Estimand SCOPE: **RATIFIED total-system** (PDR-0004 accepted 2026-07-01). Goal-1 = r0c0-specific
   system dependence; the mechanistic-enabling claim is out of scope (deferred). No placebo/DUMMY.

## Owner-gated
- **Phase 1 GPU spend (~13h)** — explicit go.
- **PDR-0004 estimand-scope** decision (independent; can settle in parallel).
