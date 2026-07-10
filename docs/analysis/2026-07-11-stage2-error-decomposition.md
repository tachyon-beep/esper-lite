# Stage-2 post-verdict analysis: main/cf error decomposition + seed-44 contrast

**Date:** 2026-07-11 · task `esper-lite-c004d67fb9` · feeds the Path A/B/C DECIDE
**Scope fence:** exploratory post-verdict read of the completed A/B arms. PDR-0059's
REJECT is final for objective-A (§7); nothing here reopens the screen.
**Data:** the ten `packet_root` arms (hashes in `2026-07-11-stage2-rescore-manifest.txt`);
floored rows excluded per §8B (2/2000).

## Method and caveats (state before results)

Per ON update: `Var(e_x) = (1 − ev_x) · denom_x` with `denom_x = max(VarG_x, 1.0)`,
and `2Cov(e_main, e_cf) = Var(e_sum) − Var(e_main) − Var(e_cf)`.

- Per-stream instantaneous `Var(G)` is **not emitted** (packet §8B "per-stream
  blindness"); the per-stream denominators use the RUNNING normalizer scales
  (`value_main_target_scale`², `cf_value_target_scale`²). The total stream has both
  running and instantaneous values; their ratio (median 0.56–1.74 by quartile/seed)
  bounds the per-stream lag error at roughly ±50–70% early, ±10–30% late.
- Consequence: `corr_e` values outside [−1, 1] appear (down to −1.5) — a
  reconstruction-error indicator, not physics. **Signs and cross-seed patterns are
  robust; magnitudes are rough.** The exact version needs the post-freeze telemetry
  addition (per-stream EV-convention `Var(G)`, normalizer-vs-instantaneous std + lag).
- `return_var_cf_share` is a covariance share on the value-free REWARD-return
  decomposition; `cov_rcf_return_share` is the same shape on the actual GAE targets
  (V_cf-contaminated). Neither is a per-stream target variance; they were not used as
  denominators.
- Validation: reconstructed full-window Δ_A per seed = −0.1002/−0.0860/−0.1008/
  +0.1737/−0.0920 vs packet −0.1022/−0.0876/−0.0978/+0.1742/−0.0928 — the pipeline
  reproduces the verdict inputs.

## Finding 1 — the deficit is late-training and seed-heterogeneous, not a warmup tax

Median Δ_A (ON ev_sum − OFF ev), burn-in discarded, split by window:

| seed | full (≈packet) | 1st half | 2nd half | last 50 |
|---|---|---|---|---|
| 41 | −0.1002 | −0.0251 | −0.0962 | −0.1016 |
| 42 | −0.0860 | −0.0241 | −0.0272 | −0.0106 |
| 43 | −0.1008 | −0.0181 | −0.1182 | −0.0998 |
| 44 | +0.1737 | −0.0004 | +0.1483 | −0.0001 |
| 45 | −0.0920 | −0.0305 | −0.0356 | −0.0280 |

First-half deltas are uniformly small (all within δ = 0.0503). The FAIL is driven by
the second half: **seeds 41/43 diverge to a persistent ~−0.10 end-state deficit**,
42/45 end mildly worse (−0.01/−0.03), 44 is transiently better then converges to
parity. A "delayed liftoff, catches up" reading was considered and is **falsified**
by the last-50 column.

## Finding 2 — Var(e_cf) dominates; implicit error cancellation partially rescues the sum

Quartile medians (all seeds, pattern identical): `Ve_main` is small and shrinking
(≈3–18 across training; ev_main reaches 0.58–0.77 by Q3–Q4 — **the main head fits its
own target well**). `Ve_cf` grows to 280–450 by Q4 and EXCEEDS `Ve_sum` from Q2
onward; the gap is closed by a consistently **negative** error covariance (2Cov ≈ −50
to −180 mid/late). ev_cf whole-run: 0.126–0.264 — the cf head never fits well.

Four-case discrimination:
- **Primary case: `Var(e_cf)` dominates — the cf target is effectively unlearnable at
  its current scale/non-stationarity** (scale drifts ~2–4 → ~24–28 within every run).
- Positive-covariance "correlated trunk mistakes": **absent** — covariance is
  negative (beneficial) everywhere except one early window.
- Main-head interference from the cf loss: **not supported** — ev_main is healthy and
  `Ve_main` shrinks through training.
- Loss-weighting/denorm inconsistency as primary cause: **not supported** — the
  components are not "individually fine"; cf is individually poor.

Nuance on the missing total-value consistency loss (confirmed structural,
`ppo_update.py:454,464`): cancellation DID emerge implicitly (negative Cov), so the
absence of the consistency term did not prevent cancellation — but it left the
cancellation unoptimized and unreliable; seeds 41/43 are the cases where it fails to
rescue the sum late in training.

## Finding 3 — seed-44 contrast: early cf-scale stabilization is the separator

| seed | cf scale Q1 | Q4 | slope | cf loss Q1 | ev_cf (run) | op_lf | slot_lf |
|---|---|---|---|---|---|---|---|
| 41 | 2.24 | 25.3 | 11.3× | 2.418 | 0.126 | 0.391 | 0.083 |
| 42 | 4.10 | 27.8 | 6.8× | 0.804 | 0.137 | 0.398 | 0.083 |
| 43 | 4.19 | 27.1 | 6.5× | 0.825 | 0.171 | 0.403 | 0.083 |
| **44** | **7.42** | 27.1 | **3.7×** | **0.293** | **0.264** | 0.401 | 0.083 |
| 45 | 3.98 | 23.7 | 6.0× | 1.883 | 0.253 | 0.401 | 0.083 |

Seed 44 — the only Δ_A-positive seed — **started with its cf target scale already
high** (7.4 vs 2.2–4.2), so its normalizer chased ~3.7× drift instead of 6–11×; it
has by far the lowest early cf loss and the best whole-run ev_cf. Decision density is
IDENTICAL across seeds (op/slot learnable fractions), ruling out a behavioral-regime
explanation. This is observational (n=1), but it is exactly the signature Path A
predicts: **when the cf target scale is stable early, the decomposition behaves.**

Side-note for the entropy-floor audit (PDR-0057): `slot` learnable fraction is 0.083
of all steps on every seed — slot choice-bearing steps are rare; the `d − floor`
feasibility check for slot (floor 0.15) is a live concern, pending the
availability-denominator correction.

## Implications for the DECIDE (advisory, not a decision)

- **Path A's premise gains observational support** (seed-44) and a concrete
  prerequisite list: stabilize the cf target scale (full PopArt with output
  preservation, or reference-scale whitening) **and** add an explicit total-value
  consistency loss (the emergent cancellation is load-bearing and currently
  unoptimized) **and** handle the early-scale transient. Head-size/loss-weight
  tweaks alone are not motivated by any finding here.
- **Path B (true de-shaping)** remains attractive on the same evidence read the other
  way: the cf head never fits (ev_cf ≤ 0.26), so keeping its error out of the actor's
  baseline entirely attacks the problem at the root — at the cost of a new-objective
  bias analysis.
- The 600-round diagnostic question ("does the cf scale plateau?") is sharpened:
  seeds differ mainly in WHEN the scale stabilizes, so the diag should read scale
  slope and EV recovery jointly.
