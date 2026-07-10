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

---

## Addendum (2026-07-11, post-review claim calibration)

A second external review (owner-relayed) calibrated the first pass. Adopted demotions,
one correction of our own framing, and two additions the review missed.

### Claim status after calibration

| Finding | Status |
|---|---|
| Objective-A REJECT | **Banked** (unchanged, PDR-0059) |
| Deficit not merely warm-up; late divergence on s41/s43 | **Banked** (exact EV inputs; reproduces packet) |
| Main target easier; main head learns it well (ev_main 0.68–0.77 Q4) | **Banked** |
| cf target large and strongly non-stationary (3.7–11.3× scale growth) | **Banked** |
| cf stream is the dominant residual burden | **Likely — qualitative**; quoted Var(e) magnitudes are running-scale approximations (lag band ±50–70% early) |
| Component errors beneficially anticorrelated | **Suggestive, NOT banked** — `corr_e` beyond ±1 proves the reconstruction is not calibrated for covariance; sign can flip in marginal cells (s42 Q4 −6.5, s45 Q3 −9.9, s45 Q1 +4.6) and the lag errors are correlated across cells, so cross-cell sign consistency is weaker evidence than it looks |
| No shared-trunk interference | **Softened**: no GROSS interference evident; interference relative to a main-only critic is UNIDENTIFIED (that arm does not exist) |
| Seed-44 = early-scale-stabilization support for Path A | **Demoted to n=1 observational association**: its advantage vanishes in the last 50 (ends at parity); its Q1 lag ratio is the LARGEST (1.735), so the high early running scale does not prove a well-matched normalizer; equal learnable fractions do not establish equal behaviour |
| "cf effectively unlearnable" | **Corrected**: partially learnable (ev_cf reaches 0.44–0.54 by Q4) — the scale grows faster than relative error falls, so the ABSOLUTE burden keeps expanding |

### Correction — the missing total-value loss, stated precisely

At the population optimum, per-component MSE heads ARE aligned with the total target
(`E[G_m|s] + E[G_cf|s] = E[G|s]`) — the first pass's "two proxies" framing overstated
the conceptual mismatch. The accurate durable statement: **Objective A did not
directly optimise total prediction error; its component objectives align with the
total only at their ideal optimum, and in this finite, shared-trunk, non-stationary,
scale-mismatched regime the total fit was left unprotected.** Still a strong reason
any Path-A′ successor carries a direct total objective; no longer an in-principle
indictment.

### Addition 1 — bootstrap feedback makes the cf target partly ENDOGENOUS

Per-stream GAE constructs each stream's target from its own head:
`returns_cf = A_cf + V_cf` (`rollout_buffer.py:678`; gate doc §0 declares this, and
`cov_rcf_return_share` is documented as "V_cf-contaminated" for the same reason). A
poorly-fit V_cf therefore pollutes its own future targets — some of the observed cf
non-stationarity is self-inflicted, not just reward-side scale drift. Consequence for
the DECIDE: a primary-`V_total` design (A′) or main-primary design (B) changes the
bootstrap TOPOLOGY, not merely the loss — the strongest mechanical reason the rejected
architecture cannot be rescued by head-size/loss-weight tuning.

### Addition 2 — sufficient-statistic telemetry supersedes the field list

Emit per update the means and full second-moment (Gram) matrix of
`x = (V_main, V_cf, G_main, G_cf)` on the raw scale (~14 scalars). Everything both
reviews asked for becomes computable offline on ANY window, forever: exact
Var(e_main)/Var(e_cf)/Cov(e_main,e_cf)/Var(e_sum), the identity assertion, normalizer
lag (against running scales), AND the held-out affine calibration-rescue test
(`Ĝ = a·V_main + b·V_cf + c`, fit early / test late — Path A′ vs B discriminator).
Note: that calibration test CANNOT run on the existing arms (no per-sample or moment
data persisted); it is forward-looking emission for the next ON-leg run. On the
600-round OFF diagnostic the matrix degenerates to (V, G) — still worth emitting for
normalizer-lag and scale-slope reads.

### Decision-rule capture (advisory, for the owner DECIDE)

- **Path A′** (primary V_total directly trained + stabilised auxiliary component
  heads + consistency term — NOT a rerun of Objective A) if: exact cf residual is
  manageable after scale stabilisation; normalizer lag explains much of the deficit;
  held-out recalibration rescues total EV; ev_cf keeps improving as scale slope falls.
- **Path B** (main-primary de-shaping; cf via auxiliary/control-variate; new objective,
  fresh PDR covering bias/optimum/exploration/safety) if: cf residual stays dominant
  after stabilisation; calibration cannot rescue the sum; cf target stays severely
  non-stationary while main stays learnable.
- **Path C** (information-first) if: both streams hit low ceilings even after
  stabilisation and offline Obs-V3 probes cannot beat the recurrent critic. Current
  evidence does NOT point here for this failure (ev_main is high).
- Current lean: **slightly toward Path B**, with A′ credible — held for the owner
  DECIDE after the schedule-correct 600-round diagnostic.
