# Committed-Shapley OFF/ON A/B (n=5) — scoring verdict (2026-07-05)

**Verdict: NULL — NOT INFORMATIVE, per the pre-registered asymmetric null.**
**Mechanism validity: ALL GREEN** (criteria i/ii pass, clamp exercised, zero
deadband violations, all guards clear, safety gates pass, no hacking signature).
The term is neither banked nor condemned; per prereg it "does NOT count against
the term until re-run with materially more accrued paid events."

**Scored strictly per:** `2026-07-03-shapley-ab-scoring-preregistration.md` +
`2026-07-04-...-ADDENDUM-gate-fossil-admission.md`. Preconditions held: addendum
§3 OFF bitwise spot-check = the PDR-0025 no-op gate (PASSED); §4 episode-idx
decode RETIRED for these runs (verified correct live); all five ON runs
integrity-clean (2400/2400 episodes, zero tracebacks, every TOPUP finite and
tau-stamped: 515/448/462/543/453 events for s41–45).

## 1. Effect-size floor — FAIL (and why that is the pre-registered null, not a defeat)

Estimator: episode J = SUM(j_per_param) per episode (seed_residency rows),
corr against episode_reward, per seed. **Estimator validated:** OFF-arm mean
corr = 0.214 vs the independently-banked phase-0 baseline 0.212 ± 0.046.

| seed | corr OFF | corr ON | Δcorr |
|---|---|---|---|
| 41 | +0.167 | +0.192 | +0.026 |
| 42 | +0.162 | +0.221 | +0.059 |
| 43 | +0.294 | +0.178 | −0.116 |
| 44 | +0.170 | +0.168 | −0.003 |
| 45 | +0.279 | +0.226 | −0.052 |

Mean Δcorr = **−0.017** (median −0.003), positive seeds **2/5**. Floor (≥ +0.10):
FAIL. Sign consistency (≥ 4/5): FAIL.

**The mechanical account.** Paid events: 59 across 12,000 ON episodes = **0.49%**
of episodes received any top-up. k≥2 coalitions: 132/12,000 = **1.10%** — the
prereg's pre-registered payment floor was 1.1%, hit exactly. A reward term that
touches 0.5% of episodes cannot move an episode-level correlation computed over
2400 episodes. The paired-delta spread (SD 0.066) equals the pure-noise
prediction for two independent draws of a ±0.046 statistic (√2·0.046 = 0.065):
the deltas are indistinguishable from noise, exactly as the asymmetric-null
clause anticipated. This is the YELLOW-band coverage caveat materializing, not
new information about the term's correctness.

## 2. Criteria (i)–(v)

| # | Result | Evidence |
|---|---|---|
| i | **PASS** | r0c0 paid in 56/126 k≥2 memberships (44%) — enabling stems are paid at par with r0c1 (50%) / r0c2 (39%) |
| ii | **PASS** | 0/2554 slot-events paid at φ−c_paid ≤ τ (structural enforcement verified) |
| iii | **clamp leg PASS; efficiency leg NOT ADJUDICABLE** | G-clamp bound 4×, ncap 27×, std_floor 0× (ceiling exercised). Paired efficiency: ON>OFF 2/5, median paired change −3.4%, per-seed swings −30%…+54% — noise-dominated (floor unbanked), and no behavioral change is expected at 0.5% payment coverage |
| iv | **PASS (no worsening beyond noise; no spike)** | fossilize/ep ratios ON/OFF: 0.99, 0.98, 1.00, 1.27, 0.97 — aggregate flat; G2 (>3×) clear by an order of magnitude |
| v | **NOT INFORMATIVE (per asymmetric null)** | Behavioral proxy: FOSSILIZE op share identical across arms (0.13–0.16% both). The S8 op-head probe was not part of the A/B telemetry; with 0.5% payment coverage a null here is uninformative by the same arithmetic. Criterion (v) falsifies only on an INFORMATIVE read ⇒ **no GATE-2 reopen** (PDR-0011 trigger not met) |

## 3. Safety gates and guards — all clear

- Final-acc median paired Δ: {+0.00, −0.12, −0.24, +0.06, −0.24} pp; median
  **−0.12 pp** ≥ −0.3 pp gate → PASS.
- Param ratio: flat (median paired Δ +0.014).
- **G1:** zero trips (all 10 runs completed 200/200 batches — an abort would have truncated).
- **G2:** clear (max ratio 1.27 vs 3× line).
- **G3:** aggregate clear (median −3.4% vs −10% line); per-seed values ±30–54%
  are noise-dominated and the paired noise floor for this read is **UNBANKED**
  (stated per prereg; blocks nothing — no positive is being banked).
- **G4:** paid events/run {10, 8, 13, 12, 16} — trigger (>36) not fired.
- Guard channels ON vs OFF (per-run means): hacking-suspected 43.2 vs 39.8
  (within seed spread), value-collapse 7.4 vs 9.6, **governor rollbacks 12.4 vs
  7.4** (ON s43 = 21 is the outlier; ≤0.9% of episodes; no pre-registered gate —
  logged as a secondary observation for any future ON leg).

## 4. G4 paid-event review — the mid-run "47% flag" RESOLVED as a reference-class error

Observed **P(pay | k≥2) = 59/132 = 44.7%** vs the transient-factorial design
prior **45.4%** (F2 calibration, computed from real transient v(S) tables). The
mid-run flag compared against ~1%, which is the **placebo null-player**
exceedance rate — the wrong reference class for real coalitions. The paid
population is not placebo-shaped: paid excesses run to p90 +9.5 pp (placebo
noise is ~0.1–0.3 pp), paid magnitude p50 = 5.08 (cap region). The mechanism
paid **exactly as calibrated**. PDR-0025's reversal trigger ("paid population
dominated by null-player-like pairs") is **not met**.

## 5. τ recalibration — labeling gap recorded (matters only for n=10 magnitude claims)

The prereg's read (P99 of φ−c_paid over **null-player** slots) is not
computable from ON telemetry alone: non-paying slots are censored at τ by
construction (top_up=0 ⟺ excess ≤ τ), and the uncensored population is a
synergy mixture whose P99 (+17.0 pp) upper-bounds nothing useful. A proper
null-player floor needs an injected-placebo leg (PIN-E style). Addendum §5
stands: τ=0.28 is an ADD-placebo floor, GATE-transfer unvalidated; 62 GATE
slot-events participated (37 non-paying) but per §5(c) any **magnitude** claim
involving gate credit stays unlicensed until a GATE-placebo PIN-E leg runs.
**No magnitude claim is made here** (direction was null), so nothing is blocked
today; this is a mandatory precondition on the n=10 path only.

## 6. Confound gate — moot

28/59 paid events had std_used < 1.0 (p50 1.02). The gate conditions on "the
direction effect is carried by…"; there is no direction effect to confound.
Recorded for any future leg.

## 7. Descriptives

- Pooled k-dist: {1: 2289, 2: 131, 3: 1} (first k=3 coalition observed, s45).
- Paid credit split (addendum §2): GATE 19%, non-GATE 81% — no gate-carried
  distortion (and nothing to distort).
- Paid magnitudes: p50 5.08, mean 5.90, max 10.0 per event — chunky (≈15–30% of
  a typical ~35-point episode reward) but touching only ~12 episodes/run.
  **Design insight: magnitude is not the problem; frequency is.**
- J/ep and reward means: flat across arms within seed noise.

## 8. What the prereg prescribes next

A null n=5 (i) does **not** fire PDR-0018 §7's pre-fossil follow-on, (ii) does
**not** count against the term, and (iii) licenses a re-run **only with
materially more accrued paid events**. Raising accrued paid events means
raising k≥2 co-fossilization frequency (currently 1.1% of episodes) or run
count/length — a **design/scope decision for the owner**, not a scoring output.
Decision recorded in PDR-0026.
