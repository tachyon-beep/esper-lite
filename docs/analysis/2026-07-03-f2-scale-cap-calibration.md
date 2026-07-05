# F2 calibration — committed-Shapley knob proposal (2026-07-03)

**Gate leg:** criterion 2 of esper-lite-f22a1d48a7 (scale/cap/normalized_cap/std_floor).
**Status:** **FROZEN 2026-07-03** — drl-expert review completed same day: all four knobs
ACCEPTED (std_floor ACCEPT-WITH-CONDITIONS); the review's blockers/majors attach to the
A/B design and enablement, not the knob values (see §Review outcome). Enabling
`shapley_synergy_scale > 0` remains owner-gated regardless.
**Method:** offline Welford reconstruction of the reward-normalizer std trajectory from the
logged reward stream (validated: order-sensitivity across env orderings = 0.000%), plus
empirical gap/G priors from the 14,199 logged multi-slot CF factorials
(scripts: `scratchpad/f2_calibration/`). n=5 control runs, read-only.

## Data

**A — the divisor (running std at episode-terminal steps).** Pooled over 12,000 terminal
steps × 5 runs: min **0.345**, p1 0.374, p50 1.055, converged **1.59–1.86**. The divisor
the credit meets varies ~5.4× across a run (early episodes ≈ 0.35, converged ≈ 1.7);
epsilon (1e-8) never binds. Reconstruction is ordering-robust because ≥1,800 samples
accumulate before the first terminal step.

**B — payment prior (transient-seed factorial proxy; declared caveat: fossil-coalition
interactions are NOT in OFF telemetry — these are the best available prior).** For k=2 the
mechanism reduces exactly to `gap = max(0, I/2 − tau)` (both slots), I = the factorial
interaction. At tau = 0.28: **P(pay) = 45.4%** of k=2 events; positive gaps p50 **1.58 pp**,
p90 6.6, p99 13.4, max 22 pp. G-clamp binds 11.5% of paying events (G p50 among paying =
13.1 pp — the clamp is a genuine tail guard, not the main bound). k=3 (751 matrices, exact
Shapley via the shipped module): P(any slot pays) = 61.7%, positive gaps p50/p99 =
1.84/10.7 pp, clamp binds 12.1%.

**C — buffer-unit context.** At converged std: fossilize package ≈ 0.83–0.98 units,
terminal bonus ≈ 1.3–1.6, tau ≈ 0.16, a p90 gap ≈ 3.6–4.2 units.

## Proposed knobs (rationale per knob)

| Knob | Proposal | Rationale |
|---|---|---|
| `shapley_synergy_scale` | **1.0** | The natural unit: credit stays in the same pp currency as the dense attribution; all bounding is done by the guards below. Any other value re-scales a quantity that is already commensurate. |
| `shapley_synergy_cap` | **5.0 pp** | Per-seed raw ceiling ≈ p85–p90 of the positive-gap prior. The p99 tail (13–22 pp) is exactly the population most likely to be single-epoch CF noise or entrenchment; it should not pay in full. Typical payments (p50 1.6 pp) pass untouched. |
| `shapley_synergy_std_floor` | **0.25** | SAFETY-ONLY floor, ~28% below the observed terminal-std minimum (0.345 over 12,000 terminal steps). Principle: the divisor exists to put the credit in the same units as its neighboring rewards, which divide by the same std — flooring above the observed range would make the credit ~3× weaker *relative to its neighbors* exactly in the early phase. The floor binds only on out-of-distribution ON-run behavior. (Alternative considered and rejected: a "stationarity floor" ≈ 1.0 to damp the 5× early amplification — rejected because neighbors are equally amplified, so unit consistency wins; flagged for drl review.) |
| `shapley_synergy_normalized_cap` | **3.0** | PRIMARY bound (buffer units). ≈ 2× the largest existing terminal package (~1.5 units), 30% of the ±10 normalizer clip. Arithmetic worst case without it: cap/std_floor = 5/0.25 = 20 units — twice the clip; with it, the credit can dominate a step but never the return. Also propose the missing config validation: `normalized_cap ≤ clip (10.0)` (criterion-2 note; validation-only change, for the review to confirm). |

Expected ON-run payment profile at these knobs, at OFF-arm behavior rates: ~26 k≥2 fossil
episodes per 200-episode run × ~45% paying prior ≈ **~12 paid events/run**, typical
credit_buf ≈ 0.9–1.6 units (capped 3.0), delivered as concentrated advantage bumps at the
FOSSILIZE decisions (retro-write; GAE linearity). Criterion (v) — does that move the
FOSSILIZE head — is precisely what the A/B tests.

## ⚠ Two findings that must reach the owner (both banked on the gate)

**1. Scope correction — PDR-0018 §6 is INVERTED.** Verified in code
(`committed_shapley.py:143-144`): for k=1, φ ≡ c_paid, so `gap = max(0, −tau) = 0` — a
lone fossil pays **structurally zero**. The term pays ONLY synergy excess in k≥2 episodes.
PDR-0018 §6 ("mostly a terminal commitment-credit experiment, not synergy credit") states
the opposite; the truth is: **it is a synergy-credit experiment whose payments are gated on
multi-fossil episodes** — k≥2 = 1.1% of episodes at OFF behavior, not the 18.5–22%
"bootstrap floor" reported at ratification. The false-negative risk is sharper than ruled
on. To be superseded by PDR at next checkpoint.

**2. The n=5 primary target has a unit problem.** "Median paired Δ`fossilize_payable_J` ≥
2×tau (+0.56 pp)" is unreachable as a per-episode median: 98.9% of episodes pay zero
structurally (median ≡ 0), and even the per-episode *mean* at OFF-behavior rates is
~0.02 pp (12 events × ~2 pp ÷ 2,400 episodes). The target only makes sense as (a) a
per-run aggregate over paying episodes, (b) conditional on payment, or (c) re-based on the
design-doc criteria (i)–(v), which do not have this problem (they test who gets credited
and whether efficiency rises, not a mass threshold). **Owner must re-specify before the
A/B is scored — surfaced now per PDR-0018's conflict rule, not at scoring time.**

## Review outcome (drl-expert, 2026-07-03 — mandated pre-freeze review)

**Per-knob verdicts:** scale=1.0 ACCEPT · cap=5.0pp ACCEPT (phase-invariant pp-domain tail
filter, complementary to normalized_cap which binds first below std≈1.66) · std_floor=0.25
ACCEPT-WITH-CONDITIONS (unit-consistency argument upheld and *strengthened*: the credit
shares BOTH the reward-std divisor and the batch-advantage z-score with every neighbor, so
a private stationarity floor would desynchronize it end-to-end; 0.5 recorded as the
risk-averse fallback, 1.0 as the ON-recalibration lever) · normalized_cap=3.0 ACCEPT
(**it is the SOLE operative bound** — the ±10 clip is not in this channel's path; the
`≤ clip` validation is sanity-only).

**Conditions attached to the freeze (required before the A/B is scored/enabled):**
1. **[BLOCKER — owner re-spec, recommendation (c)]** Replace the mis-specified
   `Δ fossilize_payable_J ≥ 2×tau` overlay: primary = design criteria (i)–(v); the
   YELLOW large-effect floor = the already-ratified **episode-level Δcorr(reward,J) ≥
   +0.10**; sign-consistency re-anchored to Δcorr or a criterion-(v) head-probability
   shift. Option (a) per-run paid mass REJECTED (Goodhart-aligned with over-fossilization);
   option (b) conditional magnitude demoted to descriptive-only.
2. **[MAJOR — owner pre-registration]** Asymmetric null: a positive n=5 is informative; a
   null is NOT (≈12 paid events/run) and must neither fire ruling §7's pre-fossil
   follow-on nor count against the term until re-run with more accrued paid events.
3. **[MAJOR — enablement precondition]** The fossilize-count guard (gate criterion 3) must
   be wired and the §3 safety gates HARD before `scale > 0`: the knobs bound per-event
   magnitude, not payment frequency — over-fossilization is the one residual Goodhart
   surface (junk pairs are structurally unpayable: paying requires measured
   superadditivity I > 2τ on a real fused-val forward pass).
4. **[MAJOR/MINOR — telemetry]** Per-paid-event `(credit_buf, met_std)` logging +
   normalized_cap bind-rate logging + a scoring-time confound gate (if the direction
   effect is carried by std<1.0 early-amplified events, re-confirm at std_floor ≥ 1.0
   before banking). These are what make std_floor=0.25 defensible.

**Learnability verdict:** MARGINAL-but-plausible for a direction read; each credit survives
normalization at ≈1.5–3σ of positive advantage on non-sparse heads (op+slot; the per-head
adv-norm lever is orthogonal). The γλ=0.945 retro-write tail also lifts the ~12–25 steps
into t_f (total mass ≈18× credit) — beneficial, but criterion-(v) reads should expect the
approach ops to shift too.

**Substrate correction (this memo, post-review):** the review's "entropy-degenerate
substrate" premise partially carries a STALE read — PDR-0006 (2026-07-01) reclassified the
entropy collapse as a telemetry artifact; decision-step conditional entropy is HEALTHY
(op ≈0.88, slot 0.30–0.70; metrics.md guardrail row). The *power* limitation (~12 paid
events/run) stands on its own; the *substrate* concern is weaker than stated, which
reduces the pressure to sequence entropy hygiene before the A/B. The occurrence-probe
doc's "entropy-degenerate population" caveat inherits the same staleness.

## Caveats
- The gap prior is transient-seed factorials; fossil-coalition interactions may differ
  (the §5.3 scope box caveat carries). First ON calibration run re-reads the actual paid
  distribution — and the PDR-0017 reversal trigger (ON P99 > 2× placebo tau) applies.
- std reconstruction assumes the normalizer consumes rewards in (episode, epoch, env)
  order; the 0.000% ordering sensitivity makes this immaterial.
- These knobs configure the ON arm of the A/B only; `scale` stays 0.0 everywhere until the
  owner enables per the gate.
