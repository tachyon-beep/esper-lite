# ON-leg Gram diagnostic (n=2) — PDR-0064 read (final, exact-decomposition)

Date: 2026-07-13 · Runs: `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/`
Commit: `92d998bd` · Completion: both seeds 600/600, rc=0, 0 crashes.
Reviewed: advisor + 2 external peers (ChatGPT, claudemain), 2026-07-13 — all integrated.
Diagnostic (not a gate); rejected HRA re-used as substrate. THIRD-PASS: supersedes the
"B favoured" (9dbf078f) and "light-A′ refuted" (a87ba291) verdicts — both were over-reads.

## Headline: further critic work is not justified until the cf term's downstream value is established

Exact residual decomposition (matched pooled moments) largely dissolves the A′/B question:
**no critic architecture in this family beats ~0.62–0.64 total EV, because ~all of the
unexplained variance lives in the cf reward stream and is not predictable from state at
this critic's capacity.** The main head is near-perfect; heavy-A′'s primary head (a direct
`V_total` critic) already exists as the OFF run and only reaches ~0.64.

Framing (per external review, GPT last-say): this does NOT erase the prior critic findings
— the op-conditioned-baseline fix (P0-1) and the Objective-A rejection had real value. The
correct statement is **"further critic optimization is not justified UNTIL the downstream
value of the cf term is established,"** not "the critic was never the problem." The lever
to test first is the cf REWARD, not the critic topology.

## Exact residual decomposition (rounds 400–600, count-weighted pooled)

`Var(e_sum) = Var(e_main) + Var(e_cf) + 2·Cov(e_main,e_cf)`, computed from the Gram moments:

| seed | Var(e_main) | Var(e_cf) | Cov | Var(e_sum) | Var(G_total) | EV_sum (pooled) | corr(e_m,e_cf) | cf share of resid |
|------|-------------|-----------|-----|------------|--------------|------------------|----------------|-------------------|
| 41 | **0.4** | 228.7 | −3.0 | 223.0 | 588.4 | 0.621 | −0.32 | 1.03 |
| 42 | **0.6** | 237.1 | −2.9 | 231.9 | 596.3 | 0.611 | −0.24 | 1.02 |

- **The main head is near-perfect** (`Var(e_main)` ≈ 0). The ENTIRE total residual is the
  cf stream (cf share ~1.0); component errors mildly ANTI-correlate (a small help, not a
  problem to fix).

## Matched comparison to the direct-total critic (= heavy-A′'s primary head)

OFF long-diag (direct-total critic, seed 41), pooled EV on matched windows:
| window | OFF direct-total pooled EV | OFF Var(e) | ON HRA pooled EV | ON Var(e_sum) |
|--------|----------------------------|-----------|-------------------|----------------|
| 400–600 | **0.644** | 228.3 | 0.621 / 0.611 | 223 / 232 |
| 250–600 | 0.653 | 258.3 | — | — |

- The direct-total critic's ABSOLUTE residual (~228) equals the ON HRA's (~223–232). The
  ON EV is ~0.02–0.03 lower only because its return variance (denominator) is smaller.
- **Heavy-A′ uses a directly-trained `V_total` head — which is the OFF critic — and it
  already only reaches ~0.64.** So heavy-A′'s upside over the rejected HRA is at most
  ~0.02–0.03 EV. The A′ pilot's ceiling is effectively known from data already on disk.
- Corollary: the LEG-A n=5 deficit (−0.093, PDR-0059) was on 200-round PRE-anneal windows;
  the matched post-anneal gap is far smaller (~0.02–0.03). Objective A stays rejected (the
  pre-registered screen stands) — but the "large total-fit deficit" framing was substantially
  a pre-anneal / statistic artifact.

## Corrections banked (over-reads from earlier passes)

- **"+0.04 recalibration lift" was a STATISTIC artifact** — pooled-recalibrated (0.621) was
  compared to *median*-of-per-update raw (0.581). Matched pooled-vs-pooled, the lift is ~0
  (raw pooled EV_sum = 0.621 = recalibrated). Affine recalibration does essentially nothing
  ⇒ `V_total` IS the near-optimal linear combination (reviewer-1's "real signal" retracted).
- **"light-A′ refuted" and "B favoured" both RETRACTED** — the affine test discriminates
  neither, and the deficit-vs-direct-total is only ~0.02–0.03.
- **"cf ceiling ~0.57" softened** — `ev_cf` ~0.60, roughly plateauing (seed 41 flat at
  ~0.60; seed 42 mildly rising 0.56→0.59). "Ceiling" is at-this-critic-capacity, NOT
  information-theoretic (that needs the Path-C probe).
- **"plateau-1 split" RETRACTED** — `Var(G_cf)` trajectories are noisy/non-monotonic (seed
  42: 505→358→488→546 per 100-round bin); no clean split, both ~490–650 wobbling. The
  −53/+41 slopes were a 2-median artifact of autocorrelated series with no error bars.
- **The `a=1.40` (seed 42) coefficient is immaterial** (Var(e_main)≈0 ⇒ reweighting a
  near-perfect head can't help). The advisor's attribution of it to the cf-normalizer lag
  was doubly wrong: `a` is the `V_main` coefficient, and `value_main_target_scale` ≈ 3
  (vs cf scale ~25). Verified from the data, not narrated.

## What actually moves the problem (reviewer-concordant, data-grounded)

1. **Does the CURRENT DENSE cf reward earn its keep? — the missing experiment.** `G_cf`
   carries ~all the return variance AND ~39% is unpredictable from state (the ~228 residual
   surviving every critic here). HYPOTHESIS (not yet a finding): a term that is
   simultaneously high-variance and low-predictability may inject advantage NOISE rather
   than reduce it — anti-shaping. But a noisy signal can still pay for itself via
   exploration / rare-motif discovery / long-horizon credit. The question is its MARGINAL
   DOWNSTREAM value. Direct test = the paired cf-off screen (§ below). Scope: it tests the
   current DENSE cf-in-reward use, NOT whether a redesigned auxiliary/control-variate cf
   could help.
2. **The missing downstream baseline:** the CAUSAL chain "critic EV → less advantage noise
   → better controller decisions → better accuracy/efficiency" is assumed, not demonstrated
   (there ARE accuracy/param/churn/governor gates, but none tie EV to them). Establish it
   before further critic work — see the program external-relevance gate (PDR-0067).
3. **Path-C ceiling probe — NOT feasible from retained data** (checked 2026-07-13): the
   event stream persists only aggregates; no per-timestep Obs V3 sequences. Would require a
   fresh run with observation-sequence logging. Removed from the cheap-parallel list.

## Caveats (uncertainty-checked)

- **Block-bootstrap (2026-07-13):** the earlier "plateau-1 split" is NOT real — `Var(G_cf)`
  slopes over 250–600 are −53 (90% CI [−50, +53]) and +41 (CI [−45, +43]); both cross zero.
  `ev_cf` slopes ~flat (±0.01). No evidence of drift or of a split; "ceiling"/"still-rising"
  language is unsupported at n=2. Seed is the replication unit; 100 autocorrelated updates
  are not 100 observations.
- "Reward-intrinsic residual" = intrinsic AT THIS LSTM CRITIC CAPACITY. A larger/different
  predictor might fit more of `G_cf` — only a fresh Path-C probe (above) tests that.
- n=2; single OFF comparator (seed 41); windows matched but runs differ in rollout.

## Process fix adopted

Every over-read this session was caught AFTER the compute yet was catchable BEFORE. Memory
notes are not controls. **Adopt a gate: a pre-committed reading gets an advisor pass BEFORE
the run launches** (amends the run-authorization discipline; recorded PDR-0066).
