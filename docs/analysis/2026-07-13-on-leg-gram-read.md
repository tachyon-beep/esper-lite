# ON-leg Gram diagnostic (n=2) — PDR-0064 read (final, exact-decomposition)

Date: 2026-07-13 · Runs: `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/`
Commit: `92d998bd` · Completion: both seeds 600/600, rc=0, 0 crashes.
Reviewed: advisor + 2 external peers (ChatGPT, claudemain), 2026-07-13 — all integrated.
Diagnostic (not a gate); rejected HRA re-used as substrate. THIRD-PASS: supersedes the
"B favoured" (9dbf078f) and "light-A′ refuted" (a87ba291) verdicts — both were over-reads.

## Headline: the critic is the wrong suspect; the residual is reward-intrinsic, and the real question is whether the cf shaping term earns its keep

Exact residual decomposition (matched pooled moments, no running-scale approximation)
resolves the A′/B question by largely dissolving it: **no critic architecture in this
family beats ~0.62–0.64 total EV, because ~all of the unexplained variance lives in the
cf reward stream and is not predictable from state at this critic's capacity.** The
main head is already near-perfect; heavy-A′'s primary head (a direct `V_total` critic)
already exists as the OFF run and only reaches ~0.64. The lever is the cf REWARD, not the
critic topology.

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

1. **Does the cf shaping term earn its keep? — the missing experiment.** `G_cf` carries
   ~all the return variance AND ~39% of it is unpredictable from state (the ~228 residual
   that survives every critic here). A shaping term that is simultaneously high-variance
   and low-predictability injects advantage NOISE — the opposite of what shaping is for.
   The direct test: **ablate the cf term, measure the DOWNSTREAM outcome** (host accuracy /
   controller decision quality / committed-J). Collapse ⇒ cf is load-bearing and now
   quantified (the baseline the epic never had). No collapse ⇒ the A/A′/B/C tree dissolves
   and a term is deleted. Cheaper than this read, worth more.
2. **The missing baseline (one level up):** 65 PDRs in, nothing connects `ev_sum` to a
   downstream metric anyone cares about. EV is instrumental (advantage-variance → policy
   quality); a PPO critic at ~0.6 is unremarkable. Establish that the current EV level
   costs something downstream BEFORE any further critic-architecture work.
3. **Path-C ceiling probe (cheap, data may be on disk):** offline predictor on the
   policy-visible Obs V3 history for `Var(G_cf|s)` — is the ~228 residual an information
   ceiling or critic-capacity? Beats the frozen-head Gram data for this question.

## Caveats

- "Reward-intrinsic residual" = intrinsic AT THIS LSTM CRITIC CAPACITY. A larger/different
  predictor might fit more of `G_cf` — the Path-C probe tests that; the vg scalars cannot.
- n=2; single OFF comparator (seed 41); windows matched but runs differ in rollout.

## Process fix adopted

Every over-read this session was caught AFTER the compute yet was catchable BEFORE. Memory
notes are not controls. **Adopt a gate: a pre-committed reading gets an advisor pass BEFORE
the run launches** (amends the run-authorization discipline; recorded PDR-0066).
