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

- **The main head's absolute residual is tiny** (`Var(e_main)` ≈ 0.4–0.6). The ENTIRE total
  residual is the cf stream (cf share ~1.0); component errors mildly ANTI-correlate (a small
  help, not a problem to fix). CAVEAT (reviewer): "near-perfect" is scale-dependent — `ev_main`
  is 0.84, so 16% of the main stream's own variance is unexplained; it is absolutely small
  only because `Var(G_main)` is small. Do not let "near-perfect" become load-bearing.

## Matched comparison to the direct-total critic (= heavy-A′'s primary head)

OFF long-diag (direct-total critic, seed 41), pooled EV on matched windows:
| window | OFF direct-total pooled EV | OFF Var(e) | ON HRA pooled EV | ON Var(e_sum) |
|--------|----------------------------|-----------|-------------------|----------------|
| 400–600 | **0.644** | 228.3 | 0.621 / 0.611 | 223 / 232 |
| 250–600 | 0.653 | 258.3 | — | — |

- The direct-total critic's ABSOLUTE residual (~228) equals the ON HRA's (~223–232). The
  ON EV is ~0.02–0.03 lower only because its return variance (denominator) is smaller.
- **Heavy-A′ uses a directly-trained `V_total` head — which is the OFF critic — and it
  already only reaches ~0.64** ⇒ heavy-A′'s upside over the rejected HRA is small. PROVISIONAL
  — this "ceiling = OFF run" move (reviewer-flagged as the 4th tidy claim) rests on three
  assumptions: (a) OFF and ON REWARD identical — **CONFIRMED** (`partition.py`: `r_main+r_cf`
  on both legs; HRA only splits the VALUE target, not the reward); (b) comparable state
  distributions — UNCHECKED (different critic → different policy → different rollouts);
  (c) heavy-A′'s auxiliary heads add nothing to the primary head's representation — UNCHECKED
  and NOT free (auxiliary value heads are a standard representation-learning trick, sometimes
  a large one). So: probably correct, not yet load-bearing.
- Corollary: the LEG-A n=5 deficit (−0.093, PDR-0059) was on 200-round PRE-anneal windows;
  the matched post-anneal gap is far smaller (~0.02–0.03). Objective A stays rejected (the
  pre-registered screen stands) — but the "large total-fit deficit" framing was substantially
  a pre-anneal / statistic artifact.

## Corrections banked (over-reads from earlier passes)

- **"+0.04 recalibration lift" was a STATISTIC artifact** — pooled-recalibrated (0.621) was
  compared to *median*-of-per-update raw (0.581). Matched pooled-vs-pooled, the lift is ~0
  (raw pooled EV_sum = 0.621 = recalibrated). Affine recalibration does essentially nothing
  ⇒ `V_total` IS the near-optimal linear combination (reviewer-1's "real signal" retracted).
- **"light-A′ refuted" STANDS (on stronger grounds); only "B favoured" is retracted.** The
  original +0.04 *empirical* argument was a statistic artifact — but the CONCLUSION is
  correct via an INFORMATION argument (reviewer, GPT-set 2): the two head outputs are two
  scalars carrying `E[G_main|s]` and `E[G_cf|s]`; if `(1,1,0)` is optimal by construction and
  matched recalibration moves EV 0.621→0.621, then NO function of them (not just affine) adds
  information. A post-hoc layer on the frozen heads (light-A′) is dead. NOTE: blanket-
  retracting after a hard hit was itself a calibration failure (sign-flipped over-correction)
  — light-A′ was over-retracted in the prior pass. "B favoured" remains retracted (the read
  can't discriminate heavy-A′ vs B; the deficit-vs-direct-total is only ~0.02–0.03).
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

0. **"Anti-shaping" RETRACTED as a premature over-read (reviewer, claudemain).** A
   state-value critic predicts `E[G|s]` BEFORE the action; the variance it CANNOT explain is
   exactly the ACTION-dependent part — i.e. `ev_cf ≈ 0.57` is what a WORKING credit signal
   looks like, not pathology. A reward term with HIGH state-EV would be one the controller
   can't influence — useless. So low `ev_cf` is not evidence against cf.
1. **The sharp question — is cf's unexplained variance CREDIT or NOISE? (SNR, cheaper than
   the ablation).** The ~39% splits into two pieces with OPPOSITE implications:
   - `Var_a(E[G_cf|s,a])` — BETWEEN-action variance = credit signal; a state-value critic
     SHOULD leave it unexplained. If it dominates ⇒ the critic is behaving correctly, EV~0.6
     is the right number, and "EV-stabilization" was mis-specified at PDR-0001 ⇒ KEEP cf,
     close the epic.
   - `E[Var(G_cf|s,a)]` — WITHIN-(s,a) variance = noise injected downstream of the decision.
     If it dominates ⇒ cf pumps variance into every gradient ⇒ delete/re-estimate cf.
   Deciding quantity: `SNR(cf) = Var_a(E[G_cf|s,a]) / E[Var(G_cf|s,a)]` — **the critic's EV
   appears nowhere in it** (the tightest statement of why 66 PDRs of critic architecture
   could never answer the question). Needs per-sample `(s,a,G_cf)`, which the `vg_*` moments
   don't carry and the rollout buffer computes but does NOT dump → one short logging run,
   then a variance decomposition. On-policy caveat: a narrow action distribution under-counts
   between-action variance — report action entropy alongside (instrument exists from the
   thermometer work).
2. **Then: does the current dense cf earn its keep DOWNSTREAM? (the ablation, sized right).**
   Direct test = the paired cf-off screen (PDR-0067). Two corrections (claudemain): (i) it is
   NOT a small screen — cf carries ~500 of ~590 return variance, so ablating it trains the
   controller on a near-flat signal (legitimate null model, but size it correctly; not
   coverage-bound like the PDR-0026 reward A/B); (ii) it must measure the thing cf was
   INTRODUCED to deliver (credit-assignment / anti-freeloading steering) — otherwise a null
   on an unrelated metric proves nothing. Find the originating decision first.
3. **The missing downstream baseline** — the chain "critic EV → advantage noise → decisions
   → accuracy/efficiency" is assumed, not demonstrated. Program external-relevance gate
   (PDR-0067) blocks banking critic work on EV alone.
4. **Path-C ceiling probe — NOT feasible from retained data** (checked 2026-07-13): only
   aggregates persisted, no Obs V3 sequences. Needs a fresh instrumented run.

## Caveats (uncertainty-checked)

- **Block-bootstrap (2026-07-13):** the earlier "plateau-1 split" is NOT real — `Var(G_cf)`
  slopes over 250–600 are −53 (90% CI [−50, +53]) and +41 (CI [−45, +43]); both cross zero.
  `ev_cf` slopes ~flat (±0.01). No evidence of drift or of a split; "ceiling"/"still-rising"
  language is unsupported at n=2. Seed is the replication unit; 100 autocorrelated updates
  are not 100 observations.
- **"Current-state / current-model residual", NOT "reward-intrinsic unpredictability"**
  (reviewer): the exact decomposition establishes the total residual is overwhelmingly
  cf-side UNDER THE CURRENT observation + recurrent model + optimizer + regime. It does NOT
  establish that ~39% of `G_cf` variance is fundamentally unpredictable from the true Markov
  state — that needs the Path-C probe AND is partly the SNR question (within-(s,a) noise vs
  between-action credit).
- n=2; single OFF comparator (seed 41); windows matched but runs differ in rollout.

## Process fix adopted

Every over-read this session was caught AFTER the compute yet was catchable BEFORE. Memory
notes are not controls. **Adopt a gate: a pre-committed reading gets an advisor pass BEFORE
the run launches** (amends the run-authorization discipline; recorded PDR-0066).
