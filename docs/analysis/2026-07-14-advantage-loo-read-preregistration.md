# Pre-registration — the offline advantage-vs-LOO read (§5 "the floor already ran the RCT") + §3 KL gate

Date: 2026-07-14 · Written BEFORE any number is read (over-read guard). Governs the two decisive zero-GPU reads that
gate the GPU spend. Runs: `telemetry/stage2_on_longdiag/seed{41,42}` longdiag (~1.08M decisions/seed). Companion:
`2026-07-13-round8-findings-evaluation.md` (§5, §3).

## Why this is a natural experiment (the identifying assumption, stated up front)
The realized op mix at HOLDING is **flat across every current-LOO bin** (~55/17/15/13, both seeds) — the anti-WAIT
floor forced FOSSILIZE at ~15-17% *independent of LOO*. That is **marginal ignorability**: assignment to FOSSILIZE is
(marginally) independent of the "dose" (LOO). So the existing telemetry already estimates the counterfactual the policy
gradient would act on. **This is MARGINAL, not conditional** — residual state-confounds (seed age, slot occupancy,
host-accuracy trend, round) must be stratified out before the differential is trusted.

## Read A — advantage-vs-LOO (§5), the reframe's only unproven bridge

**Quantity:** for HOLDING op-decisions, per-decision advantage `A` for the realized op, binned by current LOO
(`seed_contribution`), gated stage-at-decision = HOLDING, fresh non-null, NO carry-forward.

**Reconstruction (two tiers — REPORT WHICH):**
- **VALID tier:** full GAE if per-step `value` rows exist (V(s_t), V(s_{t+1}), reward-after-all-shaping, γ, GAE λ,
  terminal-vs-truncation masks, rollout+batch boundaries) — reconstruction must pass a **PARITY CHECK**: re-derived
  standardized advantages reproduce the training-time mean≈0 / std≈1 (or a logged aggregate). Parity → VALID.
- **PROXY tier:** if per-step value rows are absent (Tamiyo decisions are sparse), fall to 1-step TD
  `δ = r + γV(s′) − V(s)` at decision rows — **sign-only**, and it **CANNOT alone kill the experiment** (gpt-prime).
  Report the tier explicitly next to every conclusion.

**Primary statistic:** per LOO bin, `E[A | FOSSILIZE]` vs `E[A | SET_ALPHA_TARGET]` vs `E[A | PRUNE]`, and the slope of
`A_FOSSILIZE` (and of the differential `A_FOSSILIZE − A_SET_ALPHA`) vs LOO. Stratify by seed_age / occupancy / host-acc
trend / round. Per seed (n=2); do not pool blindly.

**Pre-registered decision rule (fixed before the number):**
| Outcome | Reading | Action |
|---|---|---|
| **GRADED** — `A_FOSS − A_SETALPHA` rises with LOO (positive slope, survives stratification) | the learning signal exists; the dead-zone is why the policy can't act on it | **PDR-0069 confirmed → build the op-isolated ρ-sweep fix** |
| **FLAT** — no slope | the immediate reward gradient is cancelled by baseline/continuation value; the floor fix alone will NOT restore LOO-selectivity | **reframe collapses → reward/critic line reopens (N2 = prime suspect)** |
| **INVERTED** — `A_FOSS` falls with LOO, or `A_FOSS < A_SETALPHA` at high LOO | committing high-LOO seeds is *bad for (critic-estimated) return*; the policy's low fossilization may be CORRECT | **the epic's founding premise ("under-fossilization is a defect") is itself the over-read → escalate a premise re-examination** |

**Validity caveat (banked now):** the outcome variable is the CRITIC's return, not ground truth. This read tells you
**what the fix would TEACH the policy**, not what is true. A mis-fit critic (the EV-stab epic's own concern) propagates
into `A`. So GRADED confirms "the signal the policy would learn from exists"; it does not prove fossilizing is optimal.
A PROXY-tier result is directional only.

## Read B — the KL trust-region gate (§3, P8)

**Quantity:** fraction of PPO updates where KL early-stop fired (`early_stop` / `early_stop_epoch`, `target_kl=0.015`),
and per-head approx-KL (op / slot / blueprint / others), per seed.

**Pre-registered decision rule:**
- **P8 < ~5% (early-stop rarely fires):** the trust region has been effectively vacuous → a **competing,
  reward-independent root cause for the Dec-2025 WAIT-collapse**, and the λ=0 necessity smoke is confounded. **The
  KL-aggregation fix (per-head or gradient-bearing-sample-only KL for early-stop) must land BEFORE any GPU arm** — a
  harness fix, and (core-training-loop change) an OWNER-GATED escalation.
- **P8 materially non-zero (early-stop fires normally):** the §3 bomb is defused; the aggregate KL is diluted but still
  binds. Note the magnitude; proceed.
- **Per-head KL:** expect op/slot/blueprint ≈ 0 (structural), learning heads materially higher — a diluted aggregate is
  confirmed if the learning-head KL is ≫ the aggregate.

## What these reads do NOT decide
Neither read authorizes a GPU run or a primitive change (owner-gated). Read A decides *whether the reframe's premise
holds*; Read B decides *the experiment's sequencing*. Both are inputs to the owner DECIDE, not the decision.

---

## OUTCOME — Read A (2026-07-14). TIER = PROXY. Classification = INVERTED (both seeds, LEVEL clause). ESCALATE.

**Tier = PROXY (1-step TD, sign-only).** Full-GAE VALID is impossible from this telemetry: per-decision V_cf, the three
normalizer stds (σ_r, σ_v, σ_cf), and the truncation bootstrap are not logged. Parity therefore not runnable. Per the
pre-reg, **a PROXY result cannot unilaterally kill the experiment.** Partial-parity anchors that let the SIGN be trusted:
σ_v confirmed = logged `value_target_scale` (in-window ≈26.6/25.1); the omitted mean V_cf continuation ≈ **−0.5** (from
logged `td_error≈0` on V_total vs the +0.50/+0.47 main-only proxy) — i.e. omitting V_cf biases advantages **too high**,
**against** finding an inversion.

**Classification = INVERTED, both seeds**, on the pre-registered LEVEL clause: `A_FOSS < A_SET_ALPHA` at every LOO bin
(top bin −2.65 s41 / −1.13 s42). GRADED fails (differential slope seed-inconsistent: s41 −0.020, s42 +0.047; never
crosses zero in-range). Driven by an **observed** main-stream continuation penalty: fossilizing drops V_main
(cont_FOSS ≈ −0.14/−0.07) while merely staying in HOLDING does not (cont_WAIT ≈ 0) — a genuine **action effect of
committing**, not horizon shrinkage. Robust across epoch-timing (FOSS/SET_ALPHA occur at the same within-episode epoch)
and five stratifications (occupancy, val-acc, round-half, seed-maturity, epoch).

**The V_cf escape hatch (the one unfalsified reversal path):** V_cf is unobserved, so this is a PARTIAL (main-only)
advantage. If V_cf is well-fit and drops LESS after FOSSILIZE than after SET_ALPHA (opposite of the mechanism), it could
lift A_FOSS at high LOO — but even then the LEVEL inversion holds at every OBSERVED bin; V_cf controls only the
slope/crossover beyond the data. Confidence: **LEVEL inversion within observed LOO = HIGH (PROXY);** interpretation that
fossilizing is genuinely suboptimal = **LOW-to-MEDIUM** (critic-return, not ground truth; v_return_corr ≈ −0.35).

**Three mechanisms fit `A_FOSS < A_SET_ALPHA` with OPPOSITE epic implications — this read CANNOT separate them (advisor):**
(a) **designed cost** — fossilizing starts paying `occupancy_rent`/`fossilized_rent` → continuation drops (the reward-spec
artifact reading: premise possibly wrong); (b) **option value** — FOSSILIZE is irreversible, SET_ALPHA/WAIT keep options
open, so a rational V assigns lower continuation to commitment *ceteris paribus* (legitimate, NOT a defect, NOT an
artifact); (c) **on-policy continuation trap** — the scored fossils are floor-FORCED and unselective, so V(s′_FOSS) is low
*because commitment happens at the wrong times under a dead-zone-crippled policy* → **the inversion is the WOUND, not the
verdict**; fixing the dead-zone could then RAISE V(s′_FOSS).
**Forced-timing liability (the sharp caveat — marginal ignorability is the RCT's weakness, not its virtue):** most
FOSSILIZE samples are floor-forced, so E[A|FOSS] measures the value of committing at RANDOM times — ~tautologically below
well-timed continuation — and is nearly silent on SELECTIVE commitment (what the epic is about). SET_ALPHA is the
argmax/dominant op in most HOLDING states, so this partly compares forced-random commits against CHOSEN alphas;
stratifying on LOO/age/occupancy does NOT fix a selection difference in the action itself.

**Facts:** γ=0.995; GAE λ=0.95 (config override, not the 0.98 default); reward = `total_reward` (raw scale, INCLUDES the
cf stream r_cf=bounded_attribution(LOO)); V(s′) = next-decision `value_estimate` = V_main normalized; decisions dense
(1 action/host-epoch); every episode boundary is a truncation bootstrap. Scripts + extracted pkls in the session
scratchpad.

**Disposition (fires PDR-0073's INVERTED trigger; escalation stated at honest altitude — advisor):**
1. **DIRECTION-INDEPENDENT RESULT (survives all three mechanisms — the robust take):** do NOT build the ρ-sweep as a
   commitment-INCREASER. Unfreezing the op gradient makes the policy act on THIS advantage ordering, which favours
   SET_ALPHA over FOSSILIZE — so the "fix" would not raise fossilization and might LOWER it. **Hold the floor-fix
   regardless of which mechanism is true.**
2. **The disambiguator (next read; runnable on the ALREADY-EXTRACTED telemetry — higher value than V_cf for the
   epic-direction question):** decompose the FOSSILIZE V_main continuation penalty into discounted designed RENT
   (`occupancy_rent`+`fossilized_rent`, logged components) vs everything-else (foregone accuracy + option value).
   Penalty ≈ rent → reward-spec reading (a) gains support; penalty ≫ rent → option/foregone value → premise SURVIVES and
   the inversion is dead-zone damage (b/c). This single read adjudicates the fork.
3. **VISION-LEVEL ESCALATION (owner), at altitude:** the premise ("under-fossilization is a defect") is now **TESTABLE in
   one specific read** — NOT "the premise is wrong." Bring the fact + the fork + the disambiguating read.
4. **V_cf instrumentation** (per-decision V_cf+V_total → PROXY→VALID) closes the sign BEYOND observed LOO, but is
   secondary to the rent-decomposition for the epic-direction question.

## G1 result (2026-07-14, reward-code read) — REFUTES the rent-dominates hypothesis; premise SURVIVES; critic is decisive
Coefficients: fossilize bonus = `0.5 + 0.1·c` (ONE-SHOT; ~1.5 at median c≈10, × legitimacy_discount≤1). Rent:
`fossilized_maintenance_cost=0.002`/step/fossil (permanent), `seed_occupancy_cost=0.01`/excess-slot/step, `free_slots=1`.
Fossilizing adds **+0.002/step** and does NOT change occupancy (the fossil still occupies its slot — 2025-01-08 note:
fossilize is not an occupancy escape hatch). Discounted permanent rent over the ~71 remaining epochs (FOSS at ~epoch
79/150, γ=0.995) ≈ **0.12**, vs a one-shot bonus ≈ **1.5** → `break_even ≈ 750 steps ≫ 71`. Read A's continuation penalty
≈ σ_v(26.6)·0.10 ≈ **2.6 raw-units**, so **designed rent explains ~5%** of it. **Claude-prime §2 (lump-sum-vs-perpetual,
negative-EV-by-construction) is REFUTED on the numbers; branch (a) reward-spec-via-rent LOSES support; the premise does
NOT die at G1.** The ~2.6 penalty is non-rent → option value / foregone accuracy / **critic mis-fit**. → G2 decides.

## The design corrections the round-9 primes forced (adopt for the next read)
- **The |H|=1 pinned simplex is an EXACTLY-randomized RCT** (constant op distribution independent of the whole state),
  not a weakness. Ignorability holds conditional on ANY covariates → estimate the **CATE over the full covariate vector**
  and look for a POSITIVE region (the selectivity question). Restrict to `op_entropy == H_min(num_valid)`, **stratify on
  num_valid**, drop |H|≥2 (confounded, use as spec-check). This answers the "forced-timing" objection at finer resolution.
- **Two real threats to Read A (the round-9 headline):** (i) **critic mis-fit** — A is two evaluations of a critic with a
  DOCUMENTED total-fit failure (Objective-A HRA, PDR-0071 §5); if it under-predicts post-commit return, Read A measures
  the critic, PDR-0066 re-opens, and the floor-fix becomes ACTIVELY DANGEROUS; (ii) **crippled continuation** — V(s′) is
  V^π for the FLOORED π; a freed capability is worth little to a dead op head → the inversion is partly self-fulfilling.
- **The disambiguator's false dichotomy is fixed:** "penalty ≫ rent" does NOT imply dead-zone damage. Report rent-fraction
  / foregone-accuracy-fraction / residual, PLUS `Δ_critic-residual = Δ_cont − Δ_MC-total` (gpt-prime), PLUS the selectivity
  CATE. Use PRECISE training reward units (V is trained on clipped/normalized reward; raw rent is not comparable to V).
- **Read B wording (gpt-prime):** record as "the configured target-KL early-stop supplies NO operative pre-emptive update
  guard under the K=1 loop" — not "no trust region at all" (PPO clip exists but is inert at ratio≈1).
- **Owner-gated reframe the primes put forward:** the epic's spine moves from "make the policy commit" to "PRICE
  commitment correctly, then let the policy decide"; transfer/`hindsight_credit` (the missing revenue line) becomes the
  PRIMARY track, the floor-fix becomes what you do AFTER the ledger is correct. A vision-level proposal for the owner.

## G2/G3 RESULT (2026-07-14) — the fork is RESOLVED: the REWARD is the binding cause; floor-fix contraindicated
Estimand: the |H|=1 dead-zone RCT (FOSS floor-drawn at conf 0.150 in every row; SET_ALPHA the pinned-dominant op at
0.55; `op_entropy≈0.853`). Realized MC = logged per-step `total_reward` summed to episode end (GROUND TRUTH, not a critic
estimate). Per seed; n(|H|=1) FOSS 1026/1111, SET_ALPHA 3378/3946.

**G2 — critic vs world = BOTH, reward binds.**
- **Critic mis-fit CONFIRMED (secondary):** V_main asserts a FOSS−SET_ALPHA continuation penalty of −6.6/−5.4, but the
  REALIZED main-stream penalty is −1.0/−1.2 → the critic OVERSTATES the task penalty **4-6×** (Δ_critic-residual
  −5.65/−4.20). Read A's inversion, on the task stream, is largely a critic artifact — the documented Objective-A HRA
  total-fit failure surfacing as a main-head bias. (So PDR-0066 is live, but it is NOT the primary story.)
- **The world's REAL penalty is a REWARD property (primary, GROUND TRUTH):** realized total-return contrast −113/−102,
  decomposing to **cf-stream (recurring `bounded_attribution`) −112/−101 = ~99%**, accuracy −0.7/−1.0 = ~0.8%, rent
  −0.25/−0.27 = **~0.3%** (rent branch refuted tighter than G1). Verified from the sequences: a HOLDING seed accrues
  `bounded_attribution` ~3.15/2.86 per epoch (87-88% of steps); after FOSSILIZE it collapses to ~0.32/0.22 (10-13%).
  **The reward pays a provisional HOLDING seed a recurring per-epoch contribution credit; fossilizing the SAME seed —
  which does not reduce its accuracy contribution — forfeits that stream (~10× drop). So the reward pays ~99% more to
  PARK a good seed than to make it permanent → commitment-avoidance is the OPTIMAL response to the reward, not a policy
  pathology.**

**G3 — selectivity: NO positive-value commitment region** on total reward (τ_total < 0 every stratum, both seeds) and at
best task-neutral on the main stream (τ_main ≈ 0; only low-LOO/short-horizon cells non-negative, never clearly positive).
**Honest bound (banked):** this is the effect of committing the seeds THIS floored policy produced — the floor + cf-credit
design suppress the whole commitment pathway, so the policy never developed seeds toward genuinely-worth-committing
states. "No positive region among floor-drawn commits" ≠ "no policy could ever benefit from committing." It does show:
under THIS reward, committing is not locally rewarded.

**α at FOSSILIZE:** not computable (`alpha_target` None on all FOSS rows — set only by SET_ALPHA). The consequent
(committing adds ~0 accuracy) is confirmed (main-ex-rent contrast ≈0); the median-α≥0.8 antecedent is not.

**Resolution + disposition (fork RESOLVED by round-10 fire-rate decomposition + a code read — it is a MEASUREMENT gap, NOT a specification):**
1. **The credit collapse is FIRE-RATE, not magnitude (claude-prime §1, arithmetic on the agent's own numbers).** HOLDING
   `bounded_attribution` = fire-rate 0.875 × magnitude 3.60 = 3.15; FOSSILIZED = 0.115 × 2.78 = 0.32. The ~10× collapse =
   **rate 7.6× (79-89% of the log-collapse)** × magnitude 1.3×. The credit did not get SMALLER at permanence — it STOPPED
   FIRING.
2. **Mechanism CONFIRMED by code (H2): FOSSILIZED seeds are EXCLUDED from the counterfactual ablation by design**
   (`vectorized_trainer.py:956-975`: *"disabling a permanently-integrated seed measures damage to the host, not the
   seed's contribution"* — skipped unless `drip_fraction>0`; the run is SHAPED/drip=0). So a fossilized seed's
   `seed_contribution` is structurally `None` → `bounded_attribution` can't fire → the reward pays a seed for its
   contribution ONLY while provisional (not at α=0 birth, not at permanence). Same `counterfactual=None` missingness that
   caused over-reads #6/#7/#8 — but in the TRAINING signal. **This is a MEASUREMENT/INSTRUMENT gap, not a reward
   specification.**
3. **RETRACT (round-9 over-read #13):** "reward-specification / branch 1 / commitment-avoidance is optimal / vindicates
   the EV-stab epic / ESCROW as the fix." The `bounded_attribution` TERM is not the defect; its INPUT (a fossil's
   counterfactual) is missing.
4. **Also RETRACT (both primes):** PDR-0069 "the gate passes" and PDR-0071 §4 "signal survives the 9-pt spread" — the
   −0.7→+8.2 was a 9-pt spread on the revenue line while a −113 cost line, scaling the same way, sat UNMEASURED; the gate
   fails by ~75× (bonus ~1.5 vs forfeited stream −113). And un-retract the PDR-0071 co-requisite (hindsight/transfer): it
   is the missing settlement, plausibly dead for the SAME measurement reason (H3, unconfirmed).
5. **The premise is UNMEASURED, not refuted (claude-prime §5).** G3's τ_total<0 was read THROUGH the −113 phantom; the
   load-bearing number is **τ_main ≈ 0** — strip the phantom and there is no evidence commitment is bad. G3 is an
   independent confirmation of the bug's magnitude, NOT evidence about commitment value.
6. **Floor + reward are a PACKAGE, not a sequence (claude-prime §3).** FOSSILIZE is floor-bound 93-94% → the ~17%
   fossilization rate is the FLOOR's number, not the policy's. Fix the gradient under this reward and fossilization → ~0
   (the floor is accidentally the entire morphogenesis pipeline). So "the policy is correctly parking" is NOT established
   (the head is gradient-frozen on 94% of commits) — only "if it could act on this reward, it would never commit."
7. **Fix = INSTRUMENT-FIRST** (round 1's abandoned recommendation, now shown to poison training): build a VALID causal
   measure of a permanent seed's contribution (the naive ablation cannot — see the code rationale). **ESCROW is
   CONTRAINDICATED** (it claws back credit on a measured-contribution collapse → would claw back the whole accrued credit
   at the fossilize measurement-zero; H4 to confirm the exact trigger). Connects to the transfer/`hindsight_credit` track.
8. **Floor-fix (ρ-sweep): CONTRAINDICATED + coupled** (do not ship without the measurement fix, or morphogenesis stops).
   Critic mis-fit is real but SECONDARY (4-6× overstatement on THIS contrast, subject to MC/value-target parity — do not
   generalize to "the critic is broken").
9. **Owner-gated refocus (PROPOSED, not enacted):** the epic's Now bet moves toward **causal permanent-seed
   instrumentation** (measure a fossil's true contribution), floor + critic held constant/deferred. Confirmatory next
   reads (zero-GPU): H3 `hindsight_credit` numerator/denominator/trigger; H4 escrow clawback trigger at FOSSILIZE; H5
   confirm `attributed` derives from the LOO (structure says yes: `contribution.py:539-552`).
