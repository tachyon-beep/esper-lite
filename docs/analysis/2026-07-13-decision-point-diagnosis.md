# Decision-point diagnosis — calibrated status (gpt-prime synthesis)

Date: 2026-07-13 · Runs: `stage2_on_longdiag/seed{41,42}`. Code + telemetry, no GPU.
Multi-model diagnostic (owner-orchestrated: advisor + ChatGPT + Claude). This file swung
many times; every miss was **a load-bearing input assumed instead of read** (invented `c`,
global epoch counter, unfiltered SET_ALPHA population, two different "contribution" fields,
a bonus computed at c≲1 when actual c≈10, "unpayable" stated as absolute). The reasoning was
consistently fine. The rule: **read the variable in the sentence before you write it.**

## READ B — CORRECTED: absorption/"scaffold retirement" REFUTED; ~30% of HOLDING-prunes discard causally-load-bearing seeds

An earlier version of this section (over-read #6, comforting direction, caught by claudeweb +
the single-seed trace) claimed "productive scaffold retirement": the prune cohort peaked high
then DECAYED to a 0.12 removal-cost as the host absorbed the value, so the gate "selects on the
right signal." **That is refuted.** Two artifacts produced it:

1. **The "0.12" was a `None`-filter artifact.** `SEED_PRUNED.counterfactual` is `None` for ~7/8
   pruned-after-HOLDING seeds (single-seed trace). The "prune cohort median 0.12" was computed
   over the small non-`None` subset — never representative. There is NO gliding decay to 0.12.
   The single-seed traces do not glide; several seeds are pruned at HIGH, RISING decision-time
   c (e.g. 4.6→13.6, 8.9→**23.8**), not decayed.
2. **The ransomware/leak split by `total_improvement` was invalid (over-read #7).**
   `total_improvement = current_val_acc − initial_val_acc` at germination (slot.py:217) — HOST
   PROGRESS over residency, explicitly warned NON-CAUSAL (slot.py:251-256, "conflates host
   training gains with seed impact"). It cannot separate ransomware from leak from anything.

**The clean read (identity-robust + missingness-aware)** — gate on the decision ROW's own
`seed_stage` (that IS the acting seed's stage → no cross-lifecycle carry, closes gpt-prime's
identity objection for a point-in-time measure), and count `seed_contribution` ONLY when
non-`None` (NO carry-forward — closes gpt-prime's Problem 2). Seeds 41/42:

| decision (0% missing unless noted) | n (41/42) | median c | %≥5 | %<0 |
|---|---|---|---|---|
| **FOSSILIZE** (commit) | 2043/2108 | **10.4** | 66/65% | 9/11% |
| **PRUNE-from-HOLDING** (discard) | 1434/1617 | **9.6** | 65/61% | 10/12% |
| **PRUNE-from-BLENDING** | 5955/5183 | **5.8** | 55/51% | 12/13% |
| PRUNE-from-TRAINING (exploration) | 74435/75307 | *unmeasured — 100% `None`* | — | — |

**The headline, both seeds, 0% missingness on the measured cohorts:**
- **Current terminal LOO does NOT separate commit from HOLDING-discard.** Fossilised seeds
  (median c≈10.4) and seeds pruned from HOLDING (median c≈9.6) have the SAME contribution
  distribution (~65% ≥5, ~10% net-negative in both). At the actual commit-vs-discard decision
  point, current contribution does not discriminate the two fates.
- There IS a stage gradient across the FULL prune population (fossil≈HOLD-prune 10 >
  BLEND-prune 5.8 > TRAIN-prune unmeasured exploration) — so "prunes are lower-contribution
  than fossils" is true ONLY because it lumps in exploration/BLENDING churn. The seeds that
  reached the last decision point (HOLDING) are contributing as much as the ones committed.
- The 74k TRAINING-stage prunes have NO counterfactual (100% `None`) — exploration churn at
  α≈0, correctly excluded; carrying their `None` forward is what produced the retracted 0.84.

**Counterfactual computation mechanism (verified — vectorized_trainer.py:1365-1394,
features.py:61-75).** `counterfactual_contribution = val_acc − solo_acc` (true LOO), computed
ONLY when a "solo" config is evaluated for the slot — i.e. for BLENDING+ seeds, NOT every epoch.
An `epochs_since_counterfactual` staleness tracker is reset on each fresh measurement, and the
policy is fed `freshness = γ^epochs_since_cf` (obs V3). Consequences: (a) TRAINING prunes are
100% `None` because TRAINING runs at **α=0** — the seed is NOT in the forward pass at all (it
trains in isolation before BLENDING ramps α up), so its counterfactual is STRUCTURALLY zero by
construction, not a data gap. Those 74k prunes reject seeds during isolated training, before they
are ever wired in. (b) the non-`None` value on any decision row is the LAST-MEASURED LOO, up to a
few epochs stale.
So the honest statement is "last-measured contribution ≈10," and the fossil-vs-HOLD-prune
comparison is robust to staleness ONLY IF both fates act on equally-fresh counterfactuals.

**Freshness residual — CLOSED (both reviewers flagged as the #1 threat; refuted).** Reconstructed
`epochs_since_counterfactual` exactly (the LOO value is written ONLY on a fresh solo-eval, so
epochs-since-value-last-changed == staleness). Both cohorts act on FRESH counterfactuals: 98-99%
at staleness=0, ~0% at ≥3, in FOSSILIZE and PRUNE alike — no confirm-before-commit asymmetry.
The decisive matched read (fresh-only, staleness=0): FOSSILIZE median 11.2/10.2 vs PRUNE-from-HOLD
median 10.3/9.2 — overlap HOLDS; prune-c does NOT collapse when forced fresh. Stale prunes (s≥3)
are n=4-5 and low-c (median 0), the opposite of the stale-high concern. **The overlap is not a
staleness artifact.**

**Still NOT measurable / open (do not resolve either way):**
- **Realised-action overlap ≠ policy-preference flatness (gpt-prime's key open read).** A
  stochastic policy could have π(FOSSILIZE) rise with LOO yet produce overlapping REALISED
  cohorts. The direct test is the op-head logit margin `log π(FOSSILIZE) − log π(PRUNE)` vs LOO,
  on the both-actions-legal population — not classifying terminal outcomes. UNTIL THEN the banked
  claim is "current LOO shows substantial distributional overlap and no obvious univariate
  separation between realised FOSSILIZE and HOLD-PRUNE," NOT "the policy cannot separate them."
- Whether the high-current-LOO HOLDING-prunes are premature prunes, productive turntable
  retirement, ransomware/dependency correction, slot-budget, or partial-α removals — no CAUSAL
  net-ensemble-value field exists (`total_improvement` is host-progress, non-causal).
- Alpha at the HOLD-prune — separating full-α from turntabled-α (partial) removals.
- Formal distribution equality — the medians/tails overlap, but ROC-AUC / Wasserstein with
  run-level (not decision-row) bootstrap is needed before "same distribution" is more than
  "substantial overlap."

## POLICY-PREFERENCE read — the MARGINAL realised op mix at HOLDING is invariant to fresh current LOO

**Headline amended (both primes, round 3):** this establishes MARGINAL invariance of the realised
op mix, NOT invariance of the JOINT op×slot policy. LOO is a PER-SEED feature; a flat op marginal
is consistent with Tamiyo fossilising at a fixed global rate and using the SLOT head to pick WHICH
seed by LOO (`P(op=FOSS | selected-seed LOO)` flat while `P(op=FOSS, slot=s | state)` LOO-sensitive).
Do NOT bank "the policy ignores LOO / the fate decision is independent of LOO." Adjudicator below.

gpt-prime's key discriminator (realised-action overlap ≠ policy-preference flatness). At HOLDING
(FOSSILIZE and PRUNE both legal, `op_masked==false`), binned by current LOO, both seeds:

| LOO bin | op_entropy | op_conf | realised op mix (SET_ALPHA / FOSSILIZE / WAIT / PRUNE) |
|---|---|---|---|
| <0 | 0.88 | 0.35 | 55 / 17 / 15 / 13 |
| 0–1 | 0.88 | 0.35 | 55 / 18 / 14 / 13 |
| 1–5 | 0.88 | 0.36 | 58 / 17 / 14 / 11 |
| 5–15 | 0.87 | 0.36 | 55 / 17 / 15 / 13 |
| ≥15 | 0.87 | 0.35 | 54 / 18 / 16 / 11 |

**The op-head's entropy, confidence, AND realised action mix are FLAT across the entire LOO range**
(both seeds). A seed contributing <0 and a seed contributing ≥15 acc-pts get the same treatment:
~55% alpha-retune, ~17% fossilize, ~13% prune. This is gpt-prime's case 2 — **current LOO is not
the axis of the HOLDING op decision** — and it directly EXPLAINS the fossil/HOLD-prune overlap:
the fates share one LOO distribution because the fossilize-vs-prune choice is made independently of
LOO. Note the dominant HOLDING op is SET_ALPHA_TARGET (~55%), not commit (~17%) or prune (~13%):
the policy overwhelmingly re-tunes rather than commits, LOO-invariantly.

Caveats (do not overclaim): (a) this is MARGINAL invariance (LOO averaged over other state) — LOO
could still matter conditional on other features and average out, though the completeness of the
flatness across a >15-pt swing argues against much hidden signal; (b) these are PER-DECISION op
rates, not per-seed terminal rates (a seed gets many HOLDING decisions); (c) n=2 trained
controllers; (d) it says LOO is not the axis, NOT that the decision is illegible — the policy may
key on host trajectory / slot pressure / LSTM history / freshness / downstream plans. The natural
next read is claudeweb's: what DOES separate fossilize from prune (age, epochs-in-HOLDING,
occupancy, blueprint, resident count, α history, host-acc trend)?

**Adjudication (round 3, both primes' concerns tested observationally), both seeds:**
- **gpt-prime's slot-hides-LOO: substantially addressed.** The slot head is FORCED (slot_entropy
  ≈0) in **97-98%** of HOLDING op-decisions — it is not spreading probability by LOO. And the
  decisive check: if the slot head selected the high-LOO seed for FOSSILIZE and a low-LOO seed for
  PRUNE, the realised cohorts would SEPARATE on selected-seed LOO — but they don't (10.4 vs 9.6).
  So slot selection is not hiding a LOO signal. (Gold-standard confirmation = direct LOO
  perturbation on a checkpoint; the observational evidence already argues against it.)
- **claudeweb's "unlearned static head": refuted in the strong form, refined.** The op mix DOES
  move with occupancy: 1-seed → ≥3-seed shifts FOSSILIZE 18%→22%, SET_ALPHA 49%→56%, PRUNE
  17%→12% (reproducible both seeds). It is NOT a fixed init-prior multinomial. BUT it stays flat
  across host-accuracy and hold-duration, AND flat across LOO. **Conclusion: the HOLDING op head
  is conditional on SLOT/OCCUPANCY PRESSURE but NOT on the seed's contribution.** The precise claim
  is "the VALUE dimension of the HOLDING decision is unlearned/underweighted," not "the HOLDING
  decision is unlearned." This still explains the ~10% negative-current-LOO fossilizations: the
  fossilize rate is set by occupancy, not down-weighted for negative LOO.
- **Definitive remaining test (gpt-prime's gold standard):** load a checkpoint + real LSTM state,
  hold all features/masks fixed, sweep the selected seed's LOO feature (and separately its LOO
  history through the recurrence), measure op/slot/joint logit change. That tests whether the
  TRAINED NETWORK uses the LOO feature at this decision, vs the observational marginal.

## CODE TRACE (drl-expert, key lines re-verified) — the finding is LEARNED/FLOORED, not structural

A code trace (not telemetry) settles what the observational reads could not:
- **The op head CAN see the LOO.** `counterfactual_contribution` is a per-slot input feature at
  `slot_offset+12`, normalized to [-1,1] (features.py:827-832, verified), with velocity (+13),
  freshness (+29), and `current_alpha` (+11) companions. It flows through the single shared
  `feature_net→LSTM` trunk that feeds ALL heads incl. op (factored_lstm.py:820-845). So "the op
  head is STRUCTURALLY blind to LOO" is REFUTED — the flatness is learned/floored behaviour, not
  an architecture gap. (The op head is GLOBAL — one op distribution for the whole board — and LOO
  is 1 of ~32 per-slot features, so dilution is plausible; op↔slot coupling is mask-level only.)
- **Anti-WAIT-collapse floors — the EXACT transform matters (both primes caught a naive-floor
  over-read of mine).** Documented collapse-to-99.9%-WAIT history (leyline/__init__.py:296-299);
  fix = op probability floor 0.15 (__init__.py:300-301) + entropy floor 0.30 (238-242). But the
  floor is NOT a naive per-action clamp — it is **floor-PRESERVING renormalization**
  (`_apply_floor_to_logits`, action_masks.py:522-581, READ): softmax → `effective_floor =
  min(0.15, 0.99/num_valid)` → **underweight** actions (raw prob < floor) set to EXACTLY the floor;
  **overweight** actions (raw ≥ floor) **scaled DOWN by factor `(1−|U|f)/Σ_H p < 1`** (relative
  proportions preserved), capped at `1−(n−1)f ≈ 0.55` for 4 legal ops; applied after masking.
  - **Algebra correction (gpt-prime, verified):** overweight actions are scaled DOWN to fund the
    raised floors, NOT amplified — my previous "amplify" was WRONG (my example had 3 "underweight"
    actions summing to >|U|·f, which is impossible). So the floor DOES cap per-state selectivity at
    ~0.55. Realised frequencies ≠ policy probabilities, and the aggregate 55/17/15/13 mixes decisions
    with different legal sets — so it alone CANNOT identify the raw policy (gpt-prime's caution).
  - **Resolves the arithmetic (PRUNE 13% < 0.15):** PRUNE needs age ≥ MIN_PRUNE_AGE=5, so it is
    MASKED (0 mass) in young-HOLDING decisions; 13% = mean of {0 masked, ≥floor legal}.
- **RETRACTED over-read #9** ("policy is AT the floor / floor provides ALL fossilize mass"): false —
  FOSSILIZE is overweight in some decisions (`alternatives` showed 0.25 > floor). Bankable status:
  *the realised HOLDING mix is consistent with floor saturation on the underweight actions; the mix
  alone cannot identify the raw policy.*
- **⭐ THE MECHANISM FINDING (gpt-prime's dead-zone — autograd-CONFIRMED, zero-GPU).** The floor
  transform is in the DIFFERENTIABLE PPO update path (`factored_lstm.py:1533`, `evaluate_actions`),
  not just sampling. A floor-bound (underweight) action's post-floor prob is a CONSTANT
  (`log(effective_floor)`), so **its PPO log-prob has ZERO gradient to the raw logits** — verified:
  FOSS-below-floor → `d logP(FOSS)/d logits = [0,0,0,0]`; FOSS-above-floor → normal gradient (0.668).
  Consequence: **when FOSSILIZE (or any op) is floor-bound, its advantage produces NO direct policy
  gradient to its own logit.** The floor GUARANTEES the action is sampled (≥15% exploration) but
  SEVERS the channel by which its outcomes would teach the policy — "forced exploration you cannot
  learn from." This is a candidate ROOT mechanism for the whole thread: once WAIT-collapse pushed the
  FOSSILIZE logit below floor, good/bad fossilize outcomes cannot climb/lower it via direct PG; it
  can move only indirectly (shared trunk, entropy-floor loss, boundary crossings). It sharpens the
  negative-LOO fossils (the penalty on a floor-forced bad fossil produces zero gradient to stop it)
  and explains why the raw commit preference stayed undeveloped where it's below floor.
  - **Caveats (not overclaiming):** the zero-gradient holds ONLY in below-floor states; above-floor
    FOSS trains normally, so it is not a total freeze — it's a severed DIRECT channel below floor,
    with indirect channels intact. NOT proven that this is the SOLE cause; the counterfactual (make
    the floor gradient-preserving / straight-through, re-run, see if commit-selectivity emerges) is
    the confirming experiment.
  - **This is a FIXABLE code-level pathology:** a straight-through gradient estimator through the
    floor, or gpt-prime's `P(non-WAIT) ≥ ε` constraint (reserve ε, redistribute by learned relative
    probs) — either prevents WAIT-collapse WITHOUT zeroing the individual ops' gradients or mandating
    random irreversible commits.
- **Pre-floor logit sweep — DOWNGRADED from "decisive" (claudeweb's challenge accepted).** With the
  dead-zone confirmed and a strong learned preference already implausible, the pre-floor read is
  mop-up on a narrow residual and would not change the decision. It is NOT worth a GPU re-run. The
  autograd test + binding audit substitute for it at zero cost.
- **Structural reframes to HOLD (hypotheses, not banked):** (claudeweb) "SET_ALPHA is the new WAIT"
  — the do-nothing/penalty-resetting attractor reached after WAIT was clamped; the floors MASK the
  collapse rather than fix it (Dec-bonus → P0-immediate-payment → anti-collapse-floors = a chain of
  symptom-suppression on a policy that doesn't want to commit). (gpt) a per-action floor that forces
  15% mass onto IRREVERSIBLE ops (FOSSILIZE/PRUNE) is a blunt primitive; a `P(non-WAIT) ≥ ε`
  constraint preserving the learned relative distribution would prevent collapse without mandating
  random irreversible commits.
- **Lineage + way-forward correction (claudeweb).** Same op-collapse failure mode as the Dec-2025
  fossilize-incentive diagnosis; floors were the FIX, side effect = ceiling on commitment selectivity.
  BUT: the floor determines the *behaviour* (the sampled mix), NOT the *preference* underneath — and
  the underlying logit preference is what the reward shaped. So "the reward may not be where this is
  determined" was TOO STRONG (my wording, retracted): a learned "prefer-SET_ALPHA / don't-commit"
  logit is still reward-shaped. The pre-floor read tells you whether it's a mild preference amplified
  by exploration mass or a strong one — but either way the reward is implicated in the preference.
- **Two invariants gate everything regardless of how the floor question lands (both primes):**
  `hindsight_credit` is inert (0.005%) and transfer is unmeasurable. No floor read, no logit read, no
  reward re-pricing changes those, and no downstream experiment is scoreable until they are fixed.

### Immediate next steps (both primes converged, ordered)

1. **Checkpoint saving = a HARNESS FIX, default-ON, no PDR / no GPU auth (claudeweb).** The seed41/42
   weights were NOT saved (only Jan `basic_plus` checkpoints exist — different config/reward family).
   Two GPU-days produced no artifact from which any "what did the policy LEARN" question can be
   re-asked — the PDR-0026 failure recurring. Fix in the harness so every future run yields the
   logits for free. Do this regardless of anything else.
2. **Floor-binding audit — INFEASIBLE from current telemetry (gpt-prime Q1 answered).**
   `alternatives` is top-2 ops EXCLUDING the chosen action (action_execution.py:1498), NOT per-legal-op
   post-floor probabilities — so the binding-rate / mandatory-mass decomposition cannot be computed
   from existing events. It needs per-op pre-/post-floor logging, which folds into the re-run (step 4).
   (Near-miss caught here: `alternatives` excluding the chosen action made SET_ALPHA appear absent from
   its own decisions — I nearly banked a false "action_name ≠ op-sample" inconsistency; reading the
   field definition resolved it. The op head DOES weakly prefer SET_ALPHA: op_confidence≈0.40 is
   SET_ALPHA's chosen-prob, WAIT is the usual runner-up, FOSSILIZE/PRUNE sit near floor — which
   SUPPORTS "SET_ALPHA is the new WAIT" and is consistent with the dead-zone.)
3. **DROP path 2 (Jan-checkpoint proxy)** — both primes: different config/reward family, low decision
   value; the transform's capacity to hide a signal is already settled by algebra + the autograd test.
4. **The one GPU arm worth buying (claudeweb): test the floor's NECESSITY, not the pre-floor logits.**
   The 99.9%-WAIT collapse was diagnosed in Dec-2025 under a reward that has since changed ≥twice; the
   floor's necessity was never re-examined. Run ONE arm with the floor made gradient-preserving
   (straight-through) OR replaced by `P(non-WAIT) ≥ ε`. If commit-selectivity emerges → the dead-zone
   was load-bearing and the fix is in hand. If the head still collapses to WAIT → that is the finding:
   the reward, as written, prefers inaction to every intervention, and every patch since Dec has
   suppressed that signal rather than heeded it. Either outcome is actionable; the pre-floor sweep is
   not.
5. **Retained/transferred-value instrumentation stays an INDEPENDENT, active track** — no floor fix
   makes transfer measurable; the two invariants are unaffected.

## ⭐⭐ ROOT-CAUSE SYNTHESIS (round 6, both primes; zero-GPU, both seeds, code-confirmed)

The dead-zone is now MEASURED and the plan corrected. Both seeds 41/42:

**1. The dead-zone blast radius is near-total for the commit ops.** `action_confidence` (=P(chosen
op), verified POST-floor by an exact spike at 0.15): **FOSSILIZE is floor-bound in 93-94% of
HOLDING decisions, PRUNE in 93-95%, WAIT ~70-86%** — i.e. ~94% of fossilize/prune samples produced
ZERO op-head gradient (claudeweb's over-read-#9 un-retracted in quantified form). **SET_ALPHA is 0%
floor-bound (median 0.55) — a genuine LEARNED preference.** So the head CAN learn (SET_ALPHA proves
it); FOSSILIZE/PRUNE specifically are floor-frozen. This weakens the entropy-bonus competing
explanation (uniform-forcing pressure could not let SET_ALPHA reach 0.55).

**2. Stronger maths (both primes, verified):** within a fixed underweight set the partition function
cancels — underweight logits are REMOVED from the transform output entirely (`∂q/∂z_U = 0` for ALL
outputs, not just their own), a piecewise policy-gradient CENSORING region. The PPO ratio for a
floor-bound op is EXACTLY 1.0 (`logp_new=logp_old=log f`) → contributes `A·1` to the surrogate and
ZERO to op-head approx-KL (a second, independently observable symptom: per-head KL biased low). Only
the op-head COMPONENT is zeroed — other heads still train around the frozen op logit.

**3. POLICY-WIDE (gpt-prime, code-confirmed).** `_apply_floor_to_logits` is applied to EVERY head
with a nonzero floor in BOTH the rollout (factored_lstm.py:1104-1106) and update
(1527-1533) legs — all 8 heads (op 0.15, slot 0.05, blueprint 0.12, tempo 0.12, …). So a rare
efficient blueprint below its 0.12 floor cannot be raised by its own good outcomes either — a
candidate for "finds excellent motifs but can't make them dominant."

**4. THE GATE PASSES — the commit reward IS strongly LOO-graded (resolves claudeweb's
invariant-contradiction).** Advantage is not logged per-decision (folds into the re-run), but the
reward-vs-LOO proxy for FOSSILIZE (both seeds): **c<0 → −0.7 ; 0–1 → +0.1 ; 1–5 → +2.0 ; 5–15 →
+4.1 ; ≥15 → +8.2** — a strong monotonic rise. claudeweb argued the arm would be unscoreable because
`hindsight_credit` is dead — but that conflates two channels: the IMMEDIATE commit reward is strongly
LOO-graded (fossilize bonus 0.5+0.1c + contribution shaping); only the DOWNSTREAM transfer settlement
is inert. And because `V(s)` is op-independent (P0-1), a fossilize reward rising this steeply implies
the fossilize ADVANTAGE rises too. So **the learning signal exists; the dead-zone is why the policy
cannot act on it.**

**The coherent root cause:** *a strong, LOO-graded commit reward that the policy CANNOT LEARN FROM,
because ~94% of commit samples are gradient-censored by the anti-WAIT floor.* Explains flat-in-LOO
fossilize, the fossil/prune LOO overlap, SET_ALPHA dominance (the one op that stayed above floor and
learned), and the negative-LOO fossils (a c<0 fossil earns −0.7 but is floor-forced; the penalty
produces zero gradient to stop it). The two problems are SEPARABLE: (A) commit-learning = dead-zone +
live reward → SCOREABLE + fixable in code; (B) transfer/developmental value = `hindsight_credit` dead
→ separate, unaffected by the floor fix.

**Corrected plan (supersedes the earlier "straight-through" and "floor-necessity" wording):**
- **Fix primitive = a DIFFERENTIABLE mixture floor, NOT straight-through** (STE back-props the
  unfloored gradient while sampling the floored dist → biased PPO ratio, the exact class of bug this
  codebase is made of). Options: `q = (1−λ)·softmax(z) + λ·uniform(legal)` (claudeweb) or gpt-prime's
  smooth `P(non-WAIT) ≥ ε` that reserves ε of WAIT's mass and redistributes by LEARNED relative probs
  (preserves relative preferences, no per-op floor on irreversible actions). Both give exact `log q`
  for a valid ratio and non-zero gradient everywhere.
- **Separate the two GPU questions (gpt-prime):** Stage-1 = a short NO-FLOOR necessity smoke (current
  reward, floor disabled, abort if WAIT-collapse recurs) — tests whether the Dec-2025 pathology still
  exists under the twice-changed reward. Stage-2 = paired hard-floor vs mixture-floor A/B (n=5), only
  if anti-WAIT is still needed. Do NOT call a mixture-floor arm a "necessity" test.
- **Zero-GPU reads still to run:** the entropy-coefficient audit (is the bonus on post-floor probs;
  coefficient magnitude — rule out the residual competing explanation); whether floor-bound (zero-
  gradient) transitions still enter global advantage standardisation (scale distortion); per-head
  floor-binding rates across all 8 heads (blueprint especially).
- **Re-run must add (gpt-prime):** pre/post-floor per-op logits + per-decision advantage + periodic
  checkpoints (weights + optimiser + normaliser + RNG + config), so the advantage-vs-LOO GATE and the
  pre-floor policy are directly readable next time.

## ROUND 7 — the dead-zone is WHOLE-HEAD (|H|=1), and the plan gains a co-requisite

**Reshaping (claudeweb, confirmed both seeds).** Take the transform to |H|=1 (one op above floor):
the output vector is a CONSTANT `[1−(n−1)f, f, …]` — gradient to EVERY op logit is zero, including
the "winner". **|H|=1 holds in 67% (s41) / 80% (s42) of HOLDING decisions** (reconstructed from
`action_confidence` + `alternatives`). SET_ALPHA's median 0.55 is the analytic CAP `1−3·0.15`, not a
learned mode — it sits at exactly 0.55 in 48-61% of SET_ALPHA choices. **RETRACT the round-6 "SET_ALPHA
is a genuine learned preference":** in the majority of HOLDING decisions the WHOLE op head emits a
constant and learns nothing; it receives gradient only in the ~20-33% minority where |H|≥2. This is a
better explanation for a structureless op head than a FOSS/PRUNE-specific freeze, and it makes the
dead-zone bigger. (The 94% FOSS-floor-bound and 67-80% |H|=1 reconcile: FOSS is rarely the single
overweight op, so it's floored ~94% of the time; the chosen op equals the cap-value op only ~half the
time within |H|=1, hence 35-42% "at cap".)

**Per-head KL not logged** (only aggregate `approx_kl`, emitters.py:471) — the "op-head KL biased to
zero" free confirmation both primes wanted is NOT available from telemetry; folds into the re-run.

**Sequencing correction — the floor fix is SCOREABLE but NOT SHIPPABLE alone (claudeweb #4, gpt).**
`hindsight_credit` (the downstream settlement telling the policy a +8.2-rewarded fossil didn't
transfer) is inert (0.005%). Unfreeze the gradient and the policy will learn to optimise a MYOPIC
proxy — greedily fossilise high CURRENT LOO — with no downstream correction → plausibly trading
flat-fossilise for **LOO-greedy commit collapse** (a new attractor, reached faster, with the
optimiser behind it). So the dead-zone may be an accidental SAFETY property. **Revised invariant:
the GPU arm is scoreable (the gate proves it); the floor fix is NOT shippable until the transfer
settlement carries signal.** Promote `hindsight_credit`/transfer from "separate track" to
**CO-REQUISITE of the floor fix** (still separate ENGINEERING, but a shipping gate).

**Gate caveat (both primes):** the −0.7→+8.2 is IMMEDIATE reward; advantage `A ≈ r + γV(s′) − V(s)`,
and `V(s′)` is NOT op-independent (fossilising CONSUMES the seed → V(s′) falls for high LOO,
partially cancelling the graded r). The 9-pt spread survives, but LOG per-decision advantage in the
re-run rather than continuing to infer it. Also reframe "the reward was doing its job" → **"a strong
LOO-graded commit reward the op head usually cannot DIRECTLY learn from"** (the immediate shaping
signal works; the reward's ability to value transfer/turntabling is unresolved).

**Experiment — ONE primitive, λ-sweep (claudeweb #5, gpt).** The mixture floor `q =
(1−λ)·softmax(z) + λ·uniform(legal)` reproduces the current guarantee at λ=0.6 (floor 0.15, cap
0.55) and IS the no-floor condition at λ=0, with `∂q/∂z = (1−λ)·J_softmax` — never zero, even at
|H|=1. So build ONE primitive and sweep **λ ∈ {0, 0.3, 0.6}**: λ=0 is the necessity smoke, the rest
is the fix — one code change, one parametrised arm, NOT two builds. (gpt's differentiable uniform
floor `q_i = f + (1−nf)·p_i` is an equivalent minimal-causal-isolation form; the smooth `P(non-WAIT)
≥ ε` is the better long-term product primitive, adopted only after the causal pilot.) **Pre-register:
the λ=0 arm most likely collapses to SET_ALPHA, not WAIT** — SET_ALPHA is the current do-nothing
attractor, so "didn't collapse to WAIT" ≠ "floor unnecessary."

**Scope correction (gpt-prime).** Do NOT claim the floor retroactively erases the prior reward/critic
nulls. Objective-A HRA had its own validated total-fit failure; the Shapley A/B was genuinely
coverage-bound; transfer stays unmeasured. Durable claim: *the hard floor is a newly-confirmed,
high-prevalence (~94% FOSS/PRUNE; ~67-80% whole-head) training-path defect that likely prevented the
op policy from learning commit/discard preferences from most realised outcomes — the leading proximal
explanation for flat HOLDING preferences, but it does not erase the independent reward/critic limits.*

## The calibrated diagnosis (authoritative)

> Tamiyo preferentially fossilises seeds that are currently important to the forward network
> (terminal-current-LOO selection is strongly POSITIVE, not inverted). But current LOO is NOT
> a measure of historical developmental value, so whether the seeds she later prunes were
> failed experiments or successful developmental modulators is UNRESOLVED. Re-blending after
> HOLDING is common; whether it is productive turntabling, recovery, or waste is UNMEASURED
> because HOLDING-origin excursions and their downstream effects have not been isolated.

## BANK (solid)

- **Current LOO shows substantial overlap and no obvious univariate separation between realised
  FOSSILIZE and HOLD-PRUNE** (fresh-confirmed). Identity-clean (stage-on-row), 0% missingness,
  n=2: FOSSILIZE last-measured c median **10.4** (66/65% ≥5, ~10% <0) vs PRUNE-from-HOLDING median
  **9.6** (65/61% ≥5, ~11% <0). BLEND-prune median 5.8; TRAIN-prune unmeasured (exploration).
  The REALISED commit-vs-discard action is not visibly separated by current contribution alone.
  (Terminology per gpt-prime: "substantial overlap," NOT proven "same distribution" — needs
  AUC/Wasserstein + run-level bootstrap. Retracts the "0.12" and "0.84" artifacts — over-read #8.)
- **Both realised fates carry ~10% negative-current-LOO and ~63-65% ≥5.** So there exist
  **negative-current-LOO fossilizations** (~10%) AND **high-current-LOO HOLDING-prunes** (~65%).
  A negative current LOO is NOT proven a harmful product decision, and a high-LOO prune is NOT
  proven a mistake — labelling them "harmful commits" / "leaks" presumes current LOO is the
  correct commit-worthiness target, which is unproven. Cause (developmental enablement / synergy /
  turntable / ransomware / budget / stale / waste) UNRESOLVED — no causal net-ensemble-value field
  exists (`total_improvement` is host-progress, non-causal).
- Current selection is IMPERFECT: substantial above-threshold non-fossilisation and some
  negative-current-LOO fossilisation both exist.
- Re-blending after HOLDING is common (75%); most re-blenders never fossilise (88%).
- Raw germinate/prune churn is largely EXPLORATION, not farming (74k TRAINING-prunes at α≈0).
- **Current LOO (`seed_contribution`) cannot measure historical modulation or seed→seed
  transfer** — a Phase-0-documented limitation, now operational not theoretical.
- Field identity (Read A): `seed_contribution` = LOO marginal in **accuracy points** (0–100).
  Fossilize bonus = `0.5 + 0.1·c` on RAW c → **~1.5 for c≈10** (NOT ≤0.6). `bounded_attribution`
  = discounted `sqrt(progress·c)`-type function × attribution_discount × timing_discount — NOT
  raw c. Decision surface ~43% free; α-throttle refuted; Objective-A rejected, PPO baseline is
  op-independent `V(s)`.

## DO NOT BANK (over-reads, incl. mine this round)

- "38% of above-threshold seeds pruned = a 38% product-quality failure" — it's
  above-threshold-current-LOO non-fossilisation; can include ransomware prunes (positive
  self-LOO, negative ensemble), field-mismatch, deliberate post-help removal, stale telemetry.
- "185–228 negative-LOO fossils = certainly harmful commits" — correct label is
  **negative-current-LOO fossilisation**; a seed can have negative instantaneous LOO yet be a
  developmental enabler / synergistic / noisy. Needs the per-cohort reward + downstream read.
- **(#6, comforting direction) "Productive scaffold retirement / gate selects on the right
  signal" — REFUTED.** Rested on the "0.12 decay" (a `None`-filter artifact) and a two-point
  interpolation across the prune boundary. The lesson repeated in the reassuring direction: a
  tidy benign story is as much a red flag as a tidy alarming one.
- **(#7) The ransomware-vs-leak split by `total_improvement` — INVALID.** `total_improvement`
  is `current_val_acc − initial_val_acc` (host progress since germination), explicitly non-causal
  (slot.py:251-256). Read as "seed's net help to the ensemble" it manufactured a false 68%/32%
  split. Any ransomware/leak claim needs a CAUSAL discriminator, which telemetry lacks today.
- **(#8) "PRUNE-after-HOLD median 0.84 / 30% ≥5 / majority-correct retirement" — ARTIFACT.**
  Caught by the user's hint to check the counterfactual code + a `None`-rate probe:
  `seed_contribution` is `None` on 91% of ALL prune rows — but that is ENTIRELY the 74k TRAINING
  prunes (α=0, no counterfactual). My read carried the last non-`None` value forward across them,
  manufacturing a low median. Gated on stage-at-decision with NO carry-forward, real lifecycle
  prunes have 0% missingness and HOLD-prune median is 9.6, not 0.84. Lesson: check a field's
  null-rate AND its computation condition before taking any statistic of it.
- "Re-blending is mostly loitering" AND "mostly useful turntabling" — both unresolved.
- "The successful modulator receives NO temporal credit" (my over-claim) — TOO STRONG. Direct
  per-seed LOO doesn't pay historical modulation, but an **indirect RL channel exists**:
  upstream action → future downstream/terminal reward → GAE return assigned to the earlier
  action. It is indirect, shared, delayed, discounted, high-variance, and weakened by
  truncation/bootstrapping — NOT absent. An explicit scaffold-hindsight term may still help,
  but the system is not mathematically incapable of learning turntabling. (`hindsight_credit`
  fires 0.005% of the time — so the EXPLICIT settlement channel is de facto inert, which is the
  real gap; ordinary future-return credit remains.)
- Any fixed multiplier for the commitment economics; any constant-`c` crossover.

## Turntabling (owner domain knowledge) + the measured screen

Turntabling is intended design: a HOLDING seed drops to partial α to modulate a downstream
seed's blend, then ramps back to 1.0 and may fossilise; some seeds exist only to modulate. So
the HOLDING excursion may be designed modulation, not loiter. **Effect-size screen
(observational, seed 41):** turntabling is RARE (2% of decisions have ≥2 active seeds — masks
enforce sequential dev); modulated fossils are NOT higher-LOO (co-resident 7.4 / upstream 2.3
vs solo 11.6 / none 11.9) — but this is confounded by LOO-dilution AND LOO's blindness to
transfer, so it is a **weak null, not a refutation**. Clean test = a scripted turntable arm
measuring the DOWNSTREAM seed's outcome.

## The decisive zero-GPU packet (gpt-prime), then stop theorising

- **A. Field/unit identity** — DONE (above): contribution in accuracy points; bonus on raw c.
  Still open: whether extreme c (±45) is a meaningful marginal or an off-manifold break;
  confirm event `counterfactual` == decision-time `seed_contribution`.
- **B. First-HOLDING-to-fate trajectories** — per seed: c at first HOLDING, peak c, c at each
  excursion, c at fate, ∫bounded_attribution after first HOLDING, time-remaining, re-blend
  count/type, fate, ransomware/auto/scheduled status. **Decides whether the 0.12 prune cohort
  was ALWAYS marginal or DECAYED after being useful** — the load-bearing unknown.
- **C. Turntable exposure vs outcome** — downstream seeds' outcomes by upstream partial-α
  exposure, matched on slot/blueprint/round/host-acc/param-budget. Positive association →
  licenses a scripted causal arm.
- **D. Reward settlement paycheck** — hindsight_credit & synergy_bonus by fate; cumulative
  reward after first HOLDING by fate; bonus/warnings/shocks/costs/rent.

## Decision table (after B–D)

| Finding | Implication |
|---|---|
| Prune cohort low FROM first HOLDING | marginal-seed exploration/recovery (benign) |
| Prune cohort high early, downstream/host benefit retained | productive scaffolding; LOO under-credits history |
| Prune cohort high early, no retained value, large reward accrued | attribution/timing problem |
| Turntable-exposed downstream materially outperform controls | build explicit history-aware scaffold credit |
| Turntable exposure no outcome association | close the modulation-credit branch |
| Same-target/no-downstream excursions common | action-hygiene fix |
| Negative-current-LOO fossils lack future benefit | commitment-gate defect |
| hindsight_credit never pays PRUNE | explicit settlement gap |

## Product baseline

Morphogenesis-improves-a-host is NOT in doubt (established, incl. degraded hosts). The open
product comparison is **Tamiyo's learned policy vs a competent scripted turntable/scaffold
controller**, primary metric terminal accuracy contribution under parameter + developmental
-compute cost. all-disabled (resident-seed effect, host learning retained) ≠ host-only control
(total developmental effect); both useful.
