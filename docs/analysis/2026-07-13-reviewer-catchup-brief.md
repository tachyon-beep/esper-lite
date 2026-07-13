# Reviewer catch-up brief — decision-point diagnosis (Esper / Tamiyo)

**For: claude-prime (claudeweb) and gpt-prime. You are each one round behind; this supersedes
both of your last reviews. Read the reconciliation section addressed to you.**

Context in one line: we are diagnosing WHY Tamiyo (an LSTM factored-action RL controller that
grows/prunes neural "seeds" during host training) rarely fossilises (commits) seeds. This brief
covers the last several correction rounds and lands on a clean, both-seeds result plus a verified
telemetry mechanism. Runs: `stage2_on_longdiag/seed{41,42}` (600-round diagnostic, ON leg).
Zero-GPU throughout — pure telemetry reads.

---

## TL;DR (the corrected, current finding)

**On current leave-one-out contribution (LOO), Tamiyo cannot be distinguished by whether she
COMMITS a seed or DISCARDS it from HOLDING.** Measured at the actual decision, identity-clean,
0% missingness, both seeds:

| decision | median last-measured c (acc-pts) | % ≥5 | % <0 |
|---|---|---|---|
| **FOSSILIZE** (commit) | **10.4** | ~66% | ~10% |
| **PRUNE-from-HOLDING** (discard) | **9.6** | ~63% | ~11% |
| PRUNE-from-BLENDING | 5.8 | ~53% | ~12% |
| PRUNE-from-TRAINING (α=0) | *unmeasured — structurally 0* | — | — |

The commit and HOLDING-discard cohorts have **the same contribution distribution.** The gate
leaks on both tails of the same causal metric in the same run: **~10% of FOSSILISED seeds are
net-negative** (harmful commits) and **~65% of HOLDING-prunes are ≥5 acc-pts** (load-bearing
discards). This is consistent with a policy optimising a signal that is *correlated with*
commit-worthiness but is not commit-worthiness — and, critically, cannot measure what the product
is actually for (retained/transferred value).

---

## What changed since your last review (both of you)

This diagnosis swung repeatedly; every miss was a **load-bearing input assumed instead of read**
(never the reasoning). The relevant recent swings:

- **Over-read #6 (banked, then retracted):** "productive scaffold retirement — the gate correctly
  retires seeds whose value the host ABSORBED." Rested on a prune-cohort "contribution 0.12" that
  turned out to be a **`None`-filter artifact**: `SEED_PRUNED.counterfactual` is `None` ~7/8 of
  the time, so a "median 0.12" over the non-`None` sliver never described the cohort. There is no
  gliding decay to 0.12. (This is the conclusion **gpt-prime reviewed** — fully retracted.)
- **Over-read #7 (caught before banking):** split the prune cohort into "ransomware vs leak" using
  `total_improvement`. That field is `current_val_acc − initial_val_acc` (host progress over the
  seed's residency), explicitly warned NON-CAUSAL in-code ("conflates host training gains with
  seed impact"). It manufactured a false 68/32 split. No causal ransomware discriminator exists.
- **Over-read #8 (caught before banking):** "PRUNE-after-HOLD median 0.84 / 30% discard ≥5 /
  majority-correct retirement." (This is the conclusion **claudeweb reviewed**.) Artifact of
  carrying the last non-`None` contribution FORWARD across the 74k TRAINING prunes that have no
  counterfactual. Gating on the decision row's own `seed_stage` with NO carry-forward gives the
  clean table above: real lifecycle prunes have **0% missingness** and HOLD-prune median is **9.6,
  not 0.84.**

---

## The verified telemetry mechanism (load-bearing — read this)

From `vectorized_trainer.py:1365-1394` and `features.py:61-75`, confirmed by owner domain
knowledge:

- `counterfactual_contribution = val_acc − solo_acc` — a true LOO (host+seed minus host+seed-off).
- It is computed **only when a "solo" config is evaluated for that slot**, i.e. for **BLENDING+
  seeds, not every epoch.** An `epochs_since_counterfactual` staleness tracker is reset on each
  fresh measurement, and the policy is fed `freshness = γ^epochs_since_cf` in its observation (obs
  V3). So the value on any decision row is the **LAST-MEASURED** LOO, possibly a few epochs stale.
- **TRAINING runs at α=0** — the seed is not wired into the forward pass at all (it trains in
  isolation to match host errors before BLENDING ramps α up). So a TRAINING seed's counterfactual
  is **structurally zero/undefined**, not a data gap. The 74k TRAINING-prunes are rejections
  during isolated training, before the seed ever enters the computation. Correctly excluded.

Two consequences: (1) the "91% of prunes have no contribution" fact is entirely these α=0 TRAINING
prunes — a red herring for the lifecycle question. (2) The clean numbers are "last-measured"
values; staleness applies to fossils and HOLD-prunes under the same cadence, so the *comparison*
is robust — **unless fossilise decisions act on systematically fresher counterfactuals than prune
decisions** (see residuals).

---

## Reconciliation — gpt-prime (you reviewed the "scaffold retirement" version)

Your review was correct and we acted on all of the free items:
- **"Not bankable: host absorbed value / gate correctly retires absorbed scaffolds / 0.12 = retained
  value / n=1 / peak includes pre-HOLDING / ∫5.29 is the full integral."** — All correct. That
  entire conclusion is retracted (#6). We no longer make any retention/absorption claim.
- **Problem 1 (identity: `(env,episode,slot)` conflates lifecycles).** Valid. Resolved for a
  point-in-time measure by gating on the decision ROW's own `seed_stage` — that is the acting
  seed's stage, so there is no cross-lifecycle carry. (Trajectory reconstruction still needs a
  real seed_id + germination reset; we did not attempt trajectories in the clean read.)
- **Problem 2 (carry-forward; `last` ≠ contribution at fate).** This was the DOMINANT bug (#8),
  not a footnote. Fixed by no-carry-forward + stage gate; real lifecycle prunes are 0% `None`.
- **Problem 3 (snapshot is per-decision, not per-residency).** Valid. The clean headline is a
  point-in-time decision measure, which does not need residency continuity; all ∫/peak/first-HOLDING
  claims are dropped.
- **Problem 5 (falling LOO ≠ retained value).** Adopted verbatim; no retention claim survives.
- **Problem 6 (seed 41 only).** The clean read is n=2 (41 and 42 agree closely).
- **Your step 4 (condition on α)** — still open (residual). **Your step 5 (measure where the value
  went)** — this is now the primary way-forward.

## Reconciliation — claudeweb (you reviewed the "median 0.84 / 30% tail" version)

- **"The median is safe — it's a decision on a real seed; the ~30% survives."** — Refuted by a
  direct probe: 91% of prune rows had NO decision-time contribution (α=0 TRAINING prunes), and my
  median was carry-forward over them. The clean HOLD-prune median is **9.6, not 0.84**, and the
  load-bearing discards are not a 30% tail — they are the **majority (~65% ≥5).** You were
  directionally right that identity corrupts trajectories not point-decisions — but the point
  measure itself was contaminated by missingness, which neither of us had checked.
- **"Distribution is the finding, not the center; the cohort is bimodal."** — Adopted. The clean
  distribution is high-c-dominated (65% ≥5), not balance-bimodal; the negatives are only ~11%.
- **"The gate leaks both ways: ~10% harmful commits, ~30% load-bearing discards → a policy
  optimising a signal correlated with but not equal to commit-worthiness."** — VINDICATED and
  stronger: ~10% harmful commits AND ~65% load-bearing HOLD-discards; the two fates share one
  contribution distribution. This is now the central supported claim.
- **Read C trap (host trains continuously; a rebound proves nothing without a background-training
  control).** — Correct and adopted into the instrument spec below.
- **"Build the instrument before the next arm."** — Adopted as the primary recommendation.

---

## Banked / Not-bankable / Live residuals

**BANKED (clean, both seeds):**
1. Current terminal LOO does NOT separate commit from HOLDING-discard (fossil 10.4 ≈ HOLD-prune 9.6).
2. The gate leaks on both tails: ~10% net-negative fossils; ~65% ≥5 HOLDING-prunes.
3. Counterfactual is computed for BLENDING+ only, staleness-tracked; TRAINING α=0 → structurally
   uncomputed.
4. `hindsight_credit` is inert (fires 0.005% of the time) — the explicit settlement channel for a
   scaffold's durable value is de-facto dead. This has survived EVERY swing, benign and alarming.
5. ~200 negative-LOO fossils exist (harmful commits), unexplained.

**NOT BANKABLE (either direction):**
- Any "the gate selects correctly / incorrectly" verdict — the *cause* of the high-c HOLD-discards
  (ransomware dependency / turntable-retirement / budget-pressure / genuine waste) is not
  separable with current telemetry. No causal net-ensemble-value field exists.
- Absorption / retained value (retracted).

**LIVE RESIDUALS (the only threats left to the headline):**
- **Freshness asymmetry** — if fossilise acts on fresher counterfactuals than prune (a
  confirm-before-commit pattern), the "same distribution" could be partly a staleness artifact.
  Checkable by joining `COUNTERFACTUAL_MATRIX_COMPUTED` timing to each decision.
- **α at the HOLD-prune** — separating full-α load-bearing from turntabled-α (partial) prunes.
- **Where the pruned value went** — the causal question, currently unmeasurable.

---

## The way-forward (convergent across all corrections)

The finding that has NOT changed in four rounds, benign or alarming: **the reward optimises a
signal (current LOO) that (a) does not separate commit from discard, (b) is only defined for
BLENDING+ seeds, and (c) cannot measure retained or transferred value — which is the mechanism the
product exists to produce.**

The single most actionable item is therefore a **schema/instrument finding, not a new reward or
critic arm**: build per-seed causal value instrumentation BEFORE the next experiment —

- host-only (all-seeds-off) accuracy captured at: seed germination, immediately pre-prune,
  immediately post-prune, and at terminal;
- with a **background-training control** (the host's own trend rate just before the event), because
  the host is training continuously and a naive post-prune rebound proves nothing;
- optionally downstream-enablement (does an upstream partial-α seed lift a downstream seed's
  eventual contribution/fossilisation vs matched controls) and compute-to-target.

Rationale: every experiment anyone has proposed (new reward term, critic redesign, HRA, escrow) is
currently **unscoreable on the quantity that matters**, because "did this seed's value transfer
into the host or a downstream seed" does not exist as a measurable field. Fix the instrument, then
run arms.

Durable record: `docs/analysis/2026-07-13-decision-point-diagnosis.md` (commit a748abda).

---

## ROUND 2 UPDATE — the two reads you both prioritised (run after your last reviews)

### (1) Freshness residual — CLOSED (you both flagged it as the #1 threat)

Reconstructed `epochs_since_counterfactual` exactly: the LOO value is written ONLY on a fresh
solo-eval (`vectorized_trainer.py:1390`), so epochs-since-the-value-last-changed == staleness.
Result, both seeds:
- **98-99% of BOTH FOSSILIZE and PRUNE-from-HOLD decisions act on staleness=0** (fresh)
  counterfactuals; ~0% at staleness ≥3. **No confirm-before-commit asymmetry.**
- Decisive matched read (fresh-only, staleness=0): FOSSILIZE median 11.2/10.2 vs PRUNE-from-HOLD
  10.3/9.2 — **the overlap HOLDS; prune-c does not collapse when forced fresh.**
- Stale prunes (s≥3) are n=4-5 and LOW-c (median 0) — the opposite of the stale-high concern.

**The overlap is not a staleness artifact.** (claudeweb bet against the residual and won.)

### (2) Policy-preference read — the op-head decision at HOLDING is INVARIANT to LOO (gpt-prime's key read)

At HOLDING (both FOSSILIZE & PRUNE legal, `op_masked==false`), binned by current LOO, both seeds:

| LOO bin | op_entropy | op_conf | realised op mix (SET_ALPHA / FOSSILIZE / WAIT / PRUNE) |
|---|---|---|---|
| <0 | 0.88 | 0.35 | 55 / 17 / 15 / 13 |
| 0–1 | 0.88 | 0.35 | 55 / 18 / 14 / 13 |
| 1–5 | 0.88 | 0.36 | 58 / 17 / 14 / 11 |
| 5–15 | 0.87 | 0.36 | 55 / 17 / 15 / 13 |
| ≥15 | 0.87 | 0.35 | 54 / 18 / 16 / 11 |

**Op-head entropy, confidence, and realised action mix are FLAT across the entire LOO range.** A
seed contributing <0 and one contributing ≥15 acc-pts get identical treatment. This is gpt-prime's
**case 2** — current LOO is not the axis of the HOLDING op decision — and it directly EXPLAINS the
fossil/HOLD-prune overlap (the fates are chosen independently of LOO), rather than just restating
it. The dominant HOLDING op is SET_ALPHA_TARGET (~55%), not commit (~17%) or prune (~13%): the
policy overwhelmingly re-tunes, LOO-invariantly.

Caveats kept honest: marginal (not conditional) invariance; per-decision (not per-seed terminal)
rates; n=2 controllers; "LOO is not the axis" ≠ "the decision is illegible" (may key on host
trajectory / slot pressure / LSTM history / freshness / downstream plans).

### (3) CODE TRACE — the mechanism (verified against source, not telemetry)

You both correctly warned the op-marginal ≠ the joint policy, and that "flat" could be structural,
learned, or a masking artifact. A drl-expert code trace (key lines re-verified by hand) resolves it:

- **Op×slot factorisation:** all 8 heads are conditionally-independent categoricals off ONE shared
  `feature_net→LSTM` trunk; op↔slot coupling is MASK-level only (no logit conditioning). The op head
  is GLOBAL — one op distribution for the whole board, not a per-seed decision.
- **The op head CAN see LOO.** `counterfactual_contribution` is a per-slot input feature
  (`features.py:827-832`), normalized ±1, feeding the shared trunk → all heads incl. op. So "op is
  structurally blind to LOO" is REFUTED — the flatness is LEARNED/FLOORED behaviour, not architecture.
  (Also addresses gpt-prime's slot-hides-LOO: slot head is FORCED — slot_entropy≈0 — in 97-98% of
  HOLDING decisions, and if it selected high-LOO for FOSS / low-LOO for PRUNE the cohorts would
  separate; they don't. And claudeweb's unlearned-static-head: refuted — the op mix DOES move with
  occupancy, 1→≥3 seeds FOSS 18→22% / SET_ALPHA 49→56%; it's conditional, just not on LOO.)
- **The load-bearing mechanism — anti-WAIT-collapse floors.** The op head has a DOCUMENTED history of
  collapsing to 99.9% WAIT (`leyline/__init__.py:296-299`). The fix: a HARD **op probability floor =
  0.15** (`__init__.py:300-301`) + **entropy floor = 0.30 normalized** (`238-242`). In the HOLDING mix,
  FOSSILIZE ~17% / WAIT ~15% / PRUNE ~13% all sit AT the ~15% floor; SET_ALPHA ~55% holds the free
  mass. The 0.15 floor over ~4 legal ops CAPS fossilize at ~55%, and the policy is at the FLOOR not
  the cap — so the learned op-logit for FOSSILIZE is at-or-below the floor: **the policy learned to
  prefer SET_ALPHA and NOT fossilize, and the floor PROVIDES the ~15% fossilize mass as
  state-independent exploration — flat in LOO by construction.** This explains, in one mechanism, the
  flat-in-LOO fossilize rate, the fossil/prune overlap, AND the ~10% negative-current-LOO
  fossilizations (the floor forces ~15% fossilize even on a harmful seed, every decision).
- **Still open:** cannot separate "value dimension never learned" from "training-time pressure
  suppressed it" — the checkpoint LOO-sweep (floors ON vs bypassed) separates them; and HOLDING's
  share of the op gradient (rarity-dilution magnitude) needs the HOLDING step fraction (telemetry).
- **Lineage & way-forward reframe:** this is the SAME op-collapse failure mode as the Dec-2025
  fossilize-incentive fix. The floors were the fix; a side effect is a ceiling on commitment
  selectivity. **A reward redesign that ignores the floors is pushing against a hard clamp** — so the
  instrument-first recommendation now has a sibling: understand the floor/collapse dynamics before
  re-pricing commitment. `hindsight_credit` inert (0.005%) unchanged.

### (4) FLOOR-TRANSFORM CORRECTION — you both caught the naive-floor arithmetic (answers your Q1/Q2)

You both flagged that a naive 0.15-per-action floor cannot yield 55/17/15/13 (PRUNE < floor). Correct.
I read the exact transform (`_apply_floor_to_logits`, action_masks.py:522-581):

- **Q1 — floor form:** NOT a naive per-action clamp. It is **floor-PRESERVING renormalization**:
  softmax → `effective_floor = min(0.15, 0.99/num_valid)` → **underweight** actions (raw < floor) set
  to EXACTLY the floor; **overweight** actions (raw ≥ floor) **scaled to fill remaining mass, relative
  proportions preserved — NOT capped**; applied after masking.
- **Resolves the arithmetic:** PRUNE 13% < 0.15 because PRUNE is MASKED (age < MIN_PRUNE_AGE=5) in
  young-HOLDING decisions → 0 mass; 13% is the mean of {0 masked, ≥floor legal}. 55/17/15/13 is NOT a
  floor-simplex extreme.
- **This cuts AGAINST the "floor masks learnt selectivity" hope, for the STRONG case:** because
  overweight actions AMPLIFY (not cap), a strong learnt "fossilize high-LOO" preference would SURVIVE
  post-floor. Binned post-floor FOSSILIZE is flat in LOO → where FOSSILIZE is overweight it does not
  rise with LOO → the floor is not hiding a *strong* LOO preference. **Only a WEAK, entirely
  sub-floor LOO gradient** (raw drifting below 0.15) would be clamped flat and masked — that is the
  one live learned-selectivity hypothesis, and the pre-floor read settles it.
- **RETRACTED (my over-read #9):** "the policy is AT the floor / the floor provides ALL fossilize
  mass." False — FOSSILIZE is overweight in some decisions (`alternatives` showed 0.25 > floor) and
  the floor doesn't cap overweight. Bankable wording (gpt's, refined by the transform): *the realised
  HOLDING mix is consistent with floor saturation of the UNDERWEIGHT actions with SET_ALPHA holding
  the free mass; a STRONG learnt LOO-fossilize preference is ruled out post-floor, a WEAK sub-floor
  one needs the pre-floor read.*
- **Q2 — pre-floor logits accessible?** YES via a checkpoint forward pass: `op_logits` are computed
  (factored_lstm.py:820-845) BEFORE `_apply_floor_to_logits` (:899/:1152). They are NOT in telemetry
  (only post-floor entropy/confidence/`alternatives` are emitted). So the decisive read is the
  checkpoint sweep (raw → pre-floor → post-floor, floors ON vs bypassed, + a LOO-history/recurrence
  variant), not a telemetry line.
- **Way-forward correction (claudeweb):** the floor sets the BEHAVIOUR, the reward shapes the
  PREFERENCE underneath — so "the reward may not be where this is determined" was too strong; a learnt
  "prefer-SET_ALPHA" logit is still reward-shaped. And the two invariants gate everything regardless:
  `hindsight_credit` dead, transfer unmeasurable — no experiment is scoreable until those are fixed.

**Net after round 2:** the headline is now two-sided and much harder to dismiss — the realised
cohorts overlap on LOO (fresh-confirmed), AND the policy's op distribution is flat in LOO. The
open question flips from "is the overlap real" (yes) to **"what DOES the HOLDING decision key on?"**
(claudeweb's list: age, epochs-in-HOLDING, occupancy, blueprint, resident count, α history,
host-acc trend). `hindsight_credit` inert (0.005%) and instrument-first way-forward both unchanged.
Durable: diagnosis doc through commit (policy-preference read).
