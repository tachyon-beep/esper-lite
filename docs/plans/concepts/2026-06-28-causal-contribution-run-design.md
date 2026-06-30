# Parallel-Control Causal-Contribution Run — Design (DRAFT, pending sign-off + specialist review)

Status: CONCEPT — DUAL-REVIEWED 2026-06-28, **NEEDS REVISION (blocking findings) before sign-off / build.** NOT launch-ready. See "Dual-review synthesis" at end.
**SUPERSEDE PENDING:** a revised v2 candidate exists — `2026-06-28-causal-contribution-run-design-v2.md` — but its load-bearing §0 change (delete the placebo arm, pivot to the TOTAL-SYSTEM estimand, reducing Goal-1's claim) is **UNRATIFIED by the owner**. Do not treat v1 as superseded until that scope call is made.
Date: 2026-06-28 · Designed by: drl-expert (advisor-vetted) · Tracker: relates esper-lite-a221da47ea (GATE −1 SURVIVE), esper-lite-3d67b09687.
Required reviewers before approval (CLAUDE.md): drl-expert + morphogenesis-reviewer + governor-design-reviewer + pytorch-expert.

## Purpose (two goals, one run set)
1. CAUSALLY resolve the (a)/(b) fork — do committed early-conv r0c0 stems (neg per-seed LOO marginal)
   causally ENABLE downstream structure that forms AFTER they commit (developmental-(b)), or contribute
   nothing (freeloader-(a))? Existing removal-cost telemetry is structurally blind to this (post-commit
   the stem is in the always-on baseline; stems commit before neighbours co-reside). GATE −1 just
   SURVIVED (cheap rescale levers don't fix the inefficiency), so the redesign is justified.
2. Produce the deferred CONTRIBUTION-GATE / ADMIT evidence: committed-per-param counterfactual value J
   of the controller's structure vs a true off-switch baseline.

## §1 Identification strategy — whole-run, matched-seed parallel control
For each seed 41–45: run the controller normally but with a runtime intervention that makes the r0c0
stem un-committable; compare to the matched control. The host RE-FORMS its downstream under the
intervention, so the contrast is exactly: factual (stem commits → downstream D → acc A) vs counterfactual
(stem absent → controller builds D′ → acc A′). D′≈D & A′≈A ⇒ freeloader (a); D′ materially smaller & A′<A
⇒ developmental enabling (b). Controller adaptation (re-germinate elsewhere) is PART of the causal pathway,
measured and absorbed by a per-param value metric — not a confound to eliminate.

Snapshot-branch REJECTED (verified infeasible): resume restores only the controller (vectorized.py:1105-1142);
no serialization of the 5 dynamic-topology host models / per-env optimizers / host RNG / LSTM hidden /
rollout buffer; static_final_replay is FOSSILIZED-only (static_final_replay.py:46-50,122-145). Even if
built, the shared recurrent controller diverges within a few steps, so matched-RNG buys only a short-horizon
counterfactual — wrong for a long-horizon developmental question. The matched-seed whole-run pairing
(shared controller init, env seeds, data order; content-addressed lifecycle RNG, action_execution.py:356-365)
is the pairing that survives to the horizon we care about.

## §2 Arms, matching, sample size
Unit of analysis = SEED (41–45). Each run = 12 envs × 200 episodes. Median-led across 5 seeds; paired
block-bootstrap B=20000 with SEEDS as the unit (never vec-envs).

| Arm | New runs | Mechanism | Purpose |
|-----|----------|-----------|---------|
| CONTROL | 0 (reuse, GATED) | existing shaped 41–45 | factual reference |
| SUPPRESS-SLOT (r0c0) — PRIMARY causal | 5 | stationary mask: r0c0 unavailable to all ops whole-run (no actor-loss change) | "stem never exists" developmental contrast |
| SUPPRESS-COMMIT (r0c0) — commitment-specific | 5 (or reallocated) | force-PRUNE at r0c0 commit + latch off re-germinate; per-env forced-flag → excluded from actor loss | isolate the COMMIT act from transient scaffolding |
| OFF_SWITCH — GOAL-2 baseline | 5 | force_wait_only (EXISTS, proof_baselines.py:57-68) | host-alone J denominator |

CONTROL reuse gated on BOTH: (1) build-skew — intervention must be bit-identical-when-off; verify by
reproducing one control seed exactly on the intervention build, else re-run all 5 control fresh.
(2) completeness — control s44 = 2389 EPISODE_OUTCOME (incomplete); re-run or use a completeness-robust
final-acc estimator (last fully-populated episode) consistently across arms.

## §3 Pre-registered decision rule (committed BEFORE running)
Power (computed from existing control variance): control between-seed final-acc SD ≈ 1.0pp (floor-compressed);
paired Δacc SD proxy (GATE −1 arms) ≈ 0.68pp ⇒ MDE(n5) ≈ 0.86pp if pairing holds, up to ~2.14pp if it
breaks. Expected (b) effect ≈ 1–3pp (off-switch headroom +6.1pp × r0c0=54% of fossils). A clear (b) drop
(≥2pp) is detectable at n=5; a TIGHT (a) accuracy-null may be unreachable even at n=10.

PRIMARY discriminator = DOWNSTREAM STRUCTURE (well-powered; control foss/ep SD ~0.005):
  Δ_struct = (committed params at r0c1+r0c2)_arm − _control (+ downstream fossil count, blueprint mix).
  (b) signature: median Δ_struct < 0 with paired-bootstrap 95% CI upper bound < 0.
SECONDARY = accuracy (precision-targeted): Δacc median, paired bootstrap. Stopping rule: run SLOT n=5;
  extend to n=10 (reallocated COMMIT budget) iff n=5 CI half-width > 1.0pp AND median in (−2.0,+1.0); cap n=10.
  Honest fallback (pre-committed): report "at achieved precision ±h, developmental enabling ≥X is/ isn't
  ruled out" — no bare "inconclusive".

| Outcome | Δ_struct | Δacc | Relocation | Verdict | Reward implication |
|---------|----------|------|-----------|---------|--------------------|
| (b) developmental enabling | <0, CI<0 | ≤−2.0pp, CI<0 | — | controller right; LOO mis-measures via developmental synergy | synergy/hindsight credit |
| (b′) direct contributor | spans 0 | ≤−2.0pp, CI<0 | — | stem's own output carried acc; co-resident LOO compression | fix the marginal estimator, not a synergy term |
| relocation / fungible | ≥0 or Δparams>0 | ⊂(−1,+1) | TRUE | function valuable, position-fungible | position-invariant credit; do NOT penalize |
| (a) freeloader | spans 0 | ⊂(−1,+1), \|med\|<0.5 | FALSE | contributes nothing yet keeps committing | penalize survival-without-contribution |
| inconclusive | — | per fallback | — | underpowered/conflicting | named follow-up |

RELOCATION operationalized (pre-registered, no post-hoc judgement): under SUPPRESS-SLOT, flag TRUE iff a
seed commits at r0c1 that is (i) early-conv blueprint family, (ii) commit-timing percentile ≤25th of the
run's fossils, (iii) absent at r0c1 in the matched control. All thresholds fixed before running.
NON-DEGENERACY health check (pre-registered): a SLOT acc drop counts as (b) only if the controller is
otherwise healthy (≥50% of control's r0c1/r0c2 fossil count; churn in-band). Degenerate (near-zero
downstream commits / runaway churn) ⇒ controller-collapse artifact, NOT banked as (b).
SLOT-vs-COMMIT cross-check: agree ⇒ robust; SLOT drops but COMMIT doesn't ⇒ transient scaffolding (not the
commit act) enables ⇒ credit transient presence.

## §4 Metrics
Causal contrast (paired vs control): Δ committed params + Δ fossil count at r0c1+r0c2 (+ blueprint mix,
re-keyed seed_lifecycle 482/482 join); Δ final committed_val_acc (vectorized_trainer.py:1458-1463);
adaptation (r0c0 re-germinate/re-commit attempts, relocation flag, churn).
Contribution-gate J: J_controller = (acc(CONTROL) − acc(OFF_SWITCH))/committed_seed_params;
J_r0c0 = (acc(CONTROL) − acc(SUPPRESS-SLOT))/r0c0_committed_params (the true causal marginal LOO mis-measures).
Denominator = added seed params (PIN-B, slot.py:1298 effective_seed_params), consistent across both J's.

## §5 Minimal code changes + budget
1. Two interventions in apply_proof_baseline_action_controls (action_execution.py:247), keyed on
   slot_config.index_for_slot_id("r0c0"): suppress_slot_r0c0 (zero slot_by_op + GERMINATE→r0c0; stationary,
   no actor-loss change); force_prune_on_commit_r0c0 (mask FOSSILIZE + force PRUNE when r0c0 in HOLDING;
   per-env latch + per-env forced-flag via the existing forced_batch_cpu path — NOT batch-level
   proof_controlled_step, which would zero all envs' gradients).
2. Two ProofBaselineMode members + cohort entries (OFF_SWITCH already exists).
3. Telemetry: INTERVENTION_APPLIED event per forced/masked step (slot, op, reason, count) — real telemetry,
   build it, do not defer. Per-run r0c0 vs downstream committed params.
4. Three configs cloned from config-3slot-3seed-baseline-shaped.json + a seeds-41–45 driver.
Budget (≤2 concurrent, ~6h/run, ≤16GB): minimal conclusive = SLOT(5)+OFF_SWITCH(5)=10 runs ≈30h (5 waves);
full = +SUPPRESS-COMMIT(5)=15 ≈45–48h; precision branch reallocates COMMIT budget to SLOT n=10 if needed;
optional control s44 re-run (+1).

## §6 Risks / what it CANNOT establish
Risks: power/floor-compression (TOP — mitigated by structure-primary + precision-targeted rule + honest
fallback); build-skew on reused control (gated); completeness s44 (gated); online-learning contamination
(COMMIT excluded from actor loss; SLOT's stationary mask yields a different learned controller — intrinsic,
separated from collapse by the non-degeneracy check); positional-vs-blueprint (keys on r0c0 position; 54%
early-conv ⇒ positional dominates; blueprint-conditioned arm is the refinement); small-n flips (median-led,
paired, pre-registered, structure-primary, operationalized relocation).
CANNOT establish: the MECHANISM of enabling (feature reuse vs optimization-path vs regularization); that a
DIFFERENT reward makes r0c0 useful (tests current shaped controller — the puzzle is a shaped-controller
behavior); transient-vs-committed split with SLOT alone (why COMMIT is included); a clean weight-level
ceteris-paribus counterfactual (only the system-level one, which is the relevant one since the redesign
also runs online).

## Open gaps to close before finalizing (drl-expert info-gaps)
1. Realized SLOT paired-Δacc SD (unknown until first 5 SLOT runs; sets n=10 trigger).
2. Per-slot r0c1/r0c2 committed-params variance across control seeds (extract before fixing Δ_struct threshold).
3. OFF_SWITCH wall-time (may be <6h; refines budget).
4. Whether r0c0 ever hosts non-early-conv blueprints (decides if a blueprint-conditioned arm is needed).
Pre-build checklist: bit-identical-when-off check; control completeness handling (s44); extract r0c1/r0c2
params on control to fix Δ_struct threshold; four reviewer sign-offs.

## Dual-review synthesis (2026-06-28) — morphogenesis-reviewer + governor-design-reviewer
Verdict: morphogenesis = CONDITIONALLY FIT, currently UNFIT for Goal-1 as written; governor = NO
safety-independence violation, but a scientific-validity confound + telemetry gap. Net: NOT sign-off-ready;
needs a v2. The reviews were load-bearing — as written the run could spend 30–48 GPU-h producing a Δ_struct<0
read as developmental-(b) that is actually a controller-trained-on-a-smaller-action-space artifact (the
expensive false positive), or a rollback-induced false-(a).

### BLOCKING (must fix before any launch)
1. **Action-space confound / no negative control (morpho, Goal-1).** SUPPRESS-SLOT re-optimizes the controller
   over an r0c0-less manifold (those steps are NOT forced → included in actor loss), so Δ_struct can move from
   action-space reduction alone; the ≥50% non-degeneracy floor sits in the danger zone. → ADD a frequency-matched
   NEGATIVE-CONTROL/placebo arm (mask an equally-often-committed but developmentally-inert option, or hold action
   cardinality constant with a dummy) to bound the artifact. A masked-late-slot (r0c2) placebo alone UNDER-bounds
   (r0c0 = 54% of fossils) and is not acceptable by itself.
2. **(b)-cell contradiction (morpho).** Prose says structure-primary but the (b) row still requires Δacc≤−2pp
   (the underpowered metric). Reconcile: make (b) = Δ_struct<0 (CI<0) AND beats placebo AND non-degenerate, with
   accuracy as CONFIRMATION not gate — or drop the "structure solves power" framing. Cannot have both.
3. **OFF_SWITCH seed pairing (morpho, Goal-2).** OFF_SWITCH default training seed = base_seed+10_000
   (proof_baselines.py:60), NOT 41–45 → J_controller would be UNPAIRED vs the paired J_r0c0. Override to seeds 41–45.
4. **Per-arm governor telemetry + rollback-parity gate (governor).** Capture GOVERNOR_ROLLBACK count/rate,
   panic-reason breakdown, preflight-veto counts by blocked_factor, snapshot cadence, + INTERVENTION_APPLIED
   (join-keyed to causal-log action_id). Pre-register: rollback≈0 across arms ⇒ assert Q3 non-issue with evidence;
   else treat as confound.
5. **Rollback-episode exclusion for SUPPRESS-COMMIT (governor).** Force-pruning a load-bearing r0c0 most likely
   trips panic → execute_rollback prunes ALL live seeds + resets (governor.py:500-503), destroying the Δ_struct
   measurement and masquerading as false-(a). Any rollback episode = distinct category, NOT pooled; add an explicit
   rollback-incidence cap to the non-degeneracy check.

### HIGH (fix before relying on results)
6. **Separate RNG streams (morpho, Disc-1/power).** Single global torch.manual_seed (vectorized.py:1148) seeds
   controller + host + blueprint init; the content-addressed rng_seed (action_execution.py:356-365) is TELEMETRY-ONLY
   (consumed by no generator) — the design's pairing justification citing it is wrong. Add a per-env host-stochasticity
   torch.Generator to remove divergence-induced host-RNG desync; re-estimate paired-Δ SD. n=5 is below the
   morphogenesis floor of 10 — treat as provisional; the n→10 trigger must consider seed-level robustness, not just
   within-seed foss/ep SD.
7. **Relabel J_r0c0 (morpho).** It is a system-level, net-of-substitution marginal (includes re-formed D′), NOT a
   per-param marginal — under full relocation its numerator→0, contradicting the "valuable-but-fungible" reading; units
   mismatched. Relabel + pin the committed-param denominator to ONE definition (the doc conflates slot.py:1298
   active_seed_params with helpers.py:131-166 effective_seed_params).
8. **Forced-PRUNE must leave alpha heads policy-sampled (governor verify).** Preflight rejects PRUNE if
   alpha_speed_steps/alpha_curve are None (governor.py:183-191) — force only op/slot, NOT the alpha heads, else every
   forced prune is silently vetoed to WAIT and the intervention is defeated.

### MEDIUM / fixes
9. Missing table cells: (Δ_struct<0, Δacc-null) = enabled redundant structure (real finding, not "inconclusive");
   (Δ_struct>0, Δacc<0) = over-commitment/degeneracy. 10. Broaden RELOCATION beyond r0c1/early-conv-only (misses r0c2
   + functionally-equivalent blueprints; "absent at r0c1 in control" can falsely fail). 11. Verify+report forced-step
   FRACTION parity across arms (masking r0c0 can push steps to WAIT-only → forced → excluded from actor loss →
   contradicts "no actor-loss change"). 12. Pin non-degeneracy churn band numerically from control variance.
13. **Hook citation fix:** apply_proof_baseline_action_controls is vectorized_trainer.py:247 (call site :2048), NOT
   action_execution.py:247.

### CLEAN (confirmed, no change)
- Governor independence (Invariant 1) holds: interventions write masks_batch upstream of panic detection
  (vectorized_trainer.py:1887) and preflight veto (action_execution.py:788); can only narrow actions / force more
  conservative ops; cannot relax a veto or suppress a panic; rollback preempts forced actions. (governor review Q1/Q2/Q4)
- Per-env forced-flag (not batch-level proof_controlled_step) is the correct choice; do NOT add the new modes to
  _PROOF_CONTROLLED_LIFECYCLE_POLICIES. Ablation-friendly schemas (Disc-2) pass.

### Budget impact of v2
Adds a NEGATIVE-CONTROL arm (≥5) and likely n→10 on the primary ⇒ minimal conclusive set grows from
SLOT(5)+OFF_SWITCH(5)=10 to ~SLOT(5–10)+PLACEBO(5)+OFF_SWITCH(5) = 15–20 runs (~45–60h). SUPPRESS-COMMIT optional.

### Next step
Produce v2 incorporating 1–8 (re-scoped budget + revised arms/decision-rule), then re-review (pytorch-expert still
owed on mask/gradient-isolation correctness), THEN owner sign-off. Reviewer agents: morphogenesis a5e206107b91d84f8,
governor aeef88659e2692db1 (resumable).
