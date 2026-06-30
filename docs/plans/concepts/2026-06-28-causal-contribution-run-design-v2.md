# Parallel-Control Causal-Contribution Run — Design v2 (FRESHLY AUTHORED, UNREVIEWED)

Status: CONCEPT — **v2 DRAFT, freshly authored 2026-06-29 from v1 + the four adversarial reviews.
UNREVIEWED as a v2. NOT launch-ready.** Incorporates v1 synthesis fixes 1–13 AND resolves the four
v2-review reports (morphogenesis / governor / pytorch / determinism) in-doc where a doc edit can resolve
them. Before launch this v2 STILL requires: (i) a fresh four-specialist re-review of *this* text and of the
*actual implementation* (the §5 mask/latch/RNG one-liners are SPEC, not code); (ii) two empirical residuals
that no document edit can close — see "Residual blockers" at end. Do NOT read this as "reviewed and nearly
fit": it is a revised design awaiting its first review.
Date: 2026-06-29 · Supersedes (PENDING owner ratification of the §0 total-system estimand pivot): 2026-06-28-causal-contribution-run-design.md (v1) · Tracker: relates
esper-lite-a221da47ea (GATE −1 SURVIVE), esper-lite-3d67b09687.
Required reviewers before approval (CLAUDE.md): drl-expert + morphogenesis-reviewer + governor-design-reviewer
+ pytorch-expert. v1 carried morpho+governor; pytorch+determinism reviewed the fix-list as the v2 *intent*.
All four must re-review this artifact.

## §0 Estimand decision (NEW — the load-bearing change in v2)
v1 straddled two incompatible estimands: §1 declared the controller's adaptation to BE the effect
(total-system) while §3 read Δ_struct as r0c0-specific MECHANISTIC enabling. The placebo arm v1 added (fix #1)
was the attempted bridge and it does not hold (morpho B2: a frequency-matched placebo's footprint is the wrong
matching dimension — the artifact scales with the removed option's developmental leverage, so a too-inert
placebo under-bounds and a leverage-matched one over-subtracts). **v2 resolves this by picking ONE estimand,
decisively: the TOTAL-SYSTEM EFFECT.**

- **What Goal-1 now answers:** does the *system* (controller + host, allowed to re-form downstream) end up
  with materially LESS committed downstream structure and/or LOWER accuracy when r0c0 can never commit, versus
  the matched control? `Δ_struct(SLOT) − Δ_struct(CONTROL)` IS the answer and is honestly system-level. There
  is no "action-space-reduction artifact to subtract" because the re-optimization over the r0c0-less manifold
  is PART of the effect, not a confound. The frequency-matched placebo-as-bound is therefore DELETED.
- **Scope reduction the owner is inheriting (state plainly):** Goal-1 can no longer claim r0c0 *mechanistically
  enables* specific downstream stems. It can only claim the system is worse off without r0c0's ability to
  commit. The mechanistic question (feature reuse vs optimization-path vs regularization) was already listed in
  v1 §6 as "CANNOT establish"; v2 makes that honest by removing the mechanistic framing from the decision rule,
  not just the caveats.
- **Mechanistic probe is demoted to EXPLORATORY:** the cardinality-fixed dummy-at-r0c0 arm (DUMMY-R0C0,
  optional) holds action cardinality and position FIXED while neutering r0c0's content. It answers a DIFFERENT
  counterfactual — "value of r0c0's *content* with the action space held fixed" — and is **explicitly NOT a
  bound for SLOT's option-removal trajectory.** It never gates a verdict; it is reported as a separate,
  cleaner mechanistic contrast if budget allows. (Implementation note: a dummy/inert-content injection hook at
  r0c0 does not exist today and must be built or the arm dropped — info-gap, not a blocker for the primary.)

## Purpose (two goals, one run set)
1. CAUSALLY resolve the total-system (a)/(b) fork — when committed early-conv r0c0 stems (neg per-seed LOO
   marginal) cannot commit, does the system build materially less downstream structure / score lower accuracy
   (system-level developmental dependence, "(b)"), or end up equivalent (freeloader, "(a)")? Existing
   removal-cost telemetry is structurally blind to this (post-commit the stem is in the always-on baseline;
   stems commit before neighbours co-reside). GATE −1 SURVIVED (cheap rescale levers don't fix the
   inefficiency), so the redesign is justified.
2. Produce the deferred GROWTH-VALUE evidence (re-scoped, see §4): committed-per-param counterfactual value of
   growth vs a true off-switch baseline. NOTE: as specified this measures growth-vs-no-growth value, NOT
   "controller skill" (no fixed-schedule baseline) — the claim is scoped to match what the baselines support.

## §1 Identification strategy — whole-run, matched-seed parallel control (total-system estimand)
For each seed 41–45: run the controller normally but with a runtime intervention that makes the r0c0 stem
un-committable; compare to the matched control. The host RE-FORMS its downstream under the intervention, so the
contrast is exactly: factual (stem commits → downstream D → acc A) vs counterfactual (stem absent → controller
builds D′ → acc A′). D′≈D & A′≈A ⇒ freeloader (a); D′ materially smaller and/or A′<A ⇒ system-level
developmental dependence (b). Controller adaptation (re-germinate elsewhere) is PART of the causal pathway,
measured and absorbed by the system-level contrast and the per-param value metric — not a confound to eliminate.

**Pairing pillars (corrected — v1's were over-claimed).** The honest pairing pillars are: shared controller
init, shared env seeds, shared data order. v1 *also* cited the "content-addressed lifecycle RNG
(action_execution.py:356-365)" as a pillar — that is WRONG and is struck: rng_seed there (action_execution.py:357)
is TELEMETRY-ONLY; it flows into LifecycleMutationCausalContext → leyline telemetry and is consumed by NO
generator (verified: no manual_seed/Generator reads it). v2 turns that false claim into a real one by WIRING
rng_seed into blueprint-init (see §5 RNG redesign). What the pairing buys is common-random-number (CRN)
variance reduction on the PRE-divergence common trajectory plus shared exogenous inputs — NOT a weight-matched
counterfactual to the final-acc horizon. Past the first divergent draw the arms have different topologies; their
host stochasticity is independent by construction and is absorbed into the per-param value metric, not
eliminated.

Snapshot-branch REJECTED (verified infeasible): resume restores only the controller (vectorized.py:1105-1142);
no serialization of the 5 dynamic-topology host models / per-env optimizers / host RNG / LSTM hidden / rollout
buffer; static_final_replay is FOSSILIZED-only.

## §2 Arms, matching, sample size
Unit of analysis = SEED (41–45). Each run = 12 envs × 200 episodes. Median-led across 5 seeds; paired
block-bootstrap B=20000 with SEEDS as the unit (never vec-envs). **The paired bootstrap's tight CIs are valid
ONLY if the realized paired-Δ correlation is positive and measured — see Residual blocker R1. Currently the
correlation is unmeasured and may be ~0 (RNG streams not yet split), which would make the CIs falsely TIGHT
(overconfident), not merely underpowered.**

| Arm | New runs | Mechanism | Role |
|-----|----------|-----------|------|
| CONTROL | 5 (re-run fresh on instrumented build — see reuse note) | existing shaped 41–45 | factual reference |
| SUPPRESS-SLOT (r0c0) — **PRIMARY** | 5 (→10 conditional) | stationary mask: r0c0 unavailable to all ops whole-run, WITH op-mask repair (§5) | "stem never exists" system-level contrast; force-prunes nothing ⇒ rollback rate ≈ control ⇒ free of the rollback-selection bias |
| SUPPRESS-COMMIT (r0c0) — **CORROBORATION + rate-evidence only** | 5 (optional) | force-PRUNE at r0c0 commit + per-env intervention-latch (§5) | isolate the COMMIT act; its rollback RATE is a reported descriptive outcome (see §3 governor rule), NOT a 95%-gated test |
| OFF_SWITCH — GROWTH-VALUE baseline | 5 | force_wait_only (EXISTS, proof_baselines.py:57-68), **seeds overridden to 41–45** | host-alone J denominator, PAIRED |
| DUMMY-R0C0 — EXPLORATORY mechanistic probe (optional) | 5 (only if injection hook built) | inert content at r0c0, cardinality + position held FIXED | DIFFERENT counterfactual ("value of r0c0 content"); never gates a verdict |

**OFF_SWITCH seed pairing (v1 fix #3, morpho B-Goal2):** OFF_SWITCH default training_seed = base_seed+10_000
(proof_baselines.py:60), which would make J_controller UNPAIRED vs the paired J_r0c0. **Override to seeds 41–45**
so both J's share the pairing. (Verified offset at proof_baselines.py:60.)

**CONTROL reuse → re-run fresh (v1 fix, determinism §2).** v1's "bit-identical-when-off" reuse gate is
UNACHIEVABLE: NO GPU-determinism configuration exists anywhere in the repo (use_deterministic_algorithms,
cudnn.deterministic, cudnn.benchmark, CUBLAS_WORKSPACE_CONFIG all absent), so on GPU+cuDNN conv-backward
atomics make bit-exact reproduction non-achievable across two builds. v2 therefore (a) DOWNGRADES the gate to a
CRN/tolerance-based equivalence check (the science needs CRN/statistical determinism, not bit-exactness; the
0.68pp paired-Δ SD is empirical on the real nondeterministic substrate, so the MDE is not refuted by this), and
(b) BUDGETS to re-run all 5 controls fresh on the instrumented build. The fresh control re-run is required
anyway for governor symmetry (next paragraph), so the reuse optimization is retired, not merely caveated.
Completeness note retained: control s44 = 2389 EPISODE_OUTCOME (incomplete) — the fresh re-run also closes this;
use a completeness-robust final-acc estimator (last fully-populated episode) consistently across arms.

## §3 Pre-registered decision rule (committed BEFORE running)

**ONE pre-registered PRIMARY test (multiplicity control — morpho HIGH).** The 7-cell table plus the relocation,
non-degeneracy, and health gates have NO family-wise error control at n=5. v2 declares a SINGLE primary
hypothesis test that carries the 95% confidence claim; every other cell, flag, and cross-check is EXPLORATORY
(reported as effect+CI, never as an independent 95% gate):

> **PRIMARY:** median `Δ_struct(SLOT − CONTROL)` < 0 with paired block-bootstrap 95% CI upper bound < 0.
> Δ_struct = (committed params at r0c1+r0c2)_SLOT − _CONTROL (+ downstream fossil count, blueprint mix as
> exploratory companions). Structure is the well-powered channel (control foss/ep SD ~0.005).

Power (provisional — see R1): control between-seed final-acc SD ≈ 1.0pp (floor-compressed); paired Δacc SD
proxy (GATE −1 arms) ≈ 0.68pp ⇒ **MDE(n5) ≈ 0.86pp IF pairing holds, up to ~2.14pp if it breaks. This MDE is
UNVALIDATED until R1 (re-measure realized paired-Δ SD after the §5 RNG split lands).** n=5 is below the
morphogenesis seed floor of 10; the pre-registered best case caps at exactly n=10 (no margin) — treat n=5 as
provisional. Stopping rule: run SLOT n=5; extend to n=10 (reallocated COMMIT/DUMMY budget) iff n=5 CI half-width
> 1.0pp AND median in (−2.0,+1.0); cap n=10. Honest fallback (pre-committed): report "at achieved precision ±h,
system-level developmental dependence ≥X is/ isn't ruled out" — no bare "inconclusive".

SECONDARY/EXPLORATORY = accuracy: Δacc median, paired bootstrap (precision-targeted, may be unreachable at
n≤10). Confirmation of the primary, never the gate.

**Verdict table (TOTAL-SYSTEM-EFFECT framing; PRIMARY = Δ_struct only; all rows below the primary are
exploratory interpretation, not independent 95% claims):**

| Outcome | Δ_struct (PRIMARY) | Δacc (exploratory) | Relocation | Health | Verdict | Reward implication |
|---------|--------------------|--------------------|-----------|--------|---------|--------------------|
| (b) system developmental dependence | <0, CI<0 | ≤−2.0pp (if reached) | FALSE | healthy | system needs r0c0's commit to reach its structure/acc | synergy/hindsight credit |
| (b′) direct contributor | spans 0 | ≤−2.0pp | FALSE | healthy | stem's own output carried acc; co-resident LOO compression | fix the marginal estimator |
| relocation / fungible | ≥0 or Δparams>0 | ⊂(−1,+1) | TRUE | healthy | function valuable, position-fungible | position-invariant credit; do NOT penalize |
| (a) freeloader | spans 0 | ⊂(−1,+1), \|med\|<0.5 | FALSE | healthy | contributes nothing yet keeps committing | penalize survival-without-contribution |
| enabled-redundant (was cell #9) | <0, CI<0 | null | FALSE | healthy | system builds less, no acc cost — redundant structure removed | informative; NOT mechanistic enabling (total-effect framing) |
| over-commitment | >0 | <0 | — | — | controller over-builds without r0c0's discipline | exploratory |
| controller-collapse artifact | <0 | any | — | **FAIL** | NOT banked — health gate fails | discard, re-run |

**Health discriminator (NEW — off the structure-count axis; morpho HIGH).** The v1 ≥50% non-degeneracy floor
sits on the SAME structure-count axis as the (b) signature, so a genuine strong (b) ("system legitimately builds
much less") can be mis-thrown as "collapse" and vice-versa. v2 adds a SINGLE INDEPENDENT health arbiter off that
axis. To avoid discarding genuinely-healthy runs by chance (a 4-way conjunction of noisy ±2σ bands at n=5 would
reject a meaningful fraction of healthy runs — the opposite failure), the arbiter is ONE aggregate z-score: the
mean standardized deviation across {controller policy entropy, value-loss, grad-norm, blueprint diversity}, each
standardized against CONTROL between-seed variance, combined into a single composite. A SLOT Δ_struct<0 is
"healthy-built-less" only if the composite health z is within its pre-registered band (pinned from the 5 control
runs); out-of-band ⇒ controller-collapse artifact, FAIL the health gate, discard (do not bank as (b)). Using one
composite (not a conjunction of four independent bands) keeps the false-discard rate controlled at n=5. The
≥50% fossil-count floor is retained as a companion but is no longer the sole arbiter.

**RELOCATION operationalized (broadened + pinned — v1 fix #10).** Under SUPPRESS-SLOT, flag TRUE iff a seed
commits a stem that is (i) early-conv OR functionally-equivalent blueprint family (equivalence relation pinned:
same blueprint `family_id` OR same `(op_type, channel_band)` signature — fixed before running), (ii) at r0c1
OR r0c2 (r0c2 inclusion added), (iii) commit-timing percentile ≤25th of the run's fossils, (iv) and the matched
control has NO equivalent commit at that position-band (replaces v1's brittle "absent at r0c1 in control" clause
with a position-band equivalence test). All thresholds fixed before running; no post-hoc latitude.

**Governor rollback rule (NEW — governor B2/B-Goal, replaces v1 fix #5's exclusion).** v1's "exclude rollback
episodes" is a DIRECTIONAL selection bias, not mere sample loss: P(rollback | force-prune r0c0) RISES with
r0c0's true contribution (the lobotomy detector, governor.py:342-358, fires when loss≈random_guess and bypasses
the consecutive-panic gate), so the catastrophic-(b) cases — removing a load-bearing r0c0 collapses the host —
are GUARANTEED to roll back and would be excluded, stripping the strongest (b) evidence and biasing survivors
toward false-(a). v2:
- **SUPPRESS-SLOT is PRIMARY** and its stationary mask never force-prunes a live seed (the host develops without
  r0c0 from step 0, no sudden loss jump), so SLOT's rollback rate ≈ control's and SLOT is FREE of this bias.
- **SUPPRESS-COMMIT's force-prune ROLLBACK RATE is a pre-registered DESCRIPTIVE outcome (reported, OUTSIDE the
  95% family — NOT a second gated test):** a high rate is corroborating evidence r0c0 is load-bearing (evidence
  FOR (b)). COMMIT is demoted to corroboration + rate-evidence; it NEVER drives the single primary verdict.
  Rollback episodes within COMMIT are a distinct category (different restored baseline —
  governor.py:199-218/483-503 restores host + fossilized seeds, asymmetric across arms), reported separately,
  never pooled into Δ_struct.
- **Rollback-parity gate (numeric, v1 fix #4 scoped — governor #4/#5 reconciled):** pre-register **0 rollback
  episodes in any of {CONTROL, SUPPRESS-SLOT, OFF_SWITCH}**; any seed with ≥1 → that seed re-run or excluded
  SYMMETRICALLY across those arms. COMMIT is EXEMPT from the parity gate (it is EXPECTED to break parity by
  construction) and is governed by the rate rule above.

**SLOT-vs-COMMIT cross-check (exploratory):** agree ⇒ robust; SLOT drops but COMMIT doesn't AND COMMIT rollback
rate is low ⇒ transient scaffolding (not the commit act) ⇒ credit transient presence. (Only interpretable when
COMMIT rollback is rare; report the rate.)

## §4 Metrics
Causal contrast (paired vs control): Δ committed params + Δ fossil count at r0c1+r0c2 (+ blueprint mix,
re-keyed seed_lifecycle 482/482 join); Δ final committed_val_acc (vectorized_trainer.py:1458-1463);
adaptation (r0c0 re-germinate/re-commit attempts, relocation flag, churn); off-axis health signals (policy
entropy, value-loss, grad-norm, blueprint diversity).

**Growth-value J (re-scoped — morpho medium).** J_controller = (acc(CONTROL) − acc(OFF_SWITCH)) / committed
seed params. **Scoped explicitly to "growth-vs-no-growth value," NOT "controller skill"** — with no
fixed-schedule (predetermined-growth) baseline, J_controller cannot decompose growth-lift from controller-skill.
Upgrade path (deferred): add a fixed-schedule arm to license the "controller value / ADMIT" claim. Until then
the contribution-gate claim is scoped to growth value.

**J_r0c0 relabel + conditioning (v1 fix #7, morpho B #7/#8).** J_r0c0 = (acc(CONTROL) − acc(SUPPRESS-SLOT)) /
r0c0_committed_params is a SYSTEM-LEVEL, NET-OF-SUBSTITUTION marginal (it includes the re-formed D′), NOT a
weight-level per-param marginal. **Report J_r0c0 ONLY when relocation=FALSE.** Under relocation=TRUE the
numerator → 0 (acc(SLOT)≈acc(CONTROL)) so J_r0c0 → 0 would read "r0c0 contributes nothing per param," directly
contradicting the same arm's relocation verdict ("function valuable, position-fungible"). Under relocation it is
UNDEFINED as a per-param contribution and MUST NOT be emitted as one. This conditioning is part of the decision
rule.

**Denominator pinned to ONE definition (v1 fix #7).** v1 conflated slot.py:1298 `active_seed_params` (counts
LIVE/active seed params, incl. mid-blending) with helpers.py:131-166 `effective_seed_params` (alpha-WEIGHTED
continuous rent: `base_slot_rent_params + current_alpha·slot_param_count`). Neither is exactly "committed
params." For a committed-per-param metric the denominator = **raw parameter count of the COMMITTED (fossilized)
seeds attributable to the arm** (r0c0 for J_r0c0; all controller-committed seeds for J_controller). For
fossilized seeds alpha is saturated so this equals the seed's full param count. Pin a single fossilized-param
accessor across BOTH J's; resolve the exact accessor at build (info-gap G2) — do NOT inherit either of the two
conflated quantities.

## §5 Minimal code changes + budget (SPEC — not yet implemented; pytorch must re-review the actual diff)
All interventions live in `apply_proof_baseline_action_controls` (**vectorized_trainer.py:247** def / **:2048**
call — v1 fix #13; v1's "action_execution.py:247" was wrong), applied to the batched `masks_batch` BEFORE
`get_action`, AFTER panic detection (vectorized_trainer.py:1887) — preserving Invariant-1 ordering. Keyed on
`slot_config.index_for_slot_id("r0c0")`. **Do NOT add the new modes to _PROOF_CONTROLLED_LIFECYCLE_POLICIES**
(vectorized_trainer.py:95) — that sets proof_controlled_step=True for ALL envs EVERY step, zeroing every actor
gradient and destroying the "controller re-forms downstream" premise (v1 CLEAN note, confirmed by pytorch).

1. **SUPPRESS-SLOT (op-mask repair — pytorch BLOCKING).** Zero `slot_by_op[:,:,r0c0_idx]`, THEN recompute op
   validity for every NON-WAIT op: `op_mask[:,op] &= slot_by_op[:,op,:].any(dim=-1)` (WAIT always True).
   Without this, when r0c0 is the only germinable slot, op_mask[GERMINATE] stays True
   (action_masks.py:300-306), the controller samples GERMINATE, `get_action` gathers an ALL-FALSE slot row;
   because MaskedCategorical.validate=False in training (vectorized.py:1102) the rollout does NOT raise — it
   uniformly samples an INVALID slot (possibly r0c0 itself, silently defeating suppression for that step) and
   the next PPO update hard-crashes on the empty-mask guard (factored_lstm.py:1547). The repair routes
   "r0c0-was-the-only-option" steps to WAIT-only (then legitimately forced — see telemetry split). Read the
   trigger rows from slot_by_op BEFORE the zeroing mutation. In-place bool mask edits are autograd-safe
   (masks built fresh per step, no grad history, sampling under inference_mode).

2. **SUPPRESS-COMMIT (per-env intervention-latch — pytorch BLOCKING; do NOT route via forced_batch_cpu).** Add
   an explicit `intervention_forced: bool[num_envs]` latch and OR it into the forced flag:
   `forced_step = proof_controlled_step or forced_batch_cpu[env_idx] or intervention_forced[env_idx]`. The
   existing `forced_batch_cpu` is WAIT-ONLY (`forced_batch = (num_valid_ops==1) & wait_valid`,
   action_execution.py:552) so a PRUNE-only mask does NOT set it ⇒ the step would otherwise enter the policy
   gradient with actor_weight=1.0 and value_weight=1.0-instead-of-0.2 (ppo_update.py:328/331), contaminating
   alpha-head gradients (left policy-sampled, see #3) and poisoning GAE for neighbouring REAL steps. The latch
   excludes the forced-PRUNE step from the actor loss as intended.

3. **Forced PRUNE: force ONLY op(=PRUNE)+slot(=r0c0); leave alpha_speed/alpha_curve POLICY-SAMPLED (v1 fix #8,
   governor + pytorch).** Governor preflight vetoes PRUNE when alpha_speed_steps or alpha_curve is None
   (governor.py:184-189); those are derived from the policy-sampled alpha heads (action_execution.py:768-772),
   so masking them to a degenerate/None mapping silently vetoes EVERY forced prune to WAIT and defeats the
   intervention with no error. **Do NOT reuse `_force_scheduled_action_masks` (vectorized_trainer.py:382/467-479)
   — its PRUNE branch FORCES the alpha heads.** Ensure (PRUNE, r0c0) is a valid `slot_by_op` combination so the
   forced action is sampleable. Verify post-hoc: INTERVENTION_APPLIED forced-prune count == actual r0c0 PRUNE
   executions (zero preflight vetoes).

4. **Force-PRUNE gated on ACTUAL PRUNE availability (pytorch HIGH).** FOSSILIZE-eligibility (stage==HOLDING,
   action_masks.py:323-327) does NOT imply PRUNE-eligibility — PRUNE additionally requires age≥MIN_PRUNE_AGE AND
   alpha_mode==HOLD (action_masks.py:330-331). Only force-PRUNE when `slot_by_op[:,PRUNE,r0c0]` is True; while
   r0c0 is HOLDING-but-not-yet-prunable, mask FOSSILIZE and latch the env to WAIT (counting it intervention-
   forced) until PRUNE becomes available. Else forcing op→PRUNE-only on an unavailable PRUNE yields an
   all-false slot row → same crash as #1.

5. **RNG three-domain split (v1 fix #6, morpho HIGH + determinism BLOCKING).** Today a single global generator
   is seeded once (vectorized.py:1148) and shared by three logically-distinct consumers: (A) policy/controller
   sampling (torch.multinomial, factored_lstm.py:1118 + forward sampled_op), (B) host stochasticity, (C)
   blueprint-init weight draws at germination. The rollout SAMPLES actions (deterministic=False,
   vectorized_trainer.py:2110), so suppressing r0c0 changes the NUMBER of blueprint-init draws between arms,
   OFFSETS the shared stream, and SHIFTS which actions the controller samples thereafter — injecting a pure RNG
   artifact straight into Δ_struct, the PRIMARY discriminator (the named expensive-false-positive channel).
   v2 splits the master seed into THREE independent child-seed domains:
   - **(A) controller:** a dedicated CUDA `torch.Generator` threaded into the controller's sampling sites
     (forward's sampled_op draw + get_action's torch.multinomial, factored_lstm.py:1118 — both accept
     `generator=`). Insulates controller sampling from host/probe draws.
   - **(C) blueprint-init:** a local CUDA `torch.Generator` **seeded from the already-computed content-addressed
     `rng_seed` (action_execution.py:357)** and threaded into germinate()/BlueprintRegistry init
     (torch 2.12 nn.init.* accept `generator=`, so this is a kwarg, not a rewrite). Content-addressed
     per-germination seeding is the ONLY thing that makes a blueprint at a given LOGICAL germination draw get
     identical weights in BOTH arms regardless of how many germinations preceded it — i.e. the only thing that
     survives the divergence point. NOTE seed weight-init constructs on CPU then `.to(device)` (slot.py:1380→1387)
     so it draws the CPU generator — thread the content-addressed generator there.
   - **(B) host stochasticity:** a per-env CUDA `torch.Generator` (reuse the env_factory.py:203 pattern),
     DETERMINISTICALLY re-seeded per content key `hash(base_seed, env_id, topology, germ_index)` — a
     free-running generator still desyncs across arms because it is advanced a different number of times. NOTE
     the CNN host has NO dropout; the live host-stochasticity consumer is the germination-time shape-probe
     `torch.randn(..., device=...)` (slot.py:1170-1182) which currently shares the controller's CUDA stream —
     this generator removes that bleed and cross-env contamination. Past the divergence point, host stochasticity
     is independent by construction (different networks) and is absorbed into the per-param value metric, not
     eliminated (do NOT claim it removes within-stream divergence-induced desync).
   Do NOT use `torch.manual_seed` mid-run (resets the global default, changes controller sampling, breaks the
   shared controller-init/data-order pairing). Controller init (vectorized.py:1155) precedes any germination
   draw and data order is keyed to an explicit seed (vectorized.py:1296/1322), so the pairing is preserved.

6. **RNG-state compare-point + declared determinism class (determinism BLOCKING + HIGH).** Declare the class:
   **CRN / statistical determinism** (NOT bit-exact — no GPU-determinism config exists; §2 reuse gate downgraded
   accordingly). Add a cheap per-env, per-step generator-state (or cumulative draw-count) hash for EACH of the
   three RNG domains, emitted on INTERVENTION_APPLIED. Pre-register two assertions: (1) the intervention-OFF run
   is CRN-identical to the fresh control up to the first r0c0 suppression event; (2) divergence onset coincides
   with the suppression event, not an earlier RNG artifact. This doubles as the empirical proof that the
   intervention is a true no-op when OFF (Residual blocker R1's verification half).

7. **Telemetry (build it — do not defer).** Two distinct shapes (governor #4): a run-start
   **INTERVENTION_CONFIGURED** record (NO action_id) describing SLOT's stationary mask; and a per-event
   **INTERVENTION_APPLIED** carrying `triggering_action_id` for COMMIT's forced prunes (joinable to
   GOVERNOR_ROLLBACK.triggering_action_id, governor.py:555). Drop the v1 "GOVERNOR_SNAPSHOT snapshot cadence"
   item — no such event exists. Per-arm derive (already available today): GOVERNOR_ROLLBACK count/rate (carries
   env_id + panic_reason + consecutive_panics), preflight-veto counts by `governor_blocked_factor`
   (MORPHOLOGY_CAUSAL_LOG, action_execution.py:805-818). **Forced-step fraction SPLIT BY REASON** (pytorch
   MEDIUM): intervention-forced (from INTERVENTION_APPLIED) vs WAIT-saturation (forced_step_ratio minus
   intervention) — a single pooled ratio conflates WAIT-saturation with intervention-forcing and hides the
   action-space effect; pre-register an expected-parity check on the WAIT-saturation component across arms.
8. **ProofBaselineMode members + cohort entries** (OFF_SWITCH exists; SUPPRESS-SLOT / SUPPRESS-COMMIT new;
   DUMMY-R0C0 only if injection hook built). **Configs** cloned from config-3slot-3seed-baseline-shaped.json +
   a seeds-41–45 driver (OFF_SWITCH seed overridden to 41–45).

Budget (≤2 concurrent, ~6h/run, ≤16GB): fresh CONTROL(5) + SLOT(5) + OFF_SWITCH(5) = 15 runs ≈ 45h minimal;
+SUPPRESS-COMMIT(5) corroboration = 20 ≈ 60h; precision branch reallocates COMMIT/DUMMY budget to SLOT n=10 if
the stopping rule trips; DUMMY-R0C0(5) only if the injection hook is built. (The reuse optimization is retired;
control is re-run fresh — see §2.)

## §6 Risks / what it CANNOT establish
Risks: power/floor-compression AND falsely-tight-CI-from-broken-pairing (TOP — mitigated by structure-primary +
single-primary-test + precision-targeted rule + honest fallback, BUT gated on R1 pilot); rollback-selection bias
(resolved: SLOT primary is bias-free, COMMIT rate-as-outcome); RNG divergence into Δ_struct (resolved by §5
three-domain split + content-addressed blueprint-init + compare-point — pending implementation R2); build-skew on
control (resolved: re-run fresh); completeness s44 (resolved: re-run fresh); controller-collapse vs
healthy-built-less (resolved: off-axis health gate); positional-vs-blueprint (keys on r0c0 position; 54%
early-conv ⇒ positional dominates; DUMMY-R0C0 is the content refinement); small-n flips (median-led, paired,
pre-registered, single-primary-test, operationalized relocation).
CANNOT establish: the MECHANISM of enabling (feature reuse vs optimization-path vs regularization — total-effect
estimand, §0); "controller skill" (no fixed-schedule baseline — J scoped to growth value); that a DIFFERENT
reward makes r0c0 useful (tests current shaped controller); a clean weight-level ceteris-paribus counterfactual
(only the system-level one, which is the relevant one since the redesign also runs online).

## Open gaps / info-gaps to close before finalizing
- G1: Realized SLOT paired-Δacc/Δstruct SD AFTER the §5 RNG split (sets n=10 trigger AND validates CI-trust — see R1).
- G2: Exact fossilized-param accessor for the J denominator (resolve at build; do NOT inherit active_/effective_seed_params).
- G3: Per-slot r0c1/r0c2 committed-params variance across control seeds (fix Δ_struct threshold).
- G4: Do the (now fresh) control runs emit GOVERNOR_ROLLBACK rows with env_id/panic_reason? (Karn query with run_dir filter; the fresh re-run on the instrumented build guarantees yes — see R3.)
- G5: Whether r0c0 ever hosts non-early-conv blueprints (decides if a blueprint-conditioned arm is needed).
- G6: Does a dummy/inert-content-at-r0c0 injection hook exist or need building? (gates DUMMY-R0C0 only — exploratory, not primary.)
- G7: OFF_SWITCH wall-time (may be <6h; refines budget).
Pre-build checklist: §5 implemented + pytorch re-review of the actual diff; RNG compare-point proves
intervention-OFF no-op; R1 pilot re-measures paired-Δ SD; extract r0c1/r0c2 control params; four reviewer
sign-offs on THIS artifact.

---

## v2 review resolution (2026-06-29)
This section records, per reviewer, what was found and how each BLOCKING/HIGH finding was resolved in-doc or why
it is deferred to a documented build-time / pilot item. **Top-line shared finding (all four reviewers): the
"-v2.md" file did not exist; only v1 did, whose synthesis ended "Produce v2 incorporating 1–8." VALID — this
document is the freshly-authored v2 that resolves it. The four reviewers reviewed the v2 *intent* (fix-list),
not an artifact; this v2 must now get its first real four-specialist review.** Code citations were independently
spot-checked here and confirmed: hook def vectorized_trainer.py:247; rng_seed telemetry-only action_execution.py:357;
single global manual_seed vectorized.py:1148; forced_batch WAIT-only action_execution.py:552; OFF_SWITCH
+10_000 proof_baselines.py:60; alpha veto governor.py:184-189; slot.py:1298 = active_seed_params (≠ effective);
effective_seed_params alpha-weighted helpers.py:131-166; _force_scheduled_action_masks forces alpha
vectorized_trainer.py:382.

### morphogenesis-reviewer (verdict: unfit → addressed)
- B "v2 does not exist": VALID — resolved by authoring this v2.
- B "placebo cannot bound the action-space artifact in either direction": VALID and load-bearing — resolved in
  §0 by DELETING the frequency-matched placebo-as-bound and adopting the total-system estimand (no artifact to
  subtract). The dummy-at-r0c0 arm is retained only as an EXPLORATORY mechanistic probe explicitly labeled a
  different counterfactual, never a bound.
- B "estimand contradiction (§1 total vs §3 mechanistic), fix #2 doesn't reconcile": VALID — resolved in §0/§1/§3
  by picking ONE estimand (total-system) decisively; all "mechanistic enabling" language removed from the
  decision rule; the scope reduction is surfaced, not buried.
- H "RNG under-scoped + weak pairing → falsely TIGHT CIs": VALID — design resolved in §5.5 (three-domain split +
  content-addressed blueprint-init) and §2 (CI-trust gated on measured pairing). The empirical half (re-measure
  paired-Δ SD; prove no-op) cannot be closed in-doc → Residual blocker R1.
- H "power below floor of 10 with zero margin; no multiplicity control": VALID — multiplicity resolved in §3 by
  declaring ONE pre-registered PRIMARY test (Δ_struct CI<0) and marking all other cells exploratory. The power
  re-derivation is gated on R1 (MDE marked provisional, n=5 marked provisional, cap n=10 retained).
- H "non-degeneracy floor collides with the (b) signature; no independent health discriminator": VALID —
  resolved in §3 by adding an OFF-AXIS health gate (policy entropy / value-loss / grad-norm / blueprint
  diversity, each pinned ±2σ from control variance); the ≥50% fossil-count floor demoted to a companion.
- H "SUPPRESS-COMMIT rollback exclusion is a selection effect biasing toward (a)": VALID (also governor B2) —
  resolved in §3 governor rule: SLOT (force-prunes nothing) is the bias-free PRIMARY; COMMIT's rollback RATE is
  a primary outcome (evidence FOR (b)); rollback episodes never pooled; COMMIT demoted to corroboration.
- H "J_r0c0 sign trap under relocation; fix #7 doesn't close it": VALID — resolved in §4: J_r0c0 reported ONLY
  when relocation=FALSE; undefined (not emitted) under relocation.
- MEDIUM (J_controller fixed-schedule baseline; relocation operationalization; cell #9 gating; fix
  implementability): addressed — Goal-2/J_controller scoped to growth-value with fixed-schedule as deferred
  upgrade (§4); relocation broadened+pinned (§3); cell #9 relabeled under total-effect framing, no longer
  presumes mechanistic enabling (§3 table); implementability one-liners flagged as SPEC in §5 with the
  alpha-head and DUMMY-injection caveats called out (G6).

### governor-design-reviewer (verdict: conditionally-fit → addressed)
- B "no v2 / recommendations not incorporated": VALID — resolved by authoring v2.
- B "rollback-exclusion is a DIRECTIONAL selection bias (lobotomy detector guarantees catastrophic-(b) rolls
  back)": VALID, the most important governor finding — resolved in §3 governor rule exactly as recommended
  (SLOT primary + bias-free; COMMIT rate-as-primary-outcome; demote COMMIT).
- B "symmetric exclusion/parity only valid if control has GOVERNOR_ROLLBACK rows — unconfirmed (Karn timeout)":
  VALID — design resolves the symmetry by re-running the control FRESH on the instrumented build (§2), which
  guarantees the rows. The empirical confirmation is Residual R3 / info-gap G4 (still must query, but the fresh
  re-run makes the answer deterministic).
- H "#4 parity gate vs #5 COMMIT-rollback-prone not reconciled": VALID — resolved in §3: parity gate SCOPED to
  {CONTROL, SLOT, OFF_SWITCH}; COMMIT explicitly EXEMPT (expected to break parity).
- H "parity gate / incidence cap not operationalized": VALID — resolved in §3 with numeric thresholds
  (0 rollback episodes in non-COMMIT arms; offending seed re-run/excluded symmetrically).
- H "telemetry partly non-existent / ill-keyed (INTERVENTION_APPLIED absent, no GOVERNOR_SNAPSHOT, SLOT mask has
  no action_id)": VALID — resolved in §5.7: two shapes (INTERVENTION_CONFIGURED run-start no-action_id for SLOT;
  INTERVENTION_APPLIED per-event triggering_action_id for COMMIT); GOVERNOR_SNAPSHOT cadence dropped.
- MEDIUM/LOW (alpha-head mechanism in body; asymmetric snapshot baseline; hook-citation; Invariant-1 clean):
  addressed — alpha-heads-sampled mechanism stated in §5.3; asymmetric-restore folded into the
  rollback-not-pooled rule (§3); hook citation corrected to vectorized_trainer.py:247/:2048 (§5); Invariant-1
  ordering preserved and the new modes kept OUT of _PROOF_CONTROLLED_LIFECYCLE_POLICIES (§5).

### pytorch-expert (verdict: conditionally-fit → addressed; SPEC only — owes a re-review of the real diff)
- B "owed pytorch sign-off": VALID — these findings are folded into §5; pytorch must re-review the actual
  implementation (Residual R2).
- B "SUPPRESS-SLOT all-masked-slot-row (silent defeat then crash)": VALID — resolved in §5.1 (op-mask repair
  `op_mask[:,op] &= slot_by_op[:,op,:].any(dim=-1)`).
- B "SUPPRESS-COMMIT forced-PRUNE leaks into actor loss (forced_batch_cpu is WAIT-only)": VALID — resolved in
  §5.2 (explicit per-env intervention-latch OR'd into forced_step; NOT routed via batch-level
  proof_controlled_step).
- H "FOSSILIZE-eligible ≠ PRUNE-eligible window → all-false PRUNE row crash": VALID — resolved in §5.4 (gate
  force-PRUNE on slot_by_op[:,PRUNE,r0c0]; latch to WAIT while HOLDING-not-prunable).
- H "forced PRUNE must leave alpha heads policy-sampled; do NOT reuse _force_scheduled_action_masks": VALID
  (confirms governor #8) — resolved in §5.3.
- H "RNG separate-stream mis-targeted (CNN has no dropout; real vector is shape-probe randn; controller needs
  its own generator)": VALID — resolved in §5.5 (dedicated controller generator; host generator scoped to
  shape-probe/norm/augmentation; CUDA generators; deterministic per-content re-seed).
- MEDIUM "forced-step fraction must be split by reason": VALID — resolved in §5.7 (split intervention-forced vs
  WAIT-saturation).
- LOW (two no-change confirmations: read trigger from slot_by_op before mutating; in-place bool mask edits
  autograd-safe): incorporated as notes in §5.1.

### determinism-reviewer (verdict: conditionally-fit → addressed; empirical half deferred)
- B "artifact missing + determinism class undeclared": VALID — resolved by authoring v2 and DECLARING the class
  (CRN/statistical) in §5.6.
- B "divergence-point confound on the PRIMARY discriminator (3 consumers share one global stream; blueprint-init
  draw-count differs between arms → RNG offset into Δ_struct)": VALID, the core determinism blocker — design
  resolved in §5.5 (three-domain split + content-addressed blueprint-init consuming the existing rng_seed). The
  EMPIRICAL proof that the fix works (intervention-OFF CRN no-op via the compare-point; re-measured SD) cannot
  be closed in-doc → Residual R1/R2.
- H "fix #6 covers the wrong consumer (host, not blueprint-init)": VALID — resolved in §5.5 (host generator
  reframed to its real scope; blueprint-init handled by the content-addressed generator; controller its own).
- H "over-claim: per-env stream 'removes divergence-induced host-RNG desync'": VALID — corrected in §1/§5.5 to
  the defensible claim (removes cross-env contamination + controller→host bleed on the PRE-divergence segment;
  post-divergence host stochasticity is independent by construction and absorbed, not eliminated).
- H "§2 bit-identical-when-off gate unachievable (no GPU-determinism config)": VALID — resolved in §2 by
  downgrading to CRN/tolerance equivalence and re-running controls fresh (reuse optimization retired).
- MEDIUM (stale §1 rng_seed pairing pillar; MDE rests on decaying pairing; no RNG compare-point): addressed —
  §1 pairing pillars corrected (rng_seed citation struck, then wired for real in §5.5); MDE marked provisional
  pending R1; RNG-state compare-point added in §5.6.

### Residual blockers (cannot be closed by a doc edit — why this is needs-another-pass)
- **R1 — RNG-pairing pilot (empirical):** implement the §5.5 three-domain split + content-addressed
  blueprint-init, then (a) re-measure the realized paired-Δ SD and (b) prove the intervention-OFF run is a CRN
  no-op up to the first suppression event via the §5.6 compare-point. Until this lands, MDE(n5)≈0.86pp and EVERY
  CI-gate driving the primary verdict are UNVALIDATED; the pairing correlation may currently be ~0 (falsely-tight
  CIs). This is the primary discriminator's trust and is irreducibly a code-spike + measurement.
- **R2 — §5 implementation + pytorch/morpho/governor re-review of the actual diff:** the four mask/latch/RNG
  one-liners (SLOT op-mask repair; COMMIT per-env latch; force-PRUNE gated-on-availability with alpha-heads
  sampled; split forced-step telemetry) are SPEC, not code. Specifying them closes the DESIGN findings, not the
  implementation. A fresh four-specialist pass on this v2 + the diff is mandatory.
- **R3 — control GOVERNOR_ROLLBACK symmetry (empirical):** the fresh control re-run (§2) is designed to
  guarantee GOVERNOR_ROLLBACK rows, but this must be confirmed by a Karn query (run_dir-filtered) before the
  parity/rate gates can be asserted symmetrically.
