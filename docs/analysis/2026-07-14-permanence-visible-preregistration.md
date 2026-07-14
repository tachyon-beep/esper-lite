# "Make permanence visible" — pre-registration + two coupled specs

**Status: OPEN. Frozen-before-data intent; nothing here is frozen.** Blocked on owner sign-off (the flags in §1.13/§2.8/§3.5) + the remaining verification reads (§1.11). Grounded in PDR-0078/0079/0080, the primes review-pack, the Stage-2 acceptance gate (discipline template), and a primary-source symbol map. Design-only; no code touched; no GPU authorized. drl-expert deliverable, 2026-07-14.

## READ-5 fold-in
READ 5 retracts the READ-4 "shelter / floor-destroys-good-seeds" half: IPCW prune-propensity mis-specified (79–82% predicted survival vs 45% observed); the SLOT head is deterministic (conf 1.0, 98%), not a random 0.15; pruning is quality-SELECTIVE (prunes lower-LOO seeds: PRUNE-destined lifetime LOO +6.1/+5.9 vs FOSS-destined +6.9/+6.6). Clean estimand: shelter ~16–20%, intrinsic-permanence residual ~80–84% of the +1.5pp. Three changes flow: (1) the settlement (arm B) is the well-motivated robust leg (priority); (2) arm C's op-primitive is re-scoped from "remove structural PRUNE probability" to the **gradient-restoring per-num_valid differentiable floor** (restore learnable op preference WITHOUT zeroing the PRUNE minimum — removing it could stop sensible quality-pruning); (3) the "reduced forced-destruction" guardrail flips to "quality-pruning selectivity **preserved/not degraded**." The design's shelter-independence (verdict rests only on B−A and C−A) is vindicated.

**Grounded anchors:** floor `_apply_floor_to_logits` (`action_masks.py:522`), per-action min `min(0.15,0.99/num_valid)`, single-source sampling+eval; `PROBABILITY_FLOOR_PER_HEAD["op"]=0.15`; `LifecycleOp` (WAIT/GERMINATE/SET_ALPHA_TARGET/PRUNE/FOSSILIZE/ADVANCE — **no REQUEST_FOSSILIZE; FOSSILIZE is immediate** via `handlers/fossilize.py:160`); `MIN_PRUNE_AGE=5`; `ContributionState`/`CounterfactualStatus` landed; per-head KL in `RatioMetrics` but **no per-op KL yet**; `escrow_fossil_settlement` fail-closed guard landed; K=4 (baseline K=1).

---

# Section 1 — Triangular attributable experiment pre-registration

**1.1 What it buys:** attributes the two interventions instead of confounding them. B−A isolates the settlement under the existing floor; C−B isolates the incremental gradient-restoration; C−A is the combined bet. The ship/no-ship verdict rests only on the shelter-independent contrasts (B−A, C−A).

**1.2 Sequencing — DECIDED: STAGED (owner, 2026-07-14).** Coupling rationale = "either-alone-degenerate": settlement-alone → commits stay 93–94% floor-forced (op head can't act selectively); floor-fix-alone (omitted D arm) → PARK under the −113-forfeiting reward → zero fossils (unsafe, never run). Staged: **Exp 1 = A vs B** (settlement, floor held; cleanest single-factor; pre-registered expected-null on selectivity = the degeneracy signature); **Exp 2 = add C** (gradient floor, settlement ON; tests coupled hypothesis). Tradeoff: A/B-first is cleanest + de-risks the settlement but can't show selective commitment (its P2 is an expected-null); triangular-in-one tests coupling in one shot but adds a factor + 3n runs.

**1.3 Arms** (all obs-v4/L1, K=4, paired fresh inits, identical telemetry): A control (SHAPED, no settlement, hard floor); B (non-selectable settlement §2, hard floor); C (settlement + gradient-restoring per-num_valid differentiable floor §3). Frozen identical: critic (HRA posture — OWNER FLAG), entropy/optimiser/lr, γ=0.995, gae_λ=0.95, K=4, n_envs=12, max_epochs=150, all reward terms except the settlement, obs-v4/L1, governor, cifar_baseline, slots r0c0-2, param_budget=500000, per_head_advantage_norm=false. Only toggles: A→B `fossil_settlement`; B→C `op_floor_primitive`. Loser deleted post-experiment (no-legacy). r9 is archival, NOT the concurrent control.

**1.4 Contrasts + shelter-independence:** B−A = settlement effect (shelter-independent, carries the settlement verdict); C−A = combined (shelter-independent, carries the coupled verdict); C−B = incremental gradient-restoration (reported, NOT a "shelter" quantity per READ 5, never gated). Staircase not full 2×2 (no D arm — unsafe). C−A=(B−A)+(C−B) is a decomposition, NOT additively bankable. **Decision-rule constraint: no PASS branch may require a C−B or destruction-selectivity magnitude — coupling carried by the degeneracy logic.**

**1.5 Regime:** r9 was K=1 (ratio≈1, clip/KL inert); K=4 makes clip + per-op KL operative for the first time (clean across arms, all K=4) → the per-op KL telemetry (§3) is load-bearing. Reversal: if the |H|=1 causal sign fails under K=4/obs-v4, r9 was regime-specific.

**1.6 Metrics.** Primary: **P1** proxy-free terminal `final_accuracy` (paired; noisy — superiority needs n=10, non-regression at n=5); **P2** contribution-sensitive commitment (behavioral primary — the CATE/slope of P(committed-FOSSILIZE) on the pre-request quote over eligible HOLDING; per-decision → high power; Exp-1 A/B: expected-null; arm C: the primary "did the coupled bet fire"). Guardrails (fail toward owner): **G-FARM** (no timing farm — dwell-to-inflate the pre-request quote window; audit quote uses only pre-request measurements); **G-QUALPRUNE** (quality-pruning selectivity preserved — baseline PRUNE-destined LOO +6.1/+5.9 < FOSS-destined +6.9/+6.6 must NOT degrade; replaces the retracted "reduced forced-destruction"); **G-STAB** (EV, KL early-stop rate, per-op KL, grad anomalies, value-collapse, NaN/Inf); **G-DESTROY-TOTAL** (no net destructive harm — a quality-selective PRUNE-rate change is not a regression; guardrail on *harmful* destruction, read with G-QUALPRUNE).

**1.7 Tiers:** n=5 paired screen (SCREEN_PASS, never ACCEPT); n=10 claim (ACCEPT only at n=10). Seeds 41–45/41–50; device-pairing per Stage-2 §11.1. Budget ≥ Stage-2 floor (200 rounds/run; K=4 ⇒ 4 inner epochs/batch). P1 terminal-acc at n=5 is coarse (+0.77pp near a 5-seed noise floor) → screen relies on P2 + guardrails; P1 superiority deferred to n=10. Pre-committed demotion path (Stage-2 LEG-B style): a primary whose screen noise floor exceeds its MDE → descriptive at that tier, escalate to owner before the ON reveal.

**1.8 Frozen decision rule:** SETTLE (B−A) = P1 non-regression ∧ guardrails on B ∧ (Exp-1: P2 may be null). COUPLED (C−A) = P2 fires (quote-graded slope C>A, survives stratification) ∧ P1 non-regression@screen/superiority@n10 ∧ all guardrails on C. Verdicts: ACCEPT-SETTLE (SETTLE clean @n10 → ship settlement); ACCEPT-COUPLED (COUPLED clean @n10 → ship settlement + differentiable floor); SCREEN_PASS; REJECT (any hard guardrail fail ∨ arm-C P2 fails to fire); INCONCLUSIVE (owner). C−B reported, never a gate. Degeneracy backstop: B alone should still show ~93–94% floor-forced commits.

**1.9 Reversal triggers:** (fired — READ 5) shelter retracted → verdict on B−A/C−A, arm C justified by degeneracy, no PASS branch changes; val@1 ordering shows common pre-measurement PRUNE → downgrade "intrinsic" to "residual beyond no-PRUNE"; settlement dwell-farm → G-FARM REJECT; K=4/v4 doesn't reproduce the |H|=1 sign → regime-specific; arm C LOO-greedy commit collapse or degraded quality-pruning → HOLD (co-requisite settlement didn't carry enough signal, round-7).

**1.10 Frozen-abort smoke** (optional, unscored): short floor-fix-alone run (diff floor ON, settlement OFF, current reward), pre-registered to ABORT on fossils→0/PARK collapse. Backstops degenerate leg 2.

**1.11 Verification reads — ALL SATISFIED by READ 5 (values confirmed from `read5_out.txt`, 2026-07-14). FROZEN guardrail baselines:**
- PRUNE-legality at pinned WAIT decisions (age≥5): **100% / 100%** (1254/1254 s41, 1448/1448 s42) → the age-mask does not bite; the G-QUALPRUNE population is the age≥5 pinned cohort.
- Empirical per-decision destruction hazard `P(op=PRUNE | pinned HOLDING, age≥5, slot-targeted)`: **0.109 (s41) / 0.127 (s42)**, k̄=1.8/1.7 legal pinned decisions/lifecycle → the **G-DESTROY-TOTAL baseline** arm C is read against.
- Quality-pruning selectivity (the **G-QUALPRUNE baseline**, must NOT degrade in C): FOSS-destined lifetime LOO **+6.89/+6.58** vs PRUNE-destined **+6.08/+5.86** (pruning removes worse seeds).
- val@1 event-ordering: pruned at delay≤1 = **9.7%/9.4%** → val@1 is **pre-mediator for ~90%** → the B−A residual is labeled "~90% intrinsic, ~10% residual-beyond-no-PRUNE."
Remaining before freeze: **owner sign-off on the §1.13/§2.8/§3.5 flags only** (no reads outstanding).

**1.12 Freeze order (no-peek):** (1) land remaining reads on r9; (2) owner ratifies flags + freeze table; (3) freeze; (4) run arms; (5) score. Calibration touches only r9/read data, never the new arms.

**1.13 Owner flags:** ~~sequencing~~ **DECIDED: staged (owner)**; HRA posture; P1 vs P2 banked-primary per leg; τ_acc + MDEs + guardrail materiality; n/seeds/budget/device.

---

# Section 2 — Non-selectable settlement spec (the priority leg)

**2.0** The commit op issues a **non-cancellable** request; the seed stays validly measurable to a **pre-determined** boundary; the quote is computed **only from measurements dated before the request**; permanent integration + a fixed annuity ledger begin at the boundary. READ-1 invariant: any policy-selected settlement *instant* is farmable + unprovable-safe offline → the instant must be schedule-determined.

**2.1 State machine:** today FOSSILIZE→`execute_fossilize` is synchronous HOLDING→FOSSILIZED, no window. Proposed (behind `fossil_settlement`): commit op → `fossilize_committed` PENDING; seed stays HOLDING (ablatable/measurable) with `committed=True` + `settlement_boundary_epoch`; uncancellable (WAIT/SET_ALPHA/PRUNE masked; α + alpha_target/speed/curve frozen at request-instant); at boundary → HOLDING→FOSSILIZED + annuity begins + excluded from ablation. **OWNER FLAG:** new `REQUEST_FOSSILIZE` op (schema bump) vs re-interpret FOSSILIZE as request-issuance (recommend re-interpret, no bump).

**2.2 Boundary (READ-1 core):** non-selectable. Recommend **fixed-cadence audit boundaries** (global schedule every W_settle epochs + min window; settle at next boundary ≥ request+min_window). Policy chooses *whether*, never *when*. Alt: terminal at max_epochs (simplest but sparse/truncation-coupled). OWNER FLAG.

**2.3 Quote (fixed before request):** `c_settle` = EWMA (or min-over-window) of the seed's per-epoch cf rate over measurements timestamped ≤ request_epoch ONLY. Window measurements used for staleness-integrity, NOT added to the quote (dwelling can't move it). EWMA-as-quote valid because the instant is non-selectable. Uses landed `ContributionState`.

**2.4 Annuity:** `H = max_epochs − boundary`. Level annuity `c_settle`/epoch over [boundary, horizon], γ=0.995; PV `S = c_settle·Σγ^k`. **Replaces** the forfeited provisional stream (never additive). Ledger keyed by `seed_generation_id`.

**2.5 Holes closed:** (1) finite-horizon+discount §2.4; (2) negative c_settle→settle 0 (noncontributing penalty −0.2 still applies); (3) stale/never-measured fail-closed — request requires FRESH (staleness 0); STALE→force fresh ablation or REJECT; NEVER_MEASURED→invalid; never settle-0-and-commit (free-commit escape); (4) **one-shot-bonus netting (exact):** the existing `0.5+0.1·c` one-shot is spot-LOO-priced = the sell-at-spike channel → DROP the +0.1·c spot term; only the annuity (pre-request quote) is contribution-graded; keep at most a FLAT commitment premium (0.5, no c); audit `fossilize_terminal_scale=3.0`/`quality_ceiling=3.0` for any c-at-commit term; (5) double-payment — disjoint intervals, annuity replaces, quote locked once, uncancellable, generation-id guards reuse; (6) terminal/truncation — annuity is ordinary per-epoch reward, truncated by construction; too-late request → settle at terminal; (7) multi-seed — per-seed boundary/quote/generation-id/ledger, per-epoch reward sums annuities; (8) quote-vs-disbursement — quote may be shown via obs dim 31 (informative-but-uncashable), only the annuity enters reward; (9) window delay cost — seed keeps earning provisional bounded_attribution, α frozen, uncancellable; `fossilize_cost=-0.01`; no extra window cost unless owner wants (FLAG).

**2.6 Invariant `G_stay≈G_fossilize`:** for constant c, stay = `c·Σ_{1..T}γ^k`; fossilize-now = provisional-in-window + annuity = `c·Σ_{1..T}γ^k` = stay. **Equal by construction** → commitment reward-NEUTRAL → −113 discontinuity removed WITHOUT a farm. The policy commits because commitment is causally good for the host (terminal accuracy via ordinary GAE future-return, PDR-0078/0079); a small FLAT premium can reflect the proven intrinsic value (FLAG on magnitude; flat, never spot-LOO).

**2.7 ESCROW guard:** the landed `escrow_fossil_settlement` fail-closed guard is exactly what this makes wireable → flip to "configured" ONLY when a concrete validated settlement-strategy OBJECT is attached (not a bare string). Annuity is schedule-frozen, never re-measures the fossil → NO clawback path → avoids the ESCROW H4 pathology.

**2.8 Owner flags:** new-op vs re-interpret (recommend re-interpret); boundary fixed-cadence (recommend) vs terminal; W_settle + min-window; EWMA (recommend) vs min-window; flat-premium magnitude (0 vs small +); stale-request REJECT (recommend) vs force-fresh; explicit window cost (recommend none).

---

# Section 3 — Gradient-restoring op-distribution spec (re-scoped per READ 5)

**3.0** Retracted: "remove structural PRUNE probability." READ 5: forced-PRUNE is NOT a destroy-good-seeds bug (slot head prunes selectively — lower-LOO seeds); removing the PRUNE minimum could stop sensible quality-pruning. The live defect is the **gradient dead-zone:** the hard floor projects an underweight action to a constant → zero PPO gradient to its logits; at |H|=1 the whole op head is constant (67–80% of HOLDING, growing over training) → the op head can't learn commit/discard on ~94% of commit decisions. **New target:** make the floor DIFFERENTIABLE (restore learnable preference) WITHOUT zeroing the PRUNE minimum.

**3.1 Primitive — differentiable per-num_valid floor (round-8 ρ-sweep):** replace the hard projection with a soft everywhere-differentiable floor reproducing the current guarantee (each legal action ≥ `f=min(0.15,0.99/num_valid)`; cap `1−(n−1)f`). Uniform-floor form (minimal isolation): `q_i = f + (1 − n·f)·p_i`, `p=softmax(masked z)`; `∂q_i/∂z=(1−nf)∂p_i/∂z ≠ 0` — no dead-zone even at |H|=1. Equivalent mixture: `q=(1−λ)softmax(z)+λ·uniform(legal)`, λ=0.6 reproduces 0.15/cap-0.55, λ=0 = no-floor smoke. **Use per-num_valid f, NOT a global λ** (round-8 N5).

**3.2 Invariants:** (retracted "no PRUNE lower bound" → **replaced:** keep the per-num_valid per-action floor incl. PRUNE minimum, make it differentiable); exact sampled AND evaluated `q` (one shared helper for get_action+evaluate_actions); differentiable everywhere (nonzero gradient); NO STE (biased ratio = the prohibited bug class); hard transform behind `op_floor_primitive: Literal["hard_per_action","diff_floor"]` as the control (arm B hard, arm C diff → C−B isolates the primitive); per-head AND **per-op** approx-KL + ratio/clip-fraction logged (new telemetry, load-bearing under K=4); explicit rate of PRUNE at the structural minimum, **conditioned on PRUNE unmasked (age≥5)**; no global λ.

**3.3 Scope — OP HEAD ONLY** (causal isolation; the dead-zone blocking commit-learning is op-specific). Other heads keep the hard floor → C−B = op-head primitive only. Other-heads dead-zone (e.g. sub-0.12 blueprint) is a deferred workstream. OWNER FLAG.

**3.4 Caveat (round-7):** the gradient-restored op head may collapse to SET_ALPHA_TARGET (the new-WAIT attractor), not WAIT → "didn't collapse to WAIT" ≠ healthy. P2 (quote-graded commitment) confirms it learns to commit GRADEDLY; the frozen-abort smoke + G-STAB catch collapse; G-QUALPRUNE catches degraded pruning.

**3.5 Owner flags:** uniform-floor form (recommend `q_i=f+(1−nf)p_i`) vs λ-sweep; f magnitude (match 0.15 vs re-tune); op-head-only (recommend) vs all-heads; λ-sweep {0,0.3,0.6} within-arm vs single diff_floor.

---

## PROPOSED FREEZE TABLE — recommended answers to every flag (owner: ✓ / ✗ / adjust)
Drafted 2026-07-14 for owner yes/no/adjust. Grounded in the Stage-2 acceptance gate (`docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`) + the frozen r9 baselines (§1.11). Design-only; nothing freezes or runs until signed. Two rows marked **[drl power-calc at freeze]** need a final MDE/power computation before the table is frozen — everything else is decision-ready.

### Experiment (Section 1)
| # | Flag | RECOMMENDATION | Why | ✓/✗/adj |
|---|---|---|---|---|
| E1 | Sequencing | **STAGED** (Exp1 A/B, Exp2 add C) | DECIDED by owner | ✓ |
| E2 | HRA critic posture | **Shipping (non-HRA) critic, held identical across arms** | HRA was the EV-stab experiment that REJECTED (PDR-0059); PDR-0074 showed the critic mis-fit was a SYMPTOM of the observation gap, which obs-v4/L1 now fixes → the plain critic on honest inputs is the cleanest baseline, and holding it constant means it's not a confound. (Re-measure critic calibration post-hoc.) | |
| E3 | Banked primary per leg | **SETTLE (B−A): P1 non-regression + G-FARM. COUPLED (C−A): P2 (quote-graded commitment) as "did it fire," P1 non-regression@screen / superiority@n10** | Settlement-alone can't move accuracy (floor still forces 94%) → its job is remove-the-discontinuity-without-farming, judged on P1-non-regression + G-FARM. P2 (per-decision, high-power) is the mechanism test for the coupled arm. | |
| E4 | τ_acc (non-regression margin) | **−0.3 pp** (val-acc floor) | Matches Stage-2 G1 (val-acc ≥ −0.3pp). | |
| E5 | P2-slope MDE | **[drl power-calc at freeze]** — target ≈ detect a quote→commit slope ≥ ~⅓ of READ-3's CATE monotonicity (+1.19→+1.87/tercile) | Per-decision N (thousands/seed) makes this well-powered; the exact MDE needs the eligible-HOLDING count under obs-v4/K=4. | |
| E6 | Guardrail materiality | **G-FARM: dwell→quote slope CI must exclude a positive assoc. G-QUALPRUNE: PRUNE-vs-FOSS LOO gap (baseline +0.81/+0.72) must not shrink >50%. G-STAB: Stage-2 thresholds (EV, KL, grad-anomaly, value-collapse, NaN). G-DESTROY-TOTAL: net harmful-destruction non-increase, read with G-QUALPRUNE** | Grounded in §1.11 baselines + Stage-2. | |
| E7 | n / tiers | **n=5 paired SCREEN (SCREEN_PASS only), n=10 CLAIM (ACCEPT only)** | Matches Stage-2 discipline (screen never accepts). | |
| E8 | Seeds | **41–45 (screen), 41–50 (claim)**; paired fresh-init | Stage-2 seed lineage; paired per §11.1. | |
| E9 | Budget | **200 rounds/run, K=4 (4 inner epochs/batch), min 150 scored updates** | Stage-2 floor. | |
| E10 | Device | **quiet-box, single GPU, device-paired per Stage-2 §11.1** | The co-tenancy starvation that degraded a prior wave (PDR-0056). | |

### Settlement (Section 2)
| # | Flag | RECOMMENDATION | Why | ✓/✗/adj |
|---|---|---|---|---|
| S1 | New op vs re-interpret FOSSILIZE | **Re-interpret FOSSILIZE** as request-issuance | No obs/action schema bump; FOSSILIZE already means "commit." | |
| S2 | Boundary policy | **Fixed-cadence audit boundaries** (not terminal) | Non-selectable instant with denser, lower-variance settlement than terminal-only. | |
| S3 | W_settle / min_window | **W_settle = 10 epochs, min_window = 5** | 5 = one MIN_PRUNE_AGE; a boundary within ≤10 epochs of request keeps the annuity horizon long (fossilize ~epoch 75/150). | |
| S4 | Quote estimator | **EWMA over pre-request measurements** (span ≈ escrow_stable_window=3–5) | Instant is non-selectable → EWMA-as-quote is safe (READ 1); span matches the existing stable-window. | |
| S5 | Flat commitment premium | **0 (neutral settlement)** | Cleanest test — let the *causal* value of commitment (host accuracy via GAE future-return) drive commitment, not a hardcoded subsidy. A small flat premium is the fallback if C shows under-commitment. | |
| S6 | Stale/never-measured request | **Fail-closed: STALE → force fresh ablation or REJECT; NEVER_MEASURED → invalid** | No settle-0-and-commit escape hatch (§2.5.3). | |
| S7 | Per-window confirmation cost | **None** (the frozen-α + delayed-integration + fossilize_cost=−0.01 are the implicit cost) | Avoid an extra tuned term. | |
| S8 | One-shot bonus | **Drop the +0.1·c spot term; keep flat 0.5 only** | The +0.1·c is the sell-at-spike side channel (§2.5.4) — must go or the farm re-enters. | |

### Gradient floor (Section 3)
| # | Flag | RECOMMENDATION | Why | ✓/✗/adj |
|---|---|---|---|---|
| F1 | Primitive form | **Uniform-floor `q_i = f + (1−n·f)·p_i`** (per-num_valid) | Minimal causal isolation (reproduces the current floor/cap exactly at the current f); differentiable everywhere. | |
| F2 | f magnitude | **Match current 0.15** (per-num_valid `min(0.15, 0.99/n)`) | Isolate the *gradient* change, not the exploration magnitude (round-8 N5). | |
| F3 | Scope | **Op head only** | The commit-learning dead-zone is op-specific; other-heads floor is a deferred workstream → clean C−B. | |
| F4 | λ-sweep vs single | **Single `diff_floor` in arm C** (no within-arm sweep); the λ=0 no-floor case only as the frozen-abort smoke | The settlement is the treatment; a within-arm λ-sweep adds factors. | |

**After your pass:** I apply the ✓/adjusted answers, get the drl power-calc for E5 (+ confirm the n=10 P1 power for E4/E7), freeze the doc (no-peek), and bring you the frozen pre-registration for the GPU authorization. The remaining verification reads are already satisfied (§1.11); READ 5 is folded in.
