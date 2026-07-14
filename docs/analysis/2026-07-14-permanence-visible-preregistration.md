# "Make permanence visible" — pre-registration + two coupled specs

**Status: OPEN — freeze on HOLD (round-15, PDR-0081). Nothing here is frozen; no GPU authorized.** Round-15 (both primes) folded in ~12 converged amendments + surfaced ONE genuine owner fork (the commitment premium). Freeze is blocked on THREE things: (1) the owner's premium call; (2) the corrected **paired-run** P2 power number + blinded-reassessment rule + frozen max-n; (3) one more adversarial pass on the integrated doc. Honest status: "round-15 amendments folded; needs another review pass," NOT "one decision from freeze" — round 15 found a fatal-class flaw (unfalsifiable ACCEPT-SETTLE) + a basic stats error (P2 pseudoreplication) after 14 rounds. The freeze table below (§ROUND-15) supersedes the older §1.13/§2.8/§3.5 flag lists. Grounded in PDR-0078/0079/0080/0081, the primes reviews, the Stage-2 acceptance gate, and a primary-source symbol map. Design-only; no code touched. drl-expert deliverable, 2026-07-14.

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

## ROUND-15 INTEGRATED FREEZE TABLE (PDR-0081) — ONE owner decision; the rest folded in
Freeze on HOLD. Both primes converged on ~12 amendments (Group B, accepted, shown for transparency) and passed the rest (Group C). They genuinely **disagree** on exactly one thing — the commitment premium (Group A) — which is the single owner decision. Design-only; nothing freezes or runs until (A) is decided, the paired-run power number lands, and one more review pass clears.

### GROUP A — THE SINGLE OWNER DECISION: the commitment premium
The proposed table contradicted itself (old S5 premium=0 vs old S8 "keep flat 0.5" — but `fossilize_base_bonus=0.5` **is** a flat premium). The primes resolve it opposite ways:

| Option | Who | The case | Cost |
|---|---|---|---|
| **A0 — premium 0 + verdict-changing raw-logit gate (RECOMMENDED, constructed middle)** | synthesis of both | Keep Exp-1's accounting test clean (premium 0); make claude-prime's **pre-floor FOSSILIZE-preference read (arm A vs B) a HARD gate**: if B's raw commit-preference is materially below A's → settlement priced commitment as *dominated* → NOT protocol-safe → add a flat un-farmable premium (+0.5–1.0) and re-run before Exp 2. Free (PDR-0070 logs pre-floor logits). | Neither prime proposed this exact form — owner accepts or bounces to primes. |
| **A+ — bake a flat +0.5–1.0 premium in NOW** | claude-prime | Under §2.6 neutrality, arm B's *complete* commit ledger is strictly negative (annuity−forfeited=0, premium 0, fossilize_cost −0.01, maint −0.002×71, frozen-α ≤0, PRUNE-option destroyed ≤0) → commitment is dominated → the floor forces 94% commits → ACCEPT-SETTLE is **bit-identical to a policy taught to avoid committing**. Flat premium is un-farmable (schedule-locked). | Muddies the accounting-continuity test (is B passing because sound, or bribed?). |
| **A0-strict — premium 0, gate stays descriptive** | gpt-prime | Exp-1's floor is load-bearing, so commitment SHOULD be floor-forced; Exp 1 = SETTLEMENT-PROTOCOL-SAFE (infra behind a flag, not a ship); commitment-preference is Exp-2's job. | Advisor: if the gate is only descriptive, claude-prime's fatal case stands untouched. → **A0 is A0-strict + the gate promoted to verdict-changing.** |

**Discriminating question for the owner:** *do you trust the pre-floor commit-preference gate (A0) to catch a dominated-commitment policy?* Yes → A0 (cleaner test; the data decides the premium in Exp 2). Not confident → A+ (hedge now). **Both primes agree the `+0.1·c` spot term dies either way** (sell-at-spike channel).

### GROUP B — CONVERGED AMENDMENTS (accepted, folded in; NOT owner questions)
| # | Amendment | Source |
|---|---|---|
| B1 | **P2 power unit = per-RUN slope, not per-decision** (old E5 was pseudoreplication). One P2 slope/run; pairwise across n=5/n=10 **runs**. **No obs-v4/K=4 pilot exists → n=10 cannot be pre-certified** → blinded post-screen variance reassessment, frozen max-n, rule written before launch. Parametric power curve = freeze-time drl task; NOT fundable-as-scoped until it exists. | gpt |
| B2 | **Two-quote design:** `q_decision=EWMA(c_{≤t−1})` (policy-visible, P2 predictor) vs `q_settle=EWMA(valid c collected AFTER request, span 5, ≥5 obs, invisible at request, locked at boundary)`. Annuity rate = `q_settle`. Removes sell-at-spike more directly than a locked pre-request EWMA. | gpt |
| B3 | **Signed-value invariant:** settlement applies identical clip/sign to provisional `bounded_attribution` across +/0/− contribution; "neg quote→annuity 0" valid only if the provisional stream applies the identical zero-plus-penalty transform. | gpt |
| B4 | **Add G-LEDGER** (hard): constant synthetic c ⇒ discounted return invariant to requesting FOSSILIZE within tol; report PV(credit), PV(annuity), net replacement error (mean/median/tail), completion/invalid rates. | gpt |
| B5 | **G-FARM → three tests:** spot-residual timing; audit-cycle phase; settlement-overpayment tail. (Selection on persistent `q_decision` intended; on transient residual/phase = farming.) | gpt |
| B6 | **G-QUALPRUNE → bootstrap the r9 baseline CI (before freezing "shrink >50%") + decision-level per-run `P(PRUNE|q_decision)`** (non-positive quote-sensitivity, top-quote not pruned more, C−B doesn't reverse). Cohort gap kept descriptive. | both |
| B7 | **G-DESTROY-TOTAL defined:** harmful = PRUNE of top-quote-tercile seed, or PRUNE→pre-defined adverse host-acc. Total PRUNE-rate alone is not harm. | gpt |
| B8 | **E2 critic:** non-HRA held-constant STANDS, but post-hoc calibration on arm A (does the commit-penalty still overstate MC by ~4–6× on honest inputs?) is a **pre-registered scored read** — if yes, PDR-0066 re-opens (over-read #14). | both |
| B9 | **F1 = "endpoint-matched," not "behaviour-identical."** C−B = an endpoint-matched differentiable primitive. Log full pre/post op dist, full **categorical** op KL, per-op prob changes, raw+transformed score-fn norms, near-floor rate `q_i≤f+ε`. | gpt |
| B10 | **S1:** version the env protocol + emit `FOSSILIZE_REQUESTED`/`FOSSILIZE_SETTLED` events. | gpt |
| B11 | **Late requests masked** (insufficient time for the ≥5-obs window + min horizon) — no underspecified "settle at terminal" fallback. | gpt |
| B12 | **Verdict taxonomy:** SETTLEMENT-PROTOCOL-SAFE (flag-land, not ship) / COUPLED-MECHANISM / SHIP-COUPLED (+powered P1 superiority C−A) / INCONCLUSIVE / REJECT. Test order P2→P1→guardrails. B−A is honestly the **whole protocol**, not pure annuity accounting. | gpt (=claude's "infra behind a flag") |

### GROUP C — AFFIRMED (both primes passed; unchanged)
S8 drop `+0.1·c` ("most important row") · S2/S3 fixed-cadence non-selectable boundary (W=10/min=5) · S1 re-interpret direction · F2 match `f=0.15` exactly · F3 op-head-only · F4 single `diff_floor` (λ=0 = unscored abort smoke) · §1.4 "no PASS branch requires a C−B magnitude" · E4 τ_acc=−0.3pp (**test: one-sided 95% LCB of paired diff > −0.3pp**) · E8 seeds 41–45/41–50 · E9 K=4/200-round (freeze burn-in/scoring window) · E10 quiet-box device-pairing · E5 left blank (correct).

### Freeze blockers (all three required)
1. Owner decides **Group A** (the premium). 2. The **paired-run** P2 power number + blinded-reassessment rule + frozen max-n (B1). 3. One more adversarial pass on the integrated doc (round 15 found a fatal-class flaw + a stats error after 14 rounds — the new-flaw rate isn't polish-level yet). **No GPU until all three clear and the doc is frozen no-peek.**
