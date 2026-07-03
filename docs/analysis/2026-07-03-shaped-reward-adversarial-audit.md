# SHAPED Reward — Adversarial Audit (2026-07-03)

**Commissioned:** adversarial audit of the SHAPED contribution-family reward ("there's an underlying flaw in the shaped result even if I can't see it").
**Scope:** three mandated lenses — per-additend adversarial review, PBRS invariance-condition audit, behavioral-sink analysis.
**Branch audited:** `feat/phase-minus1-scale-falsifier` (SHAPED mode, `max_seeds=3`, γ=0.995, λ_GAE=0.95).
**Out of scope by commission:** coalition blindness / Committed-Shapley top-up design; the known commitment-avoidance defect itself; the entropy-density telemetry artifact.
**Method:** full read of `contribution.py` (1610 lines), `shaping.py`, `partition.py`, `rewards.py`, `types.py`, `residency.py`; reward-assembly call sites in `action_execution.py` / `vectorized_trainer.py` / `helpers.py`; mask + slot-canonicalization path (`tamiyo/policy/action_masks.py`, `tamiyo/networks/factored_lstm.py`); normalizer (`simic/control/normalization.py`); GAE (`simic/agent/rollout_buffer.py`); Phase-0 evidence packet, `metrics.md`, PDR-0008/0010. No code changes, no tracker writes.

---

## Executive summary

1. **[F1 — HIGH, CONFIRMED] The PRUNE sign-flip pays the full magnitude of a harmful seed's LOO as *positive* reward, unbounded, and it bypasses the Phase −1 clip** (the clip is applied *before* the flip). Pruning a seed measured at −15 pp pays **+15**. This double-counts the relief (the environment also stops charging the per-step negative) and — combined with F2 — the negative exposure is dodgeable. It directly confounds the in-flight scale-falsifier A/B on this branch.
2. **[F2 — HIGH, CONFIRMED] The step reward is a function of the *targeted slot only*, and WAIT is hard-canonicalized to the first enabled slot (r0c0).** With ~⅔ of steps being WAIT, dense attribution income is structurally concentrated on r0c0; seeds in r0c1/r0c2 are reward-invisible (both credit *and* penalties) except on the rare steps an op targets them. This is a position-asymmetric credit scheme, a standing "aim the op at the best seed" hacking surface, and a structural ceiling on corr(reward, J) — J sums over *all* seeds, the reward samples *one*.
3. **[F3 — MED-HIGH, CONFIRMED] The PBRS invariance claim does not hold as implemented.** Φ is sampled per targeted slot (action-dependent, not state-based); un-targeted stage transitions never pay their potential deltas, so the policy can harvest positive deltas and skip negative ones; Φ(terminal) is never zeroed (fixed-horizon episodes are always bootstrapped as truncations); and the germination deposit is paid timing-discounted but clawed back undiscounted. `pbrs_bonus` is ordinary shaping, not policy-invariant shaping.
4. **[F5 — MED, CONFIRMED-in-code] The all-off counterfactual falls back to min-LOO when the all-disabled ablation is missing** — which *overstates* the baseline exactly for coalition-carried seeds, spuriously tripping the attribution discount, the FOSSILIZE suppression/penalty, and the blending warning. Frequency unknown (probe specified).
5. **[F4 — MED, CONFIRMED] The additend set spans 3 orders of magnitude**, so every "economics" term (rents, warnings, interaction, hindsight) is vestigial relative to attribution — and the two Phase −1 arms each *reorder* the whole balance differently, so neither arm tests "same design, fixed scale."

Everything above is mechanism-level (what the reward *makes optimal*), not an occurrence claim. Each finding carries a probe costed at zero GPU-hours (offline on existing telemetry) or ≤ a few GPU-hours.

---

## Findings (ranked)

### F1 — PRUNE sign-flip: unbounded, clip-exempt positive payout for harming the host — **HIGH, CONFIRMED (arithmetic in code)**

**Mechanism.** For a non-fossilized seed with measured negative LOO, the dense attribution is
`bounded_attribution = contribution_weight · seed_contribution` (negative; **no** timing discount, **no** attribution discount — those apply only on the positive branch) — `contribution.py:606-608` vs `:626-638`.
The Phase −1 knobs are then applied: `attribution_unit_normalize` (`:688-691`) and the **positive-only** clip `min(clip, ba)` (`:692-696`) — a negative value passes through unchanged.
Then: `if action == PRUNE and not escrow_mode: bounded_attribution = -bounded_attribution` (`:710-711`).

Result: pruning a seed whose live LOO is −c pays **+`contribution_weight`·c** — in accuracy *points* (the cited run's negatives reach −15.3, per the Phase −1 comment at `:323`). The `ratio_penalty` added afterwards (`:722-723`) is ≤ 0 but capped at −0.3 and requires `seed_contribution > 1.0` (positive), so it never offsets this. The prune-shaping age gates (`min_prune_bonus_age=3`, `:1408-1416`) gate only the fixed +0.15 `prune_hurting_bonus`, **not** the flip.

**Why it is a defect and not "intended relief semantics."** The intended reading ("pruning a bad seed relieves you of its harm") double-counts: after the prune, future steps *also* stop paying the per-step negative (and val_acc-linked terms improve). The +c at the prune step is paid on top of that future relief. Worse, the negative exposure during the seed's life is avoidable:

- LOO is unmeasured (None) through TRAINING and the first BLENDING step (`:639-679`, proxy path pays no negatives by design).
- Per F2, a seed in a **non-canonical slot** (r0c1/r0c2) is essentially never the reward's target during WAIT steps, so its measured negative LOO is essentially never charged, and `blending_warning` (`:740-748`) never fires for it either.

So the loop *germinate into r0c1 → let it damage the host → prune at age ≥ 5* pays the accumulated damage magnitude once, having charged almost none of it. Per-episode costs against it: `germinate_cost` −0.15, global rent ≤ 1.5/step (small for small blueprints), occupancy 0.01/step. Upside: unbounded in damage magnitude.

**Interaction with the in-flight Phase −1 experiment (this branch).** On the `shaped_attribution_clip=2.0` arm, the flip output is **the sole remaining unbounded positive dense channel**. If the policy shifts probability mass toward prune-flip harvesting under the clip, the arm's churn/tail readout is confounded — the clip arm does not test "positive tail bounded," it tests "positive tail bounded except via PRUNE."

**Affected additends:** `bounded_attribution` (dominant), `action_shaping` (prune shaping), interacts with `blending_warning` absence.

**Probe (zero GPU-hours).** On the existing n=5 control runs (`telemetry/causal_r1_n5/control_s4*`): per action_name, count steps with `bounded_attribution > 0` and their mass; specifically `SUM(bounded_attribution) FILTER (action_name='PRUNE' AND bounded_attribution>0)` vs total positive ba mass, and the distribution of `seed_contribution` at PRUNE steps. (Attempted live via the Karn `rewards` view; timed out at 30 s in-session — run as a scripted DuckDB query against the run dirs instead.) A secondary probe: correlate per-episode prune count with summed prune-step positive ba — a positive relationship is the farming signature.

### F2 — Reward keys on the targeted slot only; WAIT canonicalizes to r0c0 — **HIGH, CONFIRMED (mechanism in code)**

**Mechanism.** `seed_contribution` is computed for `target_slot` only (`action_execution.py:835-839`); `seed_info`, PBRS, warnings, and attribution all derive from that one slot. The slot head chooses the target for every op; for WAIT, the network **forces** the slot to the first enabled slot: `canonical_slot_idx = slot_mask.int().argmax(-1)` (`factored_lstm.py:1300-1304`; mask comment "WAIT: slot is irrelevant" at `action_masks.py:238-239`). With germinate/prune churn ≈ 24 ops/episode out of ~150 steps, the large majority of steps are WAIT → the large majority of dense-attribution samples read **r0c0's seed**.

**Consequences (all mechanism-level):**

1. **Position-asymmetric credit.** A high-LOO seed in r0c0 pays the policy every WAIT step; the same seed in r0c2 pays only when explicitly targeted. The reward *defines* r0c0 as the high-yield slot. Note the r0c0-centric empirical structure already banked (54% of fossils early-conv@r0c0; r0c0 the "efficiency-enabling stem," PDR-0009): a reward-side position bias is a confounded alternative/contributor to the "controller learned a stem" reading, and worth separating before that motif is treated as pure learning.
2. **Asymmetric penalty invisibility.** Negative LOO, `blending_warning`, and `holding_warning` are equally target-gated — harm and dithering in non-canonical slots are unpriced (feeds F1; also makes `holding_warning` dead for r0c1/r0c2).
3. **Standing harvest surface.** Cheap ops pay the targeted slot's full ba: `SET_ALPHA_TARGET` costs −0.005 and is valid for BLENDING/HOLDING+HOLD (`action_masks.py:343-348`). The 2026-01-08 "turntable" fix (`contribution.py:752-771`) penalizes this only in HOLDING, only when ba>0, capped at 0.3/step — no cover in BLENDING, and 0.3 ≪ ba. "Aim a no-op-ish op at the best non-canonical seed" remains profitable whenever that seed's ba exceeds ~0.3.
4. **Structural reward⊄J wedge.** J = Σ_seeds Σ_t c_t·α_t/params (residency.py) integrates *all* seeds each step; the reward samples *one* c_t chosen by the action. Even a perfectly J-aligned per-seed signal would decorrelate under this sampling. This is a candidate primary explanation for corr(reward, J) = 0.212 that is *not* coalition blindness and *not* scale.

**Affected additends:** `bounded_attribution`, `pbrs_bonus`, `blending_warning`, `holding_warning` (all target-gated); `compute_rent`/`occupancy_rent`/`alpha_shock` are global (computed across slots, `helpers.py:107-166`) — the asymmetry is credit-side only.

**Probe (zero GPU-hours).** Offline on existing runs: (a) per-slot share of summed |ba| vs per-slot share of J-integrand mass (residency view) — divergence quantifies the sampling wedge; (b) **counterfactual re-scoring**: replay logged steps, recompute the dense term as Σ over slots of c·(slot==target ? 1 : 1) vs the logged single-slot value, re-measure corr(reward, J) — if corr jumps, the wedge is confirmed as the binding misalignment; (c) fossilize rate / HOLDING dwell / germination order by slot index.

### F3 — PBRS invariance conditions violated (four ways) — **MED-HIGH, CONFIRMED**

See the per-condition table in the dedicated section below. The headline violations:

- **Φ is action-dependent, not state-based.** Per-step PBRS is computed for the targeted slot only (`contribution.py:776-778`). Transition deltas are paid only if the slot is targeted on the exact post-transition step where `epochs_in_stage == 0` (`:1297-1308`); otherwise the `else` branch (`:1309-1314`) computes the within-stage delta and the transition delta is *silently skipped* — never deferred, never paid. A policy therefore harvests positive deltas (target the slot after upward transitions — automatic for r0c0 via WAIT) and skips negative ones (fossilize r0c1, then never target it → the −0.459 weighted HOLDING(mature)→FOSSILIZED(fresh) drop is never charged). The code's own warning (`:1298-1303`) acknowledges the mistiming fragility.
- **Terminal potential never zeroed.** Episodes are fixed-horizon and *always* flagged truncated (`action_execution.py:1429-1430`) with a V(s_end) bootstrap (`vectorized_trainer.py:2557-2609`). No terminal Φ forfeiture exists in SHAPED except the germination-deposit clawback, which covers GERMINATED/TRAINING only (`action_execution.py:1042-1078`). A seed parked in BLENDING/HOLDING/FOSSILIZED at episode end keeps its accumulated potential (weighted Φ up to 2.25–2.4).
- **Deposit/clawback arithmetic mismatch.** The germination PBRS bonus is multiplied by the timing discount (floor 0.4, `contribution.py:875-881`) — itself a non-potential multiplicative factor that breaks the PBRS form — but the terminal clawback reclaims the **undiscounted** bonus, inflated by 1/γ^age (`action_execution.py:1049-1074`). An epoch-0 germination still in TRAINING at T≈150 received ~0.12 and forfeits ~0.49 (≈4×). Conversely the clawback is dodged entirely by reaching BLENDING before T (last-minute germinate→advance→advance keeps deposit + stage-climb PBRS, net ≈ +0.7–0.9 per slot; see sink S4).
- **Normalizer breaks telescoping downstream.** γ matches (0.995 enforced at `contribution.py:472-476`; GAE uses `DEFAULT_GAMMA`, `rollout_buffer.py:519,622`), but the reward the buffer consumes is `reward / running_std` clipped to ±10 with run-lifetime Welford stats (`normalization.py:208-231`, instantiated once per run `vectorized.py:1068`). A nonstationary scale + hard clip is a nonlinear transform under which PBRS terms no longer cancel in return space. (Standard practice, but it means the "policy-invariance" comment block in `shaping.py:9-53` is not a property of the deployed system.)

**Net effect on policy:** `pbrs_bonus` and the PBRS parts of `action_shaping` are just dense shaping with slot-targeting and end-of-episode loopholes. Concretely, HOLDING→FOSSILIZED carries an immediate −0.459 weighted potential drop (progress bonus resets on transition), recovered at ~+0.08/epoch only if the slot keeps being targeted for ~7 epochs — one more per-step economic bias against committing late and against committing in non-canonical slots.

**Probe (zero GPU-hours).** Replay one logged episode through a PBRS ledger per seed: Σ(paid pbrs terms for that seed) vs γ^T·Φ(s_T) − Φ(s_0). The gap, per seed and per slot, is the realized invariance leak; stratify by slot index to expose the targeting effect.

### F4 — Mixed units / component balance: the economics terms are vestigial, and both Phase −1 arms reorder them differently — **MED, CONFIRMED**

Measured magnitudes (per step, defaults): `bounded_attribution` in accuracy points, observed to |15.3| (`ba` ≈ 68% of |reward|, metrics.md); `compute_rent` ≤ 1.5 (log-capped, `contribution.py:814-818`); `occupancy_rent` 0.01·excess; `fossilized_rent` 0.002·n; `blending/holding_warning` ≤ 0.4/0.3; `alpha_shock` ≥ −1.0; `interaction_bonus` ≤ 0.1; `hindsight_credit` ≤ 0.2 (leyline:425); action costs 0.005–0.15; fossilize package ≈ 1.4–1.7; `terminal_bonus` ≈ 2–3.

Consequence: no rent or warning can discipline any attribution-carrying seed; slot economics (D2) and dithering penalties are decorative at current scales. This *is* adjacent to the known scale pathology, but the audit-relevant point is about the falsifier itself: the **clip arm (2.0)** leaves ba ≥ every other additend (ordering preserved, magnitudes compressed), while the **unit-normalize arm (÷100)** makes ba ≈ ±0.15 — *smaller than* PBRS, fossilize bonuses, and terminal terms (ordering inverted; the policy is then primarily paid in PBRS/action-shaping currency, with all of F3's loopholes dominant). The two arms are not "same design at two scales"; they are two different reward orderings. Interpret the A/B with that in mind.

**Probe:** per-additend Σ|mass| and variance-share table per arm from the in-flight Phase −1 runs (offline; the §4 decomposition machinery already exists).

### F5 — min-LOO fallback for the all-off counterfactual mis-gates coalition seeds — **MED, CONFIRMED-in-code; frequency unknown**

`action_execution.py:856-865`: when `all_disabled_accs` lacks the env, `env_all_off_acc = min(baseline_accs.values())` — the *worst single-seed-removed* accuracy, which for any multi-seed coalition is ≥ the true all-off accuracy. `counterfactual_total_improvement` is therefore understated exactly when co-resident seeds carry joint value with small individual LOOs (the enabling-stem regime). A negative or sub-threshold value then trips, on a *genuinely good* coalition: the attribution sigmoid discount (`contribution.py:527-529`), FOSSILIZE attribution suppression (`:698-708`), the invalid-fossilize penalty −0.5−damage (`:1362-1376`), and `blending_warning` (`:744-748`). This is a commitment suppressor *independent of* per-seed coalition blindness — it corrupts the joint gate, which the top-up does not touch.

**Probe (zero GPU-hours if the ablation cadence is recoverable):** count steps per run where `env_idx ∉ all_disabled_accs` at reward time (re-derive from the CF-matrix logging; else one instrumented run). If the fallback fires ≈ never in production, downgrade to latent-bug; if it fires on a schedule (e.g., ablation cadence gaps), quantify the mis-gated step fraction.

### F6 — `holding_warning` gating quirks — **LOW-MED, CONFIRMED**

Fires only when `epochs_in_stage ≥ 2` **and** `bounded_attribution > 0` (`contribution.py:765`): a HOLDING seed with zero/None LOO waits penalty-free forever; cap 0.3 ≪ ba; and per F2 it never fires for non-canonical slots. The per-step hold-income asymmetry it was meant to police (collect ba while uncommitted) is the known commitment-avoidance economics — noted here only because the deployed guard is triple-gated into irrelevance, which the terminal top-up remediation does not change.

### F7 — Residual & telemetry-less terms — **LOW, CONFIRMED (bounded)**

The designed residual is `pending_auto_prune_penalty` (−0.2 per auto-prune, ≤ 0.6/step at 3 slots; `action_execution.py:1080-1081`, `:1526-1532`). `hindsight_credit` leaks into the residual only if summary telemetry is off (populated under `collect_reward_summary`, `:1091-1092` — precondition already documented in the Phase-0 packet §4). `germination_forfeit` is folded into the `action_shaping` component (`:1077-1078`) — sign-correct, but it semantically overloads that additend at the terminal step (a terminal clawback reads as "action cost" in per-additend analytics). Sign map in `partition.py:40-54` verified against the arithmetic: `compute_rent` stored negative (+1 ✓, `contribution.py:820`), `occupancy_rent`/`fossilized_rent` stored positive and subtracted (−1 ✓, `:848-854`), `escrow_forfeit` stored negative (+1 ✓, `action_execution.py:1036`). n=5 reconciliation already validated (max|residual| 0.4–0.6, auto-prune rows only). Nothing larger than ~0.6/step can hide in the residual as wired today.

### F8 — `interaction_bonus` payable on PRUNE steps — **LOW, CONFIRMED**

The gate reads post-flip ba (`contribution.py:786-795`): pruning a seed whose ba flipped positive (F1) can additionally collect the interaction bonus if `interaction_sum > 0` and the discount gate passes. ≤ 0.1, incoherent rather than material.

### F9 — Terminal semantics are mixed episodic/continuing — **LOW-MED, CONFIRMED mechanism; impact needs the critic**

`terminal_bonus = val_acc·0.05` (+ clawbacks) treats T as an ending (`contribution.py:948-958`), while the buffer flags every episode end as *truncation* and bootstraps V(s_end) (`action_execution.py:1429-1430`, `:1498-1509`; `vectorized_trainer.py:2557-2609`). For a genuinely fixed-horizon episodic task this bootstraps continuation value out of a state that never continues; V(s_end) is never regressed against any realized target (no episode extends past it), so end-game values are anchored only by generalization. The end-game (where fossilize-vs-hold is decided, where committed-J forms) is exactly where this slack lands. **Probe:** compare critic EV and advantage sign structure on the last k steps vs mid-episode on existing runs (offline); if end-game EV is distinctly worse, a terminal-handling A/B (bootstrap vs V=0 at T) is a ≤ few-GPU-hour run.

---

## PBRS invariance-condition table (Lens 2)

Claim under audit: `pbrs_bonus` (and PBRS parts of `action_shaping`) carry the Ng–Harada–Russell guarantee (`shaping.py:9-53` asserts it).

| # | Condition | Verdict | Evidence |
|---|---|---|---|
| a1 | Φ is a function of **state** only | **VIOLATED** | Φ evaluated for the action-chosen `target_slot` only (`action_execution.py:835-870` → `contribution.py:776-778`); WAIT forced to canonical slot (`factored_lstm.py:1300-1304`). Effective shaping F(s, **a**, s′). |
| a2 | Φ stationary across the episode (no annealing/curriculum/running-stat dependence) | **PARTIAL** | `STAGE_POTENTIALS` + progress bonus are stationary state functions (`shaping.py:55-68`, `contribution.py:1291-1295`) ✓; but the germinate PBRS term is multiplied by the epoch-dependent timing discount (`contribution.py:875-881`) — a non-potential factor (epoch is in the state, but the discounted term no longer telescopes against undiscounted later potentials). |
| a3 | Every transition's delta is charged exactly once | **VIOLATED** | Transition delta paid only when the slot is targeted on the exact `epochs_in_stage==0` step (`contribution.py:1297-1308`); otherwise skipped forever (`:1309-1314`). Policy-controllable (harvest ups, skip downs). |
| b | Applied as γ·Φ(s′)−Φ(s) with the return's γ=0.995 | **MET at reward layer / broken downstream** | γ enforced equal to `DEFAULT_GAMMA` (`contribution.py:472-476`); GAE uses the same (`rollout_buffer.py:519,622`). But the buffer consumes `reward/running_std` clipped ±10 (`normalization.py:208-231`, `vectorized.py:1068`) — nonstationary scale + clip destroys telescoping in return space. |
| c | Φ(terminal) = 0 (no phantom credit survives episode end) | **VIOLATED** | No terminal Φ forfeiture in SHAPED for stages ≥ BLENDING or FOSSILIZED; only the germination-deposit clawback for GERMINATED/TRAINING (`action_execution.py:1042-1078`), which reclaims an *undiscounted* deposit that was paid *discounted* (`contribution.py:875-881` vs `action_execution.py:1049-1053`). |
| d | Truncation-vs-termination bootstrapping consistent with the potential | **CAN'T-FULLY-DETERMINE / INCOHERENT** | Fixed-horizon ends always truncated + bootstrapped (`action_execution.py:1429-1430`; `vectorized_trainer.py:2557-2609`); V(s_end) has no regression target (episodes never continue), so whether the −Φ(s) component embedded in a consistent V_shaped materializes depends entirely on critic generalization. Terminal-reward terms simultaneously treat T as an ending. |

**Deviation characterization for THIS action space:** because the per-slot potential ladder (0→1→2→3.5→5.5→6 plus a 0→2 progress bonus that resets on every transition) is sampled through the targeting mechanism, the shaping is a harvestable income stream, not an invariant. Specific optimal-policy distortions: (i) stage-advance deltas are only worth taking in slots the policy will keep targeting (canonical); (ii) HOLDING(mature)→FOSSILIZED costs −0.459 up front, recoverable only via ~7 further targeted epochs — never recoverable in the last ~7 epochs or in untargeted slots; (iii) ending the episode holding high-Φ uncommitted states is free (condition c); (iv) last-minute germinate→advance chains keep the deposit (clawback dodge).

---

## Behavioral-sink register (Lens 3)

Format: mechanism → additends → cheapest falsification probe. Sinks marked ▲ are corollaries of F1–F3 (mechanism CONFIRMED, occurrence unproven); ▽ are lower-confidence hypotheses.

| # | Sink (dominated behavior) | Mechanism | Additends | Probe (cost) |
|---|---|---|---|---|
| S1 ▲ | **Late-episode fossilization** — committing in the last ~10 epochs is strictly dominated even when J-optimal | One-time −0.459 PBRS transition drop with no re-accrual time (F3), loss of remaining ba income for that slot (`:606` gates ba on `not seed_is_fossilized`), fossilized_rent, vs fossilize package ≈1.4–1.7 that a single further HOLDING step with ba≳1.7 already beats | pbrs_bonus, bounded_attribution, action_shaping, fossilized_rent | Histogram fossilize-epoch/T on existing runs; check mass collapse in final decile (0 GPU-h) |
| S2 ▲ | **Canonical-slot commitment** — fossilizing r0c0's seed specifically dominated; non-canonical seeds under-developed | r0c0 seed is the WAIT income stream (F2): fossilizing it zeroes the majority of the episode's dense income; r0c1/r0c2 seeds earn ~nothing per-step, so their *development* (which costs rent/warnings/op costs) is undervalued | bounded_attribution, all target-gated terms | Fossilize rate, HOLDING dwell, and germination order by slot index (0 GPU-h) |
| S3 ▲ | **Pre-warmup germination** — germinating before epoch 10 dominated even where early stems are J-optimal | Timing discount (floor 0.4) on positive ba and on the germinate PBRS deposit; negatives undiscounted (`:631-638`, `:875-881`) | bounded_attribution, action_shaping | Germination-epoch histogram: discontinuity at `germination_warmup_epochs=10` (regression-discontinuity read, 0 GPU-h) |
| S4 ▲ | **Terminal germination spam / advance-chains** — last-minute germinate→advance→BLENDING nets ≈+0.7–0.9 free | Deposit + stage-climb PBRS kept at truncation; clawback only hits GERMINATED/TRAINING (`action_execution.py:1060-1063`); costs: −0.15 germinate, −0.1 advance-from-TRAINING | action_shaping, pbrs_bonus | Late-episode germination tail + stage-at-end distribution; count seeds reaching BLENDING in final 5 epochs (0 GPU-h) |
| S5 ▽ | **Blueprint-size barbell** — mid-size blueprints dominated; tiny (rent≈0) or large (rent cap-saturated) favored | rent = min(0.5·log(1+ratio), 1.5): marginal rent → 0 beyond the cap; ba is param-agnostic while J divides by params — reward-vs-J divergence grows with params | compute_rent, bounded_attribution | Scatter seed params vs lifetime (ba − rent); frequency of rent at cap (0 GPU-h). Consistent with banked +252%-params inefficiency |
| S6 ▲ | **Harm-farming loop** (germinate r0c1/r0c2 → damage → prune) | F1 + F2 composite; unbounded payout, near-zero charged cost | bounded_attribution, action_shaping | F1 probe: PRUNE-step positive-ba mass by slot index (0 GPU-h); confirm with per-episode prune-count × prune-ba correlation |
| S7 ▽ | **Policy-driven ADVANCE avoidance** — waiting for mechanical/gate-driven advancement dominated over deliberate ADVANCE from TRAINING | `advance_from_training_penalty=−0.1` (`:938-940`) with no offsetting credit at the advance step (transition PBRS pays next step, only if targeted — F3a3) | action_shaping, pbrs_bonus | op-mix stats: ADVANCE rate from TRAINING vs BLENDING; dwell in TRAINING vs G2/G3 gate readiness (0 GPU-h) |
| S8 ▽ | **Entropy-anneal interaction** — exploration mass decays fastest on exactly the ops that need it (FOSSILIZE/PRUNE are rare heads); combined with S1/S2 sinks, annealed exploration may never re-sample commitment once the hold-income policy locks in | (not an additend — exploration schedule × sink geometry) | — | On existing runs: op-head probability mass on FOSSILIZE over training time, conditioned on a HOLDING seed being present (0 GPU-h) |

The known commitment-avoidance defect is the composite of S1/S2 + the hold-income asymmetry; it is listed here only to place the *new* structure (slot asymmetry, PBRS harvesting, prune payout) around it.

---

## Why corr(reward, J) = 0.212 has structure left to explain

Four independent, code-confirmed wedges between the paid reward and J:
1. **Sampling wedge (F2):** reward reads one action-chosen slot per step; J integrates all seeds.
2. **Sign wedge (F1):** prune-flip pays +|c| for measured harm; J books the same seed's residency as ≤ 0.
3. **Commitment wedge (known, out of scope):** ba pays uncommitted flow; J weights committed residency.
4. **Currency wedge (F4/F3):** ~32% of |reward| is shaping (PBRS/action/terminal) that is *not* potential-invariant (F3) and therefore not J-neutral noise but directional pressure (toward holding, canonical-slot activity, and end-of-episode potential parking).

A useful decomposition probe (0 GPU-h): recompute corr(x, J) per additend on the n=5 runs — if corr(ba_targeted-slot-only, J) ≪ corr(Σ_slots c·α, J) on the same steps, the sampling wedge is quantified directly.

---

## What I could not check (information gaps)

1. **Occurrence, not mechanism:** whether the trained policy actually exploits F1/F2/S1–S8. All probes are specified; the in-session Karn `rewards` queries timed out (30 s limit) — they need scripted DuckDB against specific `run_dir`s.
2. **F5 frequency:** how often `all_disabled_accs` is missing at reward time (ablation cadence not audited).
3. **Exact transition-step accounting:** which lifecycle paths land the reward on the `epochs_in_stage==0` step vs skip it (order of `transition()` → `step_epoch()` per path; `slot.py:520-531`, `:2453+`). The invariance conclusion (F3a3) holds either way; the per-path ledger does not.
4. **ESCROW mode** additends beyond partition-map sign verification (production is SHAPED).
5. **The `helpers.py:784` `compute_contribution_reward` call site** (single-env/legacy trainer path) — not audited for parity with the vectorized assembly.
6. **LOSS reward family**; **BASIC/BASIC_PLUS drip** internals (read but not adversarially swept — not production).
7. **Reward-normalizer episode-boundary behavior** under multi-env desync (shared run-lifetime stats across 12 envs assumed benign; not verified).
8. Whether `slot_by_op` canonicalization ever selects a non-r0c0 slot in production configs (it picks the first *enabled* slot; audited configs enable r0c0).

---

## Confidence assessment

**Overall confidence: HIGH on mechanisms (arithmetic read directly from code), LOW-to-MODERATE on behavioral occurrence (no telemetry probes executed).**

| Finding | Confidence | Basis |
|---|---|---|
| F1 prune-flip payout, clip bypass | **Confirmed** | `contribution.py:606-608, 692-696, 710-711, 722-723` — arithmetic, no inference |
| F2 target-slot reward + WAIT canonicalization | **Confirmed** | `action_execution.py:835-839`; `factored_lstm.py:1300-1304`; `action_masks.py:238-239` |
| F3 PBRS violations (a1, a3, c) | **Confirmed** | `contribution.py:776-778, 875-881, 1297-1314`; `action_execution.py:1042-1078, 1429-1430` |
| F3 condition d (bootstrap incoherence impact) | Plausible-needs-probe | Mechanism confirmed; magnitude depends on critic generalization |
| F4 magnitude table | **Confirmed** | Config defaults + metrics.md readings |
| F5 min-LOO fallback | **Confirmed-in-code**, frequency unknown | `action_execution.py:860-865` |
| F6–F9 | Confirmed (low materiality) | cited lines |
| S1–S4, S6 sinks | Plausible (mechanism confirmed, dominance regime not measured) | derived arithmetic |
| S5, S7, S8 sinks | Hypothesis | consistent with banked readings; needs probes |
| corr(reward,J) attribution to F2 | Plausible-needs-probe | decomposition probe specified |

## Risk assessment

**Implementation risk of acting on findings:** none from this audit (no changes made). **Risk of *not* probing before the next design decision:** MEDIUM-HIGH — F1 confounds the Phase −1 clip arm currently in flight, and F2 offers a confounded alternative explanation for the r0c0-stem narrative that PDR-0009 partially banks.

| Risk | Severity | Mitigation |
|---|---|---|
| Phase −1 clip-arm verdict confounded by clip-exempt prune channel (F1) | High | Run the F1 SQL probe on both arms *before* reading the A/B verdict |
| r0c0-stem finding partially reward-artifactual (F2) | Medium-High | Per-slot ba-mass probe; re-read PDR-0009 magnitude claims after |
| PBRS treated as "safe" in future design iterations (F3) | Medium | Stop citing the Ng-guarantee for this implementation; treat pbrs as ordinary shaping in any redesign scoring |
| Top-up experiment interpreted against a gate corrupted by F5 | Medium | Measure fallback frequency before PIN-E enablement analysis |
| Audit findings inflated into a redesign mandate without occurrence data | Medium | All findings carry 0-GPU-h probes; prioritize probes over design changes |

## Caveats & required follow-ups

**Before relying on this review:**
- [ ] Run the F1 probe (PRUNE-step positive-ba mass) on `causal_r1_n5/control_s41..45` — it converts the top finding from mechanism to occurrence (or bounds it as tail-only, like the 48× ransomware events).
- [ ] Run the F2 per-slot ba-mass and corr-decomposition probes on the same runs.
- [ ] Verify F5 fallback frequency before treating it as material.

**Assumptions made:** production = SHAPED + contribution family, `max_seeds=3`, config defaults as in `ContributionRewardConfig` (no per-run overrides audited beyond the frozen control config); first enabled slot = r0c0; WAIT dominates the op mix (~⅔, inferred from churn counts, not measured).

**Deliberate scaffolding vs defect, called honestly:** the prune sign-flip semantics, the target-slot reward shape, and truncation-bootstrap are *deliberate designs* whose interaction produces the audited pathologies — the defect is compositional. The clip-before-flip ordering (F1) and the discounted-deposit/undiscounted-clawback mismatch (F3c) look like unintended arithmetic, not design choices. The PBRS invariance *claim* (comment block, tests named "pbrs properties") is contradicted by the deployed composition regardless of intent.

**Not analyzed:** ESCROW-mode behavior, LOSS family, drip internals, the non-vectorized trainer path, observation-space effects on the critic (out of reward scope), and anything covered by the standing coalition-blindness/top-up track.
