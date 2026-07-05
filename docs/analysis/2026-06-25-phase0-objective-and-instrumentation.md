# Phase 0 — Objective Formalization + Stage-0 Instrumentation (Evidence Packet)

**Status:** PINS PROPOSED (J + additends) → reviewer sign-off → telemetry (TDD) → runs.
**GATE 0: PENDING** (telemetry not yet landed; ≥5-seed control-arm runs not yet executed → PROVISIONAL by construction until then).
**Date:** 2026-06-25
**Branch:** `feat/phase-minus1-scale-falsifier` (based on `0.3.0` → carries EV-stab Stage-2 HRA + Phase −1 scale flags; the methodology doc is being brought onto this branch — see §10).
**Methodology:** `docs/plans/concepts/2026-06-24-reward-redesign-methodology.md` §0 (J + forks), §5 (Phase 0 + GATE 0), §8 (anti-hack inventory). *(That doc currently lives on `codex/sanctum-pre-ready-crash`; this packet is self-contained.)*
**Sibling (in-flight, shares the control arm + seeds):** Phase −1 scale falsifier, `docs/analysis/2026-06-25-phase-minus1-scale-falsifier.md` (A/B PENDING).

---

## 0. What Phase 0 is — and is NOT

Phase 0 is **instrument-first, scope-agnostic measurement**. It (a) makes the objective **J** computable end-to-end from logged telemetry as a *post-hoc yardstick*, and (b) lands the Stage-0 telemetry that lets **GATE 1** later choose scope (variance-only vs redesign vs both).

- **No behavioral reward change.** J is computed offline; it does **NOT** enter the reward. (Phase −1 already added the experimental scale flags; Phase 0 adds **no** reward levers.)
- **No redesign presumed.** Phase 0's output is equally valid whether GATE 1 picks variance-only, redesign, or both.
- **No critic / Stage-2 HRA edit.** `ev_main/ev_cf/ev_sum`, `V_main/V_cf` are **reused**, not touched.
- **J's functional FORM is LOCKED** (owner-ratified, methodology §0): `J = Σ_t cf_total_improvement(seed)·α_residency_t / params`. Phase 0 pins **λ** (param-cost), the **gain-baseline**, and **which counterfactual** as *outputs*; it does not re-litigate the form. Raw residency is **REJECTED** (park-the-freeloader, §8).

---

## 1. The objective J — PINNED

### 1.1 Form (locked) and the integrand

$$ J(\text{run}) \;=\; \sum_{\text{seed } s}\; \frac{1}{\text{params}(s)} \sum_{t}\; \underbrace{c_t(s)}_{\text{counterfactual}} \cdot \underbrace{\alpha_t(s)\,\mathbb{1}[\text{stage}_t(s)\ge\text{BLENDING}]}_{\text{on-output-path residency}} $$

The integrand multiplies a per-seed **counterfactual contribution** $c_t(s)$ by the seed's **on-output-path residency** $\alpha_t(s)$ (the blend weight, gated to stages on the output path, `stage ≥ BLENDING = 4`). This is the J-integrand telemetry deliverable (net-new; `grep` confirms no residency telemetry exists today).

### 1.2 PIN A — which counterfactual multiplies in (**reviewer-critical**)

There are **two** counterfactuals at the per-step reward site (`action_execution.py`):

| symbol | code | line | meaning |
|---|---|---|---|
| per-seed LOO marginal | `seed_contribution = val_acc − baseline_accs[slot]` | `:832` | drop in accuracy if **this** seed is removed (leave-one-out; `baseline_accs[slot]` = "everyone but i", `vectorized_trainer.py:1070`) |
| env joint | `counterfactual_total_improvement = val_acc − all_disabled` | `:858` | total improvement of **all** seeds vs none (same value for every seed in the env) |

**PIN A (PROPOSED): the integrand uses the per-seed LOO marginal `c_t(s) = seed_contribution(s)`**, NOT the env joint.

- **Rationale (the freeloader guard, methodology §8 + unit test (d)):** under `max_seeds=3`, a high-α zero-contribution seed co-resident with a good seed *inherits* the env's positive `counterfactual_total_improvement`. With the env-joint multiplier it would score a large **positive** J — violating "a parked freeloader scores ≤ no-op". Only the per-seed LOO marginal zeroes that seed's integrand (`seed_contribution(freeloader) ≈ 0`) regardless of co-residents. **The locked form's notation `cf_total_improvement(seed)` — a function *of the seed* — supports the per-seed reading.**
- **Spec ambiguity disclosed + RESOLVED:** methodology §0 / HARD CONSTRAINT #4 write "`val_acc − all_disabled`" (env joint); §8 names the freeloader guard the env-joint reading *cannot* satisfy. The literal env-joint text is a **spec bug** — internally inconsistent with §8 of the same owner-ratified document: under the `Σ_seed` form, env-joint (constant across co-resident seeds) factors out and is N-counted, and it breaks the §8 freeloader guard the owner ratified as the *reason* raw residency was rejected. The two readings coincide only when one seed is on-path at a time (`max_seeds=1`). **PIN A is therefore forced, not preferred** — it is the only reading consistent with BOTH the `Σ_seed` form and §8.
- **Owner resolution (2026-06-25):** owner deferred to the coherence argument — "if it's the only coherent choice, make it and explain." PIN A (per-seed LOO) stands; the literal HARD CONSTRAINT #4 formula is treated as a spec bug to correct when the methodology doc lands on this branch.
- **Availability fallback:** `seed_contribution` is `None` when `slot ∉ baseline_accs` (LOO not yet measured — the first-BLENDING-step gap, `contribution.py:596-636`). Integrand policy on a `None` step: **contribute 0** (no measured counterfactual ⇒ no credit). This is conservative and consistent with the reward's own gating (`contribution.py:563,655-665,700-702` all skip on `None`). **The readout MUST report, per seed, the fraction of on-output-path steps (α>0, stage≥BLENDING) where `seed_contribution is None`** (reviewer-required): a large fraction means J is fallback-dominated and the yardstick is unreliable — a GATE-0 quality caveat. The gap is expected to be ~1 step (the TRAINING→BLENDING transition before alpha ticks); confirm empirically.
- **Proxy term excluded — corrected rationale (reviewer):** the proxy `proxy_contribution_weight·improvement_since_stage_start` is **NOT** used in J. The correct reason is that `improvement_since_stage_start = current_val_acc − acc_at_stage_start` is a **non-counterfactual pre/post *temporal* delta** (stage-relative, NOT an ablation — `contribution.py:630-636`), self-rated at 0.3 confidence (`proxy_confidence_factor`); it is confounded by concurrent host *and* other-seed progress. (The earlier "host-drift-confounded" wording was imprecise — the proxy is explicitly stage-relative, not host-wide `acc_delta`; the conclusion — exclude from J — stands.) J credits only the clean LOO counterfactual.
- **Disclosed limitation (LOO sub-additivity, Phase-1 watch):** LOO marginals do not sum to the joint (they ignore synergy), so `J(run) = Σ marginals` is a per-seed efficiency yardstick, **not** a total-accuracy-gain figure, and two redundant-but-individually-useful seeds each score LOO≈0 (a "dual-failure" that *looks* like two freeloaders). This is acceptable for J's purpose (the null-player/freeloader guard is a lower bound that LOO satisfies exactly), but Phase 1 should cross-check J's seed ranking against the **terminal Shapley** already computed (`shapley_results`, `vectorized_trainer.py`), whose null-player axiom IS the freeloader guard *and* sums to the joint. Per-step Shapley is K!-cost, so it cannot be the integrand — LOO is correct for the per-step form.

### 1.3 PIN B — the parameter normalizer

**PIN B: `params(s) = effective_seed_params(s)`** — the seed's own active parameter count, read **per-slot** as `model.seed_slots[slot].active_seed_params` (`action_execution.py:847`; `SeedSlot.active_seed_params`, `slot.py:1298`). J is *per-seed-parameter* committed gain, so the seed's own params is the correct denominator (not `total_params`, which dilutes by the fixed host, nor `host_params`). **Citation correction (reviewer):** the *aggregate* `host.active_seed_params` (`host.py:954`) sums across ALL active seeds — it is the wrong line for "the seed's own params"; the per-slot site is `slot.py:1298`. **Time-invariance snapshot (reviewer):** because `1/params(s)` is pulled *outside* `Σ_t`, pin the snapshot convention — use the seed's **terminal/last-on-path** active-param count (fixed once the module is instantiated). Pinned in the worked example (§1.6).

### 1.4 PIN C — λ (param-cost weight)

The locked form realizes the param cost as the **per-parameter divisor** (`/params`), so for the **raw, GATE-0-computable J** the additive cost weight is **λ = 0** (no separate subtractive `−λ·params` term). λ as a *subtractive* cost is a Phase-1 lever (it trades off gain vs param burn when comparing candidates); pinning it now would be a behavioral presumption. **PIN C: λ_raw = 0; the divisor `1/effective_seed_params` IS the param cost for GATE 0.** Phase 1 may add a subtractive λ when comparing arms; flagged, not built. **Disclosure (reviewer):** the pure ratio form gain/param has a known pathology — it can favour an infinitesimal seed with infinitesimal gain (ε/ε large). This is *precisely why* Phase-1's subtractive λ exists; `λ_raw = 0` must **not** be misread as "params are free under GATE 0" — GATE 0 only asserts J is *computable*, not optimized.

### 1.5 PIN D — the gain-baseline (`off_switch`)

The run-level gain-baseline is **`off_switch`** = the `ProofBaselineMode.OFF_SWITCH` cohort (`proof_baselines.py:57-58`, `cohort_id="off_switch"`): the controller-does-nothing reference. The `J(M) − J(off_switch)` subtraction is a **Phase-1 / contribution-gate** operation, **NOT a GATE-0 requirement** (methodology §5: GATE 0 needs raw J computable, not the subtraction).

- **PIN D: `off_switch` is the gain-baseline; the subtraction needs an `off_switch` run (flagged for Phase 1, not run here).**
- **Explicitly NOT wired into J:** `STATIC_FINAL` and `FIXED_SCHEDULE_GERMINATE` (`proof_baselines.py:88,101`) are the **contribution-gate** lifecycle arms — different references, not `off_switch`. They do not enter J.
- **GATE 0 asserts NO efficacy (reviewer, un-missable):** raw J *without* the `off_switch` subtraction is **not** "controller contribution." GATE 0 certifies only (i) J is computable end-to-end and (ii) the freeloader sign property holds. Any downstream consumer that reads a positive raw J as "the controller works" is unsound — the `J(M) − J(off_switch)` subtraction (Phase 1) is what licenses an efficacy claim.

### 1.6 Worked numeric example (synthetic; the unit-test fixture pins it)

> The real-logged-episode worked example fills in once residency telemetry lands and a control-arm episode is logged (§5). The synthetic fixture below is the *definitional* worked example unit test (b) reproduces bit-for-bit.

One seed `s`, `effective_seed_params(s) = 2000`, three on-output-path steps after BLENDING entry:

| t | stage | α_t | seed_contribution c_t (pts) | integrand c_t·α_t |
|---|---|---|---|---|
| 1 | BLENDING(4) | 0.25 | 4.0 | 1.00 |
| 2 | BLENDING(4) | 0.50 | 6.0 | 3.00 |
| 3 | HOLDING(5) | 1.00 | 8.0 | 8.00 |

Residency integral = 1.00 + 3.00 + 8.00 = **12.00**. `J(s) = 12.00 / 2000 = 6.0e-3`. With one seed, `J(run) = 6.0e-3`.

**Freeloader check (test d):** a co-resident seed `f` with `α_t ≡ 1.0` but `seed_contribution ≡ 0` over the same steps contributes `Σ 0·1.0 / params = 0` ≤ a no-op (which is exactly 0). Under the env-joint multiplier it would instead score `Σ (env_total>0)·1.0`, a positive — which PIN A rejects.

---

## 2. Counterfactual noise-floor — method (PIN E)

**PIN E (method): the inert-seed placebo credit distribution is the noise-floor proxy** (methodology §5 option b), NOT re-running LOO over K resampled val minibatches (option a).

- **Why:** the placebo proxy reuses the host-drift placebo harness (deliverable 7) — an inert seed (identity/zero-output module) on the output path *should* receive `c_t ≈ 0`; the empirical spread of its credited contribution **is** the LOO counterfactual's noise floor + bias, per stage, at zero extra ablation cost. Option (a) would require K extra fused val passes per step (expensive) for a marginally tighter floor.
- **Reported, per stage:** mean (bias), std (noise floor), CoV of the inert-seed credit.
- **Non-degeneracy requirement (reviewer — must hold for the report to count):** a *true zero-output* inert seed (identity/zero module) produces **bit-identical** logits in the main pass and the leave-this-slot-out ablation pass on the same val minibatch, so `c_t ≡ 0` *deterministically* — a vacuous "noise floor" of ~0 that **understates** a real seed's estimator noise (a real seed perturbs logits and carries genuine minibatch/sampling variance). The placebo MUST therefore be a **near-inert *small-real* seed** (or carry an assertion that its credit spread is driven by the same noise sources that move a real seed's `c_t`); a deterministic-zero placebo does **not** satisfy the noise-floor deliverable.
- **Operating-range caveat:** the placebo characterizes LOO noise at *zero* contribution (the freeloader decision threshold). It does **not** characterize heteroscedastic estimator variance at non-zero magnitudes — if Phase 1 densely *pays* J, require per-magnitude (option-a) variance on a subsample.
- **Does it block GATE 0?** **No** (report-not-clear is the bar). But the non-degeneracy requirement blocks the *meaningfulness* of the reported line, so it must hold before the checkbox counts.
- **Concrete design (building blocks located 2026-06-26):** the `noop` blueprint (`kasmina/blueprints/cnn.py:21`, `param_estimate=0`, "Identity seed") is the DEGENERATE case (deterministic-zero floor) — do NOT use it. Use a **small real blueprint** (`depthwise`/`conv_light`) with `zero_init_final_layer` (`kasmina/blueprints/initialization.py:29`, `delta(x)≈0` at birth, real params), forced onto the output path via the `FIXED_SCHEDULE` proof-baseline lifecycle (`training/proof_baselines.py`), running with the residency telemetry on. The per-stage spread of its credited `seed_contribution`/`cf_weighted_integral` = the noise floor (mean=bias, std=floor, CoV). **An offline proxy from the live runs is rejected** — selecting on low-contribution seeds is circular and confounds true small-negative contribution with estimator noise. ⇒ a short dedicated run, slotted in when a GPU frees after the control sweep.

---

## 3. Term inventory — every LIVE reward term (deliverable 6)

Swept across `rewards.py` (dispatch), `contribution.py` (`compute_contribution_reward`), `shaping.py` (PBRS), `loss_primary.py`, `partition.py`, and the live lifecycle credit under `src/esper/simic/training/handlers/` (there is **no** `rewards/handlers/`). Covers BASIC / BASIC_PLUS / SIMPLIFIED / SHAPED / ESCROW.

**Classification key:** `A` = additend (sums into `total_reward`), `M` = multiplier (scales an additend), `S` = state (ledger/carry), `D` = diagnostic (not summed). C/U = committed / uncommitted credit.

| term | where (line) | modes | class | C/U | notes |
|---|---|---|---|---|---|
| `bounded_attribution` | `contribution.py:682` | SHAPED, ESCROW | **A** | U | the dense cf channel = `R_cf`. **Already contains** `ratio_penalty` (`:680`), the PRUNE sign-flip (`:668`), FOSSILIZE-suppression (`:665`), `timing_discount`/`attribution_discount` (`:583,595`). ESCROW: = `escrow_delta` (`:561`). |
| `blending_warning` | `:705` | SHAPED, ESCROW | **A** | U | ≤0 penalty for negative-cf BLENDING. |
| `holding_warning` | `:728` | SHAPED, ESCROW | **A** | U | ≤0 anti-turntable penalty. |
| `pbrs_bonus` | `:735` | SHAPED, ESCROW | **A** | U | stage-potential PBRS (`_contribution_pbrs_bonus`). |
| `synergy_bonus` | `:744` | SHAPED, ESCROW | **A** | U | `tanh·0.1` interaction bonus (`_compute_synergy_bonus`, `:1268`). |
| `compute_rent` (= −`rent_penalty`) | `:767/769` | SHAPED, ESCROW | **A** | U | param-overhead rent. |
| `alpha_shock` | `:777` | SHAPED, ESCROW | **A** | U | convex α-delta penalty (capped). |
| `occupancy_rent` | `:798` | SHAPED, ESCROW | **A**(−) | U | slot-saturation cost (subtracted). |
| `fossilized_rent` | `:803` | SHAPED, ESCROW | **A**(−) | C | per-fossil maintenance (subtracted). |
| `action_shaping` | `:893` | SHAPED, ESCROW | **A** | mixed | **composite**: germinate/prune/advance/set-α costs + germinate/prune PBRS + **immediate fossilize bonus** (`:874`) − germination_forfeit (`action_execution.py:1073`). Un-decomposable at telemetry layer. |
| `terminal_bonus` | `:907` | SHAPED, ESCROW | **A** | C | `val_acc·terminal_acc_weight` at terminal step. |
| `escrow_forfeit` | `action_execution.py:1031` | ESCROW | **A** | C | terminal clawback of unrealized escrow. |
| `hindsight_credit` | `action_execution.py:1087` | SHAPED, ESCROW | **A** | C | scaffold-hindsight (LIVE; `compute_scaffold_hindsight_credit` `contribution.py:1280`, applied `training/handlers/fossilize.py:99`). |
| `pending_auto_prune_penalty` | `action_execution.py:1075` | SHAPED, ESCROW | **A (telemetry-less)** | U | **no component field** → captured by the reconciliation `residual` (§4). |
| `ratio_penalty` | `:690` | SHAPED, ESCROW | **D / M** | — | **NOT an additend** — already folded into `bounded_attribution` (`:680`). Summing separately double-counts. |
| `fossilize_terminal_bonus` | `:910` | all | **D** | — | **always 0.0** (real bonus is inside `action_shaping`). Dead additend. |
| `attribution_discount`,`timing_discount` | `:689,695` | SHAPED, ESCROW | **M** | — | scale `bounded_attribution`; not summed. |
| `escrow_credit_prev/target/next`,`escrow_delta` | `:691-694` | ESCROW | **S** | — | ledger carry; `escrow_delta` is *inside* `bounded_attribution`. |
| `drip_this_epoch`,`drip_immediate_bonus`,`drip_deferred_total` | BASIC_PLUS | BASIC_PLUS | **A/S** | C | post-fossil drip; **not in SHAPED/ESCROW** (0 on the control arm). |
| `base_acc_delta`,`val_acc`,`host_baseline_acc`,`seed_stage`,`epoch`,`growth_ratio`,`num_*`,`seed_contribution`,`n_active_seeds` | — | all | **D** | — | context/diagnostic. |
| `compute_basic_reward` rent/pbrs/fossilize | `:967-1203` | BASIC, BASIC_PLUS | **A** | mixed | separate dispatch; control arm is SHAPED so out of scope for the variance readout but inventoried. |
| `compute_simplified/minimal/sparse_reward` | `:920-1232` | SIMPLIFIED/MINIMAL/SPARSE | **A** | C | terminal-anchored; not the control arm. |

**The SHAPED/ESCROW additend set (what the per-term decomposition reconciles over)** — see §4.

---

## 4. The additend set + reconciliation (the GATE-0 verification)

**Primary GATE-0 test (corrected after reviewer adjudication): `residual ≈ 0`, fully attributable to `pending_auto_prune_penalty`.** The naïve "shares sum to 1" criterion is a **tautology** and must NOT be used as the reconciliation test: with `residual := reward_raw − Σ(named)` defined by subtraction, the named-plus-residual set sums to `reward_raw` by construction, and by covariance linearity (population convention) `Σ_i Cov(R_i, R)/Var(R) = Cov(Σ_i R_i, R)/Var(R) = 1` *for any* classification — including a wrong one where a live term hides inside `residual`. Shares-≠-1 could then only be floating-point error, never a misclassification. **The falsifiable test is instead: `residual` is ≈ 0 on non-auto-prune steps and equals `pending_auto_prune_penalty` on auto-prune steps, and `residual`'s variance-share is negligible.** A misclassified or dropped live term inflates `residual` (and its share) while the total stays exactly 1. "Σ shares == 1" is demoted to a floating-point sanity check.

For SHAPED/ESCROW the finalized `reward_raw` (`action_execution.py:1090-1091`) decomposes as:

```
reward_raw  ==  bounded_attribution            # incl. ratio_penalty, PRUNE-flip, FOSSILIZE-suppression
              + blending_warning
              + holding_warning
              + pbrs_bonus
              + synergy_bonus
              + compute_rent                    # = −rent_penalty
              + alpha_shock
              − occupancy_rent
              − fossilized_rent
              + action_shaping                  # incl. immediate fossilize bonus, − germination_forfeit
              + terminal_bonus
              + escrow_forfeit                  # ESCROW only (= component value, already signed)
              + hindsight_credit
              + residual                        # := reward_raw − Σ(above) ≡ pending_auto_prune_penalty
```

- `residual` is defined **by subtraction** (the `partition.py` "exhaustive-by-construction" trick), so the named-plus-residual identity holds for *any* input; `residual` is ~0 except on auto-prune steps. It captures the one telemetry-less term and any future un-mirrored correction.
- **Per-term SIGNED-contribution map (reviewer — required in the telemetry; prevents a false misclassification hunt).** Component fields are stored in **mixed sign conventions**; each must enter the decomposition as its *signed contribution to `reward_raw`*:

  | term | stored as | signed contribution |
  |---|---|---|
  | `bounded_attribution`, `blending_warning`, `holding_warning`, `pbrs_bonus`, `synergy_bonus`, `alpha_shock`, `action_shaping`, `terminal_bonus`, `hindsight_credit` | already-signed | **+field** |
  | `compute_rent` | already-signed (= −rent_penalty, ≤0) | **+field** |
  | `escrow_forfeit` | already-signed (= −escrow_forfeit) | **+field** |
  | `occupancy_rent` | positive magnitude; reward `-=` it (`:798`) | **−field** |
  | `fossilized_rent` | positive magnitude; reward `-=` it (`:803`) | **−field** |

  Naïvely summing the raw fields flips `occupancy_rent`/`fossilized_rent` → `residual` ≠ 0 → a spurious "misclassified term" hunt for a non-existent bug.
- **`residual ≡ pending_auto_prune_penalty` precondition (reviewer):** `hindsight_credit` is added to `reward` *unconditionally* (`action_execution.py:1083`) but its component field is populated only inside `if collect_reward_summary` (`:1086`). With summary OFF, `hindsight_credit` would leak into `residual`. The control arm runs SHAPED with summary ON ⇒ GATE-0-valid; state the precondition so it is not assumed universally.
- **Top-line output named explicitly:** `share_attribution = Cov(bounded_attribution, reward_raw) / Var(reward_raw)` — this is the **`R_cf` share GATE 1's variance leg reads (`> 0.40`)**, the term Phase −1 perturbs. Surfaced as a first-class scalar, not buried in the per-term table.
- **Covariance convention:** population (`unbiased=False`) covariance and variance, matching the existing `cov_rcf_return_share` (`ppo_agent.py:767-777`).

---

## 5. Variance-share + residency readout (PENDING runs)

Populated after ≥5 control-arm seeds run with the new telemetry. Pre-registered outputs:
- per-term `Cov(R_i, reward_raw)/Var(reward_raw)` (signed map §4), **top-line `share_attribution`**; `residual` variance-share (must be ≈ 0);
- per-term CoV; per-term corr to final committed J and to `fossilize_count`;
- residency-integral J per seed/episode + the run-level J; the worked example on one logged episode;
- inert-seed placebo noise floor (mean/std/CoV per stage, non-degenerate per §2);
- **committed-vs-uncommitted credit split per episode — MANDATORY** (reviewer), not optional;
- **residency-length sensitivity (reviewer):** J recomputed truncated at the commit step vs full residency — does it rank-order arms differently? Surfaces survival-time accrual (§5.1);
- per-seed **fraction of on-path steps with no measured LOO** (`seed_contribution is None`) — J-reliability caveat (§1.2).

### 5.1 Survival-time residency accrual — reconciliation (reviewer headline finding)

The reward-function-reviewer flagged that the residency integral spans `stage ≥ BLENDING`, which **includes uncommitted BLENDING/HOLDING residency** (the §1.6 worked example has zero fossilized steps), so a genuinely-contributing seed that *never* fossilizes still accrues J. This is a property of the **locked residency-integral form**, not of PIN A. **Reconciliation:** the methodology owner **already ratified** (methodology §0 + §9, 2026-06-24) that "committed" is keyed to **on-output-path residency (alpha-weighted), NOT the fossilize event** — fossilization is an irreversible slot-burn, not the only contribution path, and `max_seeds=1` makes "never fossilize" partly structurally rational. So uncommitted-on-path residency *is* the ratified commitment unit, and fossilize-rate is a **diagnostic, not a gate** (§9). The reviewer's concern is therefore not an open form-vs-intent conflict but a **measurement-quality** one, addressed by: (i) the MANDATORY committed/uncommitted split, (ii) the residency-length sensitivity readout, (iii) reporting fossilize-rate as a diagnostic. The fossilize-rate-as-gate question re-opens at `max_seeds ≥ 2` (§9). This is surfaced to the owner for awareness; no form change.

**REAL-DATA validation already obtained (no new wiring, no GPU runs).** The additend/variance legs were validated offline against the cited single-seed SHAPED run `telemetry_2026-06-24_033049` (Karn `rewards` view, 45,696 logged rows) by running the §4 signed decomposition per row vs the logged `total_reward`:

| metric | value (cited single-seed run; hypothesis-generating) |
|---|---|
| per-step reconciliation: rows with `residual` exact (≤1e-6) | **43,301 / 45,696 (94.8%)** |
| rows with `1e-3 < |residual| ≤ 1e-2` (telemetry float-rounding) | 2,395 (5.2%) |
| rows with `|residual| > 1e-2` (would signal a missing term) | **0** |
| `max\|residual\|` | **0.002** (rounding-level; a missing term would be O(0.01–1)) |
| `share_residual` = Cov(residual, R)/Var(R) | **0.0** (residual carries no return variance ⇒ set complete) |
| **`share_attribution`** = Cov(bounded_attribution, R)/Var(R) | **0.819** (≫ 0.40 GATE-1 threshold) |
| `share_action_shaping` | 0.137 |
| `share_terminal` | 0.045 |
| `share_pbrs` / `share_compute_rent` / `share_synergy` | ≈ 0 |
| `stage_bonus` / `fossilize_terminal_bonus` / `ratio_penalty` nonzero rows | **0 / 0 / 0** (exclusions confirmed empirically) |
| Σ broken-out shares | ≈ 0.997 (≈ 1.0; gap = small terms not tabled + rounding) |

This **validates the additend set on real data** (my unit tests set `reward_raw` to the exact sum, so they prove arithmetic but not set-completeness — this does). `share_attribution = 0.82` reproduces the methodology's central claim through the Phase-0 machinery. **Caveat:** single-seed (`max_seeds=1`); the ≥5-paired-seed control-arm value is the actual GATE-1 input. The residency/J legs below genuinely need the new telemetry + runs.

**Karn-view gap found (telemetry-chain TODO):** the `rewards` view does NOT expose `occupancy_rent` / `fossilized_rent` (they are on the leyline dataclass but absent from the view projection), so multi-seed reconciliation from the view alone is impossible — they happened to be 0 on this single-seed run. The wiring step must add these two columns to the `rewards` view/emit path.

**MULTI-SEED control-arm results — GATE-0 LOCKED at n=5 (commit `406aeb25`, seeds 41–45, `max_seeds=3`, 200 episodes each, ~360k reward rows each).** Per-seed and aggregate:

(`committed J` column removed — structurally 0 for any policy, see retired-evidence note below.)

| seed | `share_attribution`† | corr(reward,J) | fossilize/ep | mean J/ep |
|---|---|---|---|---|
| 41 | 0.923 | 0.189 | 0.201 | 0.072 |
| 42 | 0.923 | 0.292 | 0.200 | 0.067 |
| 43 | 0.924 | 0.206 | 0.212 | 0.088 |
| 44 | 0.922 | 0.173 | 0.212 | 0.097 |
| 45 | 0.933 | 0.203 | 0.212 | 0.075 |
| **AGG n=5** | **0.925 ± 0.004**† | **0.212 ± 0.046** | **0.207 ± 0.006** | — |

† magnitude-confounded (`ba` ≈ 68% of reward magnitude) — see downgrade note.

- **Reconciliation holds across all 5** (`max|residual|` 0.4–0.6, ~6–7 rows >1e-2 of ~360k = auto-prune `residual`; `occupancy_rent`/`fossilized_rent` exercised, `−1` sign confirmed on real data).
- **`share_attribution = 0.925 ± 0.004` — DOWNGRADED, substantially mechanical** (advisor 2026-06-27). On control:41, `mean|bounded_attribution| / mean|reward| = 68%`: a term that is ⅔ of the reward *magnitude* dominates `Cov/Var` near-automatically (same tautology family as the rejected "shares-sum-to-1"). So 0.92 is NOT a clean "attribution drives the policy" signal; it is mostly "biggest addend in the sum." The `>0.40` GATE-1 threshold is met but weakly informative. Keep as a magnitude fact, not a behavioral claim.
- **committed J = 0 — RETIRED as evidence** (advisor 2026-06-27): this is a *structural tautology*, not a finding. Fossilized seeds are excluded from the LOO ablation ⇒ `seed_contribution = None` ⇒ the committed bucket can never accrue, for *any* policy. It measures the ablation design, not behavior. The real, non-circular commitment signal is **fossilize/ep** (control 0.207 ± 0.006).
- **corr(reward, J) = 0.212 ± 0.046** — weak across all 5 seeds (kept; not magnitude-confounded the way share is). Awaits placebo to certify J above noise.
- **⚠️ CONVERGENCE / BASELINE adjudication (advisor 2026-06-27, control:41) — overturns the strong thesis:**
  - **Controller DOES contribute accuracy.** Per-CF-event baseline gap (all-seeds-on − all-disabled): host-alone **40.0%** → with morphogenesis **46.1%**, **mean gap +6.1% / median +3.5%**; 69% of events add >0.5%, only 8% degrade. So "degenerate / produces no improvement" is **FALSE** — this is an **efficiency** finding (real +6% at +252% params / 7× compute), i.e. the original park-the-freeloader thesis, NOT a do-nothing attractor.
  - **Policy is NOT confidently converged.** Raw op_entropy 0.37 was a *forced-WAIT artifact* (median 0.000; >½ of op-steps single-legal-op). On **unmasked** op-steps op_entropy = **0.93 / 1.61 max ≈ 58%** — substantially stochastic. So much of the churn is genuine policy stochasticity, not a locked metronome.
  - **REWARD_HACKING_SUSPECTED fires 48×** (`ransomware_signature`: +self-LOO `seed_contribution`, −env-joint `total_improvement` — the live PIN-A divergence) but is **rare** (~0.02/episode) — a tail phenomenon, not the main story.
  - **Net:** the strong "reward makes it farm a dense signal while producing nothing" claim is **dead**. What survives: **inefficiency (modest accuracy at high param/compute cost) + churn that the cheap scale-levers (unit-norm, clip2) do not reduce.** Still leans SURVIVE, but for a milder reason than first stated.
- **Caveat:** decile cuts above are **one seed (41)** — extend to ≥5 before hardening. Arm analysis must be **seed-matched paired deltas + paired bootstrap** (advisor), NOT the unpaired n=5-vs-n=4 aggregate. STOP-vs-SURVIVE (GATE −1) still needs all arms at ≥5 paired seeds (runs in flight). No placebo yet ⇒ noise-floor leg of GATE 0 still open.

| GATE-0 leg (≥5-seed control arm) | status |
|---|---|
| per-step reconciliation (`residual`≈0) | ✅ **LOCKED n=5** |
| `share_attribution` computable + stable | ✅ computable, 0.925 ± 0.004 (n=5) — but **magnitude-confounded** (interpret as a magnitude fact, not a behavioral lever) |
| raw J computable on ≥5 seeds | ✅ **LOCKED** n=5 nonzero, stable (mean J/ep 0.072–0.097) |
| noise floor (placebo, per stage) | ⏳ placebo harness pending (the remaining code deliverable) |

> **GATE-0 instrument verdict unchanged** (the instrument is sound: J computes, reconciles, replicates). What changed 2026-06-27 is the *interpretation of the control-arm readouts*: the strong "reward-farming / no-improvement" reading is falsified (controller adds +6% acc at high cost; policy is stochastic not peaked), committed-J retired as tautological, share downgraded as magnitude-confounded. GATE −1 (cheap-fix vs redesign) must use **paired** arm deltas and is still pending the arm runs.

### 5.2 Motif/freeloader probe + the central open question (control:41, n=1 — DO NOT harden)

Motivated by the owner's observation that the controller *can* rediscover good structure (early-conv/late-stabiliser ≈ **CoAtNet**, Google 2021). Two probes on control seed 41:

**Fossilized seeds by position × blueprint family** (per-seed blending Δacc = LOO quality; ensemble CF = joint at fossilization):

| cell | n | own blendΔ (LOO quality) | ensemble CF |
|---|---|---|---|
| **early conv-family** (CoAtNet stem) | **261 (54%)** | **−0.34** | **+13.23** |
| early stabiliser | 132 | +2.85 | +1.34 |
| late conv-family | 25 | +2.80 | +2.69 |
| late stabiliser | 13 | +0.74 | +0.19 |
| mid conv-family | 35 | −0.33 | +15.06 |

**Freeloader scan** (residency, on-path seeds n=2354): freeloaders (survived ≥10 on-path steps, cf-credit ≤0) = **245 (10% of seeds, 14% of survival-time)**, mean survival **38.1 steps** vs productive seeds' 30.3 — i.e. negative-contribution seeds survive *longer*; `corr(survival, quality) = 0.165` (weak). 74% of long-survivors are genuinely productive (mean cf +128). **The defect is a real minority tail, not the dominant mode** (consistent with REWARD_HACKING firing but rare).

**⚠️ THE CENTRAL OPEN QUESTION (this fork decides the whole conclusion):** the controller's dominant fossilization (early conv, 54%) and the freeloader tail both have **low/negative per-seed LOO** but high *ensemble* value — and that pattern is ambiguous between:
- **(a) Freeloader / commitment-avoidance defect** — seeds committed despite not individually helping, wrongly credited by the surrounding ensemble. ⇒ **redesign the reward** (the J/redesign thesis).
- **(b) Foundational stem undervalued by LOO** — an early-conv *stem* (the CoAtNet motif) has low leave-one-out marginal *because it is enabling*; remove it with everything present and other paths compensate, but it **enabled the structure to form**. ⇒ the controller is correct and **J/PIN-A mis-measures it** (a measurement problem, not a controller defect).

**Neither the motif table nor the freeloader scan can disambiguate (a) from (b)** — both define quality via per-seed LOO (PIN A), which is precisely the metric blind to enabling value. **The disambiguator is SYNERGY: does removing the early-conv seed collapse the *late* seeds' contributions?** (yes → (b), real stem, J needs a synergy/Shapley-aware term; no → (a), true freeloader, redesign justified.)

### 5.3 SYNERGY result — the runs logged the FULL factorial ablation (n=5 control, median-led, advisor-vetted)

The CF matrix logs the **complete factorial** for multi-seed episodes: `(4 configs, 2 seeds): 2886/run` (full 2×2) and `(8,3): 138/run` (full 2³). So true *interventional* synergy is computable, no re-run needed. Per 2-seed factorial event: **interaction `I = acc(A,B) − acc(A,¬B) − acc(¬A,B) + acc(¬A,¬B)`** — `>0` ⇒ co-resident seeds COMPLEMENT (enabling); `≈0` ⇒ independent (the pure-freeloader signature — removing a freeloader can't change a co-resident's contribution); `<0` ⇒ substitute/redundant. Reported as the **per-event interaction MEDIAN** (paired within each factorial; robust to single-epoch CF noise), per slot-pair, per seed:

| slot pair | s41 | s42 | s43 | s44 | s45 | **mean ± sd of medians** | total n |
|---|---|---|---|---|---|---|---|
| **r0c0→r0c1** (early→adjacent) | +1.08 | +0.60 | +0.48 | +0.84 | +0.96 | **+0.79 ± 0.22** | 7339 |
| r0c0→r0c2 (early→distal) | +0.24 | +0.12 | +0.00 | +0.12 | +0.12 | +0.12 ± 0.08 | 4796 |
| r0c1→r0c2 (mid→adjacent) | +1.20 | +0.36 | +0.36 | +0.36 | +0.06 | +0.47 ± 0.38 | 808 |

**Finding (n=5, sign-consistent, median-led):** `r0c0→r0c1` synergy is **positive in all 5 seeds** (+0.48…+1.08) — the early seed *enables its neighbor* (~+0.8 acc-pts), which a pure freeloader cannot produce. This is **positive evidence that co-resident structure is complementary, not redundant**, and per-seed LOO (PIN A) is blind to it. Two honest limits: enabling is **local** (early→distal r0c0→r0c2 = +0.12, a whisker off zero — decays with distance), and **modest** (sub-acc-point). Caveats carried: (i) this is **slot-position** synergy, one inferential step short of **conv-blueprint** synergy (blueprint join owed); (ii) **selection-scoped** — measured only on pairs the controller chose to co-resident.

> **⚠️ STRUCTURAL SCOPE (advisor 2026-06-27 — do not conflate):** the factorial mask contains only **ablatable** seeds. A **fossilized** seed is baked into the host (α=1) and lives in the always-on baseline, **NOT in the ablation mask**. So this synergy result is about **co-resident *transient* (TRAINING/BLENDING) seeds — NOT the committed fossils.** It therefore does **not**, by itself, speak to the §5.2 puzzle (the 261 *fossilized* neg-blendΔ early-conv seeds). Earlier wording here ("the structure the controller commits to") was wrong and is corrected: the claim is about co-resident, not committed, structure.

**Per-population split (n=5 control, median-led).** Splitting each factorial pair by the weaker seed's own marginal `m`: `I, m_A, m_B` are mutually orthogonal contrasts of the same 4 cells (so conditioning on marginals cannot algebraically force `I` — the split is real structure):

| group | median I (all 5 seeds) | IQR of I |
|---|---|---|
| **both-help** (m>0) | **+0.84 … +1.32** | **5.04** |
| freeloader-looking (m≤0) | +0.00 (±0.000) | 0.96 |

Among m≤0 seeds, enable-rate (I>0): r0c0 38%, r0c1 49%, r0c2 47%.

**Three claims at their true confidence (advisor-gated):**
1. **SOLID:** complementarity is real and concentrated in **direct-helper** pairs (m>0, median I≈+1, IQR 5.04 = genuine range), n=5 sign-consistent. Co-resident good seeds genuinely complement.
2. **DOWNGRADED (confounded):** transient neg-marginal (m≤0) co-residents show median I=0 — BUT the m≤0 group's IQR is 0.96 vs the helpers' 5.04 (~5× tighter), so much of that "0" is **accuracy-floor range-compression, not independence.** "Weak seeds don't enable" is therefore **not cleanly established** — this dataset can *detect* enabling where it exists but cannot robustly prove its *absence* in the low-accuracy regime.
3. **OPEN — the original §5.2 fork is NOT resolved.** The fossil puzzle concerns *committed* seeds, which are absent from the factorial mask (see scope box). A feasibility check (do will-fossilize slots appear in factorials pre-commitment?) was **inconclusive**: 482 fossilizations span only 36 (env,slot) combos (~13 lifecycles/slot), so (env,slot)-matching conflates lifecycles. **Resolving the fossil fork requires the seed-lifecycle (germination→fossilization) join** — the heavy closer the advisor first pointed at; the join-free proxy answered a *related* question (do transient low-marginal co-residents enable?), not *the* question (do the committed neg-blendΔ fossils enable?).

**⇒ The (a)/(b) fork remains OPEN.** What's banked: real complementarity among co-resident helpers (pro-(b) *mechanism exists*), and a method that detects enabling where present. The two-term-credit redesign brief (synergy term + freeloader penalty) is a **hypothesis**, not a conclusion — it rests on the proxy population. **QUEUED (run fresh, not at session end): seed-lifecycle-conditioned synergy on the will-fossilize / fossilized-conv@r0c0 cohort.**

> **Closer attempt 2026-06-28 FAILED (data-model blocker, not interpreted):** the first lifecycle-join (`scratchpad/phase0runs/synergy_lifecycle.py`) is buggy and its output must NOT be trusted. `(env,slot)` is not a single-occupancy key — the controller germinates **multiple blueprint candidates in the same slot at the same episode** (morphogenesis selection), and structured germ/foss/prune events carry **no stable `seed_id`** (only in human log text) while `morph_id` keys the *action*, not the seed. So germination and fossilization of one seed share no structured key, and naive interval reconstruction yields ~23k `UNKNOWN(G→G)` intervals with the target `FOSS/conv/neg` cohort empty. The correct closer needs **seed-identity reconstruction** (isolate the single *persisting* seed per slot — `end_ep ≫ germ_ep`, excluding immediate-prune candidates — or recover `seed_id` from the residency/active stream), and must **validate the fate distribution before interpreting any synergy.** Likely warrants the morphogenesis schema / `drl-expert`.

> Bank (error pattern, bit 4× on 2026-06-27): n=1 synergy on 3 hand tables read *negative* → systematic n=5 reversed to positive; then a population-level split nearly flipped the *puzzle* verdict to (a) on a proxy population (transient ≠ committed). **No directional read off <30 hand-picked samples; and verify the population you measured is the population the question is about** before any verdict.

---

## 6. GATE 0 (instrument-first, fail-closed) — status

All must hold; **currently PROVISIONAL** (reconciliation machinery validated on logged data; residency telemetry + ≥5-seed runs pending):
- [~] Raw J computable end-to-end on ≥5 seeds of the control arm (target 10). `off_switch` subtraction NOT required (Phase 1). — **machinery VALIDATED end-to-end** (emit → production view → offline `J = SUM(j_per_param) GROUP BY` all proven on a real smoke run); the ≥5-seed runs produce the actual J *values* (the smoke's J=0 reflects no blending in 25 untrained epochs, not a plumbing gap).
- [x] **Stage-0 telemetry reconciles** — `residual ≈ 0`, `share_residual ≈ 0` (NOT the tautological "shares sum to 1", §4): **VALIDATED on 45,696 logged SHAPED rows** (max\|residual\|=0.002, share_residual=0.0, §5). *Re-confirm on the ≥5-seed run after wiring.*
- [ ] Counterfactual noise floor reported (inert-seed placebo, non-degenerate per §2). — *pending placebo harness + runs.*

**Verdict: PROVISIONAL — the reconciliation leg is validated on logged data; the J-computability and noise-floor legs require the residency telemetry + ≥5-seed runs.**

**Verification owed before the residency sweep (advisor flag):** confirm `baseline_accs` is *computed* every epoch on the control config (not a periodic ablation cadence and not merely *reset* every epoch — `vectorized_trainer.py:1122` says "rebuilt fresh each epoch"; confirm that = computed). The integrand samples per-epoch LOO; an every-K-epoch ablation cadence would silently under-sample residency and bias J.

---

## 7. Frozen reference control arm (deliverable 8)

Reuse the Phase −1 frozen arm (shared so one run-set serves GATE −1 and GATE 0):
- **Config:** `configs/config-3slot-3seed-baseline-shaped.json`, sha256 `7f199a3ac1b3` (12-hex), `reward_mode=shaped`, `max_seeds=3`, `hra_value_decomposition` **unset → OFF**.
- **Host:** capable (`cifar_baseline`), 12 vec-envs, ~180–300 episodes.
- **Paired seeds:** `41 42 43 44 45` (floor 5; extend `46–50` for target 10) — Phase −1's seed set.
- **HRA-OFF consequence (critical):** the existing per-stream EV + Stage-0 GATE metrics (`cov_rcf_return_share`, `r_main_cov`) are gated inside `if self.hra_value_decomposition:` (`ppo_agent.py:744`), so they emit **nothing** on this arm. The new Phase-0 per-term decomposition therefore **must not** be HRA-gated — it must populate on the HRA-OFF control arm or GATE 0 is unevaluable.

---

## 8. Run protocol & sequencing

- **Land telemetry first**, then run **≥5 control-arm seeds**. Because the Phase −1 A/B is **PENDING** (its packet, dated today, says so), a single control-arm run-set serves **both** GATE −1 (Phase −1) and GATE 0 (Phase 0) — same config, same seeds.
- Launch: `uv run python -m esper.scripts.train ppo --config configs/config-3slot-3seed-baseline-shaped.json --seed <s> --max-seeds 3 --device cuda:0 --gpu-preload`.
- The ≥5-seed launch is the one genuine compute decision → surfaced to the user at the launch point. GATE 0 may be reported PROVISIONAL on a seed subset.

### 8.1 Control-arm run launch (provenance)
- **Launched 2026-06-25** against commit `406aeb25` (this Phase-0 work), control arm `config-3slot-3seed-baseline-shaped.json` (sha256 `7f199a3ac1b3`, `max_seeds=3`, HRA-OFF, 12 vec-envs, n_episodes=200), `--gpu-preload`.
- **Seeds 41–45** split across 2× RTX 4060 Ti: cuda:0 = {41, 43, 45}, cuda:1 = {42, 44}. Telemetry → `telemetry/` (Karn-discoverable).
- **GATE-0 scoring queries to run on completion** (per run_dir): (1) reconciliation `residual` stats — now with `occupancy_rent`/`fossilized_rent` nonzero (re-confirms the −1 sign on real multi-seed data); (2) `share_attribution = Cov(bounded_attribution, total_reward)/Var(total_reward)` per seed + aggregate; (3) `J = SUM(j_per_param) GROUP BY run_dir, episode_idx` from the `seed_residency` view; (4) committed/uncommitted split + None-fraction (`n_none_steps/n_on_path_steps`). **Noise-floor leg** still needs the inert-seed placebo harness (not in these runs) — remains PROVISIONAL.

---

## 9. Sign-off ledger

| reviewer | scope | status |
|---|---|---|
| **OWNER** | **PIN A** (per-seed LOO vs literal env-joint) | **RESOLVED 2026-06-25** — owner deferred to the coherence argument (env-joint self-contradicts §8 under Σ_seed); per-seed LOO stands; literal HARD CONSTRAINT #4 formula = spec bug to correct when the doc lands. |
| drl-expert | J definition; **PIN A** (per-seed vs env-joint cf); λ/baseline; additend classification | **ENDORSE_WITH_CHANGES** — PIN A "the only coherent choice"; changes incorporated (§4 tautology→residual criterion; PIN E degeneracy; PIN B citation + snapshot; proxy-rationale wording; GATE-0-asserts-no-efficacy; None-fraction; Shapley cross-check) |
| yzmir-deep-rl:reward-function-reviewer | same (reward-design-critical) | **ENDORSE_WITH_CHANGES** — PIN A decisive; changes incorporated (signed-contribution map; mandatory committed/uncommitted split + residency-length sensitivity; survival-time §5.1; residual≡pending_auto_prune precondition; units verified) |
| pytorch-expert | residency hot-path hook + variance-share carry | **APPROVE_WITH_CHANGES** — caught 2 correctness blockers: (1) residency sweep MUST read PRE-action state → site is the epoch loop after `_run_fused_val_pass`, before `_run_action_transaction`, NOT `execute_actions` post-reward (else committed/uncommitted split corrupts when a seed fossilizes the same epoch); (2) do NOT ungate the `ppo_agent.py:744-778` block for variance shares — it operates on per-stream *returns* (None on HRA-OFF arm) and would crash the control arm. Variance shares → compute OFFLINE from Karn (both arms produce rewards). Perf trivial (slot.alpha/active_seed_params are CPU scalars, no sync). |
| axiom-python-engineering | leyline contracts + telemetry chain, no defensive drift | **ENDORSE_WITH_CHANGES** — residency needs the FULL 5-link chain (hook → emit per-seed at EPISODE_OUTCOME → leyline `SeedResidencyTelemetry` to_dict/from_dict serializing `j()`+uncommitted as columns → `AnalyticsSnapshotPayload` carrier → Karn `seed_residency` view) — don't stop at the dataclass (the scar). `compute_variance_shares` dict→typed if emitted live (offline keeps it a function). `getattr(components,term)` no-default is legitimate. Confirmed rewards-view rent gap (FIXED). Do NOT extend the lossy `karn/store.py:237 RewardComponents`. |

---

## 11. Implementation status (landed this session)

**Landed — pure computation cores (fully unit-tested offline, no GPU runs needed):**
- `src/esper/simic/rewards/partition.py` — `ADDITEND_SIGN_MAP` + `decompose_additends(reward_raw, components)`: the signed additend decomposition with the reviewer-mandated sign map + `residual`. Tests: `tests/simic/rewards/test_additend_decomposition.py` (6).
- `src/esper/simic/telemetry/reward_variance.py` — `compute_variance_shares(steps)`: per-term `Cov(R_i,R)/Var(R)` (population), CoV, top-line `share_attribution`, and the **corrected** `residual_share` reconciliation diagnostic (proven to flag a hidden term while `shares_sum` stays 1). Tests: `tests/simic/telemetry/test_reward_variance.py` (5).
- `src/esper/simic/rewards/residency.py` — `SeedResidencyAccumulator` + `compute_residency_j`: the J integrand (PIN A per-seed LOO, on-output-path gate, None-fallback), the freeloader guard (test d), the §1.6 worked example (test b), the committed/uncommitted split, and the raw-residency diagnostic. Tests: `tests/simic/rewards/test_residency_j.py` (8).
- **19 new tests pass; 168 reward+telemetry tests pass (no regressions); `wardline` gate clean (exit 0).**

**Landed this session (additionally):**
- **Residency sweep core** `accumulate_residency_for_env` + `SlotResidencySample` (`rewards/residency.py`) — per-env all-slots PIN-A sweep with None-gap handling. Tests: `tests/simic/rewards/test_residency_sweep.py` (6).
- **Karn `rewards` view fix** (`karn/mcp/views.py`) — added `occupancy_rent`/`fossilized_rent` columns (the confirmed gap), enabling multi-seed offline additend reconciliation.
- **Residency 5-link telemetry chain WIRED** (per the pytorch-expert + axiom-python vetting): (1) leyline `SeedResidencyTelemetry` contract (`telemetry_contracts.py`, to_dict/from_dict, `j_per_param` + committed/uncommitted as columns); (2) `AnalyticsSnapshotPayload.seed_residency`/`seed_id` fields + to_dict/from_dict + `_parse_seed_residency` (`telemetry.py`); (3) `EnvState.residency_accumulators` field + `reset_episode_state.clear()` (`parallel_env_state.py`); (4) **pre-action hook** `_accumulate_residency` in `vectorized_trainer._run_epoch` (after `baseline_accs`, before action exec); (5) **per-seed emit** at the episode-end `EPISODE_OUTCOME` block (`action_execution.py`, `kind='seed_residency'`); (6) Karn `seed_residency` view (`views.py`) → offline `J = SUM(j_per_param) GROUP BY run_dir, episode_idx`. Contract round-trip tested (`tests/simic/telemetry/test_seed_residency_telemetry.py`, 4). **29 Phase-0 unit tests pass; 300-test reward+telemetry regression clean; `wardline` gate clean (0 active).**
- **SMOKE-VALIDATED end-to-end (closes the half-wired scar):** a 1-episode `max_seeds=2` CPU run emitted real `seed_residency` events; the **production** `VIEW_DEFINITIONS['seed_residency']` SQL projected all 5 seed rows; and offline `J = SUM(j_per_param) GROUP BY run_dir, episode_idx` computed. J values were 0 because no seed reached BLENDING in 25 untrained-policy epochs (structural, not a bug — one seed had real `params=432` but `n_on_path_steps=0`); nonzero J requires the ≥5-seed scoring runs where the policy actually blends seeds. The **plumbing is proven on real data**.

**Pending — live wiring + runs (NOT landed; GATE 0 PROVISIONAL) — plan now vetted by pytorch-expert + axiom-python:**
- **Residency 5-link chain** (axiom-python) — exact sites scoped, accessors verified (`slot.state.seed_id` str, `slot.state.stage.value` int, `slot.alpha` float, `slot.active_seed_params` int — all CPU, no sync):
  1. **hook** — call `accumulate_residency_for_env` in `vectorized_trainer._run_epoch` **immediately after `baseline_accs = fused_result.baseline_accs` (line ~2485), before `_build_action_inputs`** (PRE-action; `env_states`, `baseline_accs`, `self.slots`, `env_state.val_acc` all in scope). Build one `SlotResidencySample` per slot from the live model with a `slot.state is None` guard → `seed_id=None` for dormant.
  2. `EnvState.residency_accumulators: dict[str, SeedResidencyAccumulator]` field + `.clear()` in `reset_episode_state` (`parallel_env_state.py:211`).
  3. leyline `SeedResidencyTelemetry` contract in `telemetry_contracts.py` (to_dict/from_dict; serialize `j()` + `cf_weighted_integral_uncommitted` as columns, mirroring `shaped_reward_ratio`).
  4. **emit** — at the episode-end branch in `action_execution.py:1592` (where `EPISODE_OUTCOME` already fires, `env_state` in scope), emit one ANALYTICS_SNAPSHOT per seed `kind='seed_residency'` carrying `SeedResidencyTelemetry` (extend `AnalyticsSnapshotPayload` `telemetry.py:1725` — mirror the `reward_components`/`kind='last_action'` path at `:1743,:1789`; do NOT extend the lossy `karn/store.py:237 RewardComponents`).
  5. Karn `seed_residency` view (`karn/mcp/views.py`, mirror the `rewards` view) keyed `(run_dir, episode_idx, seed_id)` so offline `J = SUM(j) GROUP BY run_dir, episode_idx`.
  Then a 1-episode `max_seeds=2` smoke (per Phase −1) to confirm `seed_residency` rows appear with plausible `j` — rows-in-the-view closes the half-wired scar; unit tests alone do not.
- **Variance shares: compute OFFLINE from Karn** (pytorch-expert blocker 2 — NOT a buffer carry, NOT ungating `ppo_agent`). The machinery is `compute_variance_shares`; the live demo (`share_attribution=0.82`) already ran via SQL. *One discriminator to confirm:* whether GATE 1 needs the share over the exact PPO `valid_mask` population (→ then a live carry of decomposed signed-additend float arrays) or just finalized SHAPED steps (→ offline suffices).
- Host-drift placebo harness (near-inert *small-real* seed per §2 non-degeneracy) as a tested negative control.
- ≥5 control-arm seeds → populate §5 readout + the real-logged-episode worked example → score GATE 0.

## 10. Branch decision (flagged to owner)

The methodology doc lives only on `codex/sanctum-pre-ready-crash`. Phase 0 must land where the doc AND the Phase −1 / Stage-2 code coexist. **Recommendation:** land Phase 0 on the current `feat/phase-minus1-scale-falsifier` (0.3.0-based — already carries Stage-2 HRA + Phase −1 flags), and `git show codex/...:<doc> > <path>` the methodology doc onto it in the same series. Flagged per the prompt; proceeding on this branch unless the owner redirects.
