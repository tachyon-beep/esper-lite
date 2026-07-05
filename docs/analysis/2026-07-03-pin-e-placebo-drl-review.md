# PIN-E Placebo Noise-Floor Harness — DRL / Measurement-Methodology Review

**Reviewer:** drl-expert (measurement-design scope)
**Plan under review:** `docs/plans/ready/2026-07-03-pin-e-placebo-harness.md`
**Method spec:** `docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md` §2 (PIN E)
**Consumer:** `docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md` (tau deadbands `gap = max(0, φ − c_paid − tau)`)
**Enablement gate:** `esper-lite-f22a1d48a7` (criteria 1–7)
**Date:** 2026-07-03

## VERDICT: APPROVED_WITH_CHANGES

The inertness engineering (single-layer residual δ, `seed_lr=0`, G2-passage argument), the No-Legacy declared-schedule generalization, the Option-B blueprint delivery, and the non-degeneracy scaffolding are sound and I endorse them. **But the tau-calibration measurement as designed is pointed at the wrong estimand.** The run is single-seed (k=1); tau deadbands `φ − c_paid`; and at k=1 the Shapley value equals the standalone marginal (`φ(s) = v({s}) − v(∅) = c_paid(s)`), so `φ − c_paid ≡ 0` **by construction**. The planned run therefore measures the spread of the *standalone LOO marginal*, which is a **different statistic** from the *synergy contrast* the deadband gates — and, worse, it measures it in the one regime (k=1) where the committed-Shapley term is *structurally inert* and never fires. This is fixable without abandoning the harness: run k≥2 co-resident placebos and assemble `φ − c_paid` offline from the full factorial that `on_counterfactual_matrix` already logs at scale=0. Details in REQUIRED-1.

Confidence in the verdict: **high**. Risk of the verdict being wrong: **low** — the k=1 ⇒ `φ = c_paid` identity is algebraic (Shapley definition + the design-doc amendment defining `c_paid = v({s}) − v(∅)`), and I confirmed against `vectorized_trainer.py:1147–1203` that the committed-Shapley masks the coalition exactly this way and that the CF-matrix factorial is logged independently of the (gated-off) block.

---

## REQUIRED CHANGES

### R1 — Calibrate tau on `φ − c_paid` at the operating k (≥2), not on the single-seed LOO spread
**What.** Replace the single-seed (k=1) placebo run with a **k≥2 co-resident near-inert placebo run** (k=3 preferred — matches the causal config's `max_seeds=3` and the term's exact-factorial ceiling). All co-resident placebos held at HOLDING α=1. The analysis assembles the committed-Shapley analog φ(s) and `c_paid(s) = v({s}) − v(∅)` **offline** from the logged `COUNTERFACTUAL_MATRIX` factorial (`solo_accs`, `pair_accs`, `baseline_accs`, `val_acc`, `all_disabled` — for k co-resident seeds this is the full 2^k family, confirmed at `vectorized_trainer.py:1626` and phase-0 §5.3 "(8,3): 138/run"). Report the terminal-window empirical distribution of `(φ − c_paid)` for a placebo (null) player; **tau = an upper-tail quantile of that distribution.**

**Why.** tau deadbands `φ − c_paid` (design-doc gap formula). At k=1, φ(s) = w(0,1)·(v({s}) − v(∅)) = v({s}) − v(∅) = c_paid(s), so `(φ − c_paid) ≡ 0`. A single-seed run has **no coalition, hence no synergy contrast to measure** — its `seed_contribution` spread (`v(C) − v(∅)` at C={s}) is the spread of `c_paid`, not of `(φ − c_paid)`. These are different statistics: under an idealized independent-noise model, std(φ − c_paid) ≈ 0.78·std(single marginal) at k=3, but that factor is model-dependent and unmeasurable at k=1. Independently, the term itself pays `max(0, φ − c_paid − tau) = max(0, −tau) = 0` at k=1 — so the entire single-seed regime is one in which the term never fires; you cannot calibrate a deadband from a regime the deadband never operates in. The correct estimator is directly measurable at scale=0 because the factorial telemetry is decoupled from the gated block.

**Sub-requirements (how to report the number defensibly):**
- **(a) Signed upper tail, not |·|.** The gap is `max(0, φ − c_paid − tau)` — only the **positive** tail of `(φ − c_paid)` leaks credit. Report the quantile of the *signed* `(φ − c_paid)`, not `|LOO|`. (For a symmetric null distribution, P95 of `|·|` ≈ P97.5 of the signed upper tail — conservative but conflates the tails; state which you use.)
- **(b) Single true-terminal-epoch sample as primary.** The term fires once, at `epoch == max_epochs`, on a (near-)converged host. Draw one `(φ − c_paid)` sample per episode at the true terminal epoch (~360/seed, one per independent episode) as the **primary** tau population. The "last-20-epochs α=1 window" mixes in pre-convergence host-drift variance and is heavily autocorrelated; keep it only as a secondary, detrended robustness readout.
- **(c) Recommend P99 with a stated false-positive budget.** Because the term is terminal-only (one φ per committed seed per episode), P95 lets ~5% of null committed seeds leak positive synergy credit. Report the full ladder (P50/P90/P95/P99/max) and recommend tau = P99 unless the cap+G-clamp are argued to absorb the P95 leakage; state the per-seed false-positive rate the chosen quantile implies.
- **(d) Block-bootstrap CI (block = episode).** The 133 HOLDING epochs/episode are near-identical at a converged terminal (effective N per episode ≈ 1, not 20). Do not assert stability from nominal sample count; report a block-bootstrap CI on the chosen quantile. Run-shape sufficiency (Q6) is **demonstrated by CI width vs tau**, not by counting correlated samples.

**Where.** §1 Objective; §2.1 ("the placebo is the only seed ⇒ its per-epoch LOO ≡ v(C)−v(∅) exactly" — this is exactly the property that makes it uncalibratable for tau); §2.4 (schedule must germinate ≥2 slots — grows WI-3); §2.6 outputs 2–3; §5 Q2; §6 AC4.

Confidence: **high**. Risk: **medium** on cost — a k≥2/k=3 declared schedule is a larger WI-3 lift than the single-slot schedule, and getting 3 placebos to co-reside and all reach HOLDING under a forced schedule needs a 3-slot skeleton. This is real added scope, but it is the only way to observe the synergy-contrast noise at scale=0.

### R2 — Split the two conflated deliverables
**What.** State explicitly that this harness produces **two distinct numbers with two distinct estimators**: (D1) the phase-0 GATE-0 per-stage noise-floor line = mean/std/CoV of the per-step LOO `seed_contribution` (the standalone marginal), for which a single-seed reading is *correct and sufficient*; and (D2) tau for the committed-Shapley term = the `(φ − c_paid)` upper quantile at k≥2 (R1). Do not let one measured number serve both roles.

**Why.** §1/§2.6 currently derive tau from the same terminal LOO used for the GATE-0 line. D1 legitimately wants the k=1 clean marginal; D2 legitimately needs k≥2. Conflating them is what produced the k=1 tau error. A k≥2 run yields D1 for free (any single marginal is readable from the factorial), so the split costs nothing analytically — it just forces the report to name which estimator feeds which consumer.

**Where.** §1 Objective; §2.6 outputs 1 vs 3; §6 AC4.

Confidence: **high**. Risk: **low**.

### R3 — Epsilon ladder must probe DOWNWARD toward the δ→0 noise plateau, with a δ-insensitivity acceptance test
**What.** Replace the single upward arm (`std=1e-2`) with a ladder that brackets the small-δ plateau, e.g. `{1e-4, 1e-3}` (optionally a `5e-5` arm to locate the degeneracy floor). Acceptance test, stated in the report: tau is **δ-insensitive** across the small-δ arms (flat within the bootstrap CI) ⇒ the measured spread is genuine δ-independent estimator noise ⇒ tau is well-defined. If tau scales monotonically with δ, the "floor" is tracking the placebo's own (small) contribution ⇒ signal-contaminated ⇒ tau is δ-anchored and not a noise floor.

**Why.** tau must be the estimator-noise floor of a *null* player — the δ→0⁺ limit that excludes the seed's own contribution. A `1e-2` arm moves the wrong way: it adds *more* real contribution (signal), so its `(φ − c_paid)` reflects the placebo's actual synergy, not noise. Two points (1e-3 baseline + 1e-2) also cannot certify δ-independence — you need ≥2 points *in the small-δ regime* to see a plateau, plus awareness that too-small δ hits the bit-identical quantization-zero degeneracy (the `noop` failure mode). The ladder maps both walls: signal-contamination above, degeneracy below; tau = the plateau between them.

**Where.** §2.2 (init std choice); §2.6 output 5; §5 Q3; risk_notes (2).

Confidence: **medium-high**. Risk: **low** — worst case the ladder confirms the floor is already flat by 1e-3 and the extra arms were cheap insurance.

### R4 — WI-5 must record the placebo's single-tensor grad-norm and gradient_health at every G2-relevant epoch, not just assert the boolean
**What.** Strengthen WI-5 so it logs (and the go/no-go asserts a *margin* on) the actual per-tensor grad L2-norm and the computed `gradient_health` at each BLENDING dwell epoch (12–17) and at the advance@17 gate — proving health is a stable 1.0 with comfortable margin above the 1e-7 vanishing threshold across the whole dwell, not at a single sampled epoch.

**Why.** The placebo has exactly **one** weight tensor. A per-tensor health metric with a hard 1e-7 threshold is a **binary step function** at k=1 tensor (health ∈ {0, 1}) — there is no averaging to smooth a transient dip. With δ~1e-3 init, the single tensor's grad-norm is O(loss-grad × host-activation × δ-path), which can be small; if it ever dips below 1e-7 (late training with tiny loss gradients, or a batch with near-zero host activations into that slot) health flips to 0 and G2 could block or flag the seed unhealthy mid-dwell. The analytic claim "health ≈ 1.0" is plausible but must be *shown numerically with margin*, because a single-tensor metric has no fault tolerance. This is the cheapest possible failure to catch before GPU time.

**Where.** §2.2 (inertness mechanism, health claim); WI-5; §6 AC2.

Confidence: **medium-high** (that the metric is binary/fragile). Risk: **low** — WI-5 already exists; this just tightens what it must observe.

### R5 — Document that placebo-derived tau is a LOWER BOUND on the operating (null-among-reals) noise, and mark it conservative-provisional
**What.** The report must state, and the enablement gate must record, that tau measured on placebos-only characterizes the noise floor of a **null player inside a homogeneous-null coalition** on a **near-identity-trained host** — which is a *lower bound* on the noise the term meets at enablement, where the committed set contains **real, high-variance co-residents** on a **co-adapted host**. Recommend: set tau with an explicit safety margin above the measured floor, and treat it as the *provisional* value that the first ON calibration run (enablement criterion 6) finalizes against the real `(φ − c_paid)` distribution.

**Why.** Three regime gaps all bias the placebo floor **downward** relative to operating conditions: (1) real co-residents swing val_acc by acc-points epoch-to-epoch, and masking one shifts the fused pass substantially, inflating the marginal-difference variance that composes φ — placebos (δ~1e-3) produce near-identical on/off configs, so their marginal-difference variance is structurally smaller; (2) the placebo host trains as if the slot were identity, so masking removes ~nothing, whereas the real host is co-adapted to load-bearing committed seeds; (3) the committed-Shapley φ fires once on a converged host, while the placebo samples a slowly-drifting one. A too-small tau ⇒ too-narrow deadband ⇒ **over-credits** (the dangerous direction — leaks synergy credit to freeloaders). The clean remedy for the operating-range gap is phase-0 §2's option (a) (K-resampled variance on a *real* seed), explicitly deferred to Phase 1 — so the honest GATE-0 posture is "conservative lower-bound tau, finalized at first ON run," not "absolute tau."

**Where.** §2.6 output 3 (tau recommendation must carry this caveat); §5 Q5; the memo posted to `esper-lite-f22a1d48a7` (criterion 1); ties to criterion 6(a).

Confidence: **medium** (directional argument, not quantified). Risk: **low** — this is a documentation/interpretation requirement, cost-free, and it prevents the gate from over-trusting the number.

---

## RECOMMENDED (optional) CHANGES

### Rec1 — Prefer, or add, a "1 forced placebo + free-controller co-residents" arm to measure the null-among-REALS floor
Running the placebo forced while the controller germinates/blends *real* seeds around it lets you assemble φ_placebo (forced-null by construction) inside a **real** coalition — directly measuring the operating-condition floor that R5 flags as unmeasured. This is **not** the circular offline-proxy the phase-0 doc rejects (§2): you are not *selecting* low-contribution seeds, you are *forcing* a known-null one and reading its φ against whatever reals appear. Caveat: messier provenance, and the placebo may not co-reside with committed seeds at the terminal. Value: it is the closest scale=0 proxy to the enabled regime. Confidence the arm is informative: **medium-high**; risk it is hard to orchestrate cleanly: **medium**.

### Rec2 — Add a real-small-seed positive control to certify "same noise sources"
Phase-0 §2's non-degeneracy clause requires the placebo spread be "driven by the same noise sources that move a real seed's c_t." The plan's three checks (nonzero-fraction, spread-spans-grid-steps, delta-scale) prove the spread is *non-vacuous* but not that it *matches a real seed's noise character*. A cheap certification: reuse the top epsilon-ladder arm (a larger-δ, still-near-null seed) or run one real seed with `zero_init_final` + lr>0 and compare its *early* (still-near-null) terminal φ spread to the placebo's. Alternatively, argue it structurally: with full-testset eval (no minibatch resampling — `vectorized_trainer.py:1355–1381`), the *only* noise sources are host-drift + cuDNN/reduction-order + quantization, which are shared by construction between placebo and real seeds. The structural argument is the cheap minimum; the positive control is the stronger evidence. Confidence: **medium**; risk: **low**.

### Rec3 — Verify the alpha-override (HOLDING) vs tensor-override (FOSSIL) masking are numerically equivalent for a null player
The offline φ is assembled by masking HOLDING seeds (alpha-override); the real term masks FOSSILIZED seeds (tensor-override, enablement criterion 6(c); GATE hazard at `vectorized_trainer.py:1165–1173`). For a δ~1e-3 null player, on-at-α=1 vs off-at-α=0 produces near-identical logits either way, so the path difference is second-order on an already-tiny effect — I judge the offline-φ approach valid to within the same second-order uncertainty criterion 6(c) already tracks. Worth a one-line numerical check (mask a HOLDING α=1 slot to 0 vs mask a FOSSILIZED α=1 slot to 0 on the same batch; assert acc delta < quantization). This straddles into pytorch-expert's kernel-identity scope. Confidence: **medium**; risk: **low**.

### Rec4 — Assert `lr == 0.0` per-epoch (catch a scheduler), and keep the fixed-δ placebo (reject the drifting 1e-6 variant)
Endorse the plan's `seed_lr=0` choice over tiny-lr (Q4): with SGD, `update = lr·momentum_buffer = 0` regardless of the (harmlessly accumulating) buffer, and coupled weight_decay + grad-clip are all lr-gated to zero, so params are bit-identical — while `requires_grad=True` keeps the backward pass alive for G2's health measurement. Crucially a **fixed** δ is *methodologically preferable* to a drifting one: a fixed-weights placebo whose output δ(x_t) still fluctuates only because the *host* inputs x_t drift gives a stationary-weights, host-drift-driven null contribution — exactly the "same noise sources" model phase-0 §2 wants, with **no learning trend** to detrend out of the terminal window. Harden WI-4 to assert the effective per-slot optimizer lr == 0.0 *at every epoch* (not just at construction), catching any LR scheduler that could override it mid-run. Confidence: **high** (lr=0 is a clean no-op); risk: **low**.

---

## Direct answers to the enumerated questions

**Q1 (enum member vs schedule-scoped override) — RL-side.** Endorse **Option B**. Option A changes `BLUEPRINT_IDS` cardinality ⇒ the blueprint action-head output dimension changes ⇒ policy-network final-layer shape and init-RNG consumption change for **every future run** — which contaminates the byte-comparability of the frozen reference control arm (phase-0 deliverable 8) that the whole GATE-0/GATE-−1 program depends on. Never mutate the global action space for a measurement fixture. Option B's forced-action-head/germinated-module mismatch is **RL-harmless in a fully-forced run**: the policy is a passenger, its actions are overridden by the declared schedule, and its learning is discarded — we read `seed_contribution`/CF-matrix telemetry (computed from live slot state, not the action trace) and the germination telemetry records the true `blueprint_id="placebo"`. One check: confirm no reward/telemetry consumer keys off the action-head index in a way that would corrupt the seed_contribution read (it shouldn't — seed_contribution is an ablation over slot state). Confidence: **high**.

**Q2 (tau statistic).** P95(|LOO|) is wrong on two axes: wrong *estimator* (single marginal, not `φ − c_paid`; see R1) and wrong *tail* (|·| conflates the tails; only the signed upper tail leaks credit; see R1a). k·std is inappropriate (the distribution is quantized and non-Gaussian by the plan's own admission — don't assume a scale family). max|LOO| is unstable (grows with sample size, dominated by the worst autocorrelated epoch). Correct: **upper-tail empirical quantile (P99 recommended) of the signed `(φ − c_paid)` at k≥2, single-terminal-epoch, with a block-bootstrap CI.** And yes — the coalition-difference correction is not a scalar factor to bolt on; it is the reason to measure `(φ − c_paid)` directly rather than inflate a single-marginal spread by a guessed constant.

**Q3 (epsilon design).** One arm is insufficient, and 1e-2 is the wrong direction. Use a downward ladder with a δ-insensitivity plateau test (R3).

**Q4 (seed_lr=0 vs tiny).** Exactly-zero is a clean no-op and is *preferable* — fixed-δ gives a stationary null contribution with no learning trend. Keep lr=0; assert it per-epoch (Rec4).

**Q5 (host-regime confound of α=1 for ~133 epochs).** The δ~1e-3 forward perturbation is negligible for host training, so the host trains as near-identity — which is precisely the confound: the placebo characterizes noise on a host that never co-adapted to a real seed, biasing tau *downward* vs the operating regime (R5). The α=1 amplitude itself matches the fossil regime; the gap is the *coalition composition* and *host co-adaptation*, not the blend amplitude.

**Q6 (run-shape sufficiency).** Nominal sample count (~7200 terminal-window samples/seed) is ample but misleading — at a converged terminal the last-20-epochs samples are near-identical (effective N/episode ≈ 1). Bind sufficiency to the **block-bootstrap CI width on the chosen quantile** (block = episode), not the raw count, and prefer the single-terminal-epoch-per-episode population (~360 independent draws/seed × 3 seeds). If the CI is wide relative to tau, add seeds — 3 may or may not suffice depending on the (unknown pre-run) `(φ − c_paid)` spread and the testset-quantization grid.

**Q(b) — the coalition-difference correction, concretely.** φ(s) = Σ_{S⊆C\{s}} w(|S|,k)·[v(S∪{s}) − v(S)], a convex-weighted (Σw=1) combination of marginals, each a difference of fused-val accs sharing the terminal pass. `(φ − c_paid)` is a **zero-sum contrast** of those marginals (weights sum to 0), i.e. literally "average-coalition marginal minus standalone marginal" = the synergy. Its noise is *not* recoverable from a single marginal's spread by a fixed factor — it depends on the cross-config correlation structure of a near-null seed against **real** coalitions, which the shared fused pass makes neither independent nor perfectly correlated. **Report:** the terminal `(φ − c_paid)` distribution assembled offline from the k≥2 factorial (mean=bias, upper quantiles), with the null-among-reals caveat (R5). That is what lets the gate set tau defensibly, instead of applying an unjustified inflation constant to the wrong statistic.

**Q(c) — HOLDING α=1 as the terminal-condition proxy.** Valid on blend *amplitude* (α=1 matches the fossil), but three named gaps, all biasing tau downward: (1) HOLDING alpha-override masking vs FOSSIL tensor-override masking (Rec3 / criterion 6c); (2) near-identity-trained host vs co-adapted host (R5); (3) 133-epoch drifting-host window vs single converged-terminal fire (R1b). The proxy is usable as a conservative lower bound, not as the operating floor.

**Q(d) — non-degeneracy sufficiency.** The three checks are necessary but do **not** discharge phase-0 §2's "same noise sources" clause — they prove non-vacuity, not source-matching. Add the structural argument (full-testset eval ⇒ only host-drift+cuDNN+quantization, shared by construction) as the minimum, and the real-small-seed positive control (Rec2) as the stronger certification.

**Q(e) — inertness RL failure modes.** Single-tensor gradient_health is binary/fragile (R4). lr=0 neutralizes momentum-buffer application, coupled weight_decay, and grad-clip (Rec4) — verify no *non-optimizer* weight decay shrinks δ over time (would drift toward degeneracy; pytorch-side). Alpha-ramp interaction: at BLENDING α=0.25 the effective δ is ~2.5e-4 ⇒ the BLENDING-stage floor will likely be quantization-limited/degenerate — expected and fine, since tau uses the HOLDING/terminal window, but report the BLENDING line as quantization-limited rather than deriving anything from it. Grad-clip on the single tiny tensor is a no-op (harmless).

---

## Endorsed as-is
- Inertness mechanism: single depthwise/1×1 conv, `normal_(std=1e-3)`, no post-normalizer, `SIGMOID_ADD` (ADD blend so `zero_init_final` doesn't fire) — correct; avoids the GroupNorm-renormalization and the requires_grad/zero-init G2-death traps the scouts found.
- Declared-schedule registry generalization with per-schedule hash guard, deleting the hardcoded dispatch (No-Legacy) — correct; note WI-3 must now carry a k≥2 schedule (R1).
- WI-5 as a hard go/no-go before GPU time — correct posture; strengthen per R4.
- `shapley_synergy_scale` stays 0.0 throughout; offline φ assembly from logged telemetry is the *only* way to read the estimator at scale=0 — correct and, per R1, sufficient.
