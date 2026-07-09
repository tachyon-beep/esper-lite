# Stage-2 HRA — MAJOR-1 Acceptance Gate (pre-registered)

- **Status:** `FORMULAS FROZEN 2026-07-08 (owner-ratified, drl-expert-reviewed) — OFF wave launched; OFF-derived scalars resolve at §10 step 3; ON launch remains owner-gated.`
- **Epic / task:** `esper-lite-f25b71c165` (EV-stabilization) → `esper-lite-2a4b56e719` (Stage 2 — de-shape the regressand via HRA value head).
- **Origin:** review-gate disposition PDR-0029 (MAJOR-1 `ev_sum` hard floor + MAJOR-3 provenance folded into Stage-2 acceptance); Stage-0 gate PDR-0032/0033; value-free-gate methodology PDR-0028.
- **Scope fence:** this gate accepts/rejects Stage-2 as a **critic / value-target decomposition intervention (actor-objective A, below)**, NOT as a reward-redesign verdict. `share_attribution` and the Stage-0 variance gate are *variance facts*, not behavioural proof.
- **Design provenance:** `drl-expert` spec (2026-07-06) merged with an independent external review; two external recommendations were rejected on code/algebra grounds (see §12).

**Freeze status: FORMULAS FROZEN 2026-07-08.** The pure scorer and wrapper/packet layer cover the §1 validity gates, §0 provenance block, burn-in `W`, §8B floored-update exclusion, §8C–G comparability guards/covariates, G3/G4 inputs, S7 diagnostic rendering, OFF-only calibration validity, spec-scoped traceback/run-log validity, and `advantage_std_floored` contamination reporting. The owner ratified the §11 threshold set (values + resolution formulas) on 2026-07-08 after a drl-expert review (2 adjustments folded in: `g4_abs_floor` 5→2; ε_rel freeze-time validation added). Formulas are committed BEFORE the OFF arms run, closing the soft-peek channel: any slot that resolves from OFF data does so through a pre-committed rule at §10 step 3. The OFF wave (seeds 41–45, 400 ep/run) launched 2026-07-08. **Same-commit discipline: no training-path code change between the OFF and ON waves** (docs-only commits acceptable; `frozen_config` §1 equality covers hyperparameters, the discipline covers code). ON launch remains blocked on step-3 scalar resolution and explicit owner approval.

Predicate ownership: *pure scorer* = LEG-A/LEG-B/MECH/G1/G2 statistics + composite (`n`-aware; ACCEPT only at n=10). *Wrapper* = everything that reads telemetry: §1, §0, burn-in, §8B, §8E covariates, §8F contamination reporting, G3/G4, and the tier-gated call into `composite_verdict`.

---

## §0. Provenance & scope declaration (MAJOR-3)

The report the scorer emits MUST carry this block verbatim, so Stage-0 evidence can never be laundered into HRA evidence:

```
Stage-0 variance gate:
    metric:  Cov(R_cf, R_total) / Var(R_total)  (value-free, raw-scale, per-return)
    reading: median 1.017 (PDR-0032/0033), Stage-2-OFF control
    licenses: running the Stage-2 HRA A/B at all
    HRA evidence: NO — it does not test the HRA implementation

Stage-2 HRA acceptance (THIS gate):
    source:  fresh paired HRA-OFF / HRA-ON A/B (§10)
    metrics: EV_main, EV_sum, advantage-path volatility, safety (§6)
```

**Actor-objective declaration (required field).** The current implementation is **A — critic/value-target decomposition only**: the policy advantage is a single GAE on `V_total = V_main + V_cf` (verified: `rollout_buffer.py` `values_total = values + cf_values`; per-stream GAE feeds `returns_main`/`returns_cf` for the *value* targets only, not the actor objective). `R_cf` therefore still shapes the trunk via the policy gradient — which is exactly why the hollow-win guard (§3) is necessary.

Objective B — actor advantage uses `R_main`, `R_cf` auxiliary — is a **reward-objective change reserved for a later PDR** with stricter behavioural gates. The scorer MUST read the run's `actor_advantage_source` and **fail the validity layer if it is not `total_reconstructed`** (i.e. if someone ran B under this gate).

---

## §1. Validity gates (fail BEFORE interpretation)

If any of the following holds for either arm, the A/B is `INVALID` and NO EV/downstream interpretation is performed (this prevents "EV_main went up, therefore it worked"):

- run incomplete (fewer scored updates than the pre-registered budget after burn-in);
- any non-finite (`NaN`/`Inf`) in a scored metric, or a traceback in the run log;
- config mismatch between the paired arms on any frozen field (§10);
- seed mismatch (the ON and OFF arm of a pair are not the same fresh-init seed);
- arms not fresh-init paired (e.g. warm-started, or unpaired seeds);
- telemetry missing `explained_variance` (both arms) or `ev_main`/`ev_cf`/`ev_sum` (ON arm);
- **OFF-arm scoring reads any hra-gated metric** — `ev_main`/`ev_cf`/`ev_sum` are emitted ONLY under `hra_value_decomposition=True` (`ppo_agent.py:751`); the OFF comparand is `explained_variance`. A rubric that reads `ev_main_OFF` is a bug, not a null.
- Stage-0 provenance block (§0) absent;
- `actor_advantage_source != total_reconstructed` (scope-fence breach, §0);
- episode↔J join failure for the safety gates (§6).

---

## §2. Data model

Primary series come from Karn `ppo_updates` filtered by `run_dir` (one run = one seed × one leg), ordered by update index. For run `r` and metric `X`, after discarding a burn-in window of `W` updates (§8-W):

- `level(r, X) = median_{u>W} X_u`
- `vol(r, X)   = IQR_{u>W} X_u = P75 − P25`  (robust to the volatility being measured)

**Pairing unit = seed.** Each seed `s` yields `r_on(s)` (`hra_value_decomposition=True`) and `r_off(s)` (`=False`), fresh-init from the same seed, everything else frozen (§10). Two tiers: **n=5 direction screen** (go/no-go) → **n=10 powered claim** (the only tier that banks ACCEPT).

Cross-leg comparand convention: OFF total-value EV = `explained_variance`; ON = `ev_sum` (byte-identical alias of `explained_variance` on the ON leg). **The floor and guards never read `ev_main` as an acceptance quantity across legs** — that is the MAJOR-1 trap.

---

## §3. LEG A — `ev_sum` non-inferiority floor (HARD, level)

The anti-hollow floor: de-shaping must not *degrade* total value fit. Per seed:

```
Δ_A(s) = level(r_on(s), "ev_sum")  −  level(r_off(s), "explained_variance")
```

**Margin δ (frozen from the A/B's own OFF arms, before any ON scoring):**

- PRIMARY: `δ = max(0.05, paired_bootstrap_SE( {level(r_off(s), "explained_variance")}_s ))` — one bootstrap SE of the OFF EV-level *estimate*, used as a proxy for the noise-dominated non-inferiority margin (it is the SE of the OFF central tendency, NOT the SE of the paired difference `Δ_A(s)` itself; a defensible, mildly conservative margin — drl-expert L1). At exactly n=5 this is a coarse 5-point median bootstrap; the `0.05` floor protects the downside (L2).
- FALLBACK (if the OFF arms are too few for a stable SE, i.e. n<5): `δ = max(0.05, 0.5 · median_s vol(r_off(s), "explained_variance"))`.
- Absolute floor `0.05` EV-points so a quiet OFF corpus cannot drive δ→0.

**Predicate:**

- **n=5 (screen):** PASS iff `≥4/5` seeds have `Δ_A(s) ≥ −δ`. (Honest OC: a non-inferiority screen passes most seeds under a true-zero effect — this is a *lenient* go/no-go, not a superiority test.)
- **n=10 (claim):** PASS iff one-sided Wilcoxon signed-rank on `{Δ_A(s) + δ}` rejects "ON worse than OFF−δ" (`p<0.05`) AND `median(Δ_A) ≥ −δ`.

**Hard override:** `LEG A == FAIL ⇒ REJECT`, regardless of any `ev_main` movement. This is the MAJOR-1 reversal trigger from PDR-0029.

---

## §4. LEG B — advantage-path volatility reduction (gated downstream signal)

The "is the win real?" signal. **Volatility, not level** — because `returns_total = advantages_total + values_total` gives the identity `pre_norm_advantage_std² = (1 − ev_sum)·Var(returns_total)` (advantage std computed at `rollout_buffer.py:745`, `correction=0`; the storage site is `ppo_agent.py:854`), so requiring an advantage-std *level* drop while pinning `ev_sum` (§3) is self-contradictory (it forces `Var(returns_total)` to collapse — a policy-regime red flag, and confounded across two different fresh-init runs). Update-to-update *volatility* of the value-fit residual is what a detached-trunk `cf_value_head` should steady. (Tucker 2018 mirage avoidance.)

**Gated statistic = the RETURN-SCALE-INVARIANT residual volatility (drl-expert M2).** Gating on `IQR_u(pre_norm_advantage_std)` is confounded: from the identity it also drops if `Var(returns_total)` merely has a steadier trajectory across updates (a return-regime artifact, no actor-path shielding). Since `sqrt(1 − ev_u) = pre_norm_advantage_std_u / sqrt(Var(returns_total)_u)`, the scale-invariant version of the *same* signal is the unexplained-variance-fraction volatility. Per leg, over non-floored updates (§8B), with `ev_u` = per-update `ev_sum` (ON) / `explained_variance` (OFF):

```
Δ_B(s) = ( IQR_u(sqrt(1 − ev_u))_on  −  IQR_u(sqrt(1 − ev_u))_off )
         / IQR_u(sqrt(1 − ev_u))_off
```

**Threshold `ε_rel = 0.10`** (≥10% median volatility reduction), frozen before ON reveal; sanity-anchored to exceed the OFF seed-to-seed relative spread.

**Predicate:**

- **n=5 (screen):** PASS iff `≥4/5` seeds `Δ_B(s) < 0`; `INCONCLUSIVE` iff `≥3/5`; else `FAIL`. (Genuine superiority screen — carries the PDR-0019 sign-test operating characteristic.)
- **n=10 (claim):** one-sided Wilcoxon (`ON < OFF`, `p<0.05`) AND `median(Δ_B) ≤ −ε_rel` ⇒ PASS; `median(Δ_B) ≥ 0` ⇒ FAIL; else `INCONCLUSIVE` → owner adjudication (asymmetric-null, PDR-0026).

**Reported covariates (drl-expert M2):**
- `IQR_u(Var(returns_total))` per leg — the confound. **Downgrade LEG-B to INCONCLUSIVE when the ON reduction in the gated statistic is explained by a matching drop in `IQR_u(Var(returns_total))`** (a return-regime artifact, not shielding). The scale-invariant gated statistic above is chosen precisely so this is a residual check, not the primary defense.
- raw `IQR_u(pre_norm_advantage_std)` + advantage-std level deltas + `Var(returns_total)` level per leg — descriptive context.
- `vol(·,"gradient_cv")` corroboration — should move the same direction. A LEG-B PASS with `gradient_cv` volatility *rising* is mirage-suspect → downgrade to INCONCLUSIVE + RCA.

---

## §5. `EV_main` — mechanism guard (necessary, weak; not a discriminator)

`V_main` regresses a definitionally lower-variance target, so `ev_main` improving is cheap — it earns **no positive credit**, but its failure signals the decomposition didn't do its own bookkeeping (structural bug). There is no `ev_main_OFF` (ON-only telemetry), so the honest baseline is the OFF single-head EV:

```
MECH holds  ⟺  vol(r_on(s),"ev_main") < vol(r_off(s),"explained_variance")  on ≥⌈0.8·n⌉ seeds
```

The count is an **80% majority that scales with the tier** (drl-expert H3): ≥4/5 at the screen, **≥8/10 at the n=10 claim tier** — never a fixed absolute `4`, which would be a 40%-threshold REJECT gate at n=10 and let structural bookkeeping bugs through at exactly the tier that banks ACCEPT.

`ev_sum` volatility is deliberately **not** required to fall — `V_total` still carries the high-variance `V_cf` head, so total-EV stability is not the expected payoff (that surfaces in LEG B). `MECH == fail ⇒ REJECT` (structural).

---

## §6. Safety gates (necessary, not sufficient — Stage-2 may be a critic win before a behavioural win, but must not damage the system)

All paired per-seed medians, thresholds frozen before ON:

- **G1 — val-acc non-regression:** `median_s[ val_acc_on(s) − val_acc_off(s) ] ≥ −τ_acc`, `τ_acc = 0.3pp`. An *informative* regression ⇒ cf term was load-bearing credit ⇒ REJECT + prescription: refold via Harutyunyan 2015 DPBA (policy-invariant potential), NOT raw dense reward.
- **G2 — params non-inflation:** `median_s[ added_params_on(s) − added_params_off(s) ] ≤ Δparam_max`. Derived from `HostSnapshot.fossilized_params` ("permanently added params"). Split from G1 so an accuracy regression cannot hide behind a param reduction.
- **G3 — churn not pathological:** germinate / prune / fossilize per-episode rates on the ON arm not materially elevated vs OFF (no reward-farm signature). Reported against the existing churn guardrail; a large ON-specific spike → RCA before banking.
- **G4 — guard channels not ON-elevated:** governor rollbacks, value-collapse, gradient-anomaly, reward-hacking detectors — no large ON-specific elevation vs OFF. (Governor-rollback asymmetry is an RCA trigger, not a hard gate, per PDR-0026.)

---

## §7. Composite predicate & decision table

The composite is **tier-aware** — ACCEPT is banked ONLY at n=10; a clean n=5 screen is `SCREEN_PASS` (drl-expert H2):

```
clean        := LEG_A==PASS ∧ LEG_B==PASS ∧ MECH==hold ∧ G1==PASS ∧ G2==hold ∧ G3==hold ∧ G4==hold
ACCEPT       ⟺ clean ∧ n==10                     # banked ONLY at the claim tier
SCREEN_PASS  ⟺ clean ∧ n==5                      # screen cleared → run n=10; NEVER an ACCEPT
REJECT       ⟺ LEG_A==FAIL                       # MAJOR-1 hard override
              ∨ MECH==fail                       # mechanism did not fire (structural)
              ∨ G1==FAIL (informative regress)   # cf load-bearing → DPBA refold
              ∨ LEG_B==FAIL                       # downstream materially worsens
INCONCLUSIVE ⟺ ¬REJECT ∧ ¬clean                  # null band / safety soft-fail → owner adjudication
```

`clean` uses `==PASS`, not `≠FAIL`: a 3-valued `INCONCLUSIVE` leg_a/g1 never clears it (drl-expert-adjacent; python-review C1). G3/G4 are **required** inputs, never defaulted (drl-expert M3).

| EV_main (vol) | EV_sum (level) | downstream (adv-vol) | Verdict |
|---|---|---|---|
| stabilises | non-regresses | reduces | **ACCEPT** |
| improves | materially regresses | any | **REJECT — hollow EV_main win** |
| stabilises | non-regresses | materially worsens | **REJECT (LEG B FAIL)** |
| stabilises | non-regresses | null band | **INCONCLUSIVE → owner** |
| flat | improves | improves | **MECHANISTIC AMBER — not Stage-2's declared win** |
| worsens (MECH fail) | any | any | **REJECT (structural)** |
| noisy / incomplete | — | — | **INVALID / not informative** (§1) |

"EV_main stabilises" ≡ **volatility (IQR-over-updates) reduction vs the OFF single-head `explained_variance` volatility** — the only honest baseline. Level of `ev_main` is deliberately unused across legs.

---

## §8. Comparability traps to control

- **(A) normalizer re-warm.** ON reconstructs `V_total = denorm_main(V_main) + denorm_cf(V_cf)` (two fresh normalizers) vs OFF's one. EV read mid-warmup makes ON look spuriously bad → controlled by burn-in `W`, identical both legs.
- **(B) floored-EV asymmetry — a SCORED PRECONDITION, not a note (drl-expert M1).** `EV = 1 − residual/max(Var(returns), floor=1.0)` (`value_metrics.py:164`). By the intervention's own premise `R_cf` is the high-CoV stream, so `returns_main = returns_total − returns_cf` is the low-variance remainder — **the quantity most likely to fall below the absolute floor 1.0**, which biases `ev_main` *up* in level and *down* in volatility (→ MECH can hold for a floor artifact, not a mechanism) and, when ON's `Var(returns_total)` dips below the floor more than OFF's, biases ON `ev_sum` *up* (→ an easier, unreal LEG-A pass). Therefore the wrapper MUST, before any level/vol statistic: (i) **exclude updates where `ev_return_variance ≤ 1.0`** from every leg's series; (ii) **REJECT/INVALID a leg if the floored fraction is materially asymmetric between arms**; (iii) **report the per-leg floored fraction in the verdict.** The wrapper now reads `ev_return_variance`, applies the exclusion, and reports per-leg fractions; the material asymmetry threshold is a freeze-time threshold slot (§11).
- **(C) moment-convention mixing.** EV family = `correction=1`; `pre_norm_advantage_std` and the value-free shares = `correction=0`. LEG A reads only `explained_variance`/`ev_sum` (both correction=1); LEG B uses only within-series IQR (convention-immune). Never compare an EV against a correction=0 sibling.
- **(D) per-stream vs total.** Only `ev_sum` (=total) is the OFF comparand; `ev_main`/`ev_cf` enter only MECH (§5).
- **(E) different `returns_total` regimes.** Report `Var(returns_total)` per leg as a covariate; if the two legs' return-variance regimes diverge materially, the paired EV comparison is confounded — surface it, don't silently score.
- **(F) advantage-norm config.** `per_head_advantage_norm` and `ADVANTAGE_STD_FLOOR` change `pre_norm_advantage_std` semantics → frozen identical both legs (per-head OFF, same floor). The wrapper rejects `per_head_advantage_norm=True` and reports per-leg `advantage_std_floored` fractions. Nonzero fractions are flagged as LEG-B contamination evidence; no hard frequency cutoff is registered unless the owner freezes one before ON.
- **(G) sequential-A/B warm-up.** `dual_ab.py` trains groups sequentially; EV/advantage/val-acc are step-indexed not time-indexed → safe, note in the run log.

**Burn-in `W`:** the slowest re-warm is `cf_value_normalizer` (high-variance `R_cf`), which the total `value_target_scale` does not track. Set `W` = later of (i) `value_warmup_batches × ppo_updates_per_batch` and (ii) the update at which `value_target_scale` plateaus AND (ON) `cf_value_loss` plateaus. Same absolute `W` discarded on both legs.

---

## §9. Diagnostic telemetry (per-stream target scale)

Two ON-leg scalars, added so a reader can tell "critic got better" from "target got trivially easy":

- `value_main_target_scale` = `value_main_normalizer.get_scale()`
- `cf_value_target_scale`   = `cf_value_normalizer.get_scale()`

(The normalizers now emit these ON-leg scalars through S7; old/all-missing telemetry renders them as `UNAVAILABLE`, never as defaults.) **Descriptive, not gating.** Interpretation folded into the report:

```
Expected non-hollow pattern:  target_main_scale ↓, ev_main ↑/steadier, ev_cf noisy/modest,
                              ev_sum non-regresses, pre_norm_advantage_std VOLATILITY ↓.
Hollow pattern:               ev_main ↑, ev_cf terrible, ev_sum ↓, adv-vol unchanged.
```

If `ev_main` improved *only* because `value_main_target_scale` collapsed, that is still reported (the report says so) but is not, by itself, a hollow-win REJECT — the EV_sum floor + LEG B carry the verdict.

---

## §10. Pre-registered run protocol

- **Design:** paired, fresh-init, per-seed ON-vs-OFF. n=5 direction screen → n=10 for any ACCEPT. Budget ≥ prior A/B (**≥200 episodes/run**; longer strictly better for the volatility estimates); pre-register the exact number.
- **ON posture:** `hra_value_decomposition=True`, `reward_mode=SHAPED`, `reward_family=CONTRIBUTION`.
- **OFF posture:** `hra_value_decomposition=False`, **same** `reward_mode=SHAPED`, **same** `reward_family=CONTRIBUTION`. The `bounded_attribution` (`R_cf`) term is in the reward on BOTH legs — folded into the single value target on OFF. **The toggle is decomposition-only, never reward-removal** (OFF is NOT a `reward_family`/`SIMPLIFIED` change — that would be objective B, out of scope §0).
- **Frozen between legs (only `hra_value_decomposition` toggles):** seed & network init; lr, clip_ratio, entropy_coef + schedules, gamma, gae_lambda, epochs/episode, n_envs, batch, `ppo_updates_per_batch`; the entire reward config (mode, family, all shaping coefficients, `R_cf` scale, rent, PBRS potentials, terminal-acc); `ev_return_variance_floor=1.0`; `per_head_advantage_norm=False` + `ADVANTAGE_STD_FLOOR`; value-warmup config; host task (`cifar_baseline`) + curriculum; governor/safety-gate config; blueprint set & slots; episode budget.
- **Freeze order (NO peeking):** (1) run all OFF arms; (2) compute `δ` from valid OFF arms only; (3) confirm fixed `ε_rel=0.10` and `τ_acc=0.3pp`, plus owner-set `Δparam_max`, `W`, floored-asymmetry materiality, G3/G4 materiality, and any advantage-floor policy; (4) freeze thresholds into this doc; (5) run + score ON arms. Calibration touches only control data.

---

## §11. Frozen thresholds

Owner-ratified 2026-07-08 (drl-expert review folded in). Formulas frozen NOW; rows marked
*step-3* resolve their scalar from OFF-arm data only, through the pre-committed rule below.

| Symbol | Meaning | Value / rule (FROZEN 2026-07-08) |
|---|---|---|
| `δ` | ev_sum non-inferiority margin | `max(0.05, paired_bootstrap_SE(OFF ev-level medians))`; fallback `max(0.05, 0.5·median OFF IQR)`. Deterministic bootstrap: 2000 resamples, `np.random.default_rng(0)` (code default, recorded). **Tier semantics: recomputed from all 10 OFF arms at the n=10 claim tier** (the 5-arm δ governs the screen only). *step-3* |
| `ε_rel` | adv-vol relative reduction floor | `0.10` fixed — **with a mandatory step-3 validation:** compute the OFF seed-to-seed relative spread of `IQR_u(sqrt(1−ev))`; if spread ≥ 0.10, LEG-B sits inside its own noise floor → escalate to owner before ON (raise ε_rel or accept an explicitly underpowered LEG-B) |
| `τ_acc` | val-acc regression deadband | `0.3pp` |
| `Δparam_max` | added-param inflation ceiling | `max(0.10 · median_s(added_params_off), IQR_s(added_params_off))`. Degenerate-zero guard: if `median_s(added_params_off) = 0`, do NOT auto-resolve — escalate to owner. *step-3* |
| `W` | burn-in updates discarded | `max(10, plateau(value_target_scale on OFF arms) + 5)`, frozen as an exact integer. `plateau(x)` := first update `u` whose trailing 8-update window has max relative change of `x` < 5%. The ON-leg `cf_value_loss` plateau is a **post-hoc validity CHECK, not a re-freeze**: if `cf_value_loss` has not plateaued by `W` on an ON arm, that arm is INVALID → re-run with a larger *pre-registered* W; never a silent post-hoc W bump. *step-3* |
| `floored_asymmetry_max` | material ON/OFF floored-EV fraction asymmetry | `0.10`, applied per seed-pair (`validate_pair` per-pair `|Δfrac|`). Symmetric heavy flooring is backstopped by the completeness gate: `budget` counts scored updates POST burn-in + POST §8B exclusion (verified in code) |
| `g3_ratio_max` | churn material-elevation ratio | `1.5`, per channel (germinate / prune / fossilize gated separately — code shape). **Step-3 confirmation:** 1.5 must exceed the observed OFF seed-to-seed churn spread; if not, escalate |
| `g4_ratio_max` / `g4_abs_floor` | guard-channel material-elevation thresholds | `2.0` / `2`. Code semantics: materially elevated ⟺ `(on − off) > abs_floor` AND `on > ratio_max · off`; `reward_hacking > 0` on the ON arm is a hardcoded zero-tolerance trip regardless of thresholds. Floor 2 (not 5) so an ON-specific alarm-channel elevation of 3+ events (value-collapse, gradient-anomaly, numerical-instability) routes to owner adjudication; the benign PDR-0026 rollback asymmetry (1.68×) still passes |
| `advantage_std_floored` policy | LEG-B contamination handling | report exact ON/OFF fractions + `OBSERVED` flag; no hard cutoff. **Pre-committed asymmetry soft-gate:** per-pair `|frac_on − frac_off| > 0.10` → LEG-B is treated INCONCLUSIVE → owner adjudication (parallel to §8B) |
| budget | pre-registered run size | **200 rounds/run hard-stop (= 200 PPO updates, 2400 env-episodes)** AND `budget = 150` minimum scored updates (post burn-in + §8B exclusion) per arm for §1 validity. AMENDED 2026-07-08 pre-data: the first ratification wrote "400 episodes" through a unit error (env-episodes vs updates — the prior A/B's "200 episodes/run" is 200 *rounds*); corrected before any OFF arm produced data. A low-update run flags INVALID rather than silently producing a noisy IQR |
| posture | base config | **`configs/config-3slot-3seed-baseline-shaped.json`** — the prior Shapley A/B base (3 slots r0c0/r0c1/r0c2, 12 envs, entropy 0.15→0.08, gae_lambda 0.95, amp on, auto-forward G1–G3, task cifar_baseline) + `return_variance_telemetry: true` + `per_head_advantage_norm: false` pinned (§8F) + the `hra_value_decomposition` toggle. Arm configs generated from ONE base so §10 config-equality holds by construction: `configs/ablations/stage2-ab-{off,on}.json` |
| seeds | pairing | paired fresh-init seeds `41–45` (screen tier), continuing the PDR-0021 convention |
| stationarity | LEG-B pre-check | **step-3 check on OFF arms:** first-half vs second-half `IQR_u(sqrt(1−ev))` of the scored window; material non-stationarity (e.g. a late germination wave) → escalate before ON, pre-registering a windowed scored region if needed |
| tiers | direction / claim | n=5 screen → n=10 claim (ACCEPT only at n=10) |

Calibration principle (recorded at owner ratification): `floored_asymmetry_max`, `g3_ratio_max`,
and `ε_rel` are noise-floor thresholds — each is validated at step 3 to exceed the observed OFF
spread rather than trusted as a round number. Every gate in this table fails toward the owner
(INVALID / INCONCLUSIVE / escalation), never toward a silent ACCEPT.

---

## §12. Rejected external recommendations (recorded for provenance)

1. **Downstream signal on advantage-std *level*.** Rejected: algebraically entangled with the ev_sum floor (§4 identity). Kept as descriptive-with-covariate only.
2. **`EV_main` as the paired *primary* gate (`ΔEV_main = EV_main_ON − EV_main_OFF`).** Rejected: `ev_main` is ON-leg-only telemetry — `EV_main_OFF` does not exist, and the target-easier effect makes level-improvement weak evidence. Kept as the MECH guard (§5).

---

## §11.1 Step-3 resolutions & pre-ON amendment (2026-07-09, owner-ratified)

Resolved from the 5 valid OFF arms (`telemetry/stage2_ab_off/seed{41..45}` @ `fe177844`,
200/200 updates each; full record: `docs/analysis/2026-07-09-stage2-off-calibration-step3.md`;
harness report: `docs/analysis/2026-07-09-stage2-off-calibration-report.txt`):

| Symbol | Resolution |
|---|---|
| `W` | **17** (per-arm plateaus 8/8/12/8/8; rule applied as written — note the rule fires early because per-update relative change in `value_target_scale` is <5% almost immediately while the cumulative level drifts ~2.7→~25; the drift is handled by the stationarity ruling below) |
| `δ` | **0.050311** (`calibrate_off`, 5 OFF arms) |
| `Δparam_max` | **1527.8** (added_params/seed 2637.6/1308.3/3373.0/879.0/2836.1; median 2637.6; IQR_s 1527.8 dominates; NOT degenerate-zero) |
| G3 spread confirm | PASS (germinate 1.006, prune 1.007, fossilize 1.379 — all < 1.5) |

**Escalations fired and owner rulings (2026-07-09):**

1. **ε_rel validation FAILED** — OFF seed-to-seed relative spread of `IQR_u(sqrt(1−ev))`
   = 0.5716 ≥ 0.10 — AND **stationarity pre-check MATERIAL** — second-half/first-half
   scored-window volatility ratios 8.41/7.70/4.50/13.55/2.38 across arms (EV lifts late
   while Var(returns) grows ~10–25× within-run; the LEG-B statistic conflates
   learning-drift with noise on this horizon).
   **RULING: LEG-B is DEMOTED TO DESCRIPTIVE at the n=5 screen** (pre-registered option
   "accept an explicitly underpowered LEG-B"). The screen composite predicate becomes
   **LEG-A ∧ MECH ∧ G1–G4**. LEG-B numbers + covariates are still computed and reported.
   The scorer's emitted composite `verdict` field composes the original §7 predicate
   (including LEG-B) and is SUPERSEDED at the screen tier by this amendment: the screen
   verdict is read from the packet's `leg_a` / `mech_hold` / `g1..g4` fields. Any
   volatility-reduction CLAIM is deferred to the n=10 tier or a redesigned
   windowed/detrended metric (pre-registered before use).
2. **Device-placement homogeneity (harness spec-vs-code):** the §10 frozen list never
   included device placement, but `_FROZEN_RUN_CONFIG_COLUMNS` swept in
   `policy_device`/`env_devices_json`, rejecting the ratified two-GPU OFF wave.
   **RULING: placement fields are provenance, excluded from cross-seed homogeneity;
   per-pair ON/OFF placement equality is enforced in `validate_pair`.** Implemented
   read-path-only with regression tests (RunMeta.placement; 182 harness tests green).
   Consequence: **each ON arm MUST run on the same device as its OFF partner**
   (41/43/45 → cuda:0, 42/44 → cuda:1).

Amendment recorded BEFORE any ON arm produced data (no ON telemetry exists at ratification).

---

## §11.2 W-rule cf-plateau VALIDITY GATE demoted to descriptive (2026-07-10, owner-ruled)

**Trigger.** A review found the §11 W-rule was only half-implemented: the acceptance
reader never selected `cf_value_loss`, so the "an ON arm whose cf_value_loss has not
plateaued by W is INVALID" clause could not fire — a still-warming cf head could reach
SCREEN_PASS on unstable EV. The fix ingested `cf_value_loss` into `UpdateRow` and
implemented the plateau gate faithfully (`plateau()`, frozen rule: first update whose
trailing-8 window has max consecutive relative change < 5%).

**Finding on real data.** With the gate implemented as frozen, **every ON arm is
INVALID at any W.** Live ON telemetry (seeds 41/42, ~28 updates): `cf_value_loss`
oscillates 0→50 per update — raw series never plateaus, and neither does its trailing-8
median. The smooth normalizer analog `cf_value_target_scale` also fails to plateau on
seed42 (step-jumps +62%/+73% past update 17). Root cause is the SAME within-run
non-stationarity that demoted LEG-B (§11.1): the cf stream's target scale grows across
the whole run (Var(returns) ↑ 10–25×), so a "cf head has stopped warming" criterion is
unsatisfiable on this horizon regardless of W.

**RULING (owner, 2026-07-10): the cf-plateau clause is DEMOTED TO DESCRIPTIVE.**
- The plateau condition is NO LONGER a validity gate — an un-plateaued `cf_value_loss`
  does not invalidate an ON arm. (Otherwise the entire pre-registered A/B is unscorable
  for a reason unrelated to HRA's effect.)
- **RETAINED as a hard gate:** `cf_value_loss` must be PRESENT on every ON update
  (telemetry-signature completeness), and non-finite `cf_value_loss` on a scored update
  fails finiteness. The reader now selects it and `validate_pair` enforces presence.
- `plateau()` is retained and its per-arm index is reported descriptively in the step-3
  / packet record (a genuinely warming-then-flat cf head would still show a finite
  plateau; the oscillatory reality is the informative datum).
- Any future warmup-stability CLAIM needs a redesigned, pre-registered criterion
  (e.g. plateau on a detrended/log cf-scale, or a fixed-fraction burn-in) — banked, not
  used at this screen.

**Consequence for the screen predicate.** Unchanged from §11.1: LEG-A ∧ MECH ∧ G1–G4,
with the cf-warmup no longer able to invalidate arms. Recorded while the ON wave was
in flight (seeds 41/42 producing telemetry; the finding is derived from cf_value_loss
shape, which HRA cannot make stationary — not from any EV/verdict quantity).
