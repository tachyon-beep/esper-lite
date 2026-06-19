# EV-Telemetry Robustness Under the Op-Marginal V(s) Critic — Design

```yaml
date: 2026-06-18
status: Approved (design) — pending spec review, then implementation planning
domain: RL telemetry / value-fit diagnostics (simic PPO controller, leyline contracts, karn views, overwatch panels)
reviewers:
  - drl-expert + yzmir-deep-rl  # estimator semantics, value-fit diagnostics, gate thresholds
  - axiom-python-engineering    # leyline contract surface, consumer audit, no-bug-hiding compliance
origin: >
  Sprint item (1) following P0-1 (op-independent V(s) baseline, commits 6a27b8e3 / 177a53aa /
  6d97391b / 31bf8cb7). P0-1 A/B revealed an explained_variance std blowout on K>1 runs that is
  a denominator artifact, not a value-head regression. This design makes EV an honest diagnostic
  under the new estimator and fixes the gates/alerts that will now false-alarm.
```

## Problem

`explained_variance` (EV) is the primary value-fit diagnostic for the simic PPO controller.
It is computed once per PPO `update()`, **pre-update** (pre-epoch-loop), over the flattened
valid-mask batch in `src/esper/simic/agent/ppo_agent.py:619-631`:

```python
valid_values  = data["values"][valid_mask]                        # 619  rollout-stored, normalized scale
valid_returns = data["returns"][valid_mask]                       # 620  raw scale
raw_values    = self.value_normalizer.denormalize(valid_values)   # 621
var_returns   = valid_returns.var()                               # 622  <-- THE DENOMINATOR (Bessel, correction=1)
if var_returns > 1e-8:                                            # 624  guard
    ev_tensor = 1.0 - (valid_returns - raw_values).var() / var_returns
    explained_variance = ev_tensor
else:
    explained_variance = torch.tensor(0.0, device=valid_returns.device)  # 628 guard -> 0.0
# emit at :631
```

EV = `1 - Var(returns - values) / Var(returns)`. This is a **ratio with a batch-estimated
denominator**. On a batch with low return spread (small `var_returns`), the ratio explodes even
when the *absolute* residual `Var(returns - values)` is small — so EV craters to a large negative
number while `value_loss` on that very same update is **low**. The existing `var_returns > 1e-8`
guard (`:624`) only catches the degenerate near-zero-variance case; it does nothing for the much
larger band of *small-but-finite* return variance where the artifact actually bites.

**Bessel convention.** `var_returns = valid_returns.var()` uses PyTorch's default Bessel
correction (`correction=1`, sample variance) over the **valid-masked flattened batch**. This is a
different population *and* a different correction from the buffer-wide `return_variance` already
emitted by `compute_value_function_metrics` (`leyline/value_metrics.py`, `returns.var(correction=0)`
over the whole buffer; aggregated at `vectorized.py:218`). The two quantities will not match. See
"Telemetry contract changes" for why the EV-denominator variance must be emitted under a *distinct*
name, and the "Correction convention" decision for how the whole EV family is reconciled.

### Why P0-1's op-marginal V(s) is more exposed

The pre-P0-1 critic was op-conditioned, `Q(s, op)`. Its targets carried **cross-op value spread**:
on any given batch the returns associated with different ops differ, inflating the effective
`var_returns` denominator and damping the blowout. P0-1 replaced this with an **op-marginal `V(s)`**.
`V(s)` sees only the *within-state* return variance on a batch; on low-return-variance batches that
denominator is tiny, so the same residual produces a far more negative EV. The estimator is more
honest about value fit overall, but **strictly more exposed** to the denominator artifact.

### The 80/20 question: WHY are these batches low-variance?

Before hardening the diagnostic, interrogate the environment/reward signal (the 80/20 rule: check
the environment and reward *before* changing the estimator math). Under an op-marginal `V(s)`, a
recurring low-`Var(returns)` batch tail is **not only** a numerical inconvenience — it is a
legitimate reward/return-regime signal. Healthy-run return std is 7–13 (`var_returns` ~ 49–169);
a batch with `var_returns < 1.0` (std < 1.0) is >7× below typical. Near-flat returns within a batch
can mean sparse/saturated reward, a short effective horizon, or a return-scale collapse — all
Phase-1/Phase-2 (environment/reward) symptoms, distinct from the Phase-3 (value-fit) question EV is
meant to answer.

Therefore the floor must **not silently swallow** this signal. The run-level count of flagged
updates (`ev_low_return_variance_count`) and the joint `(var_returns, valid-N)` distribution are
surfaced as a **first-class reward/return-regime diagnostic** (see Locked decisions and the
monitoring requirement in Risks). A *sustained* high flagged-fraction is itself an alert condition
warranting a reward/horizon/return-scale investigation — not merely noise to suppress.

### Empirical evidence that the blowout is artefactual, not a regression

From the P0-1 A/B (K=4 arm vs control):

| Signal | Control (Q(s,op)) | P0-1 (V(s)) | Reading |
|--------|-------------------|-------------|---------|
| EV mean (K=4)              | +0.572 | **−0.337** | Crater |
| EV std (K=4)               |  0.572 | **2.264**  | Blowout |
| `value_loss` (median)      | —      | **0.099 (lowest of all arms)** | Value fit is GOOD |
| median EV (across arms)    | —      | **+0.239 (best of all arms)**  | Central tendency is fine |
| K=4−K1 EV gap              | +0.137 | **+0.201** | Multi-epoch is helping |
| `head_op_grad_norm` late-alive frac | 0.05 | **0.89** | Op-head starvation fixed |

A negative-mean / high-std EV co-occurring with the **lowest** `value_loss` and the **best** median
EV is the signature of a denominator artifact on a subset of low-return-variance batches dragging
the mean and inflating the std — not a value function that has gotten worse. The mean and std are
the wrong summary statistics for a heavy-tailed ratio.

### Why this is urgent (the part with teeth)

Several **gates and alerts** key on EV mean/std and will now false-alarm on every K>1 run. The most
damaging: `src/esper/karn/mcp/views.py` `run_confounders` marks `VALUE_COLLAPSE_DETECTED` rows with
`true as proof_blocking` (views.py:649, event listed at views.py:652). A K>1 EV blowout will flag
proof confounders and **invalidate the reward-efficiency experiment verdict** even though the value
function is healthy. Full enumeration in the Consumer Audit section.

## Key reframe

EV is not broken as a *concept* — it is broken as a *summary statistic for an op-marginal estimator
on heavy-tailed, low-variance batches*. The fix is twofold:

1. Make the **raw EV emission** numerically honest about its own denominator (variance floor + an
   explicit `ev_low_return_variance` flag), so a single low-variance batch can no longer manufacture
   a −8 outlier that poisons the run-level mean/std.
2. Emit a **return-variance-robust companion metric** so consumers (gates, panels, analysts) can read
   value fit from a signal that does not divide by a tiny denominator. Several such signals already
   exist (`v_return_correlation`, `bellman_error`, `value_loss`) and are unguarded — they become the
   honest cross-check that proves the floor is numerical handling, not bug-masking.

Crucially, **no value-fit problem is hidden — and the "nothing hidden" claim is qualified precisely
for the co-occurrence case** (genuine bad fit *on the same update as* low return variance). On a
flagged update, two of the "robust" signals are themselves degraded:
  - `value_nrmse` divides by `valid_returns.std() + ev_var_floor_std`; on a genuinely small
    `returns.std()` the additive floor **dominates** the denominator, so `value_nrmse` becomes
    ~`sqrt(residual_var)/ev_var_floor_std` — a floor-stabilized residual, **not** a scale-independent
    NRMSE. It reads small precisely because the denominator is inflated by the floor.
  - `v_return_correlation` returns its **0.0 sentinel** on low-`r_std` batches
    (`value_metrics.py:77-82`: `if v_std > 1e-8 and r_std > 1e-8: ... else: correlation = 0.0`) —
    exactly the regime that flags EV. So on a flagged update it can read 0.0 for denominator reasons,
    not because the critic is bad.

Therefore, **on a FLAGGED update, regression detection rests on the genuinely unfloored,
scale-anchored signals — `value_loss` (MSE on the trained scale) and `bellman_error` (mean |TD|,
absolute residual magnitude) — which never divide by return variance at all.** `value_nrmse` is a
*secondary*, floor-stabilized signal; `v_return_correlation` is trustworthy only when the
denominator is healthy (`ev_low_return_variance == False`). If `V(s)` genuinely regresses on a
low-variance batch, `value_loss` rises and `bellman_error` grows — neither of which the floor
touches — so the regression is still detectable. (See acceptance #6 for the dedicated low-variance
collapse test.)

## Recommended design

### Recommendation: Option C — variance-floored EV + robust companion + explicit flag (all three)

Adopt the combination, not any single piece. The floor stops the artifact at the source; the
companion gives consumers a denominator-free signal to migrate gates onto; the flag keeps the
substitution auditable and lets analysts recover the raw behavior.

#### Precision invariant (applies to all of C1/C2)

EV, `value_nrmse`, and the EV-denominator variance are computed on the **fp32 rollout buffer
tensors** (`values`/`returns` are float32 at `rollout_buffer.py:279,323`). `agent.update()` —
including this block — executes under the bf16 autocast region (`policy_amp_context` at
`vectorized.py:455-456`, `helpers.py:57-76`), but autocast only down-casts matmul/conv-class ops;
pointwise reductions (`.var()`, `.std()`, `.sqrt()`) and the `denormalize()` multiply are **left in
fp32**. So the floor of `1.0` operates against a return scale of ~49–169 in fp32, where it is
numerically benign.

**Invariant the implementation MUST preserve:** do not `.half()` `values`/`returns` before the EV
block. Optionally wrap the EV/`value_nrmse` block in `torch.autocast(enabled=False)` to make the
fp32 guarantee *structural* rather than incidental — if a future change moved EV onto a bf16 tensor,
a floor of `1.0` on a return scale of ~50–170 would interact badly with bf16's ~3-decimal-digit
mantissa.

#### Correction convention (one choice across the EV family)

Pick **one** Bessel/population convention for the entire EV family and state it. The existing
`value_metrics` signals (`v_return_correlation`'s `r_std`, `bellman_error`'s `td_std`, the buffer-wide
`return_variance`) all use `correction=0` (population). The EV denominator at `ppo_agent.py:622` uses
`correction=1` (Bessel). On small/low-variance valid batches the `(n-1)/n` difference is non-trivial
and shifts where the floor binds relative to `v_return_correlation`'s own `r_std > 1e-8` guard.

**Decision:** keep the EV denominator and `value_nrmse` on `correction=1` (Bessel) — the existing,
shipped EV behavior — and document the mismatch against the `correction=0` `value_metrics` family
rather than re-baselining the shipped EV. Step 0 calibration and all test tolerances MUST be computed
with the **actual** correction used at the producing call site (`correction=1` for the EV
denominator/`value_nrmse`; `correction=0` for the `value_metrics` companions). Add a code comment at
each producing site recording its convention so implementers do not silently mix them.

#### C1. Variance-floored EV (replace the compute block, ppo_agent.py:619-631)

Introduce a principled return-variance floor and compute EV against `max(var_returns, floor)`,
emitting a flag when the floor binds:

```python
valid_values  = data["values"][valid_mask]
valid_returns = data["returns"][valid_mask]
raw_values    = self.value_normalizer.denormalize(valid_values)

residual_var = (valid_returns - raw_values).var()   # Bessel (correction=1)
var_returns  = valid_returns.var()                   # Bessel (correction=1) -- EV denominator

# Principled floor: EV's ratio is only meaningful when the target carries enough spread to
# explain. Below the floor the denominator cannot support a ratio; clamp it and flag, rather
# than emit a -8 outlier that is an artifact of dividing by ~0.
ev_var_floor = self.config.ev_return_variance_floor          # NEW config, default below
denom = torch.clamp(var_returns, min=ev_var_floor)
ev_low_return_variance = bool(var_returns < ev_var_floor)

explained_variance = 1.0 - residual_var / denom
```

- `explained_variance` is no longer special-cased to `0.0` at near-zero variance; the floor handles
  the whole low-variance band continuously (the old `1e-8` hard guard at `:624-628` is subsumed —
  delete it, do not keep both paths, per the no-legacy rule).
- `ev_low_return_variance` is a **boolean per update** indicating the floor bound (i.e. EV is
  attenuated and should be read with the scale-anchored companions).
- The `torch.clamp(var_returns, min=floor)` is a **hard clamp**: a no-op above the floor (so a
  healthy batch's EV is numerically unchanged — see acceptance #1), and a substitution only below it.

**Choosing the floor.** The floor must be on the **raw return scale** (the denominator's scale,
*before* normalization — see Risk #1 for why it is independent of the value normalizer). Two
defensible bases, to be locked during planning with a short empirical pass over recent runs:
  - **Absolute floor** keyed to observed return scale: returns in healthy runs have std 7–13
    (per the multi-epoch design doc), so `var_returns` ~ 49–169. A floor of `ev_return_variance_floor
    = 1.0` (std 1.0) excludes only batches whose return spread is <~1/7th of typical — exactly the
    pathological tail — while leaving all normal batches untouched.
  - **Relative floor** keyed to a running EMA of `var_returns` (e.g. floor = 0.05 × EMA). More
    self-tuning across reward regimes, at the cost of one more piece of state.

**Batch-size sensitivity (important).** `valid_returns.var()` is a Bessel-corrected *sample*
variance; on small valid-timestep batches the estimate itself has high sampling variance, so whether
`var_returns < 1.0` (flag fires) depends partly on the valid-N, not only the true return spread. The
"std 7–13 → var 49–169" justification is a **run-level aggregate**; per-update masked batches are
smaller and noisier, so the floor will bind *more often* than the run-level numbers suggest. Step 0
calibration MUST therefore report the joint `(var_returns, valid-N)` distribution and the flag-rate
on the genuine pathological tail vs healthy small batches, and consider whether a relative
(EMA-based) floor is more robust to N than the absolute `1.0`.

Default: **absolute floor `1.0`**, exposed as `ev_return_variance_floor` config so the relative
variant can be swapped without code change. Rationale: simplest, scale-anchored, and the run-level
return scale is stable within a training regime.

#### C2. Robust companion metric (emit alongside raw EV)

Emit a **normalized residual RMSE** as a floor-stabilized value-fit companion:

```python
value_nrmse = torch.sqrt(residual_var) / (valid_returns.std() + ev_var_floor_std)
```

where `ev_var_floor_std = sqrt(ev_return_variance_floor)`. `value_nrmse` is bounded-below at 0
(perfect fit), grows with residual magnitude, and its denominator is floored on the **std** scale so
it cannot blow up.

Two honest caveats the spec states explicitly so reviewers do not over-trust it:
  - **It is floor-stabilized, NOT scale-independent.** Unlike EV's hard clamp (a no-op above the
    floor), `value_nrmse` uses an **additive** offset (`+ ev_var_floor_std`) that biases the metric
    **downward on every batch**, by design, to keep it bounded. On a healthy batch with `return_std`
    = 10 and floor_std = 1.0 the denominator is 11, not 10 (~10% deflation). On a genuinely
    small-`returns.std()` batch the additive offset *dominates*, so `value_nrmse` ≈
    `sqrt(residual_var)/floor_std` — an unnormalized residual scaled by a constant. Gate thresholds
    for `value_nrmse` MUST be calibrated against the **floored-denominator** values (Step 0 computes
    `value_nrmse` on healthy-run batches to anchor the threshold), not against a naive `residual/std`.
  - **It shares the EV denominator's fragility on flagged batches** (it normalizes by `returns.std()`,
    the sqrt of the same quantity whose smallness causes the EV blowout). It is therefore a
    *secondary* signal, not the floor-independent ground truth. The genuinely scale-free, unfloored
    signals — `value_loss` and `bellman_error` — are the ground truth on flagged updates (see Key
    reframe).

We additionally **promote already-computed robust signals** to first-class telemetry consumers can
key on. **These already exist and are already wired into the contract** (`v_return_correlation` and
`bellman_error` are computed in `leyline/value_metrics.py`, fed from ppo_agent.py:502-537, already
written to the policy snapshot at `emitters.py:1126-1129`, already declared on `PPOUpdatePayload`
at `leyline/telemetry.py:741-744`, on `PPOUpdateMetrics` at `agent/types.py:105-108`, and already
aggregated at `ppo_metrics.py:214-217`). They are **unguarded** by the EV floor:
  - `v_return_correlation` — Pearson corr of values vs returns, clamped [−1,1], with its own
    `r_std > 1e-8` guard returning a **0.0 sentinel** on low-`r_std` batches (`value_metrics.py:77-82`).
    The honest "is the critic tracking returns at all" signal — **but only when the denominator is
    healthy**; treat 0.0 as uninformative (not "bad fit") when `ev_low_return_variance == True`.
  - `bellman_error` — mean |TD|, absolute residual magnitude, scale-anchored, **unfloored**. A primary
    ground-truth signal on flagged updates.
  - `value_loss`, `value_mean`, `value_std` — the direct low-loss-vs-crater-EV cross-check;
    `value_loss` is the other primary ground truth.

Because these three are already first-class, the routing work reduces to (a) carrying the genuinely
new fields `value_nrmse`, `ev_low_return_variance`, and the EV-denominator variance, and (b) closing
the real gap: threading the robust signals into the **gate** (they are emitted, but
`anomaly_detector.check_value_function` currently receives only `explained_variance` — see Consumer
Audit / Locked decisions).

#### C3. Explicit flag in the contract

`ev_low_return_variance` flows through the telemetry contract (leyline) so every consumer can:
  - suppress or down-weight EV mean/std aggregation on flagged updates,
  - render a "low-return-variance" badge instead of a misleading red EV gauge,
  - exclude flagged updates from `proof_blocking` confounder logic,
  - **count flagged updates as a reward/return-regime diagnostic** (see 80/20 section).

### Why a return-variance floor is legitimate numerical handling, NOT a CLAUDE.md no-bug-hiding violation

This is the load-bearing argument and the spec states it explicitly, because the codebase prohibits
defensive patterns that mask bugs.

The no-bug-hiding rule targets patterns that **suppress a symptom of a real defect** so the defect
goes unnoticed (e.g. `getattr(obj, "hallucinated_field", None)` to silence an `AttributeError` that
signals a wrong field name). The test from CLAUDE.md: *"is this defensive programming to hide a bug
that should not be possible in a well designed system, or is this legitimate type handling?"*

The variance floor passes that test as legitimate handling, on four grounds:

1. **There is no underlying value-fit bug to hide.** A small `var_returns` is not a defect — it is a
   *correct, expected* property of certain rollout batches (low reward spread within a state
   distribution; see the 80/20 caveat — it may be a *reward-regime* signal, which we surface as a
   count, not a value-fit defect). The value function is fitting fine on those batches (`value_loss`
   is low). The −8 EV is not a true value-fit signal being suppressed; it is a **numerical artifact of
   an ill-conditioned ratio** that was never informative about value fit. We are removing noise from
   the *value-fit* read, not muffling signal.

2. **The condition is surfaced, not swallowed.** A bug-hiding pattern makes the problem *disappear*.
   This design does the opposite: it emits `ev_low_return_variance=True` so the substitution is
   **observable and auditable** on every affected update, and reports the run-level flagged count as a
   first-class regime diagnostic. A consumer can count flagged updates, plot them, and gate on them.
   Swallowed exceptions have no flag; this has a first-class one.

3. **The genuinely scale-anchored cross-checks remain unguarded.** `value_loss` (MSE on the trained
   scale) and `bellman_error` (mean |TD|) are computed and emitted **without** the floor and **without
   dividing by return variance at all**. If `V(s)` genuinely regresses — including on a low-variance
   batch — both move, and no floor can mask it. (Note the honest qualification: `value_nrmse` is
   floor-stabilized and `v_return_correlation` carries its own 0.0 sentinel on low-`r_std` batches, so
   on a *flagged* update the ground truth is specifically `value_loss`/`bellman_error`, not those two.
   The floor only ever changes EV's denominator; it cannot make a broken critic look healthy because
   the unfloored, division-free signals are right there.)

4. **The floor is a standard, principled numerical guard for a ratio estimator**, in the same family
   as the existing `var_returns > 1e-8` guard (which the codebase already ships and which this design
   *replaces with a better-conditioned continuous version*), the `r_std > 1e-8` guard inside
   `v_return_correlation`, and the existing aux `clamp(min=0.01)` on the aux-EV target variance
   (`ppo_agent.py:1269`). Dividing by a near-zero estimated denominator is a known ill-conditioning,
   not a logic error. Clamping the denominator to a documented floor is the textbook handling — it is
   closer to "PyTorch tensor operations" / "numeric field type guards" in the CLAUDE.md
   legitimate-uses list than to defensive bug-masking.

What would *violate* the rule, and which this design explicitly does **not** do:
  - Silently clamping EV to `[0, 1]` to make negative values vanish (that hides genuine bad fits).
    *(Note: `HealthGauges.vue:86` already does this clamp for display — the audit flags it as a real
    masking site to fix, see below.)*
  - `try/except` around the EV compute returning a default on any error.
  - Suppressing the flag, or computing the scale-anchored companions (`value_loss`, `bellman_error`)
    with the same floored denominator so the cross-check is also blinded.

## Rejected alternatives

- **Option A — winsorize/clamp run-level EV mean/std only (no source fix).** Trim EV outliers at the
  aggregation/view layer. Rejected: it treats the symptom downstream of N consumers, requires the same
  fix replicated in karn views, overwatch panels, wandb, and the TUI, and discards the per-update
  information about *which* updates were ill-conditioned. The floor at the source is one change; this
  is many, and is closer to bug-masking (it makes outliers vanish without recording why).

- **Option B — drop raw EV entirely, emit only `v_return_correlation`.** Rejected on two counts:
  (1) EV is the field every existing gate, view, panel, and the wandb dashboard reads; removing it is a
  far larger blast radius than flooring it, and the no-legacy rule means we'd have to rip out and
  rewrite all consumers in one commit. (2) EV remains genuinely useful when the denominator is healthy
  (the median EV +0.239 result is real and informative). We want EV honest, not gone.

- **Option D — post-update EV.** Recompute EV after the K epochs against the updated value function.
  Rejected: the pre-update timing is intentional and standard (ppo_agent.py:612-615 comment) — it
  measures the value function that *generated the rollout*, which is what advantage quality depends on.
  Post-update EV measures the new critic against stale returns and is less meaningful. This also does
  nothing about the denominator artifact, which is timing-independent.

- **Option E — raise the existing `1e-8` guard to a larger constant, keep the `→ 0.0` special case.**
  Rejected: emitting `0.0` for any low-variance batch injects a *fake "average" reading* into the
  mean/std, which is its own form of masking (0.0 looks like "mediocre fit" rather than "not enough
  variance to judge"). The flag + floored-ratio continuous form is strictly more honest than a sentinel.

## Locked decisions

1. EV remains emitted; it is **floored, not removed, not post-update**.
2. Floor lives at the **source** (`ppo_agent.py` compute), not at view/panel/aggregation layers.
3. Default floor = **absolute, `ev_return_variance_floor = 1.0`** (raw return-variance scale), config-exposed.
4. The old `var_returns > 1e-8 → 0.0` special case (ppo_agent.py:624-628) is **deleted** (subsumed by
   the floor; no dual path).
5. New companion metric = **`value_nrmse`** (floor-*stabilized*, additive-offset normalized residual
   RMSE). It is a *secondary* signal; the scale-anchored ground truth on flagged updates is
   `value_loss` + `bellman_error`.
6. New per-update boolean = **`ev_low_return_variance`**, carried through leyline to all consumers; its
   run-level count (`ev_low_return_variance_count`) is a first-class reward/return-regime diagnostic.
7. **Correction convention:** the EV denominator and `value_nrmse` stay on `correction=1` (Bessel,
   the shipped EV behavior); the `value_metrics` companions stay on `correction=0`. The mismatch is
   documented at each producing site; Step 0 calibration and all test tolerances use the actual
   correction per site.
8. **Precision invariant:** EV / `value_nrmse` / EV-denominator variance run on fp32 buffer tensors;
   do not `.half()` values/returns before the EV block; optionally wrap the block in
   `torch.autocast(enabled=False)` to make it structural.
9. The EV-denominator variance is emitted under a **distinct** name (decision: `ev_return_variance`)
   — it is `valid_returns.var()` (valid-masked, Bessel) and MUST NOT reuse the existing buffer-wide
   `return_variance` key (`correction=0`, full buffer).
10. Gates migrate off raw-EV mean/std onto `value_loss` + `bellman_error` (primary, scale-anchored) and
    `value_nrmse` (secondary); `v_return_correlation` is used only when `ev_low_return_variance == False`;
    EV itself is used only when `ev_low_return_variance == False`.
11. **`HealthGauges.vue` EV display** is treated as a two-case display per the no-bug-hiding rule:
    (a) EV genuinely negative AND **not** flagged → show the raw negative value with a critical status,
    **do not clamp**; (b) **flagged** → replace the gauge with a LOW-VAR badge citing `value_nrmse`,
    hide the percentage. The existing `:86` clamp `Math.max(0, Math.min(1, ...))` is deleted.
12. **Aux EV:** floored + flagged with the **same mechanism but a SEPARATE constant** — the aux floor
    is on the contribution-prediction-target scale (currently `clamp(min=0.01)` at `ppo_agent.py:1269`,
    ~100× smaller than the return-scale floor of `1.0`). Reusing `1.0` on the aux target would massively
    over-attenuate aux EV. An empirically-calibrated `aux_ev_var_floor` (keep ~0.01 or recalibrate on
    the aux-target scale) is required; the return-scale `1.0` MUST NOT be reused.

## Telemetry contract changes (leyline)

All new fields land in `leyline` per the shared-contracts rule. Concretely:

- **`src/esper/leyline/telemetry.py`** — `PPOUpdatePayload` (current EV field `explained_variance` at
  `telemetry.py:719`) gains the **genuinely new** fields:
  - `value_nrmse: float | None` (NaN-tolerant, same convention as `explained_variance`).
  - `ev_low_return_variance: bool` (defaults False; per-update flag).
  - `ev_return_variance: float | None` (the EV-denominator variance: `valid_returns.var()`,
    valid-masked, **Bessel correction=1**). **Distinct from** the existing `return_variance` field
    (`telemetry.py:748`), which is the buffer-wide population variance (`correction=0`) from
    `compute_value_function_metrics`. Do **not** add a second `return_variance`; the two are
    semantically different (different population AND different correction), and emitting both under one
    name would silently corrupt the contract and violate the no-legacy rule. The distinct name makes
    the floor decision reconstructable from telemetry without conflating the two variances.

  **Already present — no routing work needed:** `v_return_correlation` (`telemetry.py:741`),
  `bellman_error` (`telemetry.py:744`), and `return_variance` (`telemetry.py:748`) already exist on
  `PPOUpdatePayload` and are already populated (emitter at `emitters.py:1126-1129`). The earlier
  framing of "route these into the payload" is obsolete — they are first-class. The real gap is the
  gate threading below.

- **`src/esper/simic/agent/types.py`** — `PPOUpdateMetrics` (EV field `explained_variance`) gains the
  parallel new `value_nrmse`, `ev_low_return_variance`, `ev_return_variance` fields.
  `v_return_correlation` / `bellman_error` are already declared (`agent/types.py:105-108`).

- **`src/esper/simic/agent/ppo_metrics.py`** — aggregation extends to aggregate the new fields.
  `v_return_correlation` / `bellman_error` are already aggregated (`ppo_metrics.py:214-217`).
  **Aggregation rule:** run-level EV mean/std must be computed over updates with
  `ev_low_return_variance == False`; flagged updates are excluded from the EV mean/std, and
  `ev_low_return_variance_count` is reported as a first-class regime diagnostic. `value_nrmse` /
  `v_return_correlation` / `bellman_error` aggregate over all updates. **Granularity callout:** EV and
  `ev_low_return_variance` are computed **once per `update()` invocation** (pre-epoch-loop,
  ppo_agent.py:612-631), producing a single reading per update — NOT one per K internal epoch. The
  flagged-update exclusion operates over the list of these per-update readings across the run
  (`PPOMetricsBuilder` finalize treats `metrics['explained_variance']` as a per-update singleton). If
  EV is ever moved inside the epoch loop, the aggregation semantics must be revisited.
  **Run-level std callout:** the in-process `_aggregate_ppo_metrics` (`vectorized.py:369-375`) computes
  only a **mean** for EV, not a std. The EV std in the empirical table is a downstream Karn DuckDB /
  aggregator statistic over emitted per-update rows. The flagged-update std/mean *exclusion* therefore
  must be enforced at the Karn/aggregator (SQL) layer too — the hot path only carries the flag through.

- **`src/esper/simic/telemetry/emitters.py`** — wiring at `emitters.py:986` (EV →
  `PPOUpdatePayload`) and `emitters.py:472` (EV → `AnalyticsSnapshotPayload.value_variance`). The
  latter is a **mislabel bug**: `AnalyticsSnapshotPayload.value_variance` is being fed EV
  (`value_variance=metrics.get('explained_variance', 0.0)`), not value variance, surfacing as
  `batch_stats.value_variance` in karn. Fix the mislabel as part of this change (no-legacy rule — do
  not keep the wrong name): emit EV under an EV-named field and emit the actual value variance (or the
  new `ev_return_variance`) under the value-variance field.

- **Karn DuckDB schema/views** — `src/esper/karn/mcp/views.py` `ppo_updates` (EV column at views.py:127)
  gains `value_nrmse`, `ev_low_return_variance`, `ev_return_variance` columns; the EV mean/std rollup in
  Karn views must filter on `ev_low_return_variance == False`; `run_confounders` (views.py:606-662) and
  `anomalies` (views.py:583-605) updated per the audit below.

- **Overwatch/Sanctum web types** — `sanctum types.ts:316,405` and the persistence/schema path
  (`aggregator.py:979-983`, `schema.py:969,1108`, `snapshot_copy.py:103`) carry the new fields.
  `VALUE_HEAD_SCHEMA_VERSION` is **not** bumped by this change (no checkpoint format change); only the
  telemetry payload schema grows additively.

## Consumer audit — every gate/alert/display that keys on `explained_variance`

This is a first-class deliverable: any GATE or ALERT keyed on EV mean/std will false-alarm on K>1 runs
post-P0-1. Enumerated with file:line, classified GATE/ALERT (has teeth) vs DISPLAY (misleads only).

### Producers / contracts
| Location | Role |
|----------|------|
| `src/esper/simic/agent/ppo_agent.py:619-631` | EV compute (the source fix lands here) |
| `src/esper/simic/telemetry/emitters.py:986` | EV → `PPOUpdatePayload.explained_variance` (→ `ppo_updates`) |
| `src/esper/simic/telemetry/emitters.py:472` | EV → `AnalyticsSnapshotPayload.value_variance` — **MISLABELLED**; EV under a value-variance name (→ `batch_stats.value_variance`). Fix in this change. |
| `src/esper/simic/telemetry/emitters.py:1126-1129` | `v_return_correlation` / `bellman_error` → `PPOUpdatePayload` — **already wired** (no routing work) |
| `src/esper/leyline/telemetry.py:701` (`PPOUpdatePayload`), `:719` (`explained_variance`), `:741` (`v_return_correlation`), `:744` (`bellman_error`), `:748` (`return_variance`, correction=0) | Contract surface — extend with new fields (`value_nrmse`, `ev_low_return_variance`, `ev_return_variance`) |
| `src/esper/simic/training/ppo_coordinator.py:512-524` (EV passed at `:516`, guard at `:523`) | passes EV into the gate — **the gate-threading gap lives here** |
| `src/esper/simic/agent/types.py` `PPOUpdateMetrics` (`explained_variance`; `v_return_correlation`/`bellman_error` at `:105-108`) | metrics struct |
| `src/esper/simic/agent/ppo_metrics.py` (EV aggregation; `v_return_correlation`/`bellman_error` at `:214-217`) | apply flagged-update exclusion + report `ev_low_return_variance_count` |

### GATES / ALERTS — act on the value (THESE HAVE TEETH)
| Location | Behavior | Post-P0-1 risk | Required change |
|----------|----------|----------------|-----------------|
| `src/esper/simic/training/ppo_coordinator.py:512-524` → `simic/telemetry/anomaly_detector.py:140-175` `check_value_function` (via the `check_all`-style aggregator method ~`:455`) | Phase-dependent EV thresholds (anomaly_detector.py:66-69: −0.5 / −0.2 / 0.0 / +0.1). Fires `value_collapse` anomaly. Guarded by `value_collapse_applicable = usable_actor_timesteps > 0` (ppo_coordinator.py:523). **NON-HALTING** — vectorized.py:480-493 only raises telemetry verbosity; vectorized.py:517 → `VALUE_COLLAPSE_DETECTED`. | A floored-EV outlier on a low-variance batch trips the negative thresholds spuriously. | **Gate-threading change (the true gap):** `ppo_coordinator.py:516` currently passes ONLY `explained_variance`. Thread `value_loss`, `bellman_error`, `value_nrmse`, `v_return_correlation`, and `ev_low_return_variance` through the `check_all`-style aggregator (~`:455`) down into `check_value_function` (`:140-175`); update both signatures. Gate on `value_loss`/`bellman_error` (scale-anchored, primary) + `value_nrmse` (secondary); use `v_return_correlation` only when NOT flagged; **skip the EV threshold arm where `ev_low_return_variance == True`**. Done wrong this is a runtime `KeyError`, not a unit-test failure — see acceptance #5. |
| `src/esper/karn/mcp/views.py:606-662` `run_confounders` (event at views.py:652; `true as proof_blocking` at views.py:649) | A `VALUE_COLLAPSE_DETECTED` row marks the run as proof-blocking. | **Highest teeth:** a K>1 EV blowout marks proof confounders and **invalidates the reward-efficiency experiment verdict**. | Only emit `VALUE_COLLAPSE_DETECTED` when the scale-anchored robust signal confirms (or when not `ev_low_return_variance`); driven by the gate fix above. |
| `src/esper/karn/mcp/views.py:583-605` `anomalies` | Surfaces `VALUE_COLLAPSE_DETECTED` (views.py:596). | Spurious anomaly rows on K>1. | Inherits the gate fix (fewer spurious events upstream). |
| `reward_health.py:39-41` `is_ev_healthy = ev_explained > 0.5` (fed by `aggregator.py:2085-2093`) | TUI health boolean. Display-but-**evaluative**. | Flips unhealthy on artefactual low EV. | Base `is_ev_healthy` on `value_nrmse` threshold (or EV only when not flagged). |
| `ExperimentVerdictPanel.vue:127-138` | `policyGate` → `'watch'` when `explained_variance < 0.3`. | Spurious 'watch' on K>1. | Use `value_nrmse` / `v_return_correlation`; ignore EV on flagged updates. |

### DISPLAY ONLY — charts/styling (mislead, but do not gate logic)
| Location | Note |
|----------|------|
| `src/esper/karn/mcp/views.py:127` | `ppo_updates.explained_variance` chartable column — add `value_nrmse`, flag, `ev_return_variance` columns alongside |
| `constants.py:120-121` | `EXPLAINED_VAR_WARNING=0.3` / `EXPLAINED_VAR_CRITICAL=0.0`, consumed by: `ppo_losses_panel.py:110-113, 355-358` (`_get_ev_status`); `critic_calibration_panel.py:62-64, 141-144` (`_get_ev_style`); `narrative_panel.py:320-321, 457, 500-514, 601, 624` |
| `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:21-22` (GOOD 0.5 / WARNING 0.3); **:86 clamps EV to [0,1]** | The clamp **masks the blowout in the gauge** (negative EV renders as 0). Per no-bug-hiding rule and Locked decision 11, replace with the two-case display (raw negative + critical status when not flagged; LOW-VAR badge citing `value_nrmse` when flagged). `:91-92` status |
| `aggregator.py:979-983` (persist + history); `schema.py:969,1108`; `snapshot_copy.py:103`; `sanctum types.ts:316,405` | Carry new fields |
| `src/esper/nissa/wandb_backend.py:389-390` | `ppo/explained_variance` — add `ppo/value_nrmse`, `ppo/ev_low_return_variance` |
| `vectorized.py:193` | EV registered in PPO metric set — register new metrics |

### AUX EV — same artifact family, separate metric, SEPARATE floor constant
| Location | Note |
|----------|------|
| `src/esper/simic/agent/ppo_agent.py:1268-1271` `aux_explained_variance = 1 - residual_var/target_var` with `target_var = target.var().clamp(min=0.01)` (auxiliary value head, **contribution-prediction-target scale**); emitted 1285/1314/1320/1328; `types.py:189`; `vectorized.py:241` | **Identical denominator vulnerability**, no gate located (display only). Apply the same *mechanism* (floor + flag) but a **SEPARATE, empirically-calibrated `aux_ev_var_floor`** on the aux-target scale (~0.01, ~100× smaller than the return-scale `1.0`). Reusing `1.0` would over-attenuate aux EV — see Locked decision 12. |

### Explicitly OUT OF SCOPE — do NOT touch (different signal, same name)
| Location | Why excluded |
|----------|--------------|
| `karn/triggers.py:247-259` `PolicyAnomalyDetector.check_value_collapse(value_std)` (`PolicyThresholds.VALUE_STD_COLLAPSE`) | Keys on **value_std**, not EV. Same `value_collapse` name, unrelated signal — unaffected by the EV artifact. |
| `store.py:354` `value_collapse: bool` | Also **value_std-derived**, not EV. |

## Risks

1. **Floor mis-set.** Too high a floor attenuates legitimate negative-EV signals (genuinely bad fits on
   moderate-variance batches); too low and the artifact persists. The floor is on the **raw return
   scale** (`var_returns` of `valid_returns` **before normalization**) and is therefore independent of
   the value normalizer's running scale: a stale running normalizer in early training degrades EV
   *quality* through the `denormalize()` path (it inflates the numerator `Var(returns − raw_values)`)
   but does **not** affect `var_returns` or the floor's binding decision. (Implementers MUST NOT put
   the floor on the normalized scale.) Value warmup similarly affects only the numerator, not the
   denominator. Mitigation: default `1.0` is well below typical `var_returns` (49–169); validate with
   the Step 0 empirical pass (joint `(var_returns, valid-N)` distribution, flag-rate on the
   pathological tail vs healthy small batches); the floor is config-exposed so it can be retuned
   without a code change. The flag + unfloored, division-free companions (`value_loss`,
   `bellman_error`) mean a mis-set floor degrades gracefully — it never hides a true regression.
2. **Gate migration regressions.** Changing what `run_confounders` / `check_value_function` key on could
   *mask* a real value collapse if the companion threshold is set wrong. Mitigation: on flagged updates,
   gates require a bad **scale-anchored** signal (`value_loss`/`bellman_error`) — not `value_nrmse`
   (floored) or `v_return_correlation` (0.0 sentinel) — i.e. the change makes gates *stricter about
   evidence*, not blinder. drl-expert review required on the thresholds.
3. **Mislabel fix (emitters.py:472) ripples** into `batch_stats.value_variance` consumers. Mitigation:
   audit `batch_stats.value_variance` readers before renaming; this is a correctness fix (the field never
   held value variance) but the rename touches the karn view + any panel reading it.
4. **Contract growth across many surfaces** (leyline → types → metrics → emitters → karn → overwatch →
   wandb). Mitigation: additive fields only, NaN/False-tolerant defaults, no `VALUE_HEAD_SCHEMA_VERSION`
   bump (telemetry payload, not checkpoint format).
5. **Sustained low return variance is itself a signal, not only noise (80/20).** A *persistently* high
   `ev_low_return_variance` fraction is a reward/return-regime alert (Phase 1/2): sparse/saturated
   reward, short effective horizon, or return-scale collapse — distinct from value-fit (Phase 3).
   Mitigation: surface `ev_low_return_variance_count` as a first-class regime diagnostic with a rough
   "investigate if persistently high" trigger (e.g. >20% of updates flagged), so flooring EV does not
   silence a real reward-structure signal.
6. **Late-training convergence raises the flag rate legitimately.** Entropy anneal collapses the policy
   toward deterministic actions, reducing trajectory diversity and batch return variance, so
   `ev_low_return_variance` will flag *more often* late in training even with a healthy critic. This is
   expected, not pathological. Gates should treat a rising flag rate **alongside improving `value_nrmse`
   / `v_return_correlation` / falling `bellman_error`** as a convergence signal, not a regression. The
   `ev_low_return_variance_count` longitudinal view distinguishes this benign rise from the sustained
   regime alert in Risk #5.

## Acceptance criteria

1. **EV is floored at the source.** `ppo_agent.py` EV compute divides by `max(var_returns,
   ev_return_variance_floor)`; the old `var_returns > 1e-8 → 0.0` special case (`:624-628`) is deleted
   (single code path). Unit test: a synthetic low-`var_returns` batch that previously produced EV ≪ −1
   now produces a bounded EV and `ev_low_return_variance == True`; a healthy-variance batch is
   numerically unchanged (EV within tolerance of the pre-floor value — the clamp is a no-op above the
   floor — flag False). Tolerances computed with `correction=1` (the actual EV-denominator correction).
2. **Robust companion emitted.** `value_nrmse` is computed and emitted on every update; on the same
   low-variance batch where EV would have blown out, `value_nrmse` is finite and small (reflecting the
   low `value_loss`). Test asserts `value_nrmse` is bounded and monotone in injected residual magnitude,
   and that thresholds are anchored against floored-denominator values (not naive `residual/std`).
3. **Flag plumbed end-to-end.** `ev_low_return_variance`, `value_nrmse`, and `ev_return_variance` reach
   `PPOUpdatePayload` (leyline `telemetry.py:701`), `PPOUpdateMetrics`, `ppo_metrics` aggregation, the
   karn `ppo_updates` view, and the wandb backend. Test reads them back from each contract surface. (The
   already-present `v_return_correlation` / `bellman_error` are NOT re-routed — assert they remain
   first-class, do not assert "emitter only carries explained_variance", which is already false.)
4. **Aggregation excludes flagged updates from EV mean/std.** `ppo_metrics` run-level EV mean/std is
   computed only over `ev_low_return_variance == False` updates, and `ev_low_return_variance_count` is
   reported. Test: a run with N `update()` calls, some flagged, yields an EV mean/std equal to the
   unflagged-only computation, plus a correct flagged count. (Karn-layer std/mean rollup exclusion is
   exercised separately at the view level.)
5. **Gates migrated and no longer false-alarm on the artifact — via the full coordinator path.**
   `check_value_function` / `run_confounders` do not emit `VALUE_COLLAPSE_DETECTED` / `proof_blocking`
   for an update whose only symptom is artefactual low EV (`ev_low_return_variance == True` with healthy
   `value_loss` / `bellman_error`). The test exercises the **full `ppo_coordinator.py` call path** (not
   `anomaly_detector` in isolation) so argument-threading bugs surface as a test failure, not a
   training-time `KeyError`.
6. **Genuine value collapse still fires — including the low-variance co-occurrence case.** Two
   sub-tests: (a) healthy `var_returns` + large residual (EV genuinely negative, flag False) still trips
   `check_value_function` → `VALUE_COLLAPSE_DETECTED` / `proof_blocking`; (b) **LOW `var_returns` AND a
   genuine bad fit** (flag True, low `value_loss`-but-no: high `value_loss` + high `bellman_error`) still
   fires `value_collapse` via the scale-anchored robust arm, proving the floored/flagged path does not
   blind genuine collapse on low-variance batches.
7. **Display masking fixed (two cases).** `HealthGauges.vue` no longer clamps EV to `[0,1]`: when not
   flagged and EV is negative it renders the raw negative value with a critical status; when flagged it
   renders a LOW-VAR badge citing `value_nrmse` and hides the percentage. New `HealthGauges.spec.ts`
   cases cover negative-unflagged and flagged EV.
8. **Mislabel corrected.** `emitters.py:472` no longer writes EV into `AnalyticsSnapshotPayload.value_variance`;
   EV is emitted under an EV-named field and the value-variance field carries actual value/return variance.
9. **Aux EV consistent — separate constant.** `aux_explained_variance` (ppo_agent.py:1268-1271) is
   floored and flagged with the same *mechanism* but a **distinct `aux_ev_var_floor`** on the
   aux-target scale; the return-scale `1.0` is NOT reused. Test asserts the aux floor is the aux
   constant, not `ev_return_variance_floor`.

## Files touched

- `src/esper/simic/agent/ppo_agent.py` — EV compute (619-631), aux EV (1268-1271), config reads for both floors
- `src/esper/leyline/telemetry.py` — `PPOUpdatePayload` new fields (around `:719`/`:741-748`)
- `src/esper/simic/agent/types.py` — `PPOUpdateMetrics` new fields (`v_return_correlation`/`bellman_error` already at `:105-108`)
- `src/esper/simic/agent/ppo_metrics.py` — aggregation + flagged-exclusion rule + `ev_low_return_variance_count` (`v_return_correlation`/`bellman_error` already at `:214-217`)
- `src/esper/simic/telemetry/emitters.py` — wiring (986) + mislabel fix (472); `:1126-1129` already wired
- `src/esper/simic/training/ppo_coordinator.py` — gate-threading change (512-524, `:516`)
- `src/esper/simic/telemetry/anomaly_detector.py` — `check_value_function` migration + `check_all`-style aggregator signature (66-69, ~455, 140-175)
- `src/esper/simic/training/vectorized.py` — metric registration (193, 241), event mapping (480-493, 517), autocast wrap context (455-456)
- `src/esper/karn/mcp/views.py` — `ppo_updates` (127), `run_confounders` (606-662), `anomalies` (583-605), flagged-update std/mean exclusion at rollup
- `src/esper/.../reward_health.py` — `is_ev_healthy` (39-41) + `aggregator.py` feed (2085-2093, 979-983)
- `src/esper/.../constants.py` — EV thresholds (120-121) review (display only; likely unchanged)
- Overwatch web: `ExperimentVerdictPanel.vue` (127-138), `src/esper/karn/overwatch/web/src/components/HealthGauges.vue` (21-22, delete `:86` clamp, 91-92) + `HealthGauges.spec.ts` cases, `sanctum types.ts` (316, 405)
- `src/esper/nissa/wandb_backend.py` — new metrics (389-390)
- Config surface for `ev_return_variance_floor` (default 1.0) AND `aux_ev_var_floor` (aux-target scale; default ~0.01) — PPO/agent config
- Tests under `tests/simic/` (EV compute, aggregation, full-coordinator gate path, aux floor) + karn view tests + `HealthGauges.spec.ts`

## Empirical context

P0-1 A/B (K=4 arm): EV mean −0.337 / std 2.264 (control +0.572 / 0.572) co-occurring with the **lowest**
`value_loss` (0.099), the **best** median EV (+0.239), the widest K=4−K1 EV gap (+0.201 vs control +0.137),
and the op-head-starvation fix (`head_op_grad_norm` late-alive frac 0.05 → 0.89). Healthy-run return std
is 7–13 (`var_returns` ~ 49–169) per `docs/superpowers/specs/2026-06-17-recurrent-ppo-multiepoch-design.md`,
which anchors the default floor (`ev_return_variance_floor = 1.0`) well below the normal operating band —
but note this is a **run-level aggregate**; per-update masked batches are smaller/noisier, so Step 0 must
report the joint `(var_returns, valid-N)` distribution before locking the floor. P0-1 commits: 6a27b8e3
(core), 177a53aa (telemetry), 6d97391b (sanctum), 31bf8cb7 (GAE-bootstrap subset truncation).
Checkpoint-breaking via `VALUE_HEAD_SCHEMA_VERSION=2`; **this telemetry change does not bump that version**
(additive payload schema only).
```
