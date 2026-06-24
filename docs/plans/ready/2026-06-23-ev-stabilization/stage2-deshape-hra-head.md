# Plan Metadata
id: ev-stabilization-stage2-deshape-hra-head
title: "Stage 2 — De-shape the value target via an HRA sum-of-heads decomposition"
type: ready
created: 2026-06-23
updated: 2026-06-24
owner: Claude (ralph-loop)

urgency: high
value: >
  Stop the high-CoV counterfactual contribution stream from corrupting the single
  critic's value fit. EV-liftoff is achieved (median ~0.31) but VOLATILE; the leading
  hypothesis is value-target noise from the dense counterfactual term sitting inside
  the one return the critic regresses on. Routing it to its own (head-only) value
  head, with its own normalizer, is hypothesized to lift and stabilize EV_main while
  leaving the policy-gradient objective STRUCTURE unchanged (one GAE on `V_total`) and
  adding NO new gradient to the LSTM trunk (head-only `V_cf`). The trunk's `V_main`
  value gradient intentionally retargets from the total return to `R_main` on the ON
  leg — the de-shaping mechanism, gated empirically (§6), NOT a trunk-invariance claim.

complexity: M
risk: medium
risk_notes: >
  Behavioural — adds a second value head, a second VALUE normalizer, one extra
  per-stream return recursion, and a second value-loss term. Mitigated by (a) the
  policy advantage staying on the EXISTING single-GAE path; (b) V_cf being a
  HEAD-ONLY critic (gradient detached from the LSTM trunk, exactly like q_head), so
  V_cf adds NO new gradient to the trunk — the trunk's only value gradient is still
  V_main's; the OFF leg is byte-for-byte today's (OFF-leg golden), while on the ON leg
  V_main's target retargets to R_main by design (intended de-shaping, gated by §6, NOT
  a trunk-invariance proof); (c) a subtractive, exhaustive-by-
  construction reward partition normalized at STEP TIME with a NEW no-clip method
  (clip stays a total-only property); (d) default-OFF TRUE conditional construction
  (no cf head/normalizer built when OFF — OFF leg is the current single-head path
  verbatim); (e) Stage-0 gates + attribution-fidelity gates (EV_cf floor, lifecycle
  parity); (f) fresh-init paired A/B. No env-reward change; no new reward mode.

depends_on:
  - ev-stabilization-stage0-instrumentation   # per-component arrays, per-stream EV, CoV of the EXACT subtractive R_main; its buffer-storage design must be RESOLVED first
  - ev-stabilization-stage1-marginal-v-verification   # the op-independent V(s) this head pair extends (P0-1, landed)
soft_depends:
  - recurrent-ppo-multiepoch   # anchored reference pass (landed)
blocks:
  - ev-stabilization-stage3-coma-experimental

status_notes: >
  v8 (2026-06-24) — GREEN / CLEARED FOR IMPLEMENTATION (supervised session). The final
  independent confirmation panel on v7 returned 3 GO / 2 NO-GO on ONE shared medium-
  completeness blocker (reward-fn F1 + pytorch PT-1, high-confidence): the network
  `evaluate_actions` 6→7-field NamedTuple change breaks every positional 6-unpack (incl. on
  the default-OFF leg, since the return changes unconditionally), but the plan named only
  `lstm_bundle.py:243`. v8 fixes it: §4.1 now carries the EXHAUSTIVE, grep-verified
  breaking-arity sweep (18 test sites + the 1 src consumer) with attribute-access migration,
  and the distinction that `policy.evaluate_actions` (EvalResult dataclass) is unaffected;
  §7 mandates running the full tests/simic + tests/tamiyo suites green. This blocker was a
  deterministic touch-list completeness gap (not a design issue) and the fix was
  self-verified against the live grep, so it does not require another full adversarial panel.
  The other v7 fixes (B1 §2 honesty ledger, B2 torch.compile) were CONFIRMED CORRECT by the
  panel. NO remaining critical/high/medium correctness/completeness defects. (Note: the
  panel synthesizer hit a session usage limit; verdict synthesized by the supervising
  engineer from the 5 completed reviewers.) ---
  v7 (2026-06-24) — INDEPENDENT SUPERVISED REVIEW (separate session; ralph-loop stopped,
  sole ownership taken to end the two-writer race on this uncommitted doc). An independent
  6-reviewer panel on v5/v6 (the loop's self-cleared version) found ONE genuine medium
  correctness defect the loop's 5-round self-review shared a blind spot on: §2's honesty
  ledger marked "trunk feature dynamics unchanged" as PROVABLE (and risk_notes/value echoed
  "byte-for-byte today's"), but V_main keeps FULL trunk gradient (factored_lstm.py:606) and
  its target moves returns_total→returns_main on the ON leg, so trunk dynamics intentionally
  CHANGE (it is the de-shaping mechanism). v7 fixes that ledger (§2 split into the true
  provable "V_cf adds no trunk gradient" + an honest "changed-by-design, gated" row;
  risk_notes (b) + value field corrected) and resolves the §4.6 torch.compile hedge against
  the live tree (compile is invoked WITHOUT fullgraph at lstm_bundle.py:493 → the in-forward
  conditional is fine as written). Folded low-severity implementer guards (typed _EvalOutput
  NamedTuple + unpack rewrite; ppo_agent.py:600-603 caller kwarg; train.py arg→config
  override). FINAL independent confirmation panel pending. ---
  CLEARED FOR EXECUTION (2026-06-24, v6). Five review rounds (33 specialist passes
  total). Round-5 confirmation panel (6 reviewers): reward-fn APPROVED (the lone R4
  needs-revision, q-aux major, now discharged), yzmir-pytorch APPROVED (0 findings),
  reality-check APPROVED, drl-expert / axiom-python / morphogenesis
  APPROVED-WITH-CHANGES (minors/nits only). ZERO blockers, ZERO majors across the
  panel. v6 folded the residual R5 minors/nits: §4.4 GAE-denorm normalizer routing
  (value_main_normalizer denorms V_main, NOT the total), q-aux fidelity structural
  gate (§6), EV-path denorm normalizers, q-aux ON-leg phrasing, NotRequired TypedDict,
  compile fullgraph-kwarg phrasing, spot-check telemetry surface. Severity ceiling per
  round: blocker→blocker→blocker→major→cleared. IMPLEMENTATION GATED on Stage 0
  (empirical Cov(R_cf,R)/Var(R)>0.40 + low R_main CoV) + its buffer-storage design
  decision — both genuinely empirical/unresolved and out of scope for these wiring
  reviews. Filigree esper-lite-2a4b56e719.
percent_complete: 0

# Final verdicts are from the Round-5 confirmation panel (2026-06-24). Earlier-round
# verdicts (all needs-revision) drove the v1→v6 rewrites and are summarised in the
# 00-INDEX review matrix + status_notes. CLEARED: all final verdicts are approved /
# approved-with-changes; zero blockers, zero majors.
reviewed_by:
  - reviewer: yzmir-deep-rl:reward-function-reviewer
    rounds: [1, 2, 3, 4, 5]
    date: 2026-06-24
    verdict: approved
    notes: Owned the R4 q-aux major; R5 confirmed v5 discharges it (2 prose nits, folded into v6). Partition re-verified exhaustive/double-count-free.
  - reviewer: drl-expert
    rounds: [1, 2, 3, 4, 5]
    date: 2026-06-24
    verdict: approved-with-changes
    notes: R5 "GO". §4.4 GAE-denorm-normalizer minor + EV-path nit folded into v6. Head-only V_cf + single-GAE-on-V_total + cf-bootstrap-at-truncation all RL-verified.
  - reviewer: yzmir-pytorch-engineering:pytorch-code-reviewer
    rounds: [2, 5]
    date: 2026-06-24
    verdict: approved
    notes: R5 zero findings. Compile note + 3-normalizer checkpoint + _init_weights/_ForwardOutput all confirmed.
  - reviewer: pytorch-expert
    round: 3
    date: 2026-06-24
    verdict: needs-revision
    notes: R3 (normalize_only collision; get_action path; per-stream targets; returns-line V_total) → all fixed in v4; re-verified by yzmir-pytorch in R5.
  - reviewer: axiom-python-engineering:python-code-reviewer
    rounds: [1, 3, 4, 5]
    date: 2026-06-24
    verdict: approved-with-changes
    notes: R5 confirmed _init_weights + _ForwardOutput fixes; NotRequired-TypedDict + compile-phrasing nits folded into v6.
  - reviewer: axiom-planning:plan-review-reality
    rounds: [1, 2, 3, 4, 5]
    date: 2026-06-24
    verdict: approved
    notes: R5 all R4 path/citation fixes verified; one inconsequential ~line-range nit. ~50+ citations confirmed against live code.
  - reviewer: yzmir-morphogenetic-rl:morphogenesis-reviewer
    rounds: [1, 2, 3, 4, 5]
    date: 2026-06-24
    verdict: approved-with-changes
    notes: R5 confirmed EV_cf calibration + warm-up spot-check; q-aux fidelity structural gate + spot-check telemetry surface folded into v6.
  - reviewer: axiom-planning:plan-review-architecture
    round: 2
    date: 2026-06-24
    verdict: needs-revision
    notes: R2 dead-head bypass / two versions / splitter placement → fixed in v3 (conditional construction). Not re-run (structural input fully incorporated).
  - reviewer: axiom-planning:plan-review-systems
    round: 2
    date: 2026-06-24
    verdict: needs-revision
    notes: R2 Stage-0 gating / IDR cold-start / EV_cf floor → fixed in v3. Not re-run (structural input fully incorporated).

---

# Stage 2 — De-shape the value target via an HRA sum-of-heads decomposition

## 0. TL;DR for the implementer

Add a **second value head** `cf_value_head` alongside the op-independent
`state_value_head` (read as `V_main`), **only when the flag is ON**, as a
**HEAD-ONLY critic**: its gradient is detached from the LSTM trunk (exactly like
`q_head`/`contribution_predictor`), so the trunk is shaped only by `V_main` + the
policy — as it is today.

- `V_main(s)` regresses on the de-shaped return built from `R_main = reward_raw − R_cf`.
- `V_cf(s)` regresses on the counterfactual return built from `R_cf = bounded_attribution`.

The policy baseline is `V_total = V_main + V_cf`, and **the policy advantage is the
existing single GAE on `V_total`** — no second advantage, no per-stream policy
gradient. The only new return is `returns_cf` (one extra recursion);
`returns_main = returns_total − returns_cf` (exact, GAE linearity).

Why it should help (HYPOTHESIS, gated by Stage 0 + the A/B): today one head + one
normalizer fits `R_main + R_cf`, whose scale/variance are set by the noisy `R_cf`
(CoV ~1.9–2.7). Its own head + normalizer lets `V_main` fit the lower-variance stream,
so `EV_main` should lift and *stabilize*.

Gate behind `TrainingConfig.hra_value_decomposition` (default `False`; OFF ⇒ no cf
head built ⇒ the current single-head path runs verbatim). A/B = paired **fresh-init**
runs (ON vs OFF), same seeds, via Stage-0 per-stream EV.

## 1. Corrected premise + Stage-0 gate (read first)

P0-1 landed (`6a27b8e3`): the PPO baseline is a directly-learned op-independent `V(s)`
(`src/esper/tamiyo/networks/factored_lstm.py:451-461`); `q_head` is detached aux. See
`stage1-marginal-v-verification.md`. Stage 2 is the **primary remaining lever**.

**Stage 0 is a hard EMPIRICAL gate.** Its definitions are already unified with this
plan: `stage0-instrumentation.md:56-68` now defines `R_cf = bounded_attribution` and
`R_main = reward_raw − R_cf` (exhaustive — not an enumerated subset) and measures the
residual return-variance shares of `synergy_bonus`/`alpha_shock`/`pbrs_bonus`. **No
further Stage-0 doc edit is needed** (the Round-2 mismatch is discharged). Before
implementing Stage 2, Stage 0 must be IMPLEMENTED and show BOTH:
(a) `Cov(R_cf, R)/Var(R) > 0.40`, AND
(b) low CoV of the exact subtractive `R_main`.
Stage 0 must also have RESOLVED its buffer-storage design (struct-of-arrays vs
recompute-at-update); §4.4 assumes struct-of-arrays. If a gate fails (or a dense term
like `synergy_bonus` is the `R_main`-variance culprit), STOP and re-scope, or
pre-commit to moving `synergy_bonus` into `R_cf` (a one-line splitter change).

## 2. What is provable vs what is hypothesized

| Claim | Status | Basis |
|---|---|---|
| The policy-gradient OBJECTIVE/structure is unchanged (one GAE, one advantage) | **Provable, by construction** | Advantage is the existing single GAE on `V_total`; no second advantage (§5). NOTE: the numerical advantages DO change on the ON leg (baseline is now `V_total = V_main + V_cf`, not today's single-head V), so policy *behaviour* changes — only the gradient FORM is invariant. |
| `V_cf` adds NO gradient to the shared LSTM trunk | **Provable, by construction** | `V_cf` is HEAD-ONLY (`lstm_out.detach()` before `cf_value_head`, like `q_head` at factored_lstm.py:636); the cf head cannot perturb trunk features. |
| Trunk feature dynamics on the ON leg | **Changed by design, gated** | `V_main` keeps FULL trunk gradient (`_compute_state_value`, factored_lstm.py:606) and its target moves `returns_total`→`returns_main`; the policy advantage also uses the `V_total` baseline. This IS the de-shaping mechanism, not a side effect — behavioural safety rests on the §6 empirical gates, NOT on trunk invariance. (OFF leg = byte-for-byte today's, pinned by the OFF-leg golden.) |
| `R_main_raw + R_cf_raw == reward_raw` | **Exact, by construction** | Subtractive partition on raw reward (§3). |
| `r_main_norm + r_cf_norm == reward_norm_total` | **Exact, by construction** | `r_main_norm := reward_norm_total − r_cf_norm`; `r_cf_norm` via the NEW no-clip method; clip stays on the total (§5). |
| `returns_main + returns_cf == returns_total` | **Exact, by construction** | `returns_main := returns_total − returns_cf`; requires shared mask + same `denorm_cf(V_cf)` + V_total in the returns line (§5). |
| `EV_main` lifts and its IDR shrinks | **Hypothesis** | Empirical; the separate normalizer is the lever. Gated by §6 A/B. |
| Val-acc / lifecycle decisions preserved | **Expected, gated** | Optimum-preserving objective; two trained critics need not share the single-head fixed point — within-noise drift acceptable (§6). |

(A *full-gradient* `V_cf` — trunk not detached — is a documented FOLLOW-UP A/B only,
not the default: it adds a second value-loss term to the trunk (≈2× value pressure),
which a single static `cf_value_coef` cannot hold constant against the time-varying cf
grad and the `value_coef` warmup. Head-only avoids all of that.)

## 3. The reward partition

### 3.1 Definition (subtractive, exhaustive, at the finalization site)

The PPO step reward is finalized in
`src/esper/simic/training/action_execution.py`, NOT `contribution.py`: after
`compute_contribution_reward`, it applies `escrow_forfeit` (`-=`, :1008),
`germination_forfeit` (`-=`, :1049, into `action_shaping` :1051),
`pending_auto_prune_penalty` (`+=`, :1053 — **no telemetry field**), `hindsight_credit`
(`+=`, :1061), then sets `reward_components.total_reward` (:1068) and
`action_outcome.reward_raw` (:1069); the buffer stores the **normalized** reward (:1073).

```
R_cf   = components.bounded_attribution   # the one dense counterfactual addend
R_main = reward_raw − R_cf                # everything else, by subtraction
```

`bounded_attribution` is the sole counterfactual addend hitting the reward
(`contribution.py:641`) and **already contains** `escrow_delta` (:506/516/537) and
`ratio_penalty` (:639) — re-adding either double-counts. `R_main` is exhaustive by
construction (every shaping term + the orphan `pending_auto_prune_penalty` + the
terminal corrections land in it). `action_execution.py` never mutates
`components.bounded_attribution` (reads it at :1083).

### 3.2 Home + None-handling (no defensive coalescing)

Put `split_reward_streams(reward_raw, components) -> tuple[float, float]` in
**`src/esper/simic/rewards/partition.py`** (a simic reward helper — NOT the domain-free
`leyline/telemetry_contracts.py`). Explicit None branch, never `or 0.0`:

```python
cf = components.bounded_attribution if components.bounded_attribution is not None else 0.0
# None == "no attribution this step" (bounded_attribution: float | None, telemetry_contracts.py:~57)
return reward_raw - cf, cf
```

`assert components is not None` at the call site — guaranteed by §3.3.

### 3.3 Availability (components must exist under the flag)

`return_components` is forced True only for ESCROW/BASIC_PLUS
(`_reward_components_required_for_state_transport`, `action_execution.py:189-191`;
expression at `:902-906`) — NOT SHAPED, never for non-CONTRIBUTION families. So default
SHAPED PPO has `components is None`. **Fixes:** (1) force `return_components=True` when
the flag is ON; (2) config-validation guard rejecting `hra_value_decomposition=True`
for non-CONTRIBUTION families; (3) §7 inert-decomposition assert (`R_cf` not identically
0 over an attribution-bearing rollout).

### 3.4 Stream membership (flagged for drl review)

- `escrow_forfeit`, `hindsight_credit`, `germination_forfeit` are sparse terminal
  corrections → `R_main` (low per-step variance).
- `synergy_bonus` is a separate addend (`reward += synergy_bonus`, `contribution.py:703`)
  gated by `attribution_discount >= 0.5 AND bounded_attribution > 0 AND interaction_sum
  > 0` (`contribution.py:699`, plus `interaction_sum>0` internally) — counterfactual-
  correlated, fires on a strict subset of attribution steps. Under the subtractive
  partition it sits in `R_main`. Stage 0 measures its share; **pre-commit:** if material,
  move it to `R_cf` (one-line splitter change). Do not assert `R_main` is "smooth" as
  fact (it is the §6 hypothesis).

Do NOT couple `shaped_reward_ratio` to the splitter (separate safety metric, own signs).

## 4. Verified entry points (re-mapped 2026-06-24)

### 4.1 Network (`src/esper/tamiyo/networks/factored_lstm.py`)
- `state_value_head` (= `V_main`): `:451-461` (keep the name).
- New `cf_value_head` (built only when ON): mirror `state_value_head`; declare
  **immediately after the `q_head` Sequential (ends :494) and before the
  `contribution_predictor` Sequential (`:506`; its comment block opens :496)**. In
  `_init_weights` the gain=0.1 init is a hardcoded tuple literal
  `(self.state_value_head, self.q_head)` at `:549` — **replace it with a local list and
  conditionally append** `cf_value_head` (e.g. `heads = [state_value_head, q_head]; if
  self.cf_value_head is not None: heads.append(self.cf_value_head)`), else a naive tuple
  edit raises `NameError`/`AttributeError` on the OFF path.
- New `_compute_cf_value(lstm_out)`: a **HEAD-ONLY critic** — `lstm_out = lstm_out.detach()`
  before `cf_value_head`, exactly like `_compute_q` (`:636`). (NOT full-gradient like
  `_compute_state_value` `:598-615`.)
- `_ForwardOutput` `:88-108`: add `cf_value`. Because `_ForwardOutput` is a plain `TypedDict`
  (all keys required, `total=True`), declare it as **`cf_value: NotRequired[torch.Tensor | None]`**
  (Python 3.11+ typing; if the project floor is ≤3.10 use a `total=False` extension class) so
  construction sites that omit it stay type-valid. The **single `forward()` return dict-literal
  (~:830-844) must still supply `"cf_value": None`** (OFF) or the computed tensor (ON).
  `forward()` computes `cf_value` when ON because it is the dict `get_action` and the
  telemetry forward read (§4.2; NOT for AMP-cache reasons — §4.6).
- `evaluate_actions()` `:1211-1464` (return tuple at **:1464**): compute `cf_value`
  (head-only) and return it. **Required:** replace the positional 6-tuple with a typed
  `_EvalOutput` **`NamedTuple`** (NOT a `TypedDict` — the new field must be reachable by
  attribute), appending `cf_value` as the LAST field.
- **BREAKING-ARITY SWEEP (mandatory — a 7-field NamedTuple still raises
  `ValueError: too many values to unpack (expected 6)` on every positional 6-unpack, and
  the network return changes UNCONDITIONALLY so this breaks even on the default-OFF leg).**
  Migrate EVERY positional unpack of the **NETWORK's** `evaluate_actions` to attribute
  access on `_EvalOutput` (`out = net.evaluate_actions(...); out.log_probs / out.value /
  out.cf_value`). This is the COMPLETE site list (verified by `grep -rn "evaluate_actions("`
  2026-06-24):
  - src (the only non-test consumer): `src/esper/tamiyo/policy/lstm_bundle.py:243`.
  - tests: `tests/integration/test_sanctum_head_gradients.py:484,670`;
    `tests/simic/test_tamiyo_network.py:127,193,491,519,554,722`;
    `tests/simic/test_p1sync_p1valid.py:70`;
    `tests/tamiyo/networks/test_factored_lstm.py:308,368,444` (and check `:247` which assigns
    the whole result — indexing still works, attribute is cleaner);
    `tests/tamiyo/networks/test_state_value_head.py:145,261`;
    `tests/tamiyo/policy/test_api_consistency.py:193`;
    `tests/tamiyo/policy/test_masked_logit_seam_parity.py:169,201,318`.
  - **DO NOT touch** `policy.evaluate_actions` on the lstm_bundle — it returns the
    `EvalResult` dataclass (attribute access already), so `ppo_agent.py:980/1030`, the
    `eval_result = …` sites in `test_ppo_normalization.py:179/343`, `test_anchor_reference_pass.py:218`,
    and `test_lstm_bundle.py` are UNAFFECTED. The `*args/**kwargs` monkeypatch wrappers
    (`test_anchor_reference_pass.py:552-553`, `test_q_aux_training.py:161`) are arity-resilient.
  - §7 must run the FULL `tests/simic` + `tests/tamiyo` suites green to prove no positional
    unpack was missed (the arity break is collection/runtime, not silent).

### 4.2 Rollout + bootstrap + eval contracts (the REAL paths)
The PPO rollout per-step value AND the truncation bootstrap both flow through
`get_action → GetActionResult → ActionResult`, **NOT** `ForwardResult` (which is the
SAC/`forward()` path, unused by PPO rollout). Thread `cf_value` through ALL of:
- **Network `get_action`** (`factored_lstm.py:846-1204`): it reads
  `value = output["state_value"][:, 0]` (`:1078`); add `cf_value = output["cf_value"][:, 0]`
  when ON and put it on **`GetActionResult`** (`:61`, new field `cf_value: torch.Tensor | None = None`).
- **`GetActionResult`** (`factored_lstm.py:61`, `@dataclass` line) and **`ActionResult`**
  (`src/esper/leyline/policy_protocol.py:31-51`, `@dataclass` :31) + **`EvalResult`**
  (`:54-87`, `@dataclass` :54 / `class` :55) each gain `cf_value: torch.Tensor | None = None`.
- **`lstm_bundle.get_action`** (`src/esper/tamiyo/policy/lstm_bundle.py:126`) copies
  `result.cf_value` into `ActionResult`;
  **`lstm_bundle.evaluate_actions`** (`:243` unpack, `:261` `EvalResult` build) carries
  `cf_value` (via the typed `_EvalOutput`).
- **Rollout store:** `vectorized_trainer.py:2056/2066/2079` reads `action_result.cf_value`;
  thread a `cf_values` array through `execute_actions` → `action_execution.py:540` →
  `buffer.add(...)` (`:1400`).
- **Truncation bootstrap:** `vectorized_trainer.py:2309-2328` (dedicated `get_action`
  at :2309; `bootstrap_result.value` :2318; write to `buffer.bootstrap_values[...]` :2328):
  read `bootstrap_result.cf_value`, build `cf_bootstrap_values`, write
  `buffer.cf_bootstrap_values[...]` — from the SAME forward as the main bootstrap (shared mask).
- `ppo_agent.py` reads `result.cf_value` by attribute + a hard PPO-path assert (like
  `q_value` `:1049`), present iff ON. (`ForwardResult` is SAC-only and not on the HRA
  path — leave it unchanged.)

### 4.3 Reward partition wiring
- `split_reward_streams()` in `src/esper/simic/rewards/partition.py` (§3.2).
- Call at `action_execution.py:1069`; store `R_cf_raw` and `r_cf_norm` (§5) per step.
- **No new `RewardMode`** (`00-INDEX.md:93` already records this).

### 4.4 Buffer + GAE (`src/esper/simic/agent/rollout_buffer.py`)
- (Struct-of-arrays — confirm Stage 0 chose this; §1.) `add()` (`:362`; scalar
  `value`/`bootstrap_value` `:384`/`:398`) threads `cf_value`, `cf_bootstrap_value`,
  `r_cf_norm`. Store `cf_values`, `cf_bootstrap_values`, `r_cf_norm` arrays. **The
  existing `value` arg keeps storing `V_main` only.**
- `compute_advantages_and_returns()` (`:491`, single `value_normalizer` `:495`, writes
  `self.returns`/`self.advantages` `:584-589`): extend the signature with
  `cf_value_normalizer`. **Inside the per-env loop:** `values_total = denorm_main(V_main)
  + denorm_cf(V_cf)` (denormalize `cf_values`/`cf_bootstrap_values` with
  `cf_value_normalizer` exactly as `values`/`bootstrap_values` are at `:542-543`/`:550-551`);
  `delta`, `next_value`, **and the returns line (`:589`) all use `values_total`** so
  `returns_total = advantages + values_total` (NOT `+ V_main`). Run one extra cf recursion
  on `(r_cf_norm, V_cf)` (cf bootstrap at truncation) → `returns_cf`; set
  `returns_main = returns_total − returns_cf`. `returns_total` **replaces (is the existing
  `self.returns` field renamed/reused** — `advantages + values_total` instead of
  `advantages + values`); only `returns_main`/`returns_cf` are genuinely new arrays.
  **Normalizer routing into GAE (avoid the scale-mismatch trap):** the existing
  `value_normalizer=` argument to `compute_advantages_and_returns` becomes
  **`value_main_normalizer`** — it denormalizes `self.values`/`bootstrap_values` (= `V_main`)
  at `:542-543`/`:550-551`; the NEW `cf_value_normalizer` denormalizes
  `cf_values`/`cf_bootstrap_values` (= `V_cf`). **The total `value_normalizer` is NOT passed
  into GAE at all** — it is only the q-aux + `ev_sum` target normalizer in `ppo_agent.py`
  (§4.5). (Leaving `self.values` denormalized by the total normalizer while training `V_main`
  against `value_main_normalizer` would silently corrupt the GAE deltas.) **Update the
  caller** of `compute_advantages_and_returns` in `ppo_agent.py` (`~:600-603`) to pass the
  renamed `value_main_normalizer=` kwarg AND the new `cf_value_normalizer=` — a stale kwarg
  name there raises `TypeError` at runtime. `get_batched_sequences` exports
  `returns_total` (key may stay `returns` for the unchanged q-aux path), `returns_main`,
  `returns_cf`, `cf_values`.

### 4.5 Per-stream targets, loss, normalizers, checkpoint (`ppo_agent.py`, `ppo_update.py`)
- `cf_value_head` is **head-only** (§4.1) — its loss backprops only into its own params.
  Add `cf_value_head.parameters()` to the optimizer `critic` group `:418-421` (built only
  when ON). The LSTM trunk therefore sees exactly today's value gradient (`V_main` only);
  **no `cf_value_coef` / trunk-pressure tuning is needed.**
- **Three `ValueNormalizer`s when ON** (`src/esper/simic/control/normalization.py:256-437`).
  **Do NOT rename the existing `value_normalizer`** (it has ~14 call sites in `ppo_agent.py`
  plus tests). Instead:
  - `value_normalizer` (existing, UNCHANGED): stays fed `returns_total` and remains the
    **q-aux target** + the `ev_sum` total normalizer. Already serialized at `:1649`, already
    in the load path — so q-aux's scale survives resume with *zero* new checkpoint state. Its
    "predicts the full return" semantic is byte-for-byte today's on the OFF leg (where
    `V_total == V_main ==` today's `self.returns`, pinned by the OFF-leg golden); on the ON
    leg it is the same total-return *semantic*, now reconstructed as `V_main + V_cf` (intended,
    not a defect).
  - `value_main_normalizer` (NEW): fed `returns_main`; the `V_main` target + GAE denorm for
    `V_main`.
  - `cf_value_normalizer` (NEW): fed `returns_cf`; the `V_cf` target + GAE denorm for `V_cf`.
  The single `RewardNormalizer` stays (§5).
- **Per-stream target plumbing (the error-prone wiring — spell it out):**
  `get_batched_sequences` (`rollout_buffer.py:~655`) exports `returns_total`, `returns_main`,
  `returns_cf`, `cf_values` into the `data` dict. In `ppo_agent.py` (~`:700`): feed
  `value_normalizer` ← `returns_total` (q-aux, unchanged), `value_main_normalizer` ←
  `returns_main`, `cf_value_normalizer` ← `returns_cf`. `compute_losses`
  (`ppo_update.py:246-439`) gains params `normalized_returns_main`, `normalized_returns_cf`,
  `cf_values`; `value_loss` regresses `V_main` on `normalized_returns_main`; **`q_aux`
  regresses on the unchanged `normalized_returns_total`** (`value_normalizer.normalize(returns_total)`,
  `ppo_update.py:367`) — NO third *new* stateful normalizer is introduced (the existing one
  IS the total); new `cf_value_loss` regresses `V_cf` on `normalized_returns_cf` as the
  **unconditional-MSE branch only** (NO `old_cf_values`/clip — `clip_value` is False by
  default and raises `ValueError` under `recurrent_n_epochs>1`, `ppo_agent.py:283-290`;
  comment-reference it). `LossMetrics` (`:34-46`) gains `cf_value_loss`; total loss
  `:424-430` gains `+ value_coef * cf_value_loss` (same `value_coef`; both are unit-scale
  head-only MSE — no trunk interaction).
- **EV path** `ppo_agent.py:666-680` (today: `raw_values = value_normalizer.denormalize(...)`
  at `:661`): denormalize `V_main` via **`value_main_normalizer`** and `V_cf` via
  **`cf_value_normalizer`** before scoring; `ev_main = EV(V_main, returns_main)`,
  `ev_cf = EV(V_cf, returns_cf)`, `ev_sum = EV(denorm_main(V_main)+denorm_cf(V_cf),
  returns_total)` via `compute_floored_explained_variance`. (Stage 0 *will add* the `ev_*`
  keys — grep confirms none exist in `src/` yet.)
- **Checkpoint — bump BOTH:** `CHECKPOINT_VERSION 2→3` (`ppo_agent.py:67`; serialize the
  TWO new `value_main_normalizer_state_dict` + `cf_value_normalizer_state_dict` alongside
  the existing `value_normalizer_state_dict` at `:1649`, and restore all three in the load
  path `:1793/:1898`) AND `VALUE_HEAD_SCHEMA_VERSION 2→3` (`src/esper/leyline/__init__.py:125`;
  topology guard `:1781`, update its message to the three-head topology). The checkpoint
  records `hra_value_decomposition`; the load path asserts the build flag matches and emits
  a descriptive error on mismatch. Pre-Stage-2 checkpoints will not load into an ON build —
  acceptable (no-back-compat); state it. The §7 checkpoint round-trip test must cover all
  THREE normalizers.
- Telemetry: emit `cf_value_loss`, `ev_cf`, `value_head_count`, the flag, and a
  partition-version id through the Stage-0 6-file chain + run-config snapshot.

### 4.6 Config, CLI, AMP, and the TRUE bypass
- `TrainingConfig.hra_value_decomposition: bool = False` (`config.py`). `slots=True` +
  `_validate_known_keys` (`:260-264`) rejects *unknown JSON keys*; adding the field makes
  it KNOWN, so existing `configs/*.json` (which omit it) load unchanged with the default —
  **no mass config edit.** **Thread the FULL chain (3 links):** `to_train_kwargs()`
  (`config.py:347`) → the **`train_ppo_vectorized(...)` function signature** in
  `vectorized.py` (add the param — omitting this raises `TypeError: unexpected keyword
  argument`) → the **`PPOAgent(...)` call at `vectorized.py:1145`** + the network ctor /
  `create_policy`. CLI flag `--hra-value-decomposition` on the **`ppo_parser`
  (`scripts/train.py:431`)** — NOT the `--reward-mode` block at `:409`. **Also wire the
  parsed arg into the config** at the point train.py maps args→`TrainingConfig` overrides
  (mirror an existing bool flag, e.g. `config.hra_value_decomposition = args.hra_value_decomposition`);
  the argparse entry alone does not reach `to_train_kwargs()` without this assignment.
- **TRUE bypass = conditional construction.** When OFF: do NOT construct `cf_value_head`/
  `cf_value_normalizer`; `forward()`/`evaluate_actions`/`get_action` set `cf_value=None`;
  GAE uses `V_total = V_main`; the buffer has no cf arrays. The OFF leg is the current
  single-head path verbatim (no dead weights — no-legacy). §7 OFF-leg golden pins it.
- **torch.compile note (RESOLVED against the live tree).** The bypass uses
  `if self.cf_value_head is not None:` inside `forward()`/`evaluate_actions()`. The flag is a
  **construction-time constant**, so this yields ONE compiled graph per setting (OFF/ON), not
  a per-call graph break. **Verified: `torch.compile` is invoked WITHOUT `fullgraph=True`** —
  the sole call is `lstm_bundle.py:493` `torch.compile(self._network, mode=mode, dynamic=dynamic)`
  (the only `fullgraph=True` in `src/` is an unrelated `kasmina/blend_ops.py` docstring; the
  `compile_mode` string at `config.py:483` — `default`/`max-autotune`/`reduce-overhead`/`off` —
  is the strategy, NOT a fullgraph constraint). Therefore the construction-time
  `if cf_value_head is not None` branch is **ACCEPTABLE as written** — no `torch.zeros_like`
  branch-free workaround is needed. (Guard for the future: if anyone later adds `fullgraph=True`
  to the `lstm_bundle.compile` call, switch the OFF path to a constant-zero
  `torch.zeros_like(state_value)` so the forward stays branch-free, matching the
  `contribution_predictor` precedent.)
- **AMP note (corrected).** In `update()` (~`:841` onward): the no_grad telemetry forward
  runs under `autocast(enabled=False)` and writes NOTHING to the cast cache (per its in-code
  comment). The cache-poisoning site is the **anchor `evaluate_actions` pass** later in
  `update()` (caller's BF16 autocast), evicted by a global `torch.clear_autocast_cache()`
  (the `:950-992` region). Because the anchor computes `cf_value` when ON, the existing clear
  already covers the new head — no extra guard. (With the head-only/detached cf head the
  point is doubly moot.)

## 5. The GAE construction (exact, clip-safe, policy-gradient-untouched)

**Scales (verified).** The buffer stores the **clipped normalized total**
(`reward_normalizer.update_and_normalize(reward)`, **sole** normalize site
`action_execution.py:1073`; `:1422` is just the `reward=normalized_reward` kwarg into
`buffer.add`). `RewardNormalizer` is std-only (no mean) but **clips to ±10**
(`src/esper/simic/control/normalization.py:208-231` — note the `control/` module, NOT
`simic/agent/` where most other cited files live). The existing `normalize_only` (`:233-239`)
**also clips** — do NOT reuse it. To make the split exact AND clip-safe:

1. **Add a NEW, distinctly-named method** to `RewardNormalizer`, e.g.
   `divide_by_std(reward) -> reward / std` — **no stat update, NO clip**, with the
   `count<2` branch returning the raw value unclipped. Leave the existing clipping
   `normalize_only` untouched (its callers/tests depend on the clip:
   `test_normalization.py:180` asserts `normalize_only(10.0)==5.0`;
   `ppo_coordinator.py` deliberately avoids it for that clip).
2. At `action_execution.py:1073`, right after `update_and_normalize(reward)` updates the
   running std, compute `r_cf_norm = reward_normalizer.divide_by_std(r_cf_raw)` (same
   per-step std, unclipped). Store the clipped `normalized_reward` (total, unchanged) AND
   `r_cf_norm`.
3. Define `r_main_norm := reward_norm_total − r_cf_norm` (in GAE; never stored). The clip
   residual lands in `R_main`. The identity `r_main_norm + r_cf_norm == reward_norm_total`
   is a *definition* (the clip stays a total-only property).

**Policy advantage — UNCHANGED.** The existing single recursion (`:557-589`) produces
`A_total`/`returns_total` using `V_total` (§4.4). The post-GAE global advantage
normalization (`normalize_advantages`, `rollout_buffer.py:595`) and the per-head
re-standardization (`compute_per_head_advantages`, def `advantages.py:62`, loop
`:124-152`) run **unchanged on the single `A_total`** — the nonlinear
`per_head_advantage_norm` lever acts on one advantage tensor.

**Value targets.** `returns_cf` from one extra recursion on `(r_cf_norm, V_cf)` with
`cf_bootstrap_values` at truncation; `returns_main := returns_total − returns_cf` (exact
per-stream λ-return: `returns_total − returns_cf = (A_total−A_cf)+V_main = A_main+V_main`).
Regress `V_main` on `value_main_normalizer.normalize(returns_main)`, `V_cf` on
`cf_value_normalizer.normalize(returns_cf)`. **Warm-up:** both `ValueNormalizer`s return
inputs UNCHANGED until `has_valid_stats` (`count >= _min_samples = 32`,
`normalization.py:305/315-317/375-376/395-396`); during cf warm-up `denorm_cf` is
identity, so `V_total`'s scale (and thus the GAE baseline) is transiently distorted —
another reason the A/B is **fresh-init paired** (§6).

## 6. Acceptance gates

Structural (must hold; failure = real bug):
- Partition (raw): `r_cf_raw == components.bounded_attribution`;
  `r_main_raw + r_cf_raw == reward_raw` across §7's matrix.
- `r_cf_norm == reward_normalizer.divide_by_std(r_cf_raw)` (no clip) — incl. a clip-active
  total step where `update_and_normalize` returns ±10 but `divide_by_std(r_cf)` does not.
- **q-aux fidelity pin:** with the flag ON, the q-aux regression target tensor is
  bit-identical to `value_normalizer.normalize(returns_total)` (NOT `returns_main`). This
  guards against the inversion (pointing q-aux at the main stream) that would silently
  corrupt the "q-aux unchanged" telemetry semantic the 3-normalizer fidelity argument rests
  on — and which no other gate would catch.
- `returns_main + returns_cf == returns_total` (≤1e-5, raw) incl. truncation boundaries.
- Inert-decomposition: `R_cf` not identically 0 over an attribution-bearing rollout (§3.3).
- OFF-leg golden bit-identical to `main` (conditional construction).
- `ev_sum` (ON) ≈ single-head EV — empirical near-equality, NOT a bit-tight gate.

Primary (empirical — succeeds iff):
- **EV_main lifts and stabilizes:** median `ev_main` ≥ the single-head `ev_sum` baseline
  AND its inter-decile range shrinks ≥30% vs the Stage-0 `ev_sum` IDR baseline. IDR is the
  load-bearing criterion. **Default to a fresh-init paired A/B** (both legs share warm-up);
  if measuring within one run, key the window off `cf_value_normalizer.has_valid_stats`
  (`count >= 32`), not a hardcoded sample count.

Fidelity:
- **EV_cf floor:** `ev_cf > -0.3` after the first ~500 post-warm-up updates (a dead cf
  critic de-weights the growth signal). A head-only critic is EXPECTED to clear this — but
  this is the empirical question the A/B exists to measure, not a foregone conclusion:
  `R_cf` is the HIGH-CoV stream and a trunk-detached head has strictly less capacity to fit
  it than a full-gradient head. **A grazed floor is the most informative outcome** (cf is
  genuinely hard to predict from state, which bears on whether routing it out helps `V_main`
  at all) and triggers the §2 full-gradient `V_cf` follow-up A/B — not a failure to explain
  away.
- **Lifecycle-decision parity:** germinate/fossilize/prune counts + attribution magnitudes,
  ON vs OFF on the same seeds, within the recorded run-to-run band. **Also spot-check the
  decision RATE within the first ~32 ON-leg updates** (key off
  `cf_value_normalizer.has_valid_stats`): during cf-normalizer warm-up `V_total`'s scale is
  transiently distorted (§5), and that window is exactly when early growth commitments form
  — a run-aggregate count can average a warm-up-window decision shift away. Read this from the
  existing per-update Karn fields (`decisions.decision_density` `views.py:175`,
  `decision_entropy` `:394`, `auto_pruned` `:300`, fossilization counts `:364/:517-520`) so
  the check binds to real telemetry, not an aspirational instruction.
- **Val-acc non-regression:** within run-to-run std (within-noise drift acceptable; only an
  out-of-noise regression WITH a failing structural check is a bug). DPBA refold
  (Harutyunyan 2015) is the fallback only for a future variant-B strong-de-shape.

## 7. Test plan

- **Partition exactness (keystone, NEW):** `test_reward_streams_partition` over the **full
  `action_execution.py` path**, asserting `r_main_raw + r_cf_raw == reward_raw` (≤1e-6)
  across SHAPED × ESCROW × {WAIT,GERMINATE,PRUNE,FOSSILIZE,ADVANCE} × stages that fire
  escrow credit/forfeit, occupancy/fossilized rent, hindsight, germination forfeit,
  auto-prune; `r_cf == bounded_attribution` (ESCROW case `== escrow_delta`, once). Add a
  **SHAPED leg with all telemetry emitters OFF** (proves §3.3 force-on).
- **divide_by_std correctness:** divides by current std, no stat update, **no clip**;
  explicit case `reward_raw=15, std=1`: `update_and_normalize(15)→10` (clipped) but
  `divide_by_std(r_cf)→r_cf` (unclipped), and `10 − r_cf` lands in `r_main_norm`. Assert
  the existing clipping `normalize_only` is untouched (`normalize_only(10)==5`).
- **GAE linearity / value-target consistency:** synthetic rollout (known
  `R_cf`/`V_main`/`V_cf`/`cf_bootstrap`); `returns_main + returns_cf == returns_total`
  (≤1e-5) incl. truncation; verify the returns line uses `V_total`.
- **Normalizer symmetry + warm-up:** `denorm_cf(norm_cf(x))==x`; independent scales;
  first-enabled no NaN/Inf.
- **Flag parity (true bypass):** OFF ⇒ no cf head; update bit-identical to `main`
  (`test_ppo_update_golden.py`, OFF + ON legs).
- **Head registration + detach:** `cf_value_head.parameters()` in optimizer iff ON,
  nonzero grad after one ON update; **assert no gradient reaches the LSTM via the cf head**
  (head-only) — e.g. trunk grad with the flag ON equals trunk grad with V_cf removed.
- **Checkpoint:** ON checkpoint at v3 (both versions) round-trips cf head + cf normalizer;
  loading ON↔OFF raises a descriptive flag-mismatch error.
- **Per-stream EV:** `ev_main`/`ev_cf`/`ev_sum` finite on the `ppo_updates` Karn view.

## 8. Risks / rollback

- **Reward-normalizer clip/std:** resolved by `divide_by_std` (no clip) + subtractive
  `r_main_norm`; structural tests catch regressions.
- **Wrong rollout/bootstrap contract:** v4 threads cf through `get_action`/`GetActionResult`/
  `ActionResult` + the bootstrap (§4.2); the truncation linearity test guards it.
- **Trunk-gradient confound:** ELIMINATED — `V_cf` is head-only (detached trunk).
- **Cold cf normalizer + warm-up scale distortion:** fresh-init paired A/B (§6).
- **per_head_advantage_norm confound:** explicit `× hra_value_decomposition` A/B cell.
- **Stage-0 dependency:** Stage 2 blocked until Stage 0 implemented AND its buffer-storage
  design resolved (§1, §9).
- **Memory/throughput:** one extra head + cf arrays + one extra recursion; gate on no
  rollout-throughput regression.
- **Rollback:** `hra_value_decomposition=False` ⇒ cf head/normalizer not constructed
  (current path). Full revert deletes the head, cf normalizer, splitter, `divide_by_std`,
  loss term, and reverts both version bumps — no env-reward migration.

## 9. Sequencing

1. Land Stage 0 (per-component arrays + per-stream EV + CoV of `reward_raw −
   bounded_attribution` + `synergy_bonus`/`alpha_shock` shares) — hard dependency.
2. **0b:** Confirm Stage 0 RESOLVED its buffer-storage design; if recompute-at-update,
   revise §4.4.
3. Confirm the Stage-0 gate (Cov(R_cf,R)/Var(R) > 0.40 AND low `R_main` CoV). Else STOP /
   re-scope (or move `synergy_bonus` to `R_cf`).
4. Implement §3-§5 behind the default-OFF flag (conditional construction, head-only cf);
   land §7 green.
5. Paired **fresh-init** A/B (ON vs OFF, × per_head_advantage_norm) on the capable host;
   read `ev_main` IDR, `ev_cf` floor, lifecycle parity, val-acc.
6. If EV_main stabilizes, EV_cf holds, decisions/val-acc hold → record acceptance evidence,
   promote the default; redirect Stage 1b / 3 / 3x per the index. (If EV_cf sits near its
   floor, a follow-up A/B can test a full-gradient `V_cf` with `value_coef`-matched warm-up.)

## 10. Reviewers (required before promotion)

- **drl-expert** + **yzmir-deep-rl:reward-function-reviewer** — §2 table honesty, the
  partition + `synergy_bonus` stream call (§3.4), the single-GAE construction + cf
  bootstrap + head-only `V_cf` (§4.2/§4.5/§5), q_aux target.
- **pytorch-expert** / **yzmir-pytorch-engineering:pytorch-code-reviewer** — `divide_by_std`
  + step-time normalization (§5), the `get_action`/`ActionResult` plumbing (§4.2), the
  per-stream target plumbing (§4.5), conditional construction + AMP (§4.6), both version
  bumps, dropping the dead clip path.
- **axiom-python-engineering:python-code-reviewer** — `split_reward_streams` placement +
  None branch, the `ActionResult`/`GetActionResult`/`EvalResult` contract additions, typed
  `_EvalOutput`, the `to_train_kwargs → train_ppo_vectorized → PPOAgent` chain,
  `TrainingConfig` known-keys, the 6-file telemetry contract.
- **axiom-planning:plan-review-reality** — re-verify v4 citations (esp. `factored_lstm.py:61`/
  `:846`/`:1078`/`:1201`, `policy_protocol.py:31-51`, `normalization.py:233-239`/`:305`,
  `vectorized_trainer.py:2056/2066/2309/2318`).
- **yzmir-morphogenetic-rl:morphogenesis-reviewer** — head-only `V_cf` vs attribution
  fidelity (EV_cf floor), lifecycle parity, fresh-init ablation, replay persistence.
