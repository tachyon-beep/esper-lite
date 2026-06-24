# Plan Metadata
id: ev-stabilization-stage0-instrumentation
title: "Stage 0 — Instrument per-component return variance + per-stream EV"
type: ready
created: 2026-06-23
updated: 2026-06-23
owner: Claude (ralph-loop)

urgency: high
value: Decide — with evidence, not assumption — whether the high-CoV counterfactual term actually dominates residual value-target variance, which is the gate for the (expensive) Stage 2 de-shaping. Also gives the per-stream EV baseline all later A/Bs measure against.

complexity: S
risk: low
risk_notes: Pure telemetry. No change to the reward, the policy, the critic, or the loss. The only failure modes are (a) a metric that mis-attributes variance and (b) the 6-file metric-plumbing contract drifting (mandatory-vs-optional field handling).

depends_on: []
soft_depends: []
blocks:
  - ev-stabilization-stage2-deshape

status_notes: >
  Drafted. Supervised-session review (2026-06-24): APPROVED-WITH-CHANGES. Covariance
  decomposition math correct (Σ shares == 1 holds because R = Σ R_i, on RAW pre-norm
  returns); buffer-storage Open Question DECIDED → option (A) struct-of-arrays (single-
  source, bounded memory), matching Stage 2 §4.4's assumption; R_cf/R_main split unified
  with Stage 2's subtractive partition. One consistency fix applied: partition.py is
  CREATED here (Stage 0, the earlier stage) and reused by Stage 2 — resolves the apparent
  circular dependency. Ready to implement alongside Stage 2 (flag-OFF) per the supervised
  plan. Filigree esper-lite-3d67b09687.
percent_complete: 0

reviewed_by:
  - reviewer: supervised-session (independent, code-grounded)
    date: 2026-06-24
    verdict: approved-with-changes
    notes: >
      Buffer-storage decided = option (A) struct-of-arrays. partition.py ownership moved to
      Stage 0. Variance shares + the Cov(R_cf,R)/Var(R)>0.40 gate are on RAW pre-normalization
      returns. Telemetry 6-file chain consistent with Stage 2's verified citations.

---

# Stage 0 — Instrument per-component return variance + per-stream EV

## Why (corrected context)

EV-liftoff is **already achieved** on the capable host (median ~0.31) — but it
is **volatile**, and the P0-1 critic fix (op-independent V(s)) has **already
landed** (commit `6a27b8e3`). So the leading remaining hypothesis for the
volatility is **target noise**: the dense, high-CoV counterfactual contribution
term (`seed_contribution` / `bounded_attribution`, CoV ~1.9 baseline / ~2.7
impaired) sits inside the return the single critic regresses on.

**This stage proves or refutes that hypothesis before any behavioural change.**
All four research reports insist on "instrument first." If the counterfactual
term does **not** dominate return variance, the Stage-2 de-shaping premise is
wrong and the epic re-scopes.

## Goal / acceptance gate

Emit, per PPO update, on the `ppo_updates` Karn view:
1. **Per-component return-variance share** — for each reward component
   (`seed_contribution`/attribution, `pbrs_bonus`, `escrow_delta`, `compute_rent`
   + rents, `terminal_bonus`, action-shaping/costs): its fractional contribution
   to `Var[return]`. Implemented as the variance of the per-step component stream
   accumulated to discounted return, normalized by total return variance. (A
   covariance-aware decomposition is preferred: `share_i = Cov(R_i, R)/Var(R)`,
   which sums to 1 exactly and correctly handles correlated components.)
2. **Per-stream EV** — `ev_main`, `ev_cf`, `ev_sum`. **The stream split MUST match
   Stage 2's subtractive partition exactly** (so the gate measures the stream
   `V_main` will actually regress on): `R_cf = bounded_attribution` (the single
   counterfactual addend, which already contains `escrow_delta`/`ratio_penalty`);
   `R_main = reward_raw − R_cf` (EXHAUSTIVE — NOT an enumerated PBRS+terminal+rent
   subset). `ev_main` = single V(s) vs the `R_main` return, `ev_cf` = vs the `R_cf`
   return, `ev_sum` = total-return EV (existing `explained_variance`). For Stage 0
   these are *diagnostic recomputations on the same stored values*; no second
   critic head yet (that is Stage 2). Use `simic/rewards/partition.py:split_reward_streams`,
   a tiny pure helper `split_reward_streams(reward_raw, components) -> (r_main, r_cf)`.
   **Ownership (resolves the apparent circularity):** Stage 0 lands FIRST (it is Stage 2's
   hard dependency), so **Stage 0 CREATES `partition.py`** and Stage 2 REUSES it — not the
   reverse. Single-source so the two stages cannot diverge.
3. **R_main smoothness / residual-variance shares** — CoV of the exact subtractive
   `R_main`, plus the per-component return-variance shares of the dense terms that
   REMAIN in `R_main` (notably `synergy_bonus` — gated on `bounded_attribution>0`,
   so counterfactual-correlated — `alpha_shock`, and `pbrs_bonus`). This decides
   whether `R_main` is actually the lower-variance stream Stage 2 assumes.
4. **EV volatility baseline** — record median + inter-decile range of `ev_sum`
   over a reference run so later A/Bs have a fixed comparison.

**GATE for the epic (BOTH must hold):**
(a) `Cov(R_cf, R)/Var(R) > 0.40` on the capable-host reference run (the cf stream
dominates return variance); AND
(b) low CoV of the exact subtractive `R_main` (it is genuinely the smoother stream).
⇒ Stage 2 is justified. Else ⇒ re-scope (volatility is not counterfactual-target-
noise — investigate normalizer drift / R2D2 stored-state staleness / per-head
scaling), OR, if a dense term such as `synergy_bonus` is the `R_main`-variance
culprit, pre-commit to moving it into `R_cf` (a one-line `split_reward_streams`
change Stage 2 already contemplates).

## Implementation (verified entry points)

Add scalar metrics through the existing 6-file PPO-update metric contract:

1. **`src/esper/simic/agent/types.py`** — extend the `PPOUpdateMetrics` TypedDict:
   `ev_main: float`, `ev_cf: float` (rename existing total to keep `explained_variance` as `ev_sum`-equivalent), and `return_variance_shares: dict[str, float]`.
2. **`src/esper/simic/agent/ppo_agent.py` (~:660-676)** — alongside the existing
   `compute_floored_explained_variance(raw_values, valid_returns, floor)` call:
   - Partition the stored per-step rewards into component streams (the rollout
     buffer must carry per-component rewards — see "Open question" below).
   - Recompute returns per stream (reuse the GAE/return math, or accumulate the
     component's discounted sum), and score the single `raw_values` against
     `R_main` and `R_cf` via the same floored-EV helper → `ev_main`, `ev_cf`.
   - Compute `Cov(R_i, R)/Var(R)` shares over the valid mask.
3. **`src/esper/simic/agent/ppo_metrics.py` (`finalize`, :85-141)** — add reduction
   for the new keys: scalars mean-reduced; `return_variance_shares` forwarded as a
   structured dict (mirror the `head_advantage_norm_stats` passthrough at :110-111).
4. **`src/esper/simic/telemetry/emitters.py` (`emit_ppo_update_event`, :924-1145)** —
   extract the new keys into the payload (match mandatory-vs-optional handling).
5. **`src/esper/leyline/telemetry.py` (`PPOUpdatePayload`, :705+)** — add fields
   (`ev_main`, `ev_cf`, `return_variance_shares: dict | None`).
6. **`src/esper/karn/mcp/views.py` (`ppo_updates`, :114-228)** — add
   `json_extract(data,'$.ev_main')::DOUBLE`, `'$.ev_cf'`, and the shares column.

### Open question to resolve in review
The rollout buffer currently stores the **scalar** per-step reward
(`rollout_buffer.py` `self.rewards`). Per-component variance decomposition needs
the **per-component** per-step values. Two options:
- **(A)** Carry the existing `RewardComponentsTelemetry` (already computed in
  `contribution.py`) through to the buffer as a small struct-of-arrays
  (memory cost: ~6 floats × steps × envs). Cleanest; reuses an existing contract.
- **(B)** Recompute the component split at update time from stored inputs
  (cheaper memory, but duplicates reward logic — rejected: violates single-source).
Recommend (A). Flag to reviewers; it is the only non-trivial design choice here.

## Test plan
- Unit: `tests/simic/` — a synthetic rollout with known component streams asserts
  `Σ shares == 1.0` (covariance decomposition) and that `ev_sum` matches the
  pre-existing `explained_variance` exactly (no regression).
- Contract: extend the PPO-update golden/telemetry test to include the new keys
  (mandatory-field presence; Karn view column exists).
- Karn: `mcp__esper-karn__query_sql` on `ppo_updates` returns the new columns.

## Risks / rollback
- **Low.** Telemetry-only; revert is deleting the new fields. The covariance
  decomposition is the one place to get the math right (correlated components);
  unit test pins `Σ shares == 1`.
- Carrying per-component arrays (option A) slightly increases buffer memory —
  bounded and measured; gate on no rollout-throughput regression.

## Reviewers
drl-expert (variance-decomposition correctness, the >40% gate threshold),
yzmir-deep-rl (per-stream EV semantics), axiom-python (the 6-file contract +
TypedDict/dataclass discipline, no defensive `.get()` drift).
