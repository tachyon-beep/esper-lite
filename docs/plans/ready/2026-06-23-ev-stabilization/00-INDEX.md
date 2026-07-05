# Plan Metadata
id: ev-stabilization
title: "EV-stabilization: joint value-target variance reduction for recurrent factored-action PPO"
type: ready
created: 2026-06-23
updated: 2026-06-23
owner: Claude (ralph-loop)

urgency: high
value: Stabilize critic explained-variance (EV) volatility (EV-liftoff already achieved at median ~0.31 but volatile) by removing the dominant residual value-target variance source.

complexity: L
risk: medium
risk_notes: Stage 2 changes the critic's regressand and adds a value head — behavioural; mitigated by per-stream EV measurement, reward-mode A/B, and a DPBA fallback. Stage 3 (escrow) is a correctness fix that touches the observation contract (checkpoint/normalizer shapes).

depends_on: []
soft_depends:
  - recurrent-ppo-multiepoch   # anchored reference pass (landed)
blocks: []

status_notes: >
  PREMISE CORRECTED 2026-06-23 — see below. Stage 1 (P0-1 marginal V(s)) ALREADY
  LANDED (commit 6a27b8e3); re-scoped to verification. Stage 2 (de-shape via HRA
  head) is now the primary lever. Index + Stage-0 + Stage-1-verification drafted;
  Stage 2 / 1b / 3 plans pending; specialist reviews pending.
percent_complete: 20

reviewed_by: []   # PENDING — drl-expert + yzmir-deep-rl + pytorch-expert required before promotion

---

# EV-stabilization — Plan Index

Filigree epic: **`esper-lite-f25b71c165`** (label `ev-variance-reduction`).
Research basis: 4 reports in `docs/research/` (see memory `ev-variance-research-verdict`).

## ⚠️ CRITICAL PREMISE CORRECTION (2026-06-23)

The epic and the 4 research reports were framed around a critic that is **"an
action-conditioned `Q(s,op)` used as BOTH the PPO baseline AND a single-sample
`Q(s',op')` bootstrap"** (the "Change B / P0-1" defect). **This is no longer
true.** Verified against current code + git:

- Commit **`6a27b8e3`** — *"feat(tamiyo,simic): op-independent V(s) baseline replacing Q(s,op) (P0-1)"* — is on the current branch.
- The critic is now a **directly-learned op-independent `V(s)`** (`state_value_head`, `factored_lstm.py:451-461`). The comment at `factored_lstm.py:438-446` states it explicitly.
- `Q(s,op)` (`q_head`, `factored_lstm.py:480-494`) is **detached, telemetry/aux only** — no longer the baseline.
- The GAE bootstrap uses **`V(s')`** (`rollout_buffer.py:578`: `delta = rewards[t] + gamma_t * next_value * next_non_terminal - values[t]`, where `next_value` is the stored state-value), NOT a sampled `Q(s',op')`.

**Consequence:** the reports' unanimous **#1 recommendation (marginal `V(s)`
baseline + expected bootstrap) is ALREADY SATISFIED** — by the cleaner
directly-learned-V variant the reports themselves preferred. Their Stage-2
analysis (de-shaping, HRA, the action-dependent-baseline "mirage", PBRS/DPBA,
COCOA, escrow) remains valid because it was premised on the *reward composition*,
which is unchanged.

## Per-stage status

| Stage | Filigree | Status | Verdict |
|-------|----------|--------|---------|
| **0 — Instrument** (per-component return variance + per-stream EV) | `esper-lite-3d67b09687` | OPEN — plan drafted (`stage0-instrumentation.md`) | The gate. Decides if Stage 2 is justified (counterfactual >40% of return variance). |
| **1 — Marginal V(s) / P0-1** | `esper-lite-e6382020d2` | **DONE (commit 6a27b8e3)** — re-scoped to verify (`stage1-marginal-v-verification.md`) | Already implemented; confirm acceptance evidence + close. |
| **1b — Per-head adv re-standardization** | `esper-lite-89983714fb` | OPEN — plan pending | Flag-flip + A/B; flag still OFF (`advantages.py:138`). |
| **2 — De-shape via HRA value head** | `esper-lite-2a4b56e719` | **PLAN CLEARED** (`stage2-deshape-hra-head.md`, v6; 5 review rounds, R5 panel all approved/approved-with-changes, 0 blockers/majors). Implementation gated on Stage 0. | **PRIMARY LEVER now** (B is done; residual EV volatility ⇒ target noise = A). HRA sum-of-heads (variant A); subtractive partition; single-GAE advantage; **head-only (detached-trunk) V_cf**. |
| **3 — Escrow telescoping fix** | `esper-lite-3defe42928` | OPEN — plan pending | Correctness fix; clip still at `contribution.py:532-536`. |
| **3x — COMA per-head baselines** | `esper-lite-6f30fdf089` | DEFERRED (experimental) | Redundant given the single shared V(s) baseline; only if 0/1b/2 plateau. |

## Corrected sequencing

1. **Stage 0** (instrument) — no behavioural risk; establishes the per-stream EV baseline and the >40%-of-return-variance gate.
2. **Stage 1 verification** — confirm P0-1 is implemented + the EV-liftoff acceptance evidence; close the Filigree task.
3. In parallel after Stage 0 + review: **Stage 2** (de-shape, primary), **Stage 1b** (per-head norm A/B), **Stage 3** (escrow correctness).
4. **Stage 3x (COMA)** only if 0/1b/2 plateau.

## Verified entry-point reference (mapped 2026-06-23; reuse — do not re-derive)

### Critic / GAE / baseline (`tamiyo/networks/factored_lstm.py`, `simic/agent/`)
- `state_value_head` (op-independent V(s), PPO baseline): `factored_lstm.py:451-461`; forward `_compute_state_value()` :598-615; init gain 0.1 :552.
- `q_head` (Q(s,op), telemetry/aux, detached): `factored_lstm.py:480-494`; forward `_compute_q()` :617-643.
- All-ops batched Q telemetry: `ppo_agent.py:849-862` → `op_q_values [NUM_OPS]`.
- GAE loop + bootstrap: `rollout_buffer.py:557-590`; bootstrap stored :460; denorm :550-551; `delta`/`returns` :578/:589.
- EV computation: `leyline/value_metrics.py:128-169` `compute_floored_explained_variance()`; called `ppo_agent.py:671`.
- Anchored reference pass (K-epoch frozen θ₀): `ppo_agent.py:935-988` (uses `initial_hidden_h/c`).
- ValueNormalizer (PopArt-lite, no final-layer rescale): `simic/control/normalization.py:256-407`; applied `ppo_agent.py:700-701`.
- Both value heads optimized: `ppo_agent.py:414-420` (`state_value_head` + `q_head` params).

### Telemetry / EV emission (Stage 0)
- 6-file chain to add a PPO-update scalar metric: `simic/agent/types.py` (`PPOUpdateMetrics` TypedDict) → `ppo_agent.py` (compute, ~:671) → `ppo_metrics.py` (`PPOUpdateMetricsBuilder.finalize`, :48-251) → `simic/telemetry/emitters.py` (`emit_ppo_update_event`, :924-1145; emit :1001) → `leyline/telemetry.py` (`PPOUpdatePayload`, :705+) → `karn/mcp/views.py` (`ppo_updates` view, :114-228, `json_extract(data,'$.key')`).
- Per-step reward components: `leyline/telemetry_contracts.py:40-160` (`RewardComponentsTelemetry`, ~40 fields incl. `seed_contribution`, `pbrs_bonus`, `escrow_*`, `compute_rent`, `terminal_bonus`, `total_reward`).
- Reward composed/summed: `contribution.py:377-876` `compute_contribution_reward()` (return = `advantages + values`, `rollout_buffer.py:589`).

### Value head + reward modes (Stage 2)
- **Authoritative entry-point map lives in `stage2-deshape-hra-head.md` §4 (v4, verified 2026-06-24).** The references below are a summary; on any mismatch, the Stage-2 plan wins.
- Add a second, **head-only** (detached-trunk, like `q_head`) value head `cf_value_head`: `factored_lstm.py` __init__ :451-461 (mirror `state_value_head`; declare after the `q_head` Sequential ends :494, before `contribution_predictor` :496), `_init_weights` gain-0.1 tuple :549, new `_compute_cf_value()` (sibling of `_compute_state_value()` :598-615, but **detach `lstm_out`** like `_compute_q` :636), `_ForwardOutput` :88-108, `evaluate_actions()` return tuple :1464, optimizer `critic_params` `ppo_agent.py:418-421`. **Rollout/bootstrap path:** thread `cf_value` through `GetActionResult` (`factored_lstm.py:61`) + `ActionResult` (`leyline/policy_protocol.py:31`) + `lstm_bundle.get_action` :126 + `vectorized_trainer.py:2066/2318` (NOT `ForwardResult`, which is SAC-only). Two version bumps: `CHECKPOINT_VERSION` (`ppo_agent.py:67`) + `VALUE_HEAD_SCHEMA_VERSION` (`leyline/__init__.py:125`).
- RewardMode enum: `contribution.py:92-111`. Dispatch: `rewards.py compute_reward()` :58-185. **No new `RewardMode` is needed** (earlier sketch said `DESHAPED` — superseded by Stage 2 v2): the de-shape is a SUBTRACTIVE partition of the already-emitted reward (`R_cf = bounded_attribution`, `R_main = reward_raw − R_cf`, computed at the `action_execution.py:1069` finalization site), routed to two value heads. The environment reward is unchanged. `SIMPLIFIED` (omits rent, `rewards.py:160`) is NOT the de-shaped target and is unrelated.
- Config/agent construction: `PPOAgent(...)` at `vectorized.py:1145` (NOT `:926-950`, the reward-config block); flag chain `to_train_kwargs` (`config.py:347`) → `train_ppo_vectorized` signature → `PPOAgent`. CLI `--hra-value-decomposition` on `ppo_parser` (`scripts/train.py:431`, NOT the `--reward-mode` block at `:409`).

### Observation space + escrow (Stage 3)
- Obs builder: `tamiyo/policy/features.py:655` `batch_obs_to_features()`; dims = `OBS_V3_BASE_FEATURE_SIZE(23)` + `OBS_V3_SLOT_FEATURE_SIZE(31)*num_slots` (`leyline/__init__.py:644-646`); `get_feature_size()` `features.py:630`.
- **Escrow is NOT in the observation** — `escrow_credit_prev` is only a reward input (`action_execution.py:866,929`). This is the Stage-3 lever.
- Escrow clip (breaks telescoping): `contribution.py:531-536` (`escrow_delta_clip=2.0` :150).
- PBRS telescoping test: `tests/simic/properties/test_pbrs_properties.py:40-98` (asserts Σ shaping == γΦ(s_n)−Φ(s_0) within 1e-9).
- Obs-expansion files to touch: `features.py`, `leyline/__init__.py` (dim const), `simic/training/normalizer_checkpoint.py` (shape contract), `simic/telemetry/observation_stats.py` (base/slot boundaries).

## Review tracking (required before promotion)

| Plan | drl-expert | yzmir-deep-rl | pytorch-expert | axiom-python | reality-check |
|------|-----------|---------------|----------------|--------------|---------------|
| Stage 0 | ⏳ | ⏳ | — | ⏳ | ⏳ |
| Stage 1 (verify) | ⏳ | — | — | — | ✅ (this index) |
| Stage 1b | ⏳ | ⏳ | — | — | ⏳ |
| Stage 2 | ✅ (R5) | ✅ (R5, was R4 major) | ✅ (R5) | ✅ (R5) | ✅ (R5) |
| Stage 3 | ⏳ | ⏳ | — | ⏳ | ⏳ |

Legend: ⏳ pending · ✅ approved/approved-with-changes · `R1✗` round-1 needs-revision · `R1≈` round-1 approved-with-changes · `→v2` findings addressed in the rewrite · `R2⏳` round-2 review in flight. (Stage 2's pytorch column in R2 uses the alternated `yzmir-pytorch-engineering` reviewer; reality-check spans the planning trio.)

Completion criterion (ralph-loop): a detailed plan exists for each open stage and every required reviewer verdict is `approved` / `approved-with-changes` with no `needs-revision` outstanding.
