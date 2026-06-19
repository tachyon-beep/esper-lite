# Tamiyo Policy Diagnostics Audit Findings

Audit date: 2026-06-13
Scope: `src/esper/tamiyo/`, `src/esper/simic/training/vectorized_trainer.py` decision telemetry call sites, relevant Leyline action/mask contracts, `tests/tamiyo/`, `tests/telemetry/test_decision_metrics.py`, and `tests/integration/test_q_values_telemetry.py`.
Mode: read-only source audit. Existing modified files were treated as current user-owned reality.

## Policy Diagnostic Feed Inventory

| Feed | Producer | Consumer / test coverage | Real vs placeholder assessment |
|---|---|---|---|
| Obs V3 base features: epoch, val loss, val accuracy, 5-loss history, 5-accuracy history, stage distribution, last action success/op | `src/esper/tamiyo/policy/features.py:623-697` | `tests/tamiyo/policy/test_features.py:196-266`, `tests/tamiyo/policy/test_features.py:377-422` | Real current scalar telemetry, but history missingness is left-padded as zero by `_pad_history` at `src/esper/tamiyo/policy/features.py:133-145`, so early/missing history is not distinguishable from real zero history. |
| Obs V3 per-slot lifecycle/alpha/contribution/gradient features | `src/esper/tamiyo/policy/features.py:698-794` | `tests/tamiyo/policy/test_features.py:269-331`, `tests/tamiyo/properties/test_feature_extraction_properties.py:288-363` | Mostly real `SeedStateReport` data. Missing `counterfactual_contribution` becomes neutral `0.0` at `src/esper/tamiyo/policy/features.py:721-725`; missing current telemetry becomes unhealthy zeros at `src/esper/tamiyo/policy/features.py:749-767`, which is safer than healthy defaults. |
| Counterfactual freshness feature | `src/esper/tamiyo/policy/features.py:786-790`, backed by `env_state.epochs_since_counterfactual` | `tests/tamiyo/policy/test_features.py:473-534` | Real staleness counter when a measurement exists, but newly tracked slots are initialized as fresh before proof exists (`src/esper/simic/training/parallel_env_state.py:274-277`, `src/esper/simic/training/handlers/germinate.py:114-115`). |
| Action masks and op-conditioned slot masks | `src/esper/tamiyo/policy/action_masks.py:148-359` | `tests/tamiyo/policy/test_action_masks.py:42-220`, `tests/tamiyo/properties/test_mask_properties.py` | Real physical validity masks; fail-fast on missing enabled slots at `src/esper/tamiyo/policy/action_masks.py:218-226`. Irrelevant heads intentionally canonicalize to single defaults in the network. |
| Per-head rollout confidence / entropy | `src/esper/tamiyo/networks/factored_lstm.py:1012-1015`, `src/esper/tamiyo/networks/factored_lstm.py:1061-1064`, returned via `src/esper/tamiyo/policy/lstm_bundle.py:125-133` | `tests/telemetry/test_decision_metrics.py:271-384` | Real normalized per-head entropy/confidence when `ops_telemetry_enabled` is active; disconnected when telemetry is disabled because `head_telemetry` is omitted, not emitted as a false zero. |
| Decision snapshot: action confidence, alternatives, raw op entropy, value estimate, slot states | `src/esper/simic/training/action_execution.py:1299-1363`, emitted by `src/esper/simic/telemetry/emitters.py:242-369` | `tests/telemetry/test_decision_metrics.py:166-234`, `tests/telemetry/test_decision_metrics.py:387-512`, `tests/telemetry/test_decision_metrics.py:925-1030` | Real snapshot fields when op probabilities are available. `decision_entropy` is raw op entropy (`src/esper/simic/training/action_execution.py:1335-1340`), while `HeadTelemetry.op_entropy` is normalized, so consumers must not compare them directly. |
| Per-head mask flags in decision snapshot | `src/esper/simic/training/vectorized_trainer.py:1442-1452`, copied in `src/esper/simic/training/action_execution.py:1274-1292` | `tests/telemetry/test_decision_metrics.py:714-766` | Real "some action was masked out" signal, but mislabeled by Leyline as "forced by mask"; see Finding TPD-001. |
| PPO op-conditioned Q telemetry | `src/esper/simic/agent/ppo_agent.py:680-725`, serialized by `src/esper/simic/telemetry/emitters.py:952-956` | `tests/integration/test_q_values_telemetry.py:16-176` | Real value head calls, but recurrent context is placeholder initial hidden state, not the sampled rollout hidden state; see Finding TPD-002. |
| LSTM hidden-state health | `src/esper/simic/telemetry/lstm_health.py:93-204`, collected at `src/esper/simic/agent/ppo_agent.py:823-836` | `tests/simic/telemetry/test_lstm_health.py:113-195` | Real diagnostics when recurrent evaluation returns hidden state. Missing history aggregates to `None` in `src/esper/simic/agent/ppo_metrics.py:191-200`; avoid downstream false healthy defaults. |

## Findings

### TPD-001 - Decision mask flags report "some options masked" as "forced choice"

Severity: P1

Evidence:

- Leyline defines last-action mask flags as forced-choice indicators: `src/esper/leyline/telemetry.py:1580-1588` says `True = action was forced by mask`.
- The trainer computes each flag as `~masks_batch[key].all(dim=-1)` at `src/esper/simic/training/vectorized_trainer.py:1442-1452`, which is true whenever any candidate is invalid.
- That means normal masks such as "NOOP blueprint disabled" or "only enabled slots selectable" become `*_masked=True` even when the sampled head still had multiple valid choices. The test repeats the forced wording at `tests/telemetry/test_decision_metrics.py:714-715` but only checks propagation of supplied booleans at `tests/telemetry/test_decision_metrics.py:742-766`.

Real-vs-placeholder assessment:

- The signal is real as a restriction detector.
- It is not a real forced-choice detector. It overstates loss of agency and can contaminate Tamiyo decision diagnostics and UI interpretations.

Proposed acceptance tests:

- Add a trainer/action-execution test where `blueprint_mask` has multiple valid choices but one invalid NOOP. Expected: `blueprint_masked` is false if the field keeps its "forced" meaning.
- Add a WAIT-only op-mask case with exactly one valid op. Expected: `op_masked` or a renamed `op_forced` is true.
- If "some options masked" is still useful, add a separate field such as `op_restricted` instead of reusing forced semantics.

### TPD-002 - Op Q-value telemetry recomputes with initial recurrent hidden state

Severity: P1

Evidence:

- PPO Q telemetry selects the first valid observation row at `src/esper/simic/agent/ppo_agent.py:680-689`.
- It calls `self.policy.network.forward(..., hidden=None)` at `src/esper/simic/agent/ppo_agent.py:697-703`, explicitly using the network's initial hidden state.
- The same PPO update path already has the actual rollout initial hidden tensors and passes them into recurrent evaluation at `src/esper/simic/agent/ppo_agent.py:800-818`.
- The Q vector is then computed from that initial-hidden LSTM output at `src/esper/simic/agent/ppo_agent.py:704-725`.
- The integration test only asserts presence, shape, finiteness, and non-negative spread/variance at `tests/integration/test_q_values_telemetry.py:121-130` and `tests/integration/test_q_values_telemetry.py:155-161`; it does not assert recurrent-state fidelity.

Real-vs-placeholder assessment:

- The Q values are produced by the real op-conditioned value head.
- The recurrent context is placeholder for a recurrent policy. The resulting `op_q_values`, `q_variance`, and `q_spread` can be disconnected from the policy state that actually produced the sampled decision sequence.

Proposed acceptance tests:

- Construct two identical observations with different stored `initial_hidden_h/c`; make `_compute_value` or the LSTM output hidden-sensitive; assert Q telemetry changes with hidden state.
- Assert Q telemetry uses the sampled row's stored `initial_hidden_h/c` from the rollout buffer, not `hidden=None`.
- Keep the existing finiteness/shape integration test, but add a regression that fails if `forward(... hidden=None)` is used for recurrent Q diagnostics.

### TPD-003 - New seed tracking initializes absent gradient/counterfactual evidence as healthy/fresh

Severity: P1

Evidence:

- New germinated seeds call `init_obs_v3_slot_tracking()` immediately after creation at `src/esper/simic/training/handlers/germinate.py:103-115`.
- That initializer sets `gradient_health_prev[slot_id] = 1.0` and `epochs_since_counterfactual[slot_id] = 0` at `src/esper/simic/training/parallel_env_state.py:274-277`.
- Feature extraction reads `gradient_health_prev` directly into the policy observation at `src/esper/tamiyo/policy/features.py:769-780`.
- Feature extraction also reports freshness as `DEFAULT_GAMMA ** epochs_since_counterfactual`, so an initialized `0` becomes `1.0` at `src/esper/tamiyo/policy/features.py:786-790`.
- Actual counterfactual proof only resets freshness after a solo ablation contribution is computed at `src/esper/simic/training/vectorized_trainer.py:998-1027`.
- Existing tests cover missing tracking as stale (`tests/tamiyo/policy/test_features.py:512-534`) and missing contribution as neutral (`tests/tamiyo/policy/test_features.py:537-562`), but not "tracking present at zero while contribution is still absent."

Real-vs-placeholder assessment:

- Current gradient telemetry is safely zeroed when absent (`src/esper/tamiyo/policy/features.py:749-767`).
- Previous gradient health and counterfactual freshness for a newly germinated seed are placeholders that look healthy/fresh before evidence exists.

Proposed acceptance tests:

- After germination but before any gradient telemetry or solo counterfactual measurement, assert the policy observation does not encode `gradient_health_prev=1.0` or `counterfactual_fresh=1.0` as evidence.
- Add a feature test where `epochs_since_counterfactual={"r0c0": 0}` and `counterfactual_contribution is None`; expected outcome should not look like a fresh measured zero contribution.
- Prefer an explicit measured flag/freshness contract over healthy/fresh defaults.

### TPD-004 - `HeadTelemetry.from_dict()` silently fills missing diagnostic fields with zeros

Severity: P2

Evidence:

- `HeadTelemetry` defaults every confidence and entropy field to `0.0` at `src/esper/leyline/telemetry.py:1457-1485`.
- `HeadTelemetry.from_dict()` uses `.get(..., 0.0)` for every field at `src/esper/leyline/telemetry.py:1487-1512`.
- `AnalyticsSnapshotPayload.from_dict()` parses `head_telemetry` through that helper at `src/esper/leyline/telemetry.py:1757-1800`.
- The production path correctly omits `head_telemetry` when head evidence is absent (`src/esper/simic/training/action_execution.py:149-158`), but a partial serialized payload can become a complete-looking all-zero diagnostic.

Real-vs-placeholder assessment:

- In live rollout, per-head telemetry is real when provided.
- During deserialization of partial data, missing head diagnostics are converted into real-looking zero confidence/entropy. Zero entropy is semantically meaningful, so this masks absence.

Proposed acceptance tests:

- `HeadTelemetry.from_dict({})` should fail fast or preserve missingness instead of producing an all-zero object.
- `AnalyticsSnapshotPayload.from_dict({"kind": "last_action", "episode_idx": 0, "head_telemetry": {"op_entropy": 0.5}})` should not silently default the other heads to zero.
- If backward data compatibility is required outside source, migrate historical rows explicitly rather than defaulting in the typed contract.

### TPD-005 - Observation history padding hides early/missing history as real zero values

Severity: P2

Evidence:

- `_pad_history()` left-pads short history with `0.0` at `src/esper/tamiyo/policy/features.py:133-145`.
- The vectorized feature path writes padded loss and accuracy histories directly into the policy observation at `src/esper/tamiyo/policy/features.py:675-683`.
- The property test locks this behavior in at `tests/tamiyo/properties/test_feature_extraction_properties.py:59-76`.

Real-vs-placeholder assessment:

- Existing history points are real.
- Padded history points are placeholders with no missingness indicator. For loss, `0.0` can look like perfect loss; for accuracy, `0.0` can look like real zero accuracy. The LSTM can infer early episode position from epoch, but cannot distinguish "missing history" from "observed zero history" in the history channels themselves.

Proposed acceptance tests:

- For a one-point history, assert the observation carries either a history-validity mask/count or a sentinel that cannot be confused with valid metric zero.
- Add a test that distinguishes `[0.0]` as an observed history value from `[]` as no history.

## Tracker-Ready Issue Rows

| Priority | Title | Evidence | Acceptance |
|---|---|---|---|
| P1 | Fix Tamiyo decision mask flags so forced-choice telemetry is not inflated by ordinary restrictions | `src/esper/simic/training/vectorized_trainer.py:1442-1452`; contract at `src/esper/leyline/telemetry.py:1580-1588` | Last-action forced flags are true only when the selected head has one valid action after canonical/op-conditioned masking, or the fields are renamed/split into forced vs restricted. Regression tests cover multi-valid restricted masks and WAIT-only masks. |
| P1 | Compute recurrent op Q-value telemetry with the rollout row's hidden state | `src/esper/simic/agent/ppo_agent.py:697-725`; recurrent hidden exists at `src/esper/simic/agent/ppo_agent.py:800-818` | Q diagnostics use the stored hidden state for the sampled observation row. A test proves identical observations with different hidden states produce different Q telemetry when the value path is hidden-sensitive. |
| P1 | Stop initializing new seed Obs V3 diagnostics as healthy/fresh before evidence exists | `src/esper/simic/training/parallel_env_state.py:274-277`; feature reads at `src/esper/tamiyo/policy/features.py:769-790`; proof reset at `src/esper/simic/training/vectorized_trainer.py:998-1027` | Newly germinated seeds do not expose `gradient_health_prev=1.0` or `counterfactual_fresh=1.0` until corresponding telemetry/proof exists. Tests cover post-germination pre-counterfactual observations. |
| P2 | Make `HeadTelemetry` deserialization preserve missingness instead of defaulting to zeros | `src/esper/leyline/telemetry.py:1457-1512`, `src/esper/leyline/telemetry.py:1757-1800` | Partial or empty `head_telemetry` dictionaries fail fast or retain `None`/missing fields. Tests reject silent all-zero telemetry from incomplete input. |
| P2 | Add explicit missingness for short observation histories | `src/esper/tamiyo/policy/features.py:133-145`, `src/esper/tamiyo/policy/features.py:675-683` | Policy observations distinguish absent history from observed zero-valued history. Tests cover empty, one-point zero, and full history cases. |

