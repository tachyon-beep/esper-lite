# Leyline/Nissa Contract Audit Findings

Date: 2026-06-13
Scope: read-only audit of Leyline telemetry contracts, Nissa config/output/tracker/analytics/W&B backends, and scoped Leyline/Nissa tests.

Notes:
- Existing source changes were treated as current reality and were not modified.
- Loomweave reported the index as stale, so all issue evidence below is from live filesystem reads, not indexed summaries.

## Producer/Contract/Backend Matrix

| Surface | Producer expectation | Leyline/Nissa contract | Backend behavior | Assessment |
|---|---|---|---|---|
| `TRAINING_STARTED` | Training startup emits run configuration. | `TrainingStartedPayload` requires env count, max epochs/batches, task, host params, slots, seed, PPO hyperparams, devices, and reward mode (`src/esper/leyline/telemetry.py:429-447`). | W&B only updates `n_envs`, `max_epochs`, and `task` (`src/esper/nissa/wandb_backend.py:225-231`). File/Directory preserve payload through serialization. | W&B is partial, not a full experiment contract sink. |
| `EPOCH_COMPLETED` | Per-env epoch telemetry includes env, validation metrics, inner epoch, optional training metrics, per-seed snapshots, observation stats, and episode context. | `episode_idx` is optional on the dataclass (`src/esper/leyline/telemetry.py:541-553`) but required by `from_dict()` (`src/esper/leyline/telemetry.py:555-579`). | File/Directory call payload `to_dict()` (`src/esper/nissa/output.py:75-89`, `src/esper/nissa/output.py:479-506`), but `EpochCompletedPayload.to_dict()` omits `episode_idx` (`src/esper/leyline/telemetry.py:581-601`). | Confirmed field-drop bug. |
| `BATCH_EPOCH_COMPLETED` | Batch-level commit/progress event. | Required aggregate fields plus optional resume/per-env fields; `rolling_accuracy` defaults to `0.0` and `env_accuracies` defaults to `None` (`src/esper/leyline/telemetry.py:604-642`). | Console always prints rolling accuracy (`src/esper/nissa/output.py:387-399`); W&B logs a subset (`src/esper/nissa/wandb_backend.py:276-284`). | Missing rolling accuracy is rendered as a real zero. |
| Seed lifecycle: germinated/stage/fossilized/pruned | Lifecycle producers provide slot/env identity and stage/outcome facts. | Payloads carry required `slot_id` and `env_id` for all lifecycle events (`src/esper/leyline/telemetry.py:1111-1118`, `src/esper/leyline/telemetry.py:1175-1182`, `src/esper/leyline/telemetry.py:1268-1276`, `src/esper/leyline/telemetry.py:1329-1335`). `SeedPrunedPayload.blueprint_id` is optional (`src/esper/leyline/telemetry.py:1337-1339`). | Analytics rejects pruned events when `blueprint_id` is absent (`src/esper/nissa/analytics.py:244-248`). W&B stage/fossilized/pruned handlers require envelope `event.slot_id` instead of the required payload field (`src/esper/nissa/wandb_backend.py:367-394`, `src/esper/nissa/wandb_backend.py:430-432`). | Backend contracts diverge from Leyline payload contracts. |
| Seed outcome attribution | Fossilized/pruned payloads distinguish `None` = not computed from `0.0` = computed zero. | `counterfactual` and `blending_delta` are optional (`src/esper/leyline/telemetry.py:1278-1283`, `src/esper/leyline/telemetry.py:1337-1343`). | Analytics coerces missing `blending_delta` to `0.0` and later reports it in means/tables (`src/esper/nissa/analytics.py:216-228`, `src/esper/nissa/analytics.py:253-266`, `src/esper/nissa/analytics.py:387-404`). | Unknown attribution can become a real-looking zero. |
| `PPO_UPDATE_COMPLETED` | PPO producer emits policy, value, entropy, gradient, ratio, Q-value, infrastructure, LSTM, and slot-saturation diagnostics. | Contract includes critical diagnostics such as `pre_clip_grad_norm` (`src/esper/leyline/telemetry.py:687-690`), ratio stats (`src/esper/leyline/telemetry.py:729-735`), Q-values (`src/esper/leyline/telemetry.py:832-843`), and D5 saturation metrics (`src/esper/leyline/telemetry.py:879-887`). | Console prints only losses/entropy (`src/esper/nissa/output.py:422-434`); W&B logs only a small subset (`src/esper/nissa/wandb_backend.py:308-325`). | W&B is not preserving the contract surface needed for health diagnosis. |
| `ANALYTICS_SNAPSHOT` | Discriminated union by `kind`. | `kind` is required; most fields are kind-dependent optional fields (`src/esper/leyline/telemetry.py:1536-1667`). `from_dict()` also requires `episode_idx` (`src/esper/leyline/telemetry.py:1668-1680`). | Analytics backend validates but does not aggregate snapshots (`src/esper/nissa/analytics.py:281-289`). File/Directory preserve nested telemetry through `to_dict()` (`src/esper/leyline/telemetry.py:1762-1779`). | Mostly real contract, but deserialization requires `episode_idx` even for kinds where it may be semantically irrelevant. |
| Nissa diagnostic tracker | Tracker snapshots expose gradient/loss/class/weight diagnostics from `TelemetryConfig`. | Config flags can disable norm/std/sharpness collection (`src/esper/nissa/config.py:39-52`, `src/esper/nissa/config.py:63-71`). | Disabled `norm`/`std` remain `0.0` and are serialized as if measured (`src/esper/nissa/tracker.py:31-50`, `src/esper/nissa/tracker.py:211-214`, `src/esper/nissa/tracker.py:226-235`). Tests codify this zero placeholder (`tests/nissa/test_tracker.py:204-228`, `tests/nissa/test_tracker.py:234-257`). | Non-computed metrics are indistinguishable from computed zero. |
| Telemetry levels | CLI selects `off/minimal/normal/debug`. | Runtime level gating is in Simic `TelemetryConfig.should_collect()` (`src/esper/simic/telemetry/telemetry_config.py:13-19`, `src/esper/simic/telemetry/telemetry_config.py:61-75`). | Training maps level to Simic config and only maps `debug` to console severity `debug`; Nissa `TelemetryConfig` profiles are diagnostic feature profiles, not event-level gates (`src/esper/scripts/train.py:84-88`, `src/esper/scripts/train.py:520-534`, `src/esper/nissa/config.py:82-107`). | Real behavior is split: Simic gates emission; Nissa output filters display severity only. |
| Global hub reset | Tests and repeated runs need no stale backends/events. | `reset_hub()` reuses the singleton and calls `NissaHub.reset()` (`src/esper/nissa/output.py:906-931`). | `NissaHub.reset()` closes, clears backend lists/workers, reopens, replaces queue, and resets counters (`src/esper/nissa/output.py:813-832`). Test covers no duplicate backend delivery after reset (`tests/nissa/test_global_hub_reset.py:6-40`). | No confirmed reset bug in scoped source. |

## Confirmed Issues

### LN-001 - `EpochCompletedPayload.to_dict()` drops `episode_idx`

Severity: P1

`EpochCompletedPayload` declares `episode_idx` as episode context (`src/esper/leyline/telemetry.py:541-543`) and `from_dict()` requires it (`src/esper/leyline/telemetry.py:555-579`), but `to_dict()` does not emit it (`src/esper/leyline/telemetry.py:581-601`). Nissa file and directory backends serialize via `_payload_to_dict()` and payload `to_dict()` (`src/esper/nissa/output.py:75-89`, `src/esper/nissa/output.py:479-506`), so persisted JSONL can lose the same field the contract then requires for loading.

Real-vs-placeholder assessment: real data loss. This is not a placeholder display issue; serialized events lose context and cannot round-trip through the declared parser.

Test coverage: `tests/leyline/test_telemetry.py:75-90` covers `from_dict()` accepting `observation_stats=None` when `episode_idx` is present, but there is no `to_dict()`/file-output round-trip test that would catch the missing key.

Acceptance tests:
- Add a Leyline round-trip test proving `EpochCompletedPayload(..., episode_idx=7).to_dict()["episode_idx"] == 7`.
- Add a Nissa file-output JSONL test proving `episode_idx` is present under `data`.

### LN-002 - Analytics rejects `SeedPrunedPayload.blueprint_id=None` even though Leyline marks it optional

Severity: P1

Leyline explicitly allows pruned seed payloads with unknown blueprint identity: `blueprint_id: str | None = None` (`src/esper/leyline/telemetry.py:1337-1339`), and `from_dict()` comments that blueprint may be unknown if pruning very early (`src/esper/leyline/telemetry.py:1360-1362`). `BlueprintAnalytics.emit()` raises `ValueError` when the same field is `None` (`src/esper/nissa/analytics.py:244-248`).

Real-vs-placeholder assessment: real contract/backend mismatch. A valid Leyline payload can crash one Nissa backend.

Test coverage: analytics tests only use pruned payloads with a blueprint id (`tests/nissa/test_analytics.py:134-154`). There is no acceptance test for the documented unknown-blueprint case.

Acceptance tests:
- Add `SeedPrunedPayload(slot_id=..., env_id=..., reason=..., blueprint_id=None)` through `BlueprintAnalytics.emit()`.
- Assert the backend records prune count without attributing it to a named blueprint, or define the contract to reject unknown blueprints in Leyline instead.

### LN-003 - Missing attribution fields are rendered as computed zeroes

Severity: P1

Leyline distinguishes absent attribution from computed zero with nullable fields: `counterfactual` and `blending_delta` are optional on fossilized and pruned payloads (`src/esper/leyline/telemetry.py:1278-1283`, `src/esper/leyline/telemetry.py:1337-1343`). Analytics coerces missing `blending_delta` to `0.0` (`src/esper/nissa/analytics.py:216-218`, `src/esper/nissa/analytics.py:253-255`), appends that zero into aggregate distributions (`src/esper/nissa/analytics.py:224-228`, `src/esper/nissa/analytics.py:262-266`), and prints aggregate means as numeric facts (`src/esper/nissa/analytics.py:387-404`). Empty `counterfactuals` also report `0.0` via `mean_counterfactual` (`src/esper/nissa/analytics.py:98-101`).

Real-vs-placeholder assessment: placeholder/absence becomes a real-looking fact. A missing blending analysis is indistinguishable from "exactly 0.00% blending delta" in the analytics table.

Test coverage: current tests provide explicit `counterfactual=0.0` and omit `blending_delta` without asserting absence semantics (`tests/nissa/test_analytics.py:116-124`, `tests/nissa/test_analytics.py:140-148`). That means the test suite accepts placeholder zeros.

Acceptance tests:
- Emit fossilized/pruned payloads with `blending_delta=None` and assert analytics does not append `0.0` to measured blending deltas.
- Assert summary output renders missing means as `n/a` or omits the column value until at least one measured value exists.

### LN-004 - Batch rolling accuracy defaults are displayed as measured progress

Severity: P2

`BatchEpochCompletedPayload` marks resume/progress extras as optional defaults: `start_episode=0`, `requested_episodes=0`, `rolling_accuracy=0.0`, and `env_accuracies=None` (`src/esper/leyline/telemetry.py:616-620`). `from_dict()` also defaults missing `rolling_accuracy` to `0.0` (`src/esper/leyline/telemetry.py:637-641`). `ConsoleOutput` always prints `rolling: {rolling_acc:.1f}%` when formatting the event (`src/esper/nissa/output.py:387-399`).

Real-vs-placeholder assessment: placeholder zero can be rendered as an observed rolling accuracy. This is less severe than field loss, but it makes console output evidence ambiguous.

Test coverage: console tests pass explicit rolling accuracy and assert formatting (`tests/nissa/test_console_output.py:39-58`); they do not cover absent rolling accuracy.

Acceptance tests:
- Construct `BatchEpochCompletedPayload` without `rolling_accuracy` and assert console output does not show `rolling: 0.0%` as a measured value.
- If `rolling_accuracy` is truly required at runtime, make it required in the Leyline payload and remove the default.

### LN-005 - DiagnosticTracker serializes disabled gradient metrics as zero

Severity: P2

`GradientStats` initializes `norm`, `std`, and `mean` to `0.0` and always serializes them (`src/esper/nissa/tracker.py:31-50`). `_record_grad()` documents that disabled `track_norm` and `track_std` leave those fields at `0.0` (`src/esper/nissa/tracker.py:211-214`) and only computes them when the flags are enabled (`src/esper/nissa/tracker.py:226-235`). Tests intentionally assert that disabled metrics stay at `0.0` (`tests/nissa/test_tracker.py:204-228`, `tests/nissa/test_tracker.py:234-257`).

Real-vs-placeholder assessment: placeholder/non-computed values are serialized as real scalar facts. A disabled norm collection path is indistinguishable from a measured zero gradient norm in `GradientStats.to_dict()`.

Test coverage: coverage currently locks in the placeholder behavior instead of requiring an explicit "not collected" marker.

Acceptance tests:
- Assert disabled `track_norm`/`track_std` serialize as `None`, omit the field, or include a collection flag such as `norm_collected=False`.
- Assert computed zero and not-computed are distinguishable in `EpochSnapshot.to_dict()`.

### LN-006 - W&B backend drops most PPO and training-start contract fields

Severity: P2

`PPOUpdatePayload` includes health-critical fields such as `pre_clip_grad_norm` (`src/esper/leyline/telemetry.py:687-690`), ratio/log-prob stats (`src/esper/leyline/telemetry.py:729-735`), op-conditioned Q-values (`src/esper/leyline/telemetry.py:832-843`), and slot-saturation diagnostics (`src/esper/leyline/telemetry.py:879-887`). W&B logs only policy loss, value loss, entropy, KL, clip fraction, grad norm, and optional explained variance/LR/entropy coefficient (`src/esper/nissa/wandb_backend.py:308-325`). Likewise `TrainingStartedPayload` requires many run identity fields (`src/esper/leyline/telemetry.py:429-447`), but W&B config updates only `n_envs`, `max_epochs`, and `task` (`src/esper/nissa/wandb_backend.py:225-231`).

Real-vs-placeholder assessment: real dropped-field issue for W&B as an experiment-tracking backend. File/Directory preserve fuller payloads, but W&B does not preserve enough of the declared contract for later health analysis.

Test coverage: W&B tests focus on step alignment and filtering `None` values (`tests/nissa/test_wandb_backend.py:52-175`, `tests/nissa/test_wandb_backend.py:367-420`). They do not assert preservation of critical PPO diagnostics.

Acceptance tests:
- Add W&B tests requiring `pre_clip_grad_norm`, ratio stats, `nan_grad_count`, Q variance/spread, and D5 saturation metrics when present.
- Add W&B training-start config tests requiring stable run identity: seed, reward mode, task, slot ids, devices, and param budget.

### LN-007 - W&B lifecycle handlers ignore required payload `slot_id` for several events

Severity: P2

Lifecycle payloads require `slot_id`, while the event envelope `slot_id` is optional (`src/esper/leyline/telemetry.py:127-129`, `src/esper/leyline/telemetry.py:1175-1182`, `src/esper/leyline/telemetry.py:1268-1276`, `src/esper/leyline/telemetry.py:1329-1335`). W&B germination correctly falls back to payload `slot_id` (`src/esper/nissa/wandb_backend.py:337-343`), but stage-change, fossilized, and pruned handlers require `event.slot_id` and raise when the envelope omits it (`src/esper/nissa/wandb_backend.py:367-394`, `src/esper/nissa/wandb_backend.py:430-432`).

Real-vs-placeholder assessment: real backend mismatch. Typed payloads that satisfy Leyline can still fail W&B because the backend reads a weaker optional envelope field.

Test coverage: W&B tests cover germination with both envelope and payload slot id (`tests/nissa/test_wandb_backend.py:104-124`), but do not cover stage/fossilized/pruned payload-only slot identity.

Acceptance tests:
- Add W&B tests for `SeedStageChangedPayload`, `SeedFossilizedPayload`, and `SeedPrunedPayload` with `event.slot_id=None` and payload `slot_id` present.
- Require the backend to use payload `slot_id` consistently for all typed lifecycle payloads.

## Coverage Notes

Covered:
- Nissa config rejects unknown YAML/profile keys and invalid profile structure (`tests/nissa/test_config.py`).
- Directory output creates timestamped folders and writes basic JSONL events (`tests/nissa/test_output.py:19-90`).
- Hub close/flush/reset behavior has explicit async and duplicate-backend regression coverage (`tests/nissa/test_output.py:130-231`, `tests/nissa/test_output.py:323-672`, `tests/nissa/test_global_hub_reset.py:6-40`).
- W&B step alignment, missing epoch skip behavior, and optional `None` filtering are covered (`tests/nissa/test_wandb_backend.py:52-175`, `tests/nissa/test_wandb_backend.py:367-420`).
- Nested `AnalyticsSnapshotPayload` dataclasses round-trip through Karn serialization (`tests/leyline/test_telemetry.py:27-73`, `tests/leyline/test_telemetry.py:365-440`).

Gaps:
- No all-payload round-trip test that requires `payload.to_dict()` and `payload.from_dict()` to preserve every contract field.
- No FileOutput/DirectoryOutput test for `episode_idx` preservation inside `EpochCompletedPayload`.
- No tests that assert absent optional analytics fields stay absent rather than becoming `0.0`.
- No analytics test for the documented early-prune `blueprint_id=None` case.
- No W&B tests for critical PPO diagnostics beyond the small metric subset.
- No W&B lifecycle tests requiring payload `slot_id` fallback for stage/fossilized/pruned events.
- No tracker test distinguishing "not collected" from "measured zero" for disabled gradient metrics.

## Tracker-Ready Issue Rows

| Priority | Title | Scope | Acceptance criteria |
|---|---|---|---|
| P1 | Preserve `EpochCompletedPayload.episode_idx` through serialization | `src/esper/leyline/telemetry.py`, `tests/leyline/`, `tests/nissa/` | `to_dict()` includes `episode_idx`; FileOutput JSONL includes it; `from_dict(to_dict(payload))` round-trips with `episode_idx` intact. |
| P1 | Align `SeedPrunedPayload.blueprint_id` optionality with `BlueprintAnalytics` | `src/esper/leyline/telemetry.py`, `src/esper/nissa/analytics.py`, `tests/nissa/test_analytics.py` | Either Leyline makes `blueprint_id` required or analytics accepts `None` without crashing; tests cover early-prune unknown blueprint. |
| P1 | Stop converting missing attribution metrics into zero-valued analytics facts | `src/esper/nissa/analytics.py`, `tests/nissa/test_analytics.py` | Missing `blending_delta`/`counterfactual` are excluded or rendered as `n/a`; computed `0.0` remains distinguishable from absent. |
| P2 | Make batch rolling accuracy absence explicit in console output | `src/esper/leyline/telemetry.py`, `src/esper/nissa/output.py`, `tests/nissa/test_console_output.py` | If rolling accuracy is absent, console does not print `rolling: 0.0%`; if required, payload construction/deserialization fails without it. |
| P2 | Preserve not-collected state for diagnostic gradient metrics | `src/esper/nissa/tracker.py`, `tests/nissa/test_tracker.py` | Disabled norm/std serialize as absent/`None` or with explicit collected flags; measured zero remains representable. |
| P2 | Expand W&B payload preservation for PPO and training-start diagnostics | `src/esper/nissa/wandb_backend.py`, `tests/nissa/test_wandb_backend.py` | W&B logs pre-clip gradient norm, nan/inf counts, ratio/log-prob stats, Q spread/variance, D5 metrics, and core training-start identity when present. |
| P2 | Use payload `slot_id` consistently in W&B lifecycle handlers | `src/esper/nissa/wandb_backend.py`, `tests/nissa/test_wandb_backend.py` | Stage, fossilized, and pruned events succeed when `event.slot_id is None` and typed payload `slot_id` is present. |
