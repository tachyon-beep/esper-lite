# Karn Analytics and Proof Audit

Read-only audit performed against current working tree source on 2026-06-13. Existing modified files were treated as current user-owned reality; no source files were changed.

## Scope

- Audited Karn collection/storage/import/export: `src/esper/karn/collector.py`, `src/esper/karn/store.py`, `src/esper/karn/ingest.py`, `src/esper/karn/serialization.py`.
- Audited derived metrics and proof surfaces: `src/esper/karn/health.py`, `src/esper/karn/pareto.py`, `src/esper/karn/mcp/`, `scripts/proof_packet.py`.
- Audited relevant tests under `tests/karn/`, `tests/karn/mcp/`, `tests/telemetry/`, plus the proof packet tests in `tests/scripts/test_proof_packet.py` because `scripts/proof_packet.py` is in scope.
- Used Loomweave only for orientation. Its index reported stale, so all evidence below is from direct current source reads.

## Karn/proof feed inventory

| Feed | Source | Consumer | Real vs placeholder assessment |
| --- | --- | --- | --- |
| `raw_events` | `read_json_auto('{telemetry_dir}/*/events.jsonl')` in `src/esper/karn/mcp/views.py:12-48` | All DuckDB/MCP views and proof packet | Real event envelope when parsed, but completeness is not proven because malformed rows are ignored (`ignore_errors=true`, `src/esper/karn/mcp/views.py:43-45`). |
| `runs` | `TRAINING_STARTED` fields in `src/esper/karn/mcp/views.py:49-73` | `list_runs`, `run_overview`, proof cohorts | Real run metadata. Optional `proof_baseline_mode`/`proof_baseline_pair_id` are preserved when present (`src/esper/karn/mcp/views.py:69-70`). Missing runs currently do not block proof packet verdicts. |
| `epochs` | `EPOCH_COMPLETED` in `src/esper/karn/mcp/views.py:74-91` | MCP overview and per-env accuracy/loss | Real per-env epoch feed. KarnCollector can synthesize a minimal batch fallback with default `val_loss=0.0` when no epoch events arrive (`src/esper/karn/collector.py:449-463`), so store-derived epochs are not always proof-grade. |
| `ppo_updates` | `PPO_UPDATE_COMPLETED` in `src/esper/karn/mcp/views.py:92-182` | Learnability gate, policy health | Real PPO diagnostics when emitted. Proof packet correctly blocks when learnability columns are null (`scripts/proof_packet.py:44-55`, `scripts/proof_packet.py:132`). |
| `batch_epochs` / `batch_stats` | `BATCH_EPOCH_COMPLETED` and `ANALYTICS_SNAPSHOT(kind=batch_stats)` in `src/esper/karn/mcp/views.py:183-202`, `src/esper/karn/mcp/views.py:454-476` | Progress, reproduction command derivation | Real aggregate progress. Not a substitute for `EPISODE_OUTCOME` ROI evidence. |
| `seed_lifecycle` | Seed lifecycle event rows in `src/esper/karn/mcp/views.py:223-267` | Lifecycle efficiency section | Real lifecycle counts and identity fields, including morphology IDs and RNG fields (`src/esper/karn/mcp/views.py:255-259`). |
| `morphology_causal_log` | `MORPHOLOGY_CAUSAL_LOG` in `src/esper/karn/mcp/views.py:268-298` | Causal proof joins | Real causal identity feed when emitted. Not currently used by proof packet. |
| `decisions` / `rewards` | `ANALYTICS_SNAPSHOT(kind=last_action)` in `src/esper/karn/mcp/views.py:299-355`, `src/esper/karn/mcp/views.py:397-453` | Action and reward diagnostics | Real decision/reward rows when payloads include nested reward components. KarnCollector's in-memory `PolicySnapshot` only preserves a small subset (`src/esper/karn/collector.py:598-604`). |
| `run_confounders` | Derived anomaly ledger in `src/esper/karn/mcp/views.py:518-554` | Proof packet blocking ledger | Real for selected numerical/PPO pathologies, but incomplete: `GOVERNOR_ROLLBACK` is in `anomalies` and absent from `run_confounders`; `REWARD_HACKING_SUSPECTED` and `PERFORMANCE_DEGRADATION` are not represented. |
| `episode_outcomes` | `EPISODE_OUTCOME` rows in `src/esper/karn/mcp/views.py:555-573` | Accuracy ROI and Pareto proof | Real multi-objective proof source. Missing rows currently produce "No `EPISODE_OUTCOME` events found." but do not block proof verdicts (`scripts/proof_packet.py:277-287`). |
| `pareto` | `extract_pareto_frontier` / `compute_hypervolume_2d` in `src/esper/karn/pareto.py:15-94` | Sanctum/Karn Pareto analytics | Real computation over `EpisodeOutcome`; output validity depends on correct `param_ratio` semantics. |
| `health` / `VitalSigns` | Recent store snapshots in `src/esper/karn/health.py:210-325`, `src/esper/karn/health.py:396-466` | Operator health | Derived live health, not proof evidence. Missing/zero gradients are treated as absent (`src/esper/karn/health.py:277-285`). |
| `TelemetryStore.export_jsonl` | Context, baseline, bounded epochs, bounded dense traces in `src/esper/karn/store.py:455-509` | Store roundtrip/export | Not proof-complete: no outcome collection exists in store export. |
| `TelemetryStore.import_from_nissa_dir` | Raw Nissa `events.jsonl` importer in `src/esper/karn/store.py:856-931` | Store reconstruction | Placeholder/lossy importer. It reconstructs only start, epoch, and last-action policy subset; it ignores proof-critical event families. |

## Findings

### KARN-PROOF-001 - High - Proof packet fails open when there are no runs or no episode outcomes

`build_proof_packet()` sets `verdict` to `BLOCKED` only when there are blocking confounders, missing learnability rows, or missing optional blueprint-health baselines (`scripts/proof_packet.py:159-163`). If `runs` is empty, it prints "No `TRAINING_STARTED` events found." (`scripts/proof_packet.py:174-185`) but keeps a `REVIEW` verdict. If `episode_outcomes` is empty, it prints "No `EPISODE_OUTCOME` events found." (`scripts/proof_packet.py:277-287`) and still keeps `REVIEW`.

This is proof-invalidating: `EPISODE_OUTCOME` is the contract carrying final accuracy, parameter ratio, stability, reward, and reward mode (`src/esper/leyline/episode_outcome.py:21-40`; `src/esper/karn/mcp/views.py:555-573`). A packet without those rows has no reward-efficiency evidence to review.

Existing tests cover blocking confounders, missing learnability, and optional baseline cohorts (`tests/scripts/test_proof_packet.py:72-214`), but there is no test for empty telemetry, missing `TRAINING_STARTED`, or missing `EPISODE_OUTCOME`.

Tracker-ready row:

| Field | Value |
| --- | --- |
| Title | Make proof packet fail closed on missing proof evidence |
| Severity | P1 / High |
| Files | `scripts/proof_packet.py`, `tests/scripts/test_proof_packet.py` |
| Acceptance tests | Empty telemetry dir returns `Verdict: BLOCKED`; telemetry with PPO updates but no `TRAINING_STARTED` returns `BLOCKED`; telemetry with runs but zero `EPISODE_OUTCOME` rows returns `BLOCKED`; packet explains the missing evidence explicitly. |

### KARN-PROOF-002 - High - Proof confounder ledger omits rollback and reward-hacking classes

`anomalies` includes `GOVERNOR_ROLLBACK` (`src/esper/karn/mcp/views.py:495-517`), and Leyline defines it as an emergency rollback event (`src/esper/leyline/telemetry.py:100-102`). The proof ledger does not include it: `run_confounders` only selects value collapse, ratio collapse/explosion, gradient anomaly/pathology, and numerical instability (`src/esper/karn/mcp/views.py:533-554`). The proof packet then queries only `run_confounders WHERE proof_blocking` (`scripts/proof_packet.py:123-131`).

`REWARD_HACKING_SUSPECTED` and `PERFORMANCE_DEGRADATION` also exist as telemetry event types (`src/esper/leyline/telemetry.py:86-91`) but are absent from both `anomalies` and `run_confounders`. For reward-efficiency and signal-of-life claims, those are not benign informational events; they are direct confounders.

Tracker-ready row:

| Field | Value |
| --- | --- |
| Title | Treat rollback, reward-hacking, and degradation events as proof confounders |
| Severity | P1 / High |
| Files | `src/esper/karn/mcp/views.py`, `scripts/proof_packet.py`, `tests/karn/mcp/test_views.py`, `tests/scripts/test_proof_packet.py` |
| Acceptance tests | `GOVERNOR_ROLLBACK` appears in `run_confounders` with `proof_blocking=true`; `REWARD_HACKING_SUSPECTED` and `PERFORMANCE_DEGRADATION` appear in a confounder view with enough payload detail to debug; proof packet verdict is `BLOCKED` for each event class. |

### KARN-PROOF-003 - High - JSONL parse errors are silently skipped before proof gating

The root proof view reads telemetry with `ignore_errors=true` (`src/esper/karn/mcp/views.py:27-47`). Neither `create_views()` nor `build_proof_packet()` records skipped rows or blocks on ingestion integrity (`src/esper/karn/mcp/views.py:688-714`; `scripts/proof_packet.py:58-299`).

That means a malformed anomaly, run, PPO update, or episode outcome row can disappear before the proof gates run. Combined with KARN-PROOF-001, a corrupt or partially written proof directory can still reach `REVIEW` if the remaining rows do not trip a gate.

Tracker-ready row:

| Field | Value |
| --- | --- |
| Title | Add fail-closed ingestion integrity checks for proof telemetry |
| Severity | P1 / High |
| Files | `src/esper/karn/mcp/views.py`, `scripts/proof_packet.py`, `tests/karn/mcp/test_views.py`, `tests/scripts/test_proof_packet.py` |
| Acceptance tests | A malformed JSONL line in an `events.jsonl` file is surfaced in an ingestion-integrity report; proof packet verdict is `BLOCKED` when any event file has parse failures; tests prove malformed proof-blocking anomalies cannot be skipped into a non-blocked packet. |

### KARN-PROOF-004 - High - `param_ratio` semantics make ROI and Pareto efficiency claims misleading

The `EpisodeOutcome` contract documents `param_ratio` as `total_params / host_params` (`src/esper/leyline/episode_outcome.py:31-39`). The current producer records `(model.total_params - host_params_baseline) / max(1, host_params_baseline)` (`src/esper/simic/training/action_execution.py:1537-1548`), which is parameter overage, not total/host ratio. The Karn view passes that field through as `param_ratio` (`src/esper/karn/mcp/views.py:555-573`), and the proof packet computes `mean_accuracy_roi` as `AVG(final_accuracy / NULLIF(param_ratio, 0.0))` (`scripts/proof_packet.py:133-148`).

For a no-growth run, contract ratio should be `1.0`; current overage would be `0.0`, producing `NULL` ROI. For a 20% growth run, contract ratio should be `1.2`; current overage is `0.2`, inflating accuracy-per-parameter ROI by 6x. Pareto and hypervolume also minimize `param_ratio` (`src/esper/karn/pareto.py:21-24`, `src/esper/karn/pareto.py:50-94`), so they currently reward "zero added params" as the minimum value under a field name whose contract says "total/host".

Tracker-ready row:

| Field | Value |
| --- | --- |
| Title | Align `param_ratio` contract, producer, and proof ROI math |
| Severity | P1 / High |
| Files | `src/esper/simic/training/action_execution.py`, `src/esper/leyline/episode_outcome.py`, `src/esper/karn/mcp/views.py`, `scripts/proof_packet.py`, `tests/karn/test_pareto.py`, `tests/scripts/test_proof_packet.py` |
| Acceptance tests | No-growth episode emits contract-consistent `param_ratio=1.0` or the field is renamed to `param_overage_ratio`; proof ROI uses the intended denominator; Pareto tests cover no-growth and 20% growth cases; proof packet labels the metric unambiguously. |

### KARN-PROOF-005 - Medium - KarnCollector/TelemetryStore are not proof-complete stateful aggregators

`KarnCollector._update_store()` handles training, epoch, batch, seed, PPO, anomaly, and analytics events, but has no branch for `EPISODE_OUTCOME` (`src/esper/karn/collector.py:258-270`). Store export writes context, baseline, epochs, and dense traces only (`src/esper/karn/store.py:488-507`); there is no outcome collection to export. The analytics snapshot handler only copies `total_reward`, `action_name`, and `value_estimate` into `PolicySnapshot` (`src/esper/karn/collector.py:598-604`), while the MCP `rewards` view exposes the full reward component proof surface from the raw event (`src/esper/karn/mcp/views.py:397-453`).

This creates two different Karn truths: raw DuckDB views are much richer than the stateful store/export path. Any proof or offline analysis built from `TelemetryStore.export_jsonl()` loses outcome, reward component, and detailed action-context evidence.

Tracker-ready row:

| Field | Value |
| --- | --- |
| Title | Make Karn stateful store/export preserve proof-critical events or mark it non-proof-grade |
| Severity | P2 / Medium |
| Files | `src/esper/karn/collector.py`, `src/esper/karn/store.py`, `tests/karn/test_store_export.py`, `tests/karn/test_analytics_snapshot_policy.py` |
| Acceptance tests | Emitting `EPISODE_OUTCOME` through `KarnCollector` makes the outcome available from store/export/import; `ANALYTICS_SNAPSHOT(kind=last_action)` roundtrips reward components and action context needed by `rewards`; if full preservation is intentionally out of scope, APIs/docs clearly reject using store export as a proof packet source. |

### KARN-PROOF-006 - Medium - `import_from_nissa_dir()` silently reconstructs a partial placeholder store

`TelemetryStore.import_from_nissa_dir()` says it reconstructs a store from Nissa `events.jsonl` (`src/esper/karn/store.py:856-866`), but it only handles `TRAINING_STARTED`, `EPOCH_COMPLETED`, and `ANALYTICS_SNAPSHOT(kind=last_action)` (`src/esper/karn/store.py:891-925`). It ignores seed lifecycle, PPO updates, batch epochs, anomaly/confounder events, morphology causal logs, and `EPISODE_OUTCOME`. It also defaults missing epoch metrics to `0.0` (`src/esper/karn/store.py:911-912`) and policy fields to `0.0`/empty strings (`src/esper/karn/store.py:920-925`).

That behavior is acceptable only as a explicitly partial UI convenience importer. It is not proof-grade import/export and currently does not fail closed or communicate which event families were dropped.

Tracker-ready row:

| Field | Value |
| --- | --- |
| Title | Fail closed or fully preserve proof event families in Nissa directory import |
| Severity | P2 / Medium |
| Files | `src/esper/karn/store.py`, `tests/karn/test_store_import.py` |
| Acceptance tests | Importing a Nissa directory with `EPISODE_OUTCOME`, `PPO_UPDATE_COMPLETED`, seed lifecycle, and anomaly events either roundtrips those families or raises an explicit unsupported-proof-import error; missing required metric fields do not become proof-looking zeroes. |

## Tracker-ready issue rows

| ID | Priority | Title | Acceptance tests |
| --- | --- | --- | --- |
| KARN-PROOF-001 | P1 | Make proof packet fail closed on missing runs/outcomes | Empty telemetry, missing runs, and missing `EPISODE_OUTCOME` all produce `BLOCKED` with explicit missing-evidence text. |
| KARN-PROOF-002 | P1 | Treat rollback, reward-hacking, and degradation as proof confounders | Confounder view and proof packet block on `GOVERNOR_ROLLBACK`, `REWARD_HACKING_SUSPECTED`, and `PERFORMANCE_DEGRADATION`. |
| KARN-PROOF-003 | P1 | Add fail-closed ingestion integrity checks | Malformed JSONL rows are surfaced and block proof packets instead of being skipped silently. |
| KARN-PROOF-004 | P1 | Align `param_ratio` semantics and ROI math | Contract, producer, view, Pareto, and packet tests agree on total/host ratio or explicitly renamed overage ratio. |
| KARN-PROOF-005 | P2 | Make store/export proof-complete or explicitly non-proof-grade | Store roundtrip preserves outcome/reward/action proof fields or rejects proof use. |
| KARN-PROOF-006 | P2 | Fail closed or fully preserve proof events in Nissa import | Nissa import covers proof event families or raises a clear unsupported-proof-import error. |

## Test gaps

- No proof packet test for empty telemetry directory, no `TRAINING_STARTED`, no `EPISODE_OUTCOME`, or corrupted JSONL.
- No MCP/proof test asserting rollback, reward-hacking, or performance degradation blocks proof verdicts.
- No store roundtrip test for `EPISODE_OUTCOME` or full `RewardComponentsTelemetry` preservation.
- No test that fixes `param_ratio` semantics against the Leyline contract and proof ROI output.
