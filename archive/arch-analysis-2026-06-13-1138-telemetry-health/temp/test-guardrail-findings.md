# Test and Guardrail Validation Audit Findings

Date: 2026-06-13

Scope: read-only audit of `pytest.ini`, `scripts/lint_defensive_patterns.py`, `scripts/lint_gpu_sync.py`, the focused telemetry pytest command from the task brief, and `tests/fixtures/telemetry/`. Source code was not edited. Existing untracked audit reports in this temp directory were treated as current user-owned coordination artifacts.

## Command Outcomes

The coordinator already ran the validation commands locally. I did not rerun the full pytest command because the supplied outcomes were current and sufficient for this audit; I inspected the named tests, guardrails, and smoke artifacts directly.

| Command | Outcome | Validation assessment |
| --- | --- | --- |
| `uv run python scripts/lint_defensive_patterns.py` | Exit 0. Checked 188 files, 71 checked patterns, 0 violations, 0 stale whitelist entries. | Strict no-defensive-pattern guard is green for default `src/esper` scan. It does not audit `isinstance()`/`.get()` unless run with `--audit`, so it is not a complete absence-semantics guard. |
| `uv run python scripts/lint_gpu_sync.py` | Exit 1. Checked 188 files, 117 sync points, 0 violations, 1 stale whitelist entry: `src/esper/simic/agent/rollout_buffer.py:TamiyoRolloutBuffer.mark_terminal_with_penalty:item`. | The gate is doing the right thing by failing on stale allowances. This is not a telemetry correctness failure, but CI should remain red until the stale whitelist entry is removed or updated. |
| Focused telemetry pytest command from task brief | Exit 0. `1251 passed, 5 deselected in 5.08s`. | Broad field/schema/transport coverage is passing, but the suite still misses several failure modes found by the domain audits. Passing status should be interpreted as regression coverage for many known fields, not proof that telemetry is end-to-end complete. |
| Smoke capture command | Exit 0. Output dir `telemetry/health-audit-20260613-114044`; Nissa events lines=4; Karn export lines=3. | Smoke proves a tiny heuristic run can emit JSONL and export a Karn store. It does not exercise PPO, lifecycle mutations, rollback, proof outcomes, W&B, or full import/export parity. I inspected the artifacts: the `EPOCH_COMPLETED` Nissa row lacks `episode_idx`, and the Karn export contains only `context`, `baseline`, and `epoch` rows. |

## Guardrail Assessment

`pytest.ini` excludes `integration`, `stress`, `property`, and `slow` tests by default. The focused command explicitly includes selected integration tests and all of `tests/telemetry`, which is appropriate for this campaign. However, the default lane alone would not cover the reward/Q telemetry integration tests or property suites.

`lint_defensive_patterns.py` validates strict bug-hiding patterns and stale allowlist entries. It is valuable for enforcing the local policy, but it is not a telemetry absence-semantics validator because the default mode ignores `.get()` and `isinstance()` audit patterns. Several confirmed telemetry bugs are default/absence problems rather than prohibited-call problems.

`lint_gpu_sync.py` validates GPU sync whitelist drift and currently fails only because the whitelist still names a removed or moved sync point. That is useful guardrail quality: stale performance exceptions cannot silently survive. It does not validate whether telemetry sync points are semantically real, only whether they are explicitly allowed.

## Coverage Gaps by Telemetry Failure Mode

### Missing field or data-loss regressions

Current coverage:
- `tests/leyline/test_telemetry.py` covers selected `from_dict()` paths and nested reward/head telemetry serialization.
- `tests/nissa/test_output.py` proves `DirectoryOutput` writes JSONL and the hub routes events.
- `tests/karn/mcp/test_views.py` covers several raw DuckDB views, including morphology causal fields, rewards, PPO head fields, seed lifecycle, batch stats, and one confounder class.

Gaps:
- No all-payload `to_dict()` -> `from_dict()` round-trip test across the full `TelemetryPayload` union. This would catch `EpochCompletedPayload.to_dict()` omitting `episode_idx`.
- No Nissa File/DirectoryOutput assertion that every payload contract field survives JSONL serialization.
- No import/export parity test proving `TelemetryStore.import_from_nissa_dir()` or `TelemetryStore.export_jsonl()` preserve proof-critical event families such as `PPO_UPDATE_COMPLETED`, seed lifecycle, `MORPHOLOGY_CAUSAL_LOG`, `GOVERNOR_ROLLBACK`, and `EPISODE_OUTCOME`.
- No W&B preservation test for critical `TRAINING_STARTED` and `PPO_UPDATE_COMPLETED` diagnostics beyond the small metric subset.
- No live Sanctum/Overwatch assertion that lifecycle causal IDs remain visible after aggregation.

### Fake, defaulted, or placeholder telemetry presented as measurement

Current coverage:
- Some tests intentionally preserve default behavior, for example `tests/nissa/test_tracker.py` asserts disabled norm/std remain `0.0`.
- `tests/telemetry` contains broad schema/consumer checks for UI fields and happy-path transport.

Gaps:
- No tests distinguish "not collected" from measured zero for Nissa gradient stats, batch rolling accuracy, blending/counterfactual attribution, or `HeadTelemetry.from_dict()`.
- No gate test proving permissive G2 fails when gradient telemetry has never been measured.
- No post-germination observation test proving new seed diagnostics do not expose healthy/fresh defaults before gradient or counterfactual evidence exists.
- No observation-history test distinguishing absent history from observed zero-valued history.
- No smoke/acceptance test fails when an emitted row contains placeholder-only proof evidence.

### Contract drift and miswired consumers

Current coverage:
- W&B tests cover step derivation, missing epoch handling, monotonicity, and `None` filtering.
- Decision telemetry tests cover propagation of supplied mask booleans and head telemetry values.
- Q telemetry integration tests prove Q values are present, finite, shaped correctly, and reach Sanctum.

Gaps:
- No W&B lifecycle tests requiring stage/fossilized/pruned handlers to use payload `slot_id` when envelope `event.slot_id` is absent.
- No analytics test for the documented `SeedPrunedPayload.blueprint_id=None` case.
- No test catches decision mask flags being computed as "some options restricted" while the contract says "forced".
- No recurrent-state fidelity test for op Q telemetry; current test does not fail if Q values are recomputed with `hidden=None`.
- No param-ratio contract test tying `EpisodeOutcome.param_ratio`, producer math, MCP view output, Pareto, and proof packet ROI to one meaning.

### Unwired, dead, or partial feeds

Current coverage:
- `tests/leyline/test_telemetry_events.py` verifies enum membership.
- `tests/karn/mcp/test_views.py` proves views can parse some event families when rows are hand-authored.

Gaps:
- No registry/completeness test requiring every public `TelemetryEventType` to have an intentional payload, producer, backend/store route, and consumer or an explicit tombstone.
- No test proves `GRADIENT_PATHOLOGY_DETECTED` can be emitted by the anomaly producer map.
- No live UI consumer test for structured `MORPHOLOGY_CAUSAL_LOG` fields.
- No test verifies `--export-karn` registers exactly one authoritative Karn collector and exports that same collector.

### Duplicate, stale, or split-source telemetry

Current coverage:
- GPU sync guardrail catches stale whitelist entries.
- Nissa hub reset/race tests cover backend worker lifecycle and duplicate delivery after reset.

Gaps:
- No rollback integration test asserting exactly one `GOVERNOR_ROLLBACK` per causal rollback with complete context.
- No CLI topology test catches duplicate Karn collector registration under `--export-karn`.
- No stale telemetry allowlist/fixture check outside the guardrail-specific YAML files.

### Proof-invalidating and fail-open behavior

Current coverage:
- `tests/karn/mcp/test_views.py` covers `run_confounders` for `NUMERICAL_INSTABILITY_DETECTED`.
- Existing proof tests, per Karn audit, cover blocking confounders, missing learnability, and optional baseline cohorts.

Gaps:
- No proof packet tests for empty telemetry, missing `TRAINING_STARTED`, or missing `EPISODE_OUTCOME`.
- No malformed JSONL integrity test; `raw_events` uses `ignore_errors=true`, so corrupt proof rows can be skipped before gates run.
- No proof confounder tests for `GOVERNOR_ROLLBACK`, `REWARD_HACKING_SUSPECTED`, or `PERFORMANCE_DEGRADATION`.
- No acceptance test that a smoke capture without proof outcomes is classified as smoke-only rather than evidence for reward efficiency.

### Fixture quality

`tests/fixtures/telemetry/healthy_gradients.json` is a single happy-path fixture with healthy gradient values. There are no negative fixtures for missing gradient evidence, malformed rows, partial payloads, absent optional metrics, duplicate rollback events, unknown blueprint prune events, or corrupted proof directories. This limits the suite's ability to guard the exact absence/default/fail-open failures found in this campaign.

## Tracker-Ready Test Recommendations

| ID | Priority | Title | Acceptance tests |
| --- | --- | --- | --- |
| TGV-001 | P1 | Add all-payload telemetry contract round-trip coverage | For every active `TelemetryPayload` dataclass, construct a non-default payload with every contract field populated, assert `to_dict()` includes all expected keys, assert `from_dict(to_dict(payload))` preserves values, and include a specific `EpochCompletedPayload.episode_idx` regression. |
| TGV-002 | P1 | Add Nissa JSONL field-preservation tests | Emit representative payloads through FileOutput/DirectoryOutput and assert JSONL contains required and optional non-null contract fields, including `episode_idx`, lifecycle causal IDs, PPO diagnostics, and episode outcome fields. |
| TGV-003 | P1 | Add absence-semantics tests for placeholder-prone telemetry | Disabled Nissa gradient metrics, missing rolling accuracy, missing blending/counterfactual attribution, empty/partial `HeadTelemetry`, and absent observation history must serialize as absent/`None`/explicit collected flags, not measured zeroes. |
| TGV-004 | P1 | Add gradient-evidence gate tests | A freshly constructed or fallback-synced seed with no measured gradient stats must not satisfy permissive G2; a measured healthy gradient case must still pass. |
| TGV-005 | P1 | Add post-germination missing-evidence observation tests | After germination but before gradient telemetry or solo counterfactual measurement, Obs V3 must not encode `gradient_health_prev=1.0` or counterfactual freshness as measured evidence. |
| TGV-006 | P1 | Add rollback single-source integration coverage | One panic rollback should emit exactly one `GOVERNOR_ROLLBACK` per env with `panic_reason`, `loss_at_panic`, `loss_threshold`, `consecutive_panics`, `env_id`, `device`, and episode context present. |
| TGV-007 | P1 | Add recurrent-context Q telemetry regression | Construct identical observations with different rollout hidden states and assert op Q telemetry uses the sampled row's stored hidden state rather than `hidden=None`. |
| TGV-008 | P1 | Add proof fail-closed tests | Empty telemetry, missing `TRAINING_STARTED`, missing `EPISODE_OUTCOME`, and malformed JSONL rows must produce `Verdict: BLOCKED` with explicit missing/invalid evidence text. |
| TGV-009 | P1 | Add proof confounder completeness tests | `GOVERNOR_ROLLBACK`, `REWARD_HACKING_SUSPECTED`, and `PERFORMANCE_DEGRADATION` must appear in confounder views with enough payload detail and must block proof packets. |
| TGV-010 | P1 | Add `param_ratio` semantic contract tests | No-growth and 20% growth episode outcomes must assert the intended denominator across producer, Leyline contract, MCP views, Pareto, and proof ROI labels. |
| TGV-011 | P2 | Add W&B contract-alignment tests | W&B lifecycle handlers must accept payload-only `slot_id` for stage/fossilized/pruned events; PPO and training-start tests must require critical diagnostics when present. |
| TGV-012 | P2 | Add event registry completeness tests | Every non-tombstoned `TelemetryEventType` must have a typed payload contract, a producer or explicit unsupported marker, at least one preserving backend/store path, and a consumer/view test. `ISOLATION_VIOLATION` and `GRADIENT_PATHOLOGY_DETECTED` should fail this test until resolved. |
| TGV-013 | P2 | Add Karn import/export proof-parity tests | Importing/exporting Nissa telemetry with PPO updates, lifecycle, anomalies, morphology causal logs, and episode outcomes must either preserve those families or raise an explicit non-proof-grade import/export error. |
| TGV-014 | P2 | Add live UI causal/lifecycle preservation tests | Aggregator tests should assert lifecycle timeline terminal `from_stage`/deltas are preserved and morphology proposal/verdict/mutation/RNG IDs remain visible in live snapshot or event detail state. |
| TGV-015 | P2 | Add `--export-karn` CLI topology test | A run with `--export-karn` must register one authoritative Karn collector and export the same collector's store. |
| TGV-016 | P2 | Expand telemetry fixtures beyond healthy happy path | Add fixtures for missing gradients, unknown-blueprint prune, partial head telemetry, malformed JSONL, duplicate rollback, absent episode outcome, and measured-zero vs not-collected metrics; use them in contract/backend/proof tests. |

## Bottom Line

The current focused suite is strong at field existence, schema shape, selected transport paths, and many UI consumer calculations. It is weak at negative evidence: missingness, partial payloads, malformed rows, proof fail-closed behavior, single-source event guarantees, and cross-surface semantic parity. The highest-value next tests are not more happy-path field checks; they are acceptance tests that make absence and corruption visible rather than allowing defaults, dropped rows, or partial imports to look like valid telemetry.
