# Validation Notes For Telemetry Health Report

Date: 2026-06-13

Validator scope: verify that the synthesized report is grounded in returned subagent reports, current source snippets, and command outputs from this campaign. This is not a source-code test run beyond the commands listed below.

## Artifact Validation

Expected files:

- `00-coordination.md`
- `01-telemetry-feed-inventory.md`
- `02-subagent-findings.md`
- `03-telemetry-health-report.md`
- `04-tracker-ready-issue-map.md`
- `temp/validation-telemetry-health-report.md`

Subagent source reports present:

- `temp/topology-findings.md`
- `temp/leyline-nissa-findings.md`
- `temp/simic-producer-findings.md`
- `temp/kasmina-tolaria-findings.md`
- `temp/tamiyo-diagnostics-findings.md`
- `temp/karn-analytics-findings.md`
- `temp/test-guardrail-findings.md`
- `temp/ui-consumer-findings.md`

UI subagent status: the initial UI subagent timed out and was shut down. A final UI closeout explorer later produced `temp/ui-consumer-findings.md`, and the final report set integrates those UI findings.

## Command Validation

| Command | Result | Report usage |
| --- | --- | --- |
| `uv run python scripts/lint_defensive_patterns.py` | Pass. 188 files, 0 violations, 0 stale whitelist entries. | Used as guardrail evidence only; not treated as proof that absence semantics are correct. |
| `uv run python scripts/lint_gpu_sync.py` | Fail. 0 violations, 1 stale whitelist entry: `src/esper/simic/agent/rollout_buffer.py:TamiyoRolloutBuffer.mark_terminal_with_penalty:item`. | Reported as a guardrail failure unrelated to telemetry proof correctness. |
| Focused telemetry pytest command from campaign plan | Pass. Continuation audit: `1251 passed, 5 deselected in 5.15s`. | Report states tests pass but do not cover the discovered absence/proof-fail-open gaps. |
| Tiny heuristic telemetry smoke capture | Pass. Continuation audit output under `telemetry/health-audit-20260613-120246`. | Used to validate live event production and reproduce Karn host-param loss. |

## Smoke Artifact Validation

Observed files:

- `telemetry/health-audit-20260613-114044/nissa/telemetry_2026-06-13_114046/events.jsonl`
- `telemetry/health-audit-20260613-114044/karn-export.jsonl`
- `telemetry/health-audit-20260613-120246/nissa/telemetry_2026-06-13_120247/events.jsonl`
- `telemetry/health-audit-20260613-120246/karn-export.jsonl`

Observed counts:

- Continuation Nissa JSONL: 4 rows.
- Continuation Karn export JSONL: 3 rows.

Observed field loss:

- Continuation Nissa `TRAINING_STARTED` payload includes `"host_params": 164`.
- Continuation Karn export `context.data.host_params` is `0`.
- Continuation Karn export epoch `host.host_params` is `0`.

Current-source trace supporting the field-loss finding:

- `src/esper/simic/training/helpers.py:511-529` computes and emits host params.
- `src/esper/karn/collector.py:281-287` starts episode without passing `host_params`.
- `src/esper/karn/store.py:92` and `src/esper/karn/store.py:172` default host params to `0`.
- `src/esper/karn/store.py:896-903` omits host params during Nissa import reconstruction.

## Cross-Check Against Report Standards

| Required standard | Validation result |
| --- | --- |
| Lead with top correctness blockers | `03-telemetry-health-report.md` starts with executive summary and top blockers table. |
| Include feed inventory table | `01-telemetry-feed-inventory.md` contains the consolidated table and diagram. |
| Include subsystem findings | `02-subagent-findings.md` and `03-telemetry-health-report.md` summarize each subsystem, including final UI closeout findings. |
| Include verification evidence | `00-coordination.md`, `03-telemetry-health-report.md`, and this file list commands/results. |
| Include issue map rows with acceptance tests | `04-tracker-ready-issue-map.md` includes severity, files, evidence, acceptance tests, and blocker status. |
| Classify failure modes | Issue map and subagent summary classify unwired, fake/defaulted, missing field, miswired, mutated/lost, duplicate/stale, proof-invalidating, and test gaps. |
| No automatic Filigree issue creation | No tracker tools were used to create issues. |
| No source changes | Only docs/report artifacts were written by this campaign. |

## Residual Risks

- The first UI consumer slice timed out. A final UI closeout explorer produced `temp/ui-consumer-findings.md`, but full browser-level UI behavior was not exercised.
- W&B behavior was source/test-audited, not exercised against a live W&B service.
- Smoke capture was intentionally tiny and heuristic-only; it did not exercise PPO, lifecycle mutation, rollback, proof baselines, W&B, or full UI.
- The focused pytest command passed, but many findings are confirmed specifically because current tests do not cover negative evidence and fail-closed behavior.

## Validation Conclusion

The final report is internally consistent with the completed subagent reports, current-source evidence sampled by the coordinator, and command output captured during the campaign. The strongest conclusion is that raw telemetry transport is alive, but proof-grade correctness is not yet established.
