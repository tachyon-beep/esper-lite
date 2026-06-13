# Subagent Findings

Date: 2026-06-13

This file consolidates the completed subagent reports. Full evidence and line references remain in `temp/*-findings.md`.

## Telemetry Topology

Source: `temp/topology-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| TT-001 | P3 | Unwired | `ISOLATION_VIOLATION` is defined but has no payload, producer, or consumer. |
| TT-002 | P2 | Unwired | `GRADIENT_PATHOLOGY_DETECTED` has consumers but no current producer path. |
| TT-003 | P2 | Mutated/lost | `TelemetryStore.import_from_nissa_dir()` reconstructs only a few event families from full Nissa JSONL. |
| TT-004 | P2 | Mutated/lost | `MORPHOLOGY_CAUSAL_LOG` is raw-queryable but not visible as structured live UI state. |
| TT-005 | P3 | Duplicate/stale | `--export-karn` registers a local collector and then a global collector, exporting only the latter. |

## Leyline/Nissa Contracts

Source: `temp/leyline-nissa-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| LN-001 | P1 | Missing field | `EpochCompletedPayload.to_dict()` drops `episode_idx` even though `from_dict()` requires it. |
| LN-002 | P1 | Miswired | `SeedPrunedPayload.blueprint_id=None` is contract-valid but crashes `BlueprintAnalytics`. |
| LN-003 | P1 | Fake/defaulted | Missing attribution values become analytics zeroes. |
| LN-004 | P2 | Fake/defaulted | Missing batch rolling accuracy displays as measured `0.0%`. |
| LN-005 | P2 | Fake/defaulted | Disabled Nissa gradient tracker metrics serialize as zero. |
| LN-006 | P2 | Mutated/lost | W&B drops most training-start and PPO contract fields. |
| LN-007 | P2 | Miswired | W&B lifecycle handlers read optional envelope `slot_id` instead of required payload `slot_id` for stage/fossilized/pruned events. |

## Simic Producers

Source: `temp/simic-producer-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| SIMIC-PROD-001 | P1 | Proof-invalidating | Rollback episodes skip `EPISODE_OUTCOME`; no corrected outcome event is emitted later. |
| SIMIC-PROD-002 | P2 | Mutated/lost | PPO `ratio_diagnostic` is produced and emitter-ready but dropped by the coordinator anomaly path. |
| SIMIC-PROD-003 | P2 | Miswired | PPO update LSTM health is overwritten with rollout hidden-state health. |
| SIMIC-PROD-004 | P2 | Unwired | First/second finiteness-gate skipped PPO batches emit no PPO, batch, or anomaly telemetry. |
| SIMIC-PROD-005 | P1 | Fake/defaulted | `static_final` and `fixed_schedule` proof baselines are WAIT-only placeholders but marked supported. |
| SIMIC-PROD-006 | P3 | Unwired | Rollback attribution metadata is stored in the rollout buffer but not emitted. |

## Kasmina/Tolaria Signals

Source: `temp/kasmina-tolaria-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| KTS-001 | P1 | Fake/defaulted | Permissive G2 can pass with no measured gradient evidence because defaults look healthy. |
| KTS-002 | P1 | Fake/defaulted | Morphology watch/commit/audit rows reuse current env loss rather than post-mutation evidence. |
| KTS-003 | P1 | Duplicate/stale | Rollback telemetry is emitted twice with conflicting context. |
| KTS-004 | P2 | Mutated/lost | Karn lifecycle timeline loses terminal transition origins and deltas. |
| KTS-005 | P2 | Mutated/lost | Karn drops lifecycle causal IDs even when payloads carry them. |
| KTS-006 | P3 | Missing field | Fossilize/prune schemas expose `blending_delta`, but emitters never populate it. |

## Tamiyo Policy Diagnostics

Source: `temp/tamiyo-diagnostics-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| TPD-001 | P1 | Miswired | Decision mask flags report ordinary restrictions as forced choices. |
| TPD-002 | P1 | Fake/defaulted | Op Q-value telemetry recomputes with initial recurrent hidden state, not sampled rollout hidden state. |
| TPD-003 | P1 | Fake/defaulted | New seed tracking initializes absent gradient/counterfactual evidence as healthy/fresh. |
| TPD-004 | P2 | Fake/defaulted | `HeadTelemetry.from_dict()` silently fills missing diagnostic fields with zeros. |
| TPD-005 | P2 | Fake/defaulted | Observation history padding hides early/missing history as real zero values. |

## Karn Analytics And Proof

Source: `temp/karn-analytics-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| KARN-PROOF-001 | P1 | Proof-invalidating | Proof packet fails open when there are no runs or no episode outcomes. |
| KARN-PROOF-002 | P1 | Proof-invalidating | Proof confounder ledger omits rollback, reward-hacking, and degradation classes. |
| KARN-PROOF-003 | P1 | Proof-invalidating | JSONL parse errors are silently skipped before proof gating. |
| KARN-PROOF-004 | P1 | Proof-invalidating | `param_ratio` means overage in producer but total/host in contract, invalidating ROI/Pareto semantics. |
| KARN-PROOF-005 | P2 | Mutated/lost | KarnCollector/TelemetryStore are not proof-complete stateful aggregators. |
| KARN-PROOF-006 | P2 | Mutated/lost | `import_from_nissa_dir()` silently reconstructs a partial placeholder store. |

## UI Consumers

Source: `temp/ui-consumer-findings.md`

Confirmed findings:

| ID | Severity | Failure mode | Finding |
| --- | --- | --- | --- |
| UI-001 | P1 | Unwired | `MORPHOLOGY_CAUSAL_LOG` is raw-view-only and has no structured live Sanctum/Overwatch state route. |
| UI-002 | P1 | Mutated/lost | Seed lifecycle snapshot drops causal identity already present in seed payloads. |
| UI-003 | P2 | Guardrail gap | Generated Overwatch TypeScript is mechanically in sync with Python but faithfully propagates schema omissions; no freshness guard was found in this lane. |
| UI-004 | P2 | Fake/defaulted | Web and TUI health displays can present no-data defaults as clean or measured values. |
| UI-005 | P2 | Fake/defaulted | Seed swimlane fabricates dormant rows for missing slot state. |

## Test And Guardrails

Source: `temp/test-guardrail-findings.md`

Main validation points:

- Defensive-pattern lint passed in strict default mode.
- GPU-sync lint failed because of one stale whitelist key, not because of new sync violations.
- Focused telemetry tests passed: `1251 passed, 5 deselected in 5.08s`.
- Smoke capture passed and exposed a real Karn export field-loss issue.

Highest-value test gaps:

- all-payload `to_dict()`/`from_dict()` round-trip coverage;
- Nissa JSONL field-preservation coverage;
- absence-semantics tests for placeholder-prone telemetry;
- proof fail-closed tests for empty/missing/corrupt evidence;
- recurrent-context Q telemetry regression;
- rollback single-source and corrected-outcome coverage;
- `param_ratio` semantic contract tests;
- live UI causal/lifecycle preservation tests.
