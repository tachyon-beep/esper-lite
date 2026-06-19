# Telemetry Health Report

Date: 2026-06-13

Scope: read-only audit of Esper telemetry from producers through Leyline contracts, Nissa outputs, Karn stores/MCP/proof, Sanctum/Overwatch consumers, and focused tests. No Filigree issues were created. Source edits were out of scope.

## Executive Summary

The telemetry system has strong raw-event foundations: Leyline typed payloads, Nissa JSONL, and Karn MCP raw DuckDB views preserve many real signals. The focused telemetry suite also passes. That is a useful signal of life, but it is not enough to support reward-efficiency or "solid signals of life" claims yet.

The top correctness blockers are not simple missing tests. They are semantic failures where absence looks like measurement, proof tooling fails open, store/export paths lose fields, and producers emit or omit proof-critical rows inconsistently.

Most reliable source of truth today: Nissa raw JSONL plus Karn MCP raw views, with caveats about malformed-row handling.

Least reliable source of truth today: Karn stateful store/export and proof packet verdicts, because they are partial and fail open on missing/corrupt proof evidence.

## Top Correctness Blockers

| ID | Severity | Failure mode | Why it blocks solid signals of life |
| --- | --- | --- | --- |
| KARN-PROOF-001 | P1 | Proof-invalidating | Proof packet can remain `REVIEW` with no runs or no `EPISODE_OUTCOME`; that is no proof basis. |
| KARN-PROOF-003 | P1 | Proof-invalidating | Malformed JSONL rows can be skipped before proof gates see them. |
| SIMIC-PROD-001 | P1 | Proof-invalidating | Rollback episodes disappear from `EPISODE_OUTCOME`, so bad episodes can be absent from ROI proof. |
| KARN-PROOF-004 | P1 | Miswired/proof-invalidating | `param_ratio` semantics diverge across contract, producer, Pareto, and proof ROI. Efficiency claims are mathematically misleading. |
| SIMIC-PROD-005 | P1 | Fake/defaulted | `static_final` and `fixed_schedule` baselines are marked supported but are WAIT-only placeholders. Baseline comparisons are not valid. |
| KARN-PROOF-002 | P1 | Proof-invalidating | Rollback, reward-hacking, and degradation events do not block proof packets. |
| KTS-001 | P1 | Fake/defaulted | Permissive G2 can treat absent gradient evidence as healthy gradient evidence. |
| KTS-002 | P1 | Fake/defaulted | Morphology watch/commit/audit rows claim post-action evidence but reuse same-step current loss. |
| KTS-003 | P1 | Duplicate/stale | Rollback is emitted twice with conflicting context; consumers can overwrite the better fact. |
| TPD-001 | P1 | Miswired | Decision mask telemetry overstates forced choices, corrupting policy diagnostics. |
| TPD-002 | P1 | Fake/defaulted | Q-value telemetry uses initial recurrent state, not the rollout hidden state that drove decisions. |
| TPD-003 | P1 | Fake/defaulted | New seed observations start as healthy/fresh before proof exists. |
| LN-001 | P1 | Missing field | `EPOCH_COMPLETED` loses `episode_idx` during serialization. |
| LN-003 | P1 | Fake/defaulted | Missing attribution metrics are reported as zero-valued facts. |
| UI-001 | P1 | Unwired | `MORPHOLOGY_CAUSAL_LOG` is raw-view-only, so live UI cannot prove morphology causality. |
| UI-002 | P1 | Mutated/lost | Seed lifecycle UI drops proposal/verdict/mutation/RNG identity from payloads. |
| SMOKE-001 | P1 | Mutated/lost | Tiny smoke proved `TRAINING_STARTED.host_params=164` becomes `host_params=0` in Karn export. |

## Feed Inventory

See `01-telemetry-feed-inventory.md` for the full producer -> payload -> backend -> consumer table. The short version:

- Lifecycle and training feeds are broadly emitted and preserved in raw JSONL.
- PPO and decision diagnostics are rich at the Leyline/Nissa raw layer, but W&B and Karn store/export preserve only subsets.
- Proof-critical events (`EPISODE_OUTCOME`, confounders, malformed input integrity) are not guarded tightly enough.
- Live UI snapshots are downstream of Sanctum aggregation; fields missing from Sanctum schema cannot be recovered by Overwatch.

## Subsystem Findings

### Leyline/Nissa

Leyline payloads are useful contracts, but some serialization/deserialization paths are inconsistent. The clearest bug is `EpochCompletedPayload.to_dict()` omitting `episode_idx` while `from_dict()` requires it. Several Nissa consumers also collapse missingness into zero: attribution metrics, batch rolling accuracy, and disabled tracker metrics.

W&B is currently a metric subset sink, not a faithful telemetry store. That is acceptable only if the system labels it as such; it should not be used as a source for later health proof.

### Simic Producers

Simic emits real PPO, action, reward, gradient, LSTM, and value feeds on successful paths. The weak spots are skipped/exceptional paths and proof baselines. Rollback episodes skip corrected outcome telemetry. Finiteness-gate skipped updates can continue without an auditable skipped-update event. Ratio diagnostics and rollback attribution are implemented but not wired through the production telemetry path.

### Kasmina/Tolaria Signals

Seed lifecycle and governor feeds are real, but some safety evidence is placeholder or duplicated. Permissive G2 can pass with default healthy gradient telemetry. Morphology watch/audit phases are emitted immediately using current validation loss, so they are not real post-mutation watch-window evidence. Rollback telemetry is split between Tolaria and Simic, producing duplicate events with different reason/detail semantics.

### Tamiyo Diagnostics

Tamiyo diagnostics are rich, but several fields do not mean what downstream readers would infer. Mask flags mean restriction, not forced choice. Q diagnostics use a placeholder recurrent state. New seed diagnostics initialize previous gradient health and counterfactual freshness as if proof existed. History padding and partial `HeadTelemetry` deserialization can turn missingness into measured zero.

### Karn Analytics And Proof

Karn MCP raw views are the strongest analytics surface. KarnCollector/TelemetryStore export is partial and should not be treated as proof-grade. Proof packet gating is the most urgent risk: missing evidence, corrupt rows, and omitted confounder classes can produce a non-blocked packet.

The `param_ratio` mismatch is especially damaging because it corrupts both ROI and Pareto interpretation. The Leyline contract says total/host; the producer writes overage ratio; proof divides accuracy by the field; Pareto minimizes it.

### UI Consumers

The initial UI subagent timed out, so the coordinator performed a bounded local pass and then spawned a final UI closeout explorer. The standalone UI report confirmed that Sanctum/Overwatch cannot display morphology causal logs as structured live state, seed lifecycle rows drop causal identity fields already present in payloads, generated TypeScript faithfully propagates Python schema omissions, and several web/TUI fallback paths render missing data as dormant, zero, or OK.

## Smoke Evidence

Smoke command:

```bash
AUDIT_DIR="telemetry/health-audit-$(date +%Y%m%d-%H%M%S)"
PYTHONPATH=src uv run python -m esper.scripts.train heuristic \
  --task cifar_minimal \
  --episodes 1 \
  --max-epochs 1 \
  --max-batches 1 \
  --device cpu \
  --slots r0c0 \
  --telemetry-level debug \
  --gradient-telemetry-stride 1 \
  --telemetry-dir "$AUDIT_DIR/nissa" \
  --export-karn "$AUDIT_DIR/karn-export.jsonl" \
  --no-tui
```

Continuation audit result: exit 0. Output directory: `telemetry/health-audit-20260613-120246`.

Observed records:

- Nissa `events.jsonl`: 4 rows.
- Karn export: 3 rows.
- Nissa `TRAINING_STARTED` row carried `host_params=164`.
- Karn export `context.host_params` and epoch `host.host_params` carried `0`.

Source trace:

- `src/esper/simic/training/helpers.py:511-529` computes and emits `host_params`.
- `src/esper/karn/collector.py:281-287` starts the episode without passing `host_params`.
- `src/esper/karn/store.py:92` and `src/esper/karn/store.py:172` default host params to `0`.
- `src/esper/karn/store.py:896-903` also omits `host_params` in Nissa-directory import.

## Verification Evidence

| Command | Result |
| --- | --- |
| `uv run python scripts/lint_defensive_patterns.py` | Pass: 188 files, 0 violations, 0 stale whitelist entries. |
| `uv run python scripts/lint_gpu_sync.py` | Fail: 0 violations, 1 stale whitelist entry. |
| Focused telemetry pytest command | Pass: continuation audit `1251 passed, 5 deselected in 5.15s`. |
| Tiny telemetry smoke capture | Pass: continuation audit emitted raw Nissa telemetry and Karn export; reproduced host-param field loss. |

## Health Rating

Current telemetry health for "solid signals of life": **red/yellow**.

- Red for proof validity: fail-open proof packet, missing rollback outcomes, malformed-row skip, param-ratio mismatch, and placeholder proof baselines.
- Red for live causality: raw causal identity exists, but live UI drops morphology causal-log structure and lifecycle causal IDs.
- Yellow for general observability: raw JSONL is rich, tests pass, and many producers are real, but UI/store/W&B paths lose or reinterpret fields.
- Green only for basic event transport: Nissa hub/file output and many schema/view tests are active and passing.

## Recommended Burn-Down Order

1. Fail closed in proof tooling: missing runs, missing outcomes, malformed JSONL, and omitted confounders.
2. Fix proof-critical producer gaps: rollback `EPISODE_OUTCOME`, `param_ratio`, placeholder baselines.
3. Stop fake healthy/fresh/default telemetry: G2 gradient evidence, attribution zeros, Tamiyo freshness/history/head telemetry.
4. Fix semantic miswirings: mask forced vs restricted, recurrent Q hidden state, LSTM health overwrite.
5. Make Karn store/export/import either proof-complete or explicitly non-proof-grade.
6. Restore live UI causal visibility for lifecycle and morphology causal logs.
7. Add the acceptance tests in `04-tracker-ready-issue-map.md`.

## Issue Map

The tracker-ready issue map is in `04-tracker-ready-issue-map.md`. It includes title, severity, files, evidence, acceptance tests, and whether each row blocks solid signals of life.
