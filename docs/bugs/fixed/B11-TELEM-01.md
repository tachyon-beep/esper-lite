# Finding Ticket: Q-Values Not Wired to Telemetry Pipeline (Policy V2)

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B11-TELEM-01` |
| **Severity** | `P2` |
| **Status** | `closed` |
| **Batch** | 11 |
| **Agent** | `human` |
| **Domain** | `karn/sanctum` |
| **Assignee** | |
| **Created** | 2025-12-31 |
| **Updated** | 2025-12-31 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `src/esper/leyline/telemetry.py`, `src/esper/simic/agent/ppo.py`, `src/esper/simic/telemetry/emitters.py`, `src/esper/karn/sanctum/aggregator.py` |
| **Line(s)** | Various (see Resolution) |
| **Function/Class** | `PPOUpdatePayload`, `PPOAgent.update()`, `emit_ppo_update_event()`, `SanctumAggregator._handle_ppo_update()` |

---

## Summary

**One-line summary:** Op-conditioned Q-values (Policy V2) were displayed in UI but not wired to telemetry pipeline.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [x] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

The HealthStatusPanel in Sanctum UI displays Q-values and Q-variance metrics (lines 253-281 in `health_status_panel.py`), but all values showed 0.0 defaults. The UI correctly flagged this as "NO OP COND!" critical status, indicating the critic wasn't using operation conditioning.

**Root cause:** Q-value telemetry was deferred when Policy V2 was developed. The schema was defined in `TamiyoState` and rendering was implemented in the UI, but the data pipeline was never completed:
- `PPOUpdatePayload` lacked Q-value fields
- PPO update loop didn't collect Q(s,op) values
- Telemetry emitter didn't pass Q-values
- Aggregator didn't wire Q-values to TamiyoState

This created a "partial implementation" where the feature appeared in the UI but had no backing data.

### Impact

- **Medium severity**: Feature appeared functional but displayed incorrect defaults
- **Root cause**: Deferred telemetry implementation during Policy V2 development
- **Scope**: All Q-value diagnostics non-functional (Q-variance, Q-spread, per-operation values)

---

## System Context

This violates the project's explicit telemetry mandate (CLAUDE.md):

> "If you are working on a half finished telemetry component, do not remove it as 'incomplete' or 'deferred functionality'. This pattern of behaviour is why we are several months in and have no telemetry."

The Q-values schema was added to `TamiyoState` and rendered in the UI, creating the appearance of working telemetry, but the data pipeline was never implemented.

**Relevant architectural principles:**
- **No Legacy Code Policy** - Schema existed but was unwired (partial implementation)
- **Telemetry Mandate** - Must complete telemetry components, not defer them

---

## Verification

### How to Verify the Fix

- [x] Integration test validates end-to-end Q-value flow
- [x] UI displays non-zero Q-values during training
- [x] Q-variance diagnostic correctly detects low variance
- [x] All 44 unit tests pass
- [x] Linter clean
- [x] Type checker clean

---

## Resolution

### Final Fix Description

Added Q-value collection throughout the telemetry pipeline to wire Policy V2's op-conditioned critic values to the Sanctum UI:

1. Extended `PPOUpdatePayload` with 8 Q-value fields (q_germinate, q_advance, q_fossilize, q_prune, q_wait, q_set_alpha, q_variance, q_spread)
2. Modified PPO update loop to compute Q(s,op) for all 6 operations using Policy V2's `_compute_value()` method
3. Updated telemetry emitter to pass Q-values from metrics to payload
4. Wired aggregator to map Q-values from payload to TamiyoState for UI display

### Files Changed

**Telemetry Schema:**
- `src/esper/leyline/telemetry.py:686-698` — Added 8 Q-value fields to PPOUpdatePayload
- `src/esper/leyline/telemetry.py:776-784` — Updated from_dict to parse Q-value fields
- `tests/leyline/test_telemetry.py:346-405` — Tests for Q-value serialization

**Q-Value Collection:**
- `src/esper/simic/agent/ppo.py:76-88` — Helper function for NaN initialization
- `src/esper/simic/agent/ppo.py:469-530` — Q-value collection in PPO update loop
- `src/esper/simic/agent/types.py:63-70` — Added Q-value fields to PPOUpdateMetrics TypedDict
- `tests/simic/test_ppo.py:765-845` — Test for Q-value collection

**Telemetry Emission:**
- `src/esper/simic/telemetry/emitters.py:789-797` — Pass Q-values to PPOUpdatePayload

**Aggregation:**
- `src/esper/karn/sanctum/aggregator.py:792-800` — Wire Q-values to TamiyoState
- `tests/karn/sanctum/test_aggregator.py:598-640` — Test for Q-value wiring

**Integration Testing:**
- `tests/integration/test_q_values_telemetry.py` — End-to-end Q-value flow test
- `tests/integration/conftest.py` — Shared test fixture extraction

### Impact

- Q-values now visible in Sanctum UI during training
- Q-variance diagnostic functional (detects if critic ignores op conditioning)
- Better visibility into Policy V2 value head behavior
- Enables debugging of multi-head op-conditioned critic

### Prevention Strategy

**Pattern:** When adding new UI fields to Karn/Sanctum:

1. Define the schema field in `leyline/telemetry.py` (e.g., `PPOUpdatePayload`)
2. Add a `# TODO: [FUTURE FUNCTIONALITY]` comment at the **telemetry emission site** (e.g., `simic/telemetry/emitters.py`) documenting the missing wiring
3. Add a corresponding TODO in the **aggregator** (`karn/sanctum/aggregator.py`) if field needs mapping
4. Track the telemetry implementation as a separate task before deploying UI

**Example TODO comment:**
```python
# TODO: [FUTURE FUNCTIONALITY] - Wire q_values from PPOUpdatePayload to TamiyoState
# UI field defined in HealthStatusPanel, schema in PPOUpdatePayload, but not yet emitted
```

This ensures UI feature work doesn't appear "done" when telemetry is actually missing.

---

## Appendix

### Q-Value Computation Details

Policy V2 uses an op-conditioned value head that computes Q(s,op) - the value of taking operation `op` in state `s`. The Q-values are collected by:

1. Selecting a representative state from the batch (first valid state)
2. Forward pass through LSTM to get hidden state
3. For each of 6 operations (WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE, ADVANCE):
   - Call `policy.network._compute_value(lstm_out, op_tensor)`
   - Store Q(s,op) value
4. Compute variance and spread across all 6 Q-values

Low variance indicates the critic is ignoring the operation input and collapsing to V(s) (state-value only), which is a training issue.

### Test Coverage

**Unit tests:**
- `test_ppo_update_payload_serialization()` — Tests PPOUpdatePayload with Q-values
- `test_ppo_collects_q_values()` — Tests Q-value collection in PPO update loop
- `test_aggregator_wires_q_values()` — Tests aggregator maps Q-values to TamiyoState

**Integration test:**
- `test_q_values_end_to_end()` — Full pipeline test: PPO → emitter → aggregator → UI state
