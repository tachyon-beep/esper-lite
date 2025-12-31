# B11-TELEM-01: Q-Values Not Wired to Telemetry

**Ticket ID:** B11-TELEM-01
**Status:** Fixed
**Severity:** Medium
**Domain:** Telemetry / UI
**Date Discovered:** 2025-12-31
**Date Fixed:** 2025-12-31

## One-Line Summary

Op-conditioned Q-values (Policy V2) were displayed in UI but not wired to telemetry pipeline.

## Symptom

HealthStatusPanel showed Q-values and Q-variance, but all values were 0.0 defaults. UI correctly flagged this as "NO OP COND!" critical status.

## Root Cause

Q-value telemetry was not implemented when Policy V2 was developed. The UI was built first (schema + rendering), but the data pipeline was deferred.

## Fix

Added Q-value collection, emission, and aggregation:

1. **PPOUpdatePayload** (leyline/telemetry.py): Added `q_germinate`, `q_advance`, `q_fossilize`, `q_prune`, `q_wait`, `q_set_alpha`, `q_variance`, `q_spread` fields
2. **PPO Update** (simic/agent/ppo.py): Collect Q(s,op) for all ops using `_compute_value()`, compute variance/spread
3. **Telemetry Emitter** (simic/telemetry/emitters.py): Pass Q-values to PPOUpdatePayload
4. **Aggregator** (karn/sanctum/aggregator.py): Wire Q-values to TamiyoState

## Impact

- Q-values now visible in Sanctum UI during training
- Q-variance diagnostic works (detects if critic ignores op conditioning)
- Better visibility into Policy V2 value head behavior

## Prevention

Use "schema-first + telemetry TODO" pattern: when adding UI fields, immediately add TODO comment in telemetry pipeline to track missing wiring.
