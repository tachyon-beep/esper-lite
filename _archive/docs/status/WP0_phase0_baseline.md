# WP0 Phase 0 Baseline Snapshot

- Git commit: `c9149ecb503de4c59e03a9cbe61fc19ce3e4b28e`
- Kasmina fallback metrics: 0 active, 0 events_total
- Tamiyo fallback annotations absent by default
- Tolaria emergency bypass dropped total: 0.0

## Emergency / rollback baseline (tests)

- Kasmina fallback regression: `pytest tests/kasmina/test_seed_manager.py::test_seed_manager_uses_fallback_on_failure`
- Tamiyo fallback instrumentation: `pytest tests/tamiyo/test_service.py::test_service_urza_graph_metadata_fallback_via_guard_spec`
- Tamiyo compile fallback counter: `pytest tests/tamiyo/test_service.py::test_compile_fallback_counter_exposed`
- Tamiyo GNN compile fallback: `pytest tests/tamiyo/test_policy_gnn.py::test_policy_compile_fallback`
- Tolaria emergency halt (failed epochs): `pytest tests/tolaria/test_emergency_halt.py::test_l4_halt_on_failed_epochs_streak`
- Tolaria emergency halt (rollback deadline): `pytest tests/tolaria/test_emergency_halt.py::test_l4_halt_on_rollback_deadline_exceeded`
- Tolaria rollback cache: `pytest tests/tolaria/test_rollback_cache.py`

## Rollback toggles

- Kasmina fallback instrumentation can be rolled back by reverting commit c9149ecb503de4c59e03a9cbe61fc19ce3e4b28e or flipping feature flag `KASMINA_FALLBACK_TELEMETRY=0` (to be added if needed).
- Tamiyo fallback annotations can be disabled by reverting the same commit or temporarily adding `TAMIYO_STRICT_COMMAND_IDS=1` as a guard.
- Tolaria emergency bypass logging can be reverted by removing the new metric/log block in `_build_global_packet` (commit c9149ecb503de4c59e03a9cbe61fc19ce3e4b28e).

## Metrics baseline snapshot

```json
{
  "telemetry_flags": {
    "kasmina": {
      "fallback_active": 0.0,
      "fallback_events_total": 0.0
    },
    "tamiyo": {
      "fallback_seed": 0.0,
      "fallback_blueprint": 0.0,
      "synthetic_pause": 0.0,
      "runtime_fallback": 0.0
    },
    "tolaria": {
      "emergency_bypass_dropped": 0.0
    }
  },
  "rollback_plan": {
    "kasmina": {
      "toggle": "kasmina_seed_manager_fallback_flags",
      "rollback_note": "Revert to commit 943d752ed2fdad8659c344f129fff15a22737da8"
    },
    "tamiyo": {
      "toggle": "tamiyo_policy_fallback_annotations",
      "rollback_note": "Revert to commit 943d752ed2fdad8659c344f129fff15a22737da8"
    },
    "tolaria": {
      "toggle": "tolaria_trainer_emergency_metrics",
      "rollback_note": "Revert to commit 943d752ed2fdad8659c344f129fff15a22737da8"
    }
  }
}
```
