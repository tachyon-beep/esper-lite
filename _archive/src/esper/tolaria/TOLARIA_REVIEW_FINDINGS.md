# Tolaria Review Findings

- Tolaria is pinned to PyTorch 2.8; remove pre-2.8 compatibility shims and ensure all features assume that baseline.

## aggregation.py
- PCGrad handling only projects the leading gradient and leaves the rest untouched, so conflict resolution never matches the spec. Implement pairwise projections or shuffle per PCGrad to deliver WP8 expectations.
- `combine_flat_grads` only projects the first gradient against subsequent ones, so later gradients never undergo PCGrad conflict resolution; rework to project each pair (e.g., shuffle + per-pair updates) to match spec.
- `aggregate_weighted` assumes rank-1 grads when reshaping weights; expand weights across all trailing dims to avoid shape errors on higher-rank grads.
- `grads_to_flat` returns an empty CPU float64 tensor when no grads exist, causing dtype/device mismatch for GPU/AMP; propagate expected dtype/device or bar empty input.
- `AggregationResult` is unused; remove until there is a caller rather than keeping latent compatibility scaffolding.

## emergency.py
- `EmergencyController` tracks `_bypass_cap`/`_bypass_count` but never uses them; drop the unused state or implement bypass handling rather than carrying dead scaffolding.
- `EmergencyController.escalate` silently swallows broadcast failures (`except Exception`); surface telemetry/logging or re-raise to avoid hiding delivery issues during incidents.
- `LocalEmergencySignal.clear` leaves the prior level/reason/timestamp intact, so consumers read stale metadata after a reset; mirror the shared-memory clear semantics.
- `LocalEmergencySignal.trigger` keeps `_timestamp_ms=None` when the caller omits `monotonic_ms`, leaving consumers without timeline context; populate the timestamp using `monotonic()` like the shared variant.
- Consider validating requested levels against `EmergencyLevel` enum and supporting explicit downgrades instead of locking the controller at the first high level.

## __init__.py
- Module docstring references a "legacy" spec at `old/01-tolaria.md`; if we are prototype-first, drop legacy pointers or archive the doc instead of implying backward compatibility.
- Re-export list mirrors `trainer`; consider pruning exports to the public API we truly support (e.g., hide clients if they are Tolaria-internal plumbing) to avoid committing to accidental surface area.

## lr_controller.py
- `LRController._read_current_lr` defaults to 0.0 when an optimizer lacks an `lr` key; silently switching off the learning rate is dangerous—fail fast instead of muting the optimizer.
- `CosineSchedule` clamps `step` to `[0, t_max]`, which freezes the LR once the loop passes `t_max`; consider wrapping or continuing the cosine cycle to avoid a flat tail.
- `WarmupWrapper` enforces a hard-coded 10% floor on the LR during warmup, which deviates from typical linear warmup and could overshoot for small base LRs—let the schedule decide or make the floor configurable.
- `build_controller` quietly ignores unknown policies by returning `None`; prefer raising to avoid silent misconfiguration.
- Bias between `step` and `epoch` semantics (step counted globally vs per-epoch) is undocumented; clarify or enforce one interpretation to prevent mis-scheduling.

## optimizer_manager.py
- `maybe_rebuild` blindly copies `defaults` into the new optimizer; this misses per-param options and will break optimizers that expect parameter groups with distinct settings—capture full param-group configs instead of just defaults.
- Silent fallback on `load_state_dict` failure (`except Exception: pass`) hides optimizer incompatibility and leaves us training from scratch; fail fast or surface the error so operators know state persistence failed.
- `optimizer` property exposes a mutable reference without guard; consider returning a copy or documenting that callers will mutate internal state.
- There’s no hook to update the circuit breaker when rebuild succeeds but downstream apply fails; rethink the breaker strategy or expose full telemetry to avoid hidden churn.

## profiler.py
- `active_steps` argument is dead; either implement bounded profiling windows or remove the parameter to avoid implying unused behavior.
- All exceptions from the profiler context are swallowed, which leaves operators blind to missing traces; emit telemetry or log to flag failures rather than silently succeeding.
- Export path overwrites the same file on repeated runs (`{name}.json`); include timestamps or step indices to keep distinct trace sessions instead of masking prior captures.
- We don’t guard against `torch.profiler` availability on older PyTorch (checked via `hasattr` but still assumes `ProfilerActivity` exists); tighten the capability check or gate imports.

## rollback.py
- `FastRollbackCache.put` replaces existing snapshots without subtracting their size from `_size_bytes`, so repeated inserts at the same step leak cache quota until evictions misfire.
- `FastRollbackCache.restore` swallows optimizer `load_state_dict` errors; that hides corrupted checkpoints and leaves the trainer with mismatched state—fail fast or surface telemetry instead.
- Two-tier rollback uses `ThreadPoolExecutor` within a context manager; on timeout the worker thread keeps running and `executor.__exit__` blocks until completion, so the deadline is effectively unenforced—shut down without waiting or move to multiprocess execution that can be canceled.
- Deadline signaling via `SharedDeadlineSignal` writes but never exposes the timestamp to callers of `attempt_two_tier_rollback`; plumb the signal result back so operators know when and why the deadline tripped.
- `torch.load` without `weights_only=True` (PyTorch 2.0+) leaves us exposed to arbitrary pickle execution; enforce safe loading when available.

## trainer.py
- `maybe_profile` is invoked with `trace_dir=self._settings.tolaria_profiler_dir` even when the setting is unset (`None`); `Path(None)` raises, so guard or require a concrete directory before enabling profiling (src/esper/tolaria/trainer.py:431-435).
- `_compute_loss` always pushes targets to `self._config.device`; after the runtime falls back to CPU the config still points at the original GPU, producing device-mismatch crashes—use the model/output device instead (src/esper/tolaria/trainer.py:1565-1568).
- `_invoke_tamiyo_generic` and `_apply_kasmina_command` wrap calls in a context-managed `ThreadPoolExecutor`. On timeout, `future.cancel()` is ineffective once work has started and `__exit__` blocks waiting for the worker, so the training loop still hangs—replace with a long-lived executor or a subprocess that can be cancelled (src/esper/tolaria/trainer.py:2087-2106).
- Aggregation config still honors legacy toggles like `tolaria_agg_per_layer_enabled`/`tolaria_agg_per_layer_topk`; drop the backwards-compat layer and require the new `tolaria_seed_layer_*` flags now that we are prototype-first (src/esper/tolaria/trainer.py:356-379).
- `publish_history` drops any remaining emergency signals after applying the per-minute cap, even if they failed to send; that silently loses incident data—preserve the queue or surface a hard failure instead of clearing it (src/esper/tolaria/trainer.py:2223-2249).
- Checkpointing still uses `torch.save` / `torch.load` with pickle semantics; once PyTorch 2.0+ is available switch to `weights_only=True` (and guard loads) to avoid arbitrary code execution vectors (src/esper/tolaria/trainer.py:2305-2345).

## Architectural Improvements
- Introduce a shared asynchronous mediator for Tamiyo/Kasmina calls instead of per-call ThreadPoolExecutor spawns; supports true cancellation and unified telemetry.
- Decompose the trainer into smaller components (control loop, gradient aggregation, telemetry publishing, safety) to avoid a monolith and isolate concerns.
- Create a consolidated state/rollback manager that owns both fast-cache snapshots and WAL checkpointing, so torch load/save safety hardening lives in one place.
- Replace ad-hoc EsperSettings flag probes with a typed Tolaria config schema; remove legacy keys and make breaking changes explicit.
- Centralize telemetry/event construction to a dedicated emitter, enabling batching, shared attributes, and rate-limiting before publishing via Oona.

