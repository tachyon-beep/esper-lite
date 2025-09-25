# Tamiyo Review Findings

## gnn.py
- GAT layers configure `edge_dim=config.edge_feature_dim` for every relation, but `forward` only attaches edge attributes when the input data provides them; when a relation has no features `HeteroConv` hands `edge_attr=None` to `GATConv`, which crashes with a linear-edge transform. Either require features for every relation or inject zero tensors before calling the layer.
- Edge attributes are moved with `edge_attr.to(next(iter(x_dict.values())).device)`; if the graph is already on a CUDA device this works, but if the module has been moved independent of the data the cast can pick up the wrong target device or fail when `x_dict` is empty. Drive the device choice off a module parameter (`self._policy_head.weight.device`) instead of the input dictionary.
- Node encoders are hard-wired to a fixed relation list; any new node type silently bypasses the encoder yet the relation list still references it. Prefer declaring the schema via `TamiyoGNNConfig` (or validating the data) so we fail fast when the prototype grows new node types instead of producing partially-initialised embeddings.

## graph_builder.py
- `build` falls back to `bp-unknown` when no blueprint ID is present; per prototype policy we should fail fast instead of inventing IDs (docs/prototype-delta/tamiyo/README.md emphasises strict deps).
- `_FeatureNormalizer.flush` swallows IO errors and silently keeps stale stats; expose telemetry or raise so operators know normalisation data is invalid.

- `_FeatureNormalizer.flush` silently ignores write failures; if IO keeps failing the EWMA drifts unbounded and we never surface it—emit telemetry or escalate so operators can fix the persistence path (src/esper/tamiyo/graph_builder.py:56-88).
- `TamiyoGraphBuilder.build` defaults to `packet.packet_id or training_run_id` for blueprint IDs; that couples blueprint identity to run IDs. Enforce the presence of a real blueprint ID and fail fast instead of generating `bp-unknown` graphs (src/esper/tamiyo/graph_builder.py:171-181).
- Edge population trusts metadata counts (e.g., `len(layer_ids)`), but `max_layers`/`max_parameters` caps mean some relations will be truncated while coverage stats still report full availability—validate and cap edges to avoid dangling indexes (src/esper/tamiyo/graph_builder.py:191-214).
- Feature builders inject 0.0 with a presence bit when values are missing. That masks instrumentation bugs; in the prototype we should alert Tamiyo operators instead of silently filling defaults (e.g., `_build_global_features` at src/esper/tamiyo/graph_builder.py:217-244).

## __init__.py
- Docstring points to the full detailed design; prototype-delta says Tamiyo diverges, so reference the delta doc instead of implying adherence to the legacy spec (src/esper/tamiyo/__init__.py:1-6).
- Module re-exports everything, committing us to a wide API surface; pare `__all__` down to the stable entry points (`TamiyoService`, policy) so we can evolve internals without breaking consumers (src/esper/tamiyo/__init__.py:8-21).

## persistence.py
- Docstring references the full detailed design/backlog; switch to the prototype-delta reference to avoid implying we still implement the legacy WAL contract (src/esper/tamiyo/persistence.py:1-8).
- `_load_from_disk` aborts on the first truncated/corrupt entry and discards the tail of the WAL; we should drop only the bad record and keep reading so field reports aren’t silently lost (src/esper/tamiyo/persistence.py:72-100).
- `_rewrite` replaces the WAL without fsyncing the parent directory, so a crash after rename can still lose the pointer; call `os.fsync` on the directory after `replace` (src/esper/tamiyo/persistence.py:58-69).
- Field reports lacking `issued_at` are treated as “now”, meaning they never age out; either require the timestamp or fail fast so retention can work (src/esper/tamiyo/persistence.py:44-60).

## policy.py
- `_build_command` still injects `seed-1`/`bp-demo` defaults; remove the placeholders so invalid inputs surface instead of producing synthetic commands.

- `select_action` returns a synthetic pause command whenever no seed candidates exist, masking upstream graph issues; per prototype policy we should fail fast or escalate instead of silently degrading (src/esper/tamiyo/policy.py:273-318).
- Runtime/compile failures fall back to CPU execution or a pause command, so Tamiyo continues with stale state; propagate the error so operators know the policy failed rather than issuing no-op commands (src/esper/tamiyo/policy.py:333-406).
- `_build_command` substitutes `"seed-1"`/`"bp-demo"` when IDs are missing, which invents resources that don’t exist; require real IDs or abort the command (src/esper/tamiyo/policy.py:884-902).
- Blend mode annotations instantiate a fresh `EsperSettings` per call, reading environment globals synchronously; cache a resolved settings object so policy latency isn’t dominated by config IO (src/esper/tamiyo/policy.py:644-707).

## service.py
- `evaluate_epoch` is retained purely for backwards compatibility even though 3A forbids legacy entry points; drop it and enforce the step API.
- Timeout handling relies on `future.cancel()`; the worker keeps running after the pause command returns, so we violate the strict deadline contract. Replace with cancellable workers and surface the timeout as a CRITICAL failure.

- `evaluate_epoch` is still exposed as a “backwards compatibility” entry point; drop it and enforce the step-driven API so we don’t carry legacy semantics (src/esper/tamiyo/service.py:206-212).
- `_run_policy` wraps `select_action` in a `ThreadPoolExecutor` and calls `future.cancel()` on timeout; as in Tolaria, the worker keeps running and we return a synthetic pause command instead of failing fast. Replace with a cancellable worker (process or persistent thread) and surface hard failures (src/esper/tamiyo/service.py:943-988).
- Timeout commands and metadata-fetch timeouts degrade to `COMMAND_PAUSE`, masking inference/Urza failures; per prototype policy we should escalate rather than silently pausing (src/esper/tamiyo/service.py:955-986, 1014-1050).
- `_ensure_blueprint_metadata_for_packet` submits Urza lookups to the same executor and again relies on `future.cancel()`; this suffers the same cancellation bug and can leave hanging I/O while we pretend the cache miss was fine (src/esper/tamiyo/service.py:1020-1050).
- Risk/telemetry processing sprinkles new `EsperSettings()` lookups throughout (e.g., field-report retry config, degraded-input thresholds) instead of freezing config at init; cache the resolved settings to keep the hot path deterministic (multiple sites around src/esper/tamiyo/service.py:150-340, 1700-1900).

## Architectural Improvements
- Introduce a single Tamiyo configuration object that resolves EsperSettings once and feeds policy, graph builder, risk gates, and service timeouts; eliminate scattered settings lookups and ensure consistent defaults.
- Replace per-call ThreadPoolExecutor usage with a dedicated async or persistent worker layer that handles policy inference, metadata fetch, and cancellation coherently; surface failures instead of synthesizing pause commands.
- Split risk gating, policy inference, and telemetry emission into clear components so the service orchestrator only routes results; this makes it easier to evolve risk policies without touching inference code.
- Build a proper blueprint metadata cache backed by Urza change notifications instead of ad-hoc prewarm calls, so policy sees consistent data and we can evict stale entries deterministically.
- Centralize registry/normalizer persistence (seed/layer/activation/feature stats) behind a small storage manager to coordinate flush cadence and error reporting across policy and graph builder.

