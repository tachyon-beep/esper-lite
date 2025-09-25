# Graph Metadata Schema (Urza Extras → Tamiyo)

Purpose
- Define a stable, JSON-serializable schema under `UrzaRecord.extras["graph_metadata"]` that Tamiyo can consume to hydrate structural nodes and edges without bespoke lookups.
- Keep the schema additive and backward-compatible; Tamiyo ignores unknown fields.

Location
- Persisted by Urza under the DB row JSON (`_urza.graph_metadata`), surfaced on `UrzaRecord.extras["graph_metadata"]`.
- Consumed by `TamiyoService._extract_graph_metadata` and normalized by `_normalise_graph_metadata` into `data["graph"]` for the graph builder.

Versioning (optional)
- `schema_version`: string (e.g., `"v1"`). Optional today; reserved for future hard pivots.

Top-level structure
```
{
  "layers": [LayerDescriptor, ...],
  "activations": [ActivationDescriptor, ...],
  "parameters": [ParameterDescriptor, ...],
  "capabilities": {
    "allowed_blending_methods": [string]
  },
  "adjacency": {
    "layer": [[src_idx, dst_idx], [src_idx, dst_idx, strength?], ...]
  },
  "monitors": {
    "by_seed_id": {"seed-1": [layer_idx, ...], ...},
    "by_seed_index": {"0": [layer_idx, ...], ...},
    "thresholds": {"risk_max": float, "stage_min": float}
  },
  "monitored_layers": [[seed_idx, layer_idx], [seed_idx, layer_idx, weight?], ...]
}
```

Notes
- `layers/activations/parameters` are arrays; node indices used in `adjacency.layer` refer to 0-based positions in their arrays.
- `monitors` supports multiple representations for convenience:
  - `monitored_layers` with `[seed_index, layer_index, weight?]` triples (or pairs)
  - `monitors.by_seed_id` as a map of seed IDs to layer indices
  - `monitors.by_seed_index` as a map of seed indices (string or int) to layer indices
  - `monitors.thresholds` acts as a filter for both explicit and heuristic monitors (see thresholds)
- Tamiyo ignores unknown fields; arrays and maps can be partially provided.

Descriptors
- LayerDescriptor
  - `layer_id`: string (recommended)
  - `type`: string (e.g., `"linear"`, `"layer_norm"`, ...)
  - `depth`: int (0-based; used for depth normalization)
  - `latency_ms`: float (optional)
  - `parameter_count`: int (optional)
  - `dropout_rate`: float (optional)
  - `weight_norm`: float (optional)
  - `gradient_norm`: float (optional)
  - `activation` or `activation_type`: string (optional)

- ActivationDescriptor
  - `activation_id`: string (recommended)
  - `type`: string (e.g., `"relu"`, `"gelu"`)
  - `saturation_rate`: float (optional)
  - `gradient_flow`: float (optional)
  - `computational_cost`: float (optional)
  - `nonlinearity_strength`: float (optional)

- ParameterDescriptor
  - `name`: string (required)
  - `min`: float (optional)
  - `max`: float (optional)
  - `span`: float (optional; default= max - min)
  - `default`: float (optional; default= (min + max)/2)

Edges
- Layer connects
  - `adjacency.layer`: list of pairs or triples
    - `[src_idx, dst_idx]` or `[src_idx, dst_idx, strength]`
    - When `strength` is provided, Tamiyo uses it as the last edge attribute; otherwise it uses a risk proxy.

- Seed monitors
  - Prefer explicit definitions:
    - `monitored_layers`: `[seed_idx, layer_idx]` or `[seed_idx, layer_idx, weight]`
    - `monitors.by_seed_id`: `{seed_id: [layer_idx, ...]}`
    - `monitors.by_seed_index`: `{string_or_int_index: [layer_idx, ...]}`
  - If not present, Tamiyo falls back to a heuristic using packet SeedState `layer_depth`.

Edge attribute semantics (current)
- Connects: `[src_depth_norm, dst_depth_norm, risk|strength]`
- Monitors: `[seed_stage_norm, seed_risk, weight|layer_depth_norm]`
- Attribute dimension is fixed at 3 for compatibility; future extensions can repurpose the third slot to carry strength/flow metrics.

Thresholds
- `monitors.thresholds` filters explicit and heuristic monitors:
  - `risk_max`: include only seeds with `risk <= risk_max`
  - `stage_min`: include only seeds with `stage_norm >= stage_min`
- Invalid or non-numeric thresholds are ignored.

Coverage and masks
- Feature coverage includes edge presence markers:
  - `edges.layer_connects`: boolean coverage (present/absent)
  - `edges.seed_monitors`: boolean coverage (present/absent)
- Node feature presence continues to use paired value/mask channels; categorical presence masks are included when feature dims are configured to expose them.

Fallback behavior
- If `adjacency.layer` is absent or invalid, Tamiyo falls back to a simple chain: L0→L1→…
- If monitors are absent, Tamiyo maps each seed to the layer indicated by SeedState `layer_depth` clamped to [0, len(layers)-1].
- Out-of-bounds or invalid indices are ignored.

Examples
```
{
  "layers": [
    {"layer_id": "BP-L0", "type": "linear", "depth": 0, "latency_ms": 6.0},
    {"layer_id": "BP-L1", "type": "relu", "depth": 1}
  ],
  "activations": [{"activation_id": "BP-A0", "type": "relu"}],
  "parameters": [{"name": "alpha", "min": 0.05, "max": 0.95, "span": 0.9, "default": 0.5}],
  "capabilities": {"allowed_blending_methods": ["linear", "cosine"]},
  "adjacency": {"layer": [[0, 1, 0.8]]},
  "monitors": {
    "by_seed_id": {"seed-1": [0]},
    "thresholds": {"risk_max": 0.8}
  },
  "monitored_layers": [[1, 1]]
}
```

Future extensions (non-breaking)
- `optimizer_priors`: e.g., per-parameter priors or hints
- `flow_metrics`: richer per-edge data (data rate, bandwidth, latency distributions)
- `bsds_cache_ptr`: pointer to a BSDS cache entry for large artifacts

Implementation notes
- Urza persists `_urza.graph_metadata` using fast JSON (orjson) where available.
- Tamiyo normalizes the graph block and ignores unknown fields.
- Keep lists bounded and avoid deeply nested structures for performance.

