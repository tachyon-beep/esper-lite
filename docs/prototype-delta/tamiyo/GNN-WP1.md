# Tamiyo WP1 — Strategic Policy (Hetero-GNN Upgrade)

## Objective

Replace the feed-forward Tamiyo policy stub with the 4-layer hetero-GNN architecture (GraphSAGE → GAT) defined in `docs/design/detailed_design/03-tamiyo-unified-design.md` and close the prototype deltas called out in `docs/prototype-delta/tamiyo/README.md`. Maintain ≤45 ms inference latency and integrate the PyTorch 2.8 optimisations documented in `pytorch-2.8-upgrades.md`.

## Key References

- `docs/design/detailed_design/03-tamiyo-unified-design.md`
- `docs/design/detailed_design/03.2-tamiyo-policy-training.md`
- `docs/design/detailed_design/03.4-tamiyo-integration-contracts.md`
- `docs/prototype-delta/tamiyo/pytorch-2.8-upgrades.md`
- `docs/prototype-delta/tamiyo/step-state.md`
- `docs/prototype-delta/tamiyo/telemetry.md`
- `docs/prototype-delta/tamiyo/risk-engine.md`

## Deliverables

1. New hetero-GNN policy module supporting the GraphSAGE→GAT stack with action/parameter (and optional value/risk) heads.
2. Updated `TamiyoPolicy` wiring, including graph assembly from `SystemStatePacket`, registry integration, device management, and inference optimisation (`torch.compile`, autocast, TF32).
3. Performance + correctness test suite proving inference results, regression coverage, and p95 latency ≤45 ms under representative load.
4. Documentation updates explaining the new architecture, inputs, and deployment notes.
5. Registry parity and coverage semantics (WP14–WP15):
   - Persist and share categorical registries (layer/activation/optimizer/hazard) with Simic; embed registry digests in checkpoints for load-time validation.
   - Emit coverage both as an average and as per-type ratios (node/edge families) to aid downstream diagnosis; attach the full per-key map and per-type map as command annotations.

## Work Breakdown

### 1. Design Consolidation
- Re-read the referenced design docs to capture node/edge types, feature schema, and action outputs.
- Extract prototype deltas (step-state payload, risk-engine signals) to ensure inputs are compatible.
- Decide on namespace for the new module (`src/esper/tamiyo/gnn.py` or similar).

### 2. Model Implementation
- Implement the hetero-GNN using PyTorch Geometric (or custom layers if the project avoids additional deps). Layers: two GraphSAGE + two GAT as per design, followed by MLP heads.
- Support optional value/risk heads but keep them noop-compatible until consumers land.
- Add configuration dataclass for hyperparameters (hidden dims, dropout, attention heads, device).
- Decide on parameter initialisation (e.g., Xavier uniform for linear layers, attention coefficient init per design).
- Expose forward signature accepting a dict of node-type tensors and adjacency info to align with PyG conventions.

### 3. Feature & Graph Construction
   - Expand packet encoding: build hetero graph nodes for numeric metrics, seeds, blueprints, and global context; ensure graceful handling of missing data.
   - Normalise numerical features (e.g., z-score or min-max per design); add masks where metrics are absent.
   - Maintain embedding registries for categorical IDs; persist to the same JSON files expected by Simic. Pre-seed common optimizer families (sgd/adam/adamw/rmsprop/adagrad) to stabilize indices.
   - Coverage semantics:
     - Per-key mask coverage is tracked for every feature; builder exposes `feature_coverage` and the typed aggregation `feature_coverage_types` with counts‑weighted ratios.
     - Service exports telemetry for average and per-type coverage and attaches both to AdaptationCommand annotations.

   **Normalisation Constants & Storage**
   - Metrics (`loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`) → maintain exponentially-weighted mean/variance (α=0.1) persisted under `var/tamiyo/gnn_norms.json`; initialise with design-provided priors (`loss`: μ=0.8, σ=0.3; `gradient_norm`: μ=1.0, σ=0.5; `samples_per_s`: μ=4500, σ=500; `hook_latency_ms`: μ=12, σ=6).
   - Blueprint numeric fields (`parameter_count`, `learning_rate`, `momentum`, `gradient_variance`) → normalise using log scaling (`log1p`) followed by z-score with priors derived from Urza catalog snapshots (collect baseline during integration tests and bake into fixture).
   - Seed statistics (`activity_level`, `success_rate`, `risk`) → clamp to [0,1], then feed directly; if upstream values missing, substitute 0.5 and set accompanying mask flag.
   - Store masks alongside feature tensors as additional channels so the GNN can learn to de-weight imputed values.

### 4. Inference Optimisation (PyTorch 2.8)
- Wrap forward passes with `torch.inference_mode()` and device-aware autocast (bfloat16 for CUDA, float32 otherwise).
- Attempt `torch.compile(..., dynamic=True, mode="reduce-overhead")`; log once on failure and fall back to eager.
- On CUDA, perform a best‑effort warm‑up forward on a tiny hetero‑graph at init to reduce first‑step variance; expose `tamiyo.gnn.compile_warm_ms` when available.
- Enable TF32 globally when CUDA is available (`torch.set_float32_matmul_precision('high')`, `allow_tf32=True`).
- Provide hooks to preload weights onto GPU and pin host memory buffers if needed.

### 5. Service Integration
- Update `TamiyoPolicy` to instantiate the new GNN, keep the public API (`select_action`, `last_action`, `encode_tags`).
- Ensure `TamiyoService.update_policy` can load new checkpoints (state dict compatibility) and survive hot-swaps.
- Propagate new telemetry (policy version string identifying the GNN build, optional risk/value outputs).

### 6. Testing & Validation
- **Unit tests**: graph assembly fixtures, deterministic forward outputs, compile fallback when unsupported.
- **Property tests**: round-trip registry indices, ensure missing metrics degrade to padding without crashes.
- **Performance tests**: measure inference latency on CPU (and GPU if available) with representative packets; assert p95 ≤45 ms. Add an opt‑in small builder p95 check to guard regressions.
- **Integration tests**: run `TamiyoService.evaluate_step` end-to-end using the new policy to confirm command emission, telemetry, and field reports remain valid.

### 7. Documentation & Migration
- Update module docstrings to cite the detailed design sections.
- Add notes to `docs/prototype-delta/tamiyo/README.md` summarising the architecture upgrade and perf characteristics.
- Provide operator guidance on checkpoint formats and compatibility with Simic (e.g., naming convention, version tagging).

## Detailed Coding Tasks & Required Inputs

1. **Graph Schema Definition**
   - Extract the official node specs from `03.1-tamiyo-gnn-architecture.md`:
     - `layer` nodes → 8 core features (`layer_type`, `input_channels`, `output_channels`, `activation_type`, `dropout_rate`, `weight_norm`, `gradient_norm`, `update_frequency`), raw dim 128 → encoded 256.
     - `seed` nodes → 7 features (`seed_id`, `monitoring_scope`, `activity_level`, `success_rate`, `risk_tolerance`, `specialization`, `lifecycle_stage`), raw dim 64 → encoded 256.
     - `activation` nodes → 4 features (`activation_type`, `saturation_rate`, `gradient_flow`, `computational_cost`), raw dim 32 → encoded 128.
     - `parameter` nodes → 4 features (`parameter_count`, `learning_rate`, `momentum`, `gradient_variance`), raw dim 16 → encoded 256.
   - Record required edge relations and attributes:
     - (`layer`,`connects`,`layer`) with `connection_strength`, `data_flow_volume`, `latency_ms`.
     - (`seed`,`monitors`,`layer`) with `monitoring_frequency`, `adaptation_trigger_threshold`.
     - (`layer`,`feeds`,`activation`) with `activation_frequency`, `saturation_events`.
     - (`activation`,`influences`,`layer`) with `gradient_contribution`, `nonlinearity_strength`.
     - (`parameter`,`tunes`,`layer`) with `update_magnitude`, `convergence_rate`.
   - Audit what data is currently exposed via `SystemStatePacket` (`loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`, optional `seed_states`) and blueprint caches to identify gaps; specify interim fallbacks (e.g., default/learned embeddings) for features not yet delivered.
     - Enumerate categorical vocabularies needed for embeddings (layer types, activation types, specialisations) and confirm/extend registry capacities in `TamiyoPolicyConfig` (seed_vocab, blueprint_vocab, add layer_vocab etc.).
   - Capture neighbourhood sampling settings from the design (`layer_1_neighbors=25`, `layer_2_neighbors=10`, `layer_3_neighbors=5`, `layer_4_neighbors=5`) for later batching implementation.

   **Quick-reference schema (copy into adapter module docstring)**

   | Node Type   | Raw Dim | Encoded Dim | Key Features / Source |
   | ----------- | ------- | ----------- | ---------------------- |
   | `layer`     | 128     | 256         | `layer_type` (enum → embedding 32-d), `input_channels`, `output_channels`, `activation_type` (enum), `dropout_rate`, `weight_norm`, `gradient_norm`, `update_frequency`, plus per-step metrics (`loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`) normalised via EWMA table. |
   | `seed`      | 64      | 256         | `seed_id` (registry), `monitoring_scope`, `activity_level`, `success_rate`, `risk_tolerance`, `specialization` (enum), `lifecycle_stage`, `alpha`, `risk_score`, `allowed_blending_methods` (multi-hot 4-dim), mask flags. |
   | `activation`| 32      | 128         | `activation_type`, `saturation_rate`, `gradient_flow`, `computational_cost`, fallback to learned embedding per blueprint. |
   | `parameter` | 16      | 256         | `parameter_count` (log scaled), `learning_rate`, `momentum`, `gradient_variance`, `optimizer_family` (enum). |

   | Edge Type | Attributes |
   | --------- | ---------- |
   | (`layer`,`connects`,`layer`) | `connection_strength`, `data_flow_volume`, `latency_ms` → normalise by max per-graph. |
   | (`seed`,`monitors`,`layer`) | `monitoring_frequency`, `adaptation_trigger_threshold`; binary mask if unknown. |
   | (`layer`,`feeds`,`activation`) | `activation_frequency`, `saturation_events`. |
   | (`activation`,`influences`,`layer`) | `gradient_contribution`, `nonlinearity_strength`. |
   | (`parameter`,`tunes`,`layer`) | `update_magnitude`, `convergence_rate`. |
   | (`seed`,`allowed`,`parameter`) (future) | 1-hot edges for permitted blending/optimizer combos; placeholder until Kasmina export lands. |

   **Normalisation constants cheat-sheet**

   ```python
   METRIC_PRIORS = {
       "loss": (0.8, 0.3),
       "gradient_norm": (1.0, 0.5),
       "samples_per_s": (4500.0, 500.0),
       "hook_latency_ms": (12.0, 6.0),
   }
   BLUEPRINT_PRIORS = {
       "parameter_count": ("log1p", 1.0, 0.4),
       "learning_rate": ("log1p", 0.01, 0.005),
       "momentum": (None, 0.9, 0.05),
       "gradient_variance": ("log1p", 0.001, 0.0005),
   }
   SEED_PRIORS = {
       "activity_level": (0.5, mask_if_missing=True),
       "success_rate": (0.5, mask_if_missing=True),
       "risk_score": (0.5, mask_if_missing=True),
   }
   BLENDING_METHODS = ["linear", "cosine", "warmup_hold", "sigmoid"],  # coordinate with Kasmina
   DEFAULT_METHOD = "linear"
   ```


   **Step-to-Graph Encoding Execution Plan**
   - Metrics → `layer` node features:
     - `loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms` map to continuous slots; normalise using rolling mean/variance captured from Tolaria telemetry (`var/tolaria/metrics_snapshot.json` – new artifact to emit) with epsilon=1e-6.
     - Add boolean mask feature per metric to flag missing values; default to 0 when absent, mask=0 so the GNN learns to ignore.
   - `seed_states` → `seed` nodes:
     - Pull `stage`, `quarantine_only`, `alpha`, `risk`, `policy_version` from Leyline seed state; convert enums via embedding tables (size = 64) and concatenate with numeric stats (`alpha`, `risk` normalised to [0,1]).
     - Augment node features with available Kasmina attributes needed for blending decisions (e.g., `alpha_schedule`, `allowed_blending_methods` once exported); until then, add a categorical placeholder with default `linear` method so the policy can still emit choices.
   - Blueprint/context → `parameter`/`activation` nodes:
     - Query Urza metadata cache (async warmed by TamiyoService) for parameter counts, learning rates, stage descriptions; fallback to learned embeddings if cache miss (<5 ms budget).
     - Activation nodes derived from layer metadata: if blueprint lacks explicit activation info, use default vector learned per layer type.
   - Edge construction:
     - Derive `layer→layer` connections from blueprint graph adjacency; store edge weights as floats (normalised by max fan-out) and attach to edge_attr tensor.
     - `seed→layer` edges: connect each seed node to monitored layer IDs from `seed_states.monitored_layers`; if absent, connect to top-k layers sorted by loss contributions (default k=3).
     - When Kasmina provides method capability metadata, add `seed→parameter` or `seed→blending_method` edges representing allowed transitions; until available, track capability masks in seed feature vector.
   - Batch handling:
     - For per-step inference we operate on single graphs (batch size = 1). Still, use PyG batch indices to stay compatible with tooling (all nodes share batch idx 0).
   - Data gap log:
     - Track missing feature coverage in structured telemetry (`tamiyo.gnn.feature_coverage`) to prioritise upstream exporters.

2. **Module Skeleton (`src/esper/tamiyo/gnn.py`)**
   - Create `TamiyoGNNConfig` capturing: base hidden dim 256, activation dim 128, 4 attention heads, dropout 0.2, neighbourhood sampling knobs, device, compile flag, and telemetry hook names.
   - Implement `TamiyoGNNEncoder` mirroring the spec:
     - Layers 1–2: `HeteroConv` over GraphSAGE relations with mean/max aggregations.
     - Layers 3–4: `HeteroConv` over GAT relations (`heads=4`, `concat=False`).
     - Apply GELU + dropout after each layer; include optional residual connections (design mentions “hardware-validated dimension progression” – confirm whether skip connections are mandated).
   - Implement node encoders exactly as specified (Linear→ReLU→LayerNorm per node type) with Xavier-uniform initialisation and LayerNorm eps per PyTorch defaults.
   - Expose forward signature `(x_dict, edge_index_dict, batch=None)` returning:
     - `risk_logits` (5 classes per spec), `value_estimate` (scalar),
     - `policy_logits` (32 kernel choices) and a companion `policy_params` tensor encoding continuous adjustments (e.g., LR deltas, blending schedule scalars),
     - `blending_method_logits` over available methods (`linear`, `cosine`, `warmup_hold`,…); map indexes to method names in TamiyoPolicy when constructing AdaptationCommands.
     - intermediate `graph_embedding` for telemetry.

   **Framework Choice (PyTorch Geometric vs Custom)**
   - **Decision**: adopt PyTorch Geometric (`torch-geometric>=2.5.0`, plus runtime dependencies `torch-scatter>=2.1.2`, `torch-sparse>=0.6.18`, `torch-cluster>=1.6.3`, `torch-spline-conv>=1.2.2`) as the default implementation path and install these packages as core dependencies.
     - Update CI workflows to install the base project; the Tamiyo policy now assumes PyG is present and will fail fast if imports break.
     - Document local install instructions in README, noting that PyG wheels matching the pinned torch build are mandatory for any Tamiyo deployment.
   - **Hand-rolled fallback sketch (for future work if PyG is rejected)**:
     - Implement custom hetero convolutions using `torch.mm` over adjacency matrices built via `scatter_add` (node-type specific linear layers + neighbourhood aggregation).
     - Use attention weights computed with per-edge MLP (for GAT equivalent) and normalise with `softmax` over neighbour dimension.
     - Provide performance benchmarks before committing; expect higher maintenance cost.
   - Testing strategy:
     - Introduce “no-PyG” test mode by monkeypatching wrappers to pure-PyTorch stubs for logic coverage (ensures unit tests run in constrained CI environments).

3. **Packet → Graph Adapter**
   - Build a dedicated adapter (`tamiyo.graph_builder`) responsible for:
     - Translating `SystemStatePacket.training_metrics` into numeric tensors (loss, gradient_norm, samples_per_s, hook_latency_ms) with z-score/ min-max constants derived from design appendices or recorded telemetry (action item: gather baseline stats from Tolaria runs).
     - Hydrating seed nodes from `seed_states` (if present) using Leyline enums (stage, quarantine flags) and embedding registries for `seed_id` & `specialization`.
     - Synthesising layer/activation/parameter nodes from static blueprint metadata (requires pull from Urza/Karn caches; capture dependencies) or using learned embeddings until full schema is wired.
     - Emitting hetero edge indices in COO format with per-edge attributes (store as feature tensors or encode via attention weights if attributes unavailable initially).
   - Define graceful degradation when optional data is missing: use zero vectors plus mask flags so the network learns to treat absent information consistently; log once per session for observability.

4. **Inference Pipeline**
   - Set global PyTorch precision knobs during policy init (`torch.set_float32_matmul_precision('high')`, `torch.backends.cuda.matmul.allow_tf32 = True`, etc.).
   - Wrap calls in `torch.inference_mode()` and optionally `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` when CUDA is detected; expose config knob to disable autocast for CPU-only deployments.
   - Attempt `torch.compile(self.forward, dynamic=True, mode='reduce-overhead')` during `TamiyoPolicy` construction, store compiled callable, and guard with a single warning on failure.
   - Ensure adapter outputs land on the configured device via `.to(device, non_blocking=True)`; pin CPU tensors when CUDA is enabled to minimise H2D latency.

5. **State Dict & Checkpoint Handling**
   - Register submodules (`node_encoders`, `convs`, heads) so `state_dict()` mirrors the new architecture; add migration shim that rejects legacy FFN checkpoints with clear error/telemetry (`policy_update_rejected`).
   - Extend `TamiyoPolicyConfig` with `architecture_version="gnn_v1"` and surface via telemetry/field reports (
`tamiyo_policy_version`).
   - Define checkpoint naming and metadata schema (`tamiyo_gnn_v1.pt` containing config dict + weights) for Simic to emit compatible updates; document required tensor shapes for trainers.

6. **Testing Matrix**
   - Unit tests:
     - Graph builder converts sample `SystemStatePacket` + mock blueprint metadata into expected node/edge tensors (compare shapes, masks).
     - `TamiyoGNN` forward returns risk/value/policy outputs with correct shapes and probability simplex invariants.
     - Simulate `torch.compile` failure via monkeypatch, assert fallback path and warning log.
   - Property tests:
     - Registry round-trip: repeated lookups of same seed/blueprint ID yield consistent indices and embeddings.
     - Missing optional data (no `seed_states`, zero metrics) does not crash and results contain neutral embeddings.
   - Performance tests: run N=1000 inference passes on synthetic graphs sized according to design (`typical_nodes=10k`, `typical_edges=50k`) and assert p95 latency ≤45 ms; gate on mark `@pytest.mark.perf`.
   - Integration: update `tests/tamiyo/test_service.py` to validate `evaluate_step` with new GNN policy, ensuring telemetry still includes loss delta etc., command annotations incorporate policy metadata, and `AdaptationCommand.seed_operation.parameters['blending_method']` aligns with the chosen method when multiple options are available.

7. **Telemetry & Logging Updates**
   - Inject policy version / architecture metadata into `TamiyoService` telemetry (`health_indicators['policy'] = architecture_version`, `health_indicators['gnn_compile'] = 1/0`).
   - Emit metrics:
     - `tamiyo.gnn.inference.latency_ms` (per evaluation step, p50/p95 in tests),
     - `tamiyo.gnn.compile_enabled` (gauge 0/1),
     - `tamiyo.blending.method` (enum indicator via telemetry event attribute).
   - Update `tests/tamiyo/test_service.py` to assert telemetry indicators/metrics after calling `evaluate_step`.
   - Add structured log when `torch.compile` is disabled or when fallback occurs, tagging with reason (`logger.info("tamiyo_gnn_compile_disabled", reason=...)`).

8. **Documentation & Rollout Notes**
   - Draft migration instructions for operators (expected performance, new env vars if any).
   - Document data dependencies explicitly: highlight current gaps (e.g., missing layer/parameter metrics in `SystemStatePacket`) and interim fallbacks; create backlog items for Tolaria/Kasmina/Urza exports needed to fully populate node features.
   - Coordinate with Simic team to schedule corresponding replay/training upgrades; capture action items in `docs/prototype-delta/tamiyo/implementation-roadmap.md` if follow-up is needed.
   - Update Kasmina interface docs to formalise accepted `blending_method` parameter values and required schedule fields so Tamiyo’s outputs can be consumed reliably; ensure telemetry taxonomy includes `tamiyo.blending.method`.

## Risk & Mitigation Checklist

- **Latency regression**: benchmark early, profile hotspots (attention layers, data transfer) and tune hidden sizes/attention heads if needed.
- **Dependency footprint**: if introducing PyTorch Geometric, ensure `pyproject.toml` pins versions and tests mock the dependency when unavailable.
- **Checkpoint compatibility**: document migration path from FFN weights (incompatible) and guard `TamiyoService.update_policy` to warn on old payloads.
- **Numerical stability**: validate autocast/TF32 do not skew outputs; add fallbacks to float32 when policy runs on CPU.

## Acceptance Criteria

- `TamiyoPolicy` uses the hetero-GNN architecture and passes all new and existing tests.
- Inference p95 (on dev hardware) ≤45 ms; compile fallback path logged once when disabled.
- Telemetry includes updated policy version/metadata and continues to expose metrics required by the taxonomy.
- No production behaviour regressions in Tamiyo service, persistence, or field reports.

## Full WP1 Hetero-GNN Reference

The production Tamiyo policy instantiates the four-stage hetero-GNN defined in `03.1-tamiyo-gnn-architecture.md`. The network encodes each node type with a linear → ReLU → LayerNorm stack that projects raw blueprint and telemetry fields into a shared hidden space before executing the convolution stack.

| Node Type | Raw Feature Budget | Encoder Output | Notable Fields |
| --- | --- | --- | --- |
| `layer` | 128 channels (structural telemetry, per-step loss/gradient metrics, adjacency hints) | 256-dim | Includes categorical embeddings for `layer_type`/`activation` plus log-scaled loss, gradient norm, samples/sec, latency, and mask bits for imputed slots. |
| `seed` | 64 channels (Leyline stage, activity/risk metrics, specialization) | 256-dim | Registry-based embeddings for `seed_id`, lifecycle enums, blending capability masks, and Tolaria-provided success bands. |
| `activation` | 32 channels (saturation, gradient flow, compute cost) | 128-dim | Derived from blueprint activation metadata with learned fallbacks when Urza omits values. |
| `parameter` | 16 channels (optimizer statistics, counts, variance) | 256-dim | Log-scaled parameter count, learning rate, momentum, variance, optimizer family embedding, and mask channels. |

The message-passing pipeline applies two GraphSAGE hetero-convolution stages (hidden width 256) followed by two 4-head GAT hetero-convolutions (hidden width 128) with GELU activations, dropout 0.2, residual connections, and per-relation layer norms. Pooling averages the `layer` node embeddings to produce the global summary vector consumed by the output heads.

Output heads match the WP1 contract:

- `policy_logits`: 32-way action distribution over blueprint kernels, paired with `policy_params` tensors that encode per-action continuous deltas (learning-rate shifts, schedule scalars).
- `risk_logits`: five-class safety gate consistent with the Blueprint Safety Data Sheet taxonomy.
- `value_estimate`: scalar inference of expected blueprint uplift for downstream planners.
- `blending_method_logits`: categorical selection over Kasmina-supported blending families, exported alongside per-seed schedule parameters.
- `telemetry_embedding`: auxiliary 64-dim embedding persisted for observability and health scoring.

For a canonical list of required upstream signals, normalisation constants, and mask semantics, see [`diff/gnn-inputs.md`](diff/gnn-inputs.md).
