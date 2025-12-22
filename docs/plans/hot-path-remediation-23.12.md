# Hot-Path Remediation — 2025-12-23

## Current Status
*   **Target Performance:** ~20 events per second (eps) with 3 active seeds.
*   **Observed Performance:** ~8 events per second (eps) with 1 active seed.
*   **Observed Bloat:** `src/esper/simic/training/vectorized.py` has grown by ~1500 lines due to "readability" refactors and Phase 4/5 logic, introducing significant interpreter overhead.

---

## Findings

### 1. Unstrided Gradient Telemetry (Critical Path)
The `process_train_batch` function in `vectorized.py` calls `_collect_gradient_telemetry_for_batch` for **every training batch**.
*   **Impact:** Even with PyTorch's `_foreach_norm`, this function iterates over 500k+ host parameters to compute squared norms.
*   **Observation:** The code includes comments about "striding telemetry" (e.g., every 10 steps), but the implementation currently gathers host stats every single batch.
*   **Bottleneck:** CPU-side Python overhead for list construction and kernel launch overhead for norm computation.

### 2. Synchronous Telemetry Hub (I/O Blocking)
`NissaHub.emit` is entirely synchronous. Every time an event is emitted, the training loop blocks.
*   **Impact:** `ConsoleOutput` (string formatting/printing) and `FileOutput` (JSON serialization/disk I/O) occur on the main thread.
*   **Observation:** Events like `REWARD_COMPUTED`, `EPOCH_COMPLETED`, and `ANALYTICS_SNAPSHOT` are emitted multiple times per epoch.
*   **Bottleneck:** Training throughput is tightly coupled to the latency of the slowest telemetry backend.

### 3. Interpreter Bloat in `vectorized.py`
The "readability refactor" (Commit `7c75289`) unrolled many operations, adding hundreds of lines of boilerplate and redundant checks inside the core training and validation loops.
*   **Factored Action Parsing:** Redundant indexing of `action_dict` and heavy `__debug__` assertions in every timestep.
*   **Telemetry logic:** Large `if hub:` blocks nested inside the `envs_this_batch` loop, performing dictionary allocations even if filters might later drop the event.
*   **Nested Closures:** `evaluate_fn` for counterfactuals is redefined as a closure inside the environment loop every epoch, leading to repeated object allocation and overhead.

### 4. Redundant Optimizer Validation
In every training batch, the system checks if seed optimizers need to be recreated by comparing sets of parameter IDs.
*   **Observation:** `if {id(p) for p in opt_params} != {id(p) for p in seed_params}:`.
*   **Impact:** This performs multiple Python-level iterations and set operations in the innermost loop.
*   **Bottleneck:** This logic should only trigger on seed lifecycle transitions (GERMINATE, PRUNE, FOSSILIZE), not every batch.

### 5. Counterfactual Pipeline Complexity
While counterfactuals are disabled for >3 seeds, the logic for building the matrix (solo, pair, all-disabled) is scattered across the validation loop.
*   **Observation:** Multiple `with ExitStack()` and `with force_alpha(0.0)` contexts are created and entered per batch.
*   **Impact:** This adds substantial context-manager overhead and increases the number of forward passes performed during validation without efficient batching.

### 6. Shapley Permutation Explosion (BUG-027)
`CounterfactualEngine.compute_shapley_values` materializes the full list of all $N!$ permutations before sampling.
*   **Observation:** For $N \ge 10$, this will OOM or stall. Even for $N=3$, `len(list(permutations(...)))` is performed twice.
*   **Status:** This is a latent "performance timebomb" that degrades the responsiveness of the episode-end analytics.

---

## Recommendations

### Phase 1: High-Impact "Low Hanging Fruit" (Immediate)
1.  **Stride Gradient Telemetry:** Implement a `gradient_telemetry_stride` (default 10 or 50). Skip host norm collection entirely on non-stride batches.
2.  **Optimize Optimizer Checks:** Move the optimizer parameter-set validation out of the batch loop. Use a "dirty flag" on `SeedSlot` that triggers only when parameters actually change.
3.  **Deduplicate Action Parsing:** Consolidate `action_dict` indexing and assertions into a single optimized helper function.

### Phase 2: Architectural Improvements (Structural)
1.  **Asynchronous Telemetry:** Refactor `NissaHub` to use a `queue.Queue` and a background worker thread. Decouple `emit()` calls from training progress.
2.  **Consolidate Emitters:** Move large `if hub:` blocks into a `TelemetryEmitter` class in `simic/telemetry/emitters.py`. Reduce `vectorized.py` line count and loop complexity.
3.  **Compile Core Loops:** Wrap `process_train_batch` and `process_val_batch` in `torch.compile`. Ensure they are "graph-break clean" by removing telemetry string formatting from their internals.

### Phase 3: Reliability & Safety
1.  **Fix BUG-027:** Replace `list(permutations(...))` with a random sampling approach that respects `shapley_samples` without materialization.
2.  **Externalize Counterfactuals:** Move the complex logic for solo/pair/all-disabled configuration building into `CounterfactualHelper`. Clean up the `train_ppo_vectorized` validation loop to be a simple call to `helper.process_batch(...)`.
