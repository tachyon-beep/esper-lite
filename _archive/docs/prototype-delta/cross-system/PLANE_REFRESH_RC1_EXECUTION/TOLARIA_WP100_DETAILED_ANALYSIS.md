Sweet—as requested, I’m flipping into *tech‑review* mode. Below is **Phase 1: Immediate Root‑Cause Analysis** for the eager CUDA‑graph capture failure in Tolaria on PyTorch 2.8, written with a sceptical eye and enough detail to unblock the fix.

---

# Phase 1 Technical Diagnosis — Tolaria Eager CUDA‑Graph Capture (PyTorch 2.8)

## Executive summary (TL;DR)

Your capture is failing because **a DataLoader‑side H2D transfer (or pin‑memory handoff)** is still happening **while the graph stream is in capture mode**. With PyTorch/CUDA’s *global* capture error mode, **any work on the same device from other streams/threads** (e.g., a DataLoader prefetch stream) can trigger:

```
torch.AcceleratorError: operation not permitted when stream is capturing
```

Fix: **stop all DataLoader‑initiated GPU work during capture**, pre‑allocate **static device buffers**, copy CPU→GPU **outside** the captured region on a **staging stream**, then **make the graph stream wait** on a copy‑done event before `replay()`.

(Why this matches reality: CUDA graphs only allow very specific ops during capture; interloping kernels/memcpys, late allocations, or host‑syncs will trip the guardrails. PyTorch 2.8 docs and CUDA forum discussions line up with this behaviour. ([PyTorch Documentation][1]))

---

## What’s actually going wrong

### Symptom you see

* `torch.AcceleratorError: operation not permitted when stream is capturing`
* **Origin shows a DataLoader pin‑memory worker** in the stack.
* **Timing**: during the capture phase (not replay).

### Why it happens (mechanics)

1. **Graph capture begins** on a dedicated `graph_stream` (your `*attempt*graph_capture` around **lines 1010–1044**).
2. **Another stream** (likely a prefetch stream owned by your DataLoader/prefetcher thread) **issues a H2D copy or pinned‑memory handoff** during capture.

   * Typical culprits:

     * `tensor.to(device, non_blocking=True)` from the DataLoader’s prefetch thread.
     * `dst.copy_(src)` where `src` is **CPU pinned** and `dst` is **CUDA** (H2D memcpy launched on a non‑capturing stream).
     * Late **`pin_memory()`** action or `.record_stream()` bookkeeping that touches CUDA state while capture is active.
3. **Global capture error mode** treats unrelated work on other streams as a violation and raises precisely this error. (NVIDIA forum threads show the same failure if *any* other stream issues work during capture; likewise, PyTorch documents strict constraints on graph capture.) ([NVIDIA Developer Forums][2])

### Where in Tolaria to look (by your line map)

* **947–984 `*graph*forward_backward`**
  Expect to find either:

  * microbatch CPU→GPU copies **inside** the capture scope, or
  * the graph stream assumes inputs already on device while the DataLoader **also** prefetches to device concurrently. Either way, a H2D copy can run during capture.
* **1010–1044 `*attempt*graph_capture`**
  Capture orchestration likely:

  * Creates/uses `self._graph_stream` but **doesn’t freeze or drain** the DataLoader’s GPU prefetch stream before `torch.cuda.graph(...)`.
  * Doesn’t fully **front‑load allocations** (params, grads, optimizer state) into the graph pool prior to capture—leading to **alloc during capture** if any buffer is created lazily.
* **2135–2185 stream/buffer scaffolding**
  Expect `graph_stream` and one or more **non‑default staging/prefetch streams**. Missing pieces I’d expect:

  * `graph_stream.wait_stream(prefetch_stream)` **before capture begins**.
  * A **copy‑done `Event`** recorded on the staging stream and awaited by the graph stream **before each replay**.
* **3040–3150 buffer management** (and `*prepare*graph_buffers`, `*zero*grad`)
  Watch for:

  * Device tensor/optimizer state **allocation during capture** (disallowed).
  * `optimizer.zero_grad(set_to_none=True)` **inside capture** (can create fresh grad tensors on first use). Grads must be **materialised in warm‑up** and **zeroed in‑place** during capture.
    (PyTorch’s constraints: same addresses across replays; no allocations or host syncs in capture.) ([PyTorch Documentation][3])

---

## Evidence & external constraints to keep in mind

* **CUDA Graphs constraints** (PyTorch 2.8):

  * Capture only allows a subset of CUDA ops; **host syncs** (e.g., `.item()`), **late allocations**, and **changing memory addresses** are forbidden. ([PyTorch Documentation][1])
* **Global capture invalidates** when other streams do unrelated work while a stream is capturing. Pattern reproduced in CUDA forum threads with the same error text. ([NVIDIA Developer Forums][2])
* **DataLoader pin‑memory**: it pins CPU tensors and (if you use `non_blocking=True`) enables **async** H2D memcpys that typically run on the **current stream** (or a specified stream via context). If that stream is not the capturing graph stream, you can still trip the guard. (PyTorch tutorial & Q/A references.) ([PyTorch Documentation][4])
* **Guardrails you can use**:

  * `torch.cuda.is_current_stream_capturing()` is available to instrument capture state (handy for asserts/telemetry). ([PyTorch Documentation][5])

---

## Minimal failing pattern (conceptual MWE)

```python
gstream = torch.cuda.Stream()
prefetch = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()

static = torch.empty_like(batch_cpu[0], device='cuda')  # device buffer

# Bad: prefetch still runs H2D on 'prefetch' while capture on 'gstream'
with torch.cuda.stream(prefetch):
    static.copy_(batch_cpu[0], non_blocking=True)  # H2D memcpy (not captured)

torch.cuda.synchronize()  # harmless here

with torch.cuda.graph(g, stream=gstream):
    # ... forward/backward reading from 'static' ...
    pass
# If the prefetch thread races another H2D here -> AcceleratorError
```

The fix is to **ensure no H2D occurs during capture**, and to **sequence pre‑replay copies via events** (see next section). (The “don’t do operations on other streams while capturing” behaviour and constraints are documented & discussed.) ([PyTorch Documentation][6])

---

## Recommended architecture change (safe graph pattern)

> **Goal:** Keep **all** CPU→GPU work **outside** capture; run **only device‑resident compute** inside capture.

### 1) Disable GPU prefetch when graphs are enabled

* In your custom DataLoader/prefetcher, add a toggle (e.g., `prefetch_to_device=False`) when `--enable-graphs` is set.
* Continue to **pin** CPU batches for `non_blocking` H2D, but **do not** schedule any `.to('cuda')` from the DataLoader thread while capture is active.

(If you keep DataLoader→GPU prefetch, you’ll race the graph capture and retrigger the error. Forum reports show *any* work on a different stream during capture can invalidate it.) ([NVIDIA Developer Forums][2])

### 2) Pre‑allocate **static** device buffers (outside capture)

* Build `static_inputs` (and labels) once on device with the **final shapes** you’ll use during training (microbatch granularity).
* Do a **warm‑up** forward/backward on the **graph stream** with a representative batch to **materialise** parameter grads and optimizer buffers, using `optimizer.zero_grad(set_to_none=False)` thereafter so **zeroing is in‑place** during capture (no new allocations).

(Stable addresses & no allocations during capture are required by CUDA graphs.) ([PyTorch Documentation][3])

### 3) Use a **staging stream** for per‑step H2D, and a **copy‑done event**

* Create `staging_stream = torch.cuda.Stream()` and an event `copy_done = torch.cuda.Event()`.
* **Per step (outside capture)** on `staging_stream`, copy the next CPU batch into `static_inputs` with `non_blocking=True`, then `copy_done.record(staging_stream)`.
* Make the `graph_stream` **wait** on that event **before** `g.replay()`:

```python
# outside capture:
with torch.cuda.stream(staging_stream):
    for d, s in zip(static_inputs, batch_cpu):
        d.copy_(s, non_blocking=True)
    copy_done.record(staging_stream)

# order the streams:
graph_stream.wait_event(copy_done)

# captured work only reads device buffers:
g.replay()
```

(Using events for inter‑stream dependency is idiomatic; `Event.wait()`/`Stream.wait_event()` are the right tools. ([PyTorch Documentation][7]))

### 4) Capture only **forward+backward** (optionally step)

* Safer initial scope: capture `forward` and `loss.backward()`; keep `optimizer.step()` and any host‑side scaler updates **outside** capture first.

  * If you later want to capture `step()`, ensure all optimiser state is fully allocated in warm‑up (no lazy alloc), and no host synchronisation occurs.
  * Mixed precision: if using `GradScaler`, prefer updating the scale **outside** capture (consistent with PyTorch guidance for graph stability).

### 5) Allocator & pool

* Use a dedicated graph pool: `pool = torch.cuda.graph_pool_handle()` and pass it to `torch.cuda.graph(...)` as needed, or capture on `CUDAGraph` and query `g.pool()` for reuse. This helps prevent allocator calls surfacing during capture. ([PyTorch Documentation][8])

---

## Concrete code changes (surgical)

> **File:** `src/esper/tolaria/trainer.py`

### A) `*attempt*graph_capture` (≈ 1010–1044)

* **Before capture**:

  * Ensure **no** DataLoader GPU prefetch: `self._dl.prefetch_to_device = False`.
  * Create `self._graph_stream`, `self._staging_stream`, and `self._copy_done_event`.
  * **Warm‑up on `graph_stream`** to materialise grads/optim buffers:

    * `optimizer.zero_grad(set_to_none=False)` (important for in‑place zero in capture).
    * A single fwd+bwd pass using **static device buffers**.
  * **Quiesce** any outstanding prefetch:
    `torch.cuda.synchronize()` or `graph_stream.wait_stream(prefetch_stream)` if you had one (outside capture).
* **Capture**:

  ```python
  self._graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(self._graph, stream=self._graph_stream):
      out = model(self._static_inputs)
      loss = criterion(out, self._static_targets)
      loss.backward()
  ```

  *(No H2D here; only device‑resident ops.)*

### B) `*graph*forward_backward` (≈ 947–984)

* **Do not** call `.to('cuda')` or `.copy_()` from CPU **inside** the capture.
* **Per step** (outside capture):

  ```python
  with torch.cuda.stream(self._staging_stream):
      _copy_cpu_to(self._static_inputs, batch_cpu)      # non_blocking=True
      _copy_cpu_to(self._static_targets, labels_cpu)    # non_blocking=True
      self._copy_done_event.record(self._staging_stream)

  self._graph_stream.wait_event(self._copy_done_event)
  self._graph.replay()
  ```

* If you maintain microbatches, repeat the copy/event per microbatch **outside** capture, and replay once per microbatch (or capture a microbatch body and loop at Python level).

### C) Stream/buffer scaffolding (≈ 2135–2185, 3040–3150 & around `*prepare*graph_buffers`, `*zero*grad`)

* `*prepare*graph_buffers` must **allocate all device buffers** used in the captured region **before** capture begins.
* Ensure `zero_grad` path uses `set_to_none=False` **for the captured pass**, so grads are **zeroed in‑place** during capture (no fresh allocations).
* Tie pinned CPU buffers’ lifetimes to the **staging stream** via `tensor.record_stream(self._staging_stream)` if you refcount through CUDA stream ordering; do **not** record on the graph stream.

### D) Quick instrumentation (guards/telemetry)

* **Assert capture state** at key points:

  ```python
  assert not torch.cuda.is_current_stream_capturing(), "No capture before H2D copies"
  ```

* Emit telemetry when toggling:

  * `tolaria.train.graph_enabled=1` once capture succeeds.
  * `tolaria.graph_fallback.reason="prefetch_stream_activity_during_capture"` if you detect a prefetch stream still alive.

---

## How to prove it locally (fast checks)

1. **Log capture state**: emit `is_current_stream_capturing()` in the DataLoader prefetch thread and in `*attempt*graph_capture` at entry/exit. You should see `False` in the prefetch path even while the error happens (capture is on another stream)—which is the point: *another* stream is racing you. ([PyTorch Documentation][5])
2. **Disable GPU prefetch** with a flag and rerun capture. If the error disappears, you’ve confirmed the interference.
3. **Scope capture down** to only a tiny fwd+bwd that reads pre‑filled static buffers; if this works, gradually re‑introduce the surrounding mechanics until the interference point appears.

---

## Side‑effects & guardrails (so we don’t regress `torch.compile`)

* **Gate everything** behind `--enable-graphs` (your flag) and don’t touch the Inductor/`torch.compile` path. Inductor uses CUDA graphs internally by default on GPU; we’re not changing that. ([PyTorch Documentation][9])
* **Shapes must be static** across capture/replay (batch/microbatch mustn’t change shape). If your DataLoader sometimes emits variable shapes (e.g., last batch), push those cases to an **eager fallback step** or pad. (Graph limitation.) ([PyTorch Documentation][3])
* **AMP/GradScaler**: keep `scaler.update()` **outside** capture initially to avoid control‑flow surprises. Once stable, you can capture it if all states are materialised and deterministic.

---

## Suggested patch outline (pseudo‑diff)

```diff
@@ def _attempt_graph_capture(...):   # ~1010–1044
-  # capture orchestrates work, but DL prefetch keeps copying to GPU
+  # Disable GPU prefetch while graphs are active
+  self._dl.set_prefetch_to_device(False)

+  self._graph_stream  = torch.cuda.Stream()
+  self._staging_stream = torch.cuda.Stream()
+  self._copy_done = torch.cuda.Event()
+  self._graph = torch.cuda.CUDAGraph()

+  # Allocate static device buffers for inputs/targets/outputs
+  self._static = prepare_graph_buffers(example_batch, device='cuda')

+  # Warm-up on graph stream: materialise params/grads/optim state
+  with torch.cuda.stream(self._graph_stream):
+      optimizer.zero_grad(set_to_none=False)
+      out = model(self._static.inputs)
+      loss = criterion(out, self._static.targets)
+      loss.backward()
+  torch.cuda.synchronize()

+  # Capture only device-resident compute
+  with torch.cuda.graph(self._graph, stream=self._graph_stream):
+      out = model(self._static.inputs)
+      loss = criterion(out, self._static.targets)
+      loss.backward()

@@ def _graph_forward_backward(...):   # ~947–984
-  # Inside this function we copy CPU->GPU then run compute (during capture)
+  # 1) Stage H2D on staging stream, OUTSIDE capture
+  with torch.cuda.stream(self._staging_stream):
+      copy_cpu_to_device(self._static.inputs, batch.inputs, non_blocking=True)
+      copy_cpu_to_device(self._static.targets, batch.targets, non_blocking=True)
+      self._copy_done.record(self._staging_stream)
+  # 2) Order streams and replay captured work
+  self._graph_stream.wait_event(self._copy_done)
+  self._graph.replay()
```

*(Exact names may differ in your codebase, but the sequencing is the important part.)*

---

## Risks & trade‑offs

* **Memory**: keeping static device buffers increases persistent VRAM use; for large batches, consider **microbatching** (still static shapes).
* **Prefetch overlap**: with the event‑chained staging stream, you still get H2D/compute overlap **outside** capture—so the performance model remains healthy.
* **Complexity**: the DataLoader can stay standard; you only need to **turn off GPU prefetch** when graphs are enabled. No full rewrite required.

---

## What to document (for Phase 3 later)

* Manual graphs require **fixed shapes** and **no async DL→GPU** activities during capture.
* **How to enable**: `--no-compile --enable-graphs` (and what features are incompatible).
* **Telemetry fields**: `graph_enabled`, `graph_capture_ms`, `graph_replay_ms`, `graph_fallback.reason`, `prefetch_to_device_disabled`.

---

## Confidence (WEP)

* **Root cause (“DL prefetch stream is issuing H2D during capture”)**: **Highly likely (≈ 85–90%)** given the error text, your provenance (“pin‑memory worker thread”), and common failure patterns under global capture. ([NVIDIA Developer Forums][2])
* **Proposed fix (static buffers + staging stream + event + no DL→GPU during capture)**: **Very likely (≈ 80–90%)** to eliminate the exception and allow successful replay, as it aligns with PyTorch 2.8 graph constraints and standard recipes. ([PyTorch Documentation][1])
* **No regression to `torch.compile` path** (gated changes): **Almost certain (≈ 95%)**.

---

## Benchmark Artifacts (2025-10-02)

- `baselines/perf/wp100_graph_bench/graph_bench.json`: standard benchmark run (`PYTHONPATH=. python scripts/run_graph_bench.py --epochs 3 --warmup-batches 2 --device cuda`). Shows `graph_enabled = 1` with a ~63 ms capture for the first epoch and ~5 s capture for subsequent epochs as the graph instantiates from a cold pool.
- `baselines/perf/wp100_graph_bench/graph_bench_cuda_dsa.json`: debug run captured with `TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1` to surface kernel attribution. DSA overhead inflates capture timings to ~74 ms / ~5.1 s but preserves `graph_enabled = 1`. Use this file when auditing regressions or comparing kernel attribution traces.

### References

* PyTorch 2.8 **`torch.cuda.CUDAGraph`** and **`torch.cuda.graph`** docs (constraints, memory‑pool, instantiate/replay). ([PyTorch Documentation][1])
* **CUDA graphs limitations** (same addresses, no host sync/allocs). ([PyTorch Documentation][3])
* **Global capture interference** with other streams → same error text. ([NVIDIA Developer Forums][2])
* **Pinned memory & non_blocking** behaviour with DataLoader. ([PyTorch Documentation][4])
* **`is_current_stream_capturing()`** for instrumentation. ([PyTorch Documentation][5])
* `torch.compile` notes (CUDA graphs enabled by default in some modes). ([PyTorch Documentation][9])

---

If you want the patch implemented next, the changes are confined to the four regions you listed; the diff above is the exact sequencing I’d apply to Tolaria’s `*attempt*graph_capture` and `*graph*forward_backward` to make `scripts/capture_perf_phase0_baselines.py --no-compile --enable-graphs` complete with `tolaria.train.graph_enabled=1`.

[1]: https://docs.pytorch.org/docs/2.8/generated/torch.cuda.CUDAGraph.html?utm_source=chatgpt.com "CUDAGraph — PyTorch 2.8 documentation"
[2]: https://forums.developer.nvidia.com/t/cuda-graph-capture-work-on-separated-streams-invalidates-graph-capture/325426?utm_source=chatgpt.com "CUDA Graph capture - work on separated streams ..."
[3]: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html?utm_source=chatgpt.com "CUDAGraph Trees"
[4]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html?utm_source=chatgpt.com "A guide on good usage of non_blocking and pin_memory() ..."
[5]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html?utm_source=chatgpt.com "torch.cuda.is_current_stream_capturing"
[6]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html?utm_source=chatgpt.com "graph — PyTorch 2.8 documentation"
[7]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html?utm_source=chatgpt.com "Event - torch.cuda"
[8]: https://docs.pytorch.org/docs/2.8/generated/torch.cuda.graph_pool_handle.html?utm_source=chatgpt.com "torch.cuda.graph_pool_handle"
[9]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html?utm_source=chatgpt.com "torch.compile — PyTorch 2.8 documentation"
---

## Phase 5 Observability & Reporting (2025-10-03)

- Rollout guidance: enable graphs in sandbox, verify `tolaria.graph.capture_ms < 200` and no `tolaria.graph_fallback`; promote to staging with alerts active, then production. Keep pool reuse flag `enable_graph_pool_reuse` (default) and use `warmup_graph_pool()` during service start. If alerts trigger, disable graphs and revert to compile.
- Implemented shared graph-pool reuse (`TrainingLoopConfig.enable_graph_pool_reuse`); per-trainer captures now run in ~63 ms instead of 5 s. Handled fallback by dropping the cached handle.
- **Prometheus alerts**: `infra/prometheus/alert_rules.yml` now tracks stage copy (warning >1 ms), capture (critical >8 s), replay (warning >5 ms), and fallback events (any increase over 5 min). `infra/prometheus/prometheus.yml` loads the rule file by default.
- **Grafana panel**: Nissa overview dashboard (`infra/grafana/dashboards/nissa_overview.json`, panel ID 9) plots `avg_over_time(tolaria_graph_stage_copy_ms[1m])` with thresholds to mirror the alert. Capture/replay panels remain in place for correlation.
- **Runbook linkage**: `docs/project/observability_runbook.md` now includes alert thresholds, bench commands (`scripts/run_graph_bench.py --epochs 5 --warmup-batches 2 --device cuda`), and diagnostic guidance (`CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`).
- **Telemetry snapshots**: Phase 5 pre-rollout measurements stored under `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp100_phase5_prework/` (standard + DSA runs).
- **Next steps**: optimise the 5 s pool-instantiation cost, then revisit alert thresholds and document sandbox→staging→prod rollout in the runbook.

