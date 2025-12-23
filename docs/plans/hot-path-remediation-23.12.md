# Hot-Path Remediation — 2025-12-23

## Current Status
*   **Target Performance:** ~20 events per second (eps) with 3 active seeds.
*   **Observed Performance:** ~8 events per second (eps) with 1 active seed.
*   **Root Cause Analysis:** Initial diagnosis contained several inaccuracies. Updated analysis below.

---

## Findings Audit (2025-12-23)

### Finding 1: Unstrided Gradient Telemetry — ❌ ALREADY IMPLEMENTED

**Original claim:** Gradient telemetry runs every batch.

**Actual state:** Striding is already implemented at `vectorized.py:1644-1645`:
```python
collect_gradients = use_telemetry and (
    batch_step % gradient_telemetry_stride == 0
)
```
Default stride is 10 (configurable via `TrainingConfig.gradient_telemetry_stride`).

**Collection efficiency:** Uses `torch._foreach_norm` (single kernel launch) with async pattern that defers `.item()` calls until stream sync. See `gradient_collector.py:150-170`.

**Remaining overhead:** Line 284 `torch.cat([g.view(-1) for g in grads])` allocates O(total_params) for quality checks, but this is documented as "telemetry, not hot path" and runs only on stride batches.

---

### Finding 2: Synchronous Telemetry Hub — ❌ ALREADY ASYNC

**Original claim:** `NissaHub.emit` is synchronous and blocks training.

**Actual state:** `NissaHub` already uses `queue.Queue` + background worker thread. See `output.py:406-472`:
```python
class NissaHub:
    def __init__(self, max_queue_size: int = 1000):
        self._queue: queue.Queue[TelemetryEvent | None] = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: threading.Thread | None = None
```

The `emit()` method uses non-blocking `put_nowait()` at line 533. GIL contention is minimal because CUDA kernels release the GIL during execution.

---

### Finding 3: Interpreter Bloat — ⚠️ PLAUSIBLE BUT UNQUANTIFIED

**Original claim:** Readability refactors added significant overhead.

**Actual state:** `vectorized.py` is ~2900 lines. The claim about "hundreds of lines of boilerplate" is not substantiated with profiler data. Nested closures like `_parse_sampled_action` do cause repeated object allocation.

**Recommendation:** Profile before optimizing. Use `torch.profiler` to identify actual bottlenecks.

---

### Finding 4: Redundant Optimizer Validation — ❌ ALREADY OPTIMIZED

**Original claim:** Parameter-set validation runs every batch with `{id(p) for p in opt_params}`.

**Actual state:** This pattern was already removed. See `vectorized.py:1289-1302`:
```python
# OPTIMIZATION: Removed expensive parameter-set validation from hot path.
# Rely on env_state.seed_optimizers.pop() in the action execution block.
```

---

### Finding 5: Counterfactual Pipeline Complexity — ⚠️ VALID (Low Priority)

**Status:** Valid concern but runs during validation (less frequent than training). Impact is reduced.

---

### Finding 6: Shapley Permutation Explosion (BUG-027) — ❌ ALREADY FIXED

**Original claim:** Full permutation materialization causes OOM.

**Actual state:** Fixed at `counterfactual.py:392-415`:
```python
# Sample permutations (FIX BUG-027: Avoid materializing all permutations)
for _ in range(n_perms):
    perm = list(range(n))
    random.shuffle(perm)
```

---

## Actual Optimization Opportunities

### 1. BF16 Support for Ampere+ GPUs (HIGH IMPACT)

**Current state:** AMP uses FP16 with `GradScaler` (see `vectorized.py:1022`).

**Opportunity:** BF16 has wider dynamic range and doesn't need loss scaling, eliminating `GradScaler` overhead entirely.

```python
# Current (FP16)
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Proposed (BF16 when available)
if torch.cuda.is_bf16_supported():
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        ...
    loss.backward()  # No scaler needed
    optimizer.step()
```

**Implementation:**
- Add `amp_dtype` config option (`auto`, `float16`, `bfloat16`)
- Auto-detect Ampere+ GPUs (compute capability >= 8.0)
- Skip GradScaler when using BF16

---

### 2. torch.compile Mode Tuning (MEDIUM IMPACT)

**Current state:** Policy network uses `mode="default", dynamic=True` (see `ppo.py:340`).

**Opportunity:** `mode="max-autotune"` provides better fusion but longer compile time.

```python
# Current
self.network = torch.compile(self.network, mode="default", dynamic=True)

# Proposed (configurable)
self.network = torch.compile(
    self.network,
    mode=compile_mode,  # "default" or "max-autotune"
    dynamic=True
)
```

**Trade-offs:**
- `max-autotune`: 2-5x longer compile, potentially 10-20% faster runtime
- `default`: Fast compile, good balance for development
- Add `--compile-mode` CLI flag for experimentation

---

### 3. CUDA Graph Capture for Policy Network (LOW IMPACT, HIGH COMPLEXITY)

**Opportunity:** Policy network forward is called repeatedly with fixed-size inputs. CUDA graphs eliminate kernel launch overhead.

**Complexity:** Requires fixed batch sizes. Variable batch sizes (end of episode) need multiple graphs or eager fallback.

**Recommendation:** Defer until BF16 and compile tuning are validated.

---

### 4. Gradient Hook Accumulation (LOW PRIORITY)

**Current state:** Gradient collection iterates parameters via `list(model.parameters())`.

**Opportunity:** Register post-accumulate hooks to avoid iteration:
```python
class GradientStatsAccumulator:
    def __init__(self, model: nn.Module):
        self._squared_norm_sum = torch.tensor(0.0, device="cuda")
        for p in model.parameters():
            p.register_post_accumulate_grad_hook(self._accumulate)

    def _accumulate(self, grad: torch.Tensor) -> None:
        self._squared_norm_sum += grad.pow(2).sum()
```

**Complexity:** Requires PyTorch 2.1+ for `register_post_accumulate_grad_hook`. Current `_foreach_norm` approach is already efficient.

**Recommendation:** Only implement if profiling shows parameter iteration as a bottleneck.

---

## Implementation Plan

### Phase 1: BF16 Support (This Sprint)

1. Add `amp_dtype` to `TrainingConfig` with options: `auto`, `float16`, `bfloat16`
2. Add BF16 detection helper: `torch.cuda.is_bf16_supported()`
3. Update `vectorized.py` autocast to use configured dtype
4. Skip GradScaler when dtype is bfloat16
5. Update CLI with `--amp-dtype` flag
6. Add smoke test for BF16 path

### Phase 2: Compile Mode Option (Next Sprint)

1. Add `compile_mode` to PPOAgent config
2. Expose via CLI `--compile-mode {default,max-autotune}`
3. Document trade-offs in README

### Phase 3: Profiling & Validation

1. Run `torch.profiler` on baseline vs BF16 vs max-autotune
2. Measure actual eps improvement
3. Update this document with results

---

## Profiling Recommendations

Before further optimization, profile with:

```bash
PYTHONPATH=src python -c "
import torch.profiler
from esper.simic.training.vectorized import train_ppo_vectorized

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_output'),
    record_shapes=True,
    with_stack=True,
) as prof:
    train_ppo_vectorized(n_episodes=1, n_envs=1, max_epochs=8, device='cuda:0')
"
```

Key metrics:
1. **GPU utilization** — If <80%, CPU is bottleneck
2. **CUDA kernel launch overhead** — Count of small kernels
3. **Memory allocator time** — `cudaMalloc` calls
4. **Python overhead** — Time in `_PythonFrame`
