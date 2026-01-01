# Profiler Integration Code Review

**Date:** 2026-01-01
**Reviewer:** Claude (Batch 11 debugging session)
**Scope:** Evaluate `torch.profiler` integration for memory leaks and data errors
**Files Reviewed:**
- `src/esper/simic/telemetry/profiler.py`
- `src/esper/simic/training/vectorized.py` (profiler usage)

---

## Executive Summary

The profiler integration is **fundamentally sound** but had **one critical exception safety bug (B11-CR-05)** that has been fixed. No memory leaks or data errors detected beyond the exception safety issue.

**Status after fixes:**
- ✅ Exception safety: **FIXED** (B11-CR-05)
- ✅ Memory management: **CORRECT**
- ✅ Data correctness: **CORRECT**
- ✅ Resource cleanup: **CORRECT** (after B11-CR-05 fix)

---

## Findings

### ✅ FIXED: B11-CR-05 - Exception Safety Bug

**Issue:** Profiler context not guaranteed to close on exceptions

**Impact:**
- Resource leaks (CUDA event handlers, file handles, memory)
- Incomplete/corrupted TensorBoard traces
- Silent failures on training crashes

**Fix Applied:**
Wrapped training loop in try/finally block to guarantee profiler cleanup:

```python
prof = profiler_cm.__enter__()
try:
    # Training loop
    ...
finally:
    profiler_cm.__exit__(None, None, None)
```

**Verification:** ✅ Syntax check passes, imports work correctly

---

## Memory Leak Analysis

### 1. Profiler Context Manager (`training_profiler`)

**Implementation:**
```python
@contextmanager
def training_profiler(...) -> Iterator[torch.profiler.profile | None]:
    if not enabled:
        yield None  # No resources allocated
        return

    os.makedirs(output_dir, exist_ok=True)

    with torch.profiler.profile(...) as prof:
        yield prof  # PyTorch handles cleanup
```

**Analysis:**
- ✅ **No leaks when disabled:** Returns `None` immediately, zero overhead
- ✅ **Delegates to PyTorch's context manager:** PyTorch's `profile()` handles all resource cleanup
- ✅ **No manual resource management:** Relies on battle-tested PyTorch implementation
- ✅ **Directory creation is safe:** `os.makedirs(exist_ok=True)` is idempotent

**Conclusion:** NO MEMORY LEAKS. The wrapper delegates to PyTorch's profiler, which is well-tested.

---

### 2. Profiler Usage in Training Loop

**Implementation:**
```python
# Line 1715-1726: Setup
profiler_cm = training_profiler(...)
prof = profiler_cm.__enter__()
prof_steps = 0

try:
    # Training loop
    for epoch in range(max_epochs):
        # Training step
        if prof is not None:
            prof.step()  # Line 3387
            prof_steps += 1

finally:
    # Line 3642: Cleanup
    profiler_cm.__exit__(None, None, None)
```

**Analysis:**

#### ✅ Manual `__enter__()` and `__exit__()` Pattern
- **Before B11-CR-05:** Missing try/finally → resource leak on exceptions ❌
- **After B11-CR-05:** Wrapped in try/finally → cleanup guaranteed ✅

#### ✅ prof_steps Counter
- **Initialized at line 1764:** BEFORE try block (correct placement)
- **Incremented at line 3388:** Only when `prof is not None`
- **Used in finally block (line 3652):** To warn if too few steps ran
- **No leak potential:** Simple integer counter, no resource allocation

#### ✅ prof.step() Calls
- **Gated by null check:** `if prof is not None:` prevents crashes when profiling disabled
- **Called once per epoch:** Inside epoch loop (line 3387)
- **PyTorch handles memory:** `prof.step()` is a PyTorch API, memory managed by PyTorch

**Conclusion:** NO MEMORY LEAKS after B11-CR-05 fix. Profiler cleanup guaranteed by finally block.

---

### 3. Trace Handler and File Output

**Implementation:**
```python
# In training_profiler():
with torch.profiler.profile(
    ...
    on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ...
) as prof:
    yield prof
```

**Analysis:**
- ✅ **Trace handler is a callback:** Called by PyTorch when trace is ready
- ✅ **File writes managed by PyTorch:** TensorBoard trace handler flushes and closes files
- ✅ **No manual file handles:** All I/O delegated to PyTorch's implementation

**Potential Issue (NOT A LEAK, but worth noting):**
- **Partial traces on exceptions:** If training crashes mid-profiling cycle, the trace may be incomplete
- **After B11-CR-05 fix:** Profiler `__exit__()` called in finally block → PyTorch flushes partial trace
- **Behavior:** Partial trace is better than no trace (can debug up to the crash point)

**Conclusion:** NO FILE HANDLE LEAKS. PyTorch manages all file I/O.

---

## Data Correctness Analysis

### 1. Profiling Schedule

**Configuration:**
```python
schedule = torch.profiler.schedule(
    wait=torch_profiler_wait,      # Default: 1
    warmup=torch_profiler_warmup,  # Default: 1
    active=torch_profiler_active,  # Default: 3
    repeat=torch_profiler_repeat,  # Default: 1
)
```

**Data Flow:**
1. **Wait phase:** Profiler skips first N steps (let training stabilize)
2. **Warmup phase:** Profiler warms up (discard JIT compilation overhead)
3. **Active phase:** Profiler actively records traces
4. **Repeat:** Cycle repeats M times

**Analysis:**
- ✅ **Correct schedule semantics:** Wait → Warmup → Active → Repeat matches PyTorch docs
- ✅ **Configurable via CLI args:** User can tune for their workload
- ✅ **Default values are reasonable:** 1+1+3 steps for one cycle (5 steps minimum)

**Data Correctness:**
- ✅ **No off-by-one errors:** Schedule parameters passed directly to PyTorch
- ✅ **prof.step() placement:** Called once per epoch, matches expected granularity
- ✅ **Step counting:** `prof_steps` correctly tracks total steps for warning message

---

### 2. Profiler Summary Output

**Implementation:**
```python
# Line 3643-3647: Print profiler summary
if torch_profiler_summary and prof is not None:
    print("\n=== torch.profiler: CUDA time (top 30) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print("\n=== torch.profiler: CPU time (top 30) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
```

**Analysis:**
- ✅ **Null check:** `prof is not None` prevents crashes when profiling disabled
- ✅ **PyTorch aggregation:** `key_averages()` is PyTorch's built-in aggregation
- ✅ **Correct sort keys:** `cuda_time_total` and `cpu_time_total` are valid PyTorch metrics
- ✅ **Row limit:** `row_limit=30` prevents excessive output

**Data Correctness:**
- ✅ **Summary shows aggregated data:** Averages across all profiled steps
- ✅ **No data corruption:** PyTorch computes the statistics, not user code

---

### 3. Insufficient Steps Warning

**Implementation:**
```python
# Line 3648-3658: Warn if too few steps ran
if torch_profiler:
    min_steps_for_trace = (
        torch_profiler_wait + torch_profiler_warmup + torch_profiler_active
    )
    if prof_steps < min_steps_for_trace:
        print(
            f"\n[torch.profiler] No trace captured (ran {prof_steps} steps; "
            f"need >= {min_steps_for_trace} for wait={torch_profiler_wait} "
            f"warmup={torch_profiler_warmup} active={torch_profiler_active}). "
            "Run longer or reduce --torch-profiler-wait/--torch-profiler-warmup."
        )
```

**Analysis:**
- ✅ **Correct threshold calculation:** `wait + warmup + active` matches schedule minimum
- ✅ **Helpful error message:** Tells user exactly how many more steps needed
- ✅ **Actionable guidance:** Suggests two solutions (run longer OR reduce wait/warmup)

**Data Correctness:**
- ✅ **Accurate step count:** `prof_steps` incremented only when `prof.step()` called
- ✅ **No off-by-one:** Comparison is `<` (strict less-than), correct for minimum threshold

---

## Edge Cases and Failure Modes

### 1. Profiling Disabled (`enabled=False`)

**Code Path:**
```python
if not enabled:
    yield None
    return
```

**Analysis:**
- ✅ **Zero overhead:** Returns immediately, no resource allocation
- ✅ **Null checks downstream:** `if prof is not None:` guards all `prof.step()` calls
- ✅ **No crashes:** Profiler summary checks `prof is not None` before accessing

**Conclusion:** ✅ CORRECT

---

### 2. CUDA Not Available

**Code Path:**
```python
activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)
```

**Analysis:**
- ✅ **Graceful degradation:** Falls back to CPU-only profiling
- ✅ **No crashes:** PyTorch handles CPU-only profiling correctly

**Conclusion:** ✅ CORRECT

---

### 3. Training Stops Early (Shutdown or Exception)

**Before B11-CR-05:**
```python
prof = profiler_cm.__enter__()
# Training loop (no try/finally)
profiler_cm.__exit__(None, None, None)  # ❌ Not reached on exception
```

**After B11-CR-05:**
```python
prof = profiler_cm.__enter__()
try:
    # Training loop
finally:
    profiler_cm.__exit__(None, None, None)  # ✅ Always runs
```

**Analysis:**
- ✅ **Exception safety:** finally block guarantees cleanup
- ✅ **Partial traces written:** PyTorch flushes trace on `__exit__()`
- ✅ **No resource leaks:** CUDA events, file handles, memory all cleaned up

**Conclusion:** ✅ CORRECT (after B11-CR-05 fix)

---

### 4. Directory Permissions / Disk Full

**Code Path:**
```python
os.makedirs(output_dir, exist_ok=True)
```

**Analysis:**
- ⚠️ **No error handling:** Will raise `OSError` if permissions denied or disk full
- ⚠️ **Crashes before profiling starts:** Failure happens before `__enter__()`
- ✅ **No resource leak:** Crash happens before profiler allocated

**Recommendation:**
Consider wrapping in try/except with helpful error message:
```python
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as e:
    print(f"[torch.profiler] Cannot create output dir: {e}")
    yield None
    return
```

**Severity:** Low. User will see clear error from OS, can diagnose easily.

**Conclusion:** ⚠️ MINOR - Could improve error handling, but not a correctness bug

---

### 5. Profiler Step Called Outside Epoch Loop

**Current Implementation:**
```python
# prof.step() called exactly once per epoch (line 3387)
for epoch in range(max_epochs):
    # Training step
    if prof is not None:
        prof.step()
```

**Analysis:**
- ✅ **Correct granularity:** One step = one epoch of training
- ✅ **Matches schedule:** Schedule counts steps, each epoch is one step
- ✅ **No double-stepping:** Only called once per iteration

**Conclusion:** ✅ CORRECT

---

## Performance Impact

### 1. Overhead When Disabled

**Code Path:**
```python
if not enabled:
    yield None
    return
```

**Analysis:**
- ✅ **Zero overhead:** Single branch check, returns immediately
- ✅ **No allocations:** No profiler object created
- ✅ **Null checks are cheap:** `if prof is not None:` is a single pointer comparison

**Conclusion:** ✅ NEGLIGIBLE overhead when profiling disabled

---

### 2. Overhead When Enabled

**PyTorch Profiler Overhead:**
- **CPU profiling:** ~5-10% overhead (function call tracing)
- **CUDA profiling:** ~10-20% overhead (kernel launch tracing)
- **Memory profiling:** ~20-30% overhead (allocation tracking)
- **Stack recording:** ~30-50% overhead (Python stack unwinding)

**Default configuration:**
```python
record_shapes=False,       # Lower overhead
profile_memory=False,      # Lower overhead
with_stack=False,          # Lower overhead
```

**Analysis:**
- ✅ **Defaults minimize overhead:** All expensive options disabled
- ✅ **User can enable if needed:** CLI flags expose all options
- ✅ **Schedule limits active phase:** Only profiles 3 steps per cycle by default

**Conclusion:** ✅ REASONABLE overhead, configurable by user

---

## Security Analysis

### 1. Path Traversal

**Code:**
```python
output_dir: str = "./profiler_traces"
os.makedirs(output_dir, exist_ok=True)
```

**Analysis:**
- ⚠️ **No path sanitization:** User-provided path passed directly to `os.makedirs()`
- ⚠️ **Potential path traversal:** User could pass `../../etc` (but they control CLI args anyway)
- ✅ **Limited blast radius:** User already has filesystem access (running training locally)

**Conclusion:** ⚠️ LOW RISK - User controls CLI args, so path traversal is not a real threat

---

### 2. Trace File Contents

**Code:**
```python
on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
```

**Analysis:**
- ✅ **No sensitive data in traces:** Only kernel names, durations, tensor shapes
- ✅ **No user data leaked:** Training data not written to traces
- ✅ **TensorBoard format:** Well-defined JSON format, no injection risks

**Conclusion:** ✅ NO SECURITY ISSUES

---

## Recommendations

### Critical (Must Fix)
None. All critical issues fixed.

### High Priority (Should Fix)
None.

### Medium Priority (Consider Fixing)
1. **Improve error handling for directory creation:**
   ```python
   try:
       os.makedirs(output_dir, exist_ok=True)
   except OSError as e:
       print(f"[torch.profiler] Cannot create output dir: {e}")
       yield None
       return
   ```

### Low Priority (Nice to Have)
1. **Consider using `with` statement instead of manual `__enter__`/`__exit__`:**
   ```python
   # Current (manual):
   profiler_cm = training_profiler(...)
   prof = profiler_cm.__enter__()
   try:
       ...
   finally:
       profiler_cm.__exit__(None, None, None)

   # Alternative (cleaner):
   with training_profiler(...) as prof:
       ...  # All training loop code
   ```
   **Pros:** Automatic exception safety, more Pythonic
   **Cons:** Would require re-indenting entire training loop (3000+ lines)

---

## Conclusion

**Overall Assessment:** ✅ **CORRECT AFTER B11-CR-05 FIX**

**Memory Leaks:** None detected after B11-CR-05 fix
**Data Errors:** None detected
**Resource Leaks:** Fixed by B11-CR-05 (try/finally wrapper)

**Remaining Issues:**
- ⚠️ **Minor:** Directory creation error handling could be improved (low priority)

**Recommendation:** **APPROVE** with minor improvements suggested for future iteration.

---

## Test Coverage

Existing test coverage:
- ✅ Syntax check passes (`python -m py_compile`)
- ✅ Imports work correctly
- ✅ Integration tests pass (profiler not directly tested, but training loop tested)

**Missing test coverage:**
- ❌ No explicit test for profiler exception safety (B11-CR-05 scenario)
- ❌ No test for profiler disabled mode
- ❌ No test for insufficient steps warning

**Recommendation:** Add unit test for profiler context manager cleanup:
```python
def test_profiler_cleanup_on_exception():
    """Verify profiler context closes even on exceptions."""
    with pytest.raises(ValueError):
        with training_profiler(enabled=True) as prof:
            raise ValueError("Simulated training crash")
    # Test passes if no resource leak warnings
```

---

**Review completed:** 2026-01-01
**Reviewed by:** Claude (Batch 11 debugging session)
