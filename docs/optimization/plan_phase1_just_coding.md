# Plan: Phase 1 — “Just Coding” Hot-Path Optimizations
**Date:** January 1, 2026  
**Author:** PyTorch Specialist (pragmatic pass)  
**Scope:** `src/esper/simic/training/vectorized.py` (vectorized PPO step loop)

This plan intentionally avoids architectural rewrites. It targets improvements that are:
- measurable (EPS/step-time),
- low risk to correctness,
- easy to review,
- reversible without “dual path” or compatibility code.

## 0) First Principles (Why This Phase Exists)

Vectorized training has two hard constraints:
1) Any policy decision that depends on state must occur after the state is produced.
2) Python per-env orchestration is a serial fraction; if it dominates step time, GPU utilization drops.

Phase 1 focuses on shrinking the serial Python fraction without changing the execution model.

## 1) Non-Goals (Explicitly Out of Scope)

Not Phase 1:
- A fully tensor-driven lifecycle state machine (GPU-resident SoA state, compiled control flow).
- Static-topology “sleeper seed” refactor (prealloc pools, masked execution) beyond small prototypes.
- Replacing per-env models with a single batched model (requires design-level changes in Kasmina/Tolaria).

## 2) Measurement & Baselines (Required Before/After)

Use existing scripts so comparisons are stable:
- EPS baseline: `PYTHONPATH=src uv run python scripts/benchmark_hot_path.py`
- Focused micro-profiles: `PYTHONPATH=src uv run python scripts/profile_hot_path_operations.py`
- Optional sync hunting: `PYTHONPATH=src uv run python scripts/profile_gpu_sync.py`

Record:
- GPU model + driver/CUDA version,
- `n_envs`, `max_epochs`, AMP/compile settings,
- EPS and 95p step time (if available).

## 3) Phase 1 Tasks (Ranked by ROI)

### Task A — Reduce action/value CPU materialization overhead

**Problem:** We currently do per-head `.cpu().numpy()` plus build a Python list-of-dicts for actions.

**Principle:** The goal is not “one transfer”; the goal is less Python work and fewer forced sync points.

**Candidate changes:**
- Keep action heads as NumPy arrays and pass raw ints into `_parse_sampled_action()` without building per-env dicts.
- Only construct `action_dict` when needed for buffer storage/telemetry, and keep it minimal.
- Benchmark “stacked transfer” (`torch.stack([...]).cpu().numpy()`) vs per-head transfers; pick one based on measured EPS (no runtime feature flags).

**Acceptance:** EPS improves on `scripts/benchmark_hot_path.py` with no behavior changes.

### Task B — Remove avoidable per-env Python allocations in the step loop

Hot loop allocations (examples):
- per-env dicts for transitions,
- per-env dicts for telemetry (mask flags, slot state maps),
- repeated `dict.get/set` on counters.

**Candidate changes:**
- Replace “list of per-env dicts” with a struct-of-arrays pattern where feasible (pure Python lists of primitives, then convert once).
- Move transition assembly closer to the buffer API (prefer “write tensors directly into buffer slots” over “build dict then unpack”).

**Acceptance:** Reduced Python time in profiler; EPS stable-or-up.

### Task C — Verify lifecycle ops do not introduce accidental GPU sync

Even if we still synchronize at step boundaries, lifecycle operations should not accidentally:
- allocate on the default stream in a way that forces extra waits later,
- copy tensors CPU↔GPU implicitly,
- invalidate compiled regions unnecessarily.

**Candidate changes:**
- Audit lifecycle ops called in the per-env loop for unexpected tensor ops.
- Where lifecycle ops touch CUDA tensors, ensure they execute on the intended stream/context (or document why they must not).

**Acceptance:** No new synchronizations observed in `profile_gpu_sync.py` (or equivalent).

### Task D — Telemetry stays off the critical path

Rules for Phase 1:
- No `.cpu()` inside per-env loops for telemetry.
- Prefer batched CPU transfers once per step if CPU views are required.

**Acceptance:** Telemetry-on vs telemetry-off delta stays bounded (define target after baseline).

## 4) Exit Criteria

Phase 1 is “done” when:
- We can point to a concrete EPS improvement on the benchmark script, and
- the changes are local (no cross-domain redesign), and
- tests/lints still pass.

## 5) Follow-On (Phase 2+ Candidates)

If Phase 1 still leaves a CPU trench, the next credible leap is:
- tensor-driven lifecycle/reward computation, *or*
- a constrained static-topology prototype (sleeper seeds) to validate compile/capture benefits.
