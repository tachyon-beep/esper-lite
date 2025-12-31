# Fact Finding & Risk Reduction Audit Plan (Phase 1)
**Date:** January 1, 2026
**Author:** PyTorch 2.9 Specialist Agent

This document outlines specific audits to validate assumptions and reduce risk before and during the Phase 1 optimization ("Just Coding").

## 1. Stream Leakage Audit
**Risk:** Operations within the hot path might inadvertently use the default CUDA stream, causing implicit synchronization and negating the benefits of per-environment streams.

*   **Audit 1.1: Tensor Creation in `SeedSlot`**
    *   **Goal:** Ensure all `torch.*` calls (e.g., `torch.zeros`, `torch.tensor`) inside `SeedSlot.germinate` and `forward` respect the device and current stream context.
    *   **Findings:** Initial grep shows `torch.tensor(...)` usage for cached alpha tensors.
    *   **Action:** Verify if `torch.tensor(..., device=self.device)` is used. If `device` is missing, it defaults to CPU (if data is CPU) or might cause implicit transfers. If `device` is CUDA, it uses the *current* stream. We must ensure `env_state.stream` is active when these methods are called.

*   **Audit 1.2: Host Model Operations**
    *   **Goal:** Check `MorphogeneticModel` for any operations that don't respect the active stream context.
    *   **Action:** Review `germinate_seed` and `prune_seed`. The proposed fix wraps these in `with torch.cuda.stream(...)`. Audit confirm that the internal logic of these methods doesn't explicitly synchronize (e.g., `.item()` or `print()` debugging).

## 2. Data Vectorization Readiness Audit
**Risk:** The `SeedInfo` object might have inconsistent states or `None` values that make vectorization difficult or error-prone (NaN propagation).

*   **Audit 2.1: `SeedInfo` Completeness**
    *   **Goal:** Verify if `SeedInfo` fields are ever `None` or invalid in a way that breaks tensor conversion.
    *   **Findings:** `SeedInfo` is a `NamedTuple`. `from_seed_state` handles `None` state by returning `None`.
    *   **Action:** Determine the "Null Object" pattern for tensors.
        *   *Plan:* Use `has_seed_mask` (boolean tensor).
        *   *Values:* When `has_seed_mask` is False, what should `seed_age` be? `0` or `-1`? Recommendation: `0` to avoid index errors, masked out by `has_seed_mask` in logic.

*   **Audit 2.2: `previous_stage` Logic**
    *   **Goal:** `_contribution_pbrs_bonus` relies on `previous_stage` and `previous_epochs_in_stage`.
    *   **Action:** Verify how these are tracked in `SeedState` and if they are reliable for the very first step after a transition. The Python code handles a specific edge case (`previous_epochs_in_stage=0`). The vectorized version must replicate this logic precisely.

## 3. Telemetry Overhead Audit
**Risk:** The proposed consolidated transfer might still be too slow if the data volume is massive, or if the `stack` operation on GPU is expensive.

*   **Audit 3.1: Data Volume Calculation**
    *   **Goal:** Estimate size of `actions_dict` and `log_probs`.
    *   **Calculation:** 8 heads * N envs * 4 bytes (float32). For N=128, this is tiny (~4KB).
    *   **Verdict:** `torch.stack` + `.cpu()` is likely sufficient. Pinned memory (`pin_memory=True`) is probably overkill for 4KB but harmless.

## 4. Reward Logic Purity Audit
**Risk:** `compute_reward` might have hidden side effects or rely on global state.

*   **Audit 4.1: Side Effects**
    *   **Goal:** Confirm `compute_reward` is a pure function.
    *   **Findings:** It reads `config`. It creates `RewardComponentsTelemetry` (new object). It calls `math.*`.
    *   **Action:** Confirm no modification of `env_state` or `seed_state` occurs inside `compute_reward`. (Code review suggests it's pure).

*   **Audit 4.2: Dependency Chain**
    *   **Goal:** Map all helper functions (`_contribution_pbrs_bonus`, etc.).
    *   **Action:** Ensure all helpers are also pure and vectorizable. `STAGES` and `INTERVENTION_COSTS` are module-level constants; they must be converted to tensors/lookups in Phase 2.

## 5. Verification Metrics Plan
**Risk:** We optimize but don't see improvements because we're measuring the wrong thing.

*   **Audit 5.1: Baseline Profiling**
    *   **Goal:** Establish a baseline EPS (Episodes Per Second) and "Step Latency" before applying Phase 1.
    *   **Action:** Run `scripts/benchmark_hot_path.py` *before* changes. Record the `process_train_batch` execution time specifically.
