# Fact Finding & Risk Reduction Audit Findings (Phase 1)
**Date:** January 1, 2026
**Author:** PyTorch 2.9 Specialist Agent
**Reference:** `docs/optimization/audit_plan.md`

## 1. Stream Leakage Audit Findings

### 1.1 `SeedSlot` Operations
*   **Method:** `_get_shape_probe`
*   **Code:** `torch.randn(..., device=self.device)`
*   **Finding:** Correctly uses `self.device`. However, it relies on the *current CUDA stream* at the time of execution.
*   **Method:** `forward` (Alpha Caching)
*   **Code:** `torch.tensor(self.alpha, device=host_features.device, ...)`
*   **Finding:** Correctly inherits device from input `host_features`.

### 1.2 `MorphogeneticModel` Lifecycle
*   **Method:** `germinate_seed` -> `SeedSlot.germinate`
*   **Operations Identified:**
    1.  Blueprint creation (`BlueprintRegistry.create`).
    2.  Parameter movement (`self.seed.to(self.device)`).
    3.  Shape probe creation (`torch.randn`).
    4.  Validation forward pass (`self.seed(shape_probe)`).
*   **Critical Finding:** In `vectorized.py`, the `germinate_seed` call happens in the sequential Python loop *after* the parallel stream contexts have exited (or before the next ones are entered).
    *   **Impact:** These operations run on the **Default Stream**.
    *   **Consequence:** Implicit synchronization. The default stream blocks until all other streams are idle (on older CUDA versions or strict serialization modes) or causes unintended serialization.
    *   **Verdict:** **Confirmed Stream Leakage.** The proposed fix (wrapping lifecycle ops in `with torch.cuda.stream(env_state.stream):`) is **Mandatory**.

## 2. Data Vectorization Readiness Findings

### 2.1 `SeedInfo` Structure
*   **Type:** `NamedTuple`.
*   **Nullable Fields:** `counterfactual_contribution`, `_prev_contribution`.
*   **Strategy:**
    *   Map `None` -> `torch.nan`.
    *   Use `torch.isnan()` checks in the kernel instead of `is None`.
*   **Enums:** `SeedStage` values match `src/esper/simic/rewards/rewards.py` constants.
*   **Verdict:** Ready for vectorization.

### 2.2 `previous_stage` Logic
*   **Finding:** `SeedInfo` captures `previous_stage` and `previous_epochs_in_stage` from `SeedState`.
*   **Validation:** The PBRS calculation in `_contribution_pbrs_bonus` relies on these.
    *   *Python:* `if seed_info.epochs_in_stage == 0:` branch handles the transition frame.
    *   *Vectorized:* This branching logic must be replicated using `torch.where`.
*   **Verdict:** Complex but vectorizable.

## 3. Telemetry Overhead Findings
*   **Data Volume:** ~4KB per batch (128 envs).
*   **Verdict:** Negligible. Pinned memory optimization is not required for this phase. Consolidated `stack()` + `cpu()` transfer is sufficient.

## 4. Reward Logic Purity Findings
*   **Purity:** `compute_reward` is functionally pure. It depends on `config` (immutable) and `STAGE_POTENTIALS` (constant).
*   **Side Effects:** None found. It does not mutate `SeedState`.
*   **Dependencies:**
    *   `_contribution_pbrs_bonus` (Pure)
    *   `_compute_synergy_bonus` (Pure)
    *   `INTERVENTION_COSTS` (Constant Dict) -> Needs conversion to Tensor Lookup.
*   **Verdict:** Safe to port to a standalone JIT kernel.

## 5. Verification Baseline
*   **Status:** `scripts/benchmark_hot_path.py` exists and measures EPS.
*   **Action:** Will run this script *before* applying patches to establish the baseline.
