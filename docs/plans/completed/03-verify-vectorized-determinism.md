# Plan: Verification of Vectorized Determinism

## Objective
Verify that the **Vectorized Training Environment** is strictly deterministic when seeded. This is critical for Reinforcement Learning (RL) debugging, as non-reproducible runs make it impossible to isolate the cause of policy regressions or "butterfly effect" divergences.

## Context
The training loop (`vectorized.py`) involves complex interactions:
*   **Inverted Control Flow:** Iterating over batches first, then splitting across environments.
*   **Parallel Environments:** Multiple `VectorizedHamletEnv` instances updating state independently.
*   **Shared Resources:** `SharedBatchIterator` with multi-worker prefetching.
*   **Randomness:** Action masking (dropouts), policy sampling, and environment transitions (if stochastic).

**Risks:**
*   **Race Conditions:** If random number generators (RNGs) are shared or not properly fork-safe, environment threads might consume entropy in non-deterministic orders.
*   **GPU Non-Determinism:** Certain CUDA operations (like atomic adds in `scatter_add`) are non-deterministic unless explicitly configured.
*   **DataLoader Jitter:** If the shared iterator doesn't guarantee sample order across worker processes, environments will see different data sequences.

## Strategy
We will create a standalone reproduction script `tests/integration/test_vectorized_determinism.py` that runs the full vectorized environment loop twice with the same seed and asserts exact state matching.

### 1. Test Setup
*   **Config:** `TrainingConfig` with `n_envs=2`, `max_epochs=10` (short horizon).
*   **Seed:** Fixed seed `42`.
*   **Components:** Full `train_ppo_vectorized` stack (or a slightly mocked version if full training is too slow).

### 2. Execution Steps

#### Run 1 (Reference)
1.  Set global seeds (`torch`, `numpy`, `random`).
2.  Initialize `train_ppo_vectorized`.
3.  Run for `N` steps.
4.  Record a **Trace** of events:
    *   Observation digests (hash of obs tensor).
    *   Action sequences (exact actions selected).
    *   Reward sequences.
    *   Lifecycle events (germination/prune timings).

#### Run 2 (Replay)
1.  **Reset** everything (fresh process or rigorous state clear).
2.  Set **same** global seeds.
3.  Run for `N` steps.
4.  Record Trace 2.

### 3. Verification Metrics
*   **Metric 1: Action Identity:** `Trace1.actions == Trace2.actions`.
    *   *Failure:* Policy divergence. RNG state drifted.
*   **Metric 2: Observation Identity:** `Trace1.obs_hashes == Trace2.obs_hashes`.
    *   *Failure:* Environment divergence. DataLoader or state update drift.
*   **Metric 3: Reward Identity:** `Trace1.rewards == Trace2.rewards`.

## Implementation Plan

1.  **Scaffold Test:** Create `tests/integration/test_vectorized_determinism.py`.
2.  **Instrumentation:** Use a `DeterministicAgent` (fixed weights or mock policy) to remove policy training variance from the equation first. Then test with a learning agent if stable.
    *   *Decision:* Testing with a *frozen* random policy is best for environment determinism. Testing with *training* enabled is a higher bar (requires deterministic GPU kernels). We will start with **Frozen Policy Determinism** (Environment + Data + RNG) as that is the prerequisite.
3.  **Trace Recorder:** A simple list of dicts.

## Success Criteria
*   Two independent runs produce identical traces.
*   Confirmation that `SharedBatchIterator` delivers identical data batches in the same order.
