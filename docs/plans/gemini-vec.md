# Plan: Operation Gemini-Vec (Vectorized Refactor)

**Status:** Draft
**Owner:** Gemini
**Date:** 2026-01-06

## 1. Executive Summary

The `src/esper/simic/training/vectorized.py` module has become a 4,400-line monolith that violates the "Separation of Concerns" principle. It currently mixes:
1.  **Evolution (Simic):** PPO logic, reward calculation, policy updates.
2.  **Metabolism (Tolaria):** CUDA stream management, AMP contexts, batch dispatching, hardware execution.

This refactor will extract the hardware execution logic into a new `VectorizedEngine` within `src/esper/tolaria`, transforming `Simic` into a high-level client that requests operations rather than managing GPU streams directly.

## 2. Motivation & Goals

*   **Phase 3 Readiness (Transformers):** We need to swap the underlying execution engine (e.g., to support FSDP or different parallelism strategies) without rewriting the RL agent.
*   **Maintainability:** Decouple the complex async-GPU logic from the complex PPO math.
*   **Compliance:** Adhere to the "Inverted Control Flow" and "Train Anything Protocol" commandments by formalizing the execution interface.

## 3. Architecture Transition

### Current State (The Monolith)

`src/esper/simic/training/vectorized.py`:
- Initializes generic `ParallelEnvState`.
- Manages `SharedBatchIterator`.
- manually handles `torch.cuda.Stream`.
- Runs the `while global_step < total_steps` loop.
- Calls `agent.get_action()`.
- Computes rewards inline.
- Calls `agent.update()`.

### Target State (Decoupled)

**1. `src/esper/tolaria/engine.py` (The Engine)**
- Class `VectorizedEngine`:
    - Owns: `SharedBatchIterator`, `ParallelEnvState`, CUDA Streams, AMP GradScaler.
    - Methods: `train()`, `_step_env()`, `_run_rollout()`.
    - Takes a `PolicyClient` (Simic) as a dependency.

**2. `src/esper/simic/client.py` (The Brain)**
- Class `SimicClient` (wraps PPOAgent):
    - Implements the policy interface for the engine.
    - Handles `compute_reward`, `agent.update()`, and `agent.get_action()`.

**3. `src/esper/simic/training/vectorized.py` (The Entrypoint)**
- Becomes a lightweight script that:
    1.  Configures the `PPOAgent`.
    2.  Configures the `VectorizedEngine`.
    3.  Calls `engine.train(agent)`.

## 4. Implementation Plan

### Phase 1: Preparation & Utility Extraction
*Goal: Lighten the file by moving helpers to their domain homes.*

1.  **Move Timer:** Extract `CUDATimer` to `src/esper/tolaria/runtime.py`.
2.  **Move Lifecycle Logic:** Extract `_advance_active_seed` and `_resolve_target_slot` to `src/esper/kasmina/lifecycle.py` (or `utils` if generic).
3.  **Move Metric Utils:** Extract `_aggregate_ppo_metrics` to `src/esper/simic/telemetry/aggregation.py`.
4.  **Move PPO Runner:** Extract `_run_ppo_updates` to `src/esper/simic/training/runner.py`.

### Phase 2: The Engine Scaffold
*Goal: Define the interface without breaking the current loop.*

1.  Create `src/esper/tolaria/engine.py`.
2.  Define `VectorizedEngine` class with `__init__` taking configuration.
3.  Move `SharedBatchIterator` initialization and `ParallelEnvState` setup into `VectorizedEngine`.

### Phase 3: The Loop Extraction (Critical)
*Goal: Move the execution loop.*

1.  Move the `train_ppo_vectorized` main loop logic into `VectorizedEngine.run()`.
2.  Abstract the agent calls (`get_action`, `update`) behind a `PolicyProtocol`.
3.  Abstract the reward calculation behind a `RewardProtocol` (or keep it part of the Policy client for now).

### Phase 4: Reintegration
*Goal: Wire it back up.*

1.  Update `src/esper/simic/training/vectorized.py` to instantiate `VectorizedEngine` and pass the `PPOAgent`.
2.  Verify identical behavior on CIFAR-10.

## 5. Risk Management

*   **Performance Regression:** The "inline" nature of the current code was an optimization. We must ensure the new function calls/class overhead don't introduce blocking.
    *   *Mitigation:* Use `CUDATimer` to benchmark `step_latency` before and after.
*   **Complexity:** Over-abstraction can make it harder to read.
    *   *Mitigation:* Keep the interfaces simple. `Engine` calls `Agent.act(obs)`, `Engine` gives `Agent.store(reward)`.

## 6. Verification

1.  **Benchmark:** Run `scripts/benchmark_hot_path.py` (if exists) or a standard 100-episode run to establish baseline EPS (Episodes Per Second).
2.  **Regression Test:** Ensure bitwise reproducibility with a fixed seed is maintained (the logic shouldn't change, only the code location).
