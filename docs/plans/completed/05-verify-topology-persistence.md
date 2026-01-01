# Plan: Verification of Dynamic Topology Persistence

## Objective
Verify that the system can successfully save and resume training runs where the model architecture (topology) has diverged from the initial state due to seed growth. This ensures that long-running experiments (150+ epochs) can survive interruptions and checkpoints without losing the "grown" neural structure.

## Context
Standard PyTorch checkpoints save weights (`state_dict`). They do **not** save the model architecture.
In Esper, the architecture changes at runtime:
*   **Germination:** Adds new `SeedSlot` submodules.
*   **Pruning:** Removes submodules.
*   **Lifecycle State:** Seeds have internal state (`stage`, `alpha`, `age`) that must also persist.

**The Problem:**
If you instantiate a fresh `MorphogeneticModel` (standard host) and try to `load_state_dict()` from a checkpoint with 3 active seeds, it will crash because the fresh model doesn't have the `seed_slots.r0c1...` keys.

**The Solution:**
The system must likely save a "Topology Manifest" alongside the weights, or the loading logic must be robust enough to "re-germinate" slots before loading weights.

## Strategy
We will create a standalone integration test `tests/integration/test_topology_persistence.py` that simulates a "Training -> Crash -> Resume" cycle.

### 1. Test Setup
*   **Environment:** `MorphogeneticModel` (CNNHost).
*   **Phase 1 (Train):**
    *   Germinate a seed in `r0c1`.
    *   Train it until it reaches `BLENDING` stage (`alpha=0.5`).
    *   Germinate a second seed in `r0c2` (`TRAINING` stage).
    *   **Save Checkpoint** (simulating a scheduled save or crash recovery).
*   **Phase 2 (Resume):**
    *   Destroy the model instance.
    *   Create a **fresh** model instance (which starts with NO seeds).
    *   Attempt to load the checkpoint.

### 2. Verification Metrics
*   **Metric 1: Crash Resilience:** The loading process must not raise `RuntimeError: Unexpected key(s)...`.
*   **Metric 2: Topology Restoration:** The fresh model must effectively have seeds in `r0c1` and `r0c2` after loading.
*   **Metric 3: State Fidelity:**
    *   `r0c1` must be `BLENDING` with `alpha=0.5`.
    *   `r0c2` must be `TRAINING` with `alpha=0.0`.
    *   Weights must match the saved model bit-for-bit.

## Implementation Plan

1.  **Scaffold Test:** Create `tests/integration/test_topology_persistence.py`.
2.  **Instrumentation:**
    *   Use `torch.save` and `torch.load`.
    *   Mimic the `train_ppo_vectorized` checkpointing logic (if it exists).
3.  **The Fix (Anticipated):**
    *   If the test fails (as expected), we need to implement a **Topology Restorer**.
    *   This likely involves saving a metadata file (e.g., `checkpoint.meta.json`) or embedding the topology config into the `checkpoint.pt` dictionary.
    *   We will check `src/esper/simic/training/vectorized.py` to see how it currently handles saving.

## Success Criteria
*   The test successfully loads a checkpoint with active seeds into a fresh model.
*   The resumed model's output matches the original model's output.
