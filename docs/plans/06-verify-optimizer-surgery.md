# Plan: Verification of Optimizer Surgery (Zombie State)

## Objective
Verify that the `Optimizer` correctly handles dynamic parameter birth and death. Specifically, ensure that:
1.  **Registration:** Newly germinated seeds are automatically added to the optimizer's parameter groups.
2.  **Cleanup:** Pruned seeds are removed from the optimizer's internal state (momentum buffers), preventing memory leaks ("Zombie State").

## Context
In standard deep learning, the parameter set is fixed. In Esper, it changes constantly.
PyTorch optimizers (`Adam`, `SGD`) maintain internal state dictionaries keyed by parameter pointers.
**Risk 1 (The Leak):** If we remove a parameter from the model (`slot.seed = None`) but don't tell the optimizer, the optimizer keeps the momentum buffer forever. Over thousands of epochs, this causes VRAM OOM.
**Risk 2 (The Blind Spot):** If we add a parameter (`slot.germinate()`) but don't add it to the optimizer, it will never be trained (gradients computed but never applied).

## Strategy
We will create a standalone integration test `tests/integration/test_optimizer_surgery.py`.

### 1. Test Setup
*   **Environment:** `MorphogeneticModel` (CNNHost).
*   **Optimizer:** `torch.optim.Adam` (since it has state).
*   **Manager:** We need to identify *who* is responsible for updating the optimizer. Is it `VectorizedHamletEnv`? `PPOAgent`? `Tolaria`?
    *   *Assumption:* The training loop (`vectorized.py` or `governor.py`) must have logic to re-construct or patch the optimizer. We will search for this logic first.

### 2. Execution Steps

#### Step 1: Baseline
1.  Initialize model and optimizer.
2.  Record `len(optimizer.param_groups[0]['params'])` and `len(optimizer.state)`.

#### Step 2: Germination (Birth)
1.  Germinate a seed.
2.  **Action:** Simulate the system's "Optimizer Update" hook (if it exists).
3.  **Verify:** Optimizer tracks new params. Run a step and check if seed weights change.

#### Step 3: Pruning (Death)
1.  Prune the seed.
2.  **Action:** Simulate the "Optimizer Update" hook.
3.  **Verify:** Optimizer state size shrinks back to baseline. `len(optimizer.state)` should decrease.

## Implementation Plan

1.  **Code Investigation:** Search `src/esper` for "optimizer" and "add_param_group" or similar logic to see how dynamic parameters are currently handled.
2.  **Scaffold Test:** Create `tests/integration/test_optimizer_surgery.py`.
3.  **The Fix (Likely Needed):** If no such logic exists (highly probable), we will write a helper function `prune_optimizer_state(optimizer, dead_params)` and `add_params_to_optimizer(...)` and test them.

## Success Criteria
*   New seeds get trained.
*   Pruned seeds release their optimizer memory.
