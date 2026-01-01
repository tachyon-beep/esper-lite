# Plan: Verification of Governor Rollback

## Objective
Verify that `TolariaGovernor` correctly detects catastrophic loss spikes ("Panic") and executes a **state rollback** that restores the model to a safe previous checkpoint. Crucially, we must verify that the rollback restores *both* the parameters AND the optimizer state (or resets it safely), allowing training to resume without immediately crashing again.

## Context
The Governor is a safety system that monitors loss values.
*   **Panic:** If loss exceeds a threshold (e.g., 3x rolling average or absolute > 12.0), the Governor signals a panic.
*   **Rollback:** If panics persist (min 3 times), the Governor reverts the environment state to a snapshot taken *before* the instability began.

**Risks:**
*   **Restoration Failure:** Does `load_state_dict` work here? (We verified Topology persistence, but Governor uses in-memory snapshots).
*   **Optimizer Zombie:** Does the rollback clear the optimizer's momentum? If not, the optimizer will push the restored weights right back into the NaN zone (momentum carrying the "crash" velocity).
*   **Infinite Loop:** Does the rollback actually fix the problem, or does it just loop `Crash -> Rollback -> Crash`? (The Governor usually applies a "Death Penalty" reward to tell the agent "Don't do that again").

## Strategy
We will create a standalone integration test `tests/integration/test_governor_rollback.py` that simulates a training loop and injects a catastrophe.

### 1. Test Setup
*   **Environment:** `ParallelEnvState` with `MorphogeneticModel` and `TolariaGovernor`.
*   **Scenario:**
    1.  Train normally for 10 steps (establish baseline).
    2.  **Snapshot:** Governor takes a snapshot.
    3.  **Sabotage:** Manually inject `NaN` or massive values into the model weights or gradients.
    4.  **Trigger:** Feed this corrupted state to the Governor.
    5.  **Assert:** Governor triggers rollback.
    6.  **Verify:** Model weights match the snapshot, not the sabotaged state.

### 2. Execution Steps

#### Step 1: Initialization
*   Setup Env, Model, Optimizer, Governor.

#### Step 2: Normal Operation
*   Run 5 epochs of dummy training. Ensure Governor records history.

#### Step 3: Injection
*   `model.host.blocks[0].conv.weight.data.fill_(float('nan'))`
*   Run a step. Loss will be NaN.

#### Step 4: Governor Check
*   `governor.on_epoch_end(loss=nan)` -> Should return `Panic`.
*   Repeat until `rollback` is triggered.

#### Step 5: Verification
*   Check if model weights are finite again.
*   Check if optimizer state is cleared (per the code inspection findings).

## Implementation Plan

1.  **Scaffold Test:** Create `tests/integration/test_governor_rollback.py`.
2.  **Components:** Use `ParallelEnvState` to hold the stack.
3.  **Mocking:** We might need to mock the `history` or `snapshot` mechanisms if they depend on file I/O, but `TolariaGovernor` uses in-memory snapshots (deep copies).

## Success Criteria
*   Governor detects NaN.
*   Governor executes rollback.
*   Model weights are restored to pre-NaN values.
