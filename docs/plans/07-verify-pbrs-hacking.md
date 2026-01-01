# Plan: Verification of PBRS Reward Hacking

## Objective
Verify that the Potential-Based Reward Shaping (PBRS) mechanism is robust against "Reward Hacking." Specifically, ensure that a cycle of operations that returns the system to its initial state (e.g., Germinate -> Prune) results in a net shaping reward of approximately zero.

## Context
Esper uses PBRS (Ng et al., 1999) to guide the RL agent through the seed lifecycle stages (`DORMANT` -> `GERMINATED` -> `TRAINING` -> ...).
The shaping reward is defined as $F(s, s') = \gamma \Phi(s') - \Phi(s)$.
**The Risk:** If the potential function $\Phi(s)$ is inconsistent or if the reward implementation adds raw bonuses instead of differences, the agent can exploit cycles to accumulate infinite reward without improving the model.
*   **Infinite Money Glitch:** If $Reward(Germinate) > \gamma^{-1} \times |Reward(Prune)|$, the agent will just spam Germinate/Prune.

## Strategy
We will create a standalone integration test `tests/integration/test_pbrs_hacking.py` that calculates the cumulative reward for a specific cycle.

### 1. Test Setup
*   **Environment:** `VectorizedHamletEnv` (or similar high-level env).
*   **Cycle:**
    1.  `DORMANT` -> `GERMINATED` (Germinate Action)
    2.  `GERMINATED` -> `PRUNED` (Prune Action)
    3.  `PRUNED` -> `EMBARGOED` -> `RESETTING` -> `DORMANT` (Automatic decay)

### 2. Execution Steps
We will use `compute_pbrs_bonus` directly (unit testing the math) AND/OR simulate the environment step-by-step. Unit testing the math is cleaner and isolates the logic.

#### Step 1: Define Potentials
*   Identify $\Phi(DORMANT)$, $\Phi(GERMINATED)$, $\Phi(PRUNED)$, etc.

#### Step 2: Calculate Transitions
*   $R_1 = \gamma \Phi(GERM) - \Phi(DORM)$
*   $R_2 = \gamma \Phi(PRUNED) - \Phi(GERM)$
*   ... and so on until back to `DORMANT`.

#### Step 3: Sum
*   $Sum = R_1 + \gamma R_2 + \gamma^2 R_3 + \dots$
*   The sum should effectively cancel out (telescoping sum), leaving only the potential difference between start and end (which is 0 since start==end) minus energy lost to discounting.
*   Actually, for a cycle $s_0 \to s_1 \to \dots \to s_k \to s_0$:
    *   $\sum \gamma^t F(s_t, s_{t+1}) = \Phi(s_k) \gamma^k - \Phi(s_0) + \dots$
    *   Wait, the theoretical guarantee is that the *optimal policy* is unchanged.
    *   But for "Hacking", we want to ensure the agent doesn't get *positive* reward for a useless cycle.
    *   Ideally, the net reward for a cycle should be $\le 0$.

## Implementation Plan

1.  **Code Investigation:** Locate `compute_pbrs_bonus` and potential definitions in `src/esper/simic/rewards`.
2.  **Scaffold Test:** Create `tests/integration/test_pbrs_hacking.py`.
3.  **Test Logic:**
    *   Instantiate `SeedState`.
    *   Manually transition it through the lifecycle.
    *   Compute rewards at each step.
    *   Assert `Sum <= 1e-5`.

## Success Criteria
*   Net shaping reward for a Germinate-Prune-Reset cycle is $\le 0$.

## Future Work
If verified, we can trust that the agent won't learn to farm seeds.
