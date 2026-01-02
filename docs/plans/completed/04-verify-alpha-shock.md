# Plan: Verification of Alpha Shock Transient

## Objective
Quantify and verify the "Alpha Shock" transient—the destabilizing jolt to the loss landscape that occurs when a seed transitions from `TRAINING` (Alpha=0) to `BLENDING` (Alpha>0). We must ensure this shock is within a "stability budget" to prevent the RL agent from learning to avoid blending (and thus growth) entirely.

## Context
In the `TRAINING` stage, the seed learns from the host's error but does not contribute to the output (`alpha=0`).
In the `BLENDING` stage, `alpha` ramps up from 0 to 1.
Ideally, the first step of `BLENDING` (`alpha = epsilon`) should introduce a *smooth* change in the output.
**Risk:** If the seed's output is uncorrelated with the host's error (due to poor "Incubation"), introducing it even with a small alpha can cause a spike in loss. The RL agent (`Simic`) might perceive this spike as a penalty and learn to abort blending immediately.

## Strategy
We will create a standalone integration test `tests/integration/test_alpha_shock.py` that simulates the `G1->G2` transition and measures the derivative of the loss with respect to `alpha`.

### 1. Test Setup
*   **Environment:** `MorphogeneticModel` (CNNHost) with `conv_small` seed.
*   **Data:** Fixed batch of CIFAR-10 data.
*   **State:** Seed has just finished `TRAINING`. We will simulate two conditions:
    *   **Condition A (Untrained/Random):** Seed weights are random (worst case).
    *   **Condition B (Ideal):** Seed weights are trained to minimize residual error (best case).

### 2. Execution Steps

#### Step 1: Establish Baseline
1.  Run forward pass with `alpha=0`. Record `loss_0`.

#### Step 2: Inject Shock
1.  Set `alpha = 0.01` (typical first step of a 100-step ramp).
2.  Run forward pass. Record `loss_epsilon`.

#### Step 3: Measure Sensitivity
1.  Compute `shock = loss_epsilon - loss_0`.
2.  Compute `sensitivity = d(loss)/d(alpha) approx shock / 0.01`.

### 3. Verification Metrics
*   **Metric 1: Smoothness:** The shock should be proportional to alpha.
    *   `abs(loss_epsilon - loss_0) < threshold`
*   **Metric 2: Mitigated vs. Random:**
    *   Ideally, a *trained* seed should have a *negative* shock (loss decreases immediately), or at least a smaller positive shock than a random seed.
    *   We will verify if `shock_trained < shock_random`.

## Implementation Plan

1.  **Scaffold Test:** Create `tests/integration/test_alpha_shock.py`.
2.  **Simulation:**
    *   **Random:** Germinate seed, do not train, switch to blending.
    *   **Trained:** Germinate seed, train on 1 batch using `isolate_gradients=True`, then switch.
3.  **Measurement:** Assert that `loss` doesn't explode (e.g., > 10% increase) on the first step.

## Success Criteria
*   **Random Seed:** Loss increase is measurable but bounded (not catastrophic).
*   **Trained Seed:** Loss increase is significantly lower than random seed, or ideally a decrease.
*   **Gradient Norm:** The gradient of parameters with respect to the loss shouldn't explode.

## Why this matters
If the "Incubator" works, a trained seed should *immediately* improve the loss (or stay neutral) upon blending. If it causes a spike, the Incubator failed to align the seed with the host's needs, and the RL agent will be punished for trying to grow.
