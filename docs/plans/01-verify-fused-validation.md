# Plan: Verification of Fused Validation Fidelity

## Objective
Verify that the **Fused Validation** mechanism (used for high-performance counterfactual attribution) yields results identical to standard **Sequential Validation**. This ensures the RL agent receives accurate reward signals and is not training on artifacts of the optimization strategy.

## Context
The system evaluates seed contributions by running multiple configurations (e.g., "Seed A off", "Seed B off") to measure performance drops.
*   **Sequential Approach:** Run the model $K$ times, once for each config. (Slow, correct).
*   **Fused Approach:** Replicate the batch $K$ times, apply different alpha masks to each replica, and run once. (Fast, potentially risky).

**Risks:**
*   **BatchNorm Statistics:** Tiling the batch changes the effective batch size and statistics if not strictly in `eval` mode with frozen stats.
*   **State Leakage:** Hidden states (LSTM/RNN) or internal buffers might bleed across the "virtual" batch boundaries.
*   **Alpha Override Logic:** The mechanism for forcing `alpha=0` in the fused pass might drift from the actual `forward` path logic.

## Strategy
We will create a standalone reproduction script `tests/repro_fused_fidelity.py` (or similar integration test) that instantiates a representative `SlottedHost` and runs both validation methods on identical inputs.

### 1. Test Setup
*   **Environment:** Initialize a `VectorizedHamletEnv` (or lighter weight equivalent) with a standard ResNet-like host and 2 active seeds.
*   **State:** Ensure seeds are in a mixed state (e.g., one fully `BLENDING` with alpha=0.5, one `FOSSILIZED` with alpha=1.0) to exercise the alpha override logic.
*   **Data:** Use a fixed, deterministic batch of synthetic data (frozen random seed).

### 2. Execution Steps

#### Step A: Sequential Ground Truth
1.  Define a list of configurations: `[Main (all active), Solo_A (A=0), Solo_B (B=0)]`.
2.  For each config:
    *   Manually set seed alphas on the model instances.
    *   Run `model(input)`.
    *   Record `output` tensors.
    *   **Crucial:** Reset model state between runs to ensure total isolation.

#### Step B: Fused Execution
1.  Construct the `alpha_overrides` tensor exactly as `vectorized.py` does:
    *   Shape: `[K * BatchSize, ...]`
    *   Values: Tiled alpha masks corresponding to the configs in Step A.
2.  Tile the input data: `input_fused = input.repeat(K, ...)`
3.  Run `model(input_fused, alpha_overrides=...)`.
4.  Split the result `output_fused` into $K$ chunks.

### 3. Verification Metrics
Compare `Step A` outputs vs. `Step B` chunks.

*   **Metric 1: Bitwise Equality (Strict):** `torch.allclose(sequential_out, fused_chunk, atol=1e-6)`
    *   *Success:* Passes.
    *   *Failure:* Indicates divergence. Investigate `eval()` mode, BN running stats, or masking logic.
*   **Metric 2: Loss Delta:** Compute loss for both. Ensure $\Delta < 1e-7$.
*   **Metric 3: Memory Layout:** Verify that `channels_last` or other memory format optimizations in the fused path don't alter numerical results compared to standard format.

## Implementation Plan

1.  **Scaffold Test:** Create `tests/integration/test_fused_fidelity.py`.
2.  **Mock Components:** Use real `SeedSlot` and `Kasmina` logic but maybe a smaller Host to speed up the loop.
3.  **Run Comparison:** Implement the loop described above.
4.  **Failure Analysis:** If they differ, add hooks to `SeedSlot.forward` to log intermediate activations and pinpoint divergence.

## Success Criteria
*   The test passes with `atol=1e-6` for both logits and loss values.
*   Confirmation that `model.eval()` is correctly freezing all batch norm updates during the fused pass.

## Future Work
If fused validation proves identical, we can trust the "Shapley Value" estimations built on top of it. If it differs, we must either fix the fused kernel or revert to sequential validation for the "Counterfactual" reward signal.
