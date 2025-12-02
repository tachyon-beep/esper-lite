# SME Report: esper.scripts Package

## Executive Summary

The `esper.scripts` package provides two CLI entry points for training and evaluating Simic PPO agents: `train.py` orchestrates single/vectorized PPO and heuristic baseline training with Nissa telemetry integration, while `evaluate.py` runs comprehensive diagnostics (5 categories) on trained models to assess policy learning, value calibration, and seed success metrics.

## Key Features

### train.py (141 LOC)
- **Multi-algorithm CLI**: Dispatches to heuristic baseline or PPO (single-env or vectorized)
- **Hyperparameter flexibility**: Entropy annealing (start/end/episodes), clip ratio, learning rate, discount factor, seed reproducibility
- **Nissa telemetry integration**: Console output + optional JSONL file logging for all lifecycle events (germination, fossilization, culling)
- **Task switching**: Presets for cifar10 and tinystories with automatic dataloaders
- **Checkpoint management**: Save/resume model paths for long-running training
- **Vectorized training support**: Multi-environment parallel training with per-device GPU allocation

### evaluate.py (824 LOC)
- **Diagnostic framework** (5 diagnostic categories):
  1. **Action Distribution**: Frequency analysis + entropy metrics (mean/min/max) by phase
  2. **Temporal Patterns**: Phase-dependent action distributions, germination timing, seed lifetime tracking
  3. **Value Calibration**: V(s) vs actual returns correlation, MAE, bias
  4. **State-Action Contingency**: Policy response to loss trends and seed presence
  5. **Seed Success**: Creation, fossilization, cull rates, survival percentage, accuracy deltas (with/without seeds)
- **Red flag detection**: Collapsed policies, entropy extremes, poor value calibration, missing seeds
- **Rich reporting**: Formatted text output with per-phase analysis, optional file export
- **Episode batching**: Deterministic evaluation over multiple episodes with episode seed control

## DRL/PyTorch Assessment

### DRL Implementation
- **PPO algorithm**: Standard off-policy GAE advantage estimation with clipped surrogate objective
- **Entropy scheduling**: Supports annealing from start→end over configurable episode windows
- **Value function**: Shared actor-critic architecture with separate value head
- **Discount factor**: gamma=0.99 (0.95 in vectorized variant)
- **Action space**: Discrete enum-based actions (germinate variants, fossilize, cull, idle)
- **Reward shaping**: Sparse accuracy deltas + parameter cost (penalizes seed growth)

### PyTorch Usage
- **Standard ops**: CrossEntropyLoss, SGD optimizers, no_grad() evaluation contexts
- **Gradient handling**: Direct STE backprop through seed paths with optional isolation
- **Device management**: Single-GPU (--device) and multi-GPU (--devices with distribution)
- **Model saving**: Checkpoint serialization via PPOAgent.load()

### Limitations & Concerns
- **Evaluate.py only supports deterministic inference**: Cannot probe stochastic policy behavior
- **Loss trend classification**: Uses simplistic delta thresholding (±0.05) on 3-step windows, fragile on noisy loss curves
- **Value calibration correlation**: Computed without normalization, sensitive to scale mismatch
- **No curriculum or attention masking** in base hyperparameters
- **Entropy coef defaults**: 0.01 in single-env but 0.1 in vectorized (inconsistency)

## Risks & Opportunities

### Risks
1. **Telemetry overhead**: Nissa hub integration may add latency; --no-telemetry flag only reduces feature dims, doesn't skip emission
2. **Red flag thresholds hardcoded**: Policy collapse threshold (90%), entropy bounds (0.1-1.8), survival threshold (10%) cannot be tuned
3. **Seed lifecycle validation weak**: evaluate.py does not check if fossilization gates were respected; assumes model correctly enforces stage transitions
4. **Vectorized/single-env divergence**: Different default entropy coefs and gamma values; no validation that both pathways produce comparable results
5. **Memory footprint unknown**: evaluate.py loads full trainloader/testloader; no batching cap or memory warnings for large datasets

### Opportunities
1. **Diagnostic extensibility**: Add probe for policy determinism recovery (entropy collapse over time), gradient flow analysis
2. **Hyperparameter sensitivity scans**: CLI could auto-sweep entropy_coef_start/end and report convergence time
3. **Transfer diagnostics**: evaluate.py could accept multiple model checkpoints for comparative analysis
4. **Seed blueprint effectiveness**: Breakdown success rate by blueprint_id to identify which architectural patches work best
5. **Integration with MLflow/W&B**: Nissa backends could push diagnostic reports to experiment tracking platforms

## Recommendations

1. **Standardize entropy defaults**: Align single-env and vectorized entropy_coef defaults (0.05 is reasonable middle ground)
2. **Expose red flag thresholds**: Add CLI args `--collapse-threshold`, `--entropy-bounds`, `--survival-threshold` to evaluate.py
3. **Validate seed gates in diagnostics**: Add check that fossilized seeds never transition backward; emit warning if constraint violated
4. **Add deterministic policy option**: New evaluate flag `--stochastic` to sample from policy instead of argmax, measure action variance
5. **Document reward shaping formula**: evaluate.py calls compute_shaped_reward but doesn't show the cost function; add inline docs or --show-reward-components CLI flag
6. **Benchmark telemetry impact**: Profile train.py with/without Nissa backends to quantify overhead
7. **Test vectorized convergence parity**: Add regression test comparing single-env vs multi-env results with fixed seed

