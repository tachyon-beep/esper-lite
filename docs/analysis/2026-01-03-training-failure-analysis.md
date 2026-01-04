# Training Failure Analysis: Value Function Collapse

**Date**: 2026-01-03
**Run**: `telemetry_2026-01-03_001725`
**Status**: Root cause identified and fix implemented
**Revision**: 2 (incorporating peer review feedback)

---

## 1. Problem Statement

Training run `telemetry_2026-01-03_001725` exhibited **value function collapse** - the critic failed to learn meaningful return predictions from the first batch onward. The run completed 32 PPO update batches over 2000 episodes but showed no improvement in value prediction quality, causing the policy to receive unreliable advantage estimates.

### Symptoms Observed

1. **Negative explained variance from batch 1** - critic predictions were worse than predicting the mean
2. **Q-spread collapsed to zero** - all operations predicted identical values by batch 27
3. **Policy learning stalled** - entropy dropped only 2% over 32 batches
4. **Extreme gradient clipping** - pre-clip norms were 10-73x the max_grad_norm
5. **No TRAINING_COMPLETED event** - run terminated without graceful completion

---

## 2. Evidence from Telemetry

### 2.1 Training Configuration

```
Episodes:      2000
Max Batches:   2000
Max Epochs:    150
N Envs:        64
Learning Rate: 0.0001
Entropy Coef:  0.05
Value Coef:    0.5
Max Grad Norm: 1.0
Task:          cifar_impaired
```

### 2.2 PPO Update Progression

| Batch | Value Loss | Expl.Var | Entropy | Q-Spread | Value.Std | Pre-Clip Grad | Clip Ratio |
|------:|-----------:|---------:|--------:|---------:|----------:|--------------:|-----------:|
|     1 |      87.24 |   -0.007 |   9.376 |    0.117 |     0.463 |         72.83 |      72.8x |
|     5 |      45.18 |    0.018 |   9.381 |    0.141 |     1.007 |         14.96 |      15.0x |
|    10 |      33.48 |    0.079 |   9.351 |    0.094 |     1.386 |         11.57 |      11.6x |
|    14 |      26.62 |    0.113 |   9.322 |    0.047 |     1.467 |         13.61 |      13.6x |
|    20 |      23.69 |    0.043 |   9.287 |    0.055 |     0.718 |         11.85 |      11.8x |
|    27 |       3.55 |   -0.013 |   9.201 |    0.000 |     0.398 |         32.87 |      32.9x |
|    32 |       3.15 |   -0.044 |   9.168 |    0.063 |     0.343 |         36.18 |      36.2x |

**Key observations:**
- `explained_variance` peaked at 0.113 (batch 14) but was mostly negative
- `Q-spread` collapsed from 0.117 to 0.000 by batch 27
- `pre_clip_grad_norm` was 10-73x max_grad_norm throughout (effective learning rate reduced to 1-10%)

### 2.3 Operation Distribution

| Operation | Count | Percentage |
|-----------|------:|-----------:|
| WAIT | 1,595 | 66.5% |
| GERMINATE | 379 | 15.8% |
| PRUNE | 360 | 15.0% |
| ADVANCE | 65 | 2.7% |
| SET_ALPHA_TARGET | 1 | 0.0% |
| FOSSILIZE | 0 | 0.0% |

**Implication**: With WAIT dominating at 66.5%, the op-conditioned value head primarily learned WAIT returns, contributing to Q-spread collapse as non-WAIT ops converged to WAIT-like predictions.

### 2.4 Advantage Normalization Statistics

| Batch | Advantage Mean | Advantage Std |
|------:|---------------:|--------------:|
| 1 | -3.34e-08 | 1.000 |
| 2 | 3.18e-09 | 1.000 |
| 3 | 3.18e-09 | 1.000 |

Post-normalization stats show proper normalization (mean≈0, std≈1). **Missing metric**: Pre-normalization advantage std is not logged, which would reveal if raw advantages were near-zero before normalization.

### 2.5 KL Divergence

| Metric | Observed | Expected Range | Assessment |
|--------|----------|----------------|------------|
| KL Divergence | 0.00002 | 0.01 - 0.05 | Extremely low |
| Approx KL | 0.00000 | > 0.001 | Near zero |

**Implication**: Policy updates were barely changing the network. This is consistent with normalized advantages being near-zero (when value predictions are wrong, returns - values is approximately constant, yielding minimal advantage variance before normalization).

### 2.6 Reward Distribution (first 100 actions)

```
Zero rewards:     38 (38.0%)
Positive rewards: 48
Negative rewards: 14
Mean reward:      0.135
Range:            -0.002 to 1.048
```

### 2.7 Data Gaps

The following metrics would strengthen the analysis but were not logged:
- `return_mean` / `return_std` - would contextualize value_loss magnitude
- `pre_norm_advantage_std` - would reveal if raw advantages were near-zero

---

## 3. Theories of Failure

### Classification

| Theory | Classification | Confidence |
|--------|---------------|------------|
| A: Value Head Architecture | **Root Cause Candidate** | High |
| B: Value Head Initialization | **Root Cause Candidate** | High |
| C: Value Coefficient | **Contributing Factor** | Medium |
| D: Gradient Clipping Feedback Loop | **Contributing Factor** | High |
| E: Operation Imbalance | **Contributing Factor** | Medium |
| F: Low KL Divergence | **Downstream Consequence** | High |
| G: Reward Sparsity | **Ruled Out** | High |

---

### Theory A: Value Head Architecture (ROOT CAUSE CANDIDATE)

**Hypothesis**: The 2-layer value head lacked capacity to learn the (state, op) → return mapping.

**Evidence**:
1. Explained variance started negative (-0.007) at batch 1
2. Value predictions never differentiated between operations (Q-spread → 0)
3. The value head had only 2 layers (518→256→1) while the blueprint head had 3

**Architecture before fix**:
```python
self.value_head = nn.Sequential(
    nn.Linear(518, 256),  # Aggressive 2:1 compression
    nn.ReLU(),
    nn.Linear(256, 1),    # Direct to scalar
)
```

**Caveat**: Negative explained variance at batch 1 can also occur with:
- Noisy, non-stationary returns
- Advantage normalization amplifying noise (if raw std is tiny)
- Value clipping effects with stale predictions

**Assessment**: The critic showed systematic underfitting from batch 1, suggesting either insufficient capacity or an optimization/target-scale problem. The architectural fix addresses the capacity hypothesis.

---

### Theory B: Value Head Initialization (ROOT CAUSE CANDIDATE)

**Hypothesis**: Output layer initialization with `gain=1.0` caused initial predictions to be far from zero, creating high initial value loss.

**Evidence**:
1. Initial value_loss = 87.24 (very high)
2. Policy heads used `gain=0.01`, but value head used `gain=1.0`
3. High initial error → large MSE gradients → 72.8x clipping at batch 1

**Caveat**: We did not directly measure initial value mean/std relative to returns. The causal link is plausible but not directly confirmed.

**Assessment**: `gain=1.0` likely contributed to large initial prediction magnitudes; we reduced it to match policy heads.

---

### Theory C: Value Coefficient (CONTRIBUTING FACTOR)

**Hypothesis**: With `value_coef=0.5`, the critic received insufficient gradient signal relative to policy.

**Evidence**:
1. Policy loss dominated early training
2. Value loss dropped but explained variance didn't improve proportionally
3. Combined loss: `L = L_policy + 0.5 * L_value + entropy_bonus`

**Assessment**: Increasing to 1.0 gives the critic equal priority. This alone likely wouldn't fix architectural issues but helps ensure gradient signal reaches the value head.

---

### Theory D: Gradient Clipping Feedback Loop (CONTRIBUTING FACTOR)

**Hypothesis**: Extreme gradient clipping (10-73x) throttled value function learning.

**Evidence**:
| Batch | Pre-Clip Norm | Clip Ratio | Assessment |
|-------|---------------|------------|------------|
| 1 | 72.83 | 72.8x | Severe - only 1.4% of desired update applied |
| 14 | 13.61 | 13.6x | Problem - 7% of desired update |
| 32 | 36.18 | 36.2x | Severe - getting worse |

**Healthy range**: 1.5-3x clipping is normal; 10-73x indicates architectural problem.

**Causal analysis** (from DRL expert):
```
Value Head Failure → Poor predictions → Large value_loss
    → MSE gradient ∝ 2*(pred - target) = LARGE
    → Pre-clip norm = 72x → Effective update = 1.4%
    → Value head can't catch up → remains poor → cycle continues
```

**Assessment**: The clipping is **primarily a symptom** of the value head problem, but becomes a **reinforcing cause** by preventing corrective updates. The architectural fix (gain=0.01) should reduce initial gradients dramatically, breaking the feedback loop.

---

### Theory E: Operation Imbalance (CONTRIBUTING FACTOR)

**Hypothesis**: WAIT dominating at 66.5% caused the op-conditioned value head to converge to a near-constant function.

**Evidence**:
1. WAIT: 66.5% of actions (4x more than any other op)
2. Q-spread collapsed to 0.000 by batch 27
3. Op-conditioning via one-hot input gets ignored when one op dominates

**Mechanism**: With 4x more WAIT experience, the shared layers (LSTM → value features) allocate capacity to predict WAIT returns. Other ops converge to "WAIT-like" predictions because the network optimizes for the dominant case.

**Assessment**: This is a data distribution issue, not an architecture bug. The collapsed Q-spread is expected given the op imbalance. Potential mitigations:
- Importance sampling to up-weight rare ops
- Separate value heads per op
- Reward shaping to distinguish ops

---

### Theory F: Low KL Divergence (DOWNSTREAM CONSEQUENCE)

**Hypothesis**: Policy updates were too small to drive learning.

**Evidence**:
1. KL Divergence = 0.00002 (expected: 0.01-0.05)
2. Entropy dropped only 0.2 over 32 batches

**Causal chain**:
```
Value collapse → advantages ≈ returns - (constant prediction)
    → After normalization, advantages cluster near zero
    → Policy gradient ∝ advantage × ∇log π ≈ 0
    → KL divergence ≈ 0
```

**Assessment**: This is a **consequence**, not a cause. Fixing the value head should restore advantage variance and increase KL to healthy levels.

---

### Theory G: Reward Sparsity (RULED OUT)

**Hypothesis**: Too many zero-reward timesteps reduced learning signal.

**Evidence against**:
1. 62% of actions received non-zero reward
2. Mean reward = 0.135 (small but consistent)
3. Reward range shows signal is present (-0.002 to +1.048)

**Assessment**: Reward signal is flowing; the critic couldn't learn from it due to other factors.

---

## 4. Implemented Fix

We implemented a targeted fix addressing Theories A, B, and C.

### 4.1 Value Head Architecture Redesign

**File**: `src/esper/tamiyo/networks/factored_lstm.py` (lines 281-308)

**Before** (2 layers, no normalization):
```python
self.value_head = nn.Sequential(
    nn.Linear(518, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)
```

**After** (4 layers with LayerNorm):
```python
value_input_dim = lstm_hidden_dim + NUM_OPS  # 512 + 6 = 518
self.value_head = nn.Sequential(
    # Layer 1: Feature extraction from joint (state, op) representation
    nn.Linear(value_input_dim, head_hidden),  # 518 -> 256
    nn.LayerNorm(head_hidden),
    nn.ReLU(),
    # Layer 2: Deeper representation learning
    nn.Linear(head_hidden, head_hidden // 2),  # 256 -> 128
    nn.LayerNorm(head_hidden // 2),
    nn.ReLU(),
    # Layer 3: Final feature compression
    nn.Linear(head_hidden // 2, head_hidden // 4),  # 128 -> 64
    nn.ReLU(),
    # Layer 4: Scalar value output
    nn.Linear(head_hidden // 4, 1),  # 64 -> 1
)
```

**Rationale**:
- Gradual compression (518→256→128→64→1) instead of aggressive (518→256→1)
- LayerNorm stabilizes activations, matching the policy path
- 4 layers provide capacity for learning (state, op) → return mapping
- Parameter increase: ~133K → ~175K (+31%)

### 4.2 Output Layer Initialization Fix

**File**: `src/esper/tamiyo/networks/factored_lstm.py` (lines 339-341)

**Before**: `gain=1.0`
**After**: `gain=0.01`

**Rationale**: Initial value predictions cluster near zero, reducing initial value_loss from ~87 to <10 (estimated). This should reduce pre-clip gradient norms from 72x to ~7x, breaking the clipping feedback loop.

### 4.3 Value Coefficient Increase

**File**: `src/esper/leyline/__init__.py` (line 144)

**Before**: `DEFAULT_VALUE_COEF = 0.5`
**After**: `DEFAULT_VALUE_COEF = 1.0`

**Rationale**: Equal weight for critic and policy ensures the value function receives sufficient gradient signal.

---

## 5. Expected Outcomes

### Primary Success Metrics

| Metric | Before (Collapsed) | Expected After |
|--------|-------------------|----------------|
| Initial value_loss | 87.24 | < 10 |
| Explained variance (batch 1) | -0.007 | > -0.05 (less negative) |
| Explained variance trend | Mostly negative | Upward trend |
| Q-spread | Collapsed to 0 | Remains > 0 |
| Pre-clip grad norm | 72x | < 10x (ideally 2-5x) |
| KL divergence | 0.00002 | > 0.001 |

### Success Thresholds

- **Failure**: Explained variance < 0 after batch 5 (still broken)
- **Weak Success**: Explained variance 0-0.1 (learning, but slow)
- **Strong Success**: Explained variance > 0.2 quickly (critic understands returns)

---

## 6. Monitoring Recommendations

### Immediate (First 5 Batches)

1. **Pre-clip gradient norm** - Should drop to 2-10x (from 72x)
2. **Explained variance** - Should not be persistently negative
3. **Initial value_loss** - Should be < 20 (from 87)

### If Fix Insufficient

If pre-clip grad norm remains > 10x after architectural fix:
```python
# Consider increasing max_grad_norm
DEFAULT_MAX_GRAD_NORM = 2.0  # First attempt (from 1.0)
# If still clipping hard:
DEFAULT_MAX_GRAD_NORM = 5.0  # Second attempt
```

If KL remains near-zero despite value fix:
- Consider separate critic learning rate (critic_lr > actor_lr)
- Consider disabling value clipping
- Consider return normalization

### Long-term

- Monitor op distribution; if WAIT continues dominating, consider importance sampling
- Track Q-spread as indicator of value function health
- Log `pre_norm_advantage_std` for future diagnostics

---

## 7. Conclusion

The training failure was caused by a **combination of factors** that created a reinforcing failure loop:

```
┌─────────────────────────────────────────────────────────────┐
│  Value Head Too Shallow (2 layers) + Wrong Init (gain=1.0) │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  High Initial Value Loss (87.24) + Poor Predictions        │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Extreme Gradient Clipping (72x) → Updates Throttled to 1% │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Critic Can't Catch Up → Advantages ≈ Constant             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Near-Zero KL → Policy Stalls → No Learning                │
└─────────────────────────────────────────────────────────────┘
```

The implemented fix (4-layer value head + gain=0.01 + value_coef=1.0) targets the top of this cascade. A validation training run is in progress.

---

## Appendix: Telemetry Gaps for Future Runs

Consider adding these metrics to PPO telemetry:
- `return_mean` / `return_std` per batch
- `pre_norm_advantage_std` before normalization
- `value_prediction_mean` / `value_prediction_std` for initialization diagnostics
