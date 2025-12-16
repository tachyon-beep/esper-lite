# BUG-015: TolariaGovernor rollbacks don't restore optimizer state

- **Title:** TolariaGovernor rollbacks restore weights but not optimizer state
- **Context:** Tolaria governor (`src/esper/tolaria/governor.py`)
- **Impact:** P2 – Design improvement, low practical impact
- **Environment:** Main branch
- **Status:** Deferred (downgraded from P1)

## Analysis (2025-12-17)

**Real bug but low impact.** The optimizer state is not restored on rollback:

```python
# snapshot() only captures model weights
full_state = self.model.state_dict()

# execute_rollback() only restores model weights
self.model.load_state_dict(state_on_device, strict=False)
```

After rollback:
- Model weights: restored to snapshot
- Optimizer momentum/Adam moments: stale (from pre-panic trajectory)

## Why Low Impact

### 1. Rollbacks Are Rare

Rollbacks only happen on catastrophic failures:
- NaN/Inf loss
- Loss explosion (6σ anomaly)
- Lobotomy detection (loss jumps to random guessing)

In stable training, rollbacks never occur.

### 2. SGD Momentum Recovers Naturally

The host optimizer is `SGD(momentum=0.9)`. After rollback:
- Momentum buffers are misaligned
- But momentum decays: after ~10 steps, old momentum influence is `0.9^10 ≈ 0.35`
- New gradients gradually correct the trajectory

### 3. Safeguards Exist

If post-rollback training is unstable:
- Governor would detect continued anomalies
- Multiple consecutive anomalies trigger another rollback
- System is self-correcting

## Fix Options (Future)

### Option A: Store Optimizer State (Memory Cost)
```python
def snapshot(self):
    self.last_good_state = self.model.state_dict()
    self.last_good_optimizer = optimizer.state_dict()  # Can be large!
```

Trade-off: Doubles memory for snapshot (optimizer state can match model size for Adam).

### Option B: Reset Optimizer After Rollback
```python
def execute_rollback(self):
    self.model.load_state_dict(...)
    optimizer.load_state_dict(initial_optimizer_state)  # Fresh momentum
```

Trade-off: Loses all optimizer adaptation, but ensures consistency.

### Option C: Zero Momentum After Rollback
```python
for param_group in optimizer.param_groups:
    for p in param_group['params']:
        optimizer.state[p]['momentum_buffer'].zero_()
```

Trade-off: Simplest fix, slight slowdown as momentum rebuilds.

## Links

- `src/esper/tolaria/governor.py` (snapshot/execute_rollback)
- `src/esper/simic/vectorized.py` (host_optimizer is SGD with momentum=0.9)
