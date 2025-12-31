# Tamiyo EV Fix - Linear Holding Penalty + Gradient Unclipping

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix negative Explained Variance (EV) in Tamiyo by linearizing the exponential holding penalty and relaxing gradient clipping.

**Architecture:** Two-pronged fix: (1) Replace exponential holding_warning (-1, -3, -9, -27) with linear "rent" model (-0.1 to -0.3 per epoch), reducing cumulative 5-epoch penalty from -23.0 to -0.7. (2) Increase DEFAULT_MAX_GRAD_NORM from 0.5 to 1.0 to allow critic learning.

**Tech Stack:** Python, PyTorch (PPO), Hypothesis property tests

---

## Root Cause Analysis

The systematic debugging investigation revealed:

| Symptom | Evidence | Root Cause |
|---------|----------|------------|
| EV = -0.16 (HARMFUL) | Telemetry: 981 PPO updates | Value function predictions worse than mean |
| grad_norm = 0.5 (100% saturation) | All updates hitting clip ceiling | Gradient starvation - critic cannot learn |
| Value loss = 21.75 (extremely high) | Returns have high variance | Reward scale mismatch |

**The fuel for the fire:** The exponential `holding_warning` creates massive reward spikes:
- Epoch 2: -1.0
- Epoch 3: -3.0
- Epoch 4: -9.0
- Epoch 5: -27.0 (capped to -10.0)

This creates return variance that exceeds the critic's learning capacity under aggressive gradient clipping (0.5).

## The Fix

1. **Linearize holding penalty** → Gentle "rent" pressure instead of exponential punishment
2. **Relax gradient clipping** → Allow critic to learn from high-variance returns

---

### Task 1: Write Failing Tests for Linear Holding Penalty

**Files:**
- Modify: `tests/simic/test_rewards.py:961-998`
- Modify: `tests/simic/properties/test_warning_signals.py:144-184`
- Modify: `tests/simic/test_reward_simplified.py:136,148` (stale comments only)

**Step 1: Update the unit test for linear escalation**

Replace the exponential test with linear schedule test in `tests/simic/test_rewards.py`:

```python
def test_penalty_escalates_over_epochs(self):
    """WAIT penalty should escalate linearly each epoch in HOLDING.

    Linear "rent" model prevents massive reward spikes that destabilize
    the value function. Schedule: epoch 1: 0, epoch 2: -0.1, epoch 3: -0.15,
    epoch 4: -0.2, epoch 5: -0.25, epoch 6+: -0.3 (capped)
    """
    from enum import IntEnum
    class MockAction(IntEnum):
        WAIT = 0

    def get_penalty(epochs_in_stage: int) -> float:
        seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=epochs_in_stage,
            seed_age_epochs=10 + epochs_in_stage,
        )
        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=seed,
            epoch=10 + epochs_in_stage,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )
        return components.holding_warning

    # Verify linear escalation with cap
    # Schedule: epoch 1: 0, epoch 2: -0.1, epoch 3: -0.15, epoch 4: -0.2,
    #           epoch 5: -0.25, epoch 6+: -0.3 (capped)
    assert get_penalty(1) == 0.0  # Grace period
    assert get_penalty(2) == pytest.approx(-0.1)
    assert get_penalty(3) == pytest.approx(-0.15)
    assert get_penalty(4) == pytest.approx(-0.2)
    assert get_penalty(5) == pytest.approx(-0.25)
    assert get_penalty(6) == pytest.approx(-0.3)  # Cap kicks in
    assert get_penalty(10) == pytest.approx(-0.3)  # Still capped
```

**Step 2: Update the property test to verify linear (not exponential) escalation**

Replace `test_holding_warning_exponential` in `tests/simic/properties/test_warning_signals.py`:

```python
@given(epochs_in_stage=st.integers(2, 6))
@settings(max_examples=100)
def test_holding_warning_linear(self, epochs_in_stage):
    """Holding warning should escalate linearly with a cap."""
    def get_warning(epochs: int) -> float:
        seed_info = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=0.5,
            total_improvement=2.0,
            epochs_in_stage=epochs,
            seed_params=50_000,
            previous_stage=STAGE_BLENDING,
            previous_epochs_in_stage=5,
            seed_age_epochs=epochs + 10,
        )

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=3.0,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.3,
            return_components=True,
        )
        return components.holding_warning

    # Compare consecutive epochs
    warning_n = get_warning(epochs_in_stage)
    warning_n1 = get_warning(epochs_in_stage + 1)

    # Should be more negative (linear escalation up to cap)
    if warning_n < 0 and warning_n1 < 0:
        # Linear: each step should be at most 0.05 more negative (or at cap -0.3)
        diff = warning_n - warning_n1  # positive if n1 is more negative
        assert diff >= 0 or warning_n1 == pytest.approx(-0.3), (
            f"Warning should escalate linearly or be capped "
            f"(epoch {epochs_in_stage}: {warning_n}, epoch {epochs_in_stage+1}: {warning_n1})"
        )
        # Max penalty should be capped at -0.3
        assert warning_n1 >= -0.3 - 0.01, (
            f"Warning should be capped at -0.3, got {warning_n1}"
        )
```

**Step 3: Update epoch 2 assertion**

In `test_penalty_starts_at_epoch_2` (line 957-958), change:

```python
# OLD: assert components.holding_warning == -1.0
# NEW:
assert components.holding_warning == pytest.approx(-0.1), (
    f"Epoch 2 should have -0.1 penalty: {components.holding_warning}"
)
```

**Step 4: Update the farming test assertion**

In `test_good_seed_farming_penalized` (line 1462-1463), change:

```python
# OLD: assert components.holding_warning == pytest.approx(-3.0)
# NEW (epoch 3 = -0.15):
assert components.holding_warning == pytest.approx(-0.15), (
    f"Good seed farming should receive linear HOLD penalty: {components.holding_warning}"
)
```

**Step 5: Update stale comments in test_reward_simplified.py**

In `tests/simic/test_reward_simplified.py`, update the explanatory comments (lines 136, 148):

```python
# OLD (line 136):
# With SHAPED, this would trigger severe holding_warning (-9.0 or worse)

# NEW:
# With SHAPED, this would trigger holding_warning (-0.25 at epoch 5)
```

```python
# OLD (line 148):
# Should NOT have the -9.0 holding_warning

# NEW:
# Should NOT have the -0.25 holding_warning (SIMPLIFIED skips warnings)
```

**Step 6: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_rewards.py::TestHoldingPenalty -v`
Expected: FAIL - exponential values don't match linear expectations

Run: `PYTHONPATH=src uv run pytest tests/simic/properties/test_warning_signals.py::TestHoldingWarning -v`
Expected: FAIL - exponential escalation fails linear assertions

---

### Task 2: Implement Linear Holding Penalty

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py:678-699`

**Step 1: Replace exponential formula with linear "rent" model**

Replace lines 691-696 in `rewards.py`:

```python
# OLD CODE:
# Exponential: epoch 2 -> -1.0, epoch 3 -> -3.0, epoch 4 -> -9.0
# Formula: -1.0 * (3 ** (epochs_waiting - 1))
epochs_waiting = seed_info.epochs_in_stage - 1
holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
# Cap at -10.0 (clip boundary) to avoid extreme penalties
holding_warning = max(holding_warning, -10.0)

# NEW CODE:
# Linear "rent" model: gentle pressure to make a decision
# Schedule: epoch 2: -0.1, epoch 3: -0.15, epoch 4: -0.2, ...
# Cumulative over 5 epochs: -0.7 (vs -23.0 with exponential)
# DRL Expert design 2025-12-28: prevents reward spikes that
# destabilize value function while maintaining decision pressure
epochs_waiting = seed_info.epochs_in_stage - 1
base_penalty = 0.1
ramp_penalty = max(0, epochs_waiting - 1) * 0.05
per_epoch_penalty = min(base_penalty + ramp_penalty, 0.3)
holding_warning = -per_epoch_penalty
```

**Step 2: Update the docstring comment above**

Replace lines 670-677 (the comment block) with:

```python
# === 1. Holding Warning ===
# Linear "rent" penalty for WAIT in HOLDING with positive attribution.
# Prevents seed farming (staying in HOLDING indefinitely to collect rewards).
#
# DRL Expert review 2025-12-28: Exponential penalty (-1, -3, -9, -27) created
# massive reward spikes that destabilized value function learning (EV < 0).
# Linear model caps at -0.3/epoch with cumulative 5-epoch cost of -0.7.
#
# IMPORTANT: Only apply when bounded_attribution > 0 (legitimate seed being farmed).
# For ransomware seeds (attr ~= 0 due to discount), the agent's correct action is
# PRUNE, not FOSSILIZE. Penalizing WAIT in this case provides no useful gradient -
# the attribution discount already zeroed rewards. Penalty stacking creates an
# unlearnable reward landscape where every action is punished.
```

**Step 3: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_rewards.py::TestHoldingPenalty -v`
Expected: PASS

Run: `PYTHONPATH=src uv run pytest tests/simic/properties/test_warning_signals.py::TestHoldingWarning -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/rewards/rewards.py tests/simic/test_rewards.py tests/simic/properties/test_warning_signals.py
git commit -m "fix(simic): linearize holding_warning to prevent EV collapse

Replace exponential penalty (-1, -3, -9, -27) with linear 'rent' model
(-0.1 to -0.3 per epoch). Cumulative 5-epoch cost drops from -23.0 to -0.7.

Root cause: Exponential penalties created reward spikes that exceeded
the critic's learning capacity, causing negative Explained Variance.

DRL Expert design: Linear ramp with cap maintains decision pressure
while keeping returns within learnable range.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Increase DEFAULT_MAX_GRAD_NORM

**Files:**
- Modify: `src/esper/leyline/__init__.py:97`

**Step 1: Update the constant**

Change line 97:

```python
# OLD:
DEFAULT_MAX_GRAD_NORM = 0.5

# NEW:
DEFAULT_MAX_GRAD_NORM = 1.0
```

**Step 2: Update the comment**

Change line 95-96:

```python
# OLD:
# Maximum gradient norm for clipping (prevents exploding gradients).
# 0.5 is standard; lower = more aggressive clipping.

# NEW:
# Maximum gradient norm for clipping (prevents exploding gradients).
# 1.0 allows critic learning with normalized returns; 0.5 was too aggressive
# for 12-layer LSTM, causing 100% gradient saturation and negative EV.
```

**Step 3: Run full test suite to check for regressions**

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v --tb=short`
Expected: All tests pass (no tests depend on specific grad norm value)

**Step 4: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "fix(leyline): increase DEFAULT_MAX_GRAD_NORM from 0.5 to 1.0

0.5 was too aggressive for 12-layer LSTM, causing 100% gradient saturation
on every PPO update. Critic could not learn, leading to negative EV.

Evidence from telemetry: 981 PPO updates all showed grad_norm = 0.5
(the clip ceiling), meaning gradients were always being truncated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Verification Run

**Step 1: Run a short training to verify EV improvement**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 2 --enable-tamiyo
```

**Step 2: Check telemetry for improved metrics**

Query the MCP server or check logs for:
- `explained_variance` should trend positive (> 0)
- `grad_norm` should NOT always hit 1.0 (some headroom)
- `value_loss` should decrease over training

**Step 3: Optional - compare before/after**

If you have telemetry from a pre-fix run, compare:

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| EV | -0.16 | > 0 | ✓ |
| grad_norm saturation | 100% | < 50% | ✓ |
| holding_warning range | [-10, 0] | [-0.3, 0] | ✓ |

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | tests/simic/test_rewards.py, tests/simic/properties/test_warning_signals.py | Failing tests for linear penalty |
| 2 | src/esper/simic/rewards/rewards.py | Implement linear holding_warning |
| 3 | src/esper/leyline/__init__.py | Increase max_grad_norm to 1.0 |
| 4 | (verification) | Confirm EV improvement |

**Total commits:** 2 (one for reward fix + tests, one for grad norm)
