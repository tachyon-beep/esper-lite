# Tamiyo Policy Migration - Follow-up Cleanup

> **For Claude:** These are minor improvements identified during code review of the Tamiyo Policy Migration (PR #31). Execute task-by-task.

**Goal:** Address 15 minor issues found during code review to improve code quality, test coverage, and documentation accuracy.

**Priority:** Low - these are polish items, not blocking issues.

---

## Phase 1: Code Cleanup (3 tasks)

### Task 1.1: Remove unused imports

**Files:**
- `tests/tamiyo/policy/test_protocol.py` - Remove unused `runtime_checkable` import
- `src/esper/tamiyo/policy/types.py` - Remove unused `Any` import

**Step 1: Fix test_protocol.py**

```python
# Remove this import if not used:
from typing import runtime_checkable
```

**Step 2: Fix types.py**

```python
# Remove unused Any import if present
from typing import Any  # <- Remove if unused
```

**Step 3: Verify**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_protocol.py -v
```

**Step 4: Commit**

```bash
git add tests/tamiyo/policy/test_protocol.py src/esper/tamiyo/policy/types.py
git commit -m "chore: remove unused imports in tamiyo policy"
```

---

### Task 1.2: Fix float comparison in test

**Files:**
- `tests/tamiyo/policy/test_protocol.py` (line 57)

**Issue:** Using exact `==` for float comparison instead of tolerance-based comparison.

**Step 1: Find the offending line**

```bash
grep -n "==" tests/tamiyo/policy/test_protocol.py | head -20
```

**Step 2: Replace exact comparison with pytest.approx**

```python
# Before:
assert result.value.item() == 0.5

# After:
import pytest
assert result.value.item() == pytest.approx(0.5)
```

**Step 3: Verify**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_protocol.py -v
```

**Step 4: Commit**

```bash
git add tests/tamiyo/policy/test_protocol.py
git commit -m "test: use tolerance-based float comparison in protocol tests"
```

---

### Task 1.3: Rename test method for consistency

**Files:**
- Find test file with `test_tamiyo_buffer_stores_factored_transitions`

**Issue:** Test method still references "tamiyo_buffer" after file was renamed to "rollout_buffer".

**Step 1: Find the test**

```bash
grep -r "test_tamiyo_buffer" tests/ --include="*.py"
```

**Step 2: Rename to consistent name**

```python
# Before:
def test_tamiyo_buffer_stores_factored_transitions():

# After:
def test_rollout_buffer_stores_factored_transitions():
```

**Step 3: Verify**

```bash
PYTHONPATH=src uv run pytest tests/ -v -k "rollout_buffer"
```

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: rename tamiyo_buffer test to rollout_buffer for consistency"
```

---

## Phase 2: Registry & Protocol Improvements (2 tasks)

### Task 2.1: Add duplicate registration test

**Files:**
- `tests/tamiyo/policy/test_registry.py`

**Issue:** Code handles duplicate registration with ValueError, but no test verifies this behavior.

**Step 1: Add test**

```python
def test_register_policy_duplicate_raises():
    """@register_policy should raise for duplicate names."""
    @register_policy("duplicate_test")
    class FirstPolicy(MockPolicyBundle):
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_policy("duplicate_test")
        class SecondPolicy(MockPolicyBundle):
            pass
```

**Step 2: Verify**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_registry.py::test_register_policy_duplicate_raises -v
```

**Step 3: Commit**

```bash
git add tests/tamiyo/policy/test_registry.py
git commit -m "test: add duplicate policy registration test"
```

---

### Task 2.2: Add comment explaining protocol validation

**Files:**
- `src/esper/tamiyo/policy/registry.py`

**Issue:** The hasattr-based validation approach needs explanation (why not isinstance).

**Step 1: Add explanatory comment**

Find the validation block and add context:

```python
        # NOTE: We use hasattr checks instead of isinstance(cls, PolicyBundle)
        # because PolicyBundle is a Protocol. While @runtime_checkable allows
        # isinstance() on instances, we need to validate the CLASS before
        # instantiation (policies require constructor args). hasattr on the
        # class checks that methods are defined, which is sufficient for
        # protocol compliance at registration time.
        required_methods = [
            'process_signals', 'get_action', ...
```

**Step 2: Verify no syntax errors**

```bash
PYTHONPATH=src python -c "from esper.tamiyo.policy.registry import register_policy; print('OK')"
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/policy/registry.py
git commit -m "docs: explain hasattr-based protocol validation in registry"
```

---

## Phase 3: Documentation Updates (4 tasks)

### Task 3.1: Update features.py docstring

**Files:**
- `src/esper/tamiyo/policy/features.py`

**Issue:** Module docstring still says "Simic Features" instead of "Tamiyo Policy Features".

**Step 1: Update docstring**

```python
# Before:
"""Simic Features - Feature extraction for training signals."""

# After:
"""Tamiyo Policy Features - Feature extraction for training signals.

This module was moved from simic.control to tamiyo.policy as part of
the Policy Migration (PR #31). It is re-exported from simic.control
for backwards compatibility.
"""
```

**Step 2: Commit**

```bash
git add src/esper/tamiyo/policy/features.py
git commit -m "docs: update features.py docstring for tamiyo location"
```

---

### Task 3.2: Update MaskedCategorical docstring

**Files:**
- `src/esper/tamiyo/policy/action_masks.py`

**Issue:** Docstring still references `torch.finfo().min` instead of `MASKED_LOGIT_VALUE`.

**Step 1: Find and update docstring**

```python
# Find MaskedCategorical class and update any references to torch.finfo().min
# to mention MASKED_LOGIT_VALUE from leyline
```

**Step 2: Commit**

```bash
git add src/esper/tamiyo/policy/action_masks.py
git commit -m "docs: update MaskedCategorical docstring to reference MASKED_LOGIT_VALUE"
```

---

### Task 3.3: Update off-policy error messages

**Files:**
- `src/esper/tamiyo/policy/lstm_bundle.py`

**Issue:** Error messages reference non-existent "MLPPolicyBundle".

**Step 1: Update error messages**

```python
# Before:
raise NotImplementedError(
    "LSTMPolicyBundle does not support off-policy algorithms. "
    "Use MLPPolicyBundle with SAC/TD3 instead."
)

# After:
raise NotImplementedError(
    "LSTMPolicyBundle does not support off-policy algorithms. "
    "Off-policy support requires a future MLP-based PolicyBundle implementation."
)
```

**Step 2: Commit**

```bash
git add src/esper/tamiyo/policy/lstm_bundle.py
git commit -m "docs: fix off-policy error messages (remove MLPPolicyBundle reference)"
```

---

### Task 3.4: Update simic.md specification

**Files:**
- `docs/specifications/simic.md`

**Issue:** Still references "tamiyo_buffer" instead of "rollout_buffer".

**Step 1: Find and replace**

```bash
grep -n "tamiyo_buffer" docs/specifications/simic.md
```

Replace all occurrences with "rollout_buffer".

**Step 2: Commit**

```bash
git add docs/specifications/simic.md
git commit -m "docs: update simic.md to reference rollout_buffer"
```

---

## Phase 4: LSTMPolicyBundle Improvements (3 tasks)

### Task 4.1: Fix process_signals() implementation (IMPORTANT)

**Files:**
- `src/esper/tamiyo/policy/lstm_bundle.py`

**Issue:** `process_signals()` calls `signals_to_features()` but may be missing required kwargs.

**Step 1: Review signals_to_features signature**

```bash
grep -A 20 "def signals_to_features" src/esper/simic/agent/ppo.py
```

**Step 2: Update process_signals to pass required args**

The implementation should match what the function expects. If signals_to_features requires additional parameters beyond signals and slot_config, add them.

**Step 3: Add test for process_signals**

```python
def test_lstm_bundle_process_signals(lstm_bundle):
    """process_signals should convert TrainingSignals to features."""
    from esper.leyline import TrainingSignals
    # Create minimal valid TrainingSignals
    signals = TrainingSignals(...)  # Appropriate construction
    features = lstm_bundle.process_signals(signals)
    assert features.shape[-1] == lstm_bundle.feature_dim
```

**Step 4: Commit**

```bash
git add src/esper/tamiyo/policy/lstm_bundle.py tests/tamiyo/policy/test_lstm_bundle.py
git commit -m "fix(tamiyo): fix process_signals() kwargs in LSTMPolicyBundle"
```

---

### Task 4.2: Add test for LSTMPolicyBundle.forward()

**Files:**
- `tests/tamiyo/policy/test_lstm_bundle.py`

**Issue:** Method exists but is untested.

**Step 1: Add test**

```python
def test_lstm_bundle_forward(lstm_bundle, slot_config):
    """forward() should return ForwardResult with logits."""
    features = torch.randn(1, 50)
    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, 5, dtype=torch.bool),
        "blend": torch.ones(1, 3, dtype=torch.bool),
        "op": torch.ones(1, 4, dtype=torch.bool),
    }
    hidden = lstm_bundle.initial_hidden(batch_size=1)

    result = lstm_bundle.forward(features, masks, hidden)

    assert isinstance(result, ForwardResult)
    assert "op" in result.logits
    assert result.value is not None
    assert result.hidden is not None
```

**Step 2: Verify**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py::test_lstm_bundle_forward -v
```

**Step 3: Commit**

```bash
git add tests/tamiyo/policy/test_lstm_bundle.py
git commit -m "test: add LSTMPolicyBundle.forward() test"
```

---

### Task 4.3: Add test for LSTMPolicyBundle.get_value()

**Files:**
- `tests/tamiyo/policy/test_lstm_bundle.py`

**Issue:** Method exists but is untested.

**Step 1: Add test**

```python
def test_lstm_bundle_get_value(lstm_bundle):
    """get_value() should return state value estimate."""
    features = torch.randn(1, 50)
    hidden = lstm_bundle.initial_hidden(batch_size=1)

    value = lstm_bundle.get_value(features, hidden)

    assert isinstance(value, torch.Tensor)
    assert value.shape == (1,) or value.shape == ()  # Scalar or batch
```

**Step 2: Verify**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py::test_lstm_bundle_get_value -v
```

**Step 3: Commit**

```bash
git add tests/tamiyo/policy/test_lstm_bundle.py
git commit -m "test: add LSTMPolicyBundle.get_value() test"
```

---

## Phase 5: HeuristicPolicyBundle Improvements (2 tasks)

### Task 5.1: Fix initial_hidden return type annotation

**Files:**
- `src/esper/tamiyo/policy/heuristic_bundle.py`

**Issue:** Return type annotation says `None` but should be `tuple[...] | None` for protocol compliance.

**Step 1: Update type annotation**

```python
# Before:
def initial_hidden(self, batch_size: int) -> None:

# After:
def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
```

**Step 2: Verify**

```bash
PYTHONPATH=src python -c "from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle; print('OK')"
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/policy/heuristic_bundle.py
git commit -m "fix(tamiyo): correct initial_hidden return type in HeuristicPolicyBundle"
```

---

### Task 5.2: Add HeuristicPolicyBundle property tests

**Files:**
- `tests/tamiyo/policy/test_heuristic_bundle.py`

**Issue:** Missing tests for device, dtype, to() to match LSTM bundle coverage.

**Step 1: Add tests**

```python
def test_heuristic_bundle_device():
    """device should return CPU."""
    bundle = HeuristicPolicyBundle()
    assert bundle.device == torch.device("cpu")


def test_heuristic_bundle_dtype():
    """dtype should return float32."""
    bundle = HeuristicPolicyBundle()
    assert bundle.dtype == torch.float32


def test_heuristic_bundle_to():
    """to() should be no-op but return self."""
    bundle = HeuristicPolicyBundle()
    result = bundle.to("cpu")
    assert result is bundle
```

**Step 2: Verify**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_heuristic_bundle.py -v
```

**Step 3: Commit**

```bash
git add tests/tamiyo/policy/test_heuristic_bundle.py
git commit -m "test: add HeuristicPolicyBundle device/dtype/to tests"
```

---

## Summary

| Phase | Tasks | Priority |
|-------|-------|----------|
| 1. Code Cleanup | 3 | Low |
| 2. Registry & Protocol | 2 | Low |
| 3. Documentation | 4 | Low |
| 4. LSTMPolicyBundle | 3 | Medium (4.1 is important) |
| 5. HeuristicPolicyBundle | 2 | Low |

**Total: 14 tasks** (one item was a documentation note, not a code change)

After completion, the Tamiyo Policy Migration will be fully polished with:
- Clean imports
- Accurate documentation
- Complete test coverage
- Consistent naming
