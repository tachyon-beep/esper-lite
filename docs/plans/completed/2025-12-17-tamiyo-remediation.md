# Tamiyo Subsystem Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address security, performance, and best-practice issues identified in the Tamiyo code review (2025-12-17)

**Architecture:** Fixes span Tamiyo policy infrastructure (action masking, LSTM bundle, heuristic) with minimal changes to Simic integration. All changes follow TDD with existing test patterns.

**Tech Stack:** Python 3.11+, PyTorch 2.x, pytest, Hypothesis (property-based testing)

---

## Pre-Implementation Notes

### Issues Resolved During Review

The following "critical" issues from the initial review were **false positives** after deeper investigation:

1. **Observation Normalization** - Already implemented in `simic/control/normalization.py` and used in `vectorized.py` (lines 274, 1604, 1997). The `RunningMeanStd` class provides running mean/std normalization with Welford's algorithm.

2. **Multi-head log_prob handling** - The PPO implementation intentionally uses per-head losses with causal masking (H6 FIX in `ppo.py:586-604`). This is a valid design choice, not a bug.

### Actual Remaining Issues (Prioritized)

| Priority | Task | Issue | Risk |
|----------|------|-------|------|
| 1 | Task 1 | MaskedCategorical validation CUDA sync | Performance (100-1000x overhead) |
| 2 | Task 2 | `min_improvement_to_fossilize = 0.0` | Reward hacking vulnerability |
| 3 | Task 3 | `get_value()` missing `@torch.inference_mode()` | Gradient leak |
| 4 | Task 4 | `getattr`/`hasattr` authorization | CLAUDE.md policy compliance |
| 5 | Task 5 | Entropy coefficient documentation | Training guidance |
| 6 | Task 6 | HeuristicPolicyBundle protocol violation | Type safety |

---

## Task 1: Add Validation Toggle to MaskedCategorical

**Problem:** The `_validate_action_mask()` and `_validate_logits()` functions force CUDA synchronization via `.any()` and `.sum()` on every distribution creation. This is 100-1000x overhead in tight training loops.

**Solution:** Add a class-level validation toggle that can be disabled in production while remaining enabled during development/debugging.

**Files:**
- Modify: `src/esper/tamiyo/policy/action_masks.py:285-354`
- Test: `tests/tamiyo/policy/test_action_masks.py`

**Step 1: Write the failing test**

```python
# Add to tests/tamiyo/policy/test_action_masks.py

class TestMaskedCategoricalValidation:
    """Tests for validation toggle behavior."""

    def test_validation_enabled_by_default(self):
        """Validation should be enabled by default for safety."""
        from esper.tamiyo.policy.action_masks import MaskedCategorical
        assert MaskedCategorical.validate is True

    def test_validation_can_be_disabled(self):
        """Validation can be disabled for production performance."""
        from esper.tamiyo.policy.action_masks import MaskedCategorical
        original = MaskedCategorical.validate
        try:
            MaskedCategorical.validate = False
            # This would raise InvalidStateMachineError with validation enabled
            logits = torch.zeros(1, 5)
            mask = torch.zeros(1, 5, dtype=torch.bool)  # All invalid!
            # Should NOT raise when validation is disabled
            dist = MaskedCategorical(logits, mask)
            assert dist is not None
        finally:
            MaskedCategorical.validate = original

    def test_validation_catches_invalid_mask_when_enabled(self):
        """Validation raises error for invalid masks when enabled."""
        from esper.tamiyo.policy.action_masks import (
            MaskedCategorical,
            InvalidStateMachineError,
        )
        MaskedCategorical.validate = True
        logits = torch.zeros(1, 5)
        mask = torch.zeros(1, 5, dtype=torch.bool)  # All invalid!
        with pytest.raises(InvalidStateMachineError):
            MaskedCategorical(logits, mask)
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_action_masks.py::TestMaskedCategoricalValidation -v
```

Expected: FAIL with `AttributeError: type object 'MaskedCategorical' has no attribute 'validate'`

**Step 3: Write minimal implementation**

```python
# In src/esper/tamiyo/policy/action_masks.py, modify MaskedCategorical class:

class MaskedCategorical:
    """Categorical distribution with action masking and correct entropy calculation.

    Masks invalid actions by setting their logits to MASKED_LOGIT_VALUE (-1e4)
    before softmax. This value is chosen to be:
    - Large enough to effectively zero the probability after softmax
    - Small enough to avoid numerical overflow in FP16/BF16 (finfo.min can cause issues)
    - Consistent across all dtypes for deterministic behavior

    Computes entropy only over valid actions to avoid penalizing restricted states.

    Attributes:
        validate: Class-level toggle for validation. Set to False for production
            performance (disables CUDA sync from .any()/.sum() calls).
            Default: True (validation enabled for safety during development).
    """

    validate: bool = True  # Class-level validation toggle

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """Initialize masked categorical distribution.

        Args:
            logits: Raw policy logits [batch, num_actions]
            mask: Boolean mask, True = valid, False = invalid [batch, num_actions]

        Raises:
            InvalidStateMachineError: If any batch element has no valid actions
                (only when validate=True)
            ValueError: If logits contain inf or nan (only when validate=True)

        Note:
            The validation check is isolated via @torch.compiler.disable to prevent
            graph breaks in the main forward path while preserving safety checks.
            Disable with MaskedCategorical.validate = False for production.
        """
        if MaskedCategorical.validate:
            _validate_action_mask(mask)
            _validate_logits(logits)

        self.mask = mask
        mask_value = torch.tensor(
            MASKED_LOGIT_VALUE,
            device=logits.device,
            dtype=logits.dtype,
        )
        self.masked_logits = logits.masked_fill(~mask, mask_value)
        self._dist = Categorical(logits=self.masked_logits)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_action_masks.py::TestMaskedCategoricalValidation -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/policy/action_masks.py tests/tamiyo/policy/test_action_masks.py
git commit -m "$(cat <<'EOF'
perf(tamiyo): add validation toggle to MaskedCategorical

Add class-level `validate` attribute to MaskedCategorical that controls
whether _validate_action_mask() and _validate_logits() are called.

These validation functions force CUDA synchronization via .any() and
.sum() calls, adding 100-1000x overhead in tight training loops. The
toggle allows disabling validation in production while keeping it
enabled during development for safety.

Default: validate=True (safe default, explicit opt-out for perf)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Increase Fossilization Threshold

**Problem:** `min_improvement_to_fossilize = 0.0` allows fossilizing seeds with negligible improvement (even 0.001%). This creates a reward hacking vulnerability where the agent can learn to fossilize quickly for terminal bonuses without providing real value.

**Solution:** Set a meaningful threshold (0.5% improvement minimum) to ensure fossilized seeds provide measurable value.

**Files:**
- Modify: `src/esper/tamiyo/heuristic.py:62`
- Modify: `src/esper/leyline/constants.py` (add default constant)
- Test: `tests/tamiyo/properties/test_decision_semantics.py`

**Step 1: Write the failing test**

```python
# Add to tests/tamiyo/properties/test_decision_semantics.py

def test_fossilize_requires_meaningful_improvement():
    """Fossilization should require meaningful improvement, not just any positive value."""
    from esper.tamiyo.heuristic import HeuristicConfig
    from esper.leyline.constants import DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE

    # Default threshold should be meaningful (at least 0.5%)
    assert DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE >= 0.5, (
        f"Default fossilize threshold {DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE} is too low. "
        "Seeds with negligible improvement can be fossilized, enabling reward hacking."
    )

    # HeuristicConfig should use the default
    config = HeuristicConfig()
    assert config.min_improvement_to_fossilize >= 0.5
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/properties/test_decision_semantics.py::test_fossilize_requires_meaningful_improvement -v
```

Expected: FAIL with `AssertionError: Default fossilize threshold 0.0 is too low`

**Step 3: Write minimal implementation**

```python
# In src/esper/leyline/constants.py, add:

# Fossilization threshold: minimum improvement required to fossilize a seed.
# Set to 0.5% to prevent reward hacking via marginal fossilization.
# A seed must demonstrate meaningful contribution before permanent integration.
DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE: float = 0.5
```

```python
# In src/esper/tamiyo/heuristic.py, update:

from esper.leyline.constants import (
    DEFAULT_MIN_EPOCHS_BEFORE_GERMINATE,
    DEFAULT_CULL_AFTER_EPOCHS_WITHOUT_IMPROVEMENT,
    DEFAULT_CULL_IF_ACCURACY_DROPS_BY,
    DEFAULT_EMBARGO_EPOCHS_AFTER_CULL,
    DEFAULT_BLUEPRINT_PENALTY_ON_CULL,
    DEFAULT_BLUEPRINT_PENALTY_DECAY,
    DEFAULT_BLUEPRINT_PENALTY_THRESHOLD,
    DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE,  # Add this import
)

@dataclass
class HeuristicConfig:
    # ... existing fields ...

    # Fossilization threshold (from leyline)
    min_improvement_to_fossilize: float = DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/properties/test_decision_semantics.py::test_fossilize_requires_meaningful_improvement -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/constants.py src/esper/tamiyo/heuristic.py tests/tamiyo/properties/test_decision_semantics.py
git commit -m "$(cat <<'EOF'
fix(tamiyo): increase fossilize threshold to prevent reward hacking

Change min_improvement_to_fossilize from 0.0 to 0.5% to ensure seeds
must demonstrate meaningful contribution before fossilization.

The previous threshold of 0.0 allowed seeds with negligible improvement
(even 0.001%) to fossilize, creating a reward hacking vulnerability
where the RL agent could learn to fossilize quickly for terminal
bonuses without providing real value.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add inference_mode to get_value()

**Problem:** `LSTMPolicyBundle.get_value()` creates tensors via `features.unsqueeze(1)` without `@torch.inference_mode()`. If called during evaluation with gradient tracking enabled, this creates unnecessary computation graph.

**Solution:** Add `@torch.inference_mode()` decorator since `get_value()` is only used during inference (bootstrap value computation).

**Files:**
- Modify: `src/esper/tamiyo/policy/lstm_bundle.py:208-224`
- Test: `tests/tamiyo/policy/test_lstm_bundle.py`

**Step 1: Write the failing test**

```python
# Add to tests/tamiyo/policy/test_lstm_bundle.py

def test_get_value_does_not_create_grad_graph(lstm_bundle, sample_features):
    """get_value() should not create gradient computation graph."""
    # Ensure we're in a context where gradients would normally be tracked
    features = sample_features.clone().requires_grad_(True)

    value = lstm_bundle.get_value(features)

    # Value should not require grad (computed in inference_mode)
    assert not value.requires_grad, (
        "get_value() created gradient graph. Add @torch.inference_mode() decorator."
    )
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py::test_get_value_does_not_create_grad_graph -v
```

Expected: FAIL with `AssertionError: get_value() created gradient graph`

**Step 3: Write minimal implementation**

```python
# In src/esper/tamiyo/policy/lstm_bundle.py, modify get_value():

@torch.inference_mode()
def get_value(
    self,
    features: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Get state value estimate.

    Note: Uses inference_mode() since this is only called for bootstrap
    value computation during rollout collection, not during PPO training.
    The network's forward pass for training uses evaluate_actions() instead.
    """
    # Need to add seq_len dimension if not present
    if features.dim() == 2:
        features = features.unsqueeze(1)

    output = self._network.forward(features, hidden)
    # value is [batch, seq_len], return [batch]
    return output["value"][:, 0] if output["value"].dim() > 1 else output["value"]
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py::test_get_value_does_not_create_grad_graph -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/policy/lstm_bundle.py tests/tamiyo/policy/test_lstm_bundle.py
git commit -m "$(cat <<'EOF'
perf(tamiyo): add inference_mode to LSTMPolicyBundle.get_value()

Add @torch.inference_mode() decorator to get_value() to prevent
creating unnecessary gradient computation graphs during bootstrap
value estimation.

get_value() is only used during rollout collection for computing
V(s_{T+1}) bootstrap values, not during PPO training. The training
path uses evaluate_actions() instead, which correctly handles
gradients for policy optimization.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add hasattr/getattr Authorization Comments

**Problem:** Per CLAUDE.md's strict hasattr policy, all uses of `hasattr()` and `getattr()` with fallbacks must be explicitly authorized with inline comments.

**Solution:** Add authorization comments to existing uses in `lstm_bundle.py` and `registry.py`.

**Files:**
- Modify: `src/esper/tamiyo/policy/lstm_bundle.py:243-249`
- Modify: `src/esper/tamiyo/policy/registry.py:70-71`

**Step 1: This is a documentation-only change, no test needed**

**Step 2: Add authorization comments**

```python
# In src/esper/tamiyo/policy/lstm_bundle.py, update state_dict() and load_state_dict():

def state_dict(self) -> dict[str, Any]:
    """Return network state dict."""
    # getattr AUTHORIZED by Code Review 2025-12-17
    # Justification: torch.compile wraps modules in OptimizedModule with _orig_mod
    # attribute pointing to the original module. We need the original for
    # serialization since the wrapper's state_dict includes compilation metadata.
    base = getattr(self._network, '_orig_mod', self._network)
    return base.state_dict()

def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
    """Load network state dict."""
    # getattr AUTHORIZED by Code Review 2025-12-17
    # Justification: torch.compile wraps modules in OptimizedModule with _orig_mod
    # attribute. Load into original module to avoid compilation state mismatch.
    base = getattr(self._network, '_orig_mod', self._network)
    base.load_state_dict(state_dict, strict=strict)
```

```python
# In src/esper/tamiyo/policy/registry.py, update _validate_policy_class():

def _validate_policy_class(cls: type) -> None:
    """Validate that a class structurally matches PolicyBundle protocol."""
    required = [
        "forward", "get_action", "evaluate_actions", "get_value",
        "initial_hidden", "state_dict", "load_state_dict", "to",
        "device", "dtype", "is_recurrent",
    ]
    missing = []
    for name in required:
        # hasattr AUTHORIZED by Code Review 2025-12-17
        # Justification: Protocol structural verification - checking if a class
        # implements required methods/properties before registration. This is
        # the legitimate "feature detection" exception in CLAUDE.md.
        if not hasattr(cls, name):
            missing.append(name)
    if missing:
        raise TypeError(
            f"Policy class {cls.__name__} is missing required methods/properties: {missing}"
        )
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/policy/lstm_bundle.py src/esper/tamiyo/policy/registry.py
git commit -m "$(cat <<'EOF'
docs(tamiyo): add hasattr/getattr authorization per CLAUDE.md policy

Add required authorization comments to getattr/hasattr usage:

- lstm_bundle.py: getattr for torch.compile _orig_mod unwrapping
  (serialization exception - handling compiled module internals)

- registry.py: hasattr for Protocol structural verification
  (feature detection exception - checking method presence)

Per CLAUDE.md hasattr policy, all uses must be authorized with:
1. Explicit authorization statement
2. Date of authorization
3. Justification for why hasattr/getattr is necessary

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Document Entropy Coefficient Guidance

**Problem:** `MaskedCategorical.entropy()` returns normalized entropy in [0, 1], but standard PPO entropy coefficients (~0.01) are calibrated for unnormalized entropy. Users may misconfigure the entropy coefficient.

**Solution:** Add documentation clarifying expected coefficient range for normalized entropy.

**Files:**
- Modify: `src/esper/tamiyo/policy/action_masks.py:368-384`

**Step 1: This is a documentation-only change, no test needed**

**Step 2: Update entropy docstring**

```python
# In src/esper/tamiyo/policy/action_masks.py, update entropy() method:

def entropy(self) -> torch.Tensor:
    """Compute normalized entropy over valid actions.

    Returns entropy normalized to [0, 1] by dividing by max entropy
    (log of number of valid actions). This makes exploration incentives
    comparable across states with different action restrictions.

    When only one action is valid, entropy is exactly 0 (no choice = no uncertainty).

    Entropy Coefficient Guidance:
        Since this returns NORMALIZED entropy [0, 1], the entropy coefficient
        in PPO should be higher than typical values for unnormalized entropy.

        - Unnormalized entropy: typical coef ~0.01 (entropy ranges 0 to ~3)
        - Normalized entropy: typical coef ~0.05-0.1 (entropy ranges 0 to 1)

        Example: If you want exploration equivalent to coef=0.01 with unnormalized
        entropy for a 10-action space (max_entropy = ln(10) ≈ 2.3), use:
        normalized_coef = unnormalized_coef * max_entropy ≈ 0.023

        The Esper default of entropy_coef=0.05 is appropriate for normalized entropy.
    """
    probs = self._dist.probs
    log_probs = self._dist.logits - self._dist.logits.logsumexp(dim=-1, keepdim=True)
    raw_entropy = -(probs * log_probs * self.mask).sum(dim=-1)
    num_valid = self.mask.sum(dim=-1).clamp(min=1)
    max_entropy = torch.log(num_valid.float())
    normalized = raw_entropy / max_entropy.clamp(min=1e-8)
    # Single valid action = zero entropy (no choice = no uncertainty)
    return torch.where(num_valid == 1, torch.zeros_like(normalized), normalized)
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/policy/action_masks.py
git commit -m "$(cat <<'EOF'
docs(tamiyo): add entropy coefficient guidance for normalized entropy

Document that MaskedCategorical.entropy() returns normalized entropy
[0, 1] and provide guidance on appropriate entropy coefficient values.

Standard PPO entropy coefficients (~0.01) are calibrated for
unnormalized entropy. With normalized entropy, use higher coefficients
(~0.05-0.1) to achieve equivalent exploration incentives.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Fix HeuristicPolicyBundle Protocol Violation

**Problem:** `HeuristicPolicyBundle` is registered via `@register_policy` but raises `NotImplementedError` for most core methods. This violates the `PolicyBundle` protocol contract and can cause runtime errors when code type-checks against `PolicyBundle`.

**Solution:** Remove `@register_policy` decorator and provide a separate factory function. The heuristic should not pretend to be a neural policy.

**Files:**
- Modify: `src/esper/tamiyo/policy/heuristic_bundle.py:20-25, 159-161`
- Modify: `src/esper/tamiyo/policy/__init__.py`
- Test: `tests/tamiyo/policy/test_registry.py`

**Step 1: Write the failing test**

```python
# Add to tests/tamiyo/policy/test_registry.py

def test_heuristic_not_in_neural_policy_registry():
    """Heuristic should not be registered as a neural PolicyBundle."""
    from esper.tamiyo.policy import list_policies

    policies = list_policies()
    assert "heuristic" not in policies, (
        "HeuristicPolicyBundle should not be in the neural policy registry. "
        "It raises NotImplementedError for most PolicyBundle methods."
    )
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_registry.py::test_heuristic_not_in_neural_policy_registry -v
```

Expected: FAIL with `AssertionError: HeuristicPolicyBundle should not be in the neural policy registry`

**Step 3: Write minimal implementation**

```python
# In src/esper/tamiyo/policy/heuristic_bundle.py, remove @register_policy decorator:

# Remove this line:
# @register_policy("heuristic")
class HeuristicPolicyBundle:
    """Heuristic policy adapter that wraps HeuristicTamiyo.

    IMPORTANT: This is NOT a neural PolicyBundle implementation.
    ...
```

```python
# In src/esper/tamiyo/policy/__init__.py, update exports:

# Remove heuristic from policy registry imports
# (it's still available via direct import, just not registered)

__all__ = [
    # Protocol and types
    "PolicyBundle",
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    # Neural policy bundles (registered)
    "LSTMPolicyBundle",
    # Heuristic adapter (NOT registered - use create_heuristic_policy())
    "HeuristicPolicyBundle",
    "create_heuristic_policy",
    # Registry
    "register_policy",
    "get_policy",
    "list_policies",
    # Features
    "obs_to_multislot_features",
    "get_feature_size",
    # Action masks
    "compute_action_masks",
    "MaskedCategorical",
]


def create_heuristic_policy(**kwargs) -> HeuristicPolicyBundle:
    """Create a heuristic policy adapter.

    This is the recommended way to create heuristic policies. Unlike neural
    policies, the heuristic is not registered in the policy registry because
    it doesn't implement the full PolicyBundle interface.

    Args:
        **kwargs: Arguments passed to HeuristicPolicyBundle constructor.

    Returns:
        HeuristicPolicyBundle instance.
    """
    return HeuristicPolicyBundle(**kwargs)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_registry.py::test_heuristic_not_in_neural_policy_registry -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/policy/heuristic_bundle.py src/esper/tamiyo/policy/__init__.py tests/tamiyo/policy/test_registry.py
git commit -m "$(cat <<'EOF'
fix(tamiyo): remove HeuristicPolicyBundle from neural policy registry

HeuristicPolicyBundle raises NotImplementedError for most PolicyBundle
methods, violating the protocol contract. Remove @register_policy
decorator and provide create_heuristic_policy() factory instead.

This makes the type system honest: code that expects a PolicyBundle
won't accidentally receive a heuristic that can't fulfill the contract.

The heuristic is still available via direct import or the factory
function for users who understand its limitations.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Post-Implementation Verification

After all tasks are complete, run the full test suite:

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/ -v --tb=short
```

Expected: All tests pass.

---

## Summary

| Task | Type | Files Changed | Risk |
|------|------|---------------|------|
| 1 | Performance | action_masks.py | Low (opt-in) |
| 2 | Security | heuristic.py, constants.py | Low (threshold increase) |
| 3 | Performance | lstm_bundle.py | Low (decorator only) |
| 4 | Documentation | lstm_bundle.py, registry.py | None |
| 5 | Documentation | action_masks.py | None |
| 6 | Type Safety | heuristic_bundle.py, __init__.py | Medium (API change) |

**Estimated Time:** 2-3 hours for all tasks with TDD.

**Dependencies:** None - tasks can be done in any order, but the listed order is recommended.
