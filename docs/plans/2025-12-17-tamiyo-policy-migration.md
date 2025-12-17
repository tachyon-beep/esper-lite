# Tamiyo Policy Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the policy network from Simic to Tamiyo, making Tamiyo the "brain" with hotswappable PolicyBundle implementations.

**Architecture:** Create a `tamiyo/policy/` subpackage with a `PolicyBundle` protocol. Move feature extraction and action masking from `simic/control/` to `tamiyo/policy/`. Wrap the existing LSTM network and heuristic as PolicyBundle implementations. Update Simic to consume Tamiyo's PolicyBundle interface.

**Tech Stack:** Python 3.11+, PyTorch 2.x, typing.Protocol, dataclasses

**Reference:** See `docs/plans/2025-12-17-tamiyo-policy-migration-design.md` for full design rationale and expert review feedback.

---

## Phase 1: Create PolicyBundle Protocol and Infrastructure

### Task 1.1: Add MASKED_LOGIT_VALUE constant to Leyline

**Files:**
- Modify: `src/esper/leyline/__init__.py`
- Test: `tests/leyline/test_constants.py`

**Step 1: Write the failing test**

Create test file if it doesn't exist:

```python
# tests/leyline/test_constants.py
"""Tests for leyline constants."""

import torch
from esper.leyline import MASKED_LOGIT_VALUE


def test_masked_logit_value_exists():
    """MASKED_LOGIT_VALUE should be exported from leyline."""
    assert MASKED_LOGIT_VALUE == -1e4


def test_masked_logit_value_safe_for_fp16():
    """MASKED_LOGIT_VALUE should not overflow in FP16 softmax."""
    logits = torch.tensor([0.0, MASKED_LOGIT_VALUE], dtype=torch.float16)
    probs = torch.softmax(logits, dim=0)
    assert probs[0].item() > 0.99  # Masked action should have ~0 probability
    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/leyline/test_constants.py -v
```

Expected: FAIL with `ImportError: cannot import name 'MASKED_LOGIT_VALUE'`

**Step 3: Add the constant to leyline**

```python
# In src/esper/leyline/__init__.py, add near other constants:

# Action masking constant - safe for FP16/BF16, avoids softmax overflow
# Used by MaskedCategorical to zero out invalid action probabilities
MASKED_LOGIT_VALUE: float = -1e4
```

Also add to `__all__`:
```python
__all__ = [
    # ... existing exports ...
    "MASKED_LOGIT_VALUE",
]
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/leyline/test_constants.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/__init__.py tests/leyline/test_constants.py
git commit -m "feat(leyline): add MASKED_LOGIT_VALUE constant for action masking"
```

---

### Task 1.2: Create PolicyBundle protocol

**Files:**
- Create: `src/esper/tamiyo/policy/__init__.py`
- Create: `src/esper/tamiyo/policy/protocol.py`
- Create: `src/esper/tamiyo/policy/types.py`
- Test: `tests/tamiyo/policy/test_protocol.py`

**Step 1: Create the policy directory**

```bash
mkdir -p src/esper/tamiyo/policy
mkdir -p tests/tamiyo/policy
touch tests/tamiyo/policy/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/tamiyo/policy/test_protocol.py
"""Tests for PolicyBundle protocol."""

import torch
from typing import runtime_checkable

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult


def test_policy_bundle_is_runtime_checkable():
    """PolicyBundle should be runtime_checkable for registration validation."""
    assert hasattr(PolicyBundle, '__protocol_attrs__')


def test_policy_bundle_protocol_methods():
    """PolicyBundle should define all required methods."""
    required_methods = [
        'process_signals',
        'get_action',
        'forward',
        'evaluate_actions',
        'get_q_values',
        'sync_from',
        'get_value',
        'initial_hidden',
        'state_dict',
        'load_state_dict',
        'to',
        'enable_gradient_checkpointing',
    ]
    for method in required_methods:
        assert hasattr(PolicyBundle, method), f"Missing method: {method}"


def test_policy_bundle_protocol_properties():
    """PolicyBundle should define all required properties."""
    required_properties = [
        'is_recurrent',
        'supports_off_policy',
        'device',
        'dtype',
    ]
    for prop in required_properties:
        assert hasattr(PolicyBundle, prop), f"Missing property: {prop}"


def test_action_result_dataclass():
    """ActionResult should hold action selection results."""
    result = ActionResult(
        action={'slot': 0, 'blueprint': 1, 'blend': 0, 'op': 2},
        log_prob={'slot': torch.tensor(-0.5), 'blueprint': torch.tensor(-1.0),
                  'blend': torch.tensor(-0.3), 'op': torch.tensor(-0.8)},
        value=torch.tensor(0.5),
        hidden=(torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)),
    )
    assert result.action['slot'] == 0
    assert result.value.item() == 0.5


def test_eval_result_dataclass():
    """EvalResult should hold action evaluation results."""
    result = EvalResult(
        log_prob={'op': torch.tensor(-0.5)},
        value=torch.tensor(0.3),
        entropy={'op': torch.tensor(0.7)},
        hidden=None,
    )
    assert result.entropy['op'].item() == 0.7
```

**Step 3: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_protocol.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'esper.tamiyo.policy'`

**Step 4: Create the types module**

```python
# src/esper/tamiyo/policy/types.py
"""Type definitions for PolicyBundle interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result from policy action selection.

    Attributes:
        action: Dict mapping head names to selected action indices
        log_prob: Dict mapping head names to log probabilities
        value: State value estimate
        hidden: New hidden state tuple (h, c) or None for stateless policies
    """
    action: dict[str, int]
    log_prob: dict[str, torch.Tensor]
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result from policy action evaluation (for PPO training).

    Attributes:
        log_prob: Dict mapping head names to log probabilities
        value: State value estimate
        entropy: Dict mapping head names to entropy values
        hidden: New hidden state or None
    """
    log_prob: dict[str, torch.Tensor]
    value: torch.Tensor
    entropy: dict[str, torch.Tensor]
    hidden: tuple[torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Result from policy forward pass (distribution params without sampling).

    Used by off-policy algorithms (SAC) that need to compute log_prob
    of sampled actions.

    Attributes:
        logits: Dict mapping head names to raw logits
        value: State value estimate
        hidden: New hidden state or None
    """
    logits: dict[str, torch.Tensor]
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor] | None


__all__ = [
    "ActionResult",
    "EvalResult",
    "ForwardResult",
]
```

**Step 5: Create the protocol module**

```python
# src/esper/tamiyo/policy/protocol.py
"""PolicyBundle protocol for swappable Tamiyo policy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from esper.leyline import TrainingSignals
    from esper.leyline.factored_actions import FactoredAction
    from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult


@runtime_checkable
class PolicyBundle(Protocol):
    """Interface for swappable Tamiyo policy implementations.

    Tamiyo is the "brain" of Esper - she makes strategic decisions about
    seed lifecycle (germinate, fossilize, cull, wait). Different PolicyBundle
    implementations provide different decision-making strategies:

    - LSTMPolicyBundle: Recurrent neural policy with temporal memory
    - MLPPolicyBundle: Stateless feedforward policy (simpler baseline)
    - HeuristicPolicyBundle: Rule-based expert system (for ablations)

    ## Design Rationale (from expert review)

    - Protocol over ABC: Avoids MRO conflicts with nn.Module inheritance
    - runtime_checkable: Enables validation at policy registration time
    - Explicit state_dict: Required for checkpoint compatibility
    - Device management: Essential for multi-GPU and distributed training

    ## Adding a New Policy

    1. Create `tamiyo/policy/my_bundle.py`
    2. Implement the PolicyBundle protocol
    3. Decorate with @register_policy("my_policy")
    4. Add to config: {"tamiyo": {"policy": "my_policy", ...}}

    ## On-Policy vs Off-Policy

    Policies declare their capabilities via `supports_off_policy`.
    On-policy algorithms (PPO) use `evaluate_actions()`.
    Off-policy algorithms (SAC) use `get_q_values()` and `forward()`.

    ## Recurrent Policies

    Recurrent policies (LSTM) maintain hidden state across steps.
    They must implement `initial_hidden()` and set `is_recurrent = True`.
    Simic handles hidden state threading during rollout collection.

    ## torch.compile Guidance

    Compile the inner nn.Module, NOT the PolicyBundle wrapper.
    Keep torch.compile() calls in Simic (training infrastructure).
    """

    # === Observation Processing ===
    def process_signals(self, signals: "TrainingSignals") -> torch.Tensor:
        """Convert TrainingSignals to policy-specific features."""
        ...

    # === Action Selection (both paradigms) ===
    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> "ActionResult":
        """Select action given observations.

        Uses inference_mode internally - returned tensors are non-differentiable.
        """
        ...

    # === Forward (for off-policy) ===
    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> "ForwardResult":
        """Compute action distribution parameters without sampling.

        Required for:
        - SAC: Computing log_prob of sampled actions for entropy bonus
        - TD3: Getting deterministic action for target policy
        - Offline RL: Computing action distribution for OOD detection

        Returns:
            ForwardResult with logits per head, value, new_hidden
        """
        ...

    # === On-Policy (PPO/A2C) ===
    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: "FactoredAction",
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> "EvalResult":
        """Evaluate actions for PPO update.

        Must be called with gradient tracking enabled (not in inference_mode).
        """
        ...

    # === Off-Policy (SAC/TD3) ===
    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Twin Q-values for off-policy critic.

        Returns (Q1, Q2) for clipped double-Q learning.
        Raises NotImplementedError if supports_off_policy is False.
        """
        ...

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Polyak averaging update from source policy (for target networks).

        target = tau * source + (1 - tau) * target

        Required for SAC/TD3 target network updates.
        """
        ...

    # === Value Estimation ===
    def get_value(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """State value estimate for baseline."""
        ...

    # === Recurrent State ===
    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Initial hidden state for recurrent policies (None if stateless).

        Should be called with inference_mode for efficiency.
        """
        ...

    # === Serialization ===
    def state_dict(self) -> dict[str, Any]:
        """Return policy state for checkpointing."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load policy state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
            strict: If True, keys must match exactly. If False, allows partial loading.
        """
        ...

    # === Device Management ===
    @property
    def device(self) -> torch.device:
        """Device where policy parameters reside."""
        ...

    def to(self, device: torch.device | str) -> "PolicyBundle":
        """Move policy to specified device. Returns self for chaining."""
        ...

    # === Introspection ===
    @property
    def is_recurrent(self) -> bool:
        """True if policy maintains hidden state across steps."""
        ...

    @property
    def supports_off_policy(self) -> bool:
        """True if policy supports off-policy algorithms (SAC/TD3).

        If False, get_q_values() and sync_from() raise NotImplementedError.
        """
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Data type of policy parameters (for AMP compatibility)."""
        ...

    # === Optional: Gradient Checkpointing ===
    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory efficiency.

        Optional - policies that don't support this should no-op.
        Primarily useful for Transformer-based policies.
        """
        ...


__all__ = ["PolicyBundle"]
```

**Step 6: Create the package init**

```python
# src/esper/tamiyo/policy/__init__.py
"""Tamiyo Policy - Hotswappable policy implementations.

This subpackage contains the PolicyBundle protocol and implementations:
- protocol.py: PolicyBundle interface definition
- types.py: ActionResult, EvalResult, ForwardResult dataclasses
- registry.py: Policy registration and factory (Task 1.3)
- lstm_bundle.py: LSTM-based recurrent policy (Phase 2)
- heuristic_bundle.py: Rule-based heuristic (Phase 3)
"""

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult

__all__ = [
    "PolicyBundle",
    "ActionResult",
    "EvalResult",
    "ForwardResult",
]
```

**Step 7: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_protocol.py -v
```

Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/tamiyo/policy/ tests/tamiyo/policy/
git commit -m "feat(tamiyo): add PolicyBundle protocol and types"
```

---

### Task 1.3: Create policy registry

**Files:**
- Create: `src/esper/tamiyo/policy/registry.py`
- Modify: `src/esper/tamiyo/policy/__init__.py`
- Test: `tests/tamiyo/policy/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/tamiyo/policy/test_registry.py
"""Tests for policy registry."""

import pytest
import torch

from esper.tamiyo.policy import PolicyBundle
from esper.tamiyo.policy.registry import (
    register_policy,
    get_policy,
    list_policies,
    clear_registry,
)
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult


class MockPolicyBundle:
    """Minimal PolicyBundle implementation for testing."""

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._device = torch.device("cpu")

    def process_signals(self, signals):
        return torch.zeros(1, 10)

    def get_action(self, features, masks, hidden=None, deterministic=False):
        return ActionResult(
            action={'op': 0},
            log_prob={'op': torch.tensor(0.0)},
            value=torch.tensor(0.0),
            hidden=None,
        )

    def forward(self, features, masks, hidden=None):
        return ForwardResult(
            logits={'op': torch.zeros(4)},
            value=torch.tensor(0.0),
            hidden=None,
        )

    def evaluate_actions(self, features, actions, masks, hidden=None):
        return EvalResult(
            log_prob={'op': torch.tensor(0.0)},
            value=torch.tensor(0.0),
            entropy={'op': torch.tensor(0.5)},
            hidden=None,
        )

    def get_q_values(self, features, action):
        raise NotImplementedError("Mock does not support off-policy")

    def sync_from(self, source, tau=0.005):
        raise NotImplementedError("Mock does not support off-policy")

    def get_value(self, features, hidden=None):
        return torch.tensor(0.0)

    def initial_hidden(self, batch_size):
        return None

    def state_dict(self):
        return {"hidden_dim": self.hidden_dim}

    def load_state_dict(self, state_dict, strict=True):
        self.hidden_dim = state_dict["hidden_dim"]

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    @property
    def is_recurrent(self):
        return False

    @property
    def supports_off_policy(self):
        return False

    @property
    def dtype(self):
        return torch.float32

    def enable_gradient_checkpointing(self, enabled=True):
        pass


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def test_register_policy_decorator():
    """@register_policy should add class to registry."""
    @register_policy("mock")
    class TestPolicy(MockPolicyBundle):
        pass

    assert "mock" in list_policies()


def test_get_policy_returns_instance():
    """get_policy should instantiate registered policy."""
    @register_policy("mock")
    class TestPolicy(MockPolicyBundle):
        pass

    policy = get_policy("mock", {})
    assert isinstance(policy, TestPolicy)


def test_get_policy_passes_config():
    """get_policy should pass config to constructor."""
    @register_policy("configurable")
    class ConfigurablePolicy(MockPolicyBundle):
        def __init__(self, hidden_dim: int = 64):
            super().__init__(hidden_dim)

    policy = get_policy("configurable", {"hidden_dim": 128})
    assert policy.hidden_dim == 128


def test_get_policy_unknown_raises():
    """get_policy should raise for unknown policy names."""
    with pytest.raises(ValueError, match="Unknown policy"):
        get_policy("nonexistent", {})


def test_list_policies():
    """list_policies should return all registered policy names."""
    @register_policy("policy_a")
    class PolicyA(MockPolicyBundle):
        pass

    @register_policy("policy_b")
    class PolicyB(MockPolicyBundle):
        pass

    policies = list_policies()
    assert "policy_a" in policies
    assert "policy_b" in policies


def test_register_policy_validates_protocol():
    """@register_policy should validate PolicyBundle protocol compliance."""
    # This class is missing required methods
    class InvalidPolicy:
        pass

    with pytest.raises(TypeError, match="does not implement PolicyBundle"):
        register_policy("invalid")(InvalidPolicy)
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'esper.tamiyo.policy.registry'`

**Step 3: Create the registry module**

```python
# src/esper/tamiyo/policy/registry.py
"""Policy registry for hotswappable Tamiyo policies."""

from __future__ import annotations

from typing import Type, TypeVar

from esper.tamiyo.policy.protocol import PolicyBundle

T = TypeVar("T", bound=PolicyBundle)

_REGISTRY: dict[str, Type[PolicyBundle]] = {}


def register_policy(name: str):
    """Decorator to register a PolicyBundle implementation.

    Usage:
        @register_policy("lstm")
        class LSTMPolicyBundle:
            ...

    Args:
        name: Unique name for this policy (used in config files)

    Returns:
        Decorator that registers the class

    Raises:
        TypeError: If the class doesn't implement PolicyBundle protocol
        ValueError: If name is already registered
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Validate protocol compliance
        if not isinstance(cls, type):
            raise TypeError(f"{cls} is not a class")

        # Check for required methods
        required_methods = [
            'process_signals', 'get_action', 'forward', 'evaluate_actions',
            'get_q_values', 'sync_from', 'get_value', 'initial_hidden',
            'state_dict', 'load_state_dict', 'to', 'enable_gradient_checkpointing',
        ]
        required_properties = ['is_recurrent', 'supports_off_policy', 'device', 'dtype']

        missing_methods = [m for m in required_methods if not hasattr(cls, m)]
        missing_props = [p for p in required_properties if not hasattr(cls, p)]

        if missing_methods or missing_props:
            missing = missing_methods + missing_props
            raise TypeError(
                f"{cls.__name__} does not implement PolicyBundle protocol. "
                f"Missing: {', '.join(missing)}"
            )

        if name in _REGISTRY:
            raise ValueError(f"Policy '{name}' is already registered")

        _REGISTRY[name] = cls
        return cls

    return decorator


def get_policy(name: str, config: dict) -> PolicyBundle:
    """Factory function to instantiate a policy by name.

    Args:
        name: Registered policy name (e.g., "lstm", "heuristic")
        config: Configuration dict passed to policy constructor

    Returns:
        Instantiated PolicyBundle

    Raises:
        ValueError: If name is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none registered)"
        raise ValueError(f"Unknown policy: '{name}'. Available: {available}")

    return _REGISTRY[name](**config)


def list_policies() -> list[str]:
    """List all registered policy names."""
    return list(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered policies. For testing only."""
    _REGISTRY.clear()


__all__ = [
    "register_policy",
    "get_policy",
    "list_policies",
    "clear_registry",
]
```

**Step 4: Update package init**

```python
# src/esper/tamiyo/policy/__init__.py
"""Tamiyo Policy - Hotswappable policy implementations.

This subpackage contains the PolicyBundle protocol and implementations:
- protocol.py: PolicyBundle interface definition
- types.py: ActionResult, EvalResult, ForwardResult dataclasses
- registry.py: Policy registration and factory
- lstm_bundle.py: LSTM-based recurrent policy (Phase 2)
- heuristic_bundle.py: Rule-based heuristic (Phase 3)
"""

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.tamiyo.policy.registry import (
    register_policy,
    get_policy,
    list_policies,
)

__all__ = [
    "PolicyBundle",
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    "register_policy",
    "get_policy",
    "list_policies",
]
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_registry.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/tamiyo/policy/registry.py src/esper/tamiyo/policy/__init__.py tests/tamiyo/policy/test_registry.py
git commit -m "feat(tamiyo): add policy registry for hotswappable policies"
```

---

## Phase 2: Move Feature Extraction and Action Masking

### Task 2.1: Move features.py to Tamiyo

**Files:**
- Create: `src/esper/tamiyo/policy/features.py` (copy from simic/control/)
- Modify: `src/esper/simic/control/features.py` (make it re-export from tamiyo)
- Test: Existing tests should still pass

**Step 1: Copy features.py to tamiyo/policy/**

```bash
cp src/esper/simic/control/features.py src/esper/tamiyo/policy/features.py
```

**Step 2: Run existing tests to verify they still pass**

```bash
PYTHONPATH=src uv run pytest tests/simic/control/ -v -k feature
```

Expected: PASS (tests still import from simic.control)

**Step 3: Update simic/control/features.py to re-export from tamiyo**

Replace the entire content of `src/esper/simic/control/features.py` with:

```python
"""Simic Features - Re-exports from tamiyo.policy.features.

DEPRECATED: Import from esper.tamiyo.policy.features instead.
This module exists for backwards compatibility during migration.
"""

# Re-export everything from tamiyo
from esper.tamiyo.policy.features import (
    safe,
    obs_to_multislot_features,
    MULTISLOT_FEATURE_SIZE,
    get_feature_size,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
    TaskConfig,
)

__all__ = [
    "safe",
    "obs_to_multislot_features",
    "MULTISLOT_FEATURE_SIZE",
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
]
```

**Step 4: Run tests again to verify backwards compatibility**

```bash
PYTHONPATH=src uv run pytest tests/simic/control/ -v -k feature
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/policy/features.py src/esper/simic/control/features.py
git commit -m "refactor(tamiyo): move features.py to tamiyo/policy, re-export from simic"
```

---

### Task 2.2: Move action_masks.py to Tamiyo

**Files:**
- Create: `src/esper/tamiyo/policy/action_masks.py` (copy from simic/control/)
- Modify: `src/esper/simic/control/action_masks.py` (make it re-export from tamiyo)
- Modify: `src/esper/tamiyo/policy/action_masks.py` (use MASKED_LOGIT_VALUE from leyline)
- Test: Existing tests should still pass

**Step 1: Copy action_masks.py to tamiyo/policy/**

```bash
cp src/esper/simic/control/action_masks.py src/esper/tamiyo/policy/action_masks.py
```

**Step 2: Update tamiyo/policy/action_masks.py to use MASKED_LOGIT_VALUE from leyline**

In `src/esper/tamiyo/policy/action_masks.py`, find the MaskedCategorical class and update:

```python
# Near the top, add import:
from esper.leyline import MASKED_LOGIT_VALUE

# In MaskedCategorical.__init__, replace:
#     finfo_min = torch.finfo(logits.dtype).min
#     mask_value = torch.tensor(
#         max(finfo_min, -1e4),
#         device=logits.device,
#         dtype=logits.dtype,
#     )

# With:
        mask_value = torch.tensor(
            MASKED_LOGIT_VALUE,
            device=logits.device,
            dtype=logits.dtype,
        )
```

**Step 3: Run existing tests to verify they still pass**

```bash
PYTHONPATH=src uv run pytest tests/simic/control/ -v -k mask
```

Expected: PASS

**Step 4: Update simic/control/action_masks.py to re-export from tamiyo**

Replace the entire content with:

```python
"""Action Masking - Re-exports from tamiyo.policy.action_masks.

DEPRECATED: Import from esper.tamiyo.policy.action_masks instead.
This module exists for backwards compatibility during migration.
"""

# Re-export everything from tamiyo
from esper.tamiyo.policy.action_masks import (
    MaskSeedInfo,
    build_slot_states,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
    MaskedCategorical,
    InvalidStateMachineError,
)

__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "slot_id_to_index",
    "MaskedCategorical",
    "InvalidStateMachineError",
]
```

**Step 5: Run tests again to verify backwards compatibility**

```bash
PYTHONPATH=src uv run pytest tests/simic/control/ -v -k mask
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/tamiyo/policy/action_masks.py src/esper/simic/control/action_masks.py
git commit -m "refactor(tamiyo): move action_masks.py to tamiyo/policy, use MASKED_LOGIT_VALUE"
```

---

### Task 2.3: Update tamiyo/policy/__init__.py exports

**Files:**
- Modify: `src/esper/tamiyo/policy/__init__.py`

**Step 1: Update the package init to export features and masks**

```python
# src/esper/tamiyo/policy/__init__.py
"""Tamiyo Policy - Hotswappable policy implementations.

This subpackage contains the PolicyBundle protocol and implementations:
- protocol.py: PolicyBundle interface definition
- types.py: ActionResult, EvalResult, ForwardResult dataclasses
- registry.py: Policy registration and factory
- features.py: Feature extraction for observations
- action_masks.py: Action masking for valid actions
- lstm_bundle.py: LSTM-based recurrent policy (Phase 3)
- heuristic_bundle.py: Rule-based heuristic (Phase 4)
"""

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.tamiyo.policy.registry import (
    register_policy,
    get_policy,
    list_policies,
)
from esper.tamiyo.policy.features import (
    obs_to_multislot_features,
    get_feature_size,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
    TaskConfig,
)
from esper.tamiyo.policy.action_masks import (
    compute_action_masks,
    compute_batch_masks,
    MaskedCategorical,
)

__all__ = [
    # Protocol
    "PolicyBundle",
    # Types
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    # Registry
    "register_policy",
    "get_policy",
    "list_policies",
    # Features
    "obs_to_multislot_features",
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
    # Action Masks
    "compute_action_masks",
    "compute_batch_masks",
    "MaskedCategorical",
]
```

**Step 2: Verify imports work**

```bash
PYTHONPATH=src python -c "from esper.tamiyo.policy import PolicyBundle, compute_action_masks, obs_to_multislot_features; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/tamiyo/policy/__init__.py
git commit -m "feat(tamiyo): export features and action_masks from tamiyo.policy"
```

---

## Phase 3: Create LSTM PolicyBundle

### Task 3.1: Create LSTMPolicyBundle wrapping existing network

**Files:**
- Create: `src/esper/tamiyo/policy/lstm_bundle.py`
- Modify: `src/esper/tamiyo/policy/__init__.py`
- Test: `tests/tamiyo/policy/test_lstm_bundle.py`

**Step 1: Write the failing test**

```python
# tests/tamiyo/policy/test_lstm_bundle.py
"""Tests for LSTMPolicyBundle."""

import pytest
import torch

from esper.tamiyo.policy import PolicyBundle, get_policy, list_policies
from esper.tamiyo.policy.lstm_bundle import LSTMPolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.leyline.slot_config import SlotConfig


@pytest.fixture
def slot_config():
    return SlotConfig.default()


@pytest.fixture
def lstm_bundle(slot_config):
    return LSTMPolicyBundle(
        feature_dim=50,
        hidden_dim=64,
        num_lstm_layers=1,
        slot_config=slot_config,
    )


def test_lstm_bundle_registered():
    """LSTMPolicyBundle should be registered as 'lstm'."""
    assert "lstm" in list_policies()


def test_lstm_bundle_is_recurrent(lstm_bundle):
    """LSTMPolicyBundle should be recurrent."""
    assert lstm_bundle.is_recurrent is True


def test_lstm_bundle_does_not_support_off_policy(lstm_bundle):
    """LSTMPolicyBundle should not support off-policy (for now)."""
    assert lstm_bundle.supports_off_policy is False


def test_lstm_bundle_initial_hidden(lstm_bundle):
    """initial_hidden should return LSTM hidden state tuple."""
    hidden = lstm_bundle.initial_hidden(batch_size=4)
    assert hidden is not None
    h, c = hidden
    assert h.shape == (1, 4, 64)  # (num_layers, batch, hidden_dim)
    assert c.shape == (1, 4, 64)


def test_lstm_bundle_get_action(lstm_bundle, slot_config):
    """get_action should return ActionResult."""
    features = torch.randn(1, 50)
    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, 5, dtype=torch.bool),
        "blend": torch.ones(1, 3, dtype=torch.bool),
        "op": torch.ones(1, 4, dtype=torch.bool),
    }
    hidden = lstm_bundle.initial_hidden(batch_size=1)

    result = lstm_bundle.get_action(features, masks, hidden)

    assert isinstance(result, ActionResult)
    assert "op" in result.action
    assert "op" in result.log_prob
    assert result.hidden is not None


def test_lstm_bundle_evaluate_actions(lstm_bundle, slot_config):
    """evaluate_actions should return EvalResult with gradients."""
    features = torch.randn(1, 10, 50)  # (batch, seq_len, features)
    masks = {
        "slot": torch.ones(1, 10, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, 10, 5, dtype=torch.bool),
        "blend": torch.ones(1, 10, 3, dtype=torch.bool),
        "op": torch.ones(1, 10, 4, dtype=torch.bool),
    }
    actions = {
        "slot": torch.zeros(1, 10, dtype=torch.long),
        "blueprint": torch.zeros(1, 10, dtype=torch.long),
        "blend": torch.zeros(1, 10, dtype=torch.long),
        "op": torch.zeros(1, 10, dtype=torch.long),
    }
    hidden = lstm_bundle.initial_hidden(batch_size=1)

    result = lstm_bundle.evaluate_actions(features, actions, masks, hidden)

    assert isinstance(result, EvalResult)
    assert result.log_prob["op"].requires_grad
    assert result.value.requires_grad


def test_lstm_bundle_state_dict(lstm_bundle):
    """state_dict should return serializable dict."""
    state = lstm_bundle.state_dict()
    assert isinstance(state, dict)
    # Should have network weights
    assert any("weight" in k or "bias" in k for k in state.keys())


def test_lstm_bundle_load_state_dict(lstm_bundle):
    """load_state_dict should restore policy state."""
    state = lstm_bundle.state_dict()
    lstm_bundle.load_state_dict(state)
    # Should not raise


def test_lstm_bundle_device_management(lstm_bundle):
    """to() should move policy to device."""
    assert lstm_bundle.device == torch.device("cpu")
    # Note: Can't test CUDA without GPU, but method should exist
    lstm_bundle.to("cpu")
    assert lstm_bundle.device == torch.device("cpu")


def test_lstm_bundle_dtype(lstm_bundle):
    """dtype should return network parameter dtype."""
    assert lstm_bundle.dtype == torch.float32


def test_get_policy_lstm(slot_config):
    """get_policy('lstm', ...) should return LSTMPolicyBundle."""
    policy = get_policy("lstm", {
        "feature_dim": 50,
        "hidden_dim": 64,
        "slot_config": slot_config,
    })
    assert isinstance(policy, LSTMPolicyBundle)
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'esper.tamiyo.policy.lstm_bundle'`

**Step 3: Create lstm_bundle.py**

This is a large file - it wraps the existing FactoredRecurrentActorCritic:

```python
# src/esper/tamiyo/policy/lstm_bundle.py
"""LSTM-based PolicyBundle implementation.

Wraps FactoredRecurrentActorCritic as a PolicyBundle for the registry.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch import nn

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.registry import register_policy
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.simic.agent.tamiyo_network import FactoredRecurrentActorCritic
from esper.leyline.slot_config import SlotConfig
from esper.leyline.factored_actions import HEAD_NAMES

if TYPE_CHECKING:
    from esper.leyline import TrainingSignals


@register_policy("lstm")
class LSTMPolicyBundle:
    """LSTM-based recurrent policy for seed lifecycle control.

    This PolicyBundle wraps FactoredRecurrentActorCritic, providing the
    standard PolicyBundle interface while delegating to the existing
    well-tested network implementation.

    Attributes:
        feature_dim: Input feature dimension
        hidden_dim: LSTM hidden state dimension
        slot_config: Slot configuration for action masking
    """

    def __init__(
        self,
        feature_dim: int = 50,
        hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        slot_config: SlotConfig | None = None,
        dropout: float = 0.0,
    ):
        """Initialize LSTM policy bundle.

        Args:
            feature_dim: Observation feature dimension
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            slot_config: Slot configuration (defaults to SlotConfig.default())
            dropout: Dropout rate for LSTM
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.slot_config = slot_config or SlotConfig.default()

        # Create the network
        self._network = FactoredRecurrentActorCritic(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            slot_config=self.slot_config,
            dropout=dropout,
        )

    # === Observation Processing ===

    def process_signals(self, signals: "TrainingSignals") -> torch.Tensor:
        """Convert TrainingSignals to feature tensor.

        Delegates to the signals_to_features helper which uses
        obs_to_multislot_features internally.
        """
        from esper.simic.agent.ppo import signals_to_features
        return signals_to_features(signals, self.slot_config)

    # === Action Selection ===

    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> ActionResult:
        """Select action using the LSTM network.

        This method wraps the network's get_action method.
        """
        result = self._network.get_action(
            features,
            masks,
            hidden,
            deterministic=deterministic,
        )

        return ActionResult(
            action=result["actions"],
            log_prob=result["log_probs"],
            value=result["value"],
            hidden=result["hidden"],
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ForwardResult:
        """Forward pass returning distribution parameters.

        For off-policy algorithms that need to compute log_prob separately.
        """
        output = self._network.forward(features, hidden, masks)

        return ForwardResult(
            logits={head: output[f"{head}_logits"] for head in HEAD_NAMES},
            value=output["value"],
            hidden=output["hidden"],
        )

    # === On-Policy (PPO) ===

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> EvalResult:
        """Evaluate actions for PPO training.

        This method wraps the network's evaluate_actions method.
        """
        log_probs, values, entropies, new_hidden = self._network.evaluate_actions(
            features,
            actions,
            masks,
            hidden,
        )

        return EvalResult(
            log_prob=log_probs,
            value=values,
            entropy=entropies,
            hidden=new_hidden,
        )

    # === Off-Policy (not supported for LSTM) ===

    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported for LSTM policy."""
        raise NotImplementedError(
            "LSTMPolicyBundle does not support off-policy algorithms. "
            "Use MLPPolicyBundle with SAC/TD3 instead."
        )

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Not supported for LSTM policy."""
        raise NotImplementedError(
            "LSTMPolicyBundle does not support target network updates. "
            "Use MLPPolicyBundle with SAC/TD3 instead."
        )

    # === Value Estimation ===

    def get_value(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Get state value estimate."""
        return self._network.get_value(features, hidden)

    # === Recurrent State ===

    @torch.inference_mode()
    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden state."""
        return self._network.get_initial_hidden(batch_size, self.device)

    # === Serialization ===

    def state_dict(self) -> dict[str, Any]:
        """Return network state dict."""
        # Handle torch.compile wrapper
        base = getattr(self._network, '_orig_mod', self._network)
        return base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load network state dict."""
        base = getattr(self._network, '_orig_mod', self._network)
        base.load_state_dict(state_dict, strict=strict)

    # === Device Management ===

    @property
    def device(self) -> torch.device:
        """Get device of network parameters."""
        return next(self._network.parameters()).device

    def to(self, device: torch.device | str) -> "LSTMPolicyBundle":
        """Move network to device."""
        self._network = self._network.to(device)
        return self

    # === Introspection ===

    @property
    def is_recurrent(self) -> bool:
        """LSTM is recurrent."""
        return True

    @property
    def supports_off_policy(self) -> bool:
        """LSTM does not support off-policy (needs R2D2 machinery)."""
        return False

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of network parameters."""
        return next(self._network.parameters()).dtype

    # === Optional: Gradient Checkpointing ===

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """No-op for LSTM (gradient checkpointing not beneficial)."""
        pass

    # === Network Access (for Simic's torch.compile) ===

    @property
    def network(self) -> nn.Module:
        """Access underlying network for torch.compile."""
        return self._network


__all__ = ["LSTMPolicyBundle"]
```

**Step 4: Update tamiyo/policy/__init__.py to import lstm_bundle**

Add at the end of `__init__.py`:

```python
# Import to trigger registration (must be after registry is defined)
from esper.tamiyo.policy import lstm_bundle as _lstm_bundle  # noqa: F401
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_lstm_bundle.py -v
```

Expected: PASS (or minor fixes needed based on actual network interface)

**Step 6: Commit**

```bash
git add src/esper/tamiyo/policy/lstm_bundle.py src/esper/tamiyo/policy/__init__.py tests/tamiyo/policy/test_lstm_bundle.py
git commit -m "feat(tamiyo): add LSTMPolicyBundle wrapping FactoredRecurrentActorCritic"
```

---

## Phase 4: Create Heuristic PolicyBundle

### Task 4.1: Create HeuristicPolicyBundle

**Files:**
- Create: `src/esper/tamiyo/policy/heuristic_bundle.py`
- Modify: `src/esper/tamiyo/policy/__init__.py`
- Test: `tests/tamiyo/policy/test_heuristic_bundle.py`

**Step 1: Write the failing test**

```python
# tests/tamiyo/policy/test_heuristic_bundle.py
"""Tests for HeuristicPolicyBundle."""

import pytest
import torch

from esper.tamiyo.policy import get_policy, list_policies
from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle
from esper.tamiyo.policy.types import ActionResult


def test_heuristic_bundle_registered():
    """HeuristicPolicyBundle should be registered as 'heuristic'."""
    assert "heuristic" in list_policies()


def test_heuristic_bundle_is_not_recurrent():
    """HeuristicPolicyBundle should not be recurrent."""
    bundle = HeuristicPolicyBundle()
    assert bundle.is_recurrent is False


def test_heuristic_bundle_does_not_support_off_policy():
    """HeuristicPolicyBundle should not support off-policy."""
    bundle = HeuristicPolicyBundle()
    assert bundle.supports_off_policy is False


def test_heuristic_bundle_initial_hidden():
    """initial_hidden should return None for stateless heuristic."""
    bundle = HeuristicPolicyBundle()
    assert bundle.initial_hidden(batch_size=4) is None


def test_heuristic_bundle_state_dict():
    """state_dict should return empty dict (no learnable params)."""
    bundle = HeuristicPolicyBundle()
    state = bundle.state_dict()
    assert state == {}


def test_get_policy_heuristic():
    """get_policy('heuristic', ...) should return HeuristicPolicyBundle."""
    policy = get_policy("heuristic", {})
    assert isinstance(policy, HeuristicPolicyBundle)
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_heuristic_bundle.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create heuristic_bundle.py**

```python
# src/esper/tamiyo/policy/heuristic_bundle.py
"""Heuristic PolicyBundle for ablations and debugging.

Wraps HeuristicTamiyo as a PolicyBundle for the registry.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.registry import register_policy
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig

if TYPE_CHECKING:
    from esper.leyline import TrainingSignals


@register_policy("heuristic")
class HeuristicPolicyBundle:
    """Rule-based heuristic policy for ablations and debugging.

    This PolicyBundle wraps HeuristicTamiyo, providing the standard
    PolicyBundle interface for a non-learning baseline.

    Note: This policy does not have learnable parameters. Methods like
    evaluate_actions() and get_value() return dummy values.
    """

    def __init__(
        self,
        config: HeuristicPolicyConfig | None = None,
        topology: str = "cnn",
    ):
        """Initialize heuristic policy bundle.

        Args:
            config: Heuristic policy configuration
            topology: Model topology ("cnn" or "transformer")
        """
        self._heuristic = HeuristicTamiyo(config, topology)
        self._topology = topology

    # === Observation Processing ===

    def process_signals(self, signals: "TrainingSignals") -> torch.Tensor:
        """Convert TrainingSignals to feature tensor.

        For heuristic, we just return a dummy tensor since the
        heuristic uses the signals directly.
        """
        # Return dummy features - heuristic uses signals directly
        return torch.zeros(1, 1)

    # === Action Selection ===

    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> ActionResult:
        """Not directly usable - heuristic needs TrainingSignals.

        Use decide_from_signals() instead.
        """
        raise NotImplementedError(
            "HeuristicPolicyBundle.get_action() is not supported. "
            "Use the training loop's heuristic path instead."
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ForwardResult:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no forward pass")

    # === On-Policy ===

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> EvalResult:
        """Not supported for heuristic (no learnable parameters)."""
        raise NotImplementedError("Heuristic has no learnable parameters")

    # === Off-Policy ===

    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no Q-values")

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no target network")

    # === Value Estimation ===

    def get_value(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no value function")

    # === Recurrent State ===

    def initial_hidden(self, batch_size: int) -> None:
        """Heuristic is stateless."""
        return None

    # === Serialization ===

    def state_dict(self) -> dict[str, Any]:
        """Heuristic has no learnable state."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Heuristic has no learnable state."""
        pass

    # === Device Management ===

    @property
    def device(self) -> torch.device:
        """Heuristic runs on CPU."""
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "HeuristicPolicyBundle":
        """No-op for heuristic."""
        return self

    # === Introspection ===

    @property
    def is_recurrent(self) -> bool:
        """Heuristic is stateless."""
        return False

    @property
    def supports_off_policy(self) -> bool:
        """Heuristic doesn't support any training."""
        return False

    @property
    def dtype(self) -> torch.dtype:
        """Return float32 for compatibility."""
        return torch.float32

    # === Optional ===

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """No-op for heuristic."""
        pass

    # === Heuristic-specific ===

    @property
    def heuristic(self) -> HeuristicTamiyo:
        """Access underlying heuristic for direct decision-making."""
        return self._heuristic

    def reset(self) -> None:
        """Reset heuristic state."""
        self._heuristic.reset()


__all__ = ["HeuristicPolicyBundle"]
```

**Step 4: Update tamiyo/policy/__init__.py to import heuristic_bundle**

Add:

```python
from esper.tamiyo.policy import heuristic_bundle as _heuristic_bundle  # noqa: F401
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_heuristic_bundle.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/tamiyo/policy/heuristic_bundle.py src/esper/tamiyo/policy/__init__.py tests/tamiyo/policy/test_heuristic_bundle.py
git commit -m "feat(tamiyo): add HeuristicPolicyBundle for ablations"
```

---

## Phase 5: Rename Simic Files

### Task 5.1: Rename tamiyo_buffer.py to rollout_buffer.py

**Files:**
- Rename: `src/esper/simic/agent/tamiyo_buffer.py` → `src/esper/simic/agent/rollout_buffer.py`
- Modify: `src/esper/simic/agent/__init__.py`
- Modify: All files that import tamiyo_buffer

**Step 1: Find all imports of tamiyo_buffer**

```bash
grep -r "tamiyo_buffer" src/esper --include="*.py"
```

**Step 2: Rename the file**

```bash
git mv src/esper/simic/agent/tamiyo_buffer.py src/esper/simic/agent/rollout_buffer.py
```

**Step 3: Update simic/agent/__init__.py**

Replace `tamiyo_buffer` with `rollout_buffer`:

```python
from .rollout_buffer import (
    TamiyoRolloutStep,
    TamiyoRolloutBuffer,
)
```

**Step 4: Update all other imports**

For each file found in Step 1, update the import.

**Step 5: Run tests to verify nothing broke**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor(simic): rename tamiyo_buffer.py to rollout_buffer.py"
```

---

### Task 5.2: Rename tamiyo_network.py to network.py

**Files:**
- Rename: `src/esper/simic/agent/tamiyo_network.py` → `src/esper/simic/agent/network.py`
- Modify: `src/esper/simic/agent/__init__.py`
- Modify: All files that import tamiyo_network

**Step 1: Find all imports of tamiyo_network**

```bash
grep -r "tamiyo_network" src/esper --include="*.py"
```

**Step 2: Rename the file**

```bash
git mv src/esper/simic/agent/tamiyo_network.py src/esper/simic/agent/network.py
```

**Step 3: Update simic/agent/__init__.py**

Replace `tamiyo_network` with `network`:

```python
from .network import FactoredRecurrentActorCritic
```

**Step 4: Update all other imports**

For each file found in Step 1, update the import.

**Step 5: Update lstm_bundle.py import**

In `src/esper/tamiyo/policy/lstm_bundle.py`, change:

```python
from esper.simic.agent.tamiyo_network import FactoredRecurrentActorCritic
```

to:

```python
from esper.simic.agent.network import FactoredRecurrentActorCritic
```

**Step 6: Run tests to verify nothing broke**

```bash
PYTHONPATH=src uv run pytest tests/simic/ tests/tamiyo/ -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor(simic): rename tamiyo_network.py to network.py"
```

---

## Phase 6: Update Tamiyo Root __init__.py

### Task 6.1: Export policy subpackage from tamiyo root

**Files:**
- Modify: `src/esper/tamiyo/__init__.py`

**Step 1: Update tamiyo/__init__.py**

```python
# src/esper/tamiyo/__init__.py
"""Tamiyo - Strategic decision-making for Esper.

Tamiyo is the "brain" of the Esper system. She observes training signals
and makes strategic decisions about seed lifecycle management.

## Subpackages

- tamiyo.policy: PolicyBundle protocol and implementations (LSTM, Heuristic)

## Key Components

- SignalTracker: Aggregates training metrics into TrainingSignals
- PolicyBundle: Protocol for swappable policy implementations
- get_policy(): Factory function to instantiate policies by name
"""

from esper.tamiyo.decisions import TamiyoDecision
from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import (
    TamiyoPolicy,
    HeuristicPolicyConfig,
    HeuristicTamiyo,
)

# Policy subpackage exports
from esper.tamiyo.policy import (
    PolicyBundle,
    ActionResult,
    EvalResult,
    ForwardResult,
    register_policy,
    get_policy,
    list_policies,
)

__all__ = [
    # Core
    "TamiyoDecision",
    "SignalTracker",
    # Legacy heuristic (kept for backwards compatibility)
    "TamiyoPolicy",
    "HeuristicPolicyConfig",
    "HeuristicTamiyo",
    # Policy interface
    "PolicyBundle",
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    "register_policy",
    "get_policy",
    "list_policies",
]
```

**Step 2: Verify imports work**

```bash
PYTHONPATH=src python -c "from esper.tamiyo import PolicyBundle, get_policy, list_policies; print(list_policies())"
```

Expected: `['lstm', 'heuristic']`

**Step 3: Commit**

```bash
git add src/esper/tamiyo/__init__.py
git commit -m "feat(tamiyo): export policy subpackage from tamiyo root"
```

---

## Phase 7: Integration Testing

### Task 7.1: Run full test suite

**Step 1: Run all tests**

```bash
PYTHONPATH=src uv run pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

Address any import errors or test failures from the refactoring.

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: address test failures from policy migration"
```

---

### Task 7.2: Update integration tests

**Files:**
- Modify: `tests/integration/test_tamiyo_simic.py`

**Step 1: Add test for PolicyBundle integration**

```python
# Add to tests/integration/test_tamiyo_simic.py

def test_policy_bundle_from_registry():
    """Simic should be able to get policy from Tamiyo registry."""
    from esper.tamiyo import get_policy, list_policies
    from esper.leyline.slot_config import SlotConfig

    # Verify both policies are registered
    policies = list_policies()
    assert "lstm" in policies
    assert "heuristic" in policies

    # Verify LSTM policy can be instantiated
    slot_config = SlotConfig.default()
    policy = get_policy("lstm", {
        "feature_dim": 50,
        "hidden_dim": 64,
        "slot_config": slot_config,
    })

    assert policy.is_recurrent is True
    assert policy.supports_off_policy is False
```

**Step 2: Run integration tests**

```bash
PYTHONPATH=src uv run pytest tests/integration/test_tamiyo_simic.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_tamiyo_simic.py
git commit -m "test: add PolicyBundle integration test"
```

---

## Summary

After completing all phases, the codebase will have:

1. **PolicyBundle protocol** in `tamiyo/policy/protocol.py` with full interface
2. **Policy registry** in `tamiyo/policy/registry.py` for config-time swapping
3. **LSTMPolicyBundle** wrapping existing FactoredRecurrentActorCritic
4. **HeuristicPolicyBundle** wrapping existing HeuristicTamiyo
5. **Features and action masks** moved to `tamiyo/policy/`
6. **Backwards-compatible re-exports** from `simic/control/`
7. **Renamed files** for clarity (tamiyo_buffer → rollout_buffer, tamiyo_network → network)

The architecture now matches the biological metaphor:
- **Tamiyo = Brain** - owns PolicyBundle implementations
- **Simic = Evolution** - owns PPO training, uses PolicyBundle interface
