# Tamiyo Neural Policy Migration

**Date:** 2025-12-24
**Status:** Planned
**Branch:** TBD (user will create for easy rollback)

## Overview

Wire up Tamiyo as the actual brain by migrating neural policy ownership from Simic to Tamiyo via the PolicyBundle abstraction.

### Goals

1. **Tamiyo owns the neural policy interface** - Network architecture lives in Tamiyo
2. **PPO training uses PolicyBundle** - Not direct network access
3. **Clean domain boundary** - Tamiyo = decisions, Simic = learning mechanics
4. **Preserve all existing functionality** - No behavioral changes

### Non-Goals

- Adding new policy types (HeuristicBundle is deferred)
- Changing the network architecture
- Modifying the PPO algorithm

## Current State (Problem)

```
┌─────────────────────────────────────────────────────────────┐
│                         SIMIC                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PPOAgent                                             │   │
│  │   └── self.network = FactoredRecurrentActorCritic() │   │
│  │       └── 8 action heads, LSTM, value head          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         TAMIYO                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PolicyBundle (Protocol) ←── Never used in production │   │
│  │ LSTMPolicyBundle        ←── Never instantiated       │   │
│  │ PolicyRegistry          ←── Never called             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

The PolicyBundle abstraction is fully implemented but disconnected. PPOAgent bypasses it entirely by creating the network directly.

## Target State

```
┌─────────────────────────────────────────────────────────────┐
│                         TAMIYO                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ networks/factored_lstm.py                            │   │
│  │   └── FactoredRecurrentActorCritic                   │   │
│  │                                                       │   │
│  │ policy/lstm_bundle.py                                │   │
│  │   └── LSTMPolicyBundle(PolicyBundle)                 │   │
│  │       └── wraps FactoredRecurrentActorCritic         │   │
│  │                                                       │   │
│  │ policy/factory.py                                    │   │
│  │   └── create_policy(type, config) → PolicyBundle     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ PolicyBundle
┌─────────────────────────────────────────────────────────────┐
│                         SIMIC                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PPOAgent                                             │   │
│  │   └── self.policy: PolicyBundle  ← Injected         │   │
│  │       └── calls evaluate_actions(), get_action()    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Task Dependency Chain

```
Task 1 ──► Task 2 ──► Task 3 ──► Task 4 ──► Task 5 ──► Task 6 ──► Task 7 ──► Task 8
  │           │           │           │           │           │
  │           │           │           │           │           └── Tests pass
  │           │           │           │           └── vectorized.py uses factory
  │           │           │           └── PPOAgent uses PolicyBundle
  │           │           └── Factory can create policies
  │           └── LSTMPolicyBundle imports from Tamiyo
  └── Network exists in Tamiyo
```

**IMPORTANT:** Tasks are strictly sequential. Do NOT parallelize.

---

## Implementation Tasks

### Task 1: Create Tamiyo Networks Directory

**Depends on:** Nothing (first task)

Create `src/esper/tamiyo/networks/__init__.py` and move `FactoredRecurrentActorCritic` from Simic.

**What moves to Tamiyo:**
```python
# These ALL move from simic/agent/network.py to tamiyo/networks/factored_lstm.py:

@dataclass(frozen=True, slots=True)
class GetActionResult:  # Used by lstm_bundle.py, vectorized.py
    ...

class _ForwardOutput(TypedDict):  # Internal to network
    ...

class FactoredRecurrentActorCritic(nn.Module):  # The main network
    ...
```

**What stays in Simic (delete the rest):**
```python
# simic/agent/network.py becomes EMPTY or is deleted entirely
# All contents move to Tamiyo
```

**Files:**
- CREATE: `src/esper/tamiyo/networks/__init__.py`
  ```python
  from esper.tamiyo.networks.factored_lstm import (
      FactoredRecurrentActorCritic,
      GetActionResult,
  )

  __all__ = ["FactoredRecurrentActorCritic", "GetActionResult"]
  ```
- CREATE: `src/esper/tamiyo/networks/factored_lstm.py` (move entire contents from `simic/agent/network.py`)
- DELETE: `src/esper/simic/agent/network.py` (entire file)
- MODIFY: `src/esper/simic/agent/__init__.py` (remove network imports if any)

**Test:** `uv run pytest tests/tamiyo/networks/ -v`

### Task 2: Update LSTMPolicyBundle Import

**Depends on:** Task 1 (network must exist in Tamiyo)

Change the import in LSTMPolicyBundle from Simic to local Tamiyo networks.

**Files:**
- MODIFY: `src/esper/tamiyo/policy/lstm_bundle.py`
  - FROM: `from esper.simic.agent.network import FactoredRecurrentActorCritic`
  - TO: `from esper.tamiyo.networks import FactoredRecurrentActorCritic`

**Test:** `uv run pytest tests/tamiyo/policy/test_lstm_bundle.py -v`

### Task 3: Create Policy Factory

**Depends on:** Task 2 (LSTMPolicyBundle must import from Tamiyo)

Add `create_policy()` factory function that uses the registry.

**Files:**
- CREATE: `src/esper/tamiyo/policy/factory.py`
- MODIFY: `src/esper/tamiyo/policy/__init__.py` (export factory)

```python
# factory.py
def create_policy(
    policy_type: str,
    state_dim: int,
    num_slots: int,
    device: torch.device,
    compile_mode: str = "off",  # Handle torch.compile at factory level
    **kwargs
) -> PolicyBundle:
    """Create a policy from the registry."""
    from esper.tamiyo.policy.registry import get_policy_class
    policy_cls = get_policy_class(policy_type)
    bundle = policy_cls(state_dim=state_dim, num_slots=num_slots, **kwargs).to(device)

    # Compile the inner network, not the wrapper (PyTorch expert recommendation)
    if compile_mode != "off":
        bundle._network = torch.compile(
            bundle._network,
            mode=compile_mode,
            dynamic=True
        )
    return bundle
```

**Test:** `uv run pytest tests/tamiyo/policy/test_factory.py -v`

### Task 4: Refactor PPOAgent Constructor

**Depends on:** Task 3 (factory must exist)

Change PPOAgent to accept PolicyBundle via dependency injection.

**Files:**
- MODIFY: `src/esper/simic/agent/ppo.py`

**Complete list of changes (18 references across 13 locations):**

| Line | Current Code | New Code |
|------|--------------|----------|
| 311 | `self.network = FactoredRecurrentActorCritic(...)` | DELETE (policy injected) |
| 325 | `self.network.lstm_hidden_dim` | `self.policy.network.lstm_hidden_dim` |
| 327 | `self.network.lstm_hidden_dim` | `self.policy.network.lstm_hidden_dim` |
| 329 | `self.network.num_slots` | `self.policy.network.num_slots` |
| 331 | `self.network.num_slots` | `self.policy.network.num_slots` |
| 350-351 | `self.network = torch.compile(self.network, ...)` | DELETE (factory handles compile) |
| 372-385 | `self._base_network.<head>.parameters()` | `self.policy.network.<head>.parameters()` |
| 395 | `self.network.parameters()` | `self.policy.network.parameters()` |
| 409-411 | `_base_network` property checks `_orig_mod` | DELETE property (use `self.policy.network`) |
| 554 | `self.network.evaluate_actions(...)` | `self.policy.evaluate_actions(...)` + EvalResult handling |
| 724 | `base_net = self._base_network` | `base_net = self.policy.network` |
| 745 | `nn.utils.clip_grad_norm_(self.network.parameters(), ...)` | `nn.utils.clip_grad_norm_(self.policy.network.parameters(), ...)` |
| 802 | `base_net = self._base_network` | `base_net = self.policy.network` |

**Constructor signature change:**

```python
# Before
def __init__(
    self,
    state_dim: int | None = None,
    action_dim: int = 7,  # REMOVE - unused legacy param
    hidden_dim: int = 256,  # REMOVE - unused legacy param
    ...
    compile_mode: str = "default",  # REMOVE - factory handles this
    slot_config: "SlotConfig | None" = None,
):
    self.network = FactoredRecurrentActorCritic(...)

# After
def __init__(
    self,
    policy: "PolicyBundle",  # NEW - injected policy
    slot_config: "SlotConfig | None" = None,
    ...
    # Remove: state_dim, action_dim, hidden_dim, compile_mode, lstm_hidden_dim
):
    self.policy = policy
```

**Delete the `_base_network` property entirely** - use `self.policy.network` instead.

**Test:** `uv run pytest tests/simic/test_ppo.py -v`

### Task 5: Update Vectorized Training Entry Point

**Depends on:** Task 4 (PPOAgent must accept PolicyBundle)

Wire vectorized.py to create PolicyBundle from Tamiyo before creating PPOAgent.

**Files:**
- MODIFY: `src/esper/simic/training/vectorized.py`
  - Update `GetActionResult` import: FROM `esper.simic.agent.network` TO `esper.tamiyo.networks`
  - Add `create_policy` import from `esper.tamiyo.policy`
  - Create policy before PPOAgent

**Before:**
```python
from esper.simic.agent.network import GetActionResult

agent = PPOAgent(
    state_dim=obs_dim,
    slot_config=slot_config,
    compile_mode=compile_mode,
    ...
)
```

**After:**
```python
from esper.tamiyo.networks import GetActionResult
from esper.tamiyo.policy import create_policy

policy = create_policy(
    policy_type=config.tamiyo.policy_type,  # "lstm" default
    state_dim=obs_dim,
    num_slots=slot_config.num_slots,
    device=device,
    compile_mode=compile_mode,  # Factory handles compilation
)
agent = PPOAgent(policy=policy, slot_config=slot_config, ...)
```

**Test:** `uv run pytest tests/simic/training/test_vectorized.py -v`

### Task 6: Update All Test Fixtures

**Depends on:** Task 5 (vectorized.py must be updated)

Fix all tests that create PPOAgent directly.

**Explicit file list (9 files):**

| File | What to Change |
|------|----------------|
| `tests/conftest.py` | Update `small_ppo_model_deterministic` fixture to use `create_policy()` |
| `tests/simic/test_ppo.py` | All direct PPOAgent instantiations |
| `tests/simic/test_ppo_checkpoint.py` | Checkpoint tests with PPOAgent |
| `tests/simic/training/test_policy_group.py` | Policy group tests |
| `tests/integration/test_ppo_integration.py` | Integration tests |
| `tests/integration/test_vectorized_factored.py` | Vectorized training tests |
| `tests/integration/test_dynamic_slots_e2e.py` | Dynamic slot tests |
| `tests/integration/test_slot_consistency.py` | Slot consistency tests |
| `tests/stress/test_slot_scaling.py` | Stress tests |

**Pattern for updating tests:**

```python
# Before
from esper.simic.agent import PPOAgent

agent = PPOAgent(state_dim=30, slot_config=slot_config, ...)

# After
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy

policy = create_policy(
    policy_type="lstm",
    state_dim=30,
    num_slots=slot_config.num_slots,
    device="cpu",  # Tests typically use CPU
)
agent = PPOAgent(policy=policy, slot_config=slot_config, ...)
```

**Test:** `uv run pytest tests/ -v --ignore=tests/karn`

### Task 7: Remove Dead Code

**Depends on:** Task 6 (all tests must pass first)

Delete the TODO-marked dead code that is now superseded.

**Files:**
- VERIFY DELETED: `src/esper/simic/agent/network.py` (should be gone from Task 1)
- MODIFY: `src/esper/tamiyo/policy/protocol.py` (remove `# TODO: [DEAD CODE]` comment - now ALIVE)
- MODIFY: `src/esper/tamiyo/policy/lstm_bundle.py` (remove `# TODO: [DEAD CODE]` comment - now ALIVE)
- MODIFY: `src/esper/tamiyo/policy/registry.py` (remove `# TODO: [DEAD CODE]` comment - now ALIVE)
- MODIFY: `src/esper/simic/agent/__init__.py` (remove any stale network imports)

**Test:** `uv run pytest tests/ -v --ignore=tests/karn`

### Task 8: Integration Verification

**Depends on:** Task 7 (all cleanup complete)

Full test suite and manual verification.

```bash
# Full test suite
uv run pytest tests/ -v

# Training smoke test (short run)
uv run python -m esper.scripts.train ppo --episodes 10 --num-envs 2

# Verify no imports from old location
grep -r "from esper.simic.agent.network" src/ tests/
# Should return NO results
```

**Verification checklist:**
- [ ] All tests pass
- [ ] Training runs without errors
- [ ] No imports from `esper.simic.agent.network` remain
- [ ] `FactoredRecurrentActorCritic` lives in `tamiyo/networks/`
- [ ] PPOAgent uses `self.policy` (not `self.network`)
- [ ] torch.compile handled by factory

## Rollback Plan

All changes are in a dedicated branch. If issues arise:

```bash
git checkout main
git branch -D feat/tamiyo-neural-migration  # Delete branch
```

No database migrations or external dependencies affected.

## Success Criteria

1. All tests pass
2. Training produces identical behavior (same loss curves, same seed actions)
3. `PPOAgent` no longer imports from `simic/agent/network.py`
4. `FactoredRecurrentActorCritic` lives in `tamiyo/networks/`
5. PolicyBundle protocol is used in production code path

## Architecture Diagram (Final)

```
┌───────────────────────────────────────────────────────────────────┐
│                              TAMIYO                               │
│                         "The Brain"                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ networks/                                                    │ │
│  │   └── factored_lstm.py                                      │ │
│  │       └── FactoredRecurrentActorCritic (8 heads, LSTM)      │ │
│  │                                                              │ │
│  │ policy/                                                      │ │
│  │   ├── protocol.py     → PolicyBundle interface              │ │
│  │   ├── lstm_bundle.py  → LSTMPolicyBundle (wraps network)    │ │
│  │   ├── registry.py     → @register_policy decorator          │ │
│  │   └── factory.py      → create_policy() entry point         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                                │
                                │ PolicyBundle interface
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                              SIMIC                                │
│                       "The Evolution"                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ agent/ppo.py                                                 │ │
│  │   └── PPOAgent(policy: PolicyBundle)                        │ │
│  │       └── Uses policy.evaluate_actions(), policy.get_action()│ │
│  │                                                              │ │
│  │ training/vectorized.py                                       │ │
│  │   └── Creates PolicyBundle from Tamiyo                      │ │
│  │   └── Passes to PPOAgent                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

## Expert Review Findings

### DRL Expert Review (2025-12-24)

**Verdict:** Architecturally sound, preserves PPO correctness. Three critical issues identified.

#### Critical Issues

1. **Interface Signature Mismatch** - `evaluate_actions()` uses different patterns:
   - Network: 8 individual mask kwargs (`slot_mask=...`, `blueprint_mask=...`)
   - PolicyBundle: Single dict (`masks: dict[str, Tensor]`)
   - **Action:** Task 4 must update all call sites

2. **Return Type Mismatch** - PPOAgent expects 4-tuple, PolicyBundle returns EvalResult:
   ```python
   # Current (will break)
   log_probs, values, entropy, _ = self.network.evaluate_actions(...)

   # Required change
   result = self.policy.evaluate_actions(...)
   log_probs, values, entropy = result.log_prob, result.value, result.entropy
   ```

3. **Optimizer Parameter Binding** - Must update to access via PolicyBundle:
   ```python
   # Current
   self.optimizer = torch.optim.Adam(self.network.parameters(), ...)

   # Required change
   self.optimizer = torch.optim.Adam(self.policy.network.parameters(), ...)
   ```

#### Verified Correct

- Gradient flow preserved (EvalResult is pure delegation, no `.detach()`)
- Hidden state threading correct (initial_hidden for rollouts, stored initial for training)
- On-policy integrity maintained (rollout uses get_action, training recomputes with evaluate_actions)
- Value estimation unchanged

### PyTorch Expert Review (2025-12-24)

**Verdict:** Design is sound for torch.compile, AMP, and gradient flow. One high-priority issue.

#### High Priority

1. **torch.compile Target** - PolicyBundle is not nn.Module, so compile must target inner network:
   ```python
   # WRONG - PolicyBundle is not nn.Module
   self.policy = torch.compile(self.policy, ...)

   # CORRECT - Compile the inner network
   self.policy._network = torch.compile(self.policy.network, mode=..., dynamic=True)
   ```
   **Action:** Add compile logic to factory (Task 3) or document in Task 4

#### Verified Correct

- Gradient flow: Pure delegation without detach/no_grad
- AMP compatibility: dtype property correctly delegates to parameters
- Device management: `.to(device)` at factory level handles placement
- LSTM hidden state: `initial_hidden()` correctly uses inference_mode for rollouts

#### Recommendations

- Add `fullgraph=True` test to catch graph breaks
- Consider having network return protocol-compatible types directly (eliminates wrapper conversion overhead)
- Add `parameters()` method to PolicyBundle for cleaner optimizer construction

---

## Updated Implementation Details

### Task 3 Update: Factory with torch.compile

```python
# factory.py (updated)
def create_policy(
    policy_type: str,
    state_dim: int,
    num_slots: int,
    device: torch.device,
    compile_mode: str = "off",  # NEW: Handle compile at factory level
    **kwargs
) -> PolicyBundle:
    """Create a policy from the registry."""
    from esper.tamiyo.policy.registry import get_policy_class
    policy_cls = get_policy_class(policy_type)
    bundle = policy_cls(state_dim=state_dim, num_slots=num_slots, **kwargs).to(device)

    # Compile the inner network, not the wrapper
    if compile_mode != "off":
        bundle._network = torch.compile(
            bundle._network,
            mode=compile_mode,
            dynamic=True
        )
    return bundle
```

### Task 4 Update: Critical Call Site Changes

**1. Mask consolidation in `update()`:**
```python
# Before (8 kwargs)
log_probs, values, entropy, _ = self.network.evaluate_actions(
    data["states"],
    actions,
    slot_mask=data["slot_masks"],
    blueprint_mask=data["blueprint_masks"],
    style_mask=data["style_masks"],
    # ... 5 more masks
    hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
)

# After (dict + EvalResult)
masks = {
    "slot": data["slot_masks"],
    "blueprint": data["blueprint_masks"],
    "style": data["style_masks"],
    "tempo": data["tempo_masks"],
    "alpha_target": data["alpha_target_masks"],
    "alpha_speed": data["alpha_speed_masks"],
    "alpha_curve": data["alpha_curve_masks"],
    "op": data["op_masks"],
}
result = self.policy.evaluate_actions(
    data["states"],
    actions,
    masks,
    hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
)
log_probs = result.log_prob
values = result.value
entropy = result.entropy
```

**2. Optimizer construction:**
```python
# Before
self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

# After
self.optimizer = torch.optim.Adam(self.policy.network.parameters(), lr=learning_rate)
```

**3. Buffer validation:**
```python
# Before
assert self.buffer.lstm_hidden_dim == self.network.lstm_hidden_dim

# After (add properties to PolicyBundle or access via network)
assert self.buffer.lstm_hidden_dim == self.policy.network.lstm_hidden_dim
```

**4. Checkpoint save/load:**
```python
# Before
save_dict = {'network_state_dict': self._base_network.state_dict(), ...}

# After (PolicyBundle.state_dict() handles compile wrapper)
save_dict = {'network_state_dict': self.policy.state_dict(), ...}
```

### Additional Test Coverage

```bash
# Gradient flow through bundle
uv run pytest tests/tamiyo/policy/test_lstm_bundle.py::test_evaluate_actions_gradient_flow -v

# Checkpoint round-trip
uv run pytest tests/simic/test_ppo.py::test_checkpoint_with_policy_bundle -v

# Graph break detection
uv run pytest tests/tamiyo/policy/test_lstm_bundle.py::test_no_graph_breaks -v
```

---

## Risk Assessment (Post-Review)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Gradient flow broken | Low | Critical | EvalResult is pure delegation |
| Call site mismatch | **High** | Critical | Task 4 changes are mandatory |
| Optimizer binding | **High** | Critical | Must update parameter access |
| torch.compile wrong target | Medium | High | Factory handles compile |
| Checkpoint breakage | Medium | High | Use PolicyBundle.state_dict() |
| Hidden state contamination | Low | High | Documented correctly |

---

## Notes

- The HeuristicPolicyBundle remains unimplemented (raises NotImplementedError) - this is intentional for now
- Future work: Add MLPPolicyBundle for ablation studies
- Future work: Policy hot-swapping during training
