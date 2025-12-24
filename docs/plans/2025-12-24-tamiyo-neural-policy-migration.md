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

## Implementation Tasks

### Task 1: Create Tamiyo Networks Directory

Create `src/esper/tamiyo/networks/__init__.py` and move `FactoredRecurrentActorCritic` from Simic.

**Files:**
- CREATE: `src/esper/tamiyo/networks/__init__.py`
- CREATE: `src/esper/tamiyo/networks/factored_lstm.py` (move from `simic/agent/network.py`)
- MODIFY: `src/esper/simic/agent/network.py` (remove FactoredRecurrentActorCritic, keep imports if needed)

**Test:** `uv run pytest tests/tamiyo/networks/ -v`

### Task 2: Update LSTMPolicyBundle Import

Change the import in LSTMPolicyBundle from Simic to local Tamiyo networks.

**Files:**
- MODIFY: `src/esper/tamiyo/policy/lstm_bundle.py`
  - FROM: `from esper.simic.agent.network import FactoredRecurrentActorCritic`
  - TO: `from esper.tamiyo.networks import FactoredRecurrentActorCritic`

**Test:** `uv run pytest tests/tamiyo/policy/test_lstm_bundle.py -v`

### Task 3: Create Policy Factory

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
    **kwargs
) -> PolicyBundle:
    """Create a policy from the registry."""
    from esper.tamiyo.policy.registry import get_policy_class
    policy_cls = get_policy_class(policy_type)
    return policy_cls(state_dim=state_dim, num_slots=num_slots, **kwargs).to(device)
```

**Test:** `uv run pytest tests/tamiyo/policy/test_factory.py -v`

### Task 4: Refactor PPOAgent Constructor

Change PPOAgent to accept PolicyBundle via dependency injection.

**Files:**
- MODIFY: `src/esper/simic/agent/ppo.py`
  - Remove `FactoredRecurrentActorCritic` import
  - Change `__init__` signature: `policy: PolicyBundle` instead of `state_dim: int`
  - Replace `self.network` with `self.policy`
  - Update all method calls to use PolicyBundle interface

**Before:**
```python
def __init__(self, state_dim: int, slot_config: SlotConfig, ...):
    self.network = FactoredRecurrentActorCritic(state_dim=state_dim, ...)
```

**After:**
```python
def __init__(self, policy: PolicyBundle, slot_config: SlotConfig, ...):
    self.policy = policy
```

**Test:** `uv run pytest tests/simic/test_ppo.py -v`

### Task 5: Update Vectorized Training Entry Point

Wire vectorized.py to create PolicyBundle from Tamiyo before creating PPOAgent.

**Files:**
- MODIFY: `src/esper/simic/training/vectorized.py`

**Before:**
```python
agent = PPOAgent(
    state_dim=obs_dim,
    slot_config=slot_config,
    ...
)
```

**After:**
```python
from esper.tamiyo.policy import create_policy

policy = create_policy(
    policy_type=config.tamiyo.policy_type,  # "lstm" default
    state_dim=obs_dim,
    num_slots=slot_config.num_slots,
    device=device,
)
agent = PPOAgent(policy=policy, slot_config=slot_config, ...)
```

**Test:** `uv run pytest tests/simic/training/test_vectorized.py -v`

### Task 6: Update All Test Fixtures

Fix all tests that create PPOAgent directly.

**Files:**
- MODIFY: `tests/simic/test_ppo.py`
- MODIFY: `tests/simic/test_rollout_buffer.py`
- MODIFY: `tests/integration/test_*.py` (any that use PPOAgent)

**Test:** `uv run pytest tests/ -v --ignore=tests/karn`

### Task 7: Remove Dead Code

Delete the TODO-marked dead code that is now superseded.

**Files:**
- DELETE: `src/esper/simic/agent/network.py` (if empty after move)
- MODIFY: `src/esper/tamiyo/policy/protocol.py` (remove TODO comment)
- MODIFY: `src/esper/tamiyo/policy/lstm_bundle.py` (remove TODO comment)
- MODIFY: `src/esper/tamiyo/policy/registry.py` (remove TODO comment)

### Task 8: Integration Verification

Full test suite and manual verification.

```bash
# Full test suite
uv run pytest tests/ -v

# Training smoke test (short run)
uv run python -m esper.scripts.train ppo --episodes 10 --num-envs 2
```

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

## Notes

- The HeuristicPolicyBundle remains unimplemented (raises NotImplementedError) - this is intentional for now
- Future work: Add MLPPolicyBundle for ablation studies
- Future work: Policy hot-swapping during training
