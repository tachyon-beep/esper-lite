# PR #35 Remediation Plan: Complete API Migration

**Date:** 2025-12-25
**Status:** UPDATED - Awaiting final Code Reviewer approval (PyTorch GO, DRL GO)
**Triggered by:** Automated code review feedback on PR #35

## Executive Summary

The Tamiyo Neural Policy Migration (PR #35) introduced correct architectural changes but left incomplete migrations in test files. Three specialist agents identified **4 categories of issues** totaling **~66 locations** requiring fixes.

---

## Issue Categories

### Category A: Private Attribute Access Violations (3 locations)
**Severity:** Medium
**Risk:** Abstraction leak - code breaks if PolicyBundle implementation changes

| File | Line | Current | Fix |
|------|------|---------|-----|
| `src/esper/simic/training/vectorized.py` | 391 | `agent.policy._network` | `agent.policy.network` |
| `src/esper/simic/training/vectorized.py` | 392 | `agent.policy._network` | `agent.policy.network` |
| `src/esper/simic/training/vectorized.py` | 2880 | `agent.policy._network` | `agent.policy.network` |

### Category B: PolicyBundle Compile Abstraction (1 location)
**Severity:** Medium
**Risk:** Abstraction leak - assumes `_network` backing store

| File | Line | Issue |
|------|------|-------|
| `src/esper/tamiyo/policy/factory.py` | 109 | Direct `policy._network = torch.compile(...)` assignment |

**Fix:** Add `.compile()` method to PolicyBundle protocol and implementations.

### Category C: slot_config Propagation Bugs (31+ locations)
**Severity:** High
**Risk:** Silent training corruption from slot ordering mismatch

| File | Occurrences | Pattern |
|------|-------------|---------|
| `tests/integration/test_slot_consistency.py` | 9 | `num_slots=config.num_slots` |
| `tests/integration/test_ppo_integration.py` | 13 | `num_slots=3` without slot_config |
| `tests/simic/training/test_policy_group.py` | 8 | `num_slots=3` without slot_config |
| `tests/conftest.py` | 1 | `num_slots=3` without slot_config |

### Category D: PPOAgent API Migration Gaps (32 locations)
**Severity:** CRITICAL (tests broken at runtime)
**Risk:** Tests fail with TypeError/AttributeError

**D1: Old Constructor Signature (8 locations)**

| File | Occurrences | Parameters Used |
|------|-------------|-----------------|
| `tests/integration/test_dynamic_slots_e2e.py` | 6 | `state_dim=`, `compile_network=` |
| `tests/integration/test_vectorized_factored.py` | 2 | `state_dim=`, `compile_network=` |

**D2: agent.network Access Pattern (24 locations)**

*Verified via grep 2025-12-25 - other files already migrated to agent.policy.network*

| File | Occurrences |
|------|-------------|
| `tests/integration/test_ppo_integration.py` | 15 |
| `tests/integration/test_dynamic_slots_e2e.py` | 8 |
| `tests/integration/test_vectorized_factored.py` | 1 |

---

## Fix Plan

### Phase 1: Production Code Fixes

#### Task 1.1: Fix vectorized.py Private Attribute Access
**File:** `src/esper/simic/training/vectorized.py`

```python
# Line 391 - BEFORE
gradient_stats = collect_per_layer_gradients(agent.policy._network)
# Line 391 - AFTER
gradient_stats = collect_per_layer_gradients(agent.policy.network)

# Line 392 - BEFORE
stability_report = check_numerical_stability(agent.policy._network)
# Line 392 - AFTER
stability_report = check_numerical_stability(agent.policy.network)

# Line 2880 - BEFORE
ppo_grad_norm = compute_grad_norm_surrogate(agent.policy._network)
# Line 2880 - AFTER
ppo_grad_norm = compute_grad_norm_surrogate(agent.policy.network)
```

#### Task 1.2: Add PolicyBundle.compile() Method
**Files:**
- `src/esper/tamiyo/policy/protocol.py`
- `src/esper/tamiyo/policy/lstm_bundle.py`
- `src/esper/tamiyo/policy/heuristic_bundle.py`
- `src/esper/tamiyo/policy/factory.py`

**Protocol addition:**
```python
def compile(
    self,
    mode: str = "default",
    dynamic: bool = True,
) -> None:
    """Compile the underlying network with torch.compile.

    Must be called AFTER device placement (.to(device)).

    Args:
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune", "off")
        dynamic: Enable dynamic shapes for varying batch/sequence lengths
    """
    ...

@property
def is_compiled(self) -> bool:
    """True if the network has been compiled with torch.compile."""
    ...
```

**LSTMPolicyBundle implementation:**
```python
def compile(self, mode: str = "default", dynamic: bool = True) -> None:
    if mode == "off" or self.is_compiled:
        return
    self._network = torch.compile(self._network, mode=mode, dynamic=dynamic)

@property
def is_compiled(self) -> bool:
    # hasattr AUTHORIZED by John on 2025-12-25 00:00:00 UTC
    # Justification: torch.compile detection - OptimizedModule has _orig_mod attr
    return hasattr(self._network, '_orig_mod')
```

**HeuristicPolicyBundle implementation:**
```python
def compile(self, mode: str = "default", dynamic: bool = True) -> None:
    pass  # No network to compile

@property
def is_compiled(self) -> bool:
    return False  # No network
```

**Factory update:**
```python
# BEFORE (line 102-114)
if compile_mode != "off":
    policy._network = torch.compile(
        policy.network,
        mode=compile_mode,
        dynamic=True,
    )

# AFTER
policy.compile(mode=compile_mode, dynamic=True)
```

---

### Phase 2: Test File Fixes - Critical (Broken Tests)

#### Task 2.1: Fix test_dynamic_slots_e2e.py
**File:** `tests/integration/test_dynamic_slots_e2e.py`

**Constructor fixes (6 locations):**
Pattern to apply at lines 37, 83, 131, 363, 433, 493:
```python
# BEFORE
agent = PPOAgent(
    state_dim=state_dim,
    slot_config=config,
    device="cpu",
    compile_network=False,
    num_envs=1,
    max_steps_per_env=10,
)

# AFTER
policy = create_policy(
    policy_type="lstm",
    state_dim=state_dim,
    slot_config=config,
    device="cpu",
    compile_mode="off",
)
agent = PPOAgent(
    policy=policy,
    slot_config=config,
    device="cpu",
    num_envs=1,
    max_steps_per_env=10,
)
```

**Network access fixes (8 locations):**
```python
# BEFORE
agent.network.get_action(...)
agent.network.num_slots

# AFTER
agent.policy.network.get_action(...)
agent.policy.network.num_slots
```

#### Task 2.2: Fix test_vectorized_factored.py
**File:** `tests/integration/test_vectorized_factored.py`

Same pattern as Task 2.1 for 2 constructor locations and 1 network access.

---

### Phase 3: Test File Fixes - slot_config Propagation

#### Task 3.1: Fix test_slot_consistency.py (9 locations)
**File:** `tests/integration/test_slot_consistency.py`

Lines 39, 54, 70, 97, 139, 255, 293, 310, 329:
```python
# BEFORE
create_policy(..., num_slots=config.num_slots, ...)

# AFTER
create_policy(..., slot_config=config, ...)
```

#### Task 3.2: Fix test_ppo_integration.py (13 locations)
**File:** `tests/integration/test_ppo_integration.py`

```python
# BEFORE
policy = create_policy(
    policy_type="lstm",
    state_dim=len(features),
    num_slots=3,
    device="cpu",
    compile_mode="off",
)

# AFTER
slot_config = SlotConfig.default()  # or appropriate config
policy = create_policy(
    policy_type="lstm",
    state_dim=len(features),
    slot_config=slot_config,
    device="cpu",
    compile_mode="off",
)
```

#### Task 3.3: Fix test_policy_group.py (8 locations)
**File:** `tests/simic/training/test_policy_group.py`

Same pattern as Task 3.2.

#### Task 3.4: Fix conftest.py (1 location)
**File:** `tests/conftest.py`

Same pattern as Task 3.2.

---

### Phase 4: Test File Fixes - Network Access Pattern

#### Task 4.1: Fix agent.network → agent.policy.network

**Global replacement across all test files:**

| Pattern | Replacement |
|---------|-------------|
| `agent.network.get_action(` | `agent.policy.network.get_action(` |
| `agent.network.forward(` | `agent.policy.network.forward(` |
| `agent.network.eval()` | `agent.policy.network.eval()` |
| `agent.network.train()` | `agent.policy.network.train()` |
| `agent.network.num_slots` | `agent.policy.network.num_slots` |
| `agent.network.parameters()` | `agent.policy.network.parameters()` |

**Files to update (verified 2025-12-25):**
- `tests/integration/test_ppo_integration.py` (15 locations)
- `tests/integration/test_dynamic_slots_e2e.py` (8 locations)
- `tests/integration/test_vectorized_factored.py` (1 location)

---

## Verification Plan

### V1: Run affected test files individually
```bash
PYTHONPATH=src uv run pytest tests/integration/test_dynamic_slots_e2e.py -v
PYTHONPATH=src uv run pytest tests/integration/test_vectorized_factored.py -v
PYTHONPATH=src uv run pytest tests/integration/test_ppo_integration.py -v
PYTHONPATH=src uv run pytest tests/integration/test_slot_consistency.py -v
PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -v
PYTHONPATH=src uv run pytest tests/simic/test_ppo_checkpoint.py -v
PYTHONPATH=src uv run pytest tests/stress/test_slot_scaling.py -v
```

### V2: Run full test suite
```bash
PYTHONPATH=src uv run pytest tests/ -v --tb=short
```

### V3: Verify no remaining violations
```bash
# Check for remaining _network accesses outside implementations
grep -r "policy\._network" src/esper/simic/ --include="*.py"

# Check for remaining num_slots= patterns without slot_config
grep -r "num_slots=.*\.num_slots" tests/ --include="*.py"

# Check for remaining agent.network patterns
grep -r "agent\.network\." tests/ --include="*.py"
```

### V4: Verify no old PPOAgent constructor signatures
```bash
# Check for old constructor parameters that no longer exist
grep -r "PPOAgent(" tests/ --include="*.py" -A 5 | grep -E "(state_dim=|compile_network=|action_dim=|hidden_dim=)"
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing tests | Run each test file after fixing before proceeding |
| Missing locations | Grep verification in V3 catches stragglers |
| torch.compile regression | Test with compile_mode="default" explicitly |
| slot_config ordering bugs | Defensive assertion in PPOAgent catches at runtime |

---

## Commit Strategy

1. **Commit 1:** Phase 1 production code fixes (vectorized.py + PolicyBundle.compile)
2. **Commit 2:** Phase 2 critical test fixes (broken tests)
3. **Commit 3:** Phase 3+4 remaining test fixes
4. **Commit 4:** Any stragglers found in verification

---

## Specialist Review Required

This plan requires GO/NO-GO from:
- [ ] **PyTorch Specialist:** Validate Phase 1 torch.compile abstraction approach
- [ ] **DRL Specialist:** Validate Phase 3 slot_config propagation fixes are complete
- [ ] **Code Reviewer:** Validate Phase 2+4 API migration is comprehensive

---

## Appendix: Files Modified

### Production Code
- `src/esper/simic/training/vectorized.py`
- `src/esper/tamiyo/policy/protocol.py`
- `src/esper/tamiyo/policy/lstm_bundle.py`
- `src/esper/tamiyo/policy/heuristic_bundle.py`
- `src/esper/tamiyo/policy/factory.py`

### Test Files
- `tests/integration/test_dynamic_slots_e2e.py`
- `tests/integration/test_vectorized_factored.py`
- `tests/integration/test_ppo_integration.py`
- `tests/integration/test_slot_consistency.py`
- `tests/simic/test_ppo.py`
- `tests/simic/test_ppo_checkpoint.py`
- `tests/simic/training/test_policy_group.py`
- `tests/stress/test_slot_scaling.py`
- `tests/conftest.py`
