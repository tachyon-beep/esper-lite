# Comprehensive System Audit Findings

**Date:** 2025-12-05
**Auditors:** DRL Expert + PyTorch Expert + Code Reviewer Agents
**Scope:** simic, tamiyo, kasmina, tolaria subsystems

---

## Audit History

### Phase 1: Initial System Audit (Completed)

The initial audit identified bugs across all subsystems. All fixes have been implemented and verified:

| ID | Subsystem | Issue | Status |
|----|-----------|-------|--------|
| P0-1 | tolaria | CIFAR-10 missing augmentation | **FIXED** |
| P0-2 | tolaria | CIFAR-10 wrong normalization stats | **FIXED** |
| P0-3 | simic | Truncation bootstrap value | **FIXED** |
| P1-1 | simic | train_steps increment in recurrent mode | **FIXED** |
| P1-2 | tamiyo | Stabilization check in germination | **FIXED** |
| P1-3 | simic | Advantage normalization (once per buffer) | **FIXED** |
| P2-1 | kasmina | G5 gate counterfactual alignment | **FIXED** |
| P2-2 | tamiyo | Heuristic counterfactual alignment | **FIXED** |
| P2-3 | tamiyo | Blueprint penalty decay per-epoch | **FIXED** |
| P2-4 | simic | Entropy coefficient defaults unified | **FIXED** |

---

## Phase 2: Kasmina First-Principles Review

Three specialist agents conducted a comprehensive first-principles review of kasmina:
- **DRL Expert** (primacy on RL concerns)
- **Code Reviewer** (primacy on code quality)
- **PyTorch Expert** (primacy on PyTorch concerns)

---

## CRITICAL Findings

### K-CRIT-1: Unauthorized hasattr Usage

**Severity:** CRITICAL (Policy Violation)
**File:** `src/esper/kasmina/slot.py:982`
**Reviewers:** All 3 agreed

**Current code:**
```python
if self.task_config is not None and hasattr(self.task_config, "blending_steps"):
    configured_steps = self.task_config.blending_steps
```

**Problem:** Per CLAUDE.md, `hasattr()` requires explicit operator authorization. This check is also unnecessary because `TaskConfig.blending_steps` is a guaranteed field with default value `5`.

**Fix:**
```python
total_steps = 5
if self.task_config is not None:
    configured_steps = self.task_config.blending_steps
    if isinstance(configured_steps, int) and configured_steps > 0:
        total_steps = configured_steps
```

---

## HIGH Priority Findings

### DRL Expert Domain (Primacy)

#### K-DRL-H1: Counterfactual Contribution Not in Observation Space

**Severity:** HIGH
**File:** `src/esper/leyline/signals.py` (TensorSchema)

**Problem:** The `counterfactual_contribution` field exists in `SeedMetrics` but is NOT included in the observation space exposed to the RL policy. The policy receives `SEED_IMPROVEMENT` (which is `improvement_since_stage_start`, conflating host learning with seed impact).

**Impact:** The policy cannot directly observe whether a seed is helping or hurting. This forces the agent to learn complex indirect mappings, significantly increasing sample complexity.

**Fix:** Add to `TensorSchema`:
```python
SEED_COUNTERFACTUAL = 27  # True causal attribution

# In FastTrainingSignals.to_vector():
seed_counterfactual: float  # 0.0 if alpha=0, else contribution
```

---

#### K-DRL-H2: Ransomware Reward Gaming via Rapid Fossilization

**Severity:** HIGH
**File:** `src/esper/simic/rewards.py:669-780`

**Problem:** The `min(progress, contribution)` bound assumes `progress` is independent of the seed. But during BLENDING/SHADOWING, the seed IS affecting training dynamics, so `progress` may be inflated by dependency creation.

**Impact:** Policy may learn to exploit the causal loop where seeds create dependencies that inflate both contribution and progress metrics.

**Fix:** Add temporal discount based on PROBATIONARY duration:
```python
probation_epochs = seed_info.epochs_in_stage if seed_info.stage == STAGE_PROBATIONARY else 0
legitimacy_discount = min(1.0, probation_epochs / MIN_PROBATION_EPOCHS)
return fossilize_bonus * legitimacy_discount
```

---

#### K-DRL-H3: Host Network State Not Observable

**Severity:** HIGH
**File:** `src/esper/leyline/signals.py`

**Problem:** The host network is continuously learning, but this non-stationarity is invisible to the RL policy. Missing:
- Host learning rate / scheduler state
- Host parameter/gradient norm
- Host improvement rate (cached from alpha=0 periods)

**Impact:** Optimal germination/fossilization policy changes based on host training phase. Without explicit state, the agent must learn brittle implicit timing heuristics.

**Fix:** Add host-specific signals:
```python
HOST_GRAD_NORM = 28
HOST_LEARNING_PHASE = 29  # epoch / max_epochs
HOST_IMPROVEMENT_RATE = 30  # Cached from alpha=0 periods
```

---

#### K-DRL-H4: G5 Gate Fallback Uses Confounded Metric

**Severity:** HIGH
**File:** `src/esper/kasmina/slot.py:455-460`

**Problem:** G5 gate prefers `counterfactual_contribution` but falls back to `total_improvement` which includes all host learning since germination.

**Note:** This was partially addressed in P2-1 fix, but the fallback path should be more restrictive.

**Fix:** Make G5 require counterfactual (it should only be reachable from PROBATIONARY where counterfactual is mandatory):
```python
def _check_g5(self, state: SeedState) -> GateResult:
    contribution = state.metrics.counterfactual_contribution
    if contribution is None:
        return GateResult(
            gate=GateLevel.G5,
            passed=False,
            checks_failed=["counterfactual_not_available"],
        )
    # ... use only counterfactual
```

---

### Code Reviewer Domain (Primacy)

#### K-CODE-H1: Type Annotation Lies About Optional

**Severity:** HIGH
**File:** `src/esper/kasmina/slot.py:175-183`

**Current code:**
```python
telemetry: SeedTelemetry = field(default=None)  # Type says SeedTelemetry, default is None
```

**Problem:** Type annotation says `SeedTelemetry` but default is `None`. This lies to the type checker.

**Fix:**
```python
telemetry: SeedTelemetry | None = field(default=None)
```

---

#### K-CODE-H2: Silent Exception Swallowing in Registry

**Severity:** HIGH
**File:** `src/esper/kasmina/blueprints/registry.py:52-57`

**Current code:**
```python
except Exception:
    pass
```

**Problem:** Catching bare `Exception` and silently passing can mask import errors or actual bugs.

**Fix:**
```python
except (ImportError, AttributeError, KeyError):
    # Cache invalidation is best-effort
    pass
```

---

#### K-CODE-H3: Mutable Class-Level Dictionary

**Severity:** HIGH
**File:** `src/esper/kasmina/blueprints/registry.py:30`

**Problem:** `_blueprints: dict[str, BlueprintSpec] = {}` is mutable class attribute creating global state that can cause test pollution.

**Fix:** Add `reset()` classmethod for test cleanup and document the global state behavior.

---

#### K-CODE-H4: Redundant Condition Check

**Severity:** HIGH
**File:** `src/esper/kasmina/slot.py:635`

**Problem:** `if self.seed is not None:` is redundant immediately after `self.seed = self.seed.to(self.device)`.

**Fix:** Either remove the check (trust the contract) or raise explicit error if None (fail-fast).

---

### PyTorch Expert Domain (Primacy)

#### K-PT-H1: Device Normalization Inconsistent

**Severity:** HIGH
**File:** `src/esper/kasmina/slot.py:631`

**Problem:** `self.device` may be string `"cuda:0"` or `torch.device` object, causing inconsistent behavior in comparisons and cache lookups.

**Fix:** Normalize in `__init__`:
```python
self.device = torch.device(device) if isinstance(device, str) else device
```

---

#### K-PT-H2: Shape Probe Cache Not in state_dict

**Severity:** HIGH
**File:** `src/esper/kasmina/slot.py:532, 561`

**Problem:** `_shape_probe_cache` stores tensors in plain dict, not registered buffers. They won't be moved on `.to(device)` calls.

**Fix:** Clear cache in custom `to()` override or register as buffers.

---

#### K-PT-H3: STE Lacks Defensive Gradient Assertion

**Severity:** HIGH
**File:** `src/esper/kasmina/slot.py:863-864`

**Problem:** STE pattern relies on `seed_features` retaining computation graph. If seed has internal `no_grad()` blocks, gradients won't flow.

**Fix:** Add optional debug assertion:
```python
if DEBUG_STE:
    assert seed_features.requires_grad, "STE requires seed_features to have grad"
```

---

#### K-PT-H4: Isolation Monitor Doesn't Clear Param Refs

**Severity:** HIGH
**File:** `src/esper/kasmina/isolation.py:75-78`

**Problem:** `reset()` clears `violations` but not `_host_params` and `_seed_params` lists, causing memory leaks when seeds are culled.

**Fix:**
```python
def reset(self) -> None:
    self.violations = 0
    self._host_params.clear()
    self._seed_params.clear()
```

---

## MEDIUM Priority Findings

### DRL Domain

| ID | Issue | File | Description |
|----|-------|------|-------------|
| K-DRL-M1 | No blueprint diversity bonus | `rewards.py` | No UCB-style exploration bonus for underutilized blueprints |
| K-DRL-M2 | Counterfactual doubles compute | `vectorized.py:624-701` | Two forward passes per batch for counterfactual |
| K-DRL-M3 | Missing slot history | `signals.py` | Policy can't see seeds tried, success rate at slot |
| K-DRL-M4 | PBRS may not preserve optimality | `rewards.py:911-955` | `epochs_in_stage` in potential may bias WAIT actions |

### Code Quality Domain

| ID | Issue | File | Description |
|----|-------|------|-------------|
| K-CODE-M1 | Long method | `slot.py:923-1036` | `step_epoch()` is 113 lines with 5 stage handlers |
| K-CODE-M2 | Magic number | `slot.py:424` | `0.95` alpha threshold should be named constant |
| K-CODE-M3 | Untyped deque | `slot.py:172` | `stage_history: deque` lacks type parameters |
| K-CODE-M4 | Redundant import | `slot.py:199` | `timezone` imported locally but already at module level |
| K-CODE-M5 | Inconsistent Optional handling | `slot.py` | Mix of ternary and explicit `if` for None cases |
| K-CODE-M6 | Duplicate telemetry code | `slot.py` | Stage transition telemetry repeated multiple times |

### PyTorch Domain

| ID | Issue | File | Description |
|----|-------|------|-------------|
| K-PT-M1 | torch.compiler.disable on forward | `slot.py:830` | Entire forward opts out of compilation |
| K-PT-M2 | Python float in torch.lerp | `isolation.py:57` | Causes graph break under torch.compile |
| K-PT-M3 | GroupNorm num_groups edge case | `cnn.py:12-25` | Prime channel counts degenerate to InstanceNorm |

---

## LOW Priority Findings

| ID | Domain | Issue | File |
|----|--------|-------|------|
| K-DRL-L1 | DRL | Alpha uninformative during TRAINING | `signals.py` |
| K-DRL-L2 | DRL | SHADOWING stage purpose unclear | `slot.py:1013-1036` |
| K-CODE-L1 | Code | Legacy-referencing comments | `slot.py:619`, `host.py:45` |
| K-CODE-L2 | Code | Default gate returns pass for unknown levels | `slot.py:303-305` |
| K-CODE-L3 | Code | Assert for runtime validation | `host.py:91, 211` |
| K-PT-L1 | PyTorch | TransformerAttentionSeed missing dropout | `transformer.py:62-74` |
| K-PT-L2 | PyTorch | math.tanh vs torch.tanh | `isolation.py:33-44` |
| K-PT-L3 | PyTorch | MorphogeneticModel.to() empty model edge case | `host.py:283-286` |

---

## Positive Findings

All reviewers noted these as well-implemented:

| Area | Description |
|------|-------------|
| **STE Implementation** | Correct gradient flow, well-tested |
| **Zero Initialization** | Seed output projections properly initialized |
| **Protocol Design** | `HostProtocol` using structural typing |
| **Serialization** | Proper use of `get_extra_state`/`set_extra_state` |
| **Intervention Costs** | Well-structured action cost hierarchy |
| **Dataclass Usage** | Good use of `slots=True` and `frozen=True` |
| **torch.lerp** | Efficient fused blending operation |

---

## Recommended Implementation Order

### Immediate (CRITICAL + Quick HIGH fixes)
1. K-CRIT-1: Remove hasattr violation
2. K-CODE-H1: Fix type annotation
3. K-PT-H4: Fix isolation monitor reset

### Short-term (HIGH priority)
4. K-DRL-H1: Add counterfactual to observation space
5. K-PT-H1: Normalize device handling
6. K-CODE-H2: Fix exception handling in registry

### Medium-term (Architecture improvements)
7. K-DRL-H2: Add fossilization legitimacy discount
8. K-DRL-H3: Add host state observability
9. K-DRL-H4: Make G5 require counterfactual

---

## Verification

After implementing fixes:
1. `python3 -m py_compile src/esper/kasmina/*.py`
2. `uv run pytest tests/ -x`
3. Specialist review for domain-specific changes
