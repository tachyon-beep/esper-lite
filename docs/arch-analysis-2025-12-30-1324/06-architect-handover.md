# Esper Architect Handover Document

**Analysis Date:** 2025-12-30
**Purpose:** Enable architecture improvement planning based on analysis findings

---

## 1. Executive Summary

This document provides actionable recommendations for Esper architecture improvements, organized by priority and effort. The codebase is in excellent shape with minimal technical debt - improvements focus on polish, extensibility, and preparation for future phases.

### Overall Assessment

| Category | Current State | Target State | Gap |
|----------|---------------|--------------|-----|
| Architecture | Excellent (A) | Excellent | Minimal |
| Code Quality | Very Good (A-) | Excellent | Small cleanup |
| Documentation | Good (B+) | Excellent | API docs needed |
| Testing | Good (B) | Very Good | Coverage gaps |
| Performance | Excellent (A) | Excellent | None |

---

## 2. Improvement Roadmap

### 2.1 Immediate Actions (This Sprint)

#### Task 1: Remove Dead Telemetry Code
**Priority:** Low | **Effort:** 1 hour | **Risk:** None

**What to do:**
```python
# In leyline/telemetry.py, remove these unused items:
- ISOLATION_VIOLATION event type (line ~79)
- GOVERNOR_PANIC event type (lines ~98-100)
- GOVERNOR_SNAPSHOT event type (lines ~102-104)
- PerformanceBudgets class (lines ~147-149)
- DEFAULT_BUDGETS constant
```

**Why:** Dead code causes confusion when reading contracts. Clean codebase is easier to maintain.

**Verification:** `grep -r "ISOLATION_VIOLATION\|GOVERNOR_PANIC\|GOVERNOR_SNAPSHOT\|PerformanceBudgets" src/` should return only the definitions.

---

#### Task 2: Implement CHECKPOINT_SAVED Event
**Priority:** Medium | **Effort:** 2 hours | **Risk:** Low

**What to do:**
1. Define payload in `leyline/telemetry.py`:
```python
@dataclass(frozen=True, slots=True)
class CheckpointSavedPayload:
    checkpoint_path: str
    epoch: int
    val_accuracy: float
    model_params: int
    optimizer_state_size: int
    timestamp: datetime
```

2. Emit in checkpoint save logic (create if doesn't exist):
```python
hub().emit(TelemetryEvent(
    event_type=TelemetryEventType.CHECKPOINT_SAVED,
    data=CheckpointSavedPayload(...),
    ...
))
```

**Why:** Enables tracking checkpoint history for resumption debugging and experiment reproducibility.

---

#### Task 3: Document Entropy Floor Trade-off
**Priority:** Low | **Effort:** 30 minutes | **Risk:** None

**What to do:**
Add comment in `simic/agent/ppo.py` near entropy computation:
```python
# DESIGN NOTE: Entropy floor implementation
#
# We use torch.clamp(entropy, min=FLOOR) which has zero gradient when
# entropy equals the floor. This is intentional:
#
# Pros:
#   - Prevents entropy collapse to zero
#   - Simple and stable
#
# Cons:
#   - No gradient signal at floor (learning may stall)
#
# Alternative considered: Soft floor with residual gradient
#   entropy = raw_entropy + FLOOR * sigmoid(-raw_entropy / temperature)
#
# We chose the simple clamp because:
#   1. Floor is rarely hit in practice with good exploration
#   2. Action masking naturally limits entropy in constrained states
#   3. Simpler implementation is easier to debug
#
# If learning stalls with constrained action spaces, reconsider.
```

**Why:** Future maintainers will understand the trade-off without reverse-engineering.

---

### 2.2 Short-Term Improvements (This Month)

#### Task 4: Add PPO Training Smoke Test
**Priority:** Medium | **Effort:** 4 hours | **Risk:** Low

**What to do:**
Create `tests/integration/test_ppo_smoke.py`:
```python
@pytest.mark.slow
def test_ppo_training_completes():
    """Verify PPO training runs without errors for 5 rounds."""
    config = TrainingConfig(
        n_episodes=5,
        n_envs=2,
        max_epochs=10,
        # Use fast settings
    )
    metrics = train_ppo_vectorized(config, task_spec=get_task_spec("cifar10"))

    # Assertions
    assert metrics.total_rounds == 5
    assert not math.isnan(metrics.final_policy_loss)
    assert metrics.entropy_collapsed_count == 0

@pytest.mark.slow
def test_ppo_checkpoint_roundtrip():
    """Verify checkpoint save/load preserves state."""
    # Train 3 rounds, save, load, train 2 more
    ...
```

**Why:** Prevents silent regressions in the training loop.

---

#### Task 5: Add Property-Based Tests for Leyline
**Priority:** Low | **Effort:** 4 hours | **Risk:** None

**What to do:**
Create `tests/leyline/test_contracts_properties.py`:
```python
from hypothesis import given, strategies as st

@given(row=st.integers(0, 10), col=st.integers(0, 10))
def test_slot_id_roundtrip(row, col):
    slot_id = format_slot_id(row, col)
    parsed_row, parsed_col = parse_slot_id(slot_id)
    assert (parsed_row, parsed_col) == (row, col)

@given(stage=st.sampled_from(list(SeedStage)))
def test_stage_encoding_roundtrip(stage):
    one_hot = stage_to_one_hot(stage.value)
    idx = one_hot.index(1.0)
    recovered = INDEX_TO_STAGE[idx]
    assert recovered == stage.value
```

**Why:** Property tests catch edge cases that example-based tests miss.

---

### 2.3 Medium-Term Improvements (This Quarter)

#### Task 6: Split Karn Domain
**Priority:** Low | **Effort:** 8 hours | **Risk:** Medium

**Current state:** Karn is 17,800 LOC covering core telemetry, TUI, web dashboard, and SQL interface.

**Target structure:**
```
karn/
├── __init__.py           # Re-exports for backwards compatibility
├── core/                 # Core telemetry (required)
│   ├── collector.py
│   ├── store.py
│   ├── health.py
│   ├── triggers.py
│   └── constants.py
├── sanctum/              # TUI (optional)
│   ├── __init__.py
│   ├── backend.py
│   ├── aggregator.py
│   ├── app.py
│   └── widgets/
├── overwatch/            # Web (optional)
│   ├── __init__.py
│   ├── backend.py
│   └── web/
└── mcp/                  # SQL interface (optional)
    ├── __init__.py
    ├── server.py
    └── views.py
```

**Benefits:**
- Optional installs: `pip install esper[sanctum]`, `pip install esper[overwatch]`
- Faster imports when TUI/web not needed
- Clearer ownership boundaries

**Migration steps:**
1. Create subdirectories
2. Move files with `git mv`
3. Update imports throughout codebase
4. Add `__init__.py` re-exports for backwards compatibility
5. Update `pyproject.toml` optional dependencies

---

#### Task 7: Expand Integration Test Coverage
**Priority:** Medium | **Effort:** 16 hours | **Risk:** Low

**Test scenarios to add:**

| Test | Coverage Gap |
|------|--------------|
| Multi-slot germination | Concurrent slot operations |
| Seed fossilization flow | Full lifecycle completion |
| Governor rollback | Catastrophic failure recovery |
| TUI event rendering | Widget state updates |
| WebSocket reconnection | Connection resilience |

**Implementation approach:**
1. Use `pytest-asyncio` for WebSocket tests
2. Mock CUDA for CI environments
3. Use Playwright for Overwatch E2E

---

### 2.4 Long-Term Considerations (Future Phases)

#### Task 8: Prepare for Emrakul (Immune System)
**Timeline:** Phase 4

**What it needs:**
- Withering mechanism (alpha → 0 over time)
- Efficiency auditing metrics
- Parasitic component detection

**Architecture preparation:**
```python
# In leyline/schemas.py, add:
class WitherConfig:
    wither_rate: float = 0.1  # Alpha decay per epoch
    min_contribution_threshold: float = 0.01
    audit_frequency_epochs: int = 10

# In kasmina/slot.py, add:
def wither(self, rate: float) -> None:
    """Gradually reduce alpha to audit seed necessity."""
    self.alpha_controller.retarget(
        alpha_target=max(0.0, self.alpha - rate),
        alpha_steps_total=1,
    )

def compute_churn(self) -> float:
    """Measure impact of seed removal on loss."""
    # Counterfactual evaluation with alpha=0
    ...
```

**Why prepare now:** Interface design now prevents breaking changes later.

---

#### Task 9: Prepare for Narset (Endocrine System)
**Timeline:** Phase 5

**What it needs:**
- Resource allocation signals between slot clusters
- Budget distribution across layers
- Coordination without micro-management

**Architecture preparation:**
```python
# In leyline/signals.py, add:
@dataclass
class HormonalSignal:
    signal_type: str  # "growth_factor", "stress_response", etc.
    source_cluster: str
    intensity: float  # 0.0 to 1.0
    target_clusters: list[str]

# In a future narset/ domain:
class NarsetController:
    """Coordinates resource allocation across slot clusters."""

    def emit_growth_signal(self, cluster: str, intensity: float) -> None:
        """Signal growth opportunity to target cluster."""
        ...

    def emit_stress_response(self, cluster: str) -> None:
        """Signal resource scarcity requiring consolidation."""
        ...
```

---

## 3. Technical Debt Register

| ID | Item | Type | Priority | Effort | Owner |
|----|------|------|----------|--------|-------|
| TD-001 | Dead telemetry events | Dead code | Low | 1h | - |
| TD-002 | Missing CHECKPOINT_SAVED | Feature gap | Medium | 2h | - |
| TD-003 | Entropy floor undocumented | Documentation | Low | 30m | - |
| TD-004 | Karn domain size | Refactoring | Low | 8h | - |
| TD-005 | Integration test gaps | Testing | Medium | 16h | - |
| TD-006 | Checkpoint security | Security | Low | 4h | - |

**Total estimated debt:** ~32 hours

---

## 4. Architecture Decision Log

Decisions made during this analysis that should be tracked:

### ADR-001: Keep Leyline as Single Source of Truth
**Status:** Confirmed
**Context:** All shared contracts centralized in Leyline
**Decision:** Maintain this pattern; no domain-specific contracts
**Consequences:** Clean dependencies, single place for schema updates

### ADR-002: Protocol-Based Decoupling
**Status:** Confirmed
**Context:** Critical interfaces use `@runtime_checkable` Protocols
**Decision:** Continue this pattern for new interfaces
**Consequences:** Testability, alternative implementations possible

### ADR-003: Biological Metaphor
**Status:** Confirmed
**Context:** Body/Organism for architecture, Botanical for lifecycle
**Decision:** No new metaphors; extend existing consistently
**Consequences:** Intuitive mental model, consistent documentation

### ADR-004: Inverted Control Flow
**Status:** Confirmed
**Context:** Batch-first iteration for GPU efficiency
**Decision:** Required for all new training code
**Consequences:** Performance, SharedBatchIterator dependency

---

## 5. Knowledge Transfer Checklist

### For New Contributors

- [ ] Read README.md for architecture overview
- [ ] Read ROADMAP.md for 9 principles
- [ ] Understand biological metaphor (body + botanical)
- [ ] Run `uv run pytest` to verify setup
- [ ] Try `PYTHONPATH=src uv run python -m esper.scripts.train ppo --help`

### For Architecture Work

- [ ] Understand Leyline ownership boundary
- [ ] Know the 4 key Protocols (HostProtocol, PolicyBundle, TelemetryEventLike, OutputBackend)
- [ ] Understand inverted control flow pattern
- [ ] Review SharedBatchIterator implementation
- [ ] Understand three-tiered telemetry

### For Debugging

- [ ] Use `--sanctum` for TUI debugging
- [ ] Use MCP SQL interface for telemetry queries
- [ ] Check TolariaGovernor for catastrophic failures
- [ ] Review anomaly detection thresholds in `karn/constants.py`

---

## 6. Contact Points

| Area | Resource |
|------|----------|
| Architecture questions | This document + ROADMAP.md |
| PPO implementation | `simic/agent/ppo.py` docstrings |
| Telemetry contracts | `leyline/telemetry.py` |
| TUI development | `karn/sanctum/` + Textual docs |
| Web dashboard | `karn/overwatch/` + Vue docs |

---

## 7. Success Criteria

### For This Analysis

- [x] All 7 domains cataloged with dependencies
- [x] C4 diagrams generated (Context, Container, Component)
- [x] Quality assessment completed
- [x] Technical debt inventoried
- [x] Improvement roadmap created

### For Improvement Implementation

- [ ] Dead code removed (TD-001)
- [ ] CHECKPOINT_SAVED implemented (TD-002)
- [ ] Entropy floor documented (TD-003)
- [ ] PPO smoke test added
- [ ] Property tests for Leyline added

### For Long-Term Health

- [ ] Test coverage > 80%
- [ ] No new circular dependencies
- [ ] Technical debt < 40 hours
- [ ] All new interfaces use Protocols

---

## Appendix A: File Inventory

### High-Value Files (Read These First)

| File | Why Important |
|------|---------------|
| `leyline/__init__.py` | All training constants |
| `leyline/telemetry.py` | Event contracts |
| `kasmina/slot.py` | Seed lifecycle core |
| `simic/training/vectorized.py` | Main training loop |
| `tamiyo/policy/protocol.py` | Policy interface |

### Files Requiring Attention

| File | Issue | Action |
|------|-------|--------|
| `leyline/telemetry.py` | Dead events | Remove |
| `simic/training/vectorized.py` | Large (3.4K LOC) | Document well |
| `simic/rewards/rewards.py` | Large (1.7K LOC) | Consider splitting |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Seed** | Neural module that can be grafted into host |
| **Slot** | Location where seeds can be attached |
| **Host** | Base neural network (CNN or Transformer) |
| **Fossilize** | Permanently integrate seed into host |
| **Prune** | Remove underperforming seed |
| **Alpha** | Blending weight (0=host only, 1=seed included) |
| **Gate** | Quality validator for stage transitions (G0-G5) |
| **PBRS** | Potential-Based Reward Shaping |
| **Counterfactual** | Causal attribution via what-if analysis |

---

*End of Architect Handover Document*
