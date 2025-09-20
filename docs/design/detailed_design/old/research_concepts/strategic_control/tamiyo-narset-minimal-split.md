# Tamiyo/Narset Minimal Split Implementation Plan

**Status**: PROPOSED
**Date**: 2025-01-14
**Type**: Architectural Refactoring

## Executive Summary

Implement a minimal Tamiyo/Narset split in v1.0 to establish the correct architectural foundation for future scaling. This involves moving current tactical logic to Narset while creating a minimal Tamiyo strategic layer.

## Rationale

- **Architecture is destiny**: Easier to start with the right structure than refactor later
- **Message contracts**: Changing interfaces after production is risky
- **Natural evolution path**: Provides clear location for future features (SOPs, budgets, federation)
- **Testing benefits**: Clean separation enables better mocking and testing

## Current State Analysis

### What "Tamiyo" Currently Does (Actually Tactical)
```python
# Current Tamiyo responsibilities - all tactical execution
- Decide which seeds to activate
- Manage seed lifecycle states
- Coordinate with Kasmina for execution
- Handle grafting schedules
- Make rollback decisions based on metrics
- Select adaptation strategies
```

### What Strategic Control Should Be
```python
# True strategic responsibilities (future Tamiyo)
- Set global risk tolerance
- Allocate stability/energy budgets
- Approve high-risk operations
- Define long-term objectives
- Coordinate multi-region strategies
```

## Proposed Minimal Split (v1.0)

### File Structure Changes

```
Before:
docs/architecture/migration/
  03-tamiyo-unified-design.md         # Everything mixed together
  03.1-tamiyo-gnn-architecture.md     # GNN details
  03.2-tamiyo-policy-training.md      # Training loop
  03.3-tamiyo-risk-modeling.md        # Risk assessment
  03.4-tamiyo-integration-contracts.md # Message contracts

After:
docs/architecture/migration/
  03-tamiyo-unified-design.md         # Strategic controller (minimal)
  03.1-tamiyo-policy-management.md    # Global policy settings
  03.2-tamiyo-risk-governance.md      # Risk thresholds

  14-narset-unified-design.md         # Tactical controller (current logic)
  14.1-narset-seed-management.md      # Seed lifecycle
  14.2-narset-adaptation-execution.md # Grafting/rollback
  14.3-narset-gnn-architecture.md     # GNN implementation
  14.4-narset-integration-contracts.md # Execution messages
```

### Message Contract Changes

```protobuf
// Before: Single monolithic controller
message AdaptationCommand {
  string seed_id = 1;
  AdaptationType type = 2;
  // ... all details mixed
}

// After: Clear separation
message TamiyoPolicy {
  double risk_tolerance = 1;      // 0.0 (conservative) to 1.0 (aggressive)
  uint32 max_concurrent_seeds = 2;
  uint32 stability_budget = 3;    // Future: actual energy units
  repeated string embargo_zones = 4;
}

message NarsetRequest {
  string region_id = 1;           // Future: support multiple regions
  TamiyoPolicy policy = 2;        // Strategic constraints
  SystemTelemetry telemetry = 3;
}

message NarsetDecision {
  string region_id = 1;
  repeated AdaptationCommand commands = 2;
  double risk_consumed = 3;       // How much of budget used
}
```

### Implementation Steps (Minimal)

#### Phase 1: Create Narset (Week 1)
1. **Copy** `03-tamiyo-unified-design.md` → `14-narset-unified-design.md`
2. **Move** all tactical logic sections to Narset
3. **Update** all Kasmina references to point to Narset
4. **Rename** internal logic from "Tamiyo strategic decision" to "Narset tactical execution"

#### Phase 2: Minimal Tamiyo (Week 1)
1. **Reduce** Tamiyo to policy management only:
   ```python
   class TamiyoController:
       def __init__(self):
           self.risk_tolerance = 0.7  # Single dial
           self.max_seeds = 10

       def update_policy(self, metrics):
           # Simple adaptive policy
           if metrics.recent_failures > threshold:
               self.risk_tolerance *= 0.9
           return TamiyoPolicy(...)

       def approve_high_risk(self, operation):
           # Gate for risky operations
           return operation.risk_score < self.risk_tolerance
   ```

2. **Pass-through** for v1.0:
   - Tamiyo sets policy → Narset makes all real decisions
   - High-risk operations check with Tamiyo (rare in v1.0)

#### Phase 3: Integration Updates (Week 2)
1. **Tolaria** talks to Narset for execution, Tamiyo for policy
2. **Simic** trains both policies (but Narset is primary)
3. **Jace** coordinates curriculum for both
4. **Oona** adds new topics: `tamiyo.policy.*`, `narset.decisions.*`

### Migration Impact

| Component | Change Required | Complexity |
|-----------|----------------|------------|
| Tolaria | Point to Narset for execution | Low |
| Kasmina | Subscribe to Narset commands | Low |
| Simic | Train two policies (mostly Narset) | Medium |
| Jace | Coordinate two curricula | Medium |
| Oona | New message topics | Low |
| Emrakul | Check Tamiyo for pruning budget | Low |

### Future Evolution Path (Clear with Split)

```
v1.0: Minimal Split
├── Tamiyo: Single risk dial
└── Narset: All current logic

v1.5: Budget System
├── Tamiyo: Energy/stability budgets
└── Narset: Spends allocated budget

v2.0: Federation
├── Tamiyo: Global strategy
├── Narset-1: Region A tactics
├── Narset-2: Region B tactics
└── Narset-3: Region C tactics

v3.0: Full Hierarchy
├── Tamiyo Prime: Global strategy
├── Tamiyo Regional: Per-region strategy
└── Narset Fleet: Tactical execution
```

### Testing Benefits

```python
# Clean mocking with split
def test_high_risk_operation():
    mock_tamiyo = Mock(risk_tolerance=0.5)
    narset = NarsetController(tamiyo=mock_tamiyo)

    # Test tactical decisions independently
    decision = narset.decide(telemetry)
    assert mock_tamiyo.approve_high_risk.called

# Without split - tangled concerns
def test_adaptation():
    tamiyo = TamiyoController()  # Has to mock everything
    # Hard to test strategy vs tactics separately
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Added complexity for v1.0 | Minimal implementation - mostly moving code |
| Two components to maintain | Narset has all logic, Tamiyo is trivial |
| Message contract changes | Do it now before production |
| Coordination overhead | Simple pass-through for v1.0 |

## Decision Criteria

**Do the split now if:**
- You expect to scale beyond 10B parameters ✓
- You want clean architecture from day one ✓
- You plan to implement budgets/federation ✓
- You value testing/maintainability ✓

**Defer the split if:**
- You need v1.0 in <2 weeks ✗
- You're building a proof-of-concept ✗
- You'll never scale beyond single controller ✗

## Recommendation

**IMPLEMENT THE MINIMAL SPLIT NOW**

The cost is low (1-2 weeks), the benefits are immediate (clean architecture), and the future evolution path is clear. Starting with the right structure is always easier than refactoring later.

## Implementation Checklist

- [ ] Create Narset documentation structure
- [ ] Move tactical logic from Tamiyo to Narset
- [ ] Implement minimal Tamiyo policy layer
- [ ] Update message contracts in Leyline
- [ ] Update integration points (Tolaria, Kasmina)
- [ ] Update Simic/Jace for dual training
- [ ] Test separation of concerns
- [ ] Document the split in ADR-003