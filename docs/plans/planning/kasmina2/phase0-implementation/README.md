# Phase 0 Implementation Plan: A0 + C0 Minimal Milestone

**Status:** Ready for execution
**Parent:** `../kasmina_tamiyo_submodule_intervention_roadmap.md`
**Estimated Duration:** 1-2 weeks
**Risk Level:** Low-Medium

## Objective

Give Tamiyo a **cheap, reversible** way to adjust capacity without defaulting to heavyweight blueprints, using:
- **A0:** Existing injection surfaces (no host changes)
- **C0:** One microstructured seed family (`conv_ladder`) with `GROW_INTERNAL`/`SHRINK_INTERNAL` ops

## Success Criteria (Exit Gates)

Before proceeding to Phase 1, all of the following must be true:

1. **Usage:** `GROW/SHRINK_INTERNAL` used in ≥10% of non-WAIT decisions when ladder seed present
2. **Dominance reduction:** Conv-heavy selection frequency drops vs control (without lowering ROI)
3. **Stability:** No sustained entropy collapse or governor rollback increase
4. **Anti-thrash:** `level_change_rate` bounded, `net_level_change` shows convergence (not oscillation)

## Implementation Tickets

### Track 1: Leyline Contracts (Foundation)

These must land first — everything else depends on them.

| Ticket | Description | Files | Dependencies |
|--------|-------------|-------|--------------|
| **L1** | Add `SeedInternalKind` enum | `src/esper/leyline/seed_internal.py` (new) | None |
| **L2** | Extend `SeedStateReport` with internal fields | `src/esper/leyline/reports.py` | L1 |
| **L3** | Add `GROW_INTERNAL`, `SHRINK_INTERNAL` to `LifecycleOp` | `src/esper/leyline/factored_actions.py` | None |
| **L4** | Add `CONV_LADDER` to `BlueprintAction` | `src/esper/leyline/factored_actions.py` | None |
| **L5** | Update causal masks for internal ops | `src/esper/leyline/causal_masks.py` | L3 |
| **L6** | Add `SEED_INTERNAL_LEVEL_CHANGED` telemetry event | `src/esper/leyline/telemetry.py` | L1 |
| **L7** | Make Obs dims derived from field lists | `src/esper/leyline/__init__.py` | L2, L3 |

### Track 2: Kasmina Mechanics (Seed + Slot)

| Ticket | Description | Files | Dependencies |
|--------|-------------|-------|--------------|
| **K1** | Implement `conv_ladder` blueprint | `src/esper/kasmina/blueprints/cnn.py` | L1, L4 |
| **K2** | Add internal state to `SeedSlot` | `src/esper/kasmina/slot.py` | L1, L2 |
| **K3** | Implement internal ops execution in `SeedSlot` | `src/esper/kasmina/slot.py` | K2, L3, L6 |
| **K4** | Add DDP sync for internal ops | `src/esper/kasmina/slot.py` | K3 |
| **K5** | Wire `SeedState.to_report()` for internal fields | `src/esper/kasmina/slot.py` | K2, L2 |

### Track 3: Tamiyo Policy (Observation + Masking)

| Ticket | Description | Files | Dependencies |
|--------|-------------|-------|--------------|
| **T1** | Add `internal_level_norm` to feature extraction | `src/esper/tamiyo/policy/features.py` | L2, L7 |
| **T2** | Add action masks for internal ops (stage-gated) | `src/esper/tamiyo/policy/action_masks.py` | L3, K2 |
| **T3** | Update feature net input dim assertions | `src/esper/tamiyo/networks/factored_lstm.py` | L7 |

### Track 4: Simic Training (Execution + Rewards)

| Ticket | Description | Files | Dependencies |
|--------|-------------|-------|--------------|
| **S1** | Execute internal ops in vectorized trainer | `src/esper/simic/training/vectorized.py` | K3, L3 |
| **S2** | Add intervention costs for internal ops | `src/esper/simic/rewards/rewards.py` | L3 |
| **S3** | Update rollout buffer state_dim assertions | `src/esper/simic/agent/rollout_buffer.py` | L7 |

### Track 5: Telemetry + Observability

| Ticket | Description | Files | Dependencies |
|--------|-------------|-------|--------------|
| **O1** | Add anti-thrash metrics to telemetry | `src/esper/karn/collector.py` | L6 |
| **O2** | Add counterfactual delta telemetry (observational) | `src/esper/karn/collector.py` | K3, L6 |
| **O3** | Update Karn views for internal level events | `src/esper/karn/mcp/views.py` | L6 |
| **O4** | Add Sanctum widget for internal level display | `src/esper/karn/sanctum/widgets/` | O3 |

### Track 6: Testing

| Ticket | Description | Files | Dependencies |
|--------|-------------|-------|--------------|
| **X1** | Unit tests for `conv_ladder` blueprint | `tests/kasmina/test_blueprints.py` | K1 |
| **X2** | Unit tests for internal ops execution | `tests/kasmina/test_seed_slot.py` | K3 |
| **X3** | Property tests for internal level invariants | `tests/kasmina/properties/` | K2 |
| **X4** | DDP symmetry test for internal ops | `tests/kasmina/test_seed_slot.py` | K4 |
| **X5** | Integration test: internal ops in training loop | `tests/integration/` | S1 |
| **X6** | Obs shape assertion tests | `tests/tamiyo/policy/` | T1, T3 |

## Dependency Graph

```
L1 ──┬──► L2 ──┬──► K2 ──► K3 ──► K4
     │        │         │
     │        └──► K5   └──► S1
     │
     └──► L6 ──► O1, O2, O3

L3 ──┬──► L5
     │
     ├──► T2
     │
     └──► S1, S2

L4 ──► K1

L7 ──► T1, T3, S3

K1 ──► X1
K3 ──► X2, X4, X5
K2 ──► X3
```

## Implementation Order (Suggested)

### Week 1: Foundation + Core Mechanics

**Days 1-2: Leyline contracts**
- [ ] L1: `SeedInternalKind` enum
- [ ] L2: `SeedStateReport` extension
- [ ] L3: `GROW_INTERNAL`, `SHRINK_INTERNAL` ops
- [ ] L4: `CONV_LADDER` blueprint action
- [ ] L5: Causal masks update
- [ ] L6: Telemetry event

**Days 3-4: Kasmina mechanics**
- [ ] K1: `conv_ladder` blueprint implementation
- [ ] K2: Internal state in `SeedSlot`
- [ ] K3: Internal ops execution
- [ ] K5: `to_report()` wiring

**Day 5: Testing checkpoint**
- [ ] X1: Blueprint unit tests
- [ ] X2: Internal ops unit tests
- [ ] X3: Property tests

### Week 2: Policy + Training + Observability

**Days 1-2: Observation + Policy**
- [ ] L7: Derived Obs dims
- [ ] T1: Feature extraction
- [ ] T2: Action masks
- [ ] T3: Feature net assertions

**Days 3-4: Training integration**
- [ ] S1: Vectorized trainer execution
- [ ] S2: Intervention costs
- [ ] S3: Rollout buffer assertions
- [ ] K4: DDP sync

**Day 5: Observability + Final tests**
- [ ] O1-O4: Telemetry + UI
- [ ] X4-X6: Integration + DDP tests

## Experiment Plan

After implementation, run the validation experiments:

```bash
# Control: current blueprint set (no ladder)
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_scale --task cifar_scale \
  --slots r0c0 r0c1 r0c2 r0c3 r0c4 \
  --max-seeds 2 \
  --rounds 100 --envs 8 --episode-length 150 \
  --sanctum

# Treatment: ladder blueprint + internal ops
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_scale --task cifar_scale \
  --slots r0c0 r0c1 r0c2 r0c3 r0c4 \
  --max-seeds 2 \
  --rounds 100 --envs 8 --episode-length 150 \
  --blueprints conv_ladder \
  --sanctum
```

## Key Decisions (Locked for Phase 0)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| `internal_level` range | `[0..L]` where L≤4 | Level 0 = identity; allows spend-down without PRUNE |
| Level 0 implementation | Identity-by-mask | Compile-friendly; compute scaling deferred to Phase 2 |
| Stage gating | TRAINING + BLENDING + HOLDING | Permissive default; tighten if thrash appears |
| Alpha-mode gating | OFF by default | Available as knob; not default |
| Blueprint aliases | OPTIONAL | Phase 0 ships with just `CONV_LADDER` |
| Budget semantics | Param-first | Compute is observational telemetry only |

## Risk Mitigations

| Risk | Mitigation | Telemetry Signal |
|------|------------|------------------|
| Thrash (oscillation) | Small per-op intervention cost | `level_change_rate`, `net_level_change` |
| Obs shape drift | Derived dims + fail-fast assertions | Shape mismatch errors |
| DDP desync | Explicit sync + unit test | Rank divergence in distributed runs |
| Internal ops never fire | Permissive stage gating | `SEED_INTERNAL_LEVEL_CHANGED` event rate |

## Files Changed Summary

| Domain | Files Modified | Files Created |
|--------|----------------|---------------|
| Leyline | `factored_actions.py`, `causal_masks.py`, `reports.py`, `telemetry.py`, `__init__.py` | `seed_internal.py` |
| Kasmina | `slot.py`, `blueprints/cnn.py` | — |
| Tamiyo | `features.py`, `action_masks.py`, `factored_lstm.py` | — |
| Simic | `vectorized.py`, `rewards.py`, `rollout_buffer.py` | — |
| Karn | `collector.py`, `mcp/views.py` | sanctum widget |
| Tests | Multiple test files | New integration tests |
