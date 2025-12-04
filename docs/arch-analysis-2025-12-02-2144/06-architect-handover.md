# Architect Handover Report

**Project:** esper-lite
**Analysis Date:** 2025-12-02
**Prepared For:** System Architect (axiom-system-architect transition)
**Confidence Level:** HIGH

---

## Purpose

This document provides a prioritized improvement roadmap and technical debt catalog for architectural decision-making. It synthesizes findings from the comprehensive archaeology analysis including 11 SME reports from DRL and PyTorch specialists.

---

## Executive Handover Summary

Esper-lite is a **production-ready** morphogenetic neural network training system with **clean domain-driven architecture**. The system demonstrates strong fundamentals but requires immediate attention to **3 critical bugs** in multi-GPU training before production deployment.

### Architecture Verdict

| Aspect | Assessment |
|--------|------------|
| Domain Model | EXCELLENT - Clear seed lifecycle abstraction |
| Package Boundaries | GOOD - Low coupling, high cohesion |
| RL Implementation | CORRECT - PPO and PBRS mathematically sound |
| PyTorch Patterns | GOOD - Modern practices, some optimization opportunities |
| Production Readiness | BLOCKED - Critical bugs must be fixed first |

---

## Critical Path: Blocking Issues

These issues **must be resolved before any production deployment**:

### C1: CUDA Stream Race Condition
- **Location:** `simic/vectorized.py:518-522`
- **Impact:** Training corruption in multi-GPU scenarios
- **Root Cause:** Accumulator tensors may have concurrent writes from multiple environments on same device without stream synchronization
- **Fix Approach:** Add `torch.cuda.synchronize()` or use stream-aware atomic operations
- **Effort:** 2-4 hours

### C2: Normalized vs Raw State Mismatch
- **Location:** `simic/vectorized.py:725-732`
- **Impact:** PPO ratio calculations use inconsistent state representations
- **Root Cause:** Buffer stores raw states but policy expects normalized states
- **Fix Approach:** Store normalized states in buffer OR normalize consistently at retrieval
- **Effort:** 1-2 hours

### C3: Placeholder Reward Function
- **Location:** `simic/episodes.py:301`
- **Impact:** No meaningful learning signal for PPO training
- **Root Cause:** `compute_reward()` returns `accuracy_change * 10.0` as placeholder
- **Fix Approach:** Integrate with existing PBRS implementation in `simic/rewards.py`
- **Effort:** 2-4 hours

---

## Technical Debt Catalog

### Tier 1: High-Impact Debt (Fix This Month)

| ID | Location | Issue | Impact | Effort |
|----|----------|-------|--------|--------|
| H1 | `simic/training.py:50-352` | 300+ line function with 6 elif branches | Maintainability, testing | 1 day |
| H2 | `kasmina/slot.py:887` | Unauthorized hasattr usage | Policy violation | 1 hour |
| H3 | `nissa/tracker.py` | CUDA sync in gradient hooks | 5-10x performance penalty | 4 hours |
| H4 | `simic/networks.py:463` | None stub classes cause confusing errors | Developer experience | 2 hours |
| H5 | `simic/episodes.py:450` | load_all() loads entire dataset into memory | Memory scaling | 4 hours |

### Tier 2: Medium-Impact Debt (Fix This Quarter)

| ID | Location | Issue | Effort |
|----|----------|-------|--------|
| M1 | Multiple files | Broad exception handlers masking bugs | 2 days |
| M2 | `simic/ppo.py:198` | Unsafe torch.load() without weights_only | 1 hour |
| M3 | `leyline/actions.py:6` | SimicAction legacy alias | 30 min |
| M4 | `tolaria/governor.py` | Optimizer state not in snapshots | 2 hours |
| M5 | `simic/normalization.py:55` | normalize() lacks @no_grad decorator | 30 min |
| M6 | `simic/features.py:140` | normalize_observation() returns 9, expects 27 | 1 hour |
| M7 | `simic/gradient_collector.py:45` | Private API torch._foreach_norm | 1 hour |

### Tier 3: Low-Impact Debt (Opportunistic)

| ID | Location | Issue |
|----|----------|-------|
| L1 | `simic/rewards.py:180` | Unused compute_potential() function |
| L2 | `simic/rewards.py:220` | stage_potentials dict recreated per call |
| L3 | `simic/rewards.py:290` | String matching for action names vs enum |
| L4 | `simic/normalization.py` | Missing state_dict/load_state_dict methods |
| L5 | `simic/sanity.py:12` | SANITY_CHECKS_ENABLED flag never used |
| L6 | `simic/features.py:12` | Unused TensorSchema import |

---

## Improvement Roadmap

### Phase 1: Stabilization (Week 1-2)

**Goal:** Production-ready multi-GPU training

| Task | Priority | Dependencies |
|------|----------|--------------|
| Fix CUDA stream race condition (C1) | P0 | None |
| Fix state normalization mismatch (C2) | P0 | None |
| Replace placeholder reward (C3) | P0 | None |
| Add integration tests for vectorized training | P0 | C1, C2 |

**Exit Criteria:** All tests pass, vectorized training produces consistent results

### Phase 2: Code Health (Week 3-4)

**Goal:** Improved maintainability and developer experience

| Task | Priority | Dependencies |
|------|----------|--------------|
| Refactor run_ppo_episode() into stage handlers | P1 | Phase 1 |
| Fix unauthorized hasattr (H2) | P1 | None |
| Migrate gradient collection to async pattern | P1 | None |
| Add streaming DatasetManager | P1 | None |
| Wire SANITY_CHECKS_ENABLED to checks | P2 | None |

**Exit Criteria:** No function exceeds 100 lines, hasattr compliance, gradient collection non-blocking

### Phase 3: Optimization (Week 5-8)

**Goal:** Performance and scalability improvements

| Task | Priority | Dependencies |
|------|----------|--------------|
| Add torch.compile support | P2 | Phase 2 |
| Implement multi-seed support in SignalTracker | P2 | None |
| Replace _foreach_norm with public API | P2 | None |
| Add comprehensive structured logging | P3 | None |

**Exit Criteria:** 20%+ training throughput improvement, multi-seed experiments enabled

### Phase 4: Future Architecture (Quarter 2+)

**Goal:** Strategic capability expansion

| Task | Priority | Notes |
|------|----------|-------|
| Gradient checkpointing for memory efficiency | P3 | Enables larger models |
| FSDP integration for distributed training | P3 | Scales beyond single node |
| Windowed/decay normalization variants | P3 | Non-stationary environments |
| Feature normalization layer in leyline | P3 | Standardized signal processing |

---

## Module Quality Summary

| Module | Score | Strengths | Key Issue |
|--------|-------|-----------|-----------|
| leyline | 8/10 | Clean contracts, immutable signals | Feature normalization needed |
| kasmina | 8.5/10 | Solid lifecycle, proper STE | Unauthorized hasattr |
| simic | 6.5/10 | Correct PPO, good PBRS | Critical vectorized bugs |
| tamiyo | 7.5/10 | Good baseline, clean heuristic | Multi-seed assumption |
| nissa | 7/10 | Good observer pattern | CUDA sync overhead |
| tolaria | 8.5/10 | Excellent safety features | Optimizer state not snapshotted |
| runtime | 8/10 | Clean task abstraction | Minor config duplication |
| utils | 8/10 | Production-ready loaders | None significant |
| scripts | 7.5/10 | Functional CLI | Limited error handling |

**Average Score:** 7.6/10 - Good architecture with localized issues

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Multi-GPU training corruption | HIGH | HIGH | Fix C1/C2 immediately |
| Memory exhaustion at scale | MEDIUM | MEDIUM | Streaming DatasetManager |
| Gradient explosion in incubator mode | LOW | MEDIUM | TolariaGovernor 6sigma detection |
| torch.compile graph breaks | MEDIUM | LOW | Audit dynamic control flow |

### Architectural Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| simic complexity growth | MEDIUM | MEDIUM | Refactor run_ppo_episode() |
| nissa performance bottleneck | MEDIUM | MEDIUM | Async gradient collection |
| Tamiyo single-seed limitation | LOW | LOW | SignalTracker multi-seed support |

---

## Recommendations for System Architect

### Immediate Actions (This Week)

1. **Assign senior engineer to C1/C2/C3** - These are blocking production
2. **Add CI gate for vectorized training** - Prevent regression
3. **Review hasattr authorization** in `kasmina/slot.py:887`

### Strategic Decisions Needed

1. **Training Loop Refactor Strategy**
   - Option A: Stage-specific handler pattern (recommended)
   - Option B: State machine with transition hooks
   - Option C: Policy-based dispatch

2. **Gradient Collection Architecture**
   - Option A: Async collection in dedicated thread (recommended)
   - Option B: Deferred materialization with lazy evaluation
   - Option C: GPU-side accumulation with batch sync

3. **Multi-Seed Support Scope**
   - Option A: SignalTracker per-seed isolation (minimal change)
   - Option B: Full concurrent seed support (significant refactor)
   - Option C: Sequential seed processing (no change needed)

### Documentation Gaps

1. **Missing:** Architecture Decision Records (ADRs) for key design choices
2. **Missing:** Runbook for common operational scenarios
3. **Missing:** Performance benchmarking baseline

---

## Appendix: SME Report Index

| Report | Location | Score |
|--------|----------|-------|
| leyline | sme-reports/leyline.md | 8/10 |
| kasmina | sme-reports/kasmina.md | 8.5/10 |
| simic-core | sme-reports/simic-core.md | 7/10 |
| simic-training | sme-reports/simic-training.md | 6.5/10 |
| simic-rewards-features | sme-reports/simic-rewards-features.md | 7.5/10 |
| simic-data | sme-reports/simic-data.md | 6/10 |
| tamiyo | sme-reports/tamiyo.md | 7.5/10 |
| nissa | sme-reports/nissa.md | 7/10 |
| tolaria | sme-reports/tolaria.md | 8.5/10 |
| runtime | sme-reports/runtime.md | 8/10 |
| utils-scripts | sme-reports/utils-scripts.md | 8/10 |

---

## Conclusion

Esper-lite is a well-designed system with strong domain modeling and correct RL implementation. The architecture is ready for production after addressing the 3 critical bugs in multi-GPU training.

**Recommended Next Steps:**
1. Fix C1, C2, C3 (estimated 1-2 days total)
2. Add integration tests for vectorized training
3. Begin Phase 2 refactoring for improved maintainability

The modular design enables incremental improvements without major architectural changes. The quality gate system (G0-G5) and TolariaGovernor provide strong safety guarantees once the critical bugs are resolved.

---

**Report Generated:** 2025-12-02
**Handover Status:** READY FOR ARCHITECT REVIEW
