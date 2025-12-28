# Esper Architecture Analysis - Final Report

**Analysis Date:** 2025-12-28
**Analysis Type:** Full Re-Analysis (Option C - Architect-Ready)
**Previous Analysis:** December 13, 2025
**Analyst:** Claude Code (axiom-system-archaeologist)

---

## Executive Summary

Esper is a **morphogenetic neural network framework** that dynamically grows, prunes, and adapts model topology during training. This analysis documents the architecture at ~53K LOC (44.7K Python + 8.7K Vue/TS), representing a **2.6x growth** since the previous analysis 15 days ago.

### Key Findings

| Dimension | Assessment | Evidence |
|-----------|------------|----------|
| **Architecture** | Excellent | Clean 7-domain "organ" structure with protocol decoupling |
| **Type Safety** | Excellent | 116 dataclasses, 12 protocols, strict mypy |
| **Testing** | Excellent | 1.33:1 test-to-code ratio, mutation testing |
| **Maintainability** | Good | B+ overall, one complexity hotspot |
| **Documentation** | Good | External docs strong, inline inconsistent |

### One-Sentence Summary

> Esper is a well-architected, strongly-typed RL framework for neural architecture growth, with clean domain separation and comprehensive test coverage, requiring only minor cleanup of dead code and one large file decomposition.

---

## System Overview

### What Esper Does

1. **Grows neural networks** by injecting "seed" modules at runtime
2. **Uses PPO** (Proximal Policy Optimization) to learn optimal growth strategies
3. **Manages seed lifecycle** through botanical stages (germinate → train → blend → fossilize)
4. **Provides observability** via TUI, web dashboard, and MCP SQL interface

### The Biological Metaphor

Esper uses two complementary metaphors:

| Metaphor | Scope | Terms |
|----------|-------|-------|
| **Organism** | System architecture | 7 domain "organs" |
| **Botanical** | Seed lifecycle | Germinate, train, blend, fossilize, prune |

---

## Architecture Summary

### Seven Domains ("Organs")

```
┌─────────────────────────────────────────────────────────────────┐
│                        ESPER DOMAINS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LEYLINE (3,735 LOC) ─────────────────────────────────────────  │
│  DNA/Genome: All contracts, enums, constants, telemetry types   │
│                                                                  │
│  KASMINA (5,174 LOC) ─────────────────────────────────────────  │
│  Stem Cells: SeedSlot lifecycle, quality gates, blueprints      │
│                                                                  │
│  TAMIYO (3,811 LOC) ──────────────────────────────────────────  │
│  Brain/Cortex: PolicyBundle (heuristic or neural), decisions    │
│                                                                  │
│  SIMIC (13,352 LOC) ──────────────────────────────────────────  │
│  Evolution: PPO agent, training loop, rewards, attribution      │
│                                                                  │
│  NISSA (1,969 LOC) ───────────────────────────────────────────  │
│  Sensory Organs: NissaHub telemetry routing, pub-sub backends   │
│                                                                  │
│  KARN (17,063 LOC) ───────────────────────────────────────────  │
│  Memory: TelemetryStore, Sanctum TUI, Overwatch web, MCP SQL    │
│                                                                  │
│  TOLARIA (462 LOC) ───────────────────────────────────────────  │
│  Metabolism: Governor watchdog, anomaly detection, rollback     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dependency Flow

```
Scripts → Simic → Tamiyo ──┐
                           ├──► Kasmina → Leyline
                  Karn ────┘
                    ↑
              Nissa─┴─Tolaria
```

**Key invariant:** Leyline has no outbound dependencies. All domains import contracts from Leyline.

---

## Growth Analysis

### LOC Comparison

| Domain | Dec 13 | Dec 28 | Growth |
|--------|--------|--------|--------|
| Karn | 1,700 | 17,063 | +900% |
| Tamiyo | 411 | 3,811 | +827% |
| Leyline | 844 | 3,735 | +343% |
| Kasmina | 1,400 | 5,174 | +270% |
| Simic | 7,800 | 13,352 | +71% |
| Nissa | 1,700 | 1,969 | +16% |
| Tolaria | 746 | 462 | -38% |

### Major Additions

1. **Sanctum TUI** (Karn): Full Textual-based terminal dashboard
2. **Overwatch Web** (Karn): Vue 3 real-time dashboard
3. **Neural Policies** (Tamiyo): LSTM-based PolicyBundle alongside heuristic
4. **Typed Payloads** (Leyline): 18 dataclass event types
5. **Attribution System** (Simic): Counterfactual Shapley values

---

## Technical Highlights

### Factored Action Space

Esper uses an **8-head factored policy** for seed lifecycle decisions:

| Head | Purpose |
|------|---------|
| 0 | slot_id - Which slot to operate on |
| 1 | blueprint - Which module template |
| 2 | style - Configuration variant |
| 3 | tempo - Training speed modifier |
| 4 | alpha_curve - Blend schedule shape |
| 5 | alpha_duration - Blend duration |
| 6 | lifecycle_op - GERMINATE/ADVANCE/PRUNE/HOLD |
| 7 | value - State value estimate (critic) |

### Quality Gates

Seed progression is gated by G0-G5 quality checks:
- **G0:** Germination entry
- **G1:** Training readiness
- **G2:** Blending eligibility
- **G3:** Holding confirmation
- **G4:** Fossilization pre-check
- **G5:** Final fossilization

Modes: PERMISSIVE (dev) and STRICT (prod).

### GPU-First Design

Inverted control flow for CUDA throughput:
1. DataLoader iteration first
2. Environment dispatch second
3. Pre-allocated GPU buffers

This enables `SharedGPUBatchIterator` for 8x speedup on CIFAR-10.

---

## Quality Summary

### Strengths

| Area | Evidence |
|------|----------|
| **Layering** | Leyline imports only from itself |
| **Type Safety** | 116 dataclasses, 6 minor mypy errors |
| **Testing** | 59,411 test LOC, 2,339+ tests, mutation testing |
| **Decomposition** | 7 focused domains + 3 support modules |
| **Protocols** | 12 protocol interfaces for decoupling |

### Weaknesses

| Area | Evidence | Priority |
|------|----------|----------|
| **One large file** | `vectorized.py` at 3,404 LOC | MEDIUM |
| **Dead code** | 8 dead event types in Leyline | LOW |
| **Minor type errors** | 6 easily fixable mypy errors | LOW |
| **Unwired telemetry** | `check_performance_degradation()` | LOW |

---

## Recommendations

### Immediate (This Week)

1. **Fix mypy errors** (30 min)
   - Add missing `datetime` import
   - Fix generator type in `historical_env_detail.py`
   - Correct async return type in `app.py`

2. **Remove dead event types** (1 hr)
   - Delete `ISOLATION_VIOLATION`, `GOVERNOR_PANIC`, `GOVERNOR_SNAPSHOT`, `CHECKPOINT_SAVED`
   - Remove associated formatting code in `nissa/output.py`

### Short-Term (This Month)

3. **Decompose `vectorized.py`** (1-2 days)
   - Extract environment management helpers
   - Extract checkpoint logic
   - Extract telemetry emission
   - Keep core training loop focused

4. **Wire missing telemetry** (2 hrs)
   - Call `check_performance_degradation()` at epoch end

### Long-Term (Backlog)

5. **Add module docstrings** to key files
6. **Split `tamiyo_brain.py`** into sub-widgets
7. **Document LSTM state permutation** fragility

---

## Deliverables Produced

| Document | Purpose |
|----------|---------|
| `00-coordination.md` | Analysis plan and execution log |
| `01-discovery-findings.md` | Holistic assessment, tech stack |
| `02-subsystem-catalog.md` | Detailed domain entries with APIs |
| `03-diagrams.md` | C4 architecture diagrams (10 diagrams) |
| `04-final-report.md` | This executive summary |
| `05-quality-assessment.md` | Code quality metrics and debt |
| `06-architect-handover.md` | Actionable briefing for architects |

---

## Conclusion

Esper is a **well-architected framework** that has grown 2.6x in 15 days while maintaining architectural discipline. The biological metaphor provides clear domain boundaries, and the type system enforces contracts rigorously.

The codebase is **ready for continued development** with only minor cleanup needed. The primary concern—`vectorized.py` at 3,404 LOC—is a tractable decomposition task that won't require architectural changes.

**Confidence Level:** HIGH (90%)

The analysis was conducted with parallel subagent exploration of all major domains, cross-referenced against source code and configuration files.
