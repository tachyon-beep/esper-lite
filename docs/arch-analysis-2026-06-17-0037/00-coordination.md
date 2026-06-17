# 00 — Coordination Plan

## Analysis Configuration
- **Scope**: All 8 active domains under `src/esper/` — `kasmina`, `tamiyo`, `simic`, `tolaria`, `nissa`, `karn`, `leyline`, plus `runtime`/`utils`/`scripts` as a support cluster.
- **Deliverables**: **Option C (Architect-Ready)** — discovery, subsystem catalog, C4 diagrams, final report, code-quality assessment, architect handover.
- **Depth**: All 8 active domains cataloged; each reviewed against the project's own "Nine Commandments" (ROADMAP.md) and the No-Legacy / No-Defensive-Programming policies (CLAUDE.md).
- **Strategy**: **PARALLEL** — 8 loosely-coupled domains, ~72K LOC, zero import cycles. Justifies fan-out.
- **Orchestration**: `ultracode` multi-agent Workflow (explicitly requested by user).
- **Complexity estimate**: **High** — nested-loop meta-RL system; 15+ god-files >1,000 LOC; deep telemetry stack.

## Source Facts (measured, not estimated)
- **~71,613 LOC** Python across `src/esper/` (excludes tests).
- **Loomweave index** @ git `e259169`: 10,989 entities, 27,976 edges, 316 micro-subsystems, 110 pre-extracted findings. **0 import cycles.**
- **Domain LOC**: simic 23,643 · karn 22,539 · leyline 7,277 · kasmina 5,821 · tamiyo 5,146 · nissa 3,153 · utils 1,285 · scripts 1,154 · tolaria 869 · runtime 704.
- **Coupling apex**: `esper.leyline` module (fan-in 197). Hottest functions: `compute_contribution_reward` (165), `SlotConfig.default` (158), `SanctumAggregator.get_snapshot` (141), `SeedSlot.germinate` (129).

## Orchestration Strategy (PARALLEL — justification)
| Criterion | Threshold | This codebase | Verdict |
|-----------|-----------|---------------|---------|
| Subsystem count | ≥5 → parallel | 8 active | PARALLEL |
| LOC | 20K+ → parallel | ~72K | PARALLEL |
| Coupling | loose → parallel | 0 cycles, layered via leyline | PARALLEL |

Workflow phases:
1. **Domain review** (8 parallel `codebase-explorer` agents → structured catalog entries).
2. **Validation** (per-domain `analysis-validator`, pipelined — mandatory multi-subsystem gate).
3. **Cross-cutting** (parallel specialists: `drl-expert` on Simic/rewards, `pytorch-expert` on GPU-first/tensors, commandments auditor, `debt-cataloger` on god-files, diagram/dependency synthesizer).
4. **Synthesis** (main loop authors the 6 deliverables from validated structured data).

## Execution Log
- [2026-06-17 00:37] Created workspace `docs/arch-analysis-2026-06-17-0037/`.
- [2026-06-17 00:37] Holistic scan: domain LOC, file manifest, Loomweave coupling/cycles, README + ROADMAP read.
- [2026-06-17 00:38] User selected **Option C**, **all 8 active domains**.
- [2026-06-17 00:38] Strategy fixed: PARALLEL via ultracode Workflow.
- [2026-06-17 00:39] Wrote `01-discovery-findings.md` (structure, stack, spine, constitutional claims to test).
- [2026-06-17 00:40] Launched workflow `wf_c3aafede-e75` (`esper-archaeology`): 9 domain reviewers (codebase-explorer) → 9 validators (analysis-validator) pipelined; 5 cross-cutting specialists (drl-expert RL soundness, pytorch-expert GPU-first, python-code-reviewer commandments audit, debt-cataloger god-files, codebase-explorer dependency map).
- [2026-06-17 00:54] Workflow complete: 25 agents, ~2.3M tokens, 833s. Validation gate: **3 PASS / 7 WARN / 0 BLOCK** (no blocking issues). Tolaria validator caught a commandment-numbering conflation — reconciled across all docs.
- [2026-06-17 00:55] Authored all deliverables with validator corrections merged: `02-subsystem-catalog.md`, `03-diagrams.md` (C4), `04-final-report.md`, `05-quality-assessment.md`, `06-architect-handover.md`. Validation summary written to `temp/validation-summary.md`.
- [2026-06-17 00:55] **Analysis COMPLETE.** Option C delivered in full.

## Final Deliverables
| Document | Status |
|----------|--------|
| 00-coordination.md | ✅ |
| 01-discovery-findings.md | ✅ |
| 02-subsystem-catalog.md | ✅ (10 units, all validated) |
| 03-diagrams.md | ✅ (C1–C4 + lifecycle FSM) |
| 04-final-report.md | ✅ |
| 05-quality-assessment.md | ✅ (scorecard + 18 findings + strengths) |
| 06-architect-handover.md | ✅ (P0–P3 plan, effort-sized, reviewer-routed) |
