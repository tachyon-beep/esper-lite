# Esper V1.0 Holistic Codebase Discovery

This directory contains systematic discovery analysis of the Esper codebase as of **2025-11-28**.

## Documents

### 1. **SUMMARY.md** (3.2 KB)
Quick reference executive summary. Start here for a 5-minute overview.
- What is Esper?
- 5 core subsystems (overview table)
- Key innovation (gradient isolation + alpha blending)
- 6 key design patterns
- Strengths + concerns
- Quick confidence assessment

### 2. **01-discovery-findings.md** (15 KB)
Comprehensive discovery document with full analysis.

**Sections:**
- Executive Summary
- Directory Structure (tree + organization pattern)
- Technology Stack (languages, frameworks, PyTorch patterns)
- Identified Subsystems (5-subsystem table with responsibilities)
- Entry Points (orchestrators, CLI scripts, module APIs)
- Dependency Hierarchy (import flows, hot path constraints)
- Architectural Patterns (10 design patterns observed)
- Initial Observations (strengths, concerns, design tensions, open questions)
- Confidence Level assessment

## Quick Facts

| Metric | Value |
|--------|-------|
| **Total LOC** | 9,146 |
| **Python Files** | 34 |
| **Packages** | 5 (leyline, kasmina, tamiyo, simic, nissa) |
| **Largest Package** | simic (4,615 LOC) |
| **Naming Convention** | Planeswalker characters (Magic: The Gathering lore) |
| **Core Framework** | PyTorch 2.0+ |
| **Python Version** | 3.11+ |
| **Architecture Pattern** | Domain-Driven Design (5-tier) |
| **FSM States** | 9 (DORMANT, GERMINATED, TRAINING, BLENDING, etc.) |
| **Key Innovation** | Gradient isolation hooks + alpha blending schedules |
| **Hot Path** | simic/features.py (27-dim feature extraction) |
| **Analysis Confidence** | 85% High |

## Key Subsystems at a Glance

```
┌─────────────────────────────────────────────────────┐
│  simic_overnight.py (Orchestrator)                  │
├─────────────────────────────────────────────────────┤
│  ╔════════════════════════════════════════════════╗ │
│  ║ LEYLINE (1.1 KLOC)                             ║ │
│  ║ Data contracts: FSM, signals, schemas          ║ │
│  ╚════════════════════════════════════════════════╝ │
│    ↓                                               │
│  ╔════════════════════════════════════════════════╗ │
│  ║ KASMINA (1.2 KLOC)                             ║ │
│  ║ Seed lifecycle: germinate→train→blend→fossilize║ │
│  ║ Gradient isolation, alpha blending             ║ │
│  ╚════════════════════════════════════════════════╝ │
│                                                   │
│  ╔════════════════════════════════════════════════╗ │
│  ║ TAMIYO (501 LOC)                               ║ │
│  ║ Decision engine: heuristic + learned policies  ║ │
│  ║ Signal tracking, decision history              ║ │
│  ╚════════════════════════════════════════════════╝ │
│    ↓                                               │
│  ╔════════════════════════════════════════════════╗ │
│  ║ SIMIC (4.6 KLOC)                               ║ │
│  ║ RL training: PPO/IQL, episode collection       ║ │
│  ║ Reward shaping, feature extraction             ║ │
│  ╚════════════════════════════════════════════════╝ │
│                                                   │
│  ╔════════════════════════════════════════════════╗ │
│  ║ NISSA (358 LOC)                                ║ │
│  ║ Telemetry hub: gradient health, metrics        ║ │
│  ║ Output backends (console, file)                ║ │
│  ╚════════════════════════════════════════════════╝ │
└─────────────────────────────────────────────────────┘
```

## Discovery Methodology

**Systematic scan** across codebase:
1. File enumeration (34 Python files)
2. Line counting per package
3. Public API review via `__init__.py` files (8 files read)
4. Core algorithm inspection (signals, stages, slot lifecycle)
5. Entry point analysis (scripts, orchestrators)
6. Dependency graph extraction (import analysis)
7. Pattern identification (design patterns, constraints)
8. Test coverage assessment (test files scanned)

**Total Analysis Time**: ~30 minutes
**Confidence Level**: 85% (High) - see document for gaps

## Where to Go From Here

### For Understanding the System
1. Read **SUMMARY.md** first (5 min)
2. Review **Directory Structure** section in discovery document
3. Study **Architectural Patterns** section for design insights
4. Check README.md in project root for usage examples

### For Deep Dives
- **Seed Lifecycle**: kasmina/slot.py + kasmina/isolation.py
- **Decision Making**: tamiyo/heuristic.py + simic_overnight.py
- **RL Training**: simic/ppo.py + simic/episodes.py
- **Telemetry**: nissa/tracker.py + nissa/output.py
- **Hot Path**: simic/features.py (27-feature extraction)

### For Contribution Planning
- **Test Coverage**: Expand test_leyline.py, test_simic.py
- **Documentation**: Add docstrings to PPO/IQL algorithms
- **Error Handling**: Add exception paths in core lifecycle
- **Integration**: Explore datagen system integration

## Quick Links

- **Main Package**: `/home/john/esper-lite/src/esper/`
- **Tests**: `/home/john/esper-lite/tests/`
- **README**: `/home/john/esper-lite/README.md`
- **pyproject.toml**: `/home/john/esper-lite/pyproject.toml`

---

**Discovery Date**: 2025-11-28 22:23 UTC
**Analysis Scope**: Full source tree (primary + secondary)
**Exclusions**: _archive/, __pycache__/, .venv/
