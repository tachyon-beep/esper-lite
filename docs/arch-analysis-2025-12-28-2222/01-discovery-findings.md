# Discovery Findings

**Analysis Date:** 2025-12-28
**Previous Analysis:** 2025-12-13 (17,100 LOC → current ~38,600 Python LOC + ~8,700 Vue/TS LOC)

## Executive Summary

Esper is a **morphogenetic neural network framework** that dynamically grows, prunes, and adapts model topology during training. The architecture follows a biological metaphor with 7 domain "organs" plus support modules, organized around a reinforcement learning core (PPO) that learns optimal growth strategies.

### Key Changes Since December 13th

1. **Major Growth in Karn** (+6,600 LOC): Full TUI (Sanctum) and Vue web dashboard (Overwatch)
2. **Tamiyo Expansion** (+3,400 LOC): Neural policy networks added alongside heuristic controller
3. **Simic Maturation** (+5,500 LOC): Attribution system, control module, reward engineering
4. **Leyline Expansion** (+2,900 LOC): Richer contracts, telemetry payloads, observation schemas
5. **Overall**: ~2.3x Python LOC growth (17K → 38.6K), plus 8.7K Vue/TS

## Technology Stack

### Core Languages & Frameworks

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | ≥3.11 | Primary implementation language |
| PyTorch | ≥2.8.0 | Deep learning framework |
| Torchvision | ≥0.18.0 | CIFAR-10 data loading |
| Vue 3 | Latest | Overwatch web dashboard |
| TypeScript | Latest | Type-safe frontend code |

### Key Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| RL/ML | torch, numpy | Tensor operations, PPO implementation |
| Data | datasets, transformers | TinyStories task support |
| Config | pydantic, PyYAML | Configuration validation |
| Telemetry | DuckDB | Analytical database for training metrics |
| TUI | textual, rich | Sanctum TUI, console output |
| Dashboard | FastAPI, uvicorn, websockets | Real-time web dashboard |
| MCP | mcp | Claude Code integration for telemetry queries |
| Testing | pytest, hypothesis | Unit and property-based testing |

### Development Tools

- **Package Manager:** UV (modern Python package management)
- **Linting:** Ruff (≥0.14.9)
- **Type Checking:** mypy
- **Mutation Testing:** mutmut v3 (Tamiyo + Kasmina core)
- **Frontend Build:** Vite + npm

## Directory Structure

```
src/esper/
├── karn/           # 8,341 LOC - Memory/Telemetry (TUI, Dashboard, Analytics)
│   ├── mcp/        # MCP server for Claude Code integration
│   ├── sanctum/    # Textual TUI for developer debugging
│   │   └── widgets/# 16 widget components
│   └── overwatch/  # Vue 3 web dashboard
│       └── web/    # 8,722 LOC Vue/TS frontend
├── kasmina/        # 5,174 LOC - Stem Cells (Slot mechanics, Blueprints)
│   └── blueprints/ # Neural module templates
├── leyline/        # 3,735 LOC - DNA (Contracts, Enums, Schemas)
├── nissa/          # 1,969 LOC - Sensory Organs (Observability hub)
├── simic/          # 13,352 LOC - Evolution (RL infrastructure)
│   ├── agent/      # PPO agent implementation
│   ├── attribution/# Reward attribution system
│   ├── control/    # Training flow control
│   ├── rewards/    # Reward engineering
│   ├── telemetry/  # RL-specific metrics
│   └── training/   # Training loop orchestration
├── tamiyo/         # 3,811 LOC - Brain/Cortex (Decision logic)
│   ├── networks/   # Neural policy architectures
│   └── policy/     # Policy implementations
├── tolaria/        # 462 LOC - Metabolism (Training execution)
├── runtime/        # 298 LOC - Task presets
├── scripts/        # 680 LOC - CLI entry points
└── utils/          # 802 LOC - Shared utilities

tests/              # 59,411 LOC - Comprehensive test suite
```

## Subsystems Identified (10)

| # | Subsystem | LOC | Biological Role | Status |
|---|-----------|-----|-----------------|--------|
| 1 | **Leyline** | 3,735 | DNA/Genome | Active |
| 2 | **Kasmina** | 5,174 | Stem Cells | Active |
| 3 | **Tamiyo** | 3,811 | Brain/Cortex | Active |
| 4 | **Tolaria** | 462 | Metabolism | Active |
| 5 | **Simic** | 13,352 | Evolution | Active |
| 6 | **Nissa** | 1,969 | Sensory Organs | Active |
| 7 | **Karn (Python)** | 8,341 | Memory | Active |
| 8 | **Karn (Vue/TS)** | 8,722 | Memory (Frontend) | Active |
| 9 | **Runtime** | 298 | Task Configuration | Active |
| 10 | **Scripts** | 680 | CLI Interface | Active |

**Total Python LOC:** 38,624
**Total Vue/TS LOC:** 8,722
**Total Test LOC:** 59,411
**Grand Total:** 106,757

## Entry Points

### Primary CLI

```bash
# PPO Training (main entry)
PYTHONPATH=src python -m esper.scripts.train ppo [OPTIONS]

# Heuristic Baseline
PYTHONPATH=src python -m esper.scripts.train heuristic [OPTIONS]
```

### Monitoring Interfaces

1. **Rich TUI** (default): Console-based training dashboard
2. **Sanctum** (`--sanctum`): Textual TUI for developer debugging
3. **Overwatch** (`--overwatch`): Vue 3 web dashboard at localhost:8080
4. **Dashboard** (`--dashboard`): FastAPI WebSocket dashboard at localhost:8000

### MCP Integration

```bash
# Karn MCP server for Claude Code telemetry queries
python -m esper.karn.mcp
```

## Architectural Patterns

### 1. Domain-Driven Design

Seven domains (plus two future: Emrakul, Narset) with biological metaphors:
- Clear separation of concerns
- Each domain has distinct responsibility
- Cross-domain communication via contracts in Leyline

### 2. Protocol-Based Decoupling

`HostProtocol` interface enables Kasmina to work with any model architecture:
- `injection_specs`: Available growth sites
- `register_slot()`: Slot registration
- `forward_with_slots()`: Execution integration

### 3. Lifecycle State Machine

Seeds progress through botanical stages:
```
DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → FOSSILIZED
                    ↓           ↓
                  PRUNED → EMBARGOED → RESETTING → DORMANT
```

### 4. GPU-First Iteration

Inverted control flow for CUDA throughput:
- DataLoader iteration first
- Environment dispatch second
- Pre-allocated GPU buffers for communication

### 5. Telemetry-Driven Development

Rich observability across all subsystems:
- Nissa: Central telemetry hub
- Karn: Storage, analytics, visualization
- MCP: Claude Code integration for queries

## Growth Since Previous Analysis

| Subsystem | Dec 13 LOC | Dec 28 LOC | Change |
|-----------|------------|------------|--------|
| Karn | ~1,700 | 8,341 | +6,641 (+390%) |
| Kasmina | ~1,400 | 5,174 | +3,774 (+270%) |
| Tamiyo | ~411 | 3,811 | +3,400 (+827%) |
| Simic | ~7,800 | 13,352 | +5,552 (+71%) |
| Leyline | ~844 | 3,735 | +2,891 (+343%) |
| Nissa | ~1,700 | 1,969 | +269 (+16%) |
| Tolaria | ~746 | 462 | -284 (-38%) |
| Runtime | ~224 | 298 | +74 (+33%) |
| Scripts | ~1,021 | 680 | -341 (-33%) |
| Utils | ~571 | 802 | +231 (+40%) |

**Notable:**
- Tolaria shrunk (refactoring to Simic?)
- Scripts shrunk (logic moved to library code?)
- Tamiyo grew 8x (neural policies added)
- Karn added full TUI + web dashboard

## Dependency Flow (Preliminary)

```
┌─────────────────────────────────────────────────┐
│                    Scripts                       │
│               (CLI Entry Point)                  │
└─────────────────────┬───────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │ Tamiyo  │ │  Simic  │ │  Karn   │
     │ (Brain) │ │ (Evol.) │ │ (Memory)│
     └────┬────┘ └────┬────┘ └────┬────┘
          │           │           │
          └─────┬─────┴───────────┘
                ▼
          ┌──────────┐
          │ Kasmina  │
          │(Stem Cell)│
          └────┬─────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Tolaria │ │  Nissa  │ │ Runtime │
│ (Metab.)│ │ (Sense) │ │ (Config)│
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┴───────────┘
                 │
                 ▼
          ┌──────────┐
          │ Leyline  │
          │  (DNA)   │
          └──────────┘
```

## Next Steps

1. **Detailed Subsystem Catalog** - Deep-dive into each domain
2. **Dependency Analysis** - Verify and document cross-domain imports
3. **C4 Diagrams** - Updated architecture visualizations
4. **Quality Assessment** - Code patterns, test coverage, debt catalog
5. **Architect Handover** - Actionable briefing document
