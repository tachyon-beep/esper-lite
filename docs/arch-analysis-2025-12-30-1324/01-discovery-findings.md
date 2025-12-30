# Esper Architecture Discovery Findings

**Analysis Date:** 2025-12-30
**Codebase:** esper-lite
**Total Files:** 142 Python files + Vue/TypeScript frontend
**Lines of Code:** ~35,000+ (Python) + ~5,000 (TypeScript/Vue)

## Executive Summary

Esper is a **Morphogenetic AI Framework** - a novel approach to neural network training where models dynamically grow, prune, and adapt their topology during training. Rather than static architecture engineering, Esper implements an "architectural ecology" where neural modules ("seeds") compete for integration into a host network through a biologically-inspired lifecycle.

The system implements **Phase 2.5** of its roadmap (multi-slot with reward shaping research), with PPO-based reinforcement learning controlling seed lifecycle decisions.

---

## 1. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Core ML** | PyTorch 2.x | Neural network training, torch.compile support |
| **RL Algorithm** | PPO (Proximal Policy Optimization) | On-policy seed lifecycle control |
| **Package Manager** | UV | Fast Python dependency management |
| **TUI Framework** | Textual/Rich | Sanctum developer dashboard |
| **Web Framework** | Vue 3 + Vite | Overwatch monitoring dashboard |
| **WebSocket** | FastAPI/uvicorn | Real-time telemetry streaming |
| **Database** | DuckDB (in-memory) | MCP SQL query interface |
| **Testing** | pytest, Playwright | Unit + E2E testing |

---

## 2. Directory Structure

```
src/esper/
├── kasmina/        # Stem Cells - Slot mechanics, grafting, blueprints (1,800 LOC)
│   └── blueprints/ # CNN + Transformer seed modules
├── leyline/        # DNA/Genome - Shared contracts, enums, constants (2,500 LOC)
├── tamiyo/         # Brain/Cortex - Decision logic, policy networks (3,800 LOC)
│   ├── networks/   # LSTM actor-critic architecture
│   └── policy/     # PolicyBundle protocol, action masking
├── tolaria/        # Metabolism - Training execution, governor (600 LOC)
├── simic/          # Evolution - PPO training, rewards, attribution (13,500 LOC)
│   ├── agent/      # PPO agent, rollout buffer
│   ├── rewards/    # Reward computation, PBRS
│   ├── training/   # Vectorized multi-GPU training
│   ├── telemetry/  # Gradient collection, anomaly detection
│   └── attribution/# Counterfactual evaluation
├── nissa/          # Sensory Organs - Telemetry hub, diagnostics (1,200 LOC)
├── karn/           # Memory - Research telemetry, TUI, web (17,800 LOC)
│   ├── sanctum/    # Textual TUI for debugging
│   ├── overwatch/  # Vue 3 web dashboard
│   └── mcp/        # SQL query interface
├── runtime/        # Task specifications, model factories (500 LOC)
├── utils/          # Data loading, loss computation (800 LOC)
└── scripts/        # CLI entry points (400 LOC)
```

---

## 3. Domain Organization (7 Active + 2 Future)

### Active Domains

| Domain | Biological Role | Responsibility | Size |
|--------|-----------------|----------------|------|
| **Kasmina** | Stem Cells | Pluripotent slots, seed lifecycle, grafting mechanics | 1,800 LOC |
| **Leyline** | DNA/Genome | Shared contracts, enums, constants, telemetry schemas | 2,500 LOC |
| **Tamiyo** | Brain/Cortex | Decision-making (heuristic + learned), policy networks | 3,800 LOC |
| **Tolaria** | Metabolism | Training execution, fail-safe governor | 600 LOC |
| **Simic** | Evolution | PPO infrastructure, rewards, counterfactual attribution | 13,500 LOC |
| **Nissa** | Sensory Organs | Telemetry hub, diagnostic tracking, output routing | 1,200 LOC |
| **Karn** | Memory | Research telemetry, TUI (Sanctum), web dashboard (Overwatch) | 17,800 LOC |

### Future Domains (Planned)

| Domain | Biological Role | Purpose |
|--------|-----------------|---------|
| **Emrakul** | Immune System | Efficiency auditing, parasitic component removal |
| **Narset** | Endocrine System | Hormonal coordination, resource allocation signals |

---

## 4. Key Architectural Patterns

### 4.1 Biological Metaphor System

Esper uses **two complementary metaphors**:

1. **Body/Organism** (system architecture): Domains as organs within one organism
2. **Botanical** (seed lifecycle): Individual modules undergo plant-like development

```
DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → FOSSILIZED
                                      ↓
                                    PRUNED (failure)
```

### 4.2 Inverted Control Flow

Traditional RL loops iterate environments, then sample batches. Esper inverts this:

```python
# TRADITIONAL (Python GIL bottleneck)
for env in environments:
    batch = env.get_batch()
    process(batch)

# ESPER (Batch-first, GPU-optimal)
for batch in SharedBatchIterator(combined_batch_size):
    for env_i, env_state in enumerate(envs):
        env_state.train_on_batch(batch[env_i * slice_size:(i+1)*slice_size])
```

**Benefits:** Single DataLoader for N environments, reduces worker processes from N×M to M.

### 4.3 Factored Action Space

8-dimensional action space for fine-grained control:

| Head | Purpose | Size |
|------|---------|------|
| slot | Which slot to act on | num_slots (3) |
| blueprint | Seed architecture | 13 options |
| style | Blending algorithm | 4 options |
| tempo | Blending speed | 3 options |
| alpha_target | Target amplitude | 3 options |
| alpha_speed | Ramp speed | 4 options |
| alpha_curve | Easing function | 5 options |
| op | Lifecycle operation | 6 options |

### 4.4 Causal Masking for Credit Assignment

Per-head causal masks reduce gradient noise:
- Blueprint head only receives gradients during GERMINATE
- Alpha heads only active during SET_ALPHA
- Slot head always active (except WAIT)

### 4.5 Three-Tiered Telemetry

| Tier | Fidelity | When |
|------|----------|------|
| Episode Context | Minimal | Once per run |
| Epoch Snapshots | Standard | Every epoch |
| Dense Traces | Deep | On anomaly detection |

---

## 5. Entry Points

### CLI Commands

```bash
# Heuristic baseline (rule-based)
PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar10

# PPO training (learned policy)
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 \
    --rounds 100 \
    --envs 4 \
    --sanctum  # or --overwatch for web dashboard
```

### Key Presets

| Preset | Host | Purpose |
|--------|------|---------|
| `cifar10` | Weak CNN (8ch, 3 blocks) | Default, leaves headroom for seeds |
| `cifar10_deep` | Narrow deep CNN (8ch, 5 blocks) | Better GPU utilization |
| `cifar10_blind` | No spatial context (1×1 kernels) | Seeds restore spatial awareness |
| `tinystories` | Transformer (6 layers) | Language modeling domain |

---

## 6. Cross-Domain Dependencies

```
                    ┌──────────────┐
                    │   Leyline    │ ← Single source of truth
                    │  (DNA/Genome)│
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Kasmina │      │ Tamiyo  │      │  Nissa  │
    │(Stem    │      │(Brain/  │      │(Sensory │
    │ Cells)  │      │Cortex)  │      │ Organs) │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         └────────┬───────┴────────┬───────┘
                  │                │
                  ▼                ▼
             ┌─────────┐     ┌─────────┐
             │  Simic  │────▶│  Karn   │
             │(Evolution)    │(Memory) │
             └────┬────┘     └─────────┘
                  │
                  ▼
             ┌─────────┐
             │ Tolaria │
             │(Metabolism)
             └─────────┘
```

**Dependency Rules:**
- Leyline is imported by ALL domains (no domain imports Leyline)
- Tamiyo does NOT import Simic (decoupled via PolicyBundle protocol)
- Kasmina does NOT import Simic (decoupled via contracts.py protocols)
- Nissa routes events; backends (Karn) consume events

---

## 7. Configuration Sources

| Source | Purpose | Location |
|--------|---------|----------|
| CLI args | Runtime overrides | `scripts/train.py` |
| Presets | Hyperparameter bundles | `simic/training/config.py` |
| JSON config | Full configuration | `--config-json PATH` |
| Leyline constants | Training behavior defaults | `leyline/__init__.py` |
| Karn constants | TUI/display thresholds | `karn/constants.py` |
| Nissa profiles | Telemetry collection levels | `nissa/profiles.yaml` |

---

## 8. Testing Infrastructure

| Type | Framework | Location |
|------|-----------|----------|
| Unit tests | pytest | `tests/` |
| Property tests | Hypothesis | `tests/` |
| E2E (web) | Playwright | `karn/overwatch/web/` |
| Integration | pytest | `tests/integration/` |

---

## 9. Key Design Principles (from ROADMAP.md)

1. **Sensors match capabilities** - No blind growth (Nissa tracks everything Kasmina does)
2. **Complexity pays rent** - Parameter budget with penalty in reward
3. **GPU-first iteration** - Inverted control, SharedBatchIterator
4. **Progressive curriculum** - Small worlds → big worlds
5. **Train Anything protocol** - Host-agnostic via HostProtocol
6. **Morphogenetic plane** - One Kasmina plane, many slots
7. **Governor prevents catastrophe** - TolariaGovernor fail-safe
8. **Hierarchical scaling** - Kasmina/Narset/Tamiyo separation (future)
9. **Frozen Core economy** - Train once, adapt infinitely (future PEFT)

---

## 10. Complexity Assessment

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Python LOC | ~35,000 | Medium-large codebase |
| Domain count | 7 active | Well-factored |
| External deps | Moderate | PyTorch, Textual, Vue, FastAPI |
| Circular imports | None observed | Clean dependency graph |
| Test coverage | Partial | Unit + E2E, property tests present |
| Documentation | Good | README, ROADMAP, inline docstrings |

**Overall:** Well-architected research codebase with clear domain boundaries and biological metaphor consistency.
