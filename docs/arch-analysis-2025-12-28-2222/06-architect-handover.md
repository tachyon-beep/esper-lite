# Architect Handover Briefing

**Analysis Date:** 2025-12-28
**Target Audience:** New engineers, architects, or AI assistants onboarding to Esper
**Reading Time:** 10-15 minutes

---

## Quick Start

### What Is Esper?

Esper is a **morphogenetic neural network framework** that:
1. Dynamically injects "seed" modules into neural networks during training
2. Uses reinforcement learning (PPO) to learn when/where/how to grow
3. Manages seed lifecycle through botanical stages (germinate → fossilize)
4. Provides rich observability via TUI, web dashboard, and SQL analytics

### One-Minute Architecture

```
CLI (train.py)
     │
     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│  SIMIC  │────►│ TAMIYO  │────►│ KASMINA │
│ (PPO)   │     │ (Brain) │     │ (Seeds) │
└────┬────┘     └─────────┘     └────┬────┘
     │                               │
     │         ┌─────────┐           │
     └────────►│  NISSA  │◄──────────┘
               │ (Events)│
               └────┬────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│  KARN   │   │ TOLARIA │   │ LEYLINE │
│  (TUI)  │   │ (Guard) │   │ (Types) │
└─────────┘   └─────────┘   └─────────┘
```

### Running Esper

```bash
# Install
uv sync

# Train with PPO (main workflow)
PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10

# With TUI monitoring
PYTHONPATH=src uv run python -m esper.scripts.train ppo --sanctum

# With web dashboard
PYTHONPATH=src uv run python -m esper.scripts.train ppo --overwatch
```

---

## Domain Guide

### Leyline (DNA) - READ FIRST

**Location:** `src/esper/leyline/`
**LOC:** 3,735
**Role:** Foundation layer with ALL contracts

**Key exports:**
- `SeedStage` enum (DORMANT, GERMINATED, TRAINING, BLENDING, HOLDING, FOSSILIZED, PRUNED)
- `FactoredAction` for 8-head policy outputs
- `TelemetryEvent` + 18 typed payload classes
- All `DEFAULT_*` constants

**Critical rule:** Leyline has NO outbound dependencies. Add new shared types here.

### Kasmina (Stem Cells)

**Location:** `src/esper/kasmina/`
**LOC:** 5,174
**Role:** Seed lifecycle management

**Key classes:**
- `SeedSlot` - The core lifecycle engine (2,610 LOC)
- `MorphogeneticModel` - Multi-slot host wrapper
- `AlphaController` - Time-based blend scheduling (LINEAR, COSINE, SIGMOID)

**Key patterns:**
- State machine with quality gates (G0-G5)
- Composition operators (ADD, MULTIPLY, GATE)
- DDP synchronization for distributed training

### Tamiyo (Brain)

**Location:** `src/esper/tamiyo/`
**LOC:** 3,811
**Role:** Strategic decision-making

**Key abstractions:**
- `PolicyBundle` protocol - Swappable heuristic or neural
- `HeuristicTamiyo` - Rule-based controller
- `FactoredRecurrentActorCritic` - LSTM neural policy
- `SignalTracker` - Training metric aggregation

**Decision outputs:** `TamiyoDecision(germinate=[slots], prune=[slots], advance=[slots])`

### Simic (Evolution)

**Location:** `src/esper/simic/`
**LOC:** 13,352
**Role:** PPO training infrastructure

**Subdirectories:**
- `agent/` - PPOAgent, RolloutBuffer, GAE
- `training/` - Vectorized training loop (3,404 LOC main file)
- `rewards/` - Contribution rewards, PBRS, reward modes
- `attribution/` - Counterfactual Shapley values
- `telemetry/` - Gradient collectors, anomaly detection

**Key pattern:** GPU-first iteration (DataLoader → Environments → Model)

### Karn (Memory)

**Location:** `src/esper/karn/`
**LOC:** 8,341 Python + 8,722 Vue/TS
**Role:** Telemetry, storage, visualization

**Components:**
- **Root:** TelemetryStore, KarnCollector
- **sanctum/** - Textual TUI (16 widgets)
- **overwatch/** - Vue 3 web dashboard
- **mcp/** - DuckDB SQL interface for Claude Code

**MCP views:** `runs`, `epochs`, `ppo_updates`, `seed_lifecycle`, `rewards`, `anomalies`

### Nissa (Sensory)

**Location:** `src/esper/nissa/`
**LOC:** 1,969
**Role:** Telemetry hub

**Key class:** `NissaHub` - Pub-sub event router with async queue

**Backends:** ConsoleOutput, FileOutput, DirectoryOutput

### Tolaria (Metabolism)

**Location:** `src/esper/tolaria/`
**LOC:** 462
**Role:** Training fail-safe

**Key class:** `TolariaGovernor` - Anomaly detection, RAM checkpointing, rollback

**Critical contract:** Optimizer state must be cleared after rollback.

---

## Key Concepts

### Seed Lifecycle State Machine

```
DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → FOSSILIZED
                         ↓
                       PRUNED → EMBARGOED → RESETTING → DORMANT
```

Valid transitions are enforced by `VALID_TRANSITIONS` dict in Leyline.

### Quality Gates

| Gate | Transition | Purpose |
|------|------------|---------|
| G0 | → GERMINATED | Entry validation |
| G1 | → TRAINING | Readiness check |
| G2 | → BLENDING | Eligibility check |
| G3 | → HOLDING | Stability check |
| G4 | Pre-FOSSILIZED | Final check |
| G5 | → FOSSILIZED | Permanent fusion |

### Factored Action Space

8-head policy with per-head credit assignment:
0. slot_id, 1. blueprint, 2. style, 3. tempo, 4. alpha_curve, 5. alpha_duration, 6. lifecycle_op, 7. value

Causal masking ensures only active heads receive gradients.

### Alpha Scheduling

Blend schedule for seed integration:
- **LINEAR:** Constant rate
- **COSINE:** Smooth acceleration/deceleration
- **SIGMOID:** S-curve for gradual transitions

---

## Where to Find Things

| Need | Location |
|------|----------|
| Constants, enums, types | `src/esper/leyline/` |
| Seed lifecycle code | `src/esper/kasmina/slot.py` |
| Policy decision logic | `src/esper/tamiyo/policy/` |
| PPO agent | `src/esper/simic/agent/ppo.py` |
| Training loop | `src/esper/simic/training/vectorized.py` |
| Reward engineering | `src/esper/simic/rewards/rewards.py` |
| Telemetry emission | `src/esper/nissa/output.py` |
| TUI widgets | `src/esper/karn/sanctum/widgets/` |
| Web dashboard | `src/esper/karn/overwatch/web/` |
| Task presets | `src/esper/runtime/tasks.py` |
| CLI entry | `src/esper/scripts/train.py` |

---

## Common Tasks

### Adding a New Telemetry Event

1. Add event type to `leyline/telemetry.py`:
   ```python
   class TelemetryEventType(Enum):
       MY_NEW_EVENT = auto()
   ```

2. Create typed payload dataclass:
   ```python
   @dataclass
   class MyNewEventPayload:
       field1: int
       field2: str
   ```

3. Add to `TYPED_PAYLOAD_CLASSES` mapping

4. Emit via Nissa:
   ```python
   from esper.nissa import emit
   emit(TelemetryEvent(
       event_type=TelemetryEventType.MY_NEW_EVENT,
       payload=MyNewEventPayload(field1=42, field2="hello"),
   ))
   ```

### Adding a New Blueprint

1. Create blueprint function in `kasmina/blueprints/cnn.py` or `transformer.py`:
   ```python
   @blueprint("my_seed")
   def my_seed_blueprint(host: HostProtocol, spec: InjectionSpec) -> nn.Module:
       # Return a new module to inject
       return nn.Linear(spec.in_features, spec.out_features)
   ```

2. Blueprint is automatically registered via decorator

### Adding a New Quality Gate

1. Add gate check in `kasmina/slot.py`:
   ```python
   def _check_g6(self) -> bool:
       """My new gate."""
       return self.some_condition
   ```

2. Wire into `advance_stage()` transitions

### Querying Telemetry with MCP

```bash
# Start MCP server
python -m esper.karn.mcp

# Query from Claude Code
mcp__esper-karn__query_sql("SELECT * FROM epochs WHERE epoch_number > 10")
```

---

## Gotchas and Pitfalls

### 1. Leyline Import Order

Leyline's `__init__.py` has careful import ordering. Don't add cross-imports between Leyline submodules without checking for cycles.

### 2. DDP Synchronization

When modifying SeedSlot in distributed training:
- Use `_sync_gate_decision()` for consensus
- Don't break the "all ranks agree" invariant

### 3. LSTM State Permutation

The LSTM hidden state management in Simic has fragile permutation logic. Document any changes carefully.

### 4. Optimizer State After Rollback

After `TolariaGovernor.execute_rollback()`, optimizer momentum buffers survive `load_state_dict()`. The optimizer must be re-created or cleared.

### 5. Telemetry Queue Overflow

Under extreme load, `NissaHub` will drop events. This is documented but can cause missing telemetry in CI stress tests.

---

## Testing Guidance

### Running Tests

```bash
# All tests
PYTHONPATH=src uv run pytest

# Fast subset (deselects slow)
PYTHONPATH=src uv run pytest -m "not slow"

# Specific domain
PYTHONPATH=src uv run pytest tests/kasmina/

# With mutation testing
uv run mutmut run
```

### Test Structure

Tests mirror source structure:
- `tests/kasmina/` → `src/esper/kasmina/`
- `tests/simic/` → `src/esper/simic/`
- etc.

### Property-Based Testing

Hypothesis is used extensively. Check `conftest.py` for strategies.

---

## Technical Debt to Know

| Item | Priority | Time Estimate |
|------|----------|---------------|
| 6 mypy errors | HIGH | 30 min |
| Dead event types | LOW | 1 hr |
| `vectorized.py` decomposition | MEDIUM | 1-2 days |
| Missing `check_performance_degradation()` | LOW | 2 hrs |

---

## Architecture Decision Records

Key decisions documented in codebase:

1. **Biological metaphor** - Organism for architecture, botanical for seeds
2. **Leyline as foundation** - No legacy, no backwards compatibility
3. **Protocol-based decoupling** - Simic → Kasmina via contracts
4. **GPU-first iteration** - DataLoader drives, not environments
5. **Typed telemetry** - Dataclass payloads, not dicts

---

## Getting Help

1. **Architecture docs:** `docs/arch-analysis-2025-12-28-2222/`
2. **Roadmap:** `ROADMAP.md` (the "Nine Commandments")
3. **AI guidelines:** `CLAUDE.md`
4. **MCP server:** Query telemetry with SQL

---

## Summary Checklist

Before starting work, verify you understand:

- [ ] The 7 domain "organs" and their responsibilities
- [ ] Where to add new types (Leyline)
- [ ] The seed lifecycle state machine
- [ ] How telemetry flows (Nissa → Karn)
- [ ] The GPU-first training loop design
- [ ] The DDP synchronization requirements

Welcome to Esper. The codebase is well-structured—trust the metaphors and follow the protocols.
