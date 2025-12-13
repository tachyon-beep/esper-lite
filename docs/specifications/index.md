# Esper Module Bibles

> **Purpose:** Comprehensive, AI-consumable documentation for each Esper subsystem.
>
> **Format:** [Unified Module Bible (UMB) v1.0](_TEMPLATE.md)

---

## Quick Reference

| Module | Biological Role | Layer | Criticality | Status |
|--------|-----------------|-------|-------------|--------|
| [Kasmina](kasmina.md) | Stem Cell | Core Logic | Tier-0 | 📝 Pending |
| [Leyline](leyline.md) | DNA/Proteins | Infrastructure | Tier-0 | 📝 Pending |
| [Tamiyo](tamiyo.md) | Gardener | Control | Tier-1 | ✅ Complete |
| [Tolaria](tolaria.md) | Metabolism | Core Logic | Tier-0 | 📝 Pending |
| [Simic](simic.md) | Evolution | Core Logic | Tier-0 | ✅ Complete |
| [Nissa](nissa.md) | Sensory Organs | Observation | Tier-2 | 📝 Pending |
| [Karn](karn.md) | Memory/Archivist | Observation | Tier-2 | 📝 Pending |

**Status Legend:**
- ✅ Complete - Bible written and reviewed
- 🔄 In Progress - Being written or updated
- 📝 Pending - Not yet created
- ⚠️ Stale - Needs update (code changed since last review)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ESPER SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    decisions    ┌─────────┐                       │
│   │ TAMIYO  │ ──────────────► │ TOLARIA │                       │
│   │ (Brain) │                 │ (Hands) │                       │
│   └────┬────┘                 └────┬────┘                       │
│        │                           │                             │
│        │ observes                  │ executes on                 │
│        ▼                           ▼                             │
│   ┌─────────┐    grafts to    ┌─────────┐                       │
│   │  SIMIC  │ ◄─────────────► │ KASMINA │                       │
│   │  (Gym)  │                 │ (Body)  │                       │
│   └────┬────┘                 └────┬────┘                       │
│        │                           │                             │
│        │ trains                    │ uses types from             │
│        ▼                           ▼                             │
│   ┌─────────┐                 ┌─────────┐                       │
│   │  NISSA  │ ◄───────────────│ LEYLINE │                       │
│   │(Senses) │   observes      │ (DNA)   │                       │
│   └────┬────┘                 └─────────┘                       │
│        │                                                         │
│        │ feeds                                                   │
│        ▼                                                         │
│   ┌─────────┐                                                    │
│   │  KARN   │                                                    │
│   │(Memory) │                                                    │
│   └─────────┘                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Summaries

### Kasmina - The Body (Stem Cell)
**Location:** `src/esper/kasmina/`

The morphogenetic model layer. Manages seed slots, blueprint instantiation, grafting mechanics, and the physical structure of growing neural networks.

**Key Concepts:** `SeedSlot`, `MorphogeneticModel`, `HostProtocol`, `Blueprint`

**Interacts With:** Leyline (types), Tolaria (training), Tamiyo (decisions)

---

### Leyline - The DNA (Shared Contracts)
**Location:** `src/esper/leyline/`

Infrastructure layer providing shared types, enums, tensor schemas, and data contracts used across all modules.

**Key Concepts:** `SeedStage`, `SlotID`, `FactoredActions`, observation schemas

**Interacts With:** All modules (foundational dependency)

---

### Tamiyo - The Brain (Nervous System)
**Location:** `src/esper/tamiyo/`

Strategic decision-making logic. Can be heuristic (rule-based) or neural (learned policy). Decides WHEN to germinate, advance, or cull seeds.

**Key Concepts:** `HeuristicController`, decision thresholds, lifecycle triggers

**Interacts With:** Simic (receives policy), Leyline (uses types), Tolaria (sends decisions)

---

### Tolaria - The Hands (Metabolism)
**Location:** `src/esper/tolaria/`

Execution engine running PyTorch training loops. Converts decisions into actual gradient updates and model modifications.

**Key Concepts:** `Trainer`, `TolariaGovernor`, training loop, optimizer management

**Interacts With:** Kasmina (modifies model), Tamiyo (receives decisions), Nissa (emits events)

---

### Simic - The Gym (Evolution)
**Location:** `src/esper/simic/`

RL infrastructure providing PPO training, vectorized environments, reward computation, and policy optimization.

**Key Concepts:** `PPOAgent`, `VectorizedEnv`, `RolloutBuffer`, reward shaping

**Interacts With:** Tamiyo (provides policy), Leyline (uses actions), Karn (emits telemetry)

---

### Nissa - The Senses (Sensory Organs)
**Location:** `src/esper/nissa/`

Observability hub routing telemetry and generating diagnostic narratives. Provides gradient health, loss landscape analysis, and training signals.

**Key Concepts:** Profiles, gradient diagnostics, telemetry routing

**Interacts With:** All modules (receives events), Karn (feeds data)

---

### Karn - The Memory (Archivist)
**Location:** `src/esper/karn/`

Research telemetry system with analytics, health monitoring, TUI dashboard, and web interface. Persists and visualizes training history.

**Key Concepts:** `TelemetryCollector`, `TelemetryStore`, TUI, WebSocket dashboard

**Interacts With:** Nissa (receives events), Simic (receives PPO metrics)

---

## Dependency Graph

```
             CONSUMERS
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌───────┐   ┌───────┐   ┌───────┐
│Tamiyo │   │Tolaria│   │ Simic │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    │     ┌─────┴─────┐     │
    │     ▼           ▼     │
    │ ┌───────┐   ┌───────┐ │
    └─│Kasmina│   │ Nissa │─┘
      └───┬───┘   └───┬───┘
          │           │
          │     ┌─────┘
          ▼     ▼
      ┌───────────┐
      │  Leyline  │  (Foundation - no dependencies)
      └───────────┘
            │
            ▼
      ┌───────────┐
      │   Karn    │  (Telemetry sink)
      └───────────┘
```

---

## Nine Commandments Coverage

Track which commandments each module addresses:

| Commandment | Kasmina | Leyline | Tamiyo | Tolaria | Simic | Nissa | Karn |
|-------------|---------|---------|--------|---------|-------|-------|------|
| 1. Sensors match capabilities | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | 🔵 | 🔵 |
| 2. Complexity pays rent | 🔵 | ⚪ | ⚪ | ⚪ | 🔵 | ⚪ | ⚪ |
| 3. GPU-first iteration | 🔵 | ⚪ | ⚪ | 🔵 | 🔵 | ⚪ | ⚪ |
| 4. Progressive curriculum | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 5. Train Anything protocol | 🔵 | 🔵 | ⚪ | 🔵 | ⚪ | ⚪ | ⚪ |
| 6. Morphogenetic plane | 🔵 | 🔵 | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 7. Governor prevents catastrophe | ⚪ | ⚪ | ⚪ | 🔵 | ⚪ | ⚪ | ⚪ |
| 8. Hierarchical scaling | 🔵 | ⚪ | 🔵 | ⚪ | ⚪ | ⚪ | ⚪ |
| 9. Frozen Core economy | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |

**Legend:** 🔵 Primary implementer | ⚪ Not applicable or secondary

---

## Maintenance

### When to Update a Bible

- **Code Changes:** Any modification to public API, tensor shapes, or state machines
- **New Events:** Adding telemetry events or pub/sub topics
- **Bug Discoveries:** New entries for Tribal Knowledge section
- **Dependency Changes:** Upstream/downstream relationships modified

### Review Cadence

- **Active Development:** Review weekly
- **Stable Modules:** Review monthly or on significant changes
- **Archived Modules:** Review on reactivation only

### Staleness Detection

A bible is considered stale when:
1. `last_reviewed_commit` is >20 commits behind HEAD
2. Source files have been modified since `last_updated`
3. Related bibles have been updated but cross-references weren't checked

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `_TEMPLATE.md` | UMB specification and blank template |
| `index.md` | This navigation file |
| `kasmina.md` | Kasmina module bible |
| `leyline.md` | Leyline module bible |
| `tamiyo.md` | Tamiyo module bible |
| `tolaria.md` | Tolaria module bible |
| `simic.md` | Simic module bible |
| `nissa.md` | Nissa module bible |
| `karn.md` | Karn module bible |
