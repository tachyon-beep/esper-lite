# Architecture Diagrams - Esper-Lite

## Diagram Key

This document presents the architecture of esper-lite using the C4 model (Context, Containers, Components, Code).

**C4 Levels:**
- **Level 1 (Context)**: System boundary and external actors
- **Level 2 (Container)**: Major subsystems/modules within esper-lite
- **Level 3 (Component)**: Internal components within key subsystems
- **Level 4 (Code)**: Detailed code-level structures (state machines, classes)

**Color Scheme:**
- **Blue**: Core domain subsystems (Kasmina, Leyline)
- **Green**: RL infrastructure (Simic)
- **Purple**: Decision-making (Tamiyo)
- **Orange**: Observability (Nissa)
- **Yellow**: Training orchestration (Tolaria)
- **Gray**: Utilities and external dependencies

---

## Level 1: System Context Diagram

```mermaid
flowchart TB
    subgraph External["External Actors"]
        Researcher["ML Researcher<br/><i>Designs and runs experiments</i>"]
        GPU["GPU Cluster<br/><i>Compute resources</i>"]
        Datasets["Training Datasets<br/><i>CIFAR-10, etc.</i>"]
        TensorBoard["TensorBoard<br/><i>Visualization</i>"]
    end

    Esper["<b>Esper-Lite</b><br/>Morphogenetic Neural Network<br/>Training Framework"]

    Researcher -->|"Configures experiments"| Esper
    Esper -->|"Executes training"| GPU
    Esper -->|"Loads data"| Datasets
    Esper -->|"Exports metrics"| TensorBoard
    Researcher -->|"Monitors"| TensorBoard
```

### Description

**Esper-Lite** is a morphogenetic neural network training framework that enables ML researchers to evolve neural architectures through a lifecycle-based approach. The system orchestrates training across GPU resources, automatically promoting or culling network variants (seeds) based on performance metrics.

---

## Level 2: Container Diagram

```mermaid
flowchart TB
    subgraph Esper["Esper-Lite Framework"]
        Leyline["<b>Leyline</b><br/>Data Contracts<br/><i>~600 LOC</i>"]
        Kasmina["<b>Kasmina</b><br/>Seed Mechanics<br/><i>~1100 LOC</i>"]
        Tamiyo["<b>Tamiyo</b><br/>Decision Engine<br/><i>~500 LOC</i>"]
        Simic["<b>Simic</b><br/>RL Infrastructure<br/><i>~4600 LOC</i>"]
        Tolaria["<b>Tolaria</b><br/>Training Loop<br/><i>~270 LOC</i>"]
        Nissa["<b>Nissa</b><br/>Telemetry Hub<br/><i>~1000 LOC</i>"]
        Utils["<b>Utils</b><br/>Data Loading<br/><i>~70 LOC</i>"]
    end

    subgraph External["External Dependencies"]
        PyTorch["PyTorch"]
        Pydantic["Pydantic"]
        Torchvision["Torchvision"]
    end

    Kasmina --> Leyline
    Tamiyo --> Leyline
    Tamiyo -.->|"types only"| Kasmina
    Simic --> Leyline
    Simic --> Tamiyo
    Simic --> Kasmina
    Simic --> Tolaria
    Tolaria --> Kasmina
    Nissa --> Leyline

    Kasmina --> PyTorch
    Simic --> PyTorch
    Tolaria --> PyTorch
    Nissa --> Pydantic
    Utils --> Torchvision
```

### Description

The framework is organized into 7 subsystems:

| Layer | Subsystem | Responsibility |
|-------|-----------|----------------|
| Foundation | **Leyline** | Contract-first types and protocols |
| Core Domain | **Kasmina** | Seed lifecycle, quality gates |
| Decision | **Tamiyo** | Heuristic action selection |
| RL | **Simic** | PPO/IQL training, orchestration |
| Training | **Tolaria** | Supervised learning loops |
| Observability | **Nissa** | Telemetry and metrics |
| Data | **Utils** | Dataset loading |

---

## Level 3: Component Diagrams

### Kasmina (Core Domain)

```mermaid
flowchart TB
    subgraph Kasmina["Kasmina Subsystem"]
        SeedSlot["<b>SeedSlot</b><br/>Lifecycle container<br/><i>slot.py</i>"]
        MorphModel["<b>MorphogeneticModel</b><br/>Host + Seed wrapper<br/><i>host.py</i>"]
        Gates["<b>QualityGates</b><br/>G0-G5 validation<br/><i>slot.py</i>"]
        Blueprints["<b>BlueprintCatalog</b><br/>Seed architectures<br/><i>blueprints.py</i>"]
        Isolation["<b>GradientIsolation</b><br/>Host protection<br/><i>isolation.py</i>"]
    end

    SeedSlot --> Gates
    SeedSlot --> MorphModel
    MorphModel --> Blueprints
    MorphModel --> Isolation
```

### Simic (RL Infrastructure)

```mermaid
flowchart TB
    subgraph Simic["Simic Subsystem"]
        PPO["<b>PPOAgent</b><br/>Online RL<br/><i>ppo.py - 1591 LOC</i>"]
        IQL["<b>IQL</b><br/>Offline RL<br/><i>iql.py - 1326 LOC</i>"]
        Episodes["<b>EpisodeCollector</b><br/>Trajectory data<br/><i>episodes.py</i>"]
        Rewards["<b>RewardComputer</b><br/>Shaped rewards<br/><i>rewards.py</i>"]
        Features["<b>FeatureExtractor</b><br/>HOT PATH<br/><i>features.py</i>"]
        Networks["<b>PolicyNetwork</b><br/>Actor-Critic<br/><i>networks.py</i>"]
    end

    Episodes --> PPO
    Episodes --> IQL
    Episodes --> Rewards
    Episodes --> Features
    PPO --> Networks
    IQL --> Networks
```

### Tamiyo (Decision Making)

```mermaid
flowchart LR
    subgraph Tamiyo["Tamiyo Subsystem"]
        Heuristic["<b>HeuristicTamiyo</b><br/>Rule-based decisions<br/><i>heuristic.py</i>"]
        Tracker["<b>SignalTracker</b><br/>Metric observer<br/><i>tracker.py</i>"]
        Decision["<b>TamiyoDecision</b><br/>Action output<br/><i>decisions.py</i>"]
    end

    Tracker --> Heuristic
    Heuristic --> Decision
```

---

## Level 4: Seed Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> DORMANT: Seed Created

    DORMANT --> GERMINATED: G0 Gate

    GERMINATED --> TRAINING: G1 Gate
    GERMINATED --> CULLED: Gate Failed

    TRAINING --> BLENDING: G2 Gate
    TRAINING --> CULLED: No Improvement

    BLENDING --> SHADOWING: G3 Gate
    BLENDING --> CULLED: Gate Failed

    SHADOWING --> PROBATIONARY: G4 Gate
    SHADOWING --> CULLED: Gate Failed

    PROBATIONARY --> FOSSILIZED: G5 Gate
    PROBATIONARY --> CULLED: Gate Failed

    FOSSILIZED --> [*]: Success
    CULLED --> [*]: Removed
```

### Gate Requirements

| Gate | Stage Transition | Requirements |
|------|------------------|--------------|
| G0 | DORMANT → GERMINATED | seed_id, blueprint_id present |
| G1 | GERMINATED → TRAINING | germination complete |
| G2 | TRAINING → BLENDING | improvement ≥0.5%, violations ≤10 |
| G3 | BLENDING → SHADOWING | ≥3 epochs, alpha ≥0.95 |
| G4 | SHADOWING → PROBATIONARY | shadowing complete |
| G5 | PROBATIONARY → FOSSILIZED | positive improvement, healthy |

---

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant S as Simic
    participant T as Tamiyo
    participant K as Kasmina
    participant L as Tolaria
    participant N as Nissa

    S->>K: Get seed states
    K-->>S: SeedStateReport[]

    S->>S: Extract features
    S->>T: Request decision
    T->>T: Evaluate heuristics
    T-->>S: TamiyoDecision

    S->>K: Execute action
    K->>L: Delegate training
    L->>L: Train epoch
    L-->>K: Metrics

    K->>K: Evaluate gates
    K-->>S: New state

    S->>S: Compute reward
    S->>S: Update policy

    S->>N: Log telemetry
```

### Training Loop Summary

1. **State Extraction**: Simic reads seed states from Kasmina
2. **Feature Extraction**: Convert metrics to RL observation
3. **Decision**: Tamiyo selects action (train/advance/cull)
4. **Execution**: Kasmina delegates to Tolaria for training
5. **Gate Evaluation**: Kasmina checks quality gates
6. **Reward**: Simic computes reward from improvement
7. **Policy Update**: PPO/IQL updates neural networks
8. **Telemetry**: Nissa logs all metrics

---

## Dependency Matrix

| Subsystem | Leyline | Kasmina | Tamiyo | Simic | Tolaria | Nissa | Utils |
|-----------|:-------:|:-------:|:------:|:-----:|:-------:|:-----:|:-----:|
| Leyline | - | | | | | | |
| Kasmina | ✓ | - | | | | | |
| Tamiyo | ✓ | (t) | - | | | | |
| Simic | ✓ | ✓ | ✓ | - | ✓ | | |
| Tolaria | | ✓ | | | - | | |
| Nissa | ✓ | | | | | - | |
| Utils | | | | | | | - |

**Legend:** ✓ = runtime dependency, (t) = type-only

### Dependency Layers

```
Layer 0: External (PyTorch, Pydantic, Torchvision)
Layer 1: Leyline, Utils (foundation)
Layer 2: Kasmina, Nissa (core + observability)
Layer 3: Tamiyo, Tolaria (decisions + training)
Layer 4: Simic (orchestrator)
```

---

## Architecture Principles

| Principle | Implementation |
|-----------|---------------|
| **Separation of Concerns** | 7 distinct subsystems with clear boundaries |
| **Dependency Inversion** | All depend on Leyline abstractions |
| **Single Responsibility** | Each subsystem has one purpose |
| **Layered Architecture** | Foundation → Domain → Orchestration |
| **State Machine Clarity** | Explicit lifecycle with quality gates |
