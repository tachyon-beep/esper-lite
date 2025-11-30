# Architecture Diagrams - Esper Morphogenetic Neural Networks

## Overview

This document contains C4-style architecture diagrams for Esper at multiple abstraction levels:
1. **Context**: System boundary and external interactions
2. **Container**: Internal subsystems and their relationships
3. **Component**: Detailed views of key subsystems

All diagrams use Mermaid syntax for easy rendering in GitHub, VS Code, etc.

---

## 1. System Context Diagram

Shows Esper as a black box and its interactions with external systems.

```mermaid
C4Context
    title System Context: Esper Morphogenetic Neural Network Framework

    Person(researcher, "ML Researcher", "Trains and evaluates morphogenetic models")

    System(esper, "Esper", "Framework for neural networks that dynamically grow and adapt their topology")

    System_Ext(pytorch, "PyTorch", "Deep learning framework")
    System_Ext(cuda, "CUDA", "GPU compute platform")
    System_Ext(cifar, "CIFAR-10", "Training dataset")

    Rel(researcher, esper, "Trains models via CLI")
    Rel(esper, pytorch, "Uses for neural network ops")
    Rel(esper, cuda, "GPU acceleration")
    Rel(esper, cifar, "Loads training data")
```

### Text Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌──────────────┐                                              │
│   │ ML Researcher│                                              │
│   └──────┬───────┘                                              │
│          │ trains via CLI                                       │
│          ▼                                                      │
│   ┌──────────────────────────────────────┐                     │
│   │             ESPER                     │                     │
│   │  Morphogenetic Neural Network         │                     │
│   │  Framework                            │                     │
│   └──────────────────────────────────────┘                     │
│          │                │                │                    │
│          ▼                ▼                ▼                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│   │  PyTorch │    │   CUDA   │    │ CIFAR-10 │                │
│   │(ML ops)  │    │  (GPU)   │    │ (Data)   │                │
│   └──────────┘    └──────────┘    └──────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Container Diagram

Shows the major subsystems (containers) within Esper and their relationships.

```mermaid
C4Container
    title Container Diagram: Esper Subsystems

    Person(user, "User", "ML Researcher")

    Container_Boundary(esper, "Esper Framework") {
        Container(scripts, "Scripts", "Python CLI", "Entry points: train.py, evaluate.py")
        Container(simic, "Simic", "Python", "RL infrastructure: PPO, IQL, vectorized training")
        Container(tamiyo, "Tamiyo", "Python", "Strategic decisions: heuristic or learned policy")
        Container(kasmina, "Kasmina", "Python", "Model mechanics: seeds, slots, grafting")
        Container(tolaria, "Tolaria", "Python", "Training loops: epoch functions")
        Container(leyline, "Leyline", "Python", "Contracts: stages, actions, schemas")
        Container(nissa, "Nissa", "Python", "Telemetry: diagnostics, logging")
        Container(utils, "Utils", "Python", "Utilities: data loading")
    }

    System_Ext(pytorch, "PyTorch")

    Rel(user, scripts, "CLI commands")
    Rel(scripts, simic, "trains")
    Rel(simic, tamiyo, "improves policy")
    Rel(simic, tolaria, "uses for training")
    Rel(simic, kasmina, "manages seeds")
    Rel(tamiyo, kasmina, "issues commands")
    Rel(tolaria, kasmina, "executes on model")
    Rel(kasmina, leyline, "uses contracts")
    Rel(tamiyo, leyline, "uses contracts")
    Rel(simic, leyline, "uses contracts")
    Rel(tolaria, leyline, "uses contracts")
    Rel(simic, utils, "loads data")
    Rel(simic, nissa, "optional telemetry")
    Rel(kasmina, pytorch, "neural network ops")
```

### Text Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ESPER FRAMEWORK                                │
│                                                                             │
│  ┌─────────┐                                                               │
│  │ Scripts │◄── User (CLI)                                                 │
│  └────┬────┘                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                              SIMIC (Gym)                             │  │
│  │  PPO, IQL, Vectorized Training, Rewards, Features, Networks         │  │
│  └──────────────────────┬───────────────┬──────────────────────────────┘  │
│                         │               │                                   │
│            ┌────────────┴───────┐       └──────────┐                       │
│            ▼                    ▼                  ▼                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  TAMIYO (Brain) │  │ TOLARIA (Hands) │  │   UTILS (Data)  │            │
│  │  Heuristic/RL   │  │  Training Loops │  │   CIFAR-10      │            │
│  │  Decisions      │  │  Epoch Funcs    │  └─────────────────┘            │
│  └────────┬────────┘  └────────┬────────┘                                  │
│           │                    │                                            │
│           └────────┬───────────┘                                            │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          KASMINA (Body)                              │  │
│  │  MorphogeneticModel, SeedSlot, Blueprints, Isolation, QualityGates  │  │
│  └──────────────────────────────────┬──────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          LEYLINE (Contracts)                         │  │
│  │  SeedStage, Action, TrainingSignals, TensorSchema, Telemetry        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │  NISSA (Senses) │◄── optional telemetry from Simic                     │
│  │  Diagnostics    │                                                       │
│  └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
             ┌──────────┐
             │  PyTorch │
             └──────────┘
```

---

## 3. Data Flow Diagram

Shows how data flows through the system during training.

```mermaid
flowchart LR
    subgraph External
        CIFAR[CIFAR-10 Dataset]
        GPU[CUDA GPU]
    end

    subgraph Simic[Simic - RL Training]
        VEC[Vectorized<br/>Training]
        PPO[PPO Agent]
        FEAT[Features<br/>Extraction]
        REW[Reward<br/>Shaping]
    end

    subgraph Tolaria[Tolaria - Training Loops]
        TRAIN[train_epoch_*]
        VAL[validate]
    end

    subgraph Tamiyo[Tamiyo - Decisions]
        HEUR[Heuristic<br/>Policy]
        DEC[Decision]
    end

    subgraph Kasmina[Kasmina - Model]
        MODEL[Morphogenetic<br/>Model]
        SLOT[Seed Slot]
        SEED[Seed Module]
    end

    subgraph Leyline[Leyline - Contracts]
        SIG[Training<br/>Signals]
        ACT[Actions]
        STG[Stages]
    end

    CIFAR --> VEC
    VEC --> TRAIN
    TRAIN --> MODEL
    MODEL --> SLOT
    SLOT --> SEED
    TRAIN --> VAL
    VAL --> SIG
    SIG --> FEAT
    FEAT --> PPO
    PPO --> ACT
    ACT --> DEC
    DEC --> SLOT
    SLOT --> STG
    SEED --> GPU
    MODEL --> GPU
    PPO --> REW
```

### Text Representation

```
                    TRAINING DATA FLOW

CIFAR-10 ──► Vectorized Training (Simic)
                    │
                    ▼
            train_epoch_* (Tolaria)
                    │
                    ▼
            MorphogeneticModel (Kasmina)
            ├── Host CNN
            └── SeedSlot ──► Seed Module
                    │
                    ▼
            validate() (Tolaria)
                    │
                    ▼
            TrainingSignals (Leyline)
                    │
                    ▼
            Feature Extraction (Simic)
                    │
                    ▼
            PPO Agent (Simic)
                    │
                    ▼
            Action (Leyline)
                    │
                    ▼
            Decision (Tamiyo)
                    │
                    ▼
            SeedSlot (Kasmina) ──► Stage Transition
```

---

## 4. Seed Lifecycle State Diagram

Shows the state machine for seed modules.

```mermaid
stateDiagram-v2
    [*] --> DORMANT: Empty slot

    DORMANT --> GERMINATED: Germinate action<br/>(G0 gate)

    GERMINATED --> TRAINING: Auto-start<br/>(G1 gate)

    TRAINING --> BLENDING: Advance action<br/>(G2 gate: improvement > threshold)
    TRAINING --> CULLED: Cull action<br/>(performance drop)

    BLENDING --> SHADOWING: Advance action<br/>(G3 gate: alpha >= 0.95)
    BLENDING --> CULLED: Cull action<br/>(regression)

    SHADOWING --> PROBATIONARY: Advance action<br/>(G4 gate)
    SHADOWING --> CULLED: Cull action

    PROBATIONARY --> FOSSILIZED: Advance action<br/>(G5 gate: total improvement > 0)
    PROBATIONARY --> CULLED: Cull action

    FOSSILIZED --> [*]: Terminal success

    CULLED --> EMBARGOED: Auto-transition

    EMBARGOED --> RESETTING: Cooldown complete

    RESETTING --> DORMANT: Slot recycled
```

### Text Representation

```
                          SEED LIFECYCLE STATE MACHINE

                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    [*] ──► DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING │
                │            │              │           │     │
                │            │              │           │     │
                │            │              ▼           ▼     │
                │            │           CULLED ◄──────┤     │
                │            │              │          │     │
                │            │              ▼          │     │
                │            │         EMBARGOED       │     │
                │            │              │          │     │
                │            │              ▼          │     │
                │            │         RESETTING ──────┘     │
                │            │                               │
                │            ▼                               │
                │     SHADOWING ──► PROBATIONARY ──► FOSSILIZED ──► [*]
                │          │              │
                │          └──────┬───────┘
                │                 ▼
                │              CULLED
                └─────────────────┘

    Quality Gates:
    G0: Basic sanity (seed_id, blueprint_id)
    G1: Training readiness (germinated)
    G2: Blending readiness (improvement > threshold)
    G3: Shadowing readiness (alpha >= 0.95)
    G4: Probation readiness (shadowing complete)
    G5: Fossilization readiness (total improvement > 0)
```

---

## 5. Component Diagram: Simic

Detailed view of the Simic RL subsystem components.

```mermaid
C4Component
    title Component Diagram: Simic (RL Infrastructure)

    Container_Boundary(simic, "Simic") {
        Component(ppo, "PPOAgent", "ppo.py", "On-policy RL agent")
        Component(iql, "IQL", "iql.py", "Offline RL agent")
        Component(vec, "Vectorized", "vectorized.py", "Multi-GPU training with CUDA streams")
        Component(net, "Networks", "networks.py", "ActorCritic, PolicyNetwork, Q/V Networks")
        Component(rew, "Rewards", "rewards.py", "Reward shaping, PBRS")
        Component(feat, "Features", "features.py", "Hot path feature extraction")
        Component(buf, "Buffers", "buffers.py", "RolloutBuffer, ReplayBuffer")
        Component(norm, "Normalization", "normalization.py", "RunningMeanStd")
        Component(ep, "Episodes", "episodes.py", "TrainingSnapshot, Episode")
        Component(train, "Training", "training.py", "Non-vectorized training loops")
    }

    Rel(vec, ppo, "creates and updates")
    Rel(vec, net, "batched inference")
    Rel(ppo, net, "uses ActorCritic")
    Rel(ppo, buf, "stores transitions")
    Rel(vec, feat, "extracts features")
    Rel(vec, rew, "computes rewards")
    Rel(vec, norm, "normalizes observations")
    Rel(iql, buf, "samples from ReplayBuffer")
    Rel(train, ppo, "non-vectorized training")
```

### Text Representation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SIMIC SUBSYSTEM                                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                    VECTORIZED TRAINING                        │     │
│  │    vectorized.py - Multi-GPU, CUDA streams, inverted control  │     │
│  └─────────────────────────────┬─────────────────────────────────┘     │
│                                │                                        │
│         ┌──────────────────────┼──────────────────────┐                │
│         ▼                      ▼                      ▼                │
│  ┌─────────────┐      ┌──────────────┐      ┌─────────────┐           │
│  │   PPOAgent  │      │   Networks   │      │  Buffers    │           │
│  │   ppo.py    │◄────►│ networks.py  │      │ buffers.py  │           │
│  │             │      │ ActorCritic  │      │ Rollout/    │           │
│  │             │      │ Q/V Networks │      │ Replay      │           │
│  └─────────────┘      └──────────────┘      └─────────────┘           │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────┐      ┌──────────────┐      ┌─────────────┐           │
│  │   Rewards   │      │   Features   │      │Normalization│           │
│  │ rewards.py  │      │ features.py  │      │  norm.py    │           │
│  │ PBRS, config│      │ HOT PATH     │      │RunningMean  │           │
│  └─────────────┘      └──────────────┘      └─────────────┘           │
│                                                                        │
│  ┌─────────────┐      ┌──────────────┐                                │
│  │     IQL     │      │   Episodes   │                                │
│  │   iql.py    │      │ episodes.py  │                                │
│  │ Offline RL  │      │ Snapshots    │                                │
│  └─────────────┘      └──────────────┘                                │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Component Diagram: Kasmina

Detailed view of the Kasmina model mechanics subsystem.

```mermaid
C4Component
    title Component Diagram: Kasmina (Model Mechanics)

    Container_Boundary(kasmina, "Kasmina") {
        Component(host, "Host", "host.py", "HostCNN, MorphogeneticModel")
        Component(slot, "Slot", "slot.py", "SeedSlot, SeedState, QualityGates")
        Component(bp, "Blueprints", "blueprints.py", "Seed module factories")
        Component(iso, "Isolation", "isolation.py", "Gradient isolation, alpha blending")
    }

    Container_Ext(leyline, "Leyline", "Contracts")

    Rel(host, slot, "contains SeedSlot")
    Rel(slot, bp, "creates seeds from")
    Rel(slot, iso, "uses for blending")
    Rel(host, iso, "monitors gradients")
    Rel(slot, leyline, "imports SeedStage, gates")
    Rel(bp, leyline, "implements BlueprintProtocol")
```

### Text Representation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KASMINA SUBSYSTEM                               │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                         HOST (host.py)                         │     │
│  │  ┌─────────────────┐     ┌────────────────────────────┐      │     │
│  │  │     HostCNN     │     │    MorphogeneticModel      │      │     │
│  │  │  - block1/2/3   │     │  - host: HostCNN           │      │     │
│  │  │  - classifier   │     │  - seed_slot: SeedSlot     │      │     │
│  │  │  - injection_pt │     │  - isolation_monitor       │      │     │
│  │  └─────────────────┘     └────────────┬───────────────┘      │     │
│  └───────────────────────────────────────┼───────────────────────┘     │
│                                          │                              │
│                                          ▼                              │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                        SLOT (slot.py)                          │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │     │
│  │  │  SeedSlot   │  │  SeedState  │  │   QualityGates      │   │     │
│  │  │ - germinate │  │ - stage     │  │ - G0-G5 checks      │   │     │
│  │  │ - advance   │  │ - metrics   │  │ - transition rules  │   │     │
│  │  │ - cull      │  │ - history   │  └─────────────────────┘   │     │
│  │  │ - forward   │  └─────────────┘                            │     │
│  │  └──────┬──────┘                                              │     │
│  └─────────┼─────────────────────────────────────────────────────┘     │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  BLUEPRINTS (blueprints.py)      ISOLATION (isolation.py)       │  │
│  │  ┌─────────────────────────┐     ┌───────────────────────────┐ │  │
│  │  │ ConvEnhanceSeed         │     │ AlphaSchedule             │ │  │
│  │  │ AttentionSeed           │     │ blend_with_isolation()    │ │  │
│  │  │ NormSeed                │     │ GradientIsolationMonitor  │ │  │
│  │  │ DepthwiseSeed           │     └───────────────────────────┘ │  │
│  │  │ BlueprintCatalog        │                                   │  │
│  │  └─────────────────────────┘                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │     LEYLINE     │
           │  SeedStage      │
           │  GateLevel      │
           │  GateResult     │
           └─────────────────┘
```

---

## 7. Deployment View

Shows how components map to runtime environments.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME DEPLOYMENT                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                      User Environment                            │  │
│  │                                                                   │  │
│  │   $ PYTHONPATH=src python -m esper.scripts.train ppo \           │  │
│  │       --vectorized --n-envs 4 --device cuda:0                    │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Python Process                               │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │                   Main Thread                            │    │  │
│  │  │  - CLI argument parsing                                  │    │  │
│  │  │  - Vectorized training orchestration                     │    │  │
│  │  │  - PPO updates                                           │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │               DataLoader Workers (4)                     │    │  │
│  │  │  - Parallel data prefetching                             │    │  │
│  │  │  - Independent generators per environment                │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                         GPU (cuda:0)                             │  │
│  │                                                                   │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │  CUDA Stream 0   │  CUDA Stream 1   │  CUDA Stream 2/3   │   │  │
│  │  │  - Env 0         │  - Env 1         │  - Env 2/3         │   │  │
│  │  │  - Model forward │  - Model forward │  - Model forward   │   │  │
│  │  │  - Backward      │  - Backward      │  - Backward        │   │  │
│  │  └──────────────────┴──────────────────┴────────────────────┘   │  │
│  │                                                                   │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │                    Shared GPU Memory                      │   │  │
│  │  │  - Policy network (ActorCritic)                           │   │  │
│  │  │  - Observation normalizer (RunningMeanStd)                │   │  │
│  │  └──────────────────────────────────────────────────────────┘   │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram Legend

| Symbol | Meaning |
|--------|---------|
| `──►` | Data flow direction |
| `◄──►` | Bidirectional relationship |
| `[*]` | Initial/terminal state |
| Solid box | Internal component |
| Dashed box | External system |
| `Container_Boundary` | Logical grouping |

---

## Confidence Level

**HIGH** - All diagrams are derived from verified subsystem catalog and direct code analysis. Data flows and state transitions match implementation.
