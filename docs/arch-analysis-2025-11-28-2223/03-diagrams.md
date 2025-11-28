# Architecture Diagrams: Esper V1.0

## Level 1: System Context

The System Context diagram shows Esper as a black-box system with external actors and high-level interactions. This level abstracts away all internal complexity and shows only the system boundary and major external dependencies.

**What the diagram shows:**
- **Esper System** as a unified box at the center
- **External Actors**: 
  - Developer (user interacting with CLI)
  - CIFAR-10 Dataset (training/test data source)
  - GPU Hardware (computational resource)
- **Interactions**: Data flows between actors and the system

```mermaid
graph TB
    subgraph "System Context: Esper V1.0"
        Developer["👤 Developer<br/>(CLI User)"]
        GPU["⚡ GPU Hardware<br/>(Training Device)"]
        CIFAR["📊 CIFAR-10 Dataset<br/>(Training Data)"]
        Esper["🧠 Esper System<br/>(Adaptive Neural<br/>Architecture)"]
    end
    
    Developer -->|"Runs CLI:<br/>--episodes N"| Esper
    Esper -->|"Trains on"| CIFAR
    Esper -->|"Leverages<br/>(compute)"| GPU
    Esper -->|"Reports:<br/>Metrics, Accuracy"| Developer
    
    style Esper fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style Developer fill:#50C878,stroke:#2D7A4A
    style GPU fill:#FF6B6B,stroke:#A63131
    style CIFAR fill:#FFB84D,stroke:#996600
```

---

## Level 2: Container Diagram

The Container diagram shows the major technical containers (packages/modules) within Esper and their relationships. This is the zoomed-in view of the system boundary showing how Esper is decomposed into deployable/runnable units.

**What the diagram shows:**
- **6 Core Packages** (containers):
  - Leyline (Protocol/Contract Layer)
  - Kasmina (Seed Mechanics Layer)
  - Tamiyo (Decision Engine Layer)
  - Simic (RL Training Layer)
  - Nissa (Telemetry Layer)
  - Scripts (CLI Entry Points)
- **Orchestrator**: simic_overnight.py integrates all containers
- **Dependencies**: Arrows show how containers depend on each other

```mermaid
graph TB
    subgraph "Esper Containers"
        Leyline["📋 Leyline<br/>(Protocol Layer)<br/>1,057 LOC<br/>---<br/>Data contracts,<br/>Enums, Schemas<br/>Acts as lingua franca"]
        
        Kasmina["🔧 Kasmina<br/>(Seed Mechanics)<br/>1,210 LOC<br/>---<br/>Seed lifecycle,<br/>Gradient isolation,<br/>Alpha blending"]
        
        Tamiyo["🎯 Tamiyo<br/>(Decision Engine)<br/>501 LOC<br/>---<br/>Strategic decisions,<br/>Heuristic policy,<br/>Signal tracking"]
        
        Simic["🤖 Simic<br/>(RL Training)<br/>4,615 LOC<br/>---<br/>PPO/IQL algorithms,<br/>Episode collection,<br/>Feature extraction"]
        
        Nissa["📊 Nissa<br/>(Telemetry)<br/>358 LOC<br/>---<br/>Gradient health,<br/>Loss landscape,<br/>Metrics collection"]
        
        Scripts["🖥️ Scripts<br/>(CLI Entry Points)<br/>---<br/>train.py<br/>generate.py<br/>evaluate.py"]
        
        Orchestrator["🎪 simic_overnight.py<br/>(Orchestrator)<br/>859 LOC<br/>---<br/>Integrates all subsystems,<br/>Generates episodes,<br/>Trains & evaluates"]
    end
    
    subgraph "External"
        PyTorch["⚙️ PyTorch<br/>(Deep Learning)"]
        CIFAR["📊 CIFAR-10<br/>(Dataset)"]
    end
    
    %% Dependency flow: everything depends on Leyline
    Kasmina -->|imports| Leyline
    Tamiyo -->|imports| Leyline
    Simic -->|imports| Leyline
    Nissa -->|imports| Leyline
    Scripts -->|imports| Leyline
    Orchestrator -->|imports| Leyline
    
    %% Integration dependencies
    Orchestrator -->|uses| Kasmina
    Orchestrator -->|uses| Tamiyo
    Orchestrator -->|uses| Simic
    Orchestrator -->|uses| Nissa
    Orchestrator -->|uses| Scripts
    
    %% Intelligence dependencies
    Simic -->|imports HeuristicTamiyo<br/>for PPO/IQL training| Tamiyo
    
    %% Mechanics dependencies
    Tamiyo -->|issues commands to| Kasmina
    
    %% External dependencies
    Kasmina -->|uses| PyTorch
    Simic -->|uses| PyTorch
    Orchestrator -->|loads| CIFAR
    
    style Leyline fill:#E8F4F8,stroke:#2E5C8A,stroke-width:2px
    style Kasmina fill:#FFF4E6,stroke:#996600,stroke-width:2px
    style Tamiyo fill:#F0E6FF,stroke:#663399,stroke-width:2px
    style Simic fill:#E6F4FF,stroke:#1E40AF,stroke-width:2px
    style Nissa fill:#FFF0F5,stroke:#C41E3A,stroke-width:2px
    style Scripts fill:#F0FFF4,stroke:#22863A,stroke-width:2px
    style Orchestrator fill:#FFFACD,stroke:#DAA520,stroke-width:3px
    style PyTorch fill:#FFE4E1,stroke:#8B0000
    style CIFAR fill:#FFE4B5,stroke:#8B4513
```

---

## Level 3: Component Diagrams

### Leyline Components

**Location**: `src/esper/leyline/`  
**Responsibility**: Define the data contracts, enums, and protocols that enable type-safe communication across all subsystems. This is the "lingua franca" of Esper.

**What the diagram shows:**
- **6 Core Modules**: actions, stages, signals, schemas, reports, telemetry
- **Key Abstractions**: Enums (SimicAction, SeedStage), Dataclasses (SeedMetrics), Protocols (BlueprintProtocol)
- **Dependencies**: All other subsystems import from here

```mermaid
graph TB
    subgraph "Leyline Components"
        Actions["actions.py<br/>---<br/>SimicAction (enum)<br/>8 seed lifecycle actions<br/>is_germinate(), get_blueprint_id()"]
        
        Stages["stages.py<br/>---<br/>SeedStage (IntEnum)<br/>11 lifecycle states<br/>VALID_TRANSITIONS dict<br/>is_valid_transition()"]
        
        Signals["signals.py<br/>---<br/>FastTrainingSignals (NamedTuple)<br/>TrainingMetrics (dataclass)<br/>TensorSchema (IntEnum, 27 dims)<br/>27-feature observation space"]
        
        Schemas["schemas.py<br/>---<br/>SeedOperation<br/>AdaptationCommand<br/>BlueprintProtocol<br/>GateLevel, GateResult"]
        
        Reports["reports.py<br/>---<br/>SeedMetrics<br/>SeedStateReport<br/>FieldReport<br/>Metrics & reporting types"]
        
        Telemetry["telemetry.py<br/>---<br/>TelemetryEvent<br/>TelemetryEventType<br/>PerformanceBudgets<br/>Event contracts"]
        
        Init["__init__.py<br/>---<br/>Public API exports<br/>38 re-exported symbols<br/>Curated subsystem interface"]
    end
    
    %% Internal structure
    Actions -.->|exported by| Init
    Stages -.->|exported by| Init
    Signals -.->|exported by| Init
    Schemas -.->|exported by| Init
    Reports -.->|exported by| Init
    Telemetry -.->|exported by| Init
    
    %% Logical grouping
    Actions ---|"Action Space"| Stages
    Stages ---|"State Machine"| Schemas
    Signals ---|"Observations"| Schemas
    Reports ---|"Metrics"| Telemetry
    
    style Actions fill:#E8E8FF,stroke:#1E40AF
    style Stages fill:#E8E8FF,stroke:#1E40AF
    style Signals fill:#E8E8FF,stroke:#1E40AF
    style Schemas fill:#E8E8FF,stroke:#1E40AF
    style Reports fill:#E8E8FF,stroke:#1E40AF
    style Telemetry fill:#E8E8FF,stroke:#1E40AF
    style Init fill:#D4E6F1,stroke:#1E40AF,stroke-width:2px
```

---

### Kasmina Components

**Location**: `src/esper/kasmina/`  
**Responsibility**: Manage seed module lifecycle through germination, training, blending, and fossilization. Implement gradient isolation and alpha-blending for safe integration with host model.

**What the diagram shows:**
- **4 Core Modules**: slot, blueprints, host, isolation
- **Key Classes**: SeedSlot (lifecycle), HostCNN (host model), BlueprintCatalog (architecture registry), GradientIsolationMonitor
- **Data Flow**: Seeds progress through lifecycle stages via quality gates and alpha scheduling

```mermaid
graph TB
    subgraph "Kasmina Components"
        Slot["slot.py<br/>---<br/>SeedSlot (nn.Module)<br/>SeedState, SeedMetrics<br/>QualityGates<br/>Manages lifecycle:<br/>DORMANT→FOSSILIZED<br/>607 LOC"]
        
        Blueprints["blueprints.py<br/>---<br/>ConvBlock<br/>ConvEnhanceSeed<br/>AttentionSeed<br/>NormSeed<br/>DepthwiseSeed<br/>BlueprintCatalog<br/>154 LOC"]
        
        Host["host.py<br/>---<br/>HostCNN<br/>MorphogeneticModel<br/>(Main model composition)<br/>109 LOC"]
        
        Isolation["isolation.py<br/>---<br/>AlphaSchedule<br/>blend_with_isolation()<br/>GradientIsolationMonitor<br/>Gradient hooks<br/>117 LOC"]
        
        Init["__init__.py<br/>---<br/>Re-exports:<br/>All internal components<br/>Plus Leyline contracts<br/>23 LOC"]
    end
    
    %% Data flow
    Blueprints -->|"Seed architectures"| Slot
    Slot -->|"Encapsulates"| Host
    Host -->|"Composes"| MorphoModel["MorphogeneticModel<br/>(Host + Seeds)"]
    Isolation -->|"Blends seeds<br/>with alpha schedule"| Host
    
    %% Quality gates enforcement
    Slot -->|"Quality gates"| Stages["SeedStage<br/>Transitions"]
    
    %% All exported
    Slot -.->|exported| Init
    Blueprints -.->|exported| Init
    Host -.->|exported| Init
    Isolation -.->|exported| Init
    
    style Slot fill:#FFF4E6,stroke:#996600,stroke-width:2px
    style Blueprints fill:#FFF4E6,stroke:#996600
    style Host fill:#FFF4E6,stroke:#996600
    style Isolation fill:#FFF4E6,stroke:#996600
    style Init fill:#FFE4CC,stroke:#996600,stroke-width:2px
    style MorphoModel fill:#FFCCAA,stroke:#663300,stroke-width:2px
    style Stages fill:#F0F0F0,stroke:#666666
```

---

### Tamiyo Components

**Location**: `src/esper/tamiyo/`  
**Responsibility**: Observe training signals and make strategic decisions about seed lifecycle management. Provides both heuristic baseline and learned policy implementations.

**What the diagram shows:**
- **3 Core Modules**: decisions, tracker, heuristic
- **Key Types**: TamiyoAction (enum), TamiyoDecision (dataclass), TamiyoPolicy (Protocol)
- **Decision Loop**: Observe signals → Make decision → Convert to command → Execute on Kasmina

```mermaid
graph TB
    subgraph "Tamiyo Components"
        Decisions["decisions.py<br/>---<br/>TamiyoAction (enum)<br/>7 actions: WAIT, GERMINATE,<br/>ADVANCE_STAGE, CULL, etc.<br/>TamiyoDecision (dataclass)<br/>_ACTION_TO_COMMAND mapping<br/>107 LOC"]
        
        Tracker["tracker.py<br/>---<br/>SignalTracker<br/>Decision history tracking<br/>Observation aggregation<br/>118 LOC"]
        
        Heuristic["heuristic.py<br/>---<br/>TamiyoPolicy (Protocol)<br/>HeuristicPolicyConfig<br/>HeuristicTamiyo<br/>Rule-based policy<br/>251 LOC"]
        
        Init["__init__.py<br/>---<br/>Public API exports<br/>6 symbols<br/>24 LOC"]
    end
    
    subgraph "Integration"
        Leyline["Leyline Contracts<br/>---<br/>SeedStage<br/>TrainingSignals<br/>SimicAction"]
        
        Kasmina["Kasmina Commands<br/>---<br/>AdaptationCommand<br/>SeedSlot operations"]
    end
    
    %% Decision flow
    Tracker -->|"Observes"| Leyline
    Decisions -->|"Encodes"| Leyline
    Heuristic -->|"Implements"| TamiyoPolicy["TamiyoPolicy<br/>(Interface)"]
    TamiyoPolicy -->|"Makes"| Decisions
    Decisions -->|"Converts to"| Kasmina
    
    %% Exports
    Decisions -.->|exported| Init
    Tracker -.->|exported| Init
    Heuristic -.->|exported| Init
    
    style Decisions fill:#F0E6FF,stroke:#663399,stroke-width:2px
    style Tracker fill:#F0E6FF,stroke:#663399
    style Heuristic fill:#F0E6FF,stroke:#663399
    style Init fill:#E6D5FF,stroke:#663399,stroke-width:2px
    style Leyline fill:#F0F0F0,stroke:#666666
    style Kasmina fill:#F0F0F0,stroke:#666666
    style TamiyoPolicy fill:#E6D5FF,stroke:#663399,stroke-width:2px
```

---

### Simic Components

**Location**: `src/esper/simic/`  
**Responsibility**: Train neural network policies to improve Tamiyo's seed lifecycle decisions. Provides PPO (online) and IQL/CQL (offline) training algorithms, episode collection, reward computation, and feature extraction.

**What the diagram shows:**
- **7 Core Modules**: episodes, features, rewards, networks, ppo, iql, __init__
- **Key Classes**: PolicyNetwork (nn.Module), PPOAgent, IQL (offline learner), EpisodeCollector
- **Data Pipeline**: TrainingSignals → Features → Episodes → Rewards → Training

```mermaid
graph TB
    subgraph "Simic Components"
        Episodes["episodes.py<br/>---<br/>TrainingSnapshot<br/>ActionTaken<br/>StepOutcome<br/>Episode<br/>EpisodeCollector<br/>DatasetManager<br/>719 LOC"]
        
        Features["features.py<br/>---<br/>obs_to_base_features()<br/>telemetry_to_features()<br/>safe()<br/>HOT PATH:<br/>27-dim feature vector<br/>161 LOC"]
        
        Rewards["rewards.py<br/>---<br/>RewardConfig<br/>compute_shaped_reward()<br/>compute_potential()<br/>compute_pbrs_bonus()<br/>Multi-component reward<br/>shaping<br/>376 LOC"]
        
        Networks["networks.py<br/>---<br/>PolicyNetwork<br/>(nn.Module)<br/>Actor-Critic architecture<br/>Action logits + value<br/>print_confusion_matrix()<br/>342 LOC"]
        
        PPO["ppo.py<br/>---<br/>RunningMeanStd<br/>PPOAgent<br/>train_ppo_vectorized()<br/>Vectorized environments<br/>GAE, entropy regularization<br/>1,590 LOC"]
        
        IQL["iql.py<br/>---<br/>IQL class<br/>Offline RL training<br/>CQL constraint<br/>Batch sampling<br/>1,326 LOC"]
        
        Init["__init__.py<br/>---<br/>Core exports<br/>Lazy imports for<br/>ppo, iql (heavy)<br/>101 LOC"]
    end
    
    subgraph "Integration"
        Leyline["Leyline<br/>---<br/>TensorSchema<br/>SimicAction<br/>TrainingSignals"]
        
        Tamiyo["Tamiyo<br/>(for training)<br/>---<br/>HeuristicTamiyo<br/>SignalTracker"]
    end
    
    %% Data pipeline
    Leyline -->|"TrainingSignals<br/>(observations)"| Features
    Features -->|"27-dim feature<br/>vector"| Networks
    Features -->|"feature vectors"| PPO
    Features -->|"feature vectors"| IQL
    
    Leyline -->|"SimicAction enum"| Episodes
    Episodes -->|"episode data"| Rewards
    Rewards -->|"shaped rewards"| PPO
    Rewards -->|"shaped rewards"| IQL
    
    Episodes -->|"collected data"| Networks
    Tamiyo -->|"baseline policy<br/>for comparison"| PPO
    Tamiyo -->|"baseline policy<br/>for comparison"| IQL
    
    %% Exports
    Episodes -.->|exported| Init
    Features -.->|exported| Init
    Rewards -.->|exported| Init
    Networks -.->|exported| Init
    
    style Episodes fill:#E6F4FF,stroke:#1E40AF
    style Features fill:#CCE5FF,stroke:#1E40AF,stroke-width:2px
    style Rewards fill:#E6F4FF,stroke:#1E40AF
    style Networks fill:#E6F4FF,stroke:#1E40AF
    style PPO fill:#E6F4FF,stroke:#1E40AF
    style IQL fill:#E6F4FF,stroke:#1E40AF
    style Init fill:#BFDBFE,stroke:#1E40AF,stroke-width:2px
    style Features fill:#BFDBFE,stroke:#0047AB,stroke-width:2px
    style Leyline fill:#F0F0F0,stroke:#666666
    style Tamiyo fill:#F0F0F0,stroke:#666666
```

---

### Nissa Components

**Location**: `src/esper/nissa/`  
**Responsibility**: Cross-cutting telemetry collection system. Gathers gradient health, loss landscape analysis, per-class metrics and routes to configurable output backends.

**What the diagram shows:**
- **3 Core Modules**: config, tracker, output
- **Key Classes**: TelemetryConfig (Pydantic), DiagnosticTracker (nn.Module), NissaHub (observer)
- **Pattern**: Configuration → Collection → Output routing

```mermaid
graph TB
    subgraph "Nissa Components"
        Config["config.py<br/>---<br/>TelemetryConfig<br/>GradientConfig<br/>LossLandscapeConfig<br/>PerClassConfig<br/>Profile loaders<br/>88 LOC"]
        
        Tracker["tracker.py<br/>---<br/>DiagnosticTracker<br/>GradientStats<br/>GradientHealth<br/>EpochSnapshot<br/>Narrative generation<br/>124 LOC"]
        
        Output["output.py<br/>---<br/>OutputBackend (Protocol)<br/>ConsoleOutput<br/>FileOutput<br/>NissaHub (router)<br/>get_hub(), emit()<br/>101 LOC"]
        
        Init["__init__.py<br/>---<br/>Public API exports<br/>13 symbols<br/>45 LOC"]
    end
    
    subgraph "Integration"
        Leyline["Leyline<br/>---<br/>TelemetryEvent<br/>TelemetryEventType"]
        
        Consumers["Consumers<br/>---<br/>simic_overnight.py<br/>Training loops"]
    end
    
    %% Configuration flow
    Config -->|"Profiles:<br/>diagnostic, minimal,<br/>production"| Tracker
    
    %% Collection flow
    Tracker -->|"Collects:<br/>Gradient health<br/>Loss landscape"| Leyline
    
    %% Output flow
    Tracker -->|"Emits"| Output
    Output -->|"Routes to"| ConsoleOutput["ConsoleOutput<br/>(Stdout)"]
    Output -->|"Routes to"| FileOutput["FileOutput<br/>(JSONL)"]
    NissaHub["NissaHub<br/>(Router<br/>& Multiplexer)"] -->|"Centralized<br/>telemetry hub"| ConsoleOutput
    NissaHub -->|"Centralized<br/>telemetry hub"| FileOutput
    
    %% Exports
    Config -.->|exported| Init
    Tracker -.->|exported| Init
    Output -.->|exported| Init
    
    %% Usage
    Consumers -->|"Uses"| NissaHub
    
    style Config fill:#FFF0F5,stroke:#C41E3A
    style Tracker fill:#FFF0F5,stroke:#C41E3A
    style Output fill:#FFF0F5,stroke:#C41E3A
    style Init fill:#FFE4ED,stroke:#C41E3A,stroke-width:2px
    style NissaHub fill:#FFE4ED,stroke:#C41E3A,stroke-width:2px
    style Leyline fill:#F0F0F0,stroke:#666666
    style Consumers fill:#F0F0F0,stroke:#666666
    style ConsoleOutput fill:#FFE4ED,stroke:#C41E3A
    style FileOutput fill:#FFE4ED,stroke:#C41E3A
```

---

## Data Flow Diagram

**What the diagram shows:** How data flows through the Esper system from raw training signals through decisions and actions to updated model states.

**Flow**: Training Signals → Features → Policy Decisions → Seed Commands → Model Updates → Metrics

```mermaid
graph LR
    subgraph "Data Flow Pipeline"
        Input["🔄 TrainingSignals<br/>(27 features)<br/>epoch, loss, accuracy<br/>gradient metrics"]
        
        Features["➡️ Feature Extraction<br/>(simic/features.py)<br/>obs_to_base_features()<br/>27-dim vector"]
        
        Decision["🎯 Policy Decision<br/>Heuristic or Learned<br/>TamiyoAction enum<br/>Action + metadata"]
        
        Command["⚙️ Seed Command<br/>AdaptationCommand<br/>SeedStage transition<br/>RiskLevel assignment"]
        
        Execute["🔧 Execute on Seed<br/>(kasmina/slot.py)<br/>SeedSlot.update_stage()<br/>Alpha blending"]
        
        Feedback["📊 Metrics & Telemetry<br/>(nissa/*)<br/>Gradient health<br/>Training progress"]
        
        Loop["↩️ Next Epoch<br/>(Loop back)"]
    end
    
    Input -->|quality| Features
    Features -->|inference| Decision
    Decision -->|maps to| Command
    Command -->|executes| Execute
    Execute -->|emits events| Feedback
    Feedback -->|observes<br/>in next epoch| Loop
    Loop -->|continues| Input
    
    style Input fill:#FFE4B5,stroke:#8B4513,stroke-width:2px
    style Features fill:#BFDBFE,stroke:#1E40AF,stroke-width:2px
    style Decision fill:#E6D5FF,stroke:#663399,stroke-width:2px
    style Command fill:#FFF4E6,stroke:#996600,stroke-width:2px
    style Execute fill:#FFF4E6,stroke:#996600,stroke-width:2px
    style Feedback fill:#FFE4ED,stroke:#C41E3A,stroke-width:2px
    style Loop fill:#E8F4F8,stroke:#2E5C8A,stroke-width:2px
```

---

## Dependency Graph

**What the diagram shows:** Package-level dependencies showing which subsystems depend on which others. This helps understand the architecture's layering and identifies potential circular dependency risks.

**Key Insight**: Leyline is the foundation - everything depends on it. No other subsystem has universal dependents.

```mermaid
graph BT
    subgraph "Subsystems"
        Leyline["Leyline<br/>(Foundation)<br/>No external deps"]
        Kasmina["Kasmina<br/>(Mechanics)"]
        Tamiyo["Tamiyo<br/>(Decisions)"]
        Simic["Simic<br/>(Learning)"]
        Nissa["Nissa<br/>(Telemetry)"]
        Scripts["Scripts<br/>(CLI)"]
    end
    
    %% Dependency arrows point upward to dependencies
    Kasmina -->|imports| Leyline
    Tamiyo -->|imports| Leyline
    Simic -->|imports| Leyline
    Nissa -->|imports| Leyline
    Scripts -->|imports| Leyline
    
    %% Cross-subsystem dependencies
    Simic -->|imports HeuristicTamiyo<br/>for training| Tamiyo
    Tamiyo -->|TYPE_CHECKING<br/>SeedState| Kasmina
    Scripts -->|imports| Kasmina
    Scripts -->|imports| Tamiyo
    Scripts -->|imports| Simic
    Scripts -->|imports| Nissa
    
    %% No circular dependencies
    
    subgraph "External"
        PyTorch["PyTorch<br/>(torch, torch.nn)"]
        NumPy["NumPy"]
        Pydantic["Pydantic<br/>(nissa only)"]
        TorchVision["TorchVision<br/>(CIFAR-10)"]
    end
    
    Kasmina -->|uses| PyTorch
    Simic -->|uses| PyTorch
    Simic -->|uses| NumPy
    Nissa -->|uses| Pydantic
    Scripts -->|uses| TorchVision
    
    style Leyline fill:#E8F4F8,stroke:#2E5C8A,stroke-width:3px
    style Kasmina fill:#FFF4E6,stroke:#996600,stroke-width:2px
    style Tamiyo fill:#F0E6FF,stroke:#663399,stroke-width:2px
    style Simic fill:#E6F4FF,stroke:#1E40AF,stroke-width:2px
    style Nissa fill:#FFF0F5,stroke:#C41E3A,stroke-width:2px
    style Scripts fill:#F0FFF4,stroke:#22863A,stroke-width:2px
    style PyTorch fill:#FFE4E1,stroke:#8B0000
    style NumPy fill:#FFE4E1,stroke:#8B0000
    style Pydantic fill:#FFE4E1,stroke:#8B0000
    style TorchVision fill:#FFE4E1,stroke:#8B0000
```

---

## Lifecycle State Machine Diagram

**What the diagram shows:** The finite state machine governing seed lifecycle transitions from germination through fossilization, including failure paths.

**Key States**: DORMANT → GERMINATED → TRAINING → BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED (success) or CULLED/EMBARGOED (failure)

```mermaid
stateDiagram-v2
    [*] --> DORMANT: Seed initialized
    
    DORMANT --> GERMINATED: GERMINATE action
    
    GERMINATED --> TRAINING: Begin training
    
    TRAINING --> BLENDING: Converged<br/>+ ADVANCE_STAGE
    TRAINING --> CULLED: Failed<br/>+ CULL
    
    BLENDING --> SHADOWING: Alpha = 1.0<br/>+ ADVANCE_STAGE
    BLENDING --> CULLED: Instability<br/>+ CULL
    
    SHADOWING --> PROBATIONARY: Safe blend<br/>+ ADVANCE_STAGE
    SHADOWING --> CULLED: Poor metrics<br/>+ CULL
    
    PROBATIONARY --> FOSSILIZED: Long-term<br/>improvement<br/>+ ADVANCE_STAGE
    PROBATIONARY --> EMBARGOED: Regression<br/>+ EMBARGOED_RESET
    
    EMBARGOED --> DORMANT: EMBARGOED_RESET
    
    FOSSILIZED --> [*]: Seed integrated
    CULLED --> [*]: Seed removed
    
    note right of GERMINATED
        Preparing for training
        Resource allocation
    end note
    
    note right of TRAINING
        Isolated training
        Signal collection
        Quality gate monitoring
    end note
    
    note right of BLENDING
        Alpha schedule 0 → 1
        Gradient isolation
        Preventing catastrophic
        forgetting
    end note
    
    note right of SHADOWING
        Host gradient flow
        Observe impact
        on main task
    end note
    
    note right of PROBATIONARY
        Long-term monitoring
        Check for regressions
        before final commit
    end note
    
    note right of FOSSILIZED
        Permanently integrated
        No further changes
    end note
    
    note right of CULLED
        Failed to improve
        Resource reclaimed
    end note
```

---

## Training Loop Orchestration Diagram

**What the diagram shows:** The high-level training loop orchestration showing how simic_overnight.py integrates all subsystems through a complete training cycle.

```mermaid
sequenceDiagram
    participant User as Developer
    participant Orch as simic_overnight.py<br/>(Orchestrator)
    participant Kam as Kasmina<br/>(Mechanics)
    participant Tam as Tamiyo<br/>(Decisions)
    participant Sim as Simic<br/>(Learning)
    participant Nis as Nissa<br/>(Telemetry)
    participant Data as CIFAR-10<br/>Dataset
    
    User->>Orch: --episodes N
    activate Orch
    
    Orch->>Kam: create_model()
    activate Kam
    Kam->>Kam: Initialize MorphogeneticModel
    Kam-->>Orch: model ready
    deactivate Kam
    
    Orch->>Data: load_cifar10()
    activate Data
    Data-->>Orch: trainloader, testloader
    deactivate Data
    
    loop For each episode (N times)
        Orch->>Tam: Create HeuristicTamiyo
        activate Tam
        Tam->>Tam: Initialize policy + tracker
        Tam-->>Orch: policy ready
        deactivate Tam
        
        loop For each epoch
            Orch->>Orch: Train on CIFAR-10 batch
            
            Orch->>Orch: Collect TrainingSignals
            
            Orch->>Tam: Get decision from heuristic
            activate Tam
            Tam->>Tam: Observe signals
            Tam-->>Orch: TamiyoDecision
            deactivate Tam
            
            Orch->>Kam: Execute command
            activate Kam
            Kam->>Kam: Update seed stage
            Kam->>Nis: Emit telemetry
            Kam-->>Orch: StepOutcome + rewards
            deactivate Kam
            
            Orch->>Nis: Record metrics
            activate Nis
            Nis->>Nis: Track gradient health
            Nis-->>Orch: telemetry logged
            deactivate Nis
            
            Orch->>Sim: Collect episode
            activate Sim
            Sim->>Sim: Store TrainingSnapshot
            Sim-->>Orch: episode stored
            deactivate Sim
        end
    end
    
    Orch->>Sim: train_policy(episodes)
    activate Sim
    Sim->>Sim: PolicyNetwork training loop
    Sim->>Sim: Extract features
    Sim->>Sim: Compute rewards
    Sim->>Sim: PPO/IQL gradient updates
    Sim-->>Orch: trained policy
    deactivate Sim
    
    Orch->>Orch: evaluate_policy()
    Orch->>Orch: compute accuracy
    
    Orch->>Orch: compare_policies()
    Orch->>Tam: heuristic policy
    Orch->>Sim: learned policy
    
    Orch-->>User: Results + metrics
    deactivate Orch
```

---

## Architectural Patterns Reference

### 1. Contract-Based Design (Leyline)
All subsystems communicate through Leyline data contracts:
- **Enums**: SimicAction, SeedStage, TelemetryEventType
- **Dataclasses**: SeedMetrics, TrainingSignals, SeedState
- **Protocols**: BlueprintProtocol, TamiyoPolicy, OutputBackend

### 2. Finite State Machine (Seed Lifecycle)
SeedStage forms an explicit FSM with:
- 11 states (DORMANT through FOSSILIZED)
- Valid transition rules enforced by `is_valid_transition()`
- Terminal and failure states clearly marked

### 3. Gradient Isolation Pattern (Kasmina)
Safe integration of new seeds into host model:
- Alpha-blending schedule (0 → 1 over epochs)
- Hook-based gradient interception
- Prevents catastrophic forgetting

### 4. Hot Path Isolation (Simic Features)
Feature extraction isolated for performance:
- `simic/features.py` imports only Leyline
- O(1) 27-dimensional feature computation
- Enables future JIT compilation

### 5. Strategy Pattern (Policies)
Pluggable decision implementations:
- **TamiyoPolicy** protocol defines interface
- **HeuristicTamiyo** baseline rule-based implementation
- **PolicyNetwork** learned neural implementation
- Enable side-by-side comparison and gradual migration

### 6. Observer Pattern (Telemetry)
Decoupled event emission and handling:
- **NissaHub** multiplexes to multiple backends
- **OutputBackend** protocol for pluggable outputs
- Multiple output backends (Console, File, future: Cloud)

### 7. Configuration Profiles (Nissa)
Pydantic-based configuration with presets:
- "diagnostic" - rich telemetry, high overhead
- "minimal" - sparse telemetry, low overhead
- "production" - moderate telemetry, balanced

---

## Summary

Esper V1.0 employs a **layered architecture** with clear separation of concerns:

1. **Foundation (Leyline)**: Type-safe data contracts enable loose coupling
2. **Mechanics (Kasmina)**: Seed lifecycle and gradient isolation ensure safe integration
3. **Intelligence (Tamiyo + Simic)**: Heuristic baseline + learned policies for seed management
4. **Monitoring (Nissa)**: Rich telemetry for debugging and optimization
5. **Integration (Scripts + simic_overnight.py)**: Orchestration layer ties everything together

**Key Design Principles**:
- No circular dependencies - clean layering
- Performance-aware: Hot path isolation (features.py), lazy imports, named tuples
- Type-safe: Heavy use of enums, dataclasses, protocols
- Extensible: Protocol-based design allows new implementations
- Observable: Comprehensive telemetry without polluting core logic

---

**Diagram Generation**: 2025-11-29  
**Analysis Basis**: Discovery findings (01-discovery-findings.md) and subsystem catalog (02-subsystem-catalog.md)  
**C4 Levels Covered**: Level 1 (Context), Level 2 (Container), Level 3 (Components × 4 subsystems)  
**Additional Diagrams**: Data Flow, Dependency Graph, State Machine, Orchestration Sequence

