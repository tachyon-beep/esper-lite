# C4 Architecture Diagrams: esper-lite

## Level 1: System Context

```mermaid
graph TB
    User["User/Researcher"]
    TrainingData["Training Data<br/>(CIFAR-10, TinyStories)"]

    EsperLite["esper-lite<br/>Morphogenetic Neural Networks"]

    User -->|Trains morphogenetic<br/>neural networks| EsperLite
    TrainingData -->|Feeds data into<br/>training pipeline| EsperLite
    EsperLite -->|Produces trained<br/>models & telemetry| User

    style EsperLite fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style User fill:#50C878,stroke:#2D7A4A,color:#fff
    style TrainingData fill:#FFB347,stroke:#996633,color:#fff
```

---

## Level 2: Container Diagram

```mermaid
graph TB
    User["User/Researcher"]
    TrainingData["Training Data"]

    subgraph CLI["CLI Layer"]
        Scripts["Scripts<br/>(train, evaluate)"]
    end

    subgraph Core["Core Subsystems"]
        Kasmina["Kasmina<br/>(Body/Model)<br/>Seed lifecycle,<br/>quality gates"]
        Leyline["Leyline<br/>(Nervous System)<br/>Shared contracts,<br/>enums, signals"]
        Tamiyo["Tamiyo<br/>(Brain)<br/>Heuristic policy,<br/>lifecycle decisions"]
        Tolaria["Tolaria<br/>(Hands)<br/>Training loops,<br/>governor"]
    end

    subgraph Infrastructure["Infrastructure"]
        Simic["Simic<br/>(Gym)<br/>PPO agents,<br/>feature eng"]
        Nissa["Nissa<br/>(Senses)<br/>Telemetry hub,<br/>diagnostics"]
        Runtime["Runtime<br/>Task registry,<br/>presets"]
        Utils["Utils<br/>Data loading,<br/>helpers"]
    end

    User -->|CLI commands| Scripts
    TrainingData -->|Data loading| Utils

    Scripts -->|Orchestrates| Simic
    Scripts -->|Routes events| Nissa
    Scripts -->|Uses policies| Tamiyo
    Scripts -->|Runs training| Tolaria
    Scripts -->|Task config| Runtime

    Simic -->|Observations| Leyline
    Simic -->|Calls policy| Tamiyo
    Simic -->|Telemetry| Nissa
    Simic -->|Task config| Runtime

    Tamiyo -->|Decisions| Kasmina
    Tamiyo -->|Uses signals| Leyline
    Tamiyo -->|Telemetry| Nissa

    Tolaria -->|Trains| Kasmina
    Tolaria -->|Task config| Runtime
    Tolaria -->|Uses signals| Leyline

    Kasmina -->|Implements| Leyline

    Runtime -->|Uses| Kasmina
    Runtime -->|Uses| Simic
    Runtime -->|References| Leyline
    Runtime -->|Uses| Utils

    Nissa -->|References| Leyline

    style Kasmina fill:#FF6B9D,stroke:#C41E3A,color:#fff
    style Leyline fill:#4ECDC4,stroke:#1A9B8E,color:#fff
    style Tamiyo fill:#F7DC6F,stroke:#C9A227,color:#333
    style Tolaria fill:#BB8FCE,stroke:#7D3C98,color:#fff
    style Simic fill:#85C1E2,stroke:#3498DB,color:#fff
    style Nissa fill:#F8B739,stroke:#D68910,color:#333
    style Runtime fill:#95A5A6,stroke:#566573,color:#fff
    style Utils fill:#A3E4D7,stroke:#52BE80,color:#333
    style Scripts fill:#F5B7B1,stroke:#E74C3C,color:#fff
```

---

## Level 3: Component Diagram - Kasmina

```mermaid
graph TB
    subgraph Kasmina["Kasmina (Body/Model)"]
        subgraph Lifecycle["Seed Lifecycle Management"]
            SeedSlot["SeedSlot<br/>State machine,<br/>quality gates,<br/>lifecycle control"]
            SeedState["SeedState<br/>Complete state:<br/>id, stage, metrics,<br/>alpha schedule"]
            QualityGates["QualityGates<br/>6 gate levels<br/>G0-G5<br/>Stage transitions"]
        end

        subgraph Integration["Integration & Blending"]
            AlphaSchedule["AlphaSchedule<br/>Sigmoid-based<br/>alpha curve"]
            Isolation["Gradient Isolation<br/>STE training,<br/>alpha blending,<br/>gradient monitoring"]
        end

        subgraph Hosts["Host Networks"]
            HostProtocol["HostProtocol<br/>Structural typing<br/>interface"]
            CNNHost["CNNHost<br/>Conv network<br/>with injection points"]
            TransformerHost["TransformerHost<br/>GPT decoder<br/>with injection points"]
            MorphogeneticModel["MorphogeneticModel<br/>Wrapper: host +<br/>seed slot"]
        end

        subgraph Blueprints["Seed Blueprints"]
            Registry["BlueprintRegistry<br/>Plugin system"]
            CNNBlueprints["CNN Blueprints<br/>norm, attention,<br/>depthwise, conv"]
            TransformerBlueprints["Transformer Blueprints<br/>LoRA, attention,<br/>MLP, FlexAttention"]
        end
    end

    SeedSlot -->|Manages| SeedState
    SeedSlot -->|Enforces| QualityGates
    SeedSlot -->|Uses| AlphaSchedule
    SeedSlot -->|Applies| Isolation

    MorphogeneticModel -->|Wraps| HostProtocol
    MorphogeneticModel -->|Contains| SeedSlot

    HostProtocol -.->|Implemented by| CNNHost
    HostProtocol -.->|Implemented by| TransformerHost

    Registry -->|Instantiates| CNNBlueprints
    Registry -->|Instantiates| TransformerBlueprints

    SeedSlot -->|Uses| Registry

    style SeedSlot fill:#FF6B9D,stroke:#C41E3A,color:#fff
    style SeedState fill:#FF8FAC,stroke:#C41E3A,color:#fff
    style QualityGates fill:#FFAABC,stroke:#C41E3A,color:#fff
    style AlphaSchedule fill:#F8B4D4,stroke:#C41E3A,color:#fff
    style Isolation fill:#F5C3D1,stroke:#C41E3A,color:#fff
    style HostProtocol fill:#E8B4D4,stroke:#C41E3A,color:#fff
    style CNNHost fill:#E8B4D4,stroke:#C41E3A,color:#fff
    style TransformerHost fill:#E8B4D4,stroke:#C41E3A,color:#fff
    style MorphogeneticModel fill:#FFD1DC,stroke:#C41E3A,color:#fff
    style Registry fill:#A0E7E5,stroke:#16A085,color:#333
    style CNNBlueprints fill:#B5F2E8,stroke:#16A085,color:#333
    style TransformerBlueprints fill:#B5F2E8,stroke:#16A085,color:#333
```

---

## Diagram Notes

### Key Architectural Insights

1. **Layered Architecture:**
   - **CLI Layer:** Scripts provide entry points (train, evaluate)
   - **Core Subsystems:** The five logical components handling model, decisions, training, and nervous system
   - **Infrastructure:** Supporting systems for RL, telemetry, configuration, and utilities

2. **Dependency Flow:**
   - Leyline is the foundation layer - all subsystems depend on it for contracts and enums
   - Kasmina (Body) is the central computational unit that all others manipulate
   - Tamiyo (Brain) makes decisions that influence Kasmina's state transitions
   - Tolaria (Hands) executes the PyTorch training based on Kasmina and Leyline contracts
   - Simic (Gym) provides the RL agent that learns to control the system

3. **Kasmina Internal Structure:**
   - **SeedSlot** is the core state machine managing individual seed lifecycle
   - **SeedState** holds complete state snapshot (id, stage, metrics, alpha)
   - **QualityGates** enforce gated stage transitions (G0-G5)
   - **AlphaSchedule** and **Isolation** handle gradient isolation during training
   - **Host Networks** (CNN/Transformer) are graftable injection points
   - **BlueprintRegistry** provides plugin-based seed implementations

4. **Critical Patterns:**
   - State Machine (SeedSlot with SeedState)
   - Plugin Architecture (BlueprintRegistry)
   - Structural Typing (HostProtocol)
   - Quality Gates for rollout safety
   - Gradient Isolation via STE and alpha blending
   - Telemetry-First Design (all subsystems report to Nissa)

5. **Data Flow:**
   - Training data flows through Utils into task configurations
   - Observations flow from training loops through Simic to Tamiyo
   - Decisions flow from Tamiyo to Kasmina's SeedSlot
   - Telemetry events flow to Nissa hub from all subsystems

6. **Circular Dependency Avoidance:**
   - TYPE_CHECKING imports used in Kasmina, Tamiyo, Simic
   - Lazy imports in Leyline.actions for BlueprintRegistry
   - Local imports in Runtime.tasks to break cycles
