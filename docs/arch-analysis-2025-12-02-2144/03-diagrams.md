# Architecture Diagrams

**Project:** esper-lite
**Analysis Date:** 2025-12-02
**Notation:** C4 Model with Mermaid syntax

---

## C4 Level 1: System Context

Shows esper-lite in its environment with external actors and systems.

```mermaid
C4Context
    title System Context Diagram - Esper-Lite

    Person(researcher, "ML Researcher", "Trains and evaluates morphogenetic models")

    System(esper, "Esper-Lite", "Morphogenetic neural network training system with RL-controlled seed lifecycle")

    System_Ext(pytorch, "PyTorch", "Deep learning framework")
    System_Ext(hf, "HuggingFace", "Datasets and Transformers")
    System_Ext(gpu, "CUDA GPUs", "Training hardware")

    Rel(researcher, esper, "Trains models via CLI", "train.py, evaluate.py")
    Rel(esper, pytorch, "Uses for neural networks", "torch 2.8+")
    Rel(esper, hf, "Loads datasets", "datasets, transformers")
    Rel(esper, gpu, "Executes training", "CUDA streams")
```

### Context Description

| Element | Description |
|---------|-------------|
| **Esper-Lite** | Core system implementing morphogenetic seed lifecycle with RL control |
| **ML Researcher** | Primary user who initiates training and evaluates results |
| **PyTorch** | Underlying deep learning framework (torch 2.8.0+) |
| **HuggingFace** | External data source (CIFAR-10, TinyStories via datasets) |
| **CUDA GPUs** | Hardware for training acceleration |

---

## C4 Level 2: Container Diagram

Shows the major subsystems (packages) within esper-lite and their relationships.

```mermaid
C4Container
    title Container Diagram - Esper-Lite Subsystems

    Person(user, "ML Researcher")

    Container_Boundary(esper, "Esper-Lite") {
        Container(scripts, "Scripts", "Python CLI", "Entry points: train.py, evaluate.py")

        Container(simic, "Simic", "RL Infrastructure", "PPO training, reward shaping, vectorized multi-GPU")

        Container(tamiyo, "Tamiyo", "Strategy", "Signal tracking, decision-making, heuristic policy")

        Container(kasmina, "Kasmina", "Seed Mechanics", "Seed lifecycle, blueprints, gradient isolation, blending")

        Container(tolaria, "Tolaria", "Model Training", "Epoch trainers, model factory, watchdog")

        Container(nissa, "Nissa", "Telemetry", "Event routing, diagnostics, analytics")

        Container(leyline, "Leyline", "Contracts", "Data schemas, stages, signals, actions")

        Container(runtime, "Runtime", "Config", "Task specifications")

        Container(utils, "Utils", "Utilities", "Dataset loaders")
    }

    Rel(user, scripts, "Invokes training", "CLI")
    Rel(scripts, simic, "Starts training")
    Rel(scripts, nissa, "Configures telemetry")

    Rel(simic, tamiyo, "Gets decisions")
    Rel(simic, tolaria, "Trains model")
    Rel(simic, kasmina, "Manages seeds")
    Rel(simic, nissa, "Emits events")

    Rel(tamiyo, leyline, "Uses signals")
    Rel(kasmina, leyline, "Uses stages/gates")
    Rel(tolaria, runtime, "Gets task spec")
    Rel(runtime, kasmina, "Creates hosts")
    Rel(runtime, utils, "Loads data")

    Rel(leyline, kasmina, "Lazy import blueprints", "circular avoidance")
```

### Container Descriptions

| Container | Technology | Responsibility |
|-----------|------------|----------------|
| **Scripts** | Python CLI | Entry points for training (PPO, heuristic) and evaluation |
| **Simic** | PPO, PyTorch | Core RL infrastructure - agents, buffers, rewards, vectorized training |
| **Tamiyo** | Python | Strategic decision-making - signal tracking, heuristic policy |
| **Kasmina** | PyTorch nn.Module | Seed lifecycle mechanics - blueprints, isolation, blending |
| **Tolaria** | PyTorch | Model training infrastructure - epoch trainers, watchdog |
| **Nissa** | Python | Telemetry hub - event routing, diagnostics, analytics |
| **Leyline** | Python dataclasses | Data contracts - stages, signals, actions, schemas |
| **Runtime** | Python | Task specifications and wiring |
| **Utils** | PyTorch DataLoader | Dataset loading utilities |

---

## C4 Level 3: Component Diagram - Simic (RL Core)

Shows internal components of the most complex subsystem.

```mermaid
C4Component
    title Component Diagram - Simic (RL Training Infrastructure)

    Container_Boundary(simic, "Simic") {
        Component(ppo, "PPOAgent", "RL Agent", "Online policy gradient with entropy annealing")
        Component(actor_critic, "ActorCritic", "Neural Network", "Shared features + actor/critic heads")
        Component(buffer, "RolloutBuffer", "Experience Storage", "Trajectory buffer with GAE computation")

        Component(rewards, "Reward Shaping", "PBRS", "Accuracy/loss rewards, stage bonuses, compute rent")
        Component(features, "Feature Extraction", "Hot Path", "27-dim observation vector")
        Component(normalizer, "RunningMeanStd", "Normalization", "GPU-native observation normalization")

        Component(training, "Training Loop", "Orchestration", "Single-GPU episode runner")
        Component(vectorized, "Vectorized Training", "Multi-GPU", "CUDA streams, inverted control flow")

        Component(episodes, "Episode Manager", "Data Structures", "TrainingSnapshot, Episode, DatasetManager")
        Component(gradient_coll, "Gradient Collector", "Telemetry", "Async gradient statistics")
    }

    Rel(ppo, actor_critic, "Uses")
    Rel(ppo, buffer, "Stores transitions")
    Rel(ppo, rewards, "Computes reward")

    Rel(training, ppo, "Runs agent")
    Rel(training, features, "Extracts obs")
    Rel(training, episodes, "Collects")

    Rel(vectorized, ppo, "Runs parallel")
    Rel(vectorized, normalizer, "Normalizes")
    Rel(vectorized, gradient_coll, "Collects grads")

    Rel(buffer, actor_critic, "Feeds batches")
```

### Simic Component Descriptions

| Component | Responsibility | Hot Path? |
|-----------|----------------|-----------|
| **PPOAgent** | PPO algorithm with clip ratio, entropy, value loss | Yes |
| **ActorCritic** | Shared MLP features → actor/critic heads | Yes |
| **RolloutBuffer** | Trajectory storage, GAE computation | Per-episode |
| **Reward Shaping** | PBRS-based multi-component rewards | Yes |
| **Feature Extraction** | 27-dim observation from signals | **Critical** |
| **RunningMeanStd** | GPU-native normalization | Yes |
| **Training Loop** | Single-env episode orchestration | Yes |
| **Vectorized Training** | Multi-GPU with CUDA streams | **Critical** |
| **Episode Manager** | Trajectory data structures | No |
| **Gradient Collector** | Async gradient stats for telemetry | No |

---

## C4 Level 3: Component Diagram - Kasmina (Seed Mechanics)

Shows internal components of the seed lifecycle subsystem.

```mermaid
C4Component
    title Component Diagram - Kasmina (Seed Mechanics)

    Container_Boundary(kasmina, "Kasmina") {
        Component(slot, "SeedSlot", "Lifecycle Container", "State machine, quality gates, forward pass")
        Component(state, "SeedState", "State Tracking", "Stage, metrics, telemetry sync")
        Component(gates, "QualityGates", "Validation", "G0-G5 stage transition checks")

        Component(isolation, "Gradient Isolation", "Training", "Incubator mode, alpha scheduling, blend_with_isolation")
        Component(monitor, "IsolationMonitor", "Verification", "Gradient norm checking, violation tracking")

        Component(host, "HostProtocol", "Interface", "Structural typing for graftable hosts")
        Component(cnn_host, "CNNHost", "Implementation", "CNN with injection points")
        Component(transformer_host, "TransformerHost", "Implementation", "GPT-style with injection points")
        Component(morpho, "MorphogeneticModel", "Orchestrator", "Host + SeedSlot wrapper")

        Component(registry, "BlueprintRegistry", "Plugin System", "Decorator-based registration")
        Component(cnn_blueprints, "CNN Blueprints", "Architectures", "norm, attention, depthwise, conv")
        Component(transformer_blueprints, "Transformer Blueprints", "Architectures", "norm, LoRA, attention, MLP")
    }

    Rel(slot, state, "Contains")
    Rel(slot, gates, "Checks transitions")
    Rel(slot, isolation, "Blends features")
    Rel(slot, monitor, "Verifies isolation")

    Rel(morpho, slot, "Manages")
    Rel(morpho, cnn_host, "Wraps")
    Rel(morpho, transformer_host, "Wraps")

    Rel(slot, registry, "Creates seeds from")
    Rel(registry, cnn_blueprints, "Registers")
    Rel(registry, transformer_blueprints, "Registers")
```

### Kasmina Component Descriptions

| Component | Responsibility |
|-----------|----------------|
| **SeedSlot** | Lifecycle container - germinate, advance, cull, forward |
| **SeedState** | Tracks stage, metrics, syncs telemetry |
| **QualityGates** | G0-G5 validation for stage transitions |
| **Gradient Isolation** | Incubator mode STE, alpha schedules, blending |
| **IsolationMonitor** | Verifies gradient separation, tracks violations |
| **HostProtocol** | Structural typing for injection point discovery |
| **CNNHost** | Convolutional host with block injection points |
| **TransformerHost** | GPT-style host with layer injection points |
| **MorphogeneticModel** | Orchestrates host + seed lifecycle |
| **BlueprintRegistry** | Plugin system for seed architectures |
| **CNN/Transformer Blueprints** | Concrete seed module factories |

---

## Data Flow Diagram: Training Loop

Shows how data flows through the system during one training step.

```mermaid
flowchart TB
    subgraph Input
        data[DataLoader]
        signals[TrainingSignals]
    end

    subgraph Simic["Simic (RL)"]
        features[Feature Extraction<br>27-dim vector]
        agent[PPOAgent]
        reward[Reward Shaping]
        buffer[RolloutBuffer]
    end

    subgraph Tamiyo["Tamiyo"]
        tracker[SignalTracker]
        policy[HeuristicTamiyo<br>or PPO]
        decision[TamiyoDecision]
    end

    subgraph Kasmina["Kasmina"]
        slot[SeedSlot]
        gates[QualityGates]
        blend[blend_with_isolation]
    end

    subgraph Tolaria["Tolaria"]
        trainer[train_epoch_*]
        governor[TolariaGovernor]
    end

    subgraph Model["MorphogeneticModel"]
        host[Host Network]
        seed[Seed Module]
    end

    data --> trainer
    trainer --> Model
    Model --> signals

    signals --> tracker
    tracker --> features
    features --> agent
    agent --> decision

    decision --> slot
    slot --> gates
    gates --> |valid| blend
    blend --> seed

    trainer --> |loss| reward
    reward --> buffer
    buffer --> |update| agent

    governor --> |monitors| trainer
    governor --> |rollback| Model
```

---

## State Machine: Seed Lifecycle

Shows the seed stage transitions from Leyline.

```mermaid
stateDiagram-v2
    [*] --> DORMANT: create

    DORMANT --> GERMINATED: germinate
    GERMINATED --> TRAINING: G1 gate
    TRAINING --> BLENDING: G2 gate
    BLENDING --> SHADOWING: G3 gate
    SHADOWING --> PROBATIONARY: G4 gate
    PROBATIONARY --> FOSSILIZED: G5 gate (success)

    GERMINATED --> CULLED: failure
    TRAINING --> CULLED: failure
    BLENDING --> CULLED: failure
    SHADOWING --> CULLED: failure
    PROBATIONARY --> CULLED: failure

    CULLED --> EMBARGOED: cooldown
    EMBARGOED --> RESETTING: timeout
    RESETTING --> DORMANT: reset

    FOSSILIZED --> [*]: permanent
```

### Stage Descriptions

| Stage | Description | Gate Required |
|-------|-------------|---------------|
| **DORMANT** | Slot empty, ready for germination | - |
| **GERMINATED** | Seed created from blueprint | G0 (sanity) |
| **TRAINING** | Incubator mode - STE isolation | G1 (readiness) |
| **BLENDING** | Alpha ramp 0→1, co-adaptation | G2 (improvement) |
| **SHADOWING** | Alpha ≈ 1, monitoring | G3 (alpha threshold) |
| **PROBATIONARY** | Final validation | G4 (shadowing complete) |
| **FOSSILIZED** | Permanent integration (success) | G5 (positive improvement) |
| **CULLED** | Removed (failure) | - |
| **EMBARGOED** | Cooldown after cull | - |
| **RESETTING** | Cleanup before reuse | - |

---

## Deployment Diagram: Multi-GPU Training

Shows how vectorized training distributes across GPUs.

```mermaid
flowchart TB
    subgraph Host["Host Machine"]
        cpu[CPU<br>Policy Device]

        subgraph GPU0["GPU 0"]
            env0[Environment 0]
            stream0[CUDA Stream 0]
            model0[Model + Seed]
        end

        subgraph GPU1["GPU 1"]
            env1[Environment 1]
            stream1[CUDA Stream 1]
            model1[Model + Seed]
        end

        subgraph GPUn["GPU N"]
            envn[Environment N]
            streamn[CUDA Stream N]
            modeln[Model + Seed]
        end

        agent[PPOAgent<br>on CPU]
        normalizer[RunningMeanStd<br>on GPU]
        buffer[RolloutBuffer]
    end

    cpu --> agent
    agent --> buffer

    stream0 --> |async| env0
    stream1 --> |async| env1
    streamn --> |async| envn

    env0 --> |features| agent
    env1 --> |features| agent
    envn --> |features| agent

    agent --> |actions| env0
    agent --> |actions| env1
    agent --> |actions| envn

    buffer --> |update| agent
```

### Multi-GPU Training Notes

| Aspect | Implementation |
|--------|----------------|
| **Parallelism** | Each GPU runs independent environment |
| **Synchronization** | CUDA streams with epoch-boundary sync |
| **Control Flow** | Inverted: batch-first iteration |
| **DataLoaders** | Independent per-environment to avoid GIL |
| **Policy Device** | Agent on CPU for cross-GPU coordination |
| **Normalization** | GPU-native RunningMeanStd |

---

## Package Dependency Graph

Shows import relationships between packages.

```mermaid
graph TD
    scripts[scripts] --> simic
    scripts --> nissa

    simic --> leyline
    simic --> tamiyo
    simic --> tolaria
    simic --> nissa
    simic --> runtime

    tamiyo --> leyline
    tamiyo -.-> kasmina

    tolaria --> runtime
    tolaria --> kasmina

    kasmina --> leyline

    nissa --> leyline

    runtime --> kasmina
    runtime --> leyline
    runtime --> simic
    runtime --> utils

    leyline -.-> kasmina

    style leyline fill:#e1f5fe
    style simic fill:#fff3e0
    style kasmina fill:#e8f5e9
    style tamiyo fill:#fce4ec
    style nissa fill:#f3e5f5
    style tolaria fill:#fff8e1
    style runtime fill:#efebe9
    style utils fill:#eceff1
    style scripts fill:#e0e0e0
```

**Legend:**
- Solid arrows: Direct imports
- Dashed arrows: TYPE_CHECKING or lazy imports
- Colors: Package domains (blue=contracts, orange=RL, green=seeds, pink=strategy, purple=telemetry, yellow=training)

---

## Confidence Level

**HIGH** - All diagrams derived from validated subsystem catalog and verified source code structure. Mermaid notation chosen for portability and version control compatibility.
