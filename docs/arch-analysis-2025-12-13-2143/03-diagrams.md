# Architecture Diagrams - esper-lite

**Analysis Date:** 2025-12-13
**Diagram Format:** Mermaid (C4 Model)
**Source:** Validated subsystem catalog (02-subsystem-catalog.md)

---

## 1. C4 Context Diagram

Shows esper-lite in its operational context with external systems and users.

```mermaid
C4Context
    title System Context - esper-lite Morphogenetic Training System

    Person(researcher, "ML Researcher", "Trains and evaluates morphogenetic models")
    Person(operator, "System Operator", "Monitors training runs, configures telemetry")

    System(esper, "esper-lite", "Morphogenetic neural network training system using RL-optimized seed adaptation strategies")

    System_Ext(pytorch, "PyTorch", "Deep learning framework (>=2.8)")
    System_Ext(cuda, "CUDA Runtime", "GPU compute and memory management")
    System_Ext(datasets, "HuggingFace Datasets", "Dataset loading (CIFAR-10, TinyStories)")
    System_Ext(transformers, "HuggingFace Transformers", "Transformer architectures")
    System_Ext(filesystem, "File System", "Checkpoints, telemetry output, configs")

    Rel(researcher, esper, "Trains models", "CLI commands")
    Rel(operator, esper, "Monitors/configures", "Telemetry configs")
    Rel(esper, pytorch, "Uses", "torch.compile, DDP, CUDA streams")
    Rel(esper, cuda, "Executes on", "GPU kernels, memory")
    Rel(esper, datasets, "Loads data from", "Streaming/cached")
    Rel(esper, transformers, "Uses architectures", "GPT, attention")
    Rel(esper, filesystem, "Persists to", "Checkpoints, JSONL logs")
```

---

## 2. C4 Container Diagram

Shows the major subsystems (containers) within esper-lite and their relationships.

```mermaid
C4Container
    title Container Diagram - esper-lite Subsystems

    Person(user, "User", "ML Researcher / Operator")

    System_Boundary(esper, "esper-lite") {
        Container(scripts, "Scripts", "Python CLI", "Entry points: train, evaluate")
        Container(simic, "Simic", "Python/PyTorch", "RL Infrastructure: PPO, vectorized training, rewards")
        Container(tolaria, "Tolaria", "Python/PyTorch", "Training execution engine, failure monitoring")
        Container(tamiyo, "Tamiyo", "Python", "Decision policy: heuristic controller, signal tracking")
        Container(kasmina, "Kasmina", "Python/PyTorch", "Seed lifecycle: slots, blueprints, gradient isolation")
        Container(nissa, "Nissa", "Python", "Telemetry hub: tracking, analytics, output backends")
        Container(runtime, "Runtime", "Python", "Task presets and factories")
        Container(utils, "Utils", "Python/PyTorch", "Data loading, loss computation")
        Container(leyline, "Leyline", "Python", "Data contracts: signals, stages, actions, schemas")
    }

    Rel(user, scripts, "Invokes", "python -m esper.scripts.train")
    Rel(scripts, simic, "Orchestrates RL training")
    Rel(scripts, tolaria, "Orchestrates supervised training")
    Rel(scripts, tamiyo, "Uses signal tracking")
    Rel(scripts, nissa, "Configures telemetry")
    Rel(scripts, runtime, "Gets task specs")

    Rel(simic, tolaria, "Delegates epoch training")
    Rel(simic, tamiyo, "Uses decision policy")
    Rel(simic, kasmina, "Manages seeds")
    Rel(simic, nissa, "Emits telemetry")
    Rel(simic, runtime, "Gets task configs")
    Rel(simic, utils, "Uses data/loss")
    Rel(simic, leyline, "Uses contracts")

    Rel(tolaria, kasmina, "Trains seeds")
    Rel(tolaria, runtime, "Gets model factories")
    Rel(tolaria, utils, "Uses data/loss")

    Rel(tamiyo, leyline, "Uses contracts")
    Rel(tamiyo, nissa, "Emits telemetry")

    Rel(kasmina, leyline, "Uses contracts")

    Rel(nissa, leyline, "Uses event types")

    Rel(runtime, kasmina, "Creates slots")
    Rel(runtime, utils, "Uses data loading")
    Rel(runtime, leyline, "Uses contracts")
```

---

## 3. Dependency Graph (Acyclic)

Simplified view showing the strict acyclic dependency structure.

```mermaid
flowchart TB
    subgraph "Layer 0: Foundation"
        leyline[Leyline<br/>Data Contracts]
        utils[Utils<br/>Data/Loss]
    end

    subgraph "Layer 1: Core Domain"
        kasmina[Kasmina<br/>Seed Mechanics]
        nissa[Nissa<br/>Telemetry]
    end

    subgraph "Layer 2: Policy & Execution"
        tamiyo[Tamiyo<br/>Decisions]
        tolaria[Tolaria<br/>Training]
    end

    subgraph "Layer 3: Integration"
        runtime[Runtime<br/>Task Presets]
        simic[Simic<br/>RL Infrastructure]
    end

    subgraph "Layer 4: Interface"
        scripts[Scripts<br/>CLI]
    end

    %% Dependencies (arrows point to dependencies)
    kasmina --> leyline
    nissa --> leyline

    tamiyo --> leyline
    tamiyo --> nissa
    tamiyo -.->|TYPE_CHECKING| kasmina

    tolaria --> kasmina
    tolaria --> runtime
    tolaria --> utils

    runtime --> leyline
    runtime --> kasmina
    runtime --> utils
    runtime -.->|TYPE_CHECKING| simic

    simic --> leyline
    simic --> kasmina
    simic --> tamiyo
    simic --> tolaria
    simic --> nissa
    simic --> runtime
    simic --> utils

    scripts --> simic
    scripts --> tolaria
    scripts --> tamiyo
    scripts --> nissa
    scripts --> runtime
    scripts --> utils
    scripts --> leyline

    style leyline fill:#e1f5fe
    style utils fill:#e1f5fe
    style kasmina fill:#fff3e0
    style nissa fill:#fff3e0
    style tamiyo fill:#f3e5f5
    style tolaria fill:#f3e5f5
    style runtime fill:#e8f5e9
    style simic fill:#e8f5e9
    style scripts fill:#fce4ec
```

---

## 4. Component Diagram: Simic (RL Infrastructure)

Detailed view of the largest subsystem (8,290 LOC).

```mermaid
flowchart TB
    subgraph simic["Simic Subsystem (8,290 LOC)"]
        subgraph training_core["Training Core"]
            vectorized[vectorized.py<br/>1,496 LOC<br/>Multi-GPU Parallel Training]
            training[training.py<br/>519 LOC<br/>Single Episode Training]
            ppo[ppo.py<br/>577 LOC<br/>PPO Agent]
        end

        subgraph networks["Neural Networks"]
            networks_py[networks.py<br/>482 LOC<br/>PolicyNetwork, ActorCritic]
            tamiyo_network[tamiyo_network.py<br/>334 LOC<br/>LSTM + 4-Head Output]
        end

        subgraph rewards_system["Reward System"]
            rewards[rewards.py<br/>988 LOC<br/>PBRS, Counterfactual, Rent]
            advantages[advantages.py<br/>Factored GAE]
        end

        subgraph memory["Experience Memory"]
            tamiyo_buffer[tamiyo_buffer.py<br/>395 LOC<br/>Pre-allocated Rollout Buffer]
            prioritized[prioritized_buffer.py<br/>Sum-tree PER]
        end

        subgraph features_obs["Observations"]
            features[features.py<br/>387 LOC<br/>35-dim Feature Extraction]
            action_masks[action_masks.py<br/>Valid Action Masking]
        end

        subgraph telemetry["Telemetry"]
            ppo_telemetry[ppo_telemetry.py]
            reward_telemetry[reward_telemetry.py]
            debug_telemetry[debug_telemetry.py]
            memory_telemetry[memory_telemetry.py]
        end

        subgraph config["Configuration"]
            config_py[config.py<br/>TrainingConfig, LossRewardConfig]
            telemetry_config[telemetry_config.py<br/>Tiered Telemetry Levels]
        end
    end

    vectorized --> training
    vectorized --> ppo
    vectorized --> rewards
    vectorized --> features

    training --> ppo
    training --> tamiyo_buffer

    ppo --> networks_py
    ppo --> tamiyo_network
    ppo --> advantages

    rewards --> features

    style vectorized fill:#ffcdd2
    style rewards fill:#ffcdd2
    style ppo fill:#fff9c4
    style tamiyo_network fill:#fff9c4
```

---

## 5. Component Diagram: Kasmina (Seed Mechanics)

Detailed view of the seed lifecycle management (2,935 LOC).

```mermaid
flowchart TB
    subgraph kasmina["Kasmina Subsystem (2,935 LOC)"]
        subgraph slot_mgmt["Slot Management"]
            slot[slot.py<br/>1,319 LOC<br/>SeedSlot, SeedState, Quality Gates G0-G5]
        end

        subgraph host_models["Host Models"]
            host[host.py<br/>635 LOC<br/>MorphogeneticModel, CNNHost, TransformerHost]
        end

        subgraph isolation["Gradient Isolation"]
            isolation_py[isolation.py<br/>221 LOC<br/>AlphaSchedule, STE, GradientIsolationMonitor]
            blending[blending.py<br/>Topology-aware Blending]
        end

        subgraph blueprints["Blueprint System"]
            registry[blueprints/registry.py<br/>125 LOC<br/>Decorator-based Plugin System]
            cnn_bp[blueprints/cnn.py<br/>206 LOC<br/>7 CNN Blueprints]
            transformer_bp[blueprints/transformer.py<br/>197 LOC<br/>5 Transformer Blueprints]
        end

        subgraph protocol["Protocol"]
            protocol_py[protocol.py<br/>HostProtocol ABC]
        end
    end

    slot --> host
    slot --> isolation_py
    slot --> registry

    host --> protocol_py
    host --> blending

    isolation_py --> blending

    registry --> cnn_bp
    registry --> transformer_bp

    style slot fill:#ffcdd2
    style host fill:#fff9c4
    style isolation_py fill:#c8e6c9
```

---

## 6. Seed Lifecycle State Machine

State transitions managed by Kasmina with quality gates.

```mermaid
stateDiagram-v2
    [*] --> DORMANT: Create seed

    DORMANT --> GERMINATED: germinate()
    GERMINATED --> TRAINING: G0 pass
    GERMINATED --> CULLED: G0 fail

    TRAINING --> TRAINING: continue training
    TRAINING --> BLENDING: G1+G2 pass
    TRAINING --> CULLED: anomaly detected

    BLENDING --> BLENDING: continue blending
    BLENDING --> PROBATIONARY: G3 pass
    BLENDING --> CULLED: G3 fail

    PROBATIONARY --> PROBATIONARY: monitoring
    PROBATIONARY --> FOSSILIZED: G5 pass
    PROBATIONARY --> CULLED: G5 fail

    FOSSILIZED --> [*]: Permanent integration

    CULLED --> EMBARGOED: cooldown period
    EMBARGOED --> RESETTING: cleanup
    RESETTING --> DORMANT: ready to retry

    note right of TRAINING
        Quality Gates:
        G0: Initial viability
        G1: Loss improvement
        G2: Gradient health
        G3: Blend stability
        G5: Long-term stability
        (G4 deprecated)
    end note
```

---

## 7. Data Flow: PPO Training Loop

Shows the flow of data through the RL training system.

```mermaid
flowchart LR
    subgraph environment["Vectorized Environments"]
        env1[Env 1]
        env2[Env 2]
        env3[Env 3]
        envN[Env N]
    end

    subgraph observation["Observation Pipeline"]
        signals[FastTrainingSignals<br/>from Tolaria]
        features[features.py<br/>35-dim extraction]
        obs[Observation Tensor]
    end

    subgraph policy["Policy Network"]
        lstm[LSTM<br/>Temporal Context]
        heads[4 Action Heads<br/>op, slot, blueprint, blend]
        masks[Action Masks<br/>Valid Actions Only]
    end

    subgraph action["Action Execution"]
        factored[FactoredAction<br/>Decoded Action]
        kasmina_exec[Kasmina<br/>Execute on Slot]
    end

    subgraph reward["Reward Computation"]
        pbrs[PBRS<br/>Stage Potential]
        counterfactual[Counterfactual<br/>Validation]
        rent[Compute Rent<br/>Resource Cost]
        total[Total Reward]
    end

    subgraph memory["Experience Buffer"]
        buffer[TamiyoRolloutBuffer<br/>Pre-allocated]
        gae[GAE<br/>Advantage Estimation]
    end

    subgraph update["Policy Update"]
        ppo_loss[PPO Loss<br/>Clipped Surrogate]
        value_loss[Value Loss<br/>Clipped MSE]
        entropy[Entropy Bonus]
        optimizer[AdamW<br/>Fused/Foreach]
    end

    env1 & env2 & env3 & envN --> signals
    signals --> features --> obs
    obs --> lstm --> heads
    masks --> heads
    heads --> factored --> kasmina_exec

    kasmina_exec --> pbrs & counterfactual & rent
    pbrs & counterfactual & rent --> total

    obs & factored & total --> buffer
    buffer --> gae

    gae --> ppo_loss & value_loss
    heads --> entropy
    ppo_loss & value_loss & entropy --> optimizer
    optimizer -.->|Update| lstm & heads
```

---

## 8. Telemetry Flow

Shows how telemetry flows through the Nissa hub.

```mermaid
flowchart TB
    subgraph producers["Telemetry Producers"]
        simic_tel[Simic<br/>RL Metrics]
        tolaria_tel[Tolaria<br/>Training Metrics]
        kasmina_tel[Kasmina<br/>Seed Events]
        tamiyo_tel[Tamiyo<br/>Decision Events]
    end

    subgraph nissa["Nissa Hub"]
        hub[NissaHub<br/>Central Router]
        tracker[DiagnosticTracker<br/>Gradient/Loss Stats]
        analytics[BlueprintAnalytics<br/>Performance Tracking]
    end

    subgraph backends["Output Backends"]
        console[ConsoleOutput<br/>Real-time Display]
        file[FileOutput<br/>JSONL Logs]
        directory[DirectoryOutput<br/>Structured Artifacts]
    end

    subgraph config["Configuration"]
        profiles[Profiles<br/>minimal/standard/diagnostic/research]
        levels[Telemetry Levels<br/>MINIMAL/NORMAL/DEBUG]
    end

    simic_tel & tolaria_tel & kasmina_tel & tamiyo_tel --> hub
    hub --> tracker & analytics
    hub --> console & file & directory
    profiles --> hub
    levels --> hub

    style hub fill:#fff9c4
```

---

## 9. Deployment View

Shows how esper-lite is deployed and executed.

```mermaid
flowchart TB
    subgraph user_machine["User Machine"]
        cli[CLI Invocation<br/>python -m esper.scripts.train]
    end

    subgraph compute["Compute Environment"]
        subgraph single_gpu["Single GPU Mode"]
            torch_compile[torch.compile<br/>mode=reduce-overhead]
            cuda_graphs[CUDA Graphs<br/>Kernel Fusion]
        end

        subgraph multi_gpu["Multi-GPU Mode (Future)"]
            ddp[DDP<br/>DistributedDataParallel]
            nccl[NCCL<br/>Collective Comms]
        end

        subgraph memory["Memory Management"]
            preallocated[Pre-allocated Buffers<br/>TamiyoRolloutBuffer]
            shared_batch[SharedBatchIterator<br/>Single DataLoader]
            gpu_cache[GPU Dataset Cache<br/>Amortized Loading]
        end
    end

    subgraph storage["Storage"]
        checkpoints[Checkpoints<br/>Model State + Extra State]
        telemetry_out[Telemetry<br/>JSONL + Directories]
        datasets[Datasets<br/>CIFAR-10, TinyStories]
    end

    cli --> torch_compile
    torch_compile --> cuda_graphs
    torch_compile -.->|Future| ddp
    ddp -.-> nccl

    cuda_graphs --> preallocated
    preallocated --> shared_batch
    shared_batch --> gpu_cache

    gpu_cache --> datasets
    cuda_graphs --> checkpoints
    cuda_graphs --> telemetry_out

    style ddp fill:#ffcdd2,stroke-dasharray: 5 5
    style nccl fill:#ffcdd2,stroke-dasharray: 5 5
```

---

## 10. Issue Hotspots (from Expert Reviews)

Visual representation of where critical issues were identified.

```mermaid
flowchart TB
    subgraph critical["CRITICAL Issues"]
        c1[training.py:30-31<br/>Global Mutable State<br/>USE_COMPILED_TRAIN_STEP]
    end

    subgraph high["HIGH Issues"]
        h1[training.py:193-197<br/>Missing Imports<br/>BLUEPRINT_IDS, etc.]
        h2[slot.py:1150-1270<br/>DDP Deadlock Risk<br/>Stage Divergence]
        h3[ppo.py, vectorized.py<br/>No AMP Support<br/>Missing 30-50% Speedup]
        h4[signals.py:152<br/>Counterfactual Clamping<br/>Missing Symmetry]
        h5[slot.py:190<br/>SeedState slots=True<br/>Memory Inefficiency]
    end

    subgraph medium["MEDIUM Issues"]
        m1[ppo.py:158<br/>Large value_clip=10.0]
        m2[rewards.py:549-560<br/>Reward Scale Asymmetry]
        m3[isolation.py:145<br/>Private API Usage<br/>torch._foreach_norm]
    end

    simic[Simic Subsystem] --> c1 & h1 & h3 & m1 & m2
    kasmina[Kasmina Subsystem] --> h2 & h5
    leyline[Leyline Subsystem] --> h4
    tolaria[Tolaria Subsystem] --> m3

    style c1 fill:#d32f2f,color:#fff
    style h1 fill:#f57c00,color:#fff
    style h2 fill:#f57c00,color:#fff
    style h3 fill:#f57c00,color:#fff
    style h4 fill:#f57c00,color:#fff
    style h5 fill:#f57c00,color:#fff
```

---

## Diagram Index

| # | Diagram | Type | Purpose |
|---|---------|------|---------|
| 1 | System Context | C4 Context | External systems and users |
| 2 | Container Diagram | C4 Container | Subsystem relationships |
| 3 | Dependency Graph | Flowchart | Acyclic dependency structure |
| 4 | Simic Components | C4 Component | RL infrastructure details |
| 5 | Kasmina Components | C4 Component | Seed mechanics details |
| 6 | Seed Lifecycle | State Diagram | Stage transitions and gates |
| 7 | PPO Data Flow | Flowchart | Training loop data movement |
| 8 | Telemetry Flow | Flowchart | Observability pipeline |
| 9 | Deployment View | Flowchart | Execution environment |
| 10 | Issue Hotspots | Flowchart | Critical issue locations |

---

## Notes

- All diagrams use Mermaid syntax for portability
- C4 diagrams follow the C4 model conventions (Context, Container, Component)
- Dashed lines indicate TYPE_CHECKING or future/planned features
- Color coding: Red = critical path, Yellow = important, Green = stable
