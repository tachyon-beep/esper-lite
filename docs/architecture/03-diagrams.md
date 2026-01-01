# Esper Architecture Diagrams

**Analysis Date:** 2025-12-30
**Diagram Notation:** C4 Model (Context, Container, Component)

---

## 1. System Context Diagram (C1)

High-level view of Esper and its external interactions.

```mermaid
C4Context
    title Esper System Context

    Person(researcher, "ML Researcher", "Runs morphogenetic training experiments")

    System(esper, "Esper Framework", "Morphogenetic AI training system that grows neural network topology")

    System_Ext(pytorch, "PyTorch", "Deep learning framework")
    System_Ext(gpu, "CUDA GPUs", "Training compute")
    System_Ext(datasets, "Datasets", "CIFAR-10, TinyStories")
    System_Ext(browser, "Web Browser", "Overwatch dashboard viewer")

    Rel(researcher, esper, "Runs training via CLI", "CLI/TUI")
    Rel(esper, pytorch, "Uses for tensor ops", "Python API")
    Rel(esper, gpu, "Executes training on", "CUDA")
    Rel(esper, datasets, "Loads training data from", "File I/O")
    Rel(researcher, browser, "Views dashboards in")
    Rel(browser, esper, "Connects to Overwatch", "WebSocket")
```

---

## 2. Container Diagram (C2)

Major runtime components of Esper.

```mermaid
C4Container
    title Esper Container Diagram

    Person(researcher, "ML Researcher")

    Container_Boundary(esper, "Esper Framework") {
        Container(cli, "CLI/Scripts", "Python/argparse", "Entry point for training commands")
        Container(simic, "Simic Engine", "Python/PyTorch", "PPO RL training orchestration")
        Container(tamiyo, "Tamiyo Controller", "Python/PyTorch", "Seed lifecycle decision-making")
        Container(kasmina, "Kasmina Manager", "Python/PyTorch", "Seed slot mechanics & grafting")
        Container(tolaria, "Tolaria Executor", "Python/PyTorch", "Training execution & governor")
        Container(nissa, "Nissa Hub", "Python", "Telemetry routing & diagnostics")
        Container(karn, "Karn Memory", "Python/Textual/Vue", "Research telemetry & dashboards")
        Container(leyline, "Leyline Contracts", "Python", "Shared types & constants")
    }

    Container_Ext(sanctum, "Sanctum TUI", "Textual", "Developer debugging interface")
    Container_Ext(overwatch, "Overwatch Dashboard", "Vue 3", "Web monitoring interface")
    Container_Ext(mcp, "MCP SQL Server", "DuckDB", "SQL query interface")

    Rel(researcher, cli, "Invokes", "CLI")
    Rel(cli, simic, "Starts training")
    Rel(simic, tamiyo, "Gets decisions from")
    Rel(simic, kasmina, "Manages seeds via")
    Rel(simic, tolaria, "Executes training via")
    Rel(simic, nissa, "Emits telemetry to")
    Rel(nissa, karn, "Routes events to")
    Rel(karn, sanctum, "Renders in")
    Rel(karn, overwatch, "Streams to", "WebSocket")
    Rel(karn, mcp, "Exposes views via")

    Rel(tamiyo, leyline, "Uses contracts")
    Rel(kasmina, leyline, "Uses contracts")
    Rel(simic, leyline, "Uses contracts")
```

---

## 3. Component Diagram - Simic (C3)

Internal structure of the RL training engine.

```mermaid
C4Component
    title Simic Component Diagram

    Container_Boundary(simic, "Simic (Evolution)") {
        Component(ppo, "PPOAgent", "Python class", "On-policy gradient updates with factored heads")
        Component(buffer, "TamiyoRolloutBuffer", "Python class", "Per-env experience storage with LSTM states")
        Component(advantages, "Advantage Computer", "Python module", "Per-head causal masking for GAE")
        Component(rewards, "Reward Engine", "Python module", "PBRS, counterfactual, rent computation")
        Component(vectorized, "Vectorized Trainer", "Python module", "Inverted control flow, SharedBatchIterator")
        Component(emitters, "VectorizedEmitter", "Python class", "PPO telemetry event formatting")
        Component(anomaly, "AnomalyDetector", "Python class", "Phase-dependent pathology detection")
        Component(counterfactual, "CounterfactualHelper", "Python class", "Shapley/factorial attribution")
        Component(normalization, "RunningMeanStd", "Python class", "Online observation normalization")
    }

    Component_Ext(tamiyo_policy, "PolicyBundle", "Tamiyo", "Neural policy network")
    Component_Ext(kasmina_model, "MorphogeneticModel", "Kasmina", "Host with seed slots")
    Component_Ext(nissa_hub, "NissaHub", "Nissa", "Event routing")

    Rel(vectorized, ppo, "Calls update()")
    Rel(ppo, buffer, "Samples from")
    Rel(ppo, advantages, "Computes per-head")
    Rel(vectorized, rewards, "Computes rewards")
    Rel(vectorized, tamiyo_policy, "Gets actions from")
    Rel(vectorized, kasmina_model, "Trains")
    Rel(vectorized, emitters, "Emits telemetry via")
    Rel(emitters, nissa_hub, "Routes to")
    Rel(vectorized, anomaly, "Checks for pathologies")
    Rel(rewards, counterfactual, "Uses for attribution")
    Rel(ppo, normalization, "Normalizes observations")
```

---

## 4. Component Diagram - Kasmina (C3)

Internal structure of the seed lifecycle manager.

```mermaid
C4Component
    title Kasmina Component Diagram

    Container_Boundary(kasmina, "Kasmina (Stem Cells)") {
        Component(morph_model, "MorphogeneticModel", "nn.Module", "Host network with slot orchestration")
        Component(slot, "SeedSlot", "nn.Module", "Individual seed lifecycle manager")
        Component(alpha_ctrl, "AlphaController", "Python class", "Temporal amplitude scheduling")
        Component(gated_blend, "GatedBlend", "nn.Module", "Learned per-sample gating")
        Component(blend_ops, "BlendOps", "Python module", "Composition operators (add/multiply/gate)")
        Component(isolation, "GradientHealthMonitor", "Python class", "Gradient isolation & health")
        Component(gates, "QualityGates", "Python class", "G0-G5 stage transition validators")
        Component(registry, "BlueprintRegistry", "Python class", "Plugin system for seed architectures")
        Component(cnn_blueprints, "CNN Blueprints", "nn.Module", "8 CNN seed types")
        Component(transformer_blueprints, "Transformer Blueprints", "nn.Module", "8 Transformer seed types")
    }

    Component_Ext(host, "CNNHost/TransformerHost", "Kasmina", "Backbone networks")

    Rel(morph_model, slot, "Orchestrates")
    Rel(morph_model, host, "Wraps")
    Rel(slot, alpha_ctrl, "Schedules amplitude with")
    Rel(slot, gated_blend, "Per-sample blending via")
    Rel(slot, blend_ops, "Composes with")
    Rel(slot, isolation, "Monitors gradients with")
    Rel(slot, gates, "Validates transitions with")
    Rel(slot, registry, "Creates seeds from")
    Rel(registry, cnn_blueprints, "Registers")
    Rel(registry, transformer_blueprints, "Registers")
```

---

## 5. Component Diagram - Karn (C3)

Internal structure of the telemetry and visualization system.

```mermaid
C4Component
    title Karn Component Diagram

    Container_Boundary(karn, "Karn (Memory)") {
        Component(collector, "KarnCollector", "Python class", "Central event hub & routing")
        Component(store, "TelemetryStore", "Python class", "In-memory event database")
        Component(health, "HealthMonitor", "Python class", "System vitals tracking")
        Component(triggers, "AnomalyDetector", "Python class", "Dense trace triggers")

        Component_Boundary(sanctum, "Sanctum TUI") {
            Component(sanctum_backend, "SanctumBackend", "OutputBackend", "Event → aggregator bridge")
            Component(aggregator, "SanctumAggregator", "Python class", "Stateful event processor")
            Component(sanctum_app, "SanctumApp", "Textual App", "Terminal UI application")
            Component(widgets, "Widgets", "Textual Widgets", "20+ visualization widgets")
        }

        Component_Boundary(overwatch, "Overwatch Web") {
            Component(overwatch_backend, "OverwatchBackend", "Python class", "WebSocket server + static files")
            Component(vue_app, "Vue App", "Vue 3 SPA", "Browser dashboard")
            Component(vue_components, "Components", "Vue components", "15+ visualization components")
        }

        Component_Boundary(mcp_sql, "MCP SQL") {
            Component(mcp_server, "KarnMCPServer", "Python class", "SQL query interface")
            Component(duckdb, "DuckDB Views", "SQL", "7 pre-defined views")
        }
    }

    Component_Ext(nissa_hub, "NissaHub", "Nissa", "Event source")

    Rel(nissa_hub, collector, "Routes events to")
    Rel(collector, store, "Stores in")
    Rel(collector, sanctum_backend, "Feeds")
    Rel(collector, overwatch_backend, "Feeds")
    Rel(collector, health, "Updates")
    Rel(collector, triggers, "Checks")
    Rel(sanctum_backend, aggregator, "Updates")
    Rel(aggregator, sanctum_app, "Renders in")
    Rel(sanctum_app, widgets, "Composes")
    Rel(overwatch_backend, vue_app, "Streams to", "WebSocket")
    Rel(vue_app, vue_components, "Renders")
    Rel(store, mcp_server, "Exposes via")
    Rel(mcp_server, duckdb, "Queries")
```

---

## 6. Data Flow Diagram

How data flows through Esper during training.

```mermaid
flowchart TD
    subgraph Input
        DS[("Dataset<br/>(CIFAR-10)")]
    end

    subgraph "Simic (Vectorized Training)"
        SBI[SharedBatchIterator]
        ENV1[Env 0<br/>ParallelEnvState]
        ENV2[Env 1<br/>ParallelEnvState]
        ENVN[Env N<br/>ParallelEnvState]
        PPO[PPOAgent.update]
        BUF[TamiyoRolloutBuffer]
    end

    subgraph "Kasmina (Model)"
        HOST[Host Network]
        SLOT1[Slot r0c0]
        SLOT2[Slot r0c1]
        SLOT3[Slot r0c2]
    end

    subgraph "Tamiyo (Policy)"
        LSTM[FactoredRecurrentActorCritic]
        MASK[ActionMasks]
        FEAT[Features]
    end

    subgraph "Tolaria"
        GOV[TolariaGovernor]
    end

    subgraph "Nissa → Karn"
        HUB[NissaHub]
        KARN[KarnCollector]
        TUI[Sanctum TUI]
        WEB[Overwatch Web]
    end

    DS --> SBI
    SBI --> ENV1 & ENV2 & ENVN

    ENV1 & ENV2 & ENVN --> HOST
    HOST --> SLOT1 & SLOT2 & SLOT3
    SLOT1 & SLOT2 & SLOT3 --> |loss, accuracy| ENV1 & ENV2 & ENVN

    ENV1 & ENV2 & ENVN --> |observations| FEAT
    FEAT --> LSTM
    LSTM --> |actions| MASK
    MASK --> |valid actions| ENV1 & ENV2 & ENVN

    ENV1 & ENV2 & ENVN --> |experience| BUF
    BUF --> PPO
    PPO --> |gradients| LSTM

    ENV1 & ENV2 & ENVN --> |loss| GOV
    GOV --> |rollback signal| ENV1 & ENV2 & ENVN

    ENV1 & ENV2 & ENVN --> |events| HUB
    PPO --> |metrics| HUB
    HUB --> KARN
    KARN --> TUI & WEB
```

---

## 7. Seed Lifecycle State Machine

State transitions for neural module seeds.

```mermaid
stateDiagram-v2
    [*] --> DORMANT: Slot empty

    DORMANT --> GERMINATED: GERMINATE action

    GERMINATED --> TRAINING: G1 gate pass

    TRAINING --> BLENDING: G2 gate pass<br/>(gradient activity)
    TRAINING --> PRUNED: Poor performance

    BLENDING --> HOLDING: G3 gate pass<br/>(alpha → 1.0)
    BLENDING --> PRUNED: Regression

    HOLDING --> FOSSILIZED: G5 gate pass<br/>(positive contribution)
    HOLDING --> PRUNED: Ransomware pattern

    FOSSILIZED --> [*]: Success (permanent)

    PRUNED --> EMBARGOED: Cleanup
    EMBARGOED --> RESETTING: Cooldown
    RESETTING --> DORMANT: Slot recycled
```

---

## 8. Dependency Graph

Domain import relationships.

```mermaid
graph TD
    subgraph "Shared Contracts"
        LEY[Leyline<br/>DNA/Genome]
    end

    subgraph "Core Domains"
        KAS[Kasmina<br/>Stem Cells]
        TAM[Tamiyo<br/>Brain/Cortex]
        SIM[Simic<br/>Evolution]
        TOL[Tolaria<br/>Metabolism]
    end

    subgraph "Telemetry"
        NIS[Nissa<br/>Sensory Organs]
        KAR[Karn<br/>Memory]
    end

    subgraph "Supporting"
        RUN[Runtime]
        UTL[Utils]
        SCR[Scripts]
    end

    LEY --> KAS & TAM & SIM & TOL & NIS & KAR & RUN

    KAS --> SIM
    TAM --> SIM
    NIS --> KAR

    KAS & TAM & SIM --> RUN
    UTL --> SIM & RUN

    SIM & TOL --> SCR
    KAR --> SCR

    TOL -.->|lazy| RUN
```

---

## 9. Deployment View

How Esper runs on hardware.

```mermaid
C4Deployment
    title Esper Deployment (Typical)

    Deployment_Node(workstation, "Developer Workstation", "Linux") {
        Deployment_Node(python, "Python 3.11+") {
            Container(esper, "Esper Process", "Training + TUI")
        }
        Deployment_Node(browser, "Browser") {
            Container(overwatch, "Overwatch Dashboard", "Vue 3 SPA")
        }
    }

    Deployment_Node(gpu_server, "GPU Server", "Linux + CUDA") {
        Deployment_Node(gpu0, "GPU 0", "CUDA") {
            Container(env0, "Env 0-1", "Training streams")
        }
        Deployment_Node(gpu1, "GPU 1", "CUDA") {
            Container(env1, "Env 2-3", "Training streams")
        }
    }

    Rel(esper, env0, "CUDA calls")
    Rel(esper, env1, "CUDA calls")
    Rel(esper, overwatch, "WebSocket", "localhost:8080")
```

---

## 10. Sequence Diagram - PPO Training Round

One round of PPO training.

```mermaid
sequenceDiagram
    participant CLI as Scripts/CLI
    participant VEC as Vectorized Trainer
    participant SBI as SharedBatchIterator
    participant ENV as ParallelEnvState
    participant KAS as MorphogeneticModel
    participant TAM as PolicyBundle
    participant PPO as PPOAgent
    participant NIS as NissaHub

    CLI->>VEC: train_ppo_vectorized()

    loop Each Epoch
        VEC->>SBI: next batch
        SBI-->>VEC: combined_batch

        par For each environment
            VEC->>ENV: train_on_batch(slice)
            ENV->>KAS: forward(x)
            KAS-->>ENV: loss, accuracy
            ENV->>TAM: get_action(obs, masks)
            TAM-->>ENV: action, log_prob, value
            ENV-->>VEC: experience tuple
        end

        VEC->>PPO: buffer.add(experiences)
    end

    VEC->>PPO: update()
    PPO->>TAM: evaluate_actions()
    TAM-->>PPO: log_probs, values, entropy
    PPO->>PPO: compute_loss()
    PPO->>TAM: backward()
    PPO-->>VEC: PPOUpdateMetrics

    VEC->>NIS: emit(PPO_UPDATE_COMPLETED)
    NIS-->>VEC: ok
```

---

## Diagram Legend

| Symbol | Meaning |
|--------|---------|
| Rectangle | Container/Component |
| Cylinder | Database/Store |
| Person | External user |
| Dashed arrow | Async/lazy dependency |
| Solid arrow | Synchronous call |
| `-->` | Data flow direction |
