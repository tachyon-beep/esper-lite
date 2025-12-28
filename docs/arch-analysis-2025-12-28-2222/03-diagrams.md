# Architecture Diagrams

**Analysis Date:** 2025-12-28

This document contains C4 architecture diagrams for the Esper morphogenetic neural network framework.

---

## 1. System Context Diagram (C4 Level 1)

Shows Esper in its operational environment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ESPER SYSTEM CONTEXT                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   ML Researcher │
                              │     (Actor)     │
                              └────────┬────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                         ▼             ▼             ▼
                    ┌────────┐   ┌──────────┐   ┌─────────┐
                    │  CLI   │   │ Sanctum  │   │Overwatch│
                    │Terminal│   │   TUI    │   │   Web   │
                    └────┬───┘   └────┬─────┘   └────┬────┘
                         │            │              │
                         └────────────┼──────────────┘
                                      │
                                      ▼
            ┌─────────────────────────────────────────────────────┐
            │                                                     │
            │                    ESPER SYSTEM                     │
            │                                                     │
            │   Morphogenetic Neural Network Training Framework   │
            │                                                     │
            │   • Dynamic model topology growth/pruning           │
            │   • PPO-based seed lifecycle optimization           │
            │   • Real-time telemetry and observability           │
            │                                                     │
            └───────────────────────┬─────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────┐  ┌───────────┐  ┌─────────────────┐
            │   PyTorch   │  │  DuckDB   │  │   Claude Code   │
            │   Runtime   │  │ Analytics │  │  (MCP Client)   │
            └─────────────┘  └───────────┘  └─────────────────┘
```

---

## 2. Container Diagram (C4 Level 2)

Shows the seven domain "organs" and support modules.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ESPER CONTAINER DIAGRAM                             │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           CLI / SCRIPTS (680 LOC)                        │
    │                         Entry points, arg parsing                        │
    └────────────────────────────────────┬────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│     TAMIYO      │            │      SIMIC      │            │      KARN       │
│   (3,811 LOC)   │            │   (13,352 LOC)  │            │ (8,341+8,722)   │
│                 │            │                 │            │                 │
│  Brain/Cortex   │◄──────────►│    Evolution    │───────────►│     Memory      │
│                 │            │                 │            │                 │
│ • PolicyBundle  │            │ • PPOAgent      │            │ • TelemetryStore│
│ • Heuristic     │            │ • Training Loop │            │ • Sanctum TUI   │
│ • LSTM Policy   │            │ • Rewards       │            │ • Overwatch Web │
│ • SignalTracker │            │ • Attribution   │            │ • MCP Server    │
└────────┬────────┘            └────────┬────────┘            └────────┬────────┘
         │                              │                              │
         │                              │                              │
         │            ┌─────────────────┼─────────────────┐            │
         │            │                 │                 │            │
         ▼            ▼                 ▼                 ▼            │
    ┌─────────────────────┐    ┌─────────────────┐    ┌────────────────┤
    │      KASMINA        │    │     TOLARIA     │    │                │
    │     (5,174 LOC)     │    │    (462 LOC)    │    │                │
    │                     │    │                 │    │                │
    │    Stem Cells       │    │   Metabolism    │    │                │
    │                     │    │                 │    │                │
    │ • SeedSlot          │    │ • Governor      │    │                │
    │ • Quality Gates     │    │ • Rollback      │    │                │
    │ • Blueprints        │    │ • Checkpoints   │    │                │
    │ • Alpha Controller  │    │                 │    │                │
    └──────────┬──────────┘    └────────┬────────┘    │                │
               │                        │             │                │
               │                        ▼             ▼                │
               │                  ┌─────────────────────┐              │
               │                  │       NISSA         │◄─────────────┘
               │                  │     (1,969 LOC)     │
               │                  │                     │
               │                  │   Sensory Organs    │
               │                  │                     │
               │                  │ • NissaHub          │
               │                  │ • Pub-Sub Backends  │
               │                  │ • DiagnosticTracker │
               │                  └──────────┬──────────┘
               │                             │
               └─────────────────────────────┼─────────────────────────┐
                                             │                         │
                                             ▼                         ▼
                           ┌─────────────────────────────────────────────────┐
                           │                    LEYLINE                       │
                           │                  (3,735 LOC)                     │
                           │                                                  │
                           │                DNA / Genome                      │
                           │                                                  │
                           │  • SeedStage enum, VALID_TRANSITIONS             │
                           │  • FactoredAction, ACTION_HEAD_SPECS             │
                           │  • TelemetryEvent, 18 typed payloads             │
                           │  • All DEFAULT_* constants                       │
                           │  • SlotConfig, observation schemas               │
                           └─────────────────────────────────────────────────┘

                           ┌─────────────────────────────────────────────────┐
                           │               SUPPORT MODULES                    │
                           ├──────────────────┬──────────────────────────────┤
                           │  Runtime (298)   │  Utils (802 LOC)             │
                           │  • TaskSpec      │  • SharedBatchIterator       │
                           │  • Task presets  │  • GPU preloading            │
                           └──────────────────┴──────────────────────────────┘
```

---

## 3. Component Diagram: Simic (Evolution)

The largest subsystem - PPO-based reinforcement learning infrastructure.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SIMIC COMPONENT DIAGRAM (13,352 LOC)                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │              training/                   │
                    │           (2,535 LOC)                    │
                    │                                          │
                    │  • train_ppo_vectorized()                │
                    │  • TrainingConfig                        │
                    │  • Vectorized environment loop           │
                    │  • GPU-first iteration pattern           │
                    └─────────────────┬───────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│       agent/         │  │       rewards/       │  │     attribution/     │
│     (2,695 LOC)      │  │     (1,895 LOC)      │  │      (716 LOC)       │
│                      │  │                      │  │                      │
│  • PPOAgent          │  │  • RewardMode enum   │  │  • CounterfactualEng │
│  • TamiyoRolloutBuf  │  │  • Contribution      │  │  • Shapley values    │
│  • GAE computation   │  │  • PBRS potentials   │  │  • Factorial method  │
│  • Per-env buffers   │  │  • Reward scaling    │  │  • Credit assignment │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────────────────┘
           │                         │
           │                         │
           ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│      telemetry/      │  │       control/       │
│     (2,227 LOC)      │  │      (244 LOC)       │
│                      │  │                      │
│  • Gradient collect  │  │  • RunningMean       │
│  • AnomalyDetector   │  │  • Normalization     │
│  • PPO metrics emit  │  │                      │
└──────────────────────┘  └──────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │           KEY DATA FLOWS                 │
                    ├─────────────────────────────────────────┤
                    │                                          │
                    │  Environments ──► SharedBatchIterator    │
                    │       │                  │               │
                    │       ▼                  ▼               │
                    │  Model Forward ◄── GPU Batch             │
                    │       │                                  │
                    │       ▼                                  │
                    │  Tamiyo Decisions ──► Kasmina Exec       │
                    │       │                                  │
                    │       ▼                                  │
                    │  Rewards ──► RolloutBuffer ──► PPO       │
                    │       │                                  │
                    │       ▼                                  │
                    │  Attribution ──► Per-Head Credit         │
                    │                                          │
                    └─────────────────────────────────────────┘
```

---

## 4. Component Diagram: Kasmina (Stem Cells)

Seed lifecycle management and module composition.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KASMINA COMPONENT DIAGRAM (5,174 LOC)                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────────────────┐
                           │          slot.py            │
                           │        (2,610 LOC)          │
                           │                             │
                           │  THE LIFECYCLE ENGINE       │
                           │                             │
                           │  • SeedSlot class           │
                           │  • Stage transitions        │
                           │  • Quality gates G0-G5      │
                           │  • Alpha scheduling         │
                           │  • DDP sync                 │
                           └──────────────┬──────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │     host.py       │ │   blending.py     │ │ alpha_controller  │
        │    (769 LOC)      │ │    (241 LOC)      │ │    (187 LOC)      │
        │                   │ │                   │ │                   │
        │  • CNNHost        │ │  • BlendAlgorithm │ │  • AlphaCurve     │
        │  • TransformerHost│ │  • GatedBlend     │ │  • LINEAR         │
        │  • Morphogenetic  │ │  • blend_add()    │ │  • COSINE         │
        │    Model          │ │  • blend_multiply │ │  • SIGMOID        │
        │  • HostProtocol   │ │  • blend_gate()   │ │  • Time-based     │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────┐
        │                  blueprints/                     │
        │                                                  │
        │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
        │  │ registry.py │  │   cnn.py    │  │transformer│ │
        │  │  (128 LOC)  │  │  (295 LOC)  │  │  (345)   │ │
        │  │             │  │             │  │          │ │
        │  │ @blueprint  │  │ • norm_seed │  │ • lora   │ │
        │  │ decorator   │  │ • attn_seed │  │ • attn   │ │
        │  │ factory     │  │ • bottleneck│  │ • mlp    │ │
        │  └─────────────┘  └─────────────┘  └──────────┘ │
        └─────────────────────────────────────────────────┘
```

---

## 5. Component Diagram: Karn (Memory)

Telemetry, visualization, and analytics.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         KARN COMPONENT DIAGRAM (17,063 LOC)                      │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────┐
                              │    Root (2,841 LOC)     │
                              │                         │
                              │  • TelemetryStore       │
                              │  • KarnCollector        │
                              │  • Health monitoring    │
                              │  • Triggers             │
                              │  • Serialization        │
                              └───────────┬─────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
    │    mcp/ (294 LOC)   │   │ sanctum/ (3,591+)   │   │overwatch/ (9,083+)  │
    │                     │   │                     │   │                     │
    │  Claude Code Bridge │   │   Textual TUI       │   │  Vue 3 Dashboard    │
    │                     │   │                     │   │                     │
    │  • DuckDB backend   │   │  ┌───────────────┐  │   │  ┌───────────────┐  │
    │  • SQL queries      │   │  │  aggregator   │  │   │  │  backend.py   │  │
    │  • JSONL → Views    │   │  │   (800 LOC)   │  │   │  │  (361 LOC)    │  │
    │                     │   │  │               │  │   │  │               │  │
    │  Views:             │   │  │  Raw events   │  │   │  │  WebSocket    │  │
    │  • runs             │   │  │  → Snapshot   │  │   │  │  FastAPI      │  │
    │  • epochs           │   │  │               │  │   │  │               │  │
    │  • ppo_updates      │   │  └───────────────┘  │   │  └───────────────┘  │
    │  • seed_lifecycle   │   │                     │   │                     │
    │  • rewards          │   │  ┌───────────────┐  │   │  ┌───────────────┐  │
    │  • anomalies        │   │  │  widgets/     │  │   │  │  web/ (8,722) │  │
    │                     │   │  │  (2,000+ LOC) │  │   │  │               │  │
    └─────────────────────┘   │  │               │  │   │  │  12 Vue       │  │
                              │  │  16 widgets   │  │   │  │  components   │  │
                              │  │  • PPO panel  │  │   │  │  + composables│  │
                              │  │  • Seed cards │  │   │  │               │  │
                              │  │  • Metrics    │  │   │  │  • Vite build │  │
                              │  │  • Charts     │  │   │  │  • TypeScript │  │
                              │  └───────────────┘  │   │  └───────────────┘  │
                              └─────────────────────┘   └─────────────────────┘
```

---

## 6. Seed Lifecycle State Machine

The botanical development process for neural modules.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SEED LIFECYCLE STATE MACHINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

                                 ┌─────────────┐
                                 │   DORMANT   │ ◄──────────────────────┐
                                 │             │                        │
                                 │  Inactive,  │                        │
                                 │  awaiting   │                        │
                                 │  germinate  │                        │
                                 └──────┬──────┘                        │
                                        │                               │
                                        │ germinate()                   │
                                        │                               │
                                        ▼                               │
                                 ┌─────────────┐                        │
                                 │ GERMINATED  │                        │
                                 │             │                        │
                                 │  Module     │                        │
                                 │  created,   │                        │
                                 │  init phase │                        │
                                 └──────┬──────┘                        │
                                        │                               │
                                        │ G1 gate passed                │
                                        │                               │
                                        ▼                               │
                                 ┌─────────────┐                        │
                    ┌───────────►│  TRAINING   │◄───────────┐           │
                    │            │             │            │           │
                    │            │  Learning   │            │           │
                    │            │  from host  │            │           │
                    │            │  errors     │            │           │
                    │            └──────┬──────┘            │           │
                    │                   │                   │           │
                    │          ┌────────┴────────┐          │           │
                    │          │                 │          │           │
                    │          ▼                 ▼          │           │
                    │   ┌─────────────┐   ┌─────────────┐   │           │
                    │   │  BLENDING   │   │   PRUNED    │───┼───────────┤
                    │   │             │   │             │   │           │
                    │   │  Integrating│   │  Removed    │   │           │
                    │   │  with host  │   │  (poor perf)│   │           │
                    │   └──────┬──────┘   └─────────────┘   │           │
                    │          │                            │           │
                    │          │ G3 gate passed             │           │
                    │          │                            │           │
                    │          ▼                            │           │
                    │   ┌─────────────┐                     │           │
                    │   │   HOLDING   │                     │           │
                    │   │             │                     │           │
                    │   │  Stabilizing│                     │           │
                    │   │  alpha=1.0  │                     │           │
                    │   └──────┬──────┘                     │           │
                    │          │                            │           │
                    │          │ G5 gate passed             │           │
                    │          │                            │           │
                    │          ▼                            │           │
                    │   ┌─────────────┐                     │           │
                    │   │ FOSSILIZED  │                     │           │
                    │   │             │                     │           │
                    │   │  Permanent  │                     │           │
                    │   │  fusion     │                     │           │
                    │   └─────────────┘                     │           │
                    │                                       │           │
                    │                                       │           │
                    │   ┌─────────────┐   ┌─────────────┐   │           │
                    └───┤  EMBARGOED  │◄──┤  RESETTING  │◄──┘           │
                        │             │   │             │               │
                        │  Cooldown   │   │  Cleaning   │───────────────┘
                        │  period     │   │  state      │
                        └─────────────┘   └─────────────┘


                    ┌─────────────────────────────────────────┐
                    │            QUALITY GATES                 │
                    ├─────────────────────────────────────────┤
                    │  G0: Germination entry                   │
                    │  G1: Training readiness                  │
                    │  G2: Blending eligibility                │
                    │  G3: Holding confirmation                │
                    │  G4: Fossilization pre-check             │
                    │  G5: Final fossilization                 │
                    ├─────────────────────────────────────────┤
                    │  Modes: PERMISSIVE (dev) / STRICT (prod) │
                    └─────────────────────────────────────────┘
```

---

## 7. Data Flow: Training Loop

How data flows through the system during PPO training.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING LOOP DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  SharedBatch    │    ┌─────────────────────────────────────────────────────────┐
│  Iterator       │    │                    GPU-FIRST LOOP                        │
│                 │    │                                                          │
│  • CIFAR-10     │───►│  for batch in iterator:                                  │
│  • TinyStories  │    │      # Dispatch to N environments                        │
│  • GPU-resident │    │      for env_id in range(n_envs):                        │
└─────────────────┘    │          model[env_id].forward(batch)                    │
                       │          # Compute loss, accumulate                       │
                       │                                                          │
                       └──────────────────────┬───────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TAMIYO DECISION CYCLE                                  │
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐    │
│  │ SignalTracker   │────►│ PolicyBundle    │────►│ TamiyoDecision          │    │
│  │                 │     │                 │     │                         │    │
│  │ • Loss trend    │     │ • Heuristic OR  │     │ • germinate: [slots]    │    │
│  │ • Grad health   │     │ • Neural LSTM   │     │ • prune: [slots]        │    │
│  │ • Seed metrics  │     │                 │     │ • advance: [slots]      │    │
│  └─────────────────┘     └─────────────────┘     └───────────┬─────────────┘    │
│                                                              │                   │
└──────────────────────────────────────────────────────────────┼──────────────────┘
                                                               │
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          KASMINA EXECUTION                                       │
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐    │
│  │ SeedSlot        │────►│ Quality Gates   │────►│ State Transition        │    │
│  │                 │     │                 │     │                         │    │
│  │ • Execute op    │     │ • G0-G5 checks  │     │ • DORMANT → GERMINATED  │    │
│  │ • Update alpha  │     │ • Mode toggle   │     │ • TRAINING → BLENDING   │    │
│  │ • Blend weights │     │                 │     │ • etc.                  │    │
│  └─────────────────┘     └─────────────────┘     └───────────┬─────────────┘    │
│                                                              │                   │
└──────────────────────────────────────────────────────────────┼──────────────────┘
                                                               │
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          REWARD & ATTRIBUTION                                    │
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐    │
│  │ Contribution    │────►│ PBRS Shaping    │────►│ Counterfactual          │    │
│  │                 │     │                 │     │                         │    │
│  │ • Loss delta    │     │ • Stage pots    │     │ • Shapley values        │    │
│  │ • Improvement   │     │ • Smooth reward │     │ • Per-head credit       │    │
│  └─────────────────┘     └─────────────────┘     └───────────┬─────────────┘    │
│                                                              │                   │
└──────────────────────────────────────────────────────────────┼──────────────────┘
                                                               │
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PPO UPDATE                                              │
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐    │
│  │ RolloutBuffer   │────►│ GAE Advantages  │────►│ Policy Gradient         │    │
│  │                 │     │                 │     │                         │    │
│  │ • Per-env       │     │ • Per-env       │     │ • 8 factored heads      │    │
│  │ • States        │     │ • λ=0.95        │     │ • Causal masking        │    │
│  │ • Actions       │     │                 │     │ • Clip ratio            │    │
│  │ • Rewards       │     │                 │     │                         │    │
│  └─────────────────┘     └─────────────────┘     └───────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Data Flow: Telemetry Pipeline

How telemetry events flow from source to visualization.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TELEMETRY DATA FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

   EVENT SOURCES                    NISSA HUB                     BACKENDS
   ─────────────                    ─────────                     ────────

┌─────────────────┐
│     Simic       │
│  • PPO metrics  │
│  • Gradients    │────┐
│  • Anomalies    │    │
└─────────────────┘    │
                       │     ┌─────────────────────────────┐
┌─────────────────┐    │     │         NissaHub            │
│    Kasmina      │    │     │                             │
│  • Seed events  │────┼────►│  • Typed event dispatch     │────┐
│  • Stage changes│    │     │  • Async queue worker       │    │
│  • Gate results │    │     │  • Multi-backend fanout     │    │
└─────────────────┘    │     │                             │    │
                       │     └─────────────────────────────┘    │
┌─────────────────┐    │                                        │
│    Tamiyo       │    │                                        │
│  • Decisions    │────┤                                        │
│  • Policy acts  │    │                                        │
└─────────────────┘    │                                        │
                       │                                        │
┌─────────────────┐    │                                        │
│    Tolaria      │    │                                        │
│  • Rollbacks    │────┘                                        │
│  • Panics       │                                             │
└─────────────────┘                                             │
                                                                │
                       ┌────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                              BACKENDS                                    │
   │                                                                          │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
   │  │  ConsoleOutput  │  │  FileOutput     │  │  DirectoryOutput        │  │
   │  │                 │  │                 │  │                         │  │
   │  │  Rich console   │  │  Single JSONL   │  │  Per-run directories    │  │
   │  │  formatting     │  │  file           │  │  with telemetry files   │  │
   │  └─────────────────┘  └────────┬────────┘  └───────────┬─────────────┘  │
   │                                │                       │                 │
   └────────────────────────────────┼───────────────────────┼─────────────────┘
                                    │                       │
                                    ▼                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                              KARN STORAGE                                │
   │                                                                          │
   │  ┌─────────────────────────────────────────────────────────────────┐    │
   │  │                      TelemetryStore                              │    │
   │  │                                                                  │    │
   │  │  Episode Context ──► Epoch Summaries ──► Dense Traces            │    │
   │  │      (sparse)            (periodic)         (batch-level)        │    │
   │  │                                                                  │    │
   │  └──────────────────────────────┬──────────────────────────────────┘    │
   │                                 │                                        │
   └─────────────────────────────────┼────────────────────────────────────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
   ┌─────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
   │   MCP Server    │   │   Sanctum TUI       │   │   Overwatch Web     │
   │                 │   │                     │   │                     │
   │  JSONL → DuckDB │   │  Aggregator →       │   │  WebSocket →        │
   │  → SQL views    │   │  SanctumSnapshot    │   │  Vue components     │
   │  → Claude Code  │   │  → 16 widgets       │   │  → 12 components    │
   │                 │   │                     │   │                     │
   │  Views:         │   │  • PPO panel        │   │  • Dashboard        │
   │  • runs         │   │  • Seed cards       │   │  • Seed grid        │
   │  • epochs       │   │  • Metrics          │   │  • Charts           │
   │  • ppo_updates  │   │  • Charts           │   │  • Real-time        │
   │  • rewards      │   │                     │   │                     │
   └─────────────────┘   └─────────────────────┘   └─────────────────────┘
```

---

## 9. Dependency Graph

Import relationships between domains.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DOMAIN DEPENDENCY GRAPH                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                              FOUNDATION LAYER
                    ┌─────────────────────────────────┐
                    │            LEYLINE              │
                    │                                 │
                    │  • No outbound dependencies     │
                    │  • ALL domains import from here │
                    └─────────────────────────────────┘
                                    ▲
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        │                           │                           │

   INFRASTRUCTURE LAYER
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│   NISSA     │           │   TOLARIA   │           │   RUNTIME   │
│             │           │             │           │             │
│  Sensory    │           │  Metabolism │           │  Task       │
│  (hub)      │           │  (governor) │           │  Config     │
└──────┬──────┘           └──────┬──────┘           └──────┬──────┘
       │                         │                         │
       │                         │                         │
       ▼                         ▼                         ▼

   DOMAIN LAYER
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │   TAMIYO    │   │   KASMINA   │   │        KARN         │   │
│  │             │   │             │   │                     │   │
│  │  Brain      │   │  Stem Cells │   │  Memory             │   │
│  │  (policy)   │   │  (lifecycle)│   │  (telemetry/viz)    │   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘   │
│         │                 │                      │              │
│         │                 │                      │              │
│         └────────┬────────┘                      │              │
│                  │                               │              │
│                  ▼                               │              │
│         ┌─────────────────────────────────────┐  │              │
│         │              SIMIC                  │◄─┘              │
│         │                                     │                 │
│         │  Evolution (PPO training)           │                 │
│         │                                     │                 │
│         │  Depends on:                        │                 │
│         │  • Leyline (contracts)              │                 │
│         │  • Kasmina* (via protocols)         │                 │
│         │  • Tamiyo (policy bundles)          │                 │
│         │  • Nissa (telemetry emission)       │                 │
│         │  • Tolaria (Governor)               │                 │
│         │  • Karn (collector)                 │                 │
│         │  • Utils (data loading)             │                 │
│         │                                     │                 │
│         │  * Protocol-based, not direct       │                 │
│         └─────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
       │
       │
       ▼

   ENTRY LAYER
┌─────────────────────────────────────────────────────────────────┐
│                           SCRIPTS                               │
│                                                                 │
│  CLI entry points that wire everything together                 │
│  Depends on: Simic, Nissa, Karn                                 │
└─────────────────────────────────────────────────────────────────┘


                    ┌─────────────────────────────────────────┐
                    │           DEPENDENCY MATRIX              │
                    ├─────────────────────────────────────────┤
                    │                                          │
                    │  FROM ↓  TO →  Ley Kas Tam Sim Nis Tol Kar Run Scr Uti
                    │  ──────────────────────────────────────  │
                    │  Leyline        -   -   -   -   -   -   -   -   -   -  │
                    │  Kasmina        ✓   -   -   -   -   -   -   -   -   -  │
                    │  Tamiyo         ✓   -   -   -   ✓   -   -   -   -   -  │
                    │  Simic          ✓   ✓*  ✓   -   ✓   ✓   ✓   -   -   ✓  │
                    │  Nissa          ✓   -   -   -   -   -   -   -   -   -  │
                    │  Tolaria        ✓   -   -   -   ✓   -   -   ✓   -   -  │
                    │  Karn           ✓   -   -   -   -   -   -   -   -   -  │
                    │  Runtime        ✓   ✓   ✓   -   -   -   -   -   -   ✓  │
                    │  Scripts        ✓   -   -   ✓   ✓   -   ✓   -   -   -  │
                    │  Utils          -   -   -   -   -   -   -   -   -   -  │
                    │                                          │
                    │  * Via protocol contracts, not direct    │
                    └─────────────────────────────────────────┘
```

---

## 10. Factored Action Space

The 8-head policy architecture for seed lifecycle decisions.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FACTORED ACTION SPACE (8 HEADS)                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────────┐
                         │   Observation Vector    │
                         │                         │
                         │  • Per-slot features    │
                         │  • Training signals     │
                         │  • Global state         │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │    Shared Encoder       │
                         │                         │
                         │  LSTM (if neural)       │
                         │  or direct (heuristic)  │
                         └───────────┬─────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│    HEAD 0       │       │    HEAD 1       │       │    HEAD 2       │
│    slot_id      │       │   blueprint     │       │    style        │
│                 │       │                 │       │                 │
│  Which slot to  │       │  Which module   │       │  Module config  │
│  operate on     │       │  template       │       │  variant        │
│                 │       │                 │       │                 │
│  Dim: n_slots   │       │  Dim: n_bprints │       │  Dim: n_styles  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│    HEAD 3       │       │    HEAD 4       │       │    HEAD 5       │
│    tempo        │       │   alpha_curve   │       │  alpha_duration │
│                 │       │                 │       │                 │
│  Training speed │       │  Blend schedule │       │  Blend duration │
│  modifier       │       │  shape          │       │  in batches     │
│                 │       │                 │       │                 │
│  Dim: n_tempos  │       │  Dim: n_curves  │       │  Dim: n_durs    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                   ┌─────────────────┼─────────────────┐
                   │                 │                 │
                   ▼                 ▼                 ▼
         ┌─────────────────┐ ┌─────────────────┐
         │    HEAD 6       │ │    HEAD 7       │
         │  lifecycle_op   │ │    value        │
         │                 │ │                 │
         │  Action type:   │ │  State value    │
         │  • GERMINATE    │ │  estimate       │
         │  • ADVANCE      │ │  (critic head)  │
         │  • PRUNE        │ │                 │
         │  • HOLD         │ │  Dim: 1         │
         │                 │ │                 │
         │  Dim: n_ops     │ │                 │
         └─────────────────┘ └─────────────────┘


                    ┌─────────────────────────────────────────┐
                    │           ACTION MASKING                 │
                    ├─────────────────────────────────────────┤
                    │                                          │
                    │  • Causal masking per head               │
                    │  • Only active heads receive gradients   │
                    │  • Invalid actions masked to -inf        │
                    │  • Stage-dependent validity              │
                    │                                          │
                    │  Example:                                │
                    │  If slot_id selects empty slot,          │
                    │  only GERMINATE is valid for HEAD 6      │
                    │                                          │
                    └─────────────────────────────────────────┘
```

---

## Summary

These diagrams capture the key architectural views of Esper:

| Diagram | Level | Purpose |
|---------|-------|---------|
| System Context | C4 L1 | Esper in its environment |
| Container | C4 L2 | Seven domains + support modules |
| Simic Components | C4 L3 | PPO training infrastructure |
| Kasmina Components | C4 L3 | Seed lifecycle engine |
| Karn Components | C4 L3 | Telemetry and visualization |
| Seed Lifecycle | State Machine | Botanical stage progression |
| Training Loop | Data Flow | GPU-first training iteration |
| Telemetry Pipeline | Data Flow | Event sourcing to visualization |
| Dependency Graph | Architecture | Domain import relationships |
| Factored Actions | Architecture | 8-head policy structure |

All diagrams rendered in ASCII for version control and accessibility.
