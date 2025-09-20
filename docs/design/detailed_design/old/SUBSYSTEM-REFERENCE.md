# ESPER SUBSYSTEM REFERENCE GUIDE
## Complete Catalog of All 14 Subsystems and Their Roles

**Date**: 2025-01-14
**Purpose**: Single reference document for all Esper subsystems
**Organization**: By functional plane and subsystem number

---

## ARCHITECTURE OVERVIEW

The Esper morphogenetic neural network training platform consists of **14 subsystems** (13 active services + 1 virtual):

### Active Subsystems (13)
1. **Execution Plane** - Core training execution (2 subsystems)
2. **Control Plane** - Runtime decision making (3 subsystems)
3. **Policy Training Plane** - Async policy learning (2 subsystems)
4. **Innovation Plane** - Async blueprint R&D (4 subsystems)
5. **Infrastructure Plane** - Foundation services (3 subsystems)

### Virtual Subsystem (1)
- **Leyline** (00) - Shared contract definitions (not a running service)

**Historical Note**: The architecture originally had 12 subsystems. Jace (11) was added for curriculum coordination, Elesh (13) was added for structural analysis, and Leyline (00) was added as a virtual subsystem for shared contracts, bringing the total to 14.

---

## SUBSYSTEM CATALOG

### 00. LEYLINE - Shared Contract Governance
**Type**: Virtual Subsystem (not a running service)
**Plane**: Cross-cutting (used by all planes)
**Owner**: Data Architect
**Status**: Production Ready v2.1.0

**Purpose**: Single source of truth for all shared cross-subsystem contracts, message schemas, enums, and interface definitions. Like MTG Leylines that establish foundational forces before gameplay begins.

**Key Responsibilities**:
- Define all shared Protocol Buffer message schemas
- Maintain consistent enums and constants across subsystems
- Enforce versioning and compatibility rules
- Prevent schema drift and serialization failures

**Not Responsible For**: Runtime operations (purely definitional)

---

## EXECUTION PLANE
*Manages the fundamental model training process with zero disruption*

### 01. TOLARIA - Training Orchestrator
**Role**: Master training loop controller and temporal framework
**Status**: Implementation Ready v3.0
**Integration**: Tight coupling with Kasmina-Tamiyo triad

**Purpose**: Serves as the master coordinator that provides the stable temporal framework (epochs and steps) in which the host model lives and evolves. Final authority on system stability.

**Key Responsibilities**:
- Own and execute the core PyTorch training loop
- Manage optimizer and learning rate schedules
- Coordinate epoch boundaries and checkpointing
- Execute emergency rollbacks when needed
- Maintain system heartbeat and stability

**Critical Metrics**:
- 18ms epoch boundary operations
- Zero training disruption guarantee
- 30-second rollback capability

### 02. KASMINA - Execution Layer
**Role**: Kernel execution engine and seed lifecycle manager
**Status**: Production Ready v3.0
**Integration**: Direct kernel execution, telemetry streaming

**Purpose**: Pure execution layer responsible for running compiled kernels on hardware, managing seed lifecycle, and exporting telemetry. The "hands" of the system that execute all computation.

**Key Responsibilities**:
- Execute compiled kernels on GPU hardware
- Manage seed lifecycle (creation, monitoring, termination)
- Export gradient and activation telemetry
- Apply pruning masks and architectural modifications
- Handle memory management and GPU allocation

**Critical Metrics**:
- 100μs kernel launch overhead
- 16GB GPU memory budget
- Real-time telemetry export

---

## CONTROL PLANE
*Runtime decision-making and architectural modifications*

### 03. TAMIYO - Strategic Controller
**Role**: Neural policy network for strategic decision-making
**Status**: Production Ready v3.0
**Integration**: Consumes telemetry, produces adaptation commands

**Purpose**: The "brain" of the system that uses learned policies (GNN-based) to make strategic decisions about when and how to evolve the neural architecture based on training dynamics.

**Key Responsibilities**:
- Analyze system state and training dynamics
- Make strategic adaptation decisions
- Query blueprint library for kernel selection
- Generate adaptation commands for execution
- Coordinate with Simic for policy updates

**Critical Metrics**:
- 50ms decision latency
- 1000 trajectory buffer
- 95% decision consistency

### 12. EMRAKUL - Architectural Sculptor (C&C)
**Role**: Command and control for large-scale modifications
**Status**: Production Ready v3.1 (with Leyline integration)
**Integration**: Coordinates with Elesh for analysis

**Purpose**: Pure command and control coordinator for large-scale architectural modifications. Orchestrates complex multi-step transformations including structured pruning operations.

**Key Responsibilities**:
- Coordinate architectural transformation operations
- Orchestrate checkpoint-based pruning pipeline
- Manage multi-step modification sequences
- Handle rollback coordination
- Interface between Elesh analysis and Kasmina execution

**Note**: Works as tight package with Elesh - Emrakul does C&C, Elesh does analysis

### 13. ELESH - Structural Analyzer
**Role**: Neural architecture analysis and optimization
**Status**: Production Ready v1.0
**Integration**: Tightly coupled with Emrakul

**Purpose**: Performs all structural analysis and optimization calculations for neural architectures, including importance scoring, redundancy detection, and pruning recommendations.

**Key Responsibilities**:
- Calculate channel importance scores (Taylor expansion)
- Detect attention head redundancy (cosine similarity)
- Analyze layer importance and connectivity
- Generate pruning recommendations
- Track importance metrics over time (Count-Min Sketch)

**Note**: Package deal with Emrakul - Elesh analyzes, Emrakul coordinates

---

## POLICY TRAINING PLANE
*Asynchronous policy learning and curriculum coordination*

### 04. SIMIC - Policy Trainer
**Role**: Reinforcement learning trainer for Tamiyo's policies
**Status**: Production Ready v3.0
**Integration**: Trains policies from field reports

**Purpose**: Continuously improves Tamiyo's decision-making policies through reinforcement learning on operational experience, using PPO algorithm with experience replay.

**Key Responsibilities**:
- Collect field reports from operations
- Train policy networks using PPO
- Manage experience replay buffer
- Validate policy improvements
- Deploy updated policies to Tamiyo

**Critical Metrics**:
- 1000 trajectory capacity
- 8 GPU training budget
- 90% policy improvement rate

### 11. JACE - Curriculum Coordinator
**Role**: Task and strategy alignment between Tamiyo and Simic
**Status**: Production Ready v3.0
**Integration**: Coordinates Tamiyo-Simic learning

**Purpose**: Ensures coherent learning curriculum, coordinating between Tamiyo's strategic decisions and Simic's policy training to prevent divergence.

**Key Responsibilities**:
- Coordinate curriculum between Tamiyo and Simic
- Detect and prevent strategy divergence
- Manage task complexity progression
- Ensure policy-action alignment
- Generate curriculum events

**Critical Metrics**:
- <100ms coordination overhead
- 95% curriculum coherence

**Note**: Works asynchronously like Innovation Plane - trains policies offline

---

## INNOVATION PLANE
*Continuous asynchronous improvement and validation*

### 05. KARN - Blueprint Generator
**Role**: Template selector (current), generative AI architect (planned)
**Status**: Production Ready v3.0
**Integration**: Generates blueprints for library

**Purpose**: Currently selects from template library to generate new blueprints. Future versions will use generative AI to create novel architectures based on operational feedback.

**Key Responsibilities**:
- Select appropriate templates based on requirements
- Generate BlueprintIR specifications
- (Future) Generate novel architectures via AI
- Submit blueprints to library
- Track blueprint performance metrics

**Critical Metrics**:
- 100 blueprints/day generation
- 90% compilation success rate

### 06. TEZZERET - Compilation Forge
**Role**: Blueprint compiler and optimization engine
**Status**: Production Ready v3.0
**Integration**: Compiles blueprints to kernels

**Purpose**: Asynchronously compiles blueprint specifications into optimized CUDA/Triton kernels, handling all low-level optimization and hardware-specific compilation.

**Key Responsibilities**:
- Compile BlueprintIR to CUDA/Triton kernels
- Optimize for specific hardware targets
- Handle torch.compile integration
- Manage compilation cache
- Produce compiled kernel artifacts

**Critical Metrics**:
- 8-minute compilation time
- 95% optimization success
- <1% compilation failures

### 07. URABRASK - Evaluation Engine
**Role**: Kernel validation and safety testing
**Status**: Production Ready v3.0
**Integration**: Validates compiled kernels

**Purpose**: Comprehensive validation engine that tests compiled kernels for safety, performance, and correctness before they can be deployed to production.

**Key Responsibilities**:
- Safety validation in sandboxed environment
- Performance benchmarking
- Memory leak detection
- Gradient health analysis
- Chaos testing for resilience

**Critical Metrics**:
- 30-second safety validation
- 8-minute standard benchmark
- 4GB validation GPU budget

### 08. URZA - Central Library
**Role**: Asset repository and version management
**Status**: Production Ready v3.0
**Integration**: Central storage for all subsystems

**Purpose**: Centralized repository for all blueprints, compiled kernels, and validated assets. Single source of truth for all architectural components.

**Key Responsibilities**:
- Store BlueprintIR and compiled kernels
- Manage versioning and dependencies
- Provide query API for asset discovery
- Track validation status and metrics
- Handle asset lifecycle and cleanup

**Critical Metrics**:
- <10ms query latency
- 99.99% availability
- Unlimited storage scaling

---

## INFRASTRUCTURE PLANE
*Foundation services for communication and observability*

### 09. OONA - Message Bus
**Role**: Event distribution and inter-subsystem communication
**Status**: Production Ready v2.0
**Integration**: All subsystems publish/subscribe

**Purpose**: High-performance message bus that enables asynchronous event-driven communication between all subsystems using publish/subscribe patterns.

**Key Responsibilities**:
- Route events between subsystems
- Guarantee message delivery
- Handle pub/sub patterns
- Manage message priorities
- Provide event replay capability

**Critical Metrics**:
- <1ms message latency
- 100K messages/second
- At-least-once delivery

### 10. NISSA - Observability Platform
**Role**: System monitoring, metrics, and audit trail
**Status**: Production Ready v2.0
**Integration**: Receives all telemetry

**Purpose**: Comprehensive observability platform providing metrics, logging, tracing, and dashboards for the entire Esper system.

**Key Responsibilities**:
- Collect and store metrics
- Aggregate logs from all subsystems
- Provide distributed tracing
- Generate alerts and notifications
- Maintain audit trail

**Critical Metrics**:
- 1-second metric granularity
- 30-day retention
- <100ms query response

---

## PLANE INTERACTIONS

### Runtime Operations (Synchronous)
The Execution Plane (Tolaria, Kasmina) works with Control Plane (Tamiyo, Emrakul, Elesh) for live training:
- Tolaria provides temporal framework
- Kasmina executes kernels and modifications
- Tamiyo makes strategic decisions
- Emrakul+Elesh handle architectural changes

### Async Training Operations
Policy Training Plane (Simic, Jace) operates asynchronously:
- Simic trains Tamiyo's policies offline using experience replay
- Jace coordinates curriculum between Tamiyo and Simic
- Similar to Innovation Plane - not part of runtime control loop

### Innovation Pipeline (Async)
The Innovation Plane creates new blueprints asynchronously:
- Karn generates new blueprints (will train Karn's generative AI in future)
- Tezzeret compiles them to kernels
- Urabrask validates safety and performance
- Urza stores validated assets

**Critical**: Tamiyo can only select from blueprints already in Urza's library - cannot request new ones on demand.

### Training Pipelines (Both Async)
- **Policy Training**: Simic trains Tamiyo's policies asynchronously
- **Blueprint Innovation**: Future Karn will use generative AI (also async)
- Both operate outside the runtime control loop

### Infrastructure Support
- Oona enables all inter-subsystem communication
- Nissa provides observability for all operations
- Leyline defines all shared contracts

---

## KEY ARCHITECTURAL PRINCIPLES

1. **Plane Separation**: Innovation Plane is completely async from Execution
2. **Zero Disruption**: Training never stops for morphogenetic operations
3. **Package Deals**: Emrakul+Elesh work as coordinated pair
4. **Blueprint Library**: Only pre-validated blueprints can be used
5. **Checkpoint-Based**: Major modifications happen at checkpoint boundaries
6. **Safety First**: Multiple validation layers before any change

---

## QUICK REFERENCE TABLE

| # | Subsystem | Plane | Role | Status |
|---|-----------|-------|------|--------|
| 00 | Leyline | Cross-cutting | Shared Contracts | Prod v2.1 |
| 01 | Tolaria | Training | Training Orchestrator | Impl Ready v3.0 |
| 02 | Kasmina | Training | Execution Layer | Prod v3.0 |
| 03 | Tamiyo | Control | Strategic Controller | Prod v3.0 |
| 04 | Simic | Control | Policy Trainer | Prod v3.0 |
| 05 | Karn | Innovation | Blueprint Generator | Prod v3.0 |
| 06 | Tezzeret | Innovation | Compilation Forge | Prod v3.0 |
| 07 | Urabrask | Innovation | Evaluation Engine | Prod v3.0 |
| 08 | Urza | Innovation | Central Library | Prod v3.0 |
| 09 | Oona | Infrastructure | Message Bus | Prod v2.0 |
| 10 | Nissa | Infrastructure | Observability | Prod v2.0 |
| 11 | Jace | Control | Curriculum Coordinator | Prod v3.0 |
| 12 | Emrakul | Control | Architectural Sculptor | Prod v3.1 |
| 13 | Elesh | Control | Structural Analyzer | Prod v1.0 |

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-14
**Next Review**: When new subsystems are added or major architectural changes occurod v3.0 |
| 03 | Tamiyo | Control | Strategic Controller | Prod v3.0 |
| 04 | Simic | Control | Policy Trainer | Prod v3.0 |
| 05 | Karn | Innovation | Blueprint Generator | Prod v3.0 |
| 06 | Tezzeret | Innovation | Compilation Forge | Prod v3.0 |
| 07 | Urabrask | Innovation | Evaluation Engine | Prod v3.0 |
| 08 | Urza | Innovation | Central Library | Prod v3.0 |
| 09 | Oona | Infrastructure | Message Bus | Prod v2.0 |
| 10 | Nissa | Infrastructure | Observability | Prod v2.0 |
| 11 | Jace | Control | Curriculum Coordinator | Prod v3.0 |
| 12 | Emrakul | Control | Architectural Sculptor | Prod v3.1 |
| 13 | Elesh | Control | Structural Analyzer | Prod v1.0 |
| 14 | Mycosynth | Infrastructure | Configuration Fabric | Proposed v1.0 |

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-14
**Next Review**: When new subsystems are added or major architectural changes occur