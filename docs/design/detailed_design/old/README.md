# Esper Platform Architecture Overview

Welcome to the architectural documentation for the Esper morphogenetic neural network training platform. This document serves as a starting point for understanding the high-level design and the roles of its various components.

## Introduction

The Esper platform is designed for the autonomous training and evolution of neural networks. It employs a "morphogenetic" approach, where the neural network's architecture can change and adapt during the training process. This is achieved through a collection of specialized, independent subsystems that work together to manage the training loop, make strategic decisions, generate and validate new architectural components, and provide the underlying infrastructure.

## Architectural Vision

The architecture is built on a set of core principles:

- **Plane Separation**: The system is divided into distinct functional planes (e.g., Execution, Control, Innovation). The Innovation Plane, responsible for creating and validating new network components, is fully asynchronous from the live training loop to prevent any disruption.
- **Zero Disruption**: The core training process must never be halted or delayed by the morphogenetic operations. Architectural changes are applied safely and asynchronously.
- **Safety and Resilience**: Every component is designed with safety in mind, incorporating circuit breakers, conservative operating modes, and comprehensive validation. Changes are "measured twice, cut once."
- **Asynchronous, Event-Driven Communication**: Subsystems are loosely coupled and communicate primarily through an asynchronous message bus, promoting scalability and resilience.
- **Centralized Asset Management**: All architectural assets, from design blueprints to compiled kernels, are stored in a single, versioned, and auditable repository.

## System Overview

The Esper platform is composed of 14 subsystems, organized into 5 functional planes and one virtual subsystem for shared contracts.

### The 5 Functional Planes

1.  **Execution Plane**: Manages the core training loop and executes the neural network computations.
2.  **Control Plane**: Performs real-time analysis and decision-making to guide the training and evolution process.
3.  **Policy Training Plane**: Asynchronously improves the decision-making models used by the Control Plane.
4.  **Innovation Plane**: Asynchronously designs, compiles, and validates new neural network architectures.
5.  **Infrastructure Plane**: Provides the foundational services for communication, observability, and configuration.

---

## Subsystem Catalog

Below is a brief overview of each subsystem. For a more detailed reference, see [SUBSYSTEM-REFERENCE.md](SUBSYSTEM-REFERENCE.md).

### 00. LEYLINE - Shared Contract Governance (Virtual)
- **Role**: Defines the shared data structures, schemas, and interfaces used for communication between all subsystems. It is not a running service but a collection of contracts.

### Execution Plane
- **01. TOLARIA - Training Orchestrator**: The master controller for the training loop, managing epochs, steps, and overall system stability.
- **02. KASMINA - Execution Layer**: The hands of the system, responsible for executing compiled kernels on the hardware and managing the lifecycle of network "seeds".

### Control Plane
- **03. TAMIYO - Strategic Controller**: The "brain" of the system, using a learned policy to make strategic decisions about how and when to evolve the network.
- **12. EMRAKUL - Architectural Sculptor**: The command-and-control coordinator for large-scale architectural modifications, such as structured pruning.
- **13. ELESH - Structural Analyzer**: Performs the detailed offline analysis of the network's structure to find opportunities for optimization, like identifying redundant channels or attention heads.

### Policy Training Plane

*A note on its relationship with other planes: While Simic and Jace formally constitute this plane, its primary purpose is to serve the Control and Innovation planes. In that sense, **Tamiyo** (Control) and **Karn** (Innovation) can be considered 'honorary members,' as they are the ultimate consumers of the advanced policies developed here.*
- **04. SIMIC - Policy Trainer**: The offline reinforcement learning environment that trains and improves the neural network policies for both **Tamiyo** (the strategic controller) and, in Phase 2, **Karn** (the generative architect).
- **11. JACE - Curriculum Coordinator**: Aligns the learning curriculum between the strategic decisions of Tamiyo and the policy training of Simic to ensure coherent learning.

### Innovation Plane
- **05. KARN - Blueprint Generator**: The creative engine that generates new neural network architectural designs (blueprints).
- **06. TEZZERET - Compilation Forge**: Compiles the abstract blueprints from Karn into highly-optimized, executable kernels.
- **07. URABRASK - Evaluation Engine**: The gatekeeper that rigorously validates every new kernel for safety, performance, and correctness before it can be used.
- **08. URZA - Central Library**: The single source of truth, an immutable repository for all architectural assets, including blueprints and validated kernels.

### Infrastructure Plane
- **09. OONA - Message Bus**: The high-performance, asynchronous communication backbone that allows all subsystems to interact through events.
- **10. NISSA - Observability Platform**: The eyes and ears of the system, providing comprehensive monitoring, metrics, logging, and tracing.
- **14. MYCOSYNTH - Configuration Fabric**: A proposed centralized service to manage the runtime configuration for all subsystems dynamically and safely.

---

## Getting Started

To get a deeper understanding of the architecture, it is recommended to:

1.  Read the [SUBSYSTEM-REFERENCE.md](SUBSYSTEM-REFERENCE.md) for a more detailed summary of each component.
2.  Review the unified design documents for the primary subsystem in each plane to understand the core workflows:
    -   **Execution**: [01-tolaria-unified-design.md](01-tolaria-unified-design.md)
    -   **Control**: [03-tamiyo-unified-design.md](03-tamiyo-unified-design.md)
    -   **Innovation**: [05-karn-unified-design.md](05-karn-unified-design.md)
    -   **Infrastructure**: [09-oona-unified-design.md](09-oona-unified-design.md)
3.  Examine [00-leyline-shared-contracts.md](/docs/architecture/00-leyline-shared-contracts.md) (if available) to understand the data that flows between the subsystems.
