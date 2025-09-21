# Mycosynth - The Configuration Fabric
## Unified Design Document v1.0

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Status** | PROPOSED |
| **Date** | 2025-09-14 |
| **Author** | System Architecture Team |
| **Component** | Infrastructure - Configuration Management |
| **Parent** | High-Level Design (HLD) |
| **Subdocuments** | N/A |

## Executive Summary

This document proposes the creation of Mycosynth, a new subsystem to serve as the centralized, dynamic configuration fabric for the entire Esper platform. As the project's complexity has grown, managing critical, interdependent operational parameters (e.g., timeouts, resource budgets, feature flags) across dozens of separate subsystem configurations has become a significant operational risk. Mycosynth will mitigate this risk by providing a single, versioned, and auditable source of truth for all runtime configurations.

Like the Mycosynth Lattice in MTG, which imposes a new set of rules on everything in play, this subsystem will define the operational reality for all other components, ensuring they behave consistently and predictably in any given environment.

Key characteristics:
- **Centralized Management**: A single API and UI for managing all subsystem configurations.
- **Dynamic Updates**: Push configuration changes to services in real-time without requiring redeployment.
- **Validation & Schemas**: Enforce schemas to prevent syntactically or logically invalid configurations.
- **High Availability**: Designed as a fault-tolerant, highly-available service to prevent it from becoming a single point of failure.

## Core Architecture Decision

### **Highly-Available, Centralized, Real-time Configuration Service**

- **Foundation**: A distributed key-value store (like etcd or Consul) fronted by a gRPC service for high performance and strong consistency.
- **Integration Model**: Subsystems will fetch their configuration at startup and subscribe to real-time updates for dynamic parameters. A local cache of the last-known-good configuration will ensure resilience.
- **Authority Model**: Mycosynth is the sole authority for operational configuration. Local config files will be deprecated except for the bootstrap configuration needed to connect to Mycosynth.
- **Deployment Model**: Deployed as a foundational infrastructure service with the highest level of redundancy and monitoring.

## Architectural Principles

### Non-Negotiable Requirements

1.  **High Availability**: The service must be more available than any of the services that depend on it. A failure in Mycosynth must not prevent other services from starting or running.
2.  **Consistency**: All subsystems must receive a consistent view of the configuration.
3.  **Low Latency**: Configuration reads must be extremely fast, serviced from an in-memory cache on the client side.
4.  **Auditability**: Every change to any configuration must be logged, attributed to a user or service, and versioned.
5.  **Security**: Access to modify configurations must be strictly controlled on a per-subsystem, per-environment basis.

### Design Principles

1.  **Fail-Safe by Default**: Clients (the other subsystems) MUST cache their last-known-good configuration locally and be able to start and run indefinitely if Mycosynth is unavailable.
2.  **Schema-Driven**: All configurations will be defined by a schema to enable validation and tooling.
3.  **GitOps-Friendly**: The underlying configuration can be backed by a Git repository, allowing for pull-request-based workflows for configuration changes.

### Production Safety Principles

1.  **Validation First**: No configuration change is committed without passing schema and logical validation.
2.  **Circuit Breakers**: The Mycosynth client in each subsystem will have a circuit breaker to prevent it from overwhelming the service during an outage.
3.  **Easy Rollbacks**: Every configuration change creates a new version, and rolling back to a previous version must be a trivial, one-click operation.

## System Components

### Component Overview

| Component | Purpose | Details |
|-----------|---------|---------|
| **Config Store** | The durable backend for storing configuration data. | Likely etcd, Consul, or a database like FoundationDB. |
| **Config Service** | The gRPC API that subsystems interact with. | Handles authentication, authorization, validation, and serving. |
| **Mycosynth Client** | A library integrated into every other subsystem. | Handles fetching, caching, and real-time updates. |
| **Admin UI** | A web interface for operators to view and manage configs. | Provides a user-friendly way to interact with the system. |

### Core Components Summary

**Config Service**
- Exposes a gRPC API for `GetConfiguration` and `SubscribeToChanges`.
- Enforces validation against schemas before persisting any change.
- Manages different configuration sets for each environment (`dev`, `prod`).

**Mycosynth Client**
- Fetches configuration on service startup.
- Caches the configuration to a local file (last-known-good).
- Subscribes to the Config Service for real-time updates for dynamic keys.
- Provides a simple, typed interface for the service to access config values.

## Integration Architecture

### Subsystem Dependencies

Mycosynth will be a foundational service with no dependencies on other Esper subsystems, but nearly all other subsystems will have a startup dependency on it.

### Distinguishing from Leyline

It is critical to distinguish Mycosynth's role from Leyline's:

| Concern | Owner | Description |
|---|---|---|
| **Data Contracts** | **Leyline** | Defines the *structure* of data (Protobuf schemas). Changes with code. |
| **Operational Config** | **Mycosynth** | Defines the *behavior* of services (timeouts, limits, flags). Changes at runtime. |

### Example Workflow

1.  An operator uses the Admin UI to change `Tezzeret.timeouts_ms.standard_compilation` from `250000` to `260000`.
2.  The Admin UI calls the Config Service API.
3.  The Config Service validates the change (ensuring it's an integer) and persists it, creating a new version.
4.  The Config Service publishes the change notification.
5.  The Mycosynth Client within the Tezzeret service receives the update and hot-reloads the new timeout value, without a restart.

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Client-side Read Latency | < 1ms | From local cache |
| Startup Fetch Latency | < 100ms | Time to get initial config |
| Update Propagation Time | < 1s | Time from change commit to client reload |
| Mycosynth Availability | 99.99% | Uptime of the Config Service |

## Conclusion

The introduction of Mycosynth addresses a critical and growing architectural risk. By centralizing configuration management, we enhance system reliability, improve operational agility, and provide a consistent and auditable framework for managing the behavior of the entire Esper platform. While it introduces a new critical component, the safety and operational benefits far outweigh the complexity cost for a system of this scale.

---

*Last Updated: 2025-09-14 | Next Review: 2025-09-21 | Owner: System Architecture Team*
