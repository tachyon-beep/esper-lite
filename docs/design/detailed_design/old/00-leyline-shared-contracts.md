# Leyline Shared Contracts Unified Design Document v1.0

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Status** | VIRTUAL |
| **Date** | 2025-01-15 |
| **Author** | Esper Team |
| **Component** | Virtual - Shared Contract Governance |
| **Parent** | High-Level Design (HLD) |
| **Subdocuments** | 00.1-leyline-message-contracts.md, 00.2-leyline-enums-constants.md, 00.3-leyline-governance-implementation.md |

## Executive Summary

Leyline provides the single source of truth for all shared cross-subsystem contracts in the Esper morphogenetic neural network training platform. It is a virtual subsystem - not a running service but a contract library that establishes foundational message schemas, enums, constants, and interface contracts before any subsystem implementation can succeed. Like the MTG Leylines that establish foundational forces before gameplay begins, this virtual subsystem prevents serialization failures and ensures seamless integration across all 14 subsystems.

Key characteristics:
- **Virtual Nature**: Contract library, not a running service
- **Single Source of Truth**: Prevents schema drift and contract divergence
- **Performance Optimized**: Messages designed for <80μs serialization

## Core Architecture Decision

### **Virtual Contract Governance Model**

- **Foundation**: Protocol Buffer definitions as the canonical source
- **Integration Model**: All subsystems import from Leyline namespace
- **Authority Model**: Data Architect owns contract evolution and consistency
- **Deployment Model**: Compiled into each subsystem at build time

## Architectural Principles

### Non-Negotiable Requirements

1. **Contract Consistency**: All subsystems must use identical message schemas
2. **Performance First**: Messages optimized for minimal serialization overhead
3. **Virtual Subsystem**: No runtime components, only compile-time contracts

### Design Principles

1. **Simplicity**: Greenfield optimization without legacy compatibility
2. **Type Safety**: Strong typing prevents runtime serialization failures
3. **Centralized Governance**: Single owner prevents uncontrolled schema drift

### Production Safety Principles

1. **CI/CD Enforcement**: Automated validation prevents local contract modifications
2. **Version Management**: Simple version strategy for greenfield design
3. **Breaking Changes**: Allowed during pre-production development only

## System Components

### Component Overview

| Component | Purpose | Details |
|-----------|---------|---------|
| **Message Contracts** | Core cross-subsystem messages | See: 00.1-leyline-message-contracts.md |
| **Enums & Constants** | Shared enumerations and system constants | See: 00.2-leyline-enums-constants.md |
| **Governance Implementation** | Change control and version management | See: 00.3-leyline-governance-implementation.md |

### Core Components Summary

**Message Contracts**
- SystemStatePacket for training state
- AdaptationCommand for control operations
- Structured pruning messages (C-020)
- Details: 00.1-leyline-message-contracts.md#core-messages

**Shared Enumerations**
- Lifecycle stages and health states
- Command types and operations
- Priority levels and delivery guarantees
- Details: 00.2-leyline-enums-constants.md#enumerations

**System Constants**
- Performance budgets and timing constraints
- Memory allocations and system limits
- Circuit breaker thresholds
- Details: 00.2-leyline-enums-constants.md#constants

## Integration Architecture

### Subsystem Dependencies

| Subsystem | Integration Type | Purpose |
|-----------|-----------------|---------|
| All Subsystems | Compile-time Import | Shared message schemas and constants |
| Tolaria (01) | Direct Import | SystemStatePacket, AdaptationCommand |
| Kasmina (02) | Direct Import | SystemStatePacket, checkpoint metadata |
| Tamiyo (03) | Direct Import | Strategic control messages |
| Simic (04) | Direct Import | Policy training telemetry |
| Karn (05) | Direct Import | Blueprint generation contracts |
| Tezzeret (06) | Direct Import | Compilation status messages |
| Urabrask (07) | Direct Import | Validation results and safety reports |
| Urza (08) | Direct Import | Library metadata contracts |
| Oona (09) | Direct Import | EventEnvelope for message bus |
| Nissa (10) | Direct Import | TelemetryPacket for observability |
| Jace (11) | Direct Import | Curriculum coordination messages |
| Emrakul (12) | Direct Import | Architectural analysis requests |
| Elesh (13) | Direct Import | Structural pruning messages (C-020) |

### Message Contracts

| Contract | Direction | Purpose |
|----------|-----------|---------|
| SystemStatePacket | Tolaria → All | Training state broadcast |
| AdaptationCommand | Control → Execution | Architecture modifications |
| StructuralPruningRequest | Emrakul → Elesh | Pruning analysis request (C-020) |
| StructuralPruningResponse | Elesh → Emrakul | Analysis results (C-020) |
| EventEnvelope | Any → Oona → Any | Message bus wrapper |
| TelemetryPacket | Any → Nissa | Observability data |

### Shared Contracts (Leyline)

This subsystem defines the following shared contracts:
- `leyline.SystemStatePacket` - Core training state representation
- `leyline.AdaptationCommand` - Control plane commands
- `leyline.EventEnvelope` - Message bus envelope
- `leyline.TelemetryPacket` - Observability telemetry
- `leyline.StructuralPruning*` - Structured pruning contracts (C-020)

For complete contract definitions, see: `/docs/architecture/00.1-leyline-message-contracts.md`

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| SystemStatePacket serialization | <80μs | Protobuf benchmark |
| AdaptationCommand serialization | <40μs | Protobuf benchmark |
| Message size (SystemStatePacket) | <280 bytes | Serialized size |
| GC allocations per message | ≤4 | Memory profiler |
| Contract compilation time | <2s | Build system metrics |

## Configuration

### Key Configuration Parameters

```yaml
leyline:
  # Version management
  schema_version: 1  # Current schema version

  # Performance targets
  max_message_size_bytes: 280  # Maximum serialized message size
  max_serialization_us: 80     # Maximum serialization time

  # Governance
  owner: "data-architect"       # Contract owner role
  change_approval_required: 3   # Required approvals for changes
```

## Operational Considerations

### Health Monitoring

- **Health Check Endpoint**: N/A (Virtual subsystem)
- **Key Metrics**: Contract version consistency across subsystems
- **SLO Targets**: 100% contract compatibility

### Failure Modes

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Version Mismatch | Build-time validation | Build failure, prevent deployment |
| Schema Drift | CI/CD validation | Block merge, require resolution |
| Serialization Failure | Runtime exception | Circuit breaker activation |

### Scaling Considerations

- **Horizontal Scaling**: N/A (Compile-time library)
- **Vertical Scaling**: N/A (No runtime components)
- **Resource Requirements**: Minimal - only build-time compilation

## Security Considerations

- **Authentication**: N/A (Compile-time library)
- **Authorization**: Governance model controls changes
- **Data Protection**: Messages may contain sensitive training data
- **Audit**: All contract changes tracked in version control

## Migration Notes

> **Migration Status**: Greenfield design - no migration required
> - All subsystems import from Leyline namespace
> - No legacy contracts to migrate

## Future Enhancements

### Phase 2: Schema Evolution Support
- **Description**: Add backward compatibility when production requires it
- **Trigger**: First production deployment with live systems
- **Impact**: Enable rolling updates without coordination

### Phase 3: Dynamic Contract Discovery
- **Description**: Runtime contract negotiation for heterogeneous deployments
- **Trigger**: Multi-version production environments
- **Impact**: Support for gradual rollouts and A/B testing

## Cross-References

### Subdocuments
- [[00.1-leyline-message-contracts.md]]: Complete message definitions
- [[00.2-leyline-enums-constants.md]]: Shared enumerations and constants
- [[00.3-leyline-governance-implementation.md]]: Governance and implementation details

### Related Documents
- [[01-tolaria-unified-design.md]]: Primary consumer of SystemStatePacket
- [[02-kasmina-unified-design.md]]: Execution layer integration
- [[13-elesh-unified-design.md]]: Structural pruning analyzer (C-020)
- [[00-leyline-shared-contracts.md]]: Shared contract definitions
- [ADR-018]: Greenfield optimization decision

## Implementation Status

### Current State
- [x] Component 1: Message contract definitions
- [x] Component 2: Enum and constant definitions
- [x] Component 3: C-020 structured pruning integration
- [ ] Component 4: CI/CD validation rules
- [ ] Component 5: Protobuf compilation setup

### Validation Status
- [ ] Unit tests complete
- [ ] Integration tests complete
- [ ] Performance validation
- [ ] Security review
- [ ] Production readiness review

## History & Context

### Version History
- **v1.0** (2025-01-15): Initial virtual subsystem design from migration

### Integration History
- **C-020 Structured Pruning** (2025-01-14): Added comprehensive pruning contracts
- **Greenfield Optimization** (2025-01-14): Removed legacy compatibility layers

### Critical Fixes Applied
- **PERF-001**: Optimized message size from 2KB to 280 bytes
- **CONS-001**: Established single source of truth governance model

---

*Last Updated: 2025-01-15 | Next Review: 2025-02-01 | Owner: Data Architect*