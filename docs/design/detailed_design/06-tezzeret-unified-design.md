# Tezzeret - The Compilation Forge
## Unified Design Document v4.0 (C-016 CRITICAL FIXES INTEGRATED)

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 4.0.0 |
| **Status** | PRODUCTION |
| **Date** | 2025-09-10 |
| **Author** | System Architecture Team |
| **Component** | Innovation Plane - Compilation Engine |
| **Parent** | High-Level Design (HLD) |
| **Subdocuments** | [06.1-tezzeret-compilation-internals.md](06.1-tezzeret-compilation-internals.md) |

## Executive Summary

Tezzeret serves as the dedicated, asynchronous background service that transforms architectural designs (`BlueprintIR`) into portfolios of highly-optimized, executable `CompiledKernelArtifacts`. Operating independently from the training loop with comprehensive security controls and **[C-016 CRITICAL]** WAL crash recovery guarantees, Tezzeret ensures zero training disruption while continuously optimizing the kernel library through intelligent work pipelining and value-driven recompilation strategies.

The subsystem's primary innovation is its multi-tier optimization strategy combined with complete crash recovery and circuit breaker protection. By decoupling compilation from training, Tezzeret enables continuous architectural evolution without impacting training performance, while conservative mode ensures graceful degradation under stress.

Key characteristics:
- **Asynchronous Compilation Pipeline**: Complete decoupling from training loop prevents any compilation-related pauses
- **Multi-Tier Optimization Strategy**: Three compilation levels (Fast/Standard/Aggressive) with adaptive selection plus Emergency path
- **[C-016 CRITICAL] WAL Crash Recovery**: O_DSYNC + fsync barriers with 256-byte headers and CRC validation
- **[C-016 CRITICAL] Circuit Breaker Architecture**: All assert statements replaced with circuit breakers and conservative mode

## Core Architecture Decision

### **Secure Asynchronous Blueprint Compilation and Optimization Engine with WAL Crash Recovery**

- **Foundation**: Asynchronous compilation pipeline built on PyTorch's torch.compile with sandboxed execution
- **Integration Model**: Event-driven subscription to blueprints, async delivery of compiled kernels
- **Authority Model**: Sole authority for blueprint compilation and kernel optimization in the system
- **Deployment Model**: Single instance per training cluster with Redis emergency queue for high-priority requests

## Architectural Principles

### Non-Negotiable Requirements

1. **Zero Training Disruption**: Compilation must never block, pause, or slow the training loop
2. **WAL Crash Recovery**: System must recover from any crash point with complete data integrity
3. **Circuit Breaker Protection**: System must gracefully degrade rather than crash on failures
4. **Memory Bounded Growth**: All caches and queues must have TTL-based cleanup to prevent leaks
5. **Security Sandboxing**: All compilation must execute in isolated containers with resource limits

### Design Principles

1. **Asynchronous-First**: All compilation happens in background threads/processes
2. **Multi-Tier Optimization**: Different compilation strategies for different value/urgency combinations
3. **Intelligence Idle-State Processing**: Productive recompilation during low blueprint submission periods
4. **Resource-Aware Execution**: Dynamic pipeline selection based on system load and blueprint value
5. **Protocol Standardization**: All messages use Protocol Buffers v2 with no map<> fields

### Production Safety Principles

1. **WAL Durability Semantics**: O_DSYNC + fsync barriers ensure all state survives crashes
2. **Circuit Breaker Degradation**: Replace assertions with circuit breakers that trigger conservative mode
3. **Conservative Mode Operation**: Automatic resource reduction when system under stress
4. **TTL Memory Management**: Prevent unbounded growth through periodic cleanup
5. **Chaos Engineering Validation**: Test all failure modes including NVMe namespace teardown

## System Components

### Component Overview

| Component | Purpose | Details |
|-----------|---------|---------|
| **PollingEngine** | Polls Urza for new blueprints to compile | See: [06.1#polling-engine](06.1-tezzeret-compilation-internals.md#polling-engine) |
| **CompilationCore** | Core compilation engine with torch.compile integration | See: [06.1#compilation-core](06.1-tezzeret-compilation-internals.md#compilation-core) |
| **PipelineManager** | Manages Fast/Standard/Aggressive/Emergency pipelines | See: [06.1#pipeline-management](06.1-tezzeret-compilation-internals.md#pipeline-management) |
| **WALCheckpointManager** | Write-ahead log for crash recovery | See: [06.1#wal-architecture](06.1-tezzeret-compilation-internals.md#wal-architecture) |
| **CircuitBreakerSystem** | Replaces assertions with graceful degradation | See: [06.1#circuit-breakers](06.1-tezzeret-compilation-internals.md#circuit-breakers) |
| **ConservativeModeController** | Manages reduced-resource operation mode | See: [06.1#conservative-mode](06.1-tezzeret-compilation-internals.md#conservative-mode) |
| **TTLMemoryManager** | Prevents memory leaks via TTL cleanup | See: [06.1#memory-management](06.1-tezzeret-compilation-internals.md#memory-management) |
| **SecurityValidator** | Pre-compilation security validation | See: [06.1#security-validation](06.1-tezzeret-compilation-internals.md#security-validation) |
| **SandboxExecutor** | Container-based compilation isolation | See: [06.1#sandbox-execution](06.1-tezzeret-compilation-internals.md#sandbox-execution) |

### Core Components Summary

**CompilationCore**
- torch.compile integration with multiple optimization levels
- Dynamic strategy selection based on blueprint complexity
- GPU kernel optimization and caching
- Details: [06.1#compilation-core](06.1-tezzeret-compilation-internals.md#compilation-core)

**WALCheckpointManager**
- O_DSYNC + fsync barriers with 256-byte headers and CRC validation
- Merkle root calculation including execution context
- 100% recovery from all tested crash scenarios
- Details: [06.1#wal-architecture](06.1-tezzeret-compilation-internals.md#wal-architecture)

**CircuitBreakerSystem**
- Three-state circuit breakers (CLOSED/OPEN/HALF_OPEN)
- Automatic conservative mode triggers
- Violation metrics for monitoring
- Details: [06.1#circuit-breakers](06.1-tezzeret-compilation-internals.md#circuit-breakers)

## Integration Architecture

### Subsystem Dependencies

| Subsystem | Integration Type | Purpose |
|-----------|-----------------|---------|
| Urza | Async (Leyline) | Blueprint polling and artifact storage |
| Urabrask | Async (Leyline) | Validation notification with compilation metadata |
| Karn | Async (Leyline) | Blueprint subscription with priority handling |
| Tamiyo | Async (Redis) | Emergency path for direct commands <500ms |
| Oona | Async (Leyline) | Message bus for event distribution |

### Message Contracts

| Contract | Direction | Purpose |
|----------|-----------|---------|
| BlueprintIR | Karn → Tezzeret | Blueprint submission for compilation |
| CompilationRequest | Oona → Tezzeret | Compilation request with strategy |
| CompiledKernelArtifact | Tezzeret → Urza | Compiled kernel storage |
| CompilationComplete | Tezzeret → Urabrask | Validation notification |
| EmergencyCompile | Tamiyo → Tezzeret | High-priority compilation request |

### Shared Contracts (Leyline)

This subsystem uses Protocol Buffers v2 for all message contracts:
- `leyline.BlueprintIR` - Blueprint representation (no map<> fields)
- `leyline.CompiledKernelArtifact` - Compiled kernel with metadata
- `leyline.CompilationRequest` - Request with strategy and priority
- `leyline.CompilationResult` - Result with success/failure status

For complete contract definitions and implementation details, see: [06.1#protocol-buffers](06.1-tezzeret-compilation-internals.md#protocol-buffers)

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Compilation Speed (Emergency)** | 4-10s | `tezzeret_compilation_duration_ms` histogram |
| **Compilation Speed (Fast)** | 20-65s | `tezzeret_compilation_duration_ms` histogram |
| **Compilation Speed (Standard)** | 75-250s | `tezzeret_compilation_duration_ms` histogram |
| **Compilation Speed (Aggressive)** | 250-960s | `tezzeret_compilation_duration_ms` histogram |
| **Conservative Mode Trigger** | GPU >20% for 5+ min | `tezzeret_conservative_mode_active` gauge |
| **Resource Utilization** | 25% max GPU | `tezzeret_gpu_utilization_percent` gauge |
| **Concurrent Jobs** | Max 2 | `tezzeret_concurrent_compilations` gauge |
| **Memory Budget** | 16GB total | `tezzeret_memory_usage_gb` gauge |
| **Cache Hit Rates** | L1: 5-10%, L2: 20-30%, L3: 30-40% | `tezzeret_cache_hit_rate` gauge |
| **WAL Recovery Time** | <12s | `tezzeret_wal_recovery_duration_ms` histogram |

## Configuration

### Key Configuration Parameters

```yaml
tezzeret:
  # Core compilation settings
  compilation:
    max_concurrent_jobs: 2
    default_strategy: STANDARD
    enable_emergency_compilation: true
    enable_caching: true
    gpu_utilization_limit: 0.25

  # Circuit breaker configuration
  circuit_breakers:
    enabled: true
    failure_threshold: 3
    timeout_ms: 30000

  # Conservative mode settings
  conservative_mode:
    auto_enable: true
    gpu_threshold_percent: 20
    threshold_duration_minutes: 5

  # WAL configuration
  wal:
    enabled: true
    path: "/var/lib/esper/tezzeret/wal"
    durability_mode: "O_DSYNC"

  # Memory management
  memory_management:
    ttl_cleanup_enabled: true
    cleanup_interval_seconds: 300
```

For complete configuration details, see: [06.1#configuration](06.1-tezzeret-compilation-internals.md#configuration)

## Operational Considerations

### Health Monitoring

- **Health Check Endpoint**: `/health/tezzeret`
- **Key Metrics**:
  - `tezzeret_compilation_duration_ms` - Compilation time by strategy
  - `tezzeret_circuit_breaker_state` - Circuit breaker states
  - `tezzeret_conservative_mode_active` - Conservative mode status
  - `tezzeret_wal_transactions_total` - WAL transaction counts
- **SLO Targets**:
  - P95 compilation time < strategy timeout
  - Circuit breaker open rate < 1%
  - WAL recovery success rate = 100%

### Failure Modes

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Compilation timeout | Time exceeds strategy limit | Trigger conservative mode |
| GPU memory exhaustion | Memory usage > 80% | Reduce concurrent jobs to 1 |
| Circuit breaker open | Failure threshold exceeded | Enter conservative mode |
| WAL corruption | CRC validation failure | Restore from last checkpoint |
| torch.compile crash | Process segfault | Restart with CPU-only mode |

### Scaling Considerations

- **Horizontal Scaling**: Not supported (single instance design)
- **Vertical Scaling**: Increase GPU memory and cores for better performance
- **Resource Requirements**:
  - GPU: 16GB VRAM minimum
  - CPU: 8 cores recommended
  - Memory: 32GB system RAM
  - Storage: 100GB for WAL and caches

## Security Considerations

- **Authentication**: Internal service, no external authentication
- **Authorization**: Role-based access for emergency compilation requests
- **Data Protection**: Compiled kernels signed with SHA256
- **Audit**: All compilation requests logged with requestor and outcome
- **Sandboxing**: All compilation in isolated containers with resource limits

## Migration Notes

> **Migration Status**: COMPLETE
> - C-016 Critical Fixes: Fully integrated
> - Protocol Buffers v2: Migration complete
> - Circuit breakers: All assertions replaced
> - Conservative mode: Fully operational

## Future Enhancements

### Phase 2: Distributed Compilation
- **Description**: Support multiple compilation nodes for horizontal scaling
- **Trigger**: When single-node compilation becomes bottleneck
- **Impact**: 10x compilation throughput increase

### Phase 3: ML-Driven Strategy Selection
- **Description**: Use learned model to predict optimal compilation strategy
- **Trigger**: After collecting 10,000+ compilation samples
- **Impact**: 30% reduction in average compilation time

## Cross-References

### Subdocuments
- [06.1-tezzeret-compilation-internals.md](06.1-tezzeret-compilation-internals.md): Complete implementation details

### Related Documents
- [08-urza-unified-design.md](08-urza-unified-design.md): Blueprint storage integration
- [07-urabrask-unified-design.md](07-urabrask-unified-design.md): Validation integration
- [00-leyline-shared-contracts.md](00-leyline-shared-contracts.md): Shared contract definitions
- [ADR-023-Circuit-Breakers.md]: Circuit breaker architecture decision

## Implementation Status

### Current State
- [x] WAL with Durability Semantics - O_DSYNC + fsync barriers implemented
- [x] Circuit Breaker Architecture - All assertions replaced
- [x] Memory Management - TTL-based cleanup operational
- [x] Protocol Buffers v2 - No map<> fields, validation complete
- [x] Conservative Mode - Automatic triggers configured
- [x] Core Engine - Multi-tier compilation working
- [x] All Compilation Pipelines - Fast/Standard/Aggressive/Emergency
- [x] All Integrations - Urza/Urabrask/Karn/Tamiyo connected
- [x] Security Controls - Sandboxing and validation complete
- [x] Monitoring - All metrics instrumented

### Validation Status
- [x] Unit tests complete (>90% coverage)
- [x] Integration tests complete
- [x] Performance validation (meets all targets)
- [x] Security review (passed)
- [x] Production readiness review (approved)
- [x] Chaos engineering tests (all scenarios pass)

## History & Context

### Version History
- **v1.0** (2024-06-01): Initial design with basic compilation
- **v2.0** (2024-09-01): Added multi-tier optimization
- **v3.0** (2024-12-01): C-012 consensus integration
- **v4.0** (2025-09-10): C-016 critical fixes integrated

### Integration History
- **C-012 Integration** (2024-12-01): Hardware-validated performance targets
- **C-016 Integration** (2025-09-10): Critical production safety fixes

### Critical Fixes Applied
- **C-016-001**: WAL durability with O_DSYNC + fsync
- **C-016-002**: Circuit breakers replacing assertions
- **C-016-003**: TTL memory management
- **C-016-004**: Protocol Buffers v2 migration
- **C-016-005**: Conservative mode automation
- **C-016-006**: Chaos engineering validation

---

*Last Updated: 2025-09-10 | Next Review: 2026-03-10 | Owner: System Architecture Team*