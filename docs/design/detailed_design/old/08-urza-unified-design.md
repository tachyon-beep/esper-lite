# Urza - The Central Library
## Unified Design Document v3.0 - C-016 CRITICAL FIXES INTEGRATED

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 3.0.0 |
| **Status** | PRODUCTION READY |
| **Date** | 2025-01-10 |
| **Author** | System Architecture Team |
| **Component** | Innovation Plane - Central Library |
| **Parent** | High-Level Design (HLD) |
| **Subdocuments** | [08.1-urza-internals.md](08.1-urza-internals.md) |

## Executive Summary

Urza serves as the immutable, versioned, and auditable single source of truth for all architectural assets in the Esper morphogenetic platform. It stores both abstract designs (BlueprintIRs) and their compiled, validated forms (CompiledKernelArtifacts), providing rich query capabilities for strategic kernel selection while maintaining complete lineage tracking and comprehensive metadata management.

Key characteristics:
- **Universal Metadata Schema**: Comprehensive metadata model for all assets with 100+ queryable attributes
- **Lifecycle State Management**: Track assets from generation through validation to deprecation
- **Rich Query API**: Tag-based, performance-aware kernel discovery with circuit breakers and <500ms guarantees

## Core Architecture Decision

### **Immutable Versioned Repository Pattern**

- **Foundation**: Content-addressable storage with SHA-256 integrity and Merkle tree verification
- **Integration Model**: Event-driven asset ingestion with Protocol Buffers v2 contracts
- **Authority Model**: Sole authority for storing, versioning, and serving all architectural assets
- **Deployment Model**: Distributed storage with multi-tier caching and read replicas

## Architectural Principles

### Non-Negotiable Requirements

1. **Immutability**: Once stored, assets cannot be modified - only new versions can be added
2. **Data Integrity**: SHA-256 + CRC32 checksums with Merkle tree verification for all assets
3. **Sub-10ms Retrieval**: Cached kernels must be retrievable in <10ms p50, <500ms p99
4. **Complete Auditability**: Full lineage tracking and access logging for all operations
5. **Zero Data Loss**: WAL durability with O_DSYNC + fsync barriers

### Design Principles

1. **Single Source of Truth**: All architectural assets stored in one authoritative location
2. **Rich Metadata First**: Comprehensive metadata enables intelligent query and selection
3. **Performance Through Caching**: Multi-tier cache architecture for production performance
4. **Graceful Degradation**: Conservative mode and circuit breakers prevent cascade failures
5. **Version-Aware**: Semantic versioning with dependency resolution and compatibility checking

### Production Safety Principles

1. **Circuit Breaker Protection**: All operations protected with 400ms timeout circuit breakers
2. **Conservative Mode**: Fallback to cache-only serving under load or failures
3. **Identity Kernel Fallback**: Always have safe fallback kernel available
4. **Memory Leak Prevention**: Periodic garbage collection with TTL-based cleanup
5. **Query Timeout Enforcement**: Hard 500ms timeout with fallback to identity kernel

## System Components

### Component Overview

| Component | Purpose | Details |
|-----------|---------|---------|
| **Object Store** | S3-based immutable binary storage with compression | See: [08.1#object-store](08.1-urza-internals.md#object-store) |
| **Metadata DB** | PostgreSQL for rich queryable metadata | See: [08.1#metadata-database](08.1-urza-internals.md#metadata-database) |
| **Cache Layer** | Multi-tier caching (Memory/Redis/DB/S3) | See: [08.1#cache-architecture](08.1-urza-internals.md#cache-architecture) |
| **Query Engine** | Advanced query with circuit breakers | See: [08.1#query-engine](08.1-urza-internals.md#query-engine) |
| **Version Manager** | Semantic versioning and dependency resolution | See: [08.1#versioning-system](08.1-urza-internals.md#versioning-system) |
| **Lifecycle Manager** | Asset state transitions and retention | See: [08.1#lifecycle-management](08.1-urza-internals.md#lifecycle-management) |

### Core Components Summary

**Content-Addressable Storage**
- SHA-256 based addressing for immutability
- Zstd compression for 40% space savings
- 100MB kernel size limit enforced
- Details: [08.1#storage-architecture](08.1-urza-internals.md#storage-architecture)

**Advanced Query Engine**
- Circuit breaker with 400ms timeout
- Conservative mode for graceful degradation
- Tag-based and performance-aware queries
- Details: [08.1#query-engine](08.1-urza-internals.md#query-engine)

**Multi-Tier Cache**
- L1: Process memory (sub-µs)
- L2: Redis 32GB cluster (sub-ms)
- L3: PostgreSQL replicas (ms)
- L4: S3 with CDN (10s ms)
- Details: [08.1#cache-architecture](08.1-urza-internals.md#cache-architecture)

## Integration Architecture

### Subsystem Dependencies

| Subsystem | Integration Type | Purpose |
|-----------|-----------------|---------|
| Karn | Async (Leyline) | Receive generated blueprints for storage |
| Tezzeret | Async (Leyline) | Receive compiled kernel artifacts |
| Urabrask | Async (Leyline) | Process validation reports and update metadata |
| Tamiyo | Async (Leyline) | Serve kernel queries for strategic decisions |
| Emrakul | Async (Leyline) | Provide architectural lineage data |
| Urza | Internal | Self-referential for version management |

### Message Contracts

| Contract | Direction | Purpose |
|----------|-----------|---------|
| BlueprintProto | Karn → Urza | Blueprint ingestion |
| KernelProto | Tezzeret → Urza | Kernel artifact storage |
| ValidationReport | Urabrask → Urza | Update kernel validation status |
| KernelQuery | Tamiyo → Urza | Strategic kernel queries |
| EventEnvelope | Oona ↔ Urza | Event bus integration |

### Shared Contracts (Leyline)

This subsystem uses Protocol Buffers v2 for all contracts:
- `leyline.BlueprintMetadata` - Blueprint metadata schema
- `leyline.KernelMetadata` - Kernel metadata with versioning
- `leyline.KernelQuery` - Query request/response format
- `leyline.StateTransition` - Lifecycle state changes

For complete contract definitions, see: [/docs/architecture/00-leyline-shared-contracts.md](./00-leyline-shared-contracts.md)

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query Latency (p50) | <10ms | `urza_query_duration_ms` histogram |
| Query Latency (p95) | <200ms | `urza_query_duration_ms` histogram |
| Query Latency (p99) | <500ms | `urza_query_duration_ms` histogram |
| Cache Hit Rate | >95% | `urza_cache_hits_total` / `urza_queries_total` |
| Query Throughput | 1000 qps | `urza_queries_total` counter rate |
| Storage Efficiency | 40% dedup | `urza_storage_dedup_ratio` gauge |
| Metadata Dimensions | 100 attrs | `urza_metadata_attributes` gauge |

## Configuration

### Key Configuration Parameters

```yaml
urza:
  # Storage configuration
  storage:
    object_store:
      type: s3
      bucket: urza-kernels
      compression: zstd
      max_kernel_size_mb: 100

    metadata_db:
      max_connections: 200
      pool_size: 50
      read_replicas: 3

    cache:
      redis_memory_gb: 32
      memory_cache_mb: 1024
      ttl_seconds: 3600

  # Query configuration
  query:
    timeout_ms: 500  # C-016 CRITICAL
    circuit_breaker:
      failure_threshold: 3
      timeout_ms: 400
    conservative_mode:
      enabled: true
      cache_only_serving: true
```

## Operational Considerations

### Health Monitoring

- **Health Check Endpoint**: `/health` with subsystem checks
- **Key Metrics**: `urza_query_duration_ms`, `urza_cache_hit_rate`, `urza_circuit_breaker_state`
- **SLO Targets**: 99.9% availability, <500ms p99 latency

### Failure Modes

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Query timeout | >500ms execution | Return identity kernel |
| Circuit breaker open | 3 failures in 60s | Conservative mode |
| Cache failure | Redis unreachable | Fallback to DB |
| DB overload | Connection pool exhausted | Read replica failover |
| Memory leak | >32GB usage | Trigger GC cycle |

### Scaling Considerations

- **Horizontal Scaling**: Add PostgreSQL read replicas and Redis shards
- **Vertical Scaling**: Increase cache memory up to 64GB
- **Resource Requirements**: 32GB RAM, 500GB SSD, 4 CPU cores minimum

## Security Considerations

- **Authentication**: Internal service, Oona message bus authentication
- **Authorization**: Read-only for most subsystems, write for Karn/Tezzeret/Urabrask
- **Data Protection**: AES-256 encryption at rest, TLS in transit
- **Audit**: Complete access logging with caller identification

## Migration Notes

> **Migration Status**: COMPLETE
> - All C-016 critical fixes integrated
> - Protocol Buffers v2 migration complete
> - Circuit breakers and conservative mode implemented

## Future Enhancements

### Phase 2: Distributed Consensus
- **Description**: Multi-master replication with Raft consensus
- **Trigger**: When single-master becomes bottleneck
- **Impact**: Improved write throughput and availability

### Phase 3: ML-Powered Indexing
- **Description**: Learned indexes for kernel similarity search
- **Trigger**: When metadata dimensions exceed 200
- **Impact**: Sub-linear query complexity for similarity queries

## Cross-References

### Subdocuments
- [08.1-urza-internals.md](08.1-urza-internals.md): Complete implementation details

### Related Documents
- [05-karn-unified-design.md](05-karn-unified-design.md): Blueprint generation
- [06-tezzeret-unified-design.md](06-tezzeret-unified-design.md): Kernel compilation
- [07-urabrask-unified-design.md](07-urabrask-unified-design.md): Validation reports
- [00-leyline-shared-contracts.md](00-leyline-shared-contracts.md): Shared contracts

## Implementation Status

### Current State
- [x] Core Storage Layer - S3 + PostgreSQL implemented
- [x] Query Engine - Circuit breakers and timeouts complete
- [x] Cache Architecture - 32GB Redis cluster deployed
- [x] Lifecycle Management - State machine with GC
- [x] Integration Interfaces - Protocol Buffers v2
- [x] Data Integrity - SHA256 + CRC32 verification
- [x] Versioning System - Semantic versioning complete
- [x] Dependency Resolution - Tarjan's algorithm implemented
- [x] Compatibility Matrix - Multi-dimensional validation
- [x] Rollback Computation - Dijkstra's algorithm
- [x] RESTful API - Protocol Buffer endpoints
- [x] Conservative Mode - Graceful degradation
- [x] Memory Management - GC and TTL cleanup

### Validation Status
- [x] Unit tests complete (>90% coverage)
- [x] Integration tests complete
- [x] Performance validation (<500ms p99)
- [x] Security review passed
- [x] Production readiness review complete

## History & Context

### Version History
- **v1.0** (2024-12-15): Initial design with basic storage
- **v2.0** (2024-12-30): Added query engine and caching
- **v3.0** (2025-01-10): C-016 critical fixes integrated

### Integration History
- **C-016 External Review** (2025-01-08): Critical issues identified
- **C-016 Integration** (2025-01-10): All fixes implemented and validated

### Critical Fixes Applied
- **C-016-01**: Query timeout reduced from 5000ms to 500ms
- **C-016-02**: Memory leak prevention with garbage collection
- **C-016-03**: WAL durability with O_DSYNC + fsync
- **C-016-04**: Protocol Buffers v2 without map fields
- **C-016-05**: SHA256 + CRC32 integrity verification
- **C-016-06**: Conservative mode for graceful degradation
- **C-016-07**: Circuit breakers replace assert statements
- **C-016-08**: Duration standardization to _ms suffix
- **C-016-09**: Semantic versioning with dependency resolution
- **C-016-10**: Dijkstra rollback path computation

---

*Last Updated: 2025-01-10 | Next Review: 2025-07-10 | Owner: System Architecture Team*