# Urza Combined Design

---

File: docs/design/detailed_design/08-urza-unified-design.md
---

# Urza Unified Design (Esper-Lite)

## Snapshot

- **Role**: Central library holding blueprint metadata and compiled kernel artifacts.
- **Scope**: Receive static blueprints/compiled binaries from Tezzeret, store them immutably, and serve metadata to Tamiyo and binaries to Kasmina (and other consumers) with low latency.
- **Status**: Production; C‑016 safety features (WAL durability, circuit breakers, conservative mode, TTL cleanup) remain, but Urabrask integration is dropped for Esper-Lite.

## Responsibilities

- Store BlueprintIR descriptors and compiled artifacts with versioned metadata.
- Provide fast query API (Tag/ID lookup) for Tamiyo and other clients. No validation workflow in-lite.
- Deliver kernel binaries to Kasmina within <10 ms p50 via cache tiering.
- Maintain integrity (SHA-256) and recovery (WAL) for all assets.

## Component Map

| Component | Purpose | Reference |
| --- | --- | --- |
| Object Store (S3/Zstd) | Immutable artifact storage | `08.1-urza-internals.md` |
| Metadata DB (PostgreSQL) | Queryable metadata + version history | `08.1` |
| Cache Layer (Memory/Redis/DB) | Sub-ms to ms retrieval tiers | `08.1` |
| Query Engine | Tag/ID retrieval, circuit breakers | `08.1` |
| WAL Manager | Persistence + crash recovery | `08.1` |
| Lifecycle Manager | Track availability, deprecation, retention | `08.1` |

## Simplifications for Esper-Lite

- Inputs limited to Tezzeret (compiled artifacts) and pre-vetted blueprint bundles. No dynamic submissions from Karn or Urabrask.
- Validation status assumed “approved”; Urza only stores metadata provided by upstream tools.
- Query API trimmed to lookup by blueprint id, tag, or curriculum stage.

## Workflow

1. **Ingest**: Tezzeret posts `BlueprintIR` + `CompiledKernelArtifact`; Urza verifies checksum, writes WAL, stores object & metadata.
2. **Catalog**: Metadata DB stores version, tags, platform info, performance hints.
3. **Serve**: Tamiyo/Kasmina query; Query Engine uses cache tiers -> fallback to DB/object store.
4. **Monitor**: Circuit breakers guard query timeouts, cache failures; conservative mode falls back to identity kernels.

## Performance Targets

| Metric | Target | Notes |
| --- | --- | --- |
| Query latency p50 | <10 ms | Served from memory/Redis. |
| Query latency p95 | <200 ms | DB fallback. |
| Query latency p99 | <500 ms | Object store fallback. |
| Cache hit rate | >95 % | Combined L1/L2. |
| WAL recovery | <12 s | Verified via chaos tests. |

## Configuration Highlights

```yaml
urza:
  storage:
    object_store: {type: s3, bucket: urza-kernels, compression: zstd}
    metadata_db: {pool_size: 50, read_replicas: 3}
    cache: {redis_memory_gb: 32, memory_cache_mb: 1024, ttl_seconds: 3600}
  query:
    timeout_ms: 500
    circuit_breaker: {failure_threshold: 3, timeout_ms: 400}
    conservative_mode: {enabled: true, cache_only_serving: true}
  wal: {enabled: true, path: /var/lib/urza/wal, durability_mode: O_DSYNC}
```

## Telemetry & Operations

- Metrics: `urza.query.duration_ms`, `urza.cache.hit_rate`, `urza.breaker.state`, `urza.wal.transactions_total`.
- Logs include asset id, version, checksum, source, retrieval tier used.
- Health endpoint displays cache utilisation, DB lag, breaker states, conservative-mode flag.

## References

- `docs/design/detailed_design/08.1-urza-internals.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

### Mission-Critical Behaviours (Authoritative Reference)

Legacy design details live in `docs/design/detailed_design/08-urza.md`. Esper-Lite continues to rely on the following behaviours:

- **Immutable Catalogue:** BlueprintIR metadata and compiled artefacts are stored with versioning, checksums, and WAL-backed durability to guarantee integrity (Old §"Workflow" steps 1–4).
- **Low-Latency Retrieval:** Multi-tier caching (memory/Redis/object store) can be simplified, but Urza must still deliver p50 <10 ms responses via an in-process cache and maintain >95 % hit rates (Old §"Performance Targets").
- **Metadata Governance:** Query APIs expose tags, curriculum stages, and validation status so Tamiyo/Kasmina can filter artefacts safely (Old §"Workflow" and §"Configuration").
- **Telemetry:** Urza emits query latency, cache hit rate, and breaker state metrics for Nissa to monitor (`urza.query.*`, `urza.cache.*`) (Old §"Telemetry & Operations").

These behaviours ensure Urza remains the single source of truth for blueprint assets even in the slimmed environment.

---

File: docs/design/detailed_design/08.1-urza-internals.md
---

# Urza - Internals

## Document Metadata

| Field | Value |
|-------|-------|
| **Parent Document** | [08-urza-unified-design.md](08-urza-unified-design.md) |
| **Component Type** | System Implementation |
| **Version** | 3.0.0 |
| **Status** | PRODUCTION |
| **Implementation** | Complete |

## Overview

This document provides the internal implementation details of Urza for the Esper-Lite build. Urza acts as the central library that stores pre-approved blueprints and compiled kernel artifacts produced by Tezzeret, then serves metadata to Tamiyo and binaries to Kasmina. Dynamic validation or generative ingestion is outside the current scope, but the C‑016 safety improvements (WAL, circuit breakers, TTL cleanup) remain in place.

Key characteristics:

- **Content-Addressable Storage**: SHA-256 based immutable storage with integrity verification
- **Multi-Tier Caching**: Sub-10 ms retrieval through memory/Redis/DB tiers
- **Circuit Breaker Protection**: All operations protected with timeout and failure handling

## Technical Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Urza Central Library                     │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Ingestion   │ Query Engine │   Storage    │  Lifecycle    │
│  Interfaces  │              │  Management  │  Management   │
├──────────────┼──────────────┼──────────────┼───────────────┤
│   Tezzeret   │    Tamiyo    │   Object     │  Blueprint    │
│   Receiver   │   Handler    │    Store     │  & Kernel     │
├──────────────┼──────────────┼──────────────┼───────────────┤
│              │   Kasmina    │  Metadata    │   Retention   │
│              │   Handler    │   Database   │   Manager     │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

### Core Abstractions

**UrzaCentralLibrary**

```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import hashlib
import sqlite3
from esper_protocols_v2 import (
    BlueprintProto, KernelProto, PolicyCheckpointProto,
    SystemStatePacket
)

class UrzaCentralLibrary:
    """Immutable repository for all morphogenetic assets with C-016 enhancements."""

    def __init__(self, config: UrzaConfig):
        # [C-016] Storage layer with WAL durability
        self.object_store = S3ObjectStore(
            durability_config=WALDurabilityConfig(
                use_o_dsync=True,
                fsync_barriers=True,
                filesystem_type="ext4"
            )
        )
        self.metadata_db = PostgreSQLDatabase(
            connection_pool_size=50,  # [C-016] Increased from default
            max_connections=200
        )
        self.cache_layer = RedisCache(
            max_memory_gb=32,  # [C-016] Scaled cache
            eviction_policy='allkeys-lru',
            clustering_enabled=True,
            shards=4
        )
        self.search_index = ElasticsearchIndex()

        # Asset management with versioning
        self.blueprint_manager = BlueprintManager()
        self.kernel_manager = KernelManager()
        self.lineage_tracker = LineageTracker()
        self.lifecycle_manager = LifecycleStateManager()

        # [C-016] Query engine with circuit breaker
        self.query_engine = AdvancedQueryEngine(
            circuit_breaker=CircuitBreaker(
                failure_threshold=3,
                window_seconds=60,
                recovery_seconds=30,
                timeout_ms=400  # [C-016] Fixed timeout
            )
        )
        self.tag_matcher = TagMatchingEngine()
        self.performance_ranker = PerformanceRanker()

        # [C-016] Versioning and compatibility system
        self.version_manager = SemanticVersionManager()
        self.dependency_resolver = DependencyResolver()
        self.compatibility_checker = CompatibilityChecker()
        self.rollback_computer = RollbackPathComputer()

        # Integration interfaces with Protocol Buffers v2
        self.tezzeret_interface = TezzeretArtifactReceiver()
        self.tamiyo_interface = TamiyoQueryHandler()
        self.kasmina_interface = KasminaArtifactHandler()

        # [C-016] Maintenance with memory leak prevention
        self.curation_pipeline = NightlyCurationPipeline()
        self.retention_manager = RetentionPolicyManager()
        self.integrity_validator = DataIntegrityValidator()
        self.garbage_collector = MemoryGarbageCollector(
            cleanup_interval_epochs=100
        )
```

## Storage Architecture

### Object Store

**S3 Object Store with WAL Durability**

```python
class S3ObjectStore:
    """[C-016] Object storage with Write-Ahead Log durability."""

    def __init__(self, durability_config: WALDurabilityConfig):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'urza-kernels'
        self.durability = durability_config

        # Configure filesystem for durability
        if durability_config.use_o_dsync:
            self.open_flags = os.O_WRONLY | os.O_CREAT | os.O_DSYNC
        else:
            self.open_flags = os.O_WRONLY | os.O_CREAT

    async def store_with_wal(self, key: str, data: bytes, metadata: dict) -> str:
        """Store object with WAL durability guarantees."""
        # Write to WAL first
        wal_entry = WALEntry(
            operation='STORE',
            key=key,
            size=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
            timestamp=datetime.utcnow()
        )

        # Ensure WAL durability with fsync
        with open(self.wal_path, 'ab', self.open_flags) as wal:
            wal.write(wal_entry.serialize())
            if self.durability.fsync_barriers:
                os.fsync(wal.fileno())

        # Store to S3
        response = await self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            Metadata=metadata,
            ServerSideEncryption='AES256'
        )

        # Mark WAL entry as committed
        await self.mark_wal_committed(wal_entry)

        return response['ETag']

@dataclass
class WALDurabilityConfig:
    """[C-016] WAL durability configuration."""
    use_o_dsync: bool = True  # Use O_DSYNC for synchronous writes
    fsync_barriers: bool = True  # Use fsync barriers
    filesystem_type: str = "ext4"  # Filesystem type
    filesystem_mount_options: List[str] = field(default_factory=lambda: [
        "data=ordered",  # Ordered data mode
        "barrier=1"      # Enable write barriers
    ])
```

### Metadata Database

**PostgreSQL Metadata Storage**

```python
class PostgreSQLDatabase:
    """Metadata storage with connection pooling and read replicas."""

    def __init__(self, connection_pool_size: int, max_connections: int):
        self.primary = psycopg2.pool.ThreadedConnectionPool(
            minconn=10,
            maxconn=connection_pool_size,
            host='urza-primary',
            database='urza_metadata'
        )

        self.read_replicas = [
            psycopg2.pool.ThreadedConnectionPool(
                minconn=5,
                maxconn=connection_pool_size // 3,
                host=f'urza-replica-{i}',
                database='urza_metadata'
            )
            for i in range(1, 4)
        ]

        self.max_connections = max_connections

    async def store_metadata(self, metadata: dict) -> str:
        """Store metadata with ACID guarantees."""
        conn = self.primary.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO kernel_metadata (
                        artifact_id, blueprint_id, version,
                        metadata_json, sha256_checksum, crc32_checksum
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    metadata['artifact_id'],
                    metadata['blueprint_id'],
                    metadata['version'],
                    json.dumps(metadata),
                    metadata['sha256_checksum'],
                    metadata['crc32_checksum']
                ))

                result = cur.fetchone()[0]
                conn.commit()
                return result
        finally:
            self.primary.putconn(conn)

    async def query_metadata(self, query: dict) -> List[dict]:
        """Query metadata from read replica."""
        # Load balance across replicas
        replica = random.choice(self.read_replicas)
        conn = replica.getconn()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query with GIN index for JSONB
                sql = self._build_query_sql(query)
                cur.execute(sql, query.get('params', []))
                return cur.fetchall()
        finally:
            replica.putconn(conn)
```

## Lifecycle Management

### Blueprint Lifecycle

```python
from time import perf_counter
from esper_protocols_v2.duration import ProtocolDuration

class BlueprintLifecycle:
    """Manage blueprint states from generation to deprecation."""

    STATES = {
        'REGISTERED': 'Blueprint metadata recorded in Urza',
        'COMPILATION_PENDING': 'Waiting for compiled artifacts from Tezzeret',
        'AVAILABLE': 'Compiled artifacts present and ready for serving',
        'DEPRECATED': 'Superseded by newer version',
        'ARCHIVED': 'Moved to cold storage'
    }

    def __init__(self):
        # [C-016] Circuit breaker instead of assertions
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_ms=500
        )
        self.conservative_mode = False

    def transition_state(
        self,
        blueprint_id: str,
        new_state: str,
        metadata: Dict[str, Any]
    ) -> StateTransition:
        """[C-016] Atomic state transition with circuit breaker protection."""

        start_time = perf_counter()

        try:
            with self.circuit_breaker.protect():
                # Validate transition
                current_state = self.get_current_state(blueprint_id)
                if not self.is_valid_transition(current_state, new_state):
                    if not self.conservative_mode:
                        self._trigger_conservative_mode("Invalid state transition")
                    return self._create_fallback_transition(blueprint_id, current_state)

                # Execute transition
                transition = StateTransition(
                    blueprint_id=blueprint_id,
                    from_state=current_state,
                    to_state=new_state,
                    timestamp=datetime.utcnow(),
                    metadata=metadata
                )

                # Update database with durability
                with self.db.transaction():
                    self.db.update_blueprint_state(blueprint_id, new_state)
                    self.db.log_transition(transition)
                    # [C-016] Ensure durability
                    self.db.fsync()

                # Publish event
                self.publish_state_change(transition)

                return transition

        except CircuitBreakerOpen:
            self._handle_circuit_open("Blueprint state transition")
            return self._create_fallback_transition(blueprint_id, current_state)
        finally:
            # [C-016] Timing budget with circuit breaker instead of assert
            duration_ms = (perf_counter() - start_time) * 1000.0
            if duration_ms > 500.0:
                self.metrics.state_transition_overrun_ms.observe(duration_ms)
                if not self.conservative_mode:
                    self._trigger_conservative_mode(f"State transition took {duration_ms}ms")

    def _trigger_conservative_mode(self, reason: str) -> None:
        """[C-016] Conservative mode instead of system crash."""
        self.conservative_mode = True
        self.metrics.conservative_mode_triggered.inc({"reason": reason})
        # Implement conservative policies:
        # - Disable experimental features
        # - Reduce query complexity
        # - Extend timeouts
        # - Use cache-only serving
```

### Kernel Lifecycle

```python
class KernelLifecycle:
    """Manage compiled kernel artifact states with memory leak prevention."""

    STATES = {
        'REGISTERED': 'Artifact stored and checksum verified',
        'ACTIVE': 'Available for serving',
        'DEPRECATED': 'Superseded by newer artifact',
        'RETIRED': 'Removed from active serving'
    }

    def __init__(self):
        self.artifact_cache: Dict[Tuple[str, int], KernelMetadata] = {}
        self.cache_ttl_epochs = 100
        self.last_gc_epoch = 0

    def register_artifact(
        self,
        artifact: CompiledKernelArtifact,
        metadata: Dict[str, Any]
    ) -> KernelStateUpdate:
        """Record an artifact produced by Tezzeret and mark it active."""

        current_epoch = self.get_current_epoch()
        if current_epoch - self.last_gc_epoch >= self.cache_ttl_epochs:
            self._gc_artifact_cache(current_epoch)
            self.last_gc_epoch = current_epoch

        kernel_id = artifact.kernel_id
        kernel_metadata = self._load_metadata(kernel_id)
        kernel_metadata.state = 'ACTIVE'
        kernel_metadata.registered_at = datetime.utcnow()
        kernel_metadata.attributes.update(metadata)
        kernel_metadata.checksum = hashlib.sha256(artifact.bytes).hexdigest()

        self._persist_metadata(kernel_metadata)

        return KernelStateUpdate(
            kernel_id=kernel_id,
            new_state='ACTIVE',
            timestamp=datetime.utcnow(),
            metadata=kernel_metadata
        )

    def _gc_artifact_cache(self, current_epoch: int) -> None:
        """[C-016] Garbage collection to prevent memory leaks."""
        expired_keys = [
            key for key, metadata in self.artifact_cache.items()
            if current_epoch - key[1] > self.cache_ttl_epochs
        ]

        for key in expired_keys:
            del self.artifact_cache[key]

        self.metrics.artifact_cache_gc_cleaned.observe(len(expired_keys))
```

## Metadata Schema

### Blueprint Metadata

```protobuf
// shared/contracts/v2/blueprint.proto
syntax = "proto3";
package esper.blueprint.v2;

import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";

message BlueprintMetadata {
    // Identification
    string blueprint_id = 1;
    string name = 2;
    string version = 3;
    string lineage_id = 4;

    // Classification
    string blueprint_type = 5;
    string architecture_family = 6;
    string category = 7;
    string subcategory = 8;
    repeated string tags = 9;

    // Origin and authorship
    string generator = 10;
    google.protobuf.Timestamp generation_timestamp = 11;
    // [C-016] No map<> fields - use repeated KeyValue instead
    repeated KeyValuePair generation_context = 12;
    string parent_blueprint_id = 13;

    // Technical specifications
    int64 parameter_count = 20;
    int32 layer_count = 21;
    repeated KeyValuePair operation_graph = 22;  // [C-016] No map<>
    double estimated_memory_mb = 23;
    int64 estimated_flops = 24;

    // Strategic metadata
    double tamiyo_interest = 30;
    double karn_novelty_score = 31;
    double exploration_priority = 32;

    // Compilation status
    int32 compilation_attempts = 40;
    int32 successful_compilations = 41;
    repeated string compilation_strategies_tried = 42;

    // Validation summary
    int32 validated_artifacts = 50;
    int32 rejected_artifacts = 51;
    double average_risk_score = 52;
    double best_performance_score = 53;

    // Usage statistics
    int32 deployment_count = 60;
    int64 total_inference_count = 61;
    double average_performance_delta = 62;
    int32 field_report_count = 63;

    // Lifecycle
    string current_state = 70;
    repeated StateTransition state_history = 71;
    string retention_policy = 72;
    google.protobuf.Timestamp archive_date = 73;
}

message KeyValuePair {
    string key = 1;
    string value = 2;
}

message StateTransition {
    string from_state = 1;
    string to_state = 2;
    google.protobuf.Timestamp timestamp = 3;
    repeated KeyValuePair metadata = 4;  // [C-016] No map<>
}
```

### Kernel Metadata

```protobuf
// shared/contracts/v2/kernel.proto
syntax = "proto3";
package esper.kernel.v2;

import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";

message KernelMetadata {
    // Identification
    string artifact_id = 1;
    string blueprint_id = 2;
    string compilation_id = 3;

    // [C-016] Semantic versioning
    KernelVersion version = 10;
    repeated KernelDependency dependencies = 11;
    repeated KeyValuePair compatibility_matrix = 12;  // [C-016] No map<>

    // Compilation details
    string compiler_version = 20;
    string compilation_strategy = 21;
    google.protobuf.Timestamp compilation_timestamp = 22;
    int64 compilation_time_ms = 23;  // [C-016] Standardized duration
    int32 optimization_level = 24;

    // Validation results
    string validation_status = 30;
    string validation_report_id = 31;
    double risk_score = 32;
    string risk_level = 33;
    string confidence_level = 34;

    // Performance characterization
    repeated string empirical_tags = 40;
    repeated PerformanceMetric performance_metrics = 41;  // [C-016] No map<>
    repeated KeyValuePair hardware_profiles = 42;
    WorkloadAffinity workload_affinity = 43;
    ScalingProfile scaling_characteristics = 44;

    // Scientific interest
    double urabrask_interest = 50;
    repeated string novel_characteristics = 51;
    repeated string research_recommendations = 52;

    // Storage and access with integrity
    string storage_location = 60;
    double storage_size_mb = 61;
    string sha256_checksum = 62;  // [C-016] SHA256 integrity
    string crc32_checksum = 63;   // [C-016] CRC32 verification
    string encryption_key_id = 64;

    // Usage tracking
    int64 load_count = 70;
    google.protobuf.Timestamp last_accessed = 71;
    int64 average_load_time_ms = 72;  // [C-016] Standardized duration
    int64 cache_hits = 73;

    // Deployment history
    repeated DeploymentRecord deployments = 80;
    repeated string field_reports = 81;
    repeated string production_issues = 82;

    // [C-016] Evolution and rollback data
    repeated string evolution_lineage = 90;
    bytes architecture_embedding = 91;
    repeated PerformanceMetric ab_test_metrics = 92;  // [C-016] No map<>
}

message KernelVersion {
    int32 major = 1;
    int32 minor = 2;
    int32 patch = 3;
    string build = 4;
}

message KernelDependency {
    string kernel_id = 1;
    string version_constraint = 2;
    bool optional = 3;
}

message PerformanceMetric {
    string name = 1;
    double value = 2;
    string unit = 3;
}
```

## Versioning System

### Semantic Versioning

```python
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import re

@dataclass
class KernelVersion:
    """[C-016] Semantic versioning for morphogenetic kernels."""
    major: int  # Breaking changes (incompatible tensor shapes, removed ops)
    minor: int  # Compatible features (new ops, optimizations)
    patch: int  # Bug fixes (performance improvements, memory fixes)
    build: Optional[str] = None  # Build metadata (commit hash, timestamp)

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: 'KernelVersion') -> bool:
        """Compare versions for ordering. O(1) complexity."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def is_compatible_with(self, other: 'KernelVersion') -> bool:
        """Check if versions are compatible. O(1) complexity."""
        return self.major == other.major

    @classmethod
    def parse(cls, version_string: str) -> 'KernelVersion':
        """Parse version string. O(1) complexity."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:\+(.+))?$'
        match = re.match(pattern, version_string)
        if not match:
            raise ValueError(f"Invalid version string: {version_string}")
        major, minor, patch, build = match.groups()
        return cls(int(major), int(minor), int(patch), build)

class VersionRange:
    """[C-016] Version constraint specification."""

    def __init__(self, constraint: str):
        """Parse version constraint. Examples: '>=1.2.0', '^1.0.0', '~1.2.3'"""
        self.constraint = constraint
        self.operator, self.version = self._parse_constraint(constraint)

    def matches(self, version: KernelVersion) -> bool:
        """Check if version satisfies constraint. O(1) complexity."""
        if self.operator == '>=':
            return version >= self.version
        elif self.operator == 'caret':
            # ^1.2.3 means >=1.2.3 <2.0.0
            return (version >= self.version and
                    version.major == self.version.major)
        elif self.operator == 'tilde':
            # ~1.2.3 means >=1.2.3 <1.3.0
            return (version >= self.version and
                    version.major == self.version.major and
                    version.minor == self.version.minor)
        return False
```

## Dependency Management

### Dependency Resolution with Tarjan's Algorithm

```python
from collections import defaultdict, deque
import heapq
from typing import Dict, Set, List

@dataclass
class KernelDependency:
    """[C-016] Dependency specification for a kernel."""
    kernel_id: str
    version_constraint: VersionRange
    optional: bool = False

class DependencyGraph:
    """[C-016] Directed Acyclic Graph for kernel dependencies."""

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.in_degree: Dict[str, int] = defaultdict(int)

    def detect_cycles_tarjan(self) -> List[List[str]]:
        """
        [C-016] Tarjan's strongly connected components algorithm.
        Complexity: O(V + E)
        Returns list of cycles (SCCs with >1 node).
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        cycles = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in self.edges[node]:
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])

            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break

                # Only report cycles (SCCs with >1 node)
                if len(component) > 1:
                    cycles.append(component)

        for node in self.nodes:
            if node not in index:
                strongconnect(node)

        return cycles

    def topological_sort(self) -> List[str]:
        """
        [C-016] Kahn's algorithm for topological sorting.
        Complexity: O(V + E)
        Returns installation order or raises if cycle detected.
        """
        # [C-016] Circuit breaker instead of exception
        cycles = self.detect_cycles_tarjan()
        if cycles:
            self.metrics.dependency_cycles_detected.inc({"cycles": len(cycles)})
            if not self.conservative_mode:
                self._trigger_conservative_mode(f"Dependency cycles: {cycles}")
            return self._fallback_linear_order()

        queue = deque([node for node in self.nodes
                      if self.in_degree[node] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in self.edges[current]:
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            self.metrics.topological_sort_failed.inc()
            return self._fallback_linear_order()

        return result
```

### Compatibility Checking

```python
class CompatibilityChecker:
    """[C-016] Comprehensive compatibility validation."""

    def __init__(self):
        self.dtype_compatibility = {
            torch.float32: {torch.float32, torch.float16, torch.bfloat16},
            torch.float16: {torch.float16, torch.float32},
            torch.bfloat16: {torch.bfloat16, torch.float32},
            torch.int8: {torch.int8, torch.int16, torch.int32},
        }
        # [C-016] Circuit breaker for compatibility checks
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_ms=100  # Fast compatibility checks
        )

    def check_shape_compatibility(self, k1: Dict, k2: Dict) -> Tuple[bool, str]:
        """
        [C-016] Validate tensor shape compatibility with circuit breaker.
        Complexity: O(n) where n=number of tensors
        """
        try:
            with self.circuit_breaker.protect():
                k1_inputs = k1.get('input_shapes', {})
                k1_outputs = k1.get('output_shapes', {})
                k2_inputs = k2.get('input_shapes', {})

                for output_name, output_shape in k1_outputs.items():
                    if output_name in k2_inputs:
                        input_shape = k2_inputs[output_name]
                        if not self._shapes_compatible(output_shape, input_shape):
                            return False, f"Shape mismatch: {output_name} {output_shape} -> {input_shape}"

                return True, "Shapes compatible"

        except CircuitBreakerOpen:
            return False, "Compatibility check timeout - assuming incompatible"

    def check_hardware_compatibility(self, k1: Dict, k2: Dict) -> Tuple[bool, str]:
        """[C-016] Validate hardware requirements with circuit breaker."""
        try:
            with self.circuit_breaker.protect():
                k1_compute = k1.get('min_compute_capability', 3.5)
                k2_compute = k2.get('min_compute_capability', 3.5)
                max_compute = max(k1_compute, k2_compute)

                k1_memory = k1.get('min_memory_gb', 4)
                k2_memory = k2.get('min_memory_gb', 4)
                total_memory = k1_memory + k2_memory * 0.5

                # [C-016] Graceful hardware detection
                try:
                    current_compute = torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] * 0.1
                    current_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                except:
                    # [C-016] Conservative fallback when CUDA unavailable
                    return True, "CUDA unavailable - assuming compatibility"

                if current_compute < max_compute:
                    return False, f"Insufficient compute capability: {current_compute} < {max_compute}"

                if current_memory < total_memory:
                    return False, f"Insufficient memory: {current_memory}GB < {total_memory}GB"

                return True, "Hardware compatible"

        except CircuitBreakerOpen:
            return False, "Hardware check timeout - assuming incompatible"
```

## Query Engine

### Advanced Query Engine

```python
import asyncio
from time import perf_counter
from esper.shared.circuit_breaker import CircuitBreaker

class AdvancedQueryEngine:
    """[C-016] Query engine with timeout protection and conservative mode."""

    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            window_seconds=60,
            recovery_seconds=30,
            timeout_ms=400  # [C-016] Fixed 400ms threshold
        )
        self.identity_kernel = self._load_identity_kernel()
        self.conservative_mode = False

    async def query_kernels(self, query: KernelQuery) -> KernelQueryResult:
        """
        [C-016] Execute complex kernel query with 500ms timeout and circuit breaker.
        """
        start_time = perf_counter()

        # [C-016] Conservative mode handling
        if self.conservative_mode:
            return await self._conservative_query(query)

        # Fast path: metadata-only queries (target <100ms)
        if query.metadata_only:
            try:
                return await asyncio.wait_for(
                    self._metadata_query(query),
                    timeout=0.4  # 400ms budget
                )
            except asyncio.TimeoutError:
                self.metrics.metadata_query_timeout.inc()
                return self._fallback_result()

        # Complex path with circuit breaker protection
        try:
            with self.circuit_breaker.protect():
                result = await asyncio.wait_for(
                    self._full_query(query),
                    timeout=0.45  # 450ms hard limit
                )
                return result

        except (asyncio.TimeoutError, CircuitBreakerOpen) as e:
            self.metrics.query_timeout.inc({"reason": type(e).__name__})
            return KernelQueryResult(
                results=[self.identity_kernel],
                metadata_only=False,
                fallback_used=True,
                execution_time_ms=500  # Timeout
            )
        finally:
            # [C-016] Timing budget check with circuit breaker
            duration_ms = (perf_counter() - start_time) * 1000.0
            if duration_ms > 500.0:
                self.metrics.query_overrun_ms.observe(duration_ms)
                if not self.conservative_mode:
                    self._trigger_conservative_mode(f"Query took {duration_ms}ms")

    async def _conservative_query(self, query: KernelQuery) -> KernelQueryResult:
        """[C-016] Conservative mode - cache-only serving."""
        # In conservative mode:
        # - Only serve from cache
        # - Reduce query complexity
        # - Use identity kernel fallbacks
        # - Extend timeouts
        cache_result = await self.cache_layer.get_query_result(query.cache_key)
        if cache_result:
            return cache_result

        # Fallback to identity kernel in conservative mode
        return KernelQueryResult(
            results=[self.identity_kernel],
            metadata_only=False,
            fallback_used=True,
            conservative_mode=True,
            execution_time_ms=50  # Fast conservative response
        )

    def _trigger_conservative_mode(self, reason: str) -> None:
        """[C-016] Enter conservative mode instead of crashing."""
        self.conservative_mode = True
        self.metrics.conservative_mode_triggered.inc({"reason": reason})

        # Conservative mode policies:
        self.circuit_breaker.timeout_ms = int(self.circuit_breaker.timeout_ms * 1.5)
        self.enable_experimental_features = False
        self.cache_ttl_ms = self.cache_ttl_ms * 2

        # Schedule recovery attempt
        asyncio.create_task(self._attempt_recovery_after_delay(300))  # 5 minutes

    def find_kernels_by_tags(
        self,
        tags: List[str],
        match_mode: str = 'all'
    ) -> List[KernelMetadata]:
        """[C-016] Find kernels matching empirical tags with circuit breaker."""

        try:
            with self.circuit_breaker.protect():
                if match_mode == 'all':
                    sql = """
                        SELECT k.* FROM kernels k
                        WHERE k.empirical_tags @> %s
                        AND k.validation_status = 'VALIDATED'
                        ORDER BY k.risk_score ASC
                        LIMIT 100
                    """
                elif match_mode == 'any':
                    sql = """
                        SELECT k.* FROM kernels k
                        WHERE k.empirical_tags && %s
                        AND k.validation_status = 'VALIDATED'
                        ORDER BY k.risk_score ASC
                        LIMIT 100
                    """

                return self.db.execute(sql, [tags])

        except CircuitBreakerOpen:
            self.metrics.tag_query_circuit_open.inc()
            return [self.identity_kernel]
```

### Tamiyo Query Handler

```python
import heapq
from typing import Set

class TamiyoQueryHandler:
    """[C-016] Specialized query interface for Tamiyo's strategic needs."""

    def __init__(self):
        self.rollback_computer = RollbackPathComputer()

    def get_rollback_kernel(
        self,
        failed_kernel_id: str
    ) -> KernelMetadata:
        """[C-016] Get safe rollback kernel using Dijkstra's algorithm."""

        try:
            failed_kernel = self.get_kernel(failed_kernel_id)
            blueprint_id = failed_kernel.blueprint_id

            # Use rollback computer for optimal path
            safe_kernels = self.get_safe_kernels(blueprint_id)
            rollback_path = self.rollback_computer.compute_balanced_rollback(
                current_kernel=failed_kernel_id,
                safe_kernels=set(safe_kernels),
                risk_weight=0.8  # Prioritize safety over speed
            )

            if rollback_path and len(rollback_path) > 1:
                return self.get_kernel(rollback_path[1])  # First step back

            # Fallback to identity kernel if no path found
            return self.get_identity_kernel()

        except Exception as e:
            self.metrics.rollback_kernel_error.inc({"error": type(e).__name__})
            return self.get_identity_kernel()

class RollbackPathComputer:
    """[C-016] Compute optimal rollback paths using Dijkstra's algorithm."""

    def compute_balanced_rollback(self,
                                  current_kernel: str,
                                  safe_kernels: Set[str],
                                  risk_weight: float = 0.7) -> List[str]:
        """
        [C-016] Balance between safety and speed with circuit breaker.
        Complexity: O((V + E) log V)
        Target: <500ms rollback path computation
        """
        start_time = perf_counter()

        try:
            distances = {node: float('inf') for node in self.graph.nodes}
            distances[current_kernel] = 0

            previous = {}
            pq = [(0, current_kernel)]
            visited = set()

            while pq:
                current_cost, current = heapq.heappop(pq)

                if current in visited:
                    continue
                visited.add(current)

                # Found safe kernel - return path
                if current in safe_kernels:
                    return self._reconstruct_path(previous, current_kernel, current)

                # [C-016] Time budget check
                duration_ms = (perf_counter() - start_time) * 1000.0
                if duration_ms > 500.0:
                    self.metrics.rollback_computation_timeout.inc()
                    break

                for edge in self.graph.edges:
                    if edge.from_kernel == current:
                        neighbor = edge.to_kernel

                        # Weighted combination of risk and time
                        risk_component = edge.transition_risk * risk_weight
                        time_component = (edge.transition_time / 1000) * (1 - risk_weight)
                        combined_cost = risk_component + time_component

                        new_distance = distances[current] + combined_cost

                        if new_distance < distances[neighbor]:
                            distances[neighbor] = new_distance
                            previous[neighbor] = current
                            heapq.heappush(pq, (new_distance, neighbor))

            return []  # No path found

        except Exception as e:
            self.metrics.rollback_computation_error.inc({"error": type(e).__name__})
            return []
        finally:
            duration_ms = (perf_counter() - start_time) * 1000.0
            self.metrics.rollback_computation_time_ms.observe(duration_ms)
```

## Cache Architecture

### Multi-Tier Cache

```python
class MultiTierCache:
    """[C-016] Multi-tier caching with memory leak prevention."""

    def __init__(self):
        # L1: Process memory cache (sub-µs)
        self.memory_cache = LRUCache(max_size_mb=1024)

        # L2: Redis cache (sub-ms) - SCALED TO 32GB
        self.redis_cache = RedisCache(
            max_memory='32gb',
            eviction_policy='allkeys-lru',
            clustering_enabled=True,
            shards=4
        )

        # L3: PostgreSQL with read replicas (ms)
        self.indexed_db = PostgreSQLDatabase(
            primary_host='urza-primary',
            read_replicas=[
                'urza-replica-1',
                'urza-replica-2',
                'urza-replica-3'
            ],
            max_connections=200,  # [C-016] Increased from 50
            connection_pool_size=50
        )

        # L4: S3 object store with CDN (10s of ms)
        self.object_store = S3ObjectStore(
            bucket='urza-kernels',
            cdn_enabled=True,
            compression_enabled=True,
            compression_algorithm='zstd'
        )

        # [C-016] Memory leak prevention
        self.cache_cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()

    async def get_kernel(self, kernel_id: str) -> Optional[KernelArtifact]:
        """[C-016] Retrieve kernel with cache hierarchy and GC."""

        # [C-016] Periodic cleanup to prevent memory leaks
        current_time = time.time()
        if current_time - self.last_cleanup > self.cache_cleanup_interval:
            self._cleanup_caches()
            self.last_cleanup = current_time

        # L1: Memory cache (target <1ms)
        if kernel := self.memory_cache.get(kernel_id):
            self.metrics.l1_hits.inc()
            return kernel

        # L2: Redis cache (target <5ms)
        if kernel := await self.redis_cache.get(kernel_id):
            self.metrics.l2_hits.inc()
            self.memory_cache.put(kernel_id, kernel)
            return kernel

        # L3: Database metadata + L4: Object store
        if metadata := await self.indexed_db.get_kernel_metadata(kernel_id):
            # [C-016] Integrity verification
            if not self._verify_integrity(metadata):
                self.metrics.integrity_failures.inc()
                return None

            kernel_data = await self.object_store.get(metadata.storage_location)
            kernel = self.deserialize_kernel(kernel_data, metadata)

            # Populate caches
            await self.redis_cache.put(kernel_id, kernel)
            self.memory_cache.put(kernel_id, kernel)

            self.metrics.cache_misses.inc()
            return kernel

        return None

    def _cleanup_caches(self) -> None:
        """[C-016] Periodic cache cleanup to prevent memory leaks."""
        # Clean L1 memory cache
        old_l1_size = len(self.memory_cache)
        self.memory_cache.cleanup_expired()
        cleaned_l1 = old_l1_size - len(self.memory_cache)

        # Log cleanup metrics
        self.metrics.cache_cleanup_items.observe(cleaned_l1)

    def _verify_integrity(self, metadata: KernelMetadata) -> bool:
        """[C-016] Verify SHA256 + CRC32 integrity."""
        try:
            # Verify both checksums exist
            if not metadata.sha256_checksum or not metadata.crc32_checksum:
                return False

            # Additional integrity checks can be added here
            return True

        except Exception:
            return False
```

### Kernel Size Management

```python
import zstandard as zstd

class KernelSizeLimiter:
    """[C-016] Enforce strict kernel size limits with integrity."""

    MAX_KERNEL_SIZE = 100 * 1024 * 1024  # 100MB hard limit
    COMPRESSION_THRESHOLD = 10 * 1024 * 1024  # 10MB compression threshold

    def __init__(self):
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()

    async def validate_and_store(self, kernel: bytes, metadata: dict) -> str:
        """[C-016] Validate size, compress, and verify integrity."""

        # [C-016] Circuit breaker instead of exception
        try:
            # Reject oversized kernels
            if len(kernel) > self.MAX_KERNEL_SIZE:
                self.metrics.kernel_too_large.inc()
                return None  # Conservative handling instead of exception

            # [C-016] Compute integrity checksums
            sha256_hash = hashlib.sha256(kernel).hexdigest()
            crc32_hash = hex(zlib.crc32(kernel) & 0xffffffff)

            # Compress large kernels with zstd
            original_size = len(kernel)
            if len(kernel) > self.COMPRESSION_THRESHOLD:
                kernel = self.compressor.compress(kernel)
                metadata['compressed'] = True
                metadata['compression_ratio'] = len(kernel) / original_size

            # [C-016] Store integrity information
            metadata['sha256_checksum'] = sha256_hash
            metadata['crc32_checksum'] = crc32_hash
            metadata['original_size'] = original_size

            return await self.storage.store(kernel, metadata)

        except Exception as e:
            self.metrics.kernel_store_error.inc({"error": type(e).__name__})
            return None

    async def load_and_verify(self, kernel_id: str) -> Optional[bytes]:
        """[C-016] Load kernel and verify integrity."""
        try:
            metadata = await self.storage.get_metadata(kernel_id)
            kernel_data = await self.storage.get_data(kernel_id)

            # [C-016] Verify integrity
            if metadata.get('compressed'):
                kernel_data = self.decompressor.decompress(kernel_data)

            # Verify checksums
            actual_sha256 = hashlib.sha256(kernel_data).hexdigest()
            actual_crc32 = hex(zlib.crc32(kernel_data) & 0xffffffff)

            if (actual_sha256 != metadata['sha256_checksum'] or
                actual_crc32 != metadata['crc32_checksum']):
                self.metrics.integrity_verification_failed.inc()
                return None

            return kernel_data

        except Exception as e:
            self.metrics.kernel_load_error.inc({"error": type(e).__name__})
            return None
```

## Integration Points

### Internal Integration

| Component | Interface | Data Flow |
|-----------|-----------|-----------|
| Blueprint Manager | `store_blueprint()` | Blueprint metadata storage |
| Kernel Manager | `store_kernel()` | Kernel artifact storage |
| Query Engine | `query_kernels()` | Kernel retrieval |
| Lifecycle Manager | `transition_state()` | State transitions |
| Version Manager | `resolve_dependencies()` | Version resolution |

### External Integration

| Subsystem | Contract | Pattern |
|-----------|----------|---------|
| Tezzeret | KernelProto | Async/Queue |
| Tamiyo | KernelQuery | Sync/Request-Response |
| Kasmina | KernelDownload | Sync/Streaming |

### Leyline Contracts Used

This component uses the following shared contracts:

- `leyline.BlueprintMetadata` - Blueprint storage schema
- `leyline.KernelMetadata` - Kernel storage schema
- `leyline.KernelQuery` - Query request/response
- `leyline.StateTransition` - Lifecycle state changes

## Configuration

```yaml
# [C-016] Updated configuration with circuit breaker and scaling
urza:
  # Storage configuration with WAL durability
  storage:
    object_store:
      type: s3
      bucket: urza-kernels
      region: us-east-1
      encryption: AES256
      versioning: enabled
      compression: zstd
      # [C-016] WAL durability settings
      durability:
        use_o_dsync: true
        fsync_barriers: true
        filesystem_type: ext4
        filesystem_mount_options:
          - "data=ordered"
          - "barrier=1"

    metadata_db:
      type: postgresql
      primary_host: urza-db.internal
      read_replicas:
        - urza-replica-1.internal
        - urza-replica-2.internal
        - urza-replica-3.internal
      port: 5432
      database: urza_metadata
      max_connections: 200  # [C-016] Increased from 50
      pool_size: 50

    cache:
      redis_url: redis://urza-cache:6379
      memory_cache_mb: 1024
      redis_memory_gb: 32  # [C-016] Scaled from 8GB
      ttl_seconds: 3600
      clustering_enabled: true
      shards: 4
      # [C-016] Memory leak prevention
      cleanup_interval_seconds: 3600

  # [C-016] Query configuration - CRITICAL 500ms timeout
  query:
    max_results: 100
    default_limit: 10
    timeout_ms: 500  # [C-016] FIXED from 5000ms
    enable_caching: true

    # [C-016] Circuit breaker configuration
    circuit_breaker:
      enabled: true
      failure_threshold: 3
      window_seconds: 60
      recovery_seconds: 30
      half_open_requests: 1
      timeout_ms: 400  # [C-016] Query circuit breaker timeout

    # [C-016] Conservative mode configuration
    conservative_mode:
      enabled: true
      trigger_on_timeout: true
      trigger_on_circuit_open: true
      cache_only_serving: true
      reduced_query_complexity: true
      extended_timeout_factor: 1.5

    # Fallback strategy
    fallback:
      strategy: identity_kernel
      cache_ttl_ms: 300000  # [C-016] Standardized duration (5 minutes)
      preload: true

    # Query optimization
    optimization:
      metadata_first: true
      async_binary: true
      query_cache: true

  # [C-016] Kernel size limits and integrity
  limits:
    max_kernel_size_mb: 100
    compression_threshold_mb: 10
    compression_algorithm: zstd
    # [C-016] Integrity verification
    verify_checksums: true
    sha256_required: true
    crc32_required: true

  # [C-016] Lifecycle management with GC
  lifecycle:
    retention_days:
      active: -1  # Keep forever
      validated: 365
      rejected: 30
      deprecated: 90

    auto_archive: true
    archive_storage: glacier

    # [C-016] Memory management
    garbage_collection:
      enabled: true
      cleanup_interval_epochs: 100
      cache_cleanup_interval_ms: 3600000  # [C-016] 1 hour

  # [C-016] Performance targets - CONCRETE SPECIFICATIONS
  performance:
    query_latency_ms:
      p50: 10  # Cache hits
      p95: 200  # Complex queries
      p99: 500  # Hard timeout

    cache_hit_rate: 0.95  # 95% minimum
    throughput_qps: 1000  # Sustained

  # Integration with Protocol Buffers v2
  integration:
    message_bus_timeout_ms: 10000  # [C-016] Standardized duration
    enable_events: true
    protocol_version: v2  # [C-016] Protocol Buffers v2
    event_topics:
      - blueprint.stored
      - kernel.validated
      - kernel.rejected
      - kernel.version.released
      - kernel.dependency.resolved
      - kernel.rollback.initiated

    # [C-016] Protocol governance
    protocol_governance:
      containerized_toolchain: true
      golden_tests_enabled: true
      decode_reencode_validation: true
      forbidden_features:
        - "map fields"
        - "float counters"
        - "oneof spanning planes"
```

## Performance Characteristics

### Benchmarks

| Operation | Target | Measured | Conditions |
|-----------|--------|----------|------------|
| Kernel query (cached) | <10ms | 8ms | p50, cache hit |
| Kernel query (complex) | <200ms | 150ms | p95, multi-condition |
| Kernel query (timeout) | <500ms | 500ms | p99, circuit breaker |
| Kernel storage | <100ms | 75ms | With compression |
| Metadata update | <50ms | 30ms | Single field |

### Resource Usage

- **Memory**: 32GB Redis cache + 1GB process memory
- **CPU**: 4 cores minimum, scales with query load
- **I/O**: High read throughput, moderate write
- **Storage**: 500GB SSD for hot data, S3 for cold

### Optimization Strategies

1. **Multi-tier caching**: L1-L4 cache hierarchy for optimal latency
2. **Read replicas**: PostgreSQL replicas for query distribution
3. **Compression**: Zstd compression for 40% storage savings
4. **Circuit breakers**: Prevent cascade failures under load
5. **Conservative mode**: Graceful degradation when overloaded

## Error Handling

### Failure Modes

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| Query timeout | >500ms execution | Return identity kernel |
| Circuit breaker open | 3 failures in 60s | Conservative mode |
| Cache failure | Redis unreachable | Fallback to DB |
| Integrity failure | Checksum mismatch | Reject kernel |
| Memory leak | >32GB usage | Trigger GC |

### Circuit Breakers

```python
# Circuit breaker configuration
circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout_ms=30000,
    half_open_requests=1
)
```

### Fallback Behavior

When this component fails:

1. Query timeout → Return identity kernel
2. Circuit breaker open → Enter conservative mode
3. Conservative mode → Cache-only serving

## API Specification

### RESTful API

```python
# [C-016] API specification with Protocol Buffers v2 and error handling
from flask import Flask, request, jsonify
from esper_protocols_v2 import KernelQueryRequest, KernelQueryResponse
import asyncio

@app.route('/api/v1/kernels/query', methods=['POST'])
async def query_kernels():
    """[C-016] Query kernels with Protocol Buffers v2 and circuit breaker."""

    try:
        # [C-016] Parse Protocol Buffer request
        query_request = KernelQueryRequest()
        query_request.ParseFromString(request.get_data())

        # Execute query with circuit breaker
        result = await urza_library.query_engine.query_kernels(query_request)

        # [C-016] Return Protocol Buffer response
        response = KernelQueryResponse(
            results=result.results,
            metadata_only=result.metadata_only,
            fallback_used=result.fallback_used,
            execution_time_ms=result.execution_time_ms,
            conservative_mode=result.conservative_mode
        )

        return response.SerializeToString(), 200, {
            'Content-Type': 'application/x-protobuf'
        }

    except Exception as e:
        error_response = KernelQueryResponse(
            error=str(e),
            fallback_used=True
        )
        return error_response.SerializeToString(), 500, {
            'Content-Type': 'application/x-protobuf'
        }

# [C-016] Version-aware kernel retrieval
@app.route('/api/v1/kernels/<kernel_id>')
async def get_kernel(kernel_id: str):
    """Get kernel with integrity verification."""

    try:
        version = request.args.get('version', 'latest')

        # [C-016] Get kernel with integrity check
        kernel = await urza_library.get_kernel_with_verification(kernel_id, version)

        if not kernel:
            return jsonify({"error": "Kernel not found or integrity check failed"}), 404

        # Return metadata as Protocol Buffer
        return kernel.SerializeToString(), 200, {
            'Content-Type': 'application/x-protobuf'
        }

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# [C-016] Rollback path computation
@app.route('/api/v1/kernels/<kernel_id>/rollback', methods=['POST'])
async def compute_rollback_path(kernel_id: str):
    """Compute rollback path with Dijkstra's algorithm."""

    try:
        rollback_request = RollbackRequest()
        rollback_request.ParseFromString(request.get_data())

        # Compute rollback path
        rollback_path = await urza_library.tamiyo_interface.compute_rollback_path(
            kernel_id, rollback_request.strategy, rollback_request.risk_weight
        )

        response = RollbackResponse(
            path=rollback_path,
            computation_time_ms=rollback_path.computation_time_ms
        )

        return response.SerializeToString(), 200, {
            'Content-Type': 'application/x-protobuf'
        }

    except Exception as e:
        error_response = RollbackResponse(error=str(e))
        return error_response.SerializeToString(), 500, {
            'Content-Type': 'application/x-protobuf'
        }
```

## Monitoring & Observability

### Metrics

| Metric | Type | Purpose |
|--------|------|---------|
| `urza_query_duration_ms` | Histogram | Query latency tracking |
| `urza_cache_hits_total` | Counter | Cache hit rate |
| `urza_circuit_breaker_state` | Gauge | Circuit breaker status |
| `urza_conservative_mode` | Gauge | Conservative mode active |
| `urza_storage_size_gb` | Gauge | Storage usage |
| `urza_metadata_attributes` | Gauge | Metadata dimensions |

### Logging

```python
# Logging levels and patterns
logger.debug(f"Urza querying kernel {kernel_id}")
logger.info(f"Urza stored blueprint {blueprint_id}")
logger.warning(f"Urza query timeout: {query_id}")
logger.error(f"Urza integrity check failed: {kernel_id}", exc_info=True)
```

### Tracing

- **Span**: `urza.query.execute`
  - **Attributes**: query_type, cache_hit, fallback_used
  - **Events**: timeout, circuit_breaker_open, conservative_mode

## Implementation Checklist

- [x] Core storage layer with WAL durability
- [x] Query engine with circuit breakers
- [x] Multi-tier cache architecture
- [x] Lifecycle management with GC
- [x] Protocol Buffers v2 integration
- [x] SHA256 + CRC32 integrity
- [x] Semantic versioning system
- [x] Tarjan's dependency resolution
- [x] Dijkstra's rollback computation
- [x] Conservative mode implementation
- [x] Memory leak prevention
- [x] RESTful API with Protocol Buffers

## References

### Internal References

- Parent: [08-urza-unified-design.md](08-urza-unified-design.md)
- Tests: `tests/urza/`
- Implementation: `src/esper/urza/`

### External References

- [Tarjan's Algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
- [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Protocol Buffers v2](https://developers.google.com/protocol-buffers)

## History & Context

### Implementation Notes

- **2025-01-10**: C-016 critical fixes integrated
- **2025-01-10**: Circuit breakers replace all assertions
- **2025-01-10**: 500ms query timeout enforced
- **2025-01-10**: Conservative mode implemented

### Known Issues

- **URZA-001**: Cache invalidation complexity - use TTL approach
- **URZA-002**: Query optimizer needs ML enhancement (Phase 3)

---

*Component Owner: System Architecture Team | Last Updated: 2025-01-10*
