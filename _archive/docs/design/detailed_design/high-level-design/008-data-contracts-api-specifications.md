# Esper HLD - Data Contracts & API Specifications

**Context:** This is part 8 of the Esper High Level Design document breakdown. Complete reference: `/home/john/esper/docs/architecture/hld-sections/`

**Cross-References:**
- Previous: [Component Specifications](./007-component-specifications.md)
- Next: [Appendices and Technical Details](./009-appendices-technical-details.md)
- Related: [System Design & Data Flow](./006-system-design-data-flow.md)
- Architecture: [Reference Architecture Overview](./005-reference-architecture-overview.md)

---

## 8. Data Architecture

A morphogenetic system generates and consumes a rich variety of data, from high-frequency seed telemetry to the immutable genetic code of blueprints and their compiled artifacts. A robust data architecture is therefore essential for ensuring system stability, auditability, and the effectiveness of the learning-based agents. This chapter defines the core data models, flow patterns, and storage strategies for the `Esper` reference implementation, updated to reflect the asynchronous compilation and validation pipeline.

## 8.0 Protocol Governance Framework

**This section implements the C-016 External Review requirements for preventing protocol drift and ensuring system-wide compatibility across all 16 subsystems.**

### 8.0.1 Core Governance Principles

The Esper platform implements comprehensive protocol governance to make schema drift impossible, not just unlikely. This framework ensures byte-for-byte compatibility across all subsystems while supporting controlled evolution.

**Governance Invariants:**
- **I1**: Protocol changes require explicit version bump
- **I2**: All subsystems use generated stubs only
- **I3**: Wire format compatibility is tested continuously
- **I4**: No manual serialization/deserialization code
- **I5**: No map<> fields in cross-plane messages (prevents non-deterministic serialization)
- **I6**: All messages must pass decode-reencode validation

### 8.0.2 Containerized Protocol Toolchain

**Single Source of Truth**: All protocol definitions exist in versioned `.proto` files with exact toolchain versions and SHA256 digests for reproducibility.

#### Directory Structure

```
/home/john/esper/
├── shared/
│   ├── contracts/                    # Single source of protocol truth
│   │   ├── v1/                      # Version 1 protocols (frozen)
│   │   │   ├── system_state.proto
│   │   │   ├── adaptation.proto
│   │   │   └── telemetry.proto
│   │   ├── v2/                      # Version 2 protocols (current)
│   │   │   ├── system_state.proto
│   │   │   ├── adaptation.proto
│   │   │   ├── telemetry.proto
│   │   │   ├── rollback.proto
│   │   │   └── pause.proto
│   │   ├── build/                   # Build configuration
│   │   │   ├── protobuf.yaml       # Proto compilation config
│   │   │   ├── versions.yaml       # Version manifest
│   │   │   └── toolchain.yaml      # Containerized toolchain spec
│   │   └── golden/                  # Golden test fixtures
│   │       ├── v1/
│   │       └── v2/
│   │
│   ├── generated/                   # Auto-generated code (CI only)
│   │   ├── python/
│   │   │   ├── esper_protocols_v1/
│   │   │   └── esper_protocols_v2/
│   │   └── go/                     # Foreign target for validation
│   └── tests/
│       ├── compatibility/          # Wire format tests
│       ├── golden/                 # Golden test runner
│       ├── decode_reencode/        # Decode-reencode validation
│       └── cross_language/         # Multi-language validation
```

#### Containerized Toolchain Configuration

**File**: `shared/contracts/build/toolchain.yaml`

```yaml
# Protocol toolchain with exact versions and digests
toolchain:
  name: "esper-protocol-toolchain"
  version: "2025.1.0"
  
containers:
  protoc:
    image: "protobuf/protoc:3.21.12"
    digest: "sha256:a1b2c3d4e5f6..."
    command: ["protoc", "--version"]
    
  python_generator:
    image: "grpcio/protoc-gen-python:4.24.4"
    digest: "sha256:f6e5d4c3b2a1..."
    
  go_generator:
    image: "golang/protoc-gen-go:1.31.0"
    digest: "sha256:1a2b3c4d5e6f..."

# Forbidden features in cross-plane messages
forbidden_features:
  - "map fields"              # Causes non-deterministic serialization
  - "float counters"          # Precision issues across languages
  - "oneof spanning planes"   # Cross-boundary coupling

# Required validations
validations:
  - "golden_bytes_match"
  - "decode_reencode_identity"
  - "cross_language_compatibility"
  - "no_manual_serialization"
```

#### Dockerfile for Reproducible Builds

```dockerfile
# shared/contracts/build/Dockerfile
FROM protobuf/protoc:3.21.12@sha256:a1b2c3d4e5f6... AS protoc
FROM grpcio/protoc-gen-python:4.24.4@sha256:f6e5d4c3b2a1... AS python-gen
FROM golang/protoc-gen-go:1.31.0@sha256:1a2b3c4d5e6f... AS go-gen

# Multi-stage build ensures exact toolchain versions
FROM ubuntu:22.04

COPY --from=protoc /usr/bin/protoc /usr/bin/protoc
COPY --from=python-gen /usr/bin/protoc-gen-python /usr/bin/protoc-gen-python
COPY --from=go-gen /usr/bin/protoc-gen-go /usr/bin/protoc-gen-go

# Install Python protobuf runtime (exact version)
RUN pip install protobuf==4.24.4 grpcio==1.58.0

# Install Go protobuf runtime (exact version)
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.31.0

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/protoc"]
```

### 8.0.3 Protocol Buffer v2 Definitions

All message definitions use Protocol Buffer v2 specifications with enhanced fields and strict compatibility rules.

#### SystemStatePacket v2

```protobuf
syntax = "proto3";
package esper.system_state.v2;

import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";

message SystemStatePacket {
    // Metadata
    string packet_id = 1;  // UUID for correlation
    int64 epoch = 2;
    google.protobuf.Timestamp timestamp = 3;
    string source_subsystem = 4;
    
    // Training state
    TrainingMetrics training_metrics = 10;
    ModelState model_state = 11;
    
    // Seed states (NO MAP FIELDS - prevents non-deterministic serialization)
    repeated SeedState seed_states = 20;
    
    // System health
    SystemHealth system_health = 30;
    
    // Reserved for future use
    reserved 40 to 49;
    reserved "deprecated_field";
}

message TrainingMetrics {
    double loss = 1;
    double learning_rate = 2;
    int64 global_step = 3;
    google.protobuf.Duration epoch_duration = 4;  // Use Duration helpers
    double grad_norm = 5;
    
    // Performance metrics
    double throughput_samples_per_sec = 10;
    double gpu_utilization = 11;
    int64 memory_used_bytes = 12;
}

message ModelState {
    string model_version = 1;
    int64 parameter_count = 2;
    repeated LayerInfo layers = 3;
    string checkpoint_path = 4;
}

message LayerInfo {
    string name = 1;
    string type = 2;
    int64 param_count = 3;
    bool frozen = 4;
}

message SeedState {
    int32 seed_id = 1;
    string seed_type = 2;
    bool active = 3;
    double local_loss = 4;
    SeedHealth health = 5;
}

message SeedHealth {
    enum Status {
        UNKNOWN = 0;
        HEALTHY = 1;
        DEGRADED = 2;
        CRITICAL = 3;
    }
    Status status = 1;
    double gradient_variance = 2;
    double weight_drift = 3;
    int32 consecutive_failures = 4;
}

message SystemHealth {
    bool training_stable = 1;
    bool memory_pressure = 2;
    repeated string warnings = 3;
    // NO MAP FIELDS - use repeated key-value pairs instead
    repeated ResourceUsage resource_usage = 4;
}

message ResourceUsage {
    string resource_name = 1;
    double usage_value = 2;
}
```

#### AdaptationCommand v2

```protobuf
syntax = "proto3";
package esper.adaptation.v2;

import "google/protobuf/timestamp.proto";

message AdaptationCommand {
    // Command metadata
    string command_id = 1;       // UUID for idempotency
    int64 target_epoch = 2;       // Epoch to apply at
    google.protobuf.Timestamp created_at = 3;
    int32 ttl_epochs = 4;         // Auto-expire after N epochs
    
    // Command payload
    oneof command {
        LearningRateUpdate lr_update = 10;
        SeedActivation seed_activation = 11;
        ArchitectureChange arch_change = 12;
        RollbackCommand rollback = 13;
        PauseCommand pause = 14;
    }
    
    // Execution constraints
    ExecutionConstraints constraints = 20;
    
    // Reserved for extensions
    reserved 30 to 39;
}

message LearningRateUpdate {
    // NO MAP FIELDS - use repeated key-value pairs
    repeated LearningRateGroup group_rates = 1;
    string policy_name = 2;
    bool apply_warmup = 3;
}

message LearningRateGroup {
    string param_group = 1;
    double learning_rate = 2;
}

message SeedActivation {
    repeated int32 activate_seeds = 1;
    repeated int32 deactivate_seeds = 2;
    string activation_strategy = 3;
}

message ArchitectureChange {
    string change_type = 1;
    bytes serialized_config = 2;  // Pickled config for now
    string validation_hash = 3;
}

message RollbackCommand {
    enum RollbackType {
        FAST = 0;   // Memory-only
        FULL = 1;   // From checkpoint
    }
    RollbackType type = 1;
    string checkpoint_id = 2;
    string reason = 3;
}

message PauseCommand {
    bool pause = 1;
    int32 max_duration_ms = 2;
    string reason = 3;
    bytes pause_token = 4;  // Signed token
}

message ExecutionConstraints {
    int32 max_latency_ms = 1;
    bool require_confirmation = 2;
    repeated string required_subsystems = 3;
}
```

### 8.0.4 Protocol Duration Helpers

**Standardized Duration Handling**: All time values use millisecond standardization to prevent unit confusion across subsystems.

```python
from google.protobuf import duration_pb2
from typing import Union

class ProtocolDuration:
    """Standardized duration handling - always use milliseconds internally"""
    
    @staticmethod
    def from_ms(ms: int) -> duration_pb2.Duration:
        """Convert milliseconds to protobuf Duration"""
        d = duration_pb2.Duration()
        d.seconds, rem = divmod(int(ms), 1000)
        d.nanos = (rem * 1_000_000)
        return d
        
    @staticmethod
    def to_ms(d: duration_pb2.Duration) -> int:
        """Convert protobuf Duration to milliseconds"""
        return d.seconds * 1000 + d.nanos // 1_000_000
        
    @staticmethod
    def from_seconds(s: Union[int, float]) -> duration_pb2.Duration:
        """Convert seconds to protobuf Duration"""
        return ProtocolDuration.from_ms(int(s * 1000))

# Usage example:
# packet.training_metrics.epoch_duration = ProtocolDuration.from_ms(18)
# timeout_ms = ProtocolDuration.to_ms(command.constraints.timeout)
```

### 8.0.5 Forbidden Features Enforcement

**Linting Rules**: CI pipeline enforces forbidden patterns to prevent protocol drift.

```python
# scripts/validate_forbidden_features.py

import re
from pathlib import Path
from typing import List

class ForbiddenFeatureChecker:
    """Detect and prevent forbidden protocol features"""
    
    FORBIDDEN_PATTERNS = {
        'map_fields': r'map<[^>]+>',
        'float_counters': r'float\s+\w*count\w*',
        'manual_serialization': r'(SerializeToString|ParseFromString)',
    }
    
    def check_proto_files(self, proto_dir: Path) -> List[str]:
        """Check .proto files for forbidden patterns"""
        violations = []
        
        for proto_file in proto_dir.glob("*.proto"):
            content = proto_file.read_text()
            
            for pattern_name, pattern in self.FORBIDDEN_PATTERNS.items():
                if re.search(pattern, content):
                    violations.append(
                        f"{proto_file}: {pattern_name} forbidden in cross-plane messages"
                    )
        
        return violations
```

### 8.0.6 Decode-Reencode Validation Framework

**Wire Format Stability**: Ensures messages survive serialization round-trip without corruption.

```python
# shared/tests/decode_reencode/test_wire_stability.py

import hashlib
from pathlib import Path
from typing import Any

class DecodeReencodeValidator:
    """Ensures decode-reencode identity: golden == encode(decode(golden))"""
    
    def test_system_state_packet(self):
        """Test SystemStatePacket decode-reencode identity"""
        # Load golden fixture
        golden_path = Path("shared/contracts/golden/v2/system_state.bin")
        with open(golden_path, "rb") as f:
            golden_bytes = f.read()
        
        # Decode and re-encode
        packet = SystemStatePacket()
        packet.ParseFromString(golden_bytes)
        reencoded = packet.SerializeToString()
        
        # Verify identity
        assert reencoded == golden_bytes, \
            f"Decode-reencode identity violation! " \
            f"Original SHA256: {hashlib.sha256(golden_bytes).hexdigest()}, " \
            f"Reencoded SHA256: {hashlib.sha256(reencoded).hexdigest()}"
    
    def test_adaptation_command(self):
        """Test AdaptationCommand decode-reencode identity"""
        fixtures = [
            "adaptation_lr_update.bin",
            "adaptation_seed_activation.bin", 
            "adaptation_rollback.bin",
            "adaptation_pause.bin"
        ]
        
        for fixture in fixtures:
            golden_path = Path(f"shared/contracts/golden/v2/{fixture}")
            with open(golden_path, "rb") as f:
                golden_bytes = f.read()
            
            command = AdaptationCommand()
            command.ParseFromString(golden_bytes)
            reencoded = command.SerializeToString()
            
            assert reencoded == golden_bytes, \
                f"Decode-reencode failure for {fixture}"
```

### 8.0.7 Golden Test Framework

**Byte-for-byte Compatibility**: Golden tests ensure wire format never changes unexpectedly.

```python
# shared/tests/golden/test_golden_wire.py

class GoldenWireTest:
    """Ensures byte-for-byte compatibility of protocol messages"""
    
    def __init__(self, version: str):
        self.version = version
        self.golden_dir = Path(f"shared/contracts/golden/{version}")
        
    def test_system_state_packet(self):
        """Test SystemStatePacket wire format stability"""
        # Load golden fixture
        with open(self.golden_dir / "system_state.bin", "rb") as f:
            golden_bytes = f.read()
        
        # Create equivalent message
        packet = create_test_system_state()
        
        # Test 1: Byte-for-byte compatibility
        serialized = packet.SerializeToString()
        assert serialized == golden_bytes, \
            f"Wire format changed! SHA256: {hashlib.sha256(serialized).hexdigest()}"
            
        # Test 2: Decode-reencode identity
        parsed = SystemStatePacket()
        parsed.ParseFromString(golden_bytes)
        reencoded = parsed.SerializeToString()
        assert reencoded == golden_bytes, \
            "Decode-reencode identity violation!"
            
        # Test 3: Cross-language compatibility
        self._verify_cross_language_compatibility(golden_bytes)
    
    def generate_golden_fixtures(self):
        """Generate new golden fixtures (use with extreme caution)"""
        if self.golden_dir.exists():
            raise RuntimeError(
                "Golden fixtures already exist! Use update_golden() instead"
            )
        
        self.golden_dir.mkdir(parents=True)
        
        # Generate all fixture types
        fixtures = {
            "system_state.bin": create_test_system_state(),
            "adaptation_lr_update.bin": create_lr_update_command(),
            "adaptation_seed_activation.bin": create_seed_activation_command(),
            "telemetry_emergency.bin": create_emergency_telemetry(),
        }
        
        for name, message in fixtures.items():
            with open(self.golden_dir / name, "wb") as f:
                f.write(message.SerializeToString())
```

### 8.0.8 CI/CD Integration

**Continuous Protocol Governance**: CI pipeline enforces all protocol rules automatically.

```yaml
# .github/workflows/protocol-governance.yml

name: Protocol Governance

on:
  pull_request:
    paths:
      - 'shared/contracts/**/*.proto'
      - 'shared/tests/**'

jobs:
  validate-protocols:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check for version bump
        run: |
          # Ensure version is incremented if protos changed
          python scripts/check_version_bump.py
      
      - name: Load containerized toolchain
        run: |
          docker pull protobuf/protoc:3.21.12@sha256:a1b2c3d4e5f6...
          docker pull grpcio/protoc-gen-python:4.24.4@sha256:f6e5d4c3b2a1...
      
      - name: Generate Python stubs (containerized)
        run: |
          docker run --rm -v $(pwd):/workspace \
            protobuf/protoc:3.21.12@sha256:a1b2c3d4e5f6... \
            --python_out=/workspace/shared/generated/python \
            --proto_path=/workspace/shared/contracts/v2 \
            /workspace/shared/contracts/v2/*.proto
      
      - name: Generate Go stubs (containerized)
        run: |
          docker run --rm -v $(pwd):/workspace \
            golang/protoc-gen-go:1.31.0@sha256:1a2b3c4d5e6f... \
            --go_out=/workspace/shared/generated/go \
            --proto_path=/workspace/shared/contracts/v2 \
            /workspace/shared/contracts/v2/*.proto
      
      - name: Run golden tests
        run: |
          pytest shared/tests/golden/ -v
      
      - name: Run decode-reencode tests
        run: |
          pytest shared/tests/decode_reencode/ -v
      
      - name: Check for forbidden features
        run: |
          # No map<> fields in cross-plane messages
          ! grep -r "map<" shared/contracts/v2/ --include="*.proto"
          # No float counters
          ! grep -r "float.*count" shared/contracts/v2/ --include="*.proto"
          
      - name: Check for manual serialization
        run: |
          # Fail if any manual proto serialization found
          ! grep -r "SerializeToString\|ParseFromString" src/ \
            --include="*.py" \
            --exclude-dir="generated" \
            --exclude-dir="tests"
      
      - name: Run cross-language tests
        run: |
          pytest shared/tests/cross_language/ -v
      
      - name: Validate compatibility rules
        run: |
          python scripts/validate_compatibility.py
```

### 8.1 Data Models (Contracts)

To ensure stable, versioned communication between subsystems, the framework relies on a set of well-defined data contracts. These are specified below using Protocol Buffer v2 schemas for their cross-language compatibility and runtime validation capabilities. The central change is the evolution of `Urza` into an asset hub that manages both the abstract design (`Blueprint`) and its concrete, executable instances (`CompiledKernelArtifact`).

#### 8.1.1 The Asset Hierarchy: Blueprints & Kernels

The architecture now distinguishes between a blueprint's abstract design (its Intermediate Representation or IR) and the multiple compiled kernel artifacts generated from it.

```protobuf
# shared/contracts/v2/blueprint.proto
syntax = "proto3";
package esper.blueprint.v2;

import "google/protobuf/timestamp.proto";

message Blueprint {
    // Core identification
    string id = 1;                          // The canonical hash of the Blueprint's IR
    BlueprintStatus status = 2;
    string architecture_ref = 3;            // Reference to stored IR
    
    // Metadata
    google.protobuf.Timestamp created_at = 4;
    string creator_subsystem = 5;
    repeated MetadataEntry metadata = 6;    // NO MAP FIELDS - use repeated pairs
    
    // Compiled artifacts
    repeated CompiledKernelArtifact compiled_artifacts = 10;
}

message MetadataEntry {
    string key = 1;
    string value = 2;
}

enum BlueprintStatus {
    UNVALIDATED = 0;
    COMPILING = 1;
    INVALID = 2;
}

message CompiledKernelArtifact {
    string id = 1;                          // Unique kernel identifier: sha256:xyz-pipeline_A
    KernelStatus status = 2;
    string compilation_pipeline = 3;        // Which Tezzeret pipeline generated this
    string kernel_binary_ref = 4;           // Reference to stored kernel binary
    
    // Compilation details
    CompilationRecord compilation_metadata = 10;
    UrabraskValidationReport validation_report = 11;
}

enum KernelStatus {
    PENDING_BENCHMARKING = 0;
    VALIDATED = 1;
    REJECTED = 2;
}

message CompilationRecord {
    string pipeline = 1;
    int32 compilation_duration_ms = 2;      // Standardized to milliseconds
    double peak_memory_gb = 3;
    string toolchain_version = 4;
    repeated string optimization_flags = 5;
}

message UrabraskValidationReport {
    // Empirical measurements
    repeated BenchmarkResult benchmark_results = 1;    // NO MAP FIELDS
    repeated string generated_tags = 2;
    repeated BaselineComparison baseline_comparisons = 3;
    bool passed_validation = 4;
    
    // Hardware context
    repeated HardwareAttribute hardware_context = 5;
    google.protobuf.Timestamp report_timestamp = 6;
}

message BenchmarkResult {
    string metric_name = 1;
    double metric_value = 2;
}

message BaselineComparison {
    string baseline_name = 1;
    double comparison_value = 2;
}

message HardwareAttribute {
    string attribute_name = 1;
    string attribute_value = 2;
}
```

#### 8.1.2 Core Operational Entities

These models define the information passed between components during runtime operations, using Protocol Buffer v2 specifications.

```protobuf
# shared/contracts/v2/operational.proto
syntax = "proto3";
package esper.operational.v2;

import "google/protobuf/timestamp.proto";

message HardwareContext {
    string device_type = 1;             // "GPU", "TPU", etc.
    double memory_available_gb = 2;
    bool supports_quantization = 3;
    repeated HardwareCapability capabilities = 4;  // NO MAP FIELDS
}

message HardwareCapability {
    string capability_name = 1;
    string capability_value = 2;
}

message AdaptationSignal {
    string signal_id = 1;                   // UUID for correlation
    bool optimizer_rebuild_required = 2;
    bool emergency_rollback_required = 3;
    google.protobuf.Timestamp created_at = 4;
    int32 ttl_epochs = 5;
}

message FieldReport {
    string report_id = 1;                   // UUID for correlation
    string blueprint_id = 2;
    string kernel_artifact_id = 3;         // Specify which variant was used
    string seed_id = 4;
    
    // Performance metrics
    double pre_metric = 10;
    double post_metric = 11;
    string outcome = 12;                    // 'FOSSILIZED', 'CULLED', 'ROLLED_BACK'
    
    // Evaluation details
    repeated EvaluationMetric eval_metrics = 20;
    google.protobuf.Timestamp timestamp = 21;
}

message EvaluationMetric {
    string metric_name = 1;
    double metric_value = 2;
}

message KernelQuery {
    string query_id = 1;                    // UUID for correlation
    string blueprint_id = 2;               // Target blueprint hash
    
    // Tag filters
    repeated string required_tags = 10;
    repeated string preferred_tags = 11;
    repeated string excluded_tags = 12;
    
    // Performance constraints
    int32 max_latency_ms = 20;              // Standardized to milliseconds
    int32 max_memory_mb = 21;
    string target_gpu_arch = 22;
    
    google.protobuf.Timestamp created_at = 30;
}

message KernelQueryResult {
    string query_id = 1;                    // Correlation with request
    repeated CompiledKernelArtifact matching_kernels = 2;
    bool query_successful = 3;
    string error_message = 4;               // If query_successful = false
    google.protobuf.Timestamp timestamp = 5;
}
```

### 8.2 Data Management & Flow Patterns

The architecture's data flows are built on modern, event-driven patterns to ensure scalability and auditability, with all messages using Protocol Buffer v2 schemas.

- **Event Sourcing:** The `Oona` message bus functions as the primary event log. Every significant state change in the system—a seed reporting telemetry, a controller issuing a command, a `Blueprint` being submitted, or a `KernelArtifact` being validated—is published as an immutable event using Protocol Buffer serialization. This creates a complete, replayable history of the system.

- **Command Query Responsibility Segregation (CQRS):** The system naturally separates state-changing operations (**Commands**) from data-reading operations (**Queries**).

  - **Commands:** Are lightweight, asynchronous messages (e.g., `AdaptationCommand`) published to specific topics on the bus using Protocol Buffer serialization. They are designed to be processed by a single consumer.
  - **Queries:** Are performed by components requesting data. `Tamiyo` performs a `KernelQuery` against the `Urza` API using Protocol Buffer messages. `Nissa` queries the event stream from `Oona` to build observability dashboards.

- **Data Lifecycle & Retention:**

  - **Blueprints & Kernels (Urza):** Retained indefinitely. Both designs and their compiled artifacts are versioned and never hard-deleted; underperforming ones are moved to an "archived" state.
  - **Operational Events (Oona):** High-frequency telemetry is retained on the bus for a short period (e.g., 72 hours). Critical events like `FieldReports` and compilation status updates are consumed and persisted indefinitely by the relevant subsystems.
  - **Checkpoints (Tolaria):** A rolling window of the last N known-good checkpoints is maintained (e.g., N=5).

### 8.3 Data Storage Architecture with WAL Durability

The reference architecture uses a polyglot persistence strategy, choosing the best storage technology for each type of data, with enhanced WAL durability semantics from C-016.

| Data Type                        | Storage System                    | Phase 1 Implementation                | Phase 2 Implementation                      | Rationale                                                                                                                   |
| :------------------------------- | :-------------------------------- | :------------------------------------ | :------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------- |
| **Blueprint & Kernel Artifacts** | Object Storage                    | Local Filesystem / MinIO              | AWS S3, GCS, or other cloud object store    | Ideal for storing large, immutable binary files with O_DSYNC and fsync durability requirements. |
| **Blueprint & Kernel Metadata** | Relational Database               | SQLite / PostgreSQL                   | Managed PostgreSQL (e.g., AWS RDS)          | Supports complex queries, transactions, and relational integrity with WAL durability. |
| **Time-Series Metrics** | Time-Series Database (TSDB)       | Prometheus                            | Managed Prometheus / VictoriaMetrics        | Optimized for high-throughput ingestion with millisecond timestamp standardization.        |
| **Event Stream / Messages** | Message Broker                    | Redis Streams                         | Apache Pulsar                               | Provides real-time, persistent, pub/sub backbone with Protocol Buffer message validation.  |
| **RL Replay Buffer** | Hybrid In-Memory + Data Lake      | In-memory Python `deque`              | Redis (Hot Storage) + Object Store (Cold)  | Balances fast sampling with WAL durability for critical replay data. |
| **System Checkpoints** | Object Storage with WAL          | Local Filesystem with WAL             | AWS S3, GCS with WAL durability    | Enhanced WAL with O_DSYNC, fsync barriers, and Merkle tree integrity.       |

#### WAL Durability Specifications

**Enhanced Write-Ahead Logging**: All critical storage operations use WAL with O_DSYNC and fsync barriers for durability.

```python
# src/esper/common/wal_writer.py

import os
import struct
import hashlib
from typing import Dict, Any
from pathlib import Path

class WALWriter:
    """Enhanced WAL writer with C-016 durability semantics"""
    
    # 256-byte WAL header with endian markers and CRC
    WAL_HEADER_SIZE = 256
    WAL_MAGIC = b'ESPR'
    WAL_VERSION = 2
    
    def __init__(self, wal_path: Path):
        self.wal_path = wal_path
        self.wal_fd = None
        
    def open_wal(self) -> None:
        """Open WAL with O_DSYNC for immediate durability"""
        # O_DSYNC ensures data reaches storage before write() returns
        self.wal_fd = os.open(
            str(self.wal_path), 
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_DSYNC
        )
        
    def write_entry(self, entry_type: str, data: bytes) -> None:
        """Write WAL entry with 256-byte header and CRC"""
        if not self.wal_fd:
            raise RuntimeError("WAL not opened")
            
        # Create 256-byte header
        header = bytearray(self.WAL_HEADER_SIZE)
        
        # Header structure:
        # [0:4]   Magic number 'ESPR'
        # [4:8]   Version (2)
        # [8:16]  Timestamp (8 bytes)
        # [16:20] Entry type length
        # [20:52] Entry type (32 bytes max)
        # [52:60] Data length (8 bytes)
        # [60:92] CRC32 of data (32 bytes)
        # [92:100] Endian marker (8 bytes)
        # [100:256] Reserved/padding
        
        struct.pack_into('<4sI', header, 0, self.WAL_MAGIC, self.WAL_VERSION)
        struct.pack_into('<Q', header, 8, int(time.time() * 1000))  # milliseconds
        
        entry_type_bytes = entry_type.encode('utf-8')[:32]
        struct.pack_into('<I', header, 16, len(entry_type_bytes))
        struct.pack_into(f'<{len(entry_type_bytes)}s', header, 20, entry_type_bytes)
        
        struct.pack_into('<Q', header, 52, len(data))
        
        # CRC32 for integrity
        crc = hashlib.crc32(data)
        struct.pack_into('<I', header, 60, crc)
        
        # Endian marker for cross-platform compatibility
        struct.pack_into('<Q', header, 92, 0x1234567890ABCDEF)
        
        # Write header + data atomically
        os.write(self.wal_fd, header)
        os.write(self.wal_fd, data)
        
        # Explicit fsync for durability guarantee
        os.fsync(self.wal_fd)
        
    def close_wal(self) -> None:
        """Close WAL with final fsync"""
        if self.wal_fd:
            os.fsync(self.wal_fd)
            os.close(self.wal_fd)
            self.wal_fd = None
```

#### Merkle Tree Integration

**Enhanced Checkpoint Integrity**: Checkpoints include Merkle tree metadata for tamper detection.

```python
# src/esper/common/merkle_checkpoint.py

import hashlib
from typing import List, Optional

class MerkleCheckpoint:
    """Checkpoint with Merkle tree integrity verification"""
    
    def __init__(self):
        self.leaves: List[bytes] = []
        self.root_hash: Optional[bytes] = None
        
    def add_component(self, component_name: str, component_data: bytes) -> None:
        """Add component to Merkle tree"""
        # Hash component with name for uniqueness
        leaf_data = f"{component_name}:".encode() + component_data
        leaf_hash = hashlib.sha256(leaf_data).digest()
        self.leaves.append(leaf_hash)
        
    def compute_root(self, execution_context: Dict[str, Any]) -> bytes:
        """Compute Merkle root with execution context"""
        if not self.leaves:
            raise ValueError("No components added to checkpoint")
            
        # Include execution context in root calculation
        context_str = "|".join(f"{k}={v}" for k, v in sorted(execution_context.items()))
        context_hash = hashlib.sha256(context_str.encode()).digest()
        
        # Build Merkle tree bottom-up
        current_level = self.leaves + [context_hash]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(left + right).digest()
                next_level.append(combined)
            current_level = next_level
            
        self.root_hash = current_level[0]
        return self.root_hash
        
    def verify_integrity(self, expected_root: bytes) -> bool:
        """Verify checkpoint integrity against expected root"""
        return self.root_hash == expected_root
```

### 8.4 Protocol Version Management

**Semantic Versioning for Protocols**: All Protocol Buffer schemas follow strict semantic versioning rules.

#### Version Manifest

```yaml
# shared/contracts/build/versions.yaml
versions:
  v1:
    status: frozen
    released: 2025-01-01
    deprecated: 2025-01-10
    packages:
      - esper.system_state.v1
      - esper.adaptation.v1
      - esper.telemetry.v1
  
  v2:
    status: active
    released: 2025-01-10
    packages:
      - esper.system_state.v2
      - esper.adaptation.v2
      - esper.telemetry.v2
      - esper.rollback.v2
      - esper.pause.v2
    
migration:
  v1_to_v2:
    tool: scripts/migrate_v1_to_v2.py
    compatibility: read_v1_write_v2
```

#### Compatibility Rules

**ALLOWED Changes (Minor Version Bump)**:
- Add new optional fields (with new field numbers)
- Add new message types
- Add new enum values (at the end)
- Add new RPC methods

**FORBIDDEN Changes (Require Major Version)**:
- Remove fields (mark as `reserved` instead)
- Change field numbers
- Change field types
- Rename fields (use `reserved` and add new)
- Reorder enum values
- Add map<> fields in cross-plane messages

## 9. Integration Architecture

A key strength of the Morphogenetic Framework is its modular, decoupled design enhanced by comprehensive protocol governance. This chapter specifies how the eleven subsystems integrate internally and how the platform interfaces with external systems.

### 9.1 Internal Integration: The Protocol-Governed Backbone

All internal communication between the major planes (Training, Control, Innovation) is mediated by the `Oona` message bus using Protocol Buffer v2 messages with strict schema validation.

- **Mechanism:** The system uses a "Fat Envelope, Thin Payload" approach. Every message published to the bus is wrapped in a standard Protocol Buffer envelope containing rich metadata, while the payload contains the specific data model for that event.

- **Key Topics and Payloads:** The following table defines the primary communication channels on the message bus:

| Topic | Publisher | Consumer(s) | Payload Data Model | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `telemetry.seed.health` | `Kasmina` | `Tamiyo`, `Nissa` | `SeedTelemetry` (PB v2) | High-frequency reporting using Protocol Buffer serialization |
| `control.kasmina.commands` | `Tamiyo` | `Kasmina` | `AdaptationCommand` (PB v2) | Command to load and execute specific kernel artifacts |
| `innovation.field_reports` | `Tamiyo` | `Karn`, `Simic` | `FieldReport` (PB v2) | Real-world adaptation outcomes |
| `compilation.blueprint.submitted` | `Karn` / `Urza` | `Tezzeret` | `BlueprintSubmittedEvent` (PB v2) | New blueprint available for compilation |
| `compilation.kernel.ready` | `Tezzeret` / `Urza`| `Urabrask` | `KernelReadyEvent` (PB v2) | New kernel artifact ready for benchmarking |
| `validation.kernel.characterized`| `Urabrask` / `Urza`| `Tamiyo`, `Nissa`, `Karn` | `KernelCharacterizedEvent` (PB v2) | Validated kernel with performance profile |
| `system.events.epoch` | `Tolaria` | `Tamiyo`, `Nissa` | `EpochEvent` (PB v2) | Master heartbeat signal for epoch boundaries |

### 9.2 External Integration: The MLOps Ecosystem

The `Esper` platform integrates with the larger MLOps ecosystem through Protocol Buffer APIs and standardized interfaces.

- **Model Ingestion & Export:**

  - **Ingestion:** `Esper` consumes a standard, pre-trained PyTorch `nn.Module`. The injection of `Kasmina` execution layers is handled by a helper utility.
  - **Export:** The final, evolved model is exported with complete lineage including exact `kernel_artifact_id` for each adaptation, ensuring bit-for-bit reproducibility.

- **Monitoring & Observability (`Nissa`):**

  - The `Nissa` subsystem exposes all platform metrics via Prometheus `/metrics` endpoint with standardized millisecond timestamps and Protocol Buffer telemetry aggregation.

- **CI/CD Pipeline Integration:**

  - Protocol governance is enforced through GitHub Actions workflows that validate schema changes, generate stubs, and run comprehensive compatibility tests.

### 9.3 Primary API & SDK Specification

This section defines the primary interfaces for interacting with `Esper`, all using Protocol Buffer v2 message formats.

#### 9.3.1 Researcher-Facing Python SDK

The primary interaction model uses Protocol Buffer messages throughout:

```python
import esper
from esper_protocols_v2.system_state_pb2 import SystemStatePacket
from esper_protocols_v2.adaptation_pb2 import AdaptationCommand

# Load configurations using Protocol Buffer schemas
config = esper.load_config("./configs/phase1_mnist.yaml")

# Instantiate components with protocol validation
model = MySimpleCNN()
morphable_model = esper.wrap(model, config.kasmina)

tamiyo_controller = esper.TamiyoController(morphable_model, config.tamiyo)
tolaria_trainer = esper.TolariaTrainer(morphable_model, tamiyo_controller, config.tolaria)

# Start training with protocol-governed communication
final_model, training_history = tolaria_trainer.train()

# Save with complete Protocol Buffer lineage
esper.save_model(final_model, "./outputs/evolved_cnn")
```

#### 9.3.2 Control & Query REST API

Key components expose REST APIs that accept and return Protocol Buffer messages:

- **`Urza`: Central Library API**

  - `POST /api/v1/kernels/query`: Accept `KernelQuery` Protocol Buffer, return `KernelQueryResult`
  - `GET /api/v1/blueprints/{blueprint_id}`: Return `Blueprint` Protocol Buffer with all compiled variants

- **`Tamiyo`: System Control API**

  - `GET /api/v1/system/status`: Return `SystemStatePacket` with current training state
  - `POST /api/v1/control/pause`: Accept `PauseCommand` Protocol Buffer for administrative control

## 10. Security Architecture

The security architecture is enhanced with Protocol Buffer message validation and cryptographic verification of all data contracts.

### 10.1 Protocol Security

- **Message Integrity:** All Protocol Buffer messages include CRC32 checksums and are validated against golden test fixtures
- **Schema Validation:** Runtime guards ensure only generated Protocol Buffer stubs are used, preventing manual message crafting
- **Version Verification:** All messages include version metadata and are validated for compatibility

### 10.2 Data Security with Protocol Governance

- **Data in Transit:** All Protocol Buffer messages are transmitted over TLS 1.3 with additional message-level integrity checks
- **Data at Rest:** Protocol Buffer schemas are digitally signed and stored with SHA256 checksums for tamper detection
- **Schema Integrity:** Golden test failures immediately trigger security alerts for potential schema tampering

## 11. Implementation Timeline

### Phase 1: Protocol Governance Foundation (Day 1)
- Create directory structure for Protocol Buffer schemas
- Define v2 message definitions with no map<> fields
- Set up containerized toolchain with exact versions and SHA256 digests
- Implement duration helpers for millisecond standardization

### Phase 2: Testing Framework (Day 1-2)
- Create golden test fixtures for all message types
- Implement decode-reencode validation framework
- Set up cross-language compatibility tests
- Add CI pipeline with containerized toolchain

### Phase 3: Integration (Day 2)
- Update all subsystems to use generated Protocol Buffer stubs
- Remove manual serialization code
- Implement WAL durability with O_DSYNC and fsync barriers
- Add Merkle tree integration for checkpoints

### Phase 4: Enforcement (Day 2)
- Enable all CI gates for protocol governance
- Deploy runtime guards for message validation
- Complete documentation and team training
- Validate end-to-end system with Protocol Buffer v2

## Acceptance Criteria

### Technical Requirements
- [ ] All protocols defined in `.proto` files under `shared/contracts/`
- [ ] Python and Go stubs generate without errors using containerized toolchain
- [ ] Golden tests pass for all message types with decode-reencode validation
- [ ] Cross-language serialization validated
- [ ] CI gates prevent breaking changes and forbidden features
- [ ] No manual serialization code in codebase
- [ ] WAL durability with O_DSYNC and fsync implemented
- [ ] Duration helpers used for all time values (millisecond standardization)

### Protocol Governance Requirements
- [ ] No map<> fields in any cross-plane messages
- [ ] All message types have golden test coverage
- [ ] Decode-reencode tests pass for all message types
- [ ] Containerized toolchain with SHA256 digests deployed
- [ ] Version negotiation implemented and tested
- [ ] Runtime guards active and validated

This enhanced Data Contracts specification ensures the Esper platform has a rock-solid foundation for preventing protocol drift while supporting the morphogenetic training system's advanced capabilities. The Protocol Governance Framework makes schema inconsistencies impossible rather than just unlikely, providing the reliability needed for production deployment.