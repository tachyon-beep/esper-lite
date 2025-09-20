# Elesh Unified Design Document v3.0

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 3.0 - PRODUCTION READY WITH STRUCTURED PRUNING |
| **Status** | APPROVED |
| **Date** | 2025-01-14 |
| **Author** | C-020 Conclave (Structured Pruning Implementation) |
| **Component** | Innovation Plane - Structural Analysis Workers |
| **Parent** | High-Level Design (HLD) |
| **Subdocuments** | 13.1-elesh-importance-tracking.md, 13.2-elesh-analysis-algorithms.md |

## Executive Summary

Elesh provides structured pruning analysis execution for the Esper morphogenetic platform using the "measure twice, cut once" approach with checkpoint-based analysis. Following C-020 conclave consensus, this subsystem operates offline during checkpoint saves, enabling sophisticated structural analysis (channels, attention heads, layers) without runtime performance constraints.

Key characteristics:
- **Structured Pruning Analysis**: Channel importance calculation, attention head redundancy detection, layer pruning analysis
- **Checkpoint-Based Analysis Workers**: Offline analysis during checkpoint saves with unlimited compute budget
- **Zero Runtime Impact**: All analysis occurs during checkpoint saves with no training disruption
- **Enhanced Safety**: Progressive pruning schedules with comprehensive validation gates
- **Non-Blocking Failures**: System continues training without pruning if analysis fails

**C-020 INNOVATION**: Transforms structured pruning from complex runtime optimization to elegant checkpoint-based analysis using Taylor expansion importance metrics, attention pattern analysis, and skip connection evaluation.

## Core Architecture Decision

### **Structured Pruning Analysis with Checkpoint-Based Coordination**

- **Foundation**: Multi-level analysis workers (channel/head/layer) with offline importance calculation during checkpoint saves
- **Integration Model**: Receives analysis requests from Emrakul, performs structured analysis, returns pruning decisions to Emrakul for coordination with Kasmina
- **Authority Model**: Owns structural importance scoring and pruning recommendation generation during offline windows
- **Deployment Model**: Multiple analysis workers activated only during checkpoint operations with layer-group assignment

## 2. Core Responsibilities

### Enhanced Parameter Analysis (v3.0)
- **Structural analysis of neural network architectures** for channel, head, and layer-level pruning opportunities
- **Channel importance calculation using Taylor expansion** approximation: ΔL ≈ |∂L/∂a_c · a_c|
- **Attention head redundancy detection via cosine similarity** with entropy and pattern diversity analysis
- **Layer pruning recommendations based on gradient flow** and skip connection potential assessment
- **Progressive pruning schedule management** with safety thresholds and rollback coordination

### Legacy Parameter Analysis (Preserved)
- Checkpoint-based parameter analysis execution during checkpoint saves
- Enhanced memory efficiency through larger Count-Min Sketches offline
- Statistical validation with error bounds and confidence scoring
- Conservative mask generation with safety margins

## 3. Key Components

### 3.1 ImportanceTracker

**Enhanced Count-Min Sketch for Offline Analysis (v3.0)**
- Multi-metric scoring system combining Taylor expansion, magnitude, and activation frequency
- Adaptive phase-based weighting (early/mid/late training stages)
- EMA tracking with momentum=0.99 for stable importance estimates
- Dynamic threshold selection with architectural constraints

**Configuration**:
- Sketch width: 100,000 elements (10x larger than runtime for enhanced accuracy)
- Sketch depth: 10 levels (improved collision resistance)
- Historical importance tracking: 500 epochs for long-term trend analysis
- Enhanced statistical sampling: 10,000 samples for offline accuracy validation

### 3.2 StructuralAnalyzer (NEW in v3.0)

**Channel Analysis Engine**:
- Taylor expansion importance calculation with gradient × activation products
- Channel correlation analysis using cosine similarity (redundancy threshold: 0.95)
- Progressive removal schedules with polynomial/exponential/linear strategies
- Minimum channel retention constraints (16 channels minimum per layer)

**Attention Head Analysis Engine**:
- Attention pattern entropy calculation for head importance scoring
- Jensen-Shannon divergence for pattern diversity assessment
- Cosine similarity clustering for redundancy group identification
- Multi-factor scoring combining entropy, diversity, and gradient sensitivity

**Layer Analysis Engine**:
- Skip connection potential evaluation with dimension compatibility assessment
- Feature similarity analysis between layer inputs and outputs
- Gradient flow strength measurement for layer importance scoring
- Architecture modification feasibility with residual path establishment

### 3.3 Safety Validation Engine (Enhanced v3.0)
- Pre-pruning architecture consistency and gradient flow validation
- Post-pruning accuracy retention verification (95% minimum threshold)
- Go/No-Go automatic decision logic with comprehensive safety gates
- Progressive safety schedule: validation → conservative → moderate → aggressive phases

### 3.4 Legacy Components (Preserved)
- **Analysis Executor**: Offline analysis and logical mask generation
- **Telemetry Collector**: Analysis performance metrics aggregation
- **Offline Cache**: Historical importance data across checkpoint cycles

## 4. Message Contracts

### 4.1 Structured Pruning Request/Response (NEW v3.0)

**StructuralAnalysisRequest**: Emrakul → Elesh
```protobuf
message StructuralAnalysisRequest {
  string request_id = 1;
  string idempotency_key = 2;                // {epoch}:{model_hash}:{request_id}
  PrunedArchitecture current_architecture = 3;
  AnalysisParameters parameters = 4;
  PerformanceRequirements requirements = 5;
  google.protobuf.Duration timeout = 6;      // 2-5 minute checkpoint window
}
```

**StructuralAnalysisResponse**: Elesh → Emrakul
```protobuf
message StructuralPruningDecision {
  string idempotency_key = 1;                // Replay safety
  repeated ChannelPruningTarget channels = 10;
  repeated AttentionHeadTarget attention_heads = 11;
  repeated LayerPruningTarget layers = 12;
  PruningStatistics statistics = 15;
  SafetyValidation safety = 20;
  bool approved_for_application = 21;
  string rejection_reason = 22;              // Safety validation failures
}
```

### 4.2 Channel Pruning Data

**ChannelImportanceData**: Kasmina → Elesh
```protobuf
message ChannelImportanceData {
  string layer_name = 1;
  repeated float taylor_importance = 2;      // |∂L/∂a_c · a_c| per channel
  repeated float magnitude_importance = 3;   // Weight magnitude per channel
  repeated float activation_frequency = 4;   // Sparsity-aware activation rates
  torch.Tensor channel_correlations = 5;    // Channel similarity matrix
}
```

### 4.3 Attention Head Analysis

**AttentionHeadRedundancy**: Analysis output
```protobuf
message AttentionHeadTarget {
  string layer_name = 1;
  repeated uint32 head_indices = 2;          // Heads to remove
  repeated HeadRedundancy redundancy_pairs = 5;
  float redundancy_threshold = 6;            // 0.95 similarity threshold
  repeated float head_importance = 7;        // Combined importance scores
  repeated float attention_entropy = 8;      // Pattern entropy per head
}
```

### 4.4 Legacy Message Contracts (Preserved)
- **AnalysisRequest**: Emrakul → Elesh (Parameter analysis parameters and thresholds)
- **ImportanceData**: Kasmina → Elesh (Accumulated importance statistics at checkpoint)
- **PruningMasks**: Elesh → Emrakul (Generated parameter masks with confidence scores)
- **MaskApplication**: Emrakul → Kasmina (Validated masks for checkpoint embedding)
- **AnalysisReport**: Elesh → Emrakul (Analysis results and next-cycle recommendations)

### 4.5 Shared Contracts (Leyline)

This subsystem uses the following shared contracts from Leyline:
- `leyline.ParameterIdentifier` - Unique parameter identification
- `leyline.CheckpointMetadata` - Checkpoint coordination and metadata
- `leyline.AnalysisMetrics` - Analysis performance telemetry
- `leyline.StructuralPruningDecision` - Complete pruning decision with idempotency
- `leyline.PrunedArchitecture` - Architecture descriptors with pruning history

For complete contract definitions, see: `/docs/architecture/00-leyline-shared-contracts.md`

## Architectural Principles

### Non-Negotiable Requirements

1. **Gradient Isolation**: Must maintain ∇L_host ∩ ∇L_seed = ∅ invariant
2. **Checkpoint Boundaries**: All structural analysis must occur only at checkpoint boundaries
3. **Conservative Safety**: "Measure twice, cut once" with comprehensive offline validation
4. **Structural Integrity**: Channel/head/layer removals must preserve network connectivity

### Design Principles

1. **Offline Analysis**: Unlimited compute budget enables sophisticated multi-level importance calculations
2. **Natural Synchronization**: Checkpoint boundaries eliminate need for complex coordination protocols
3. **Statistical Accuracy**: Enhanced probabilistic data structures for improved accuracy
4. **Progressive Pruning**: Gradual removal with safety thresholds and rollback capability

### Production Safety Principles

1. **Non-Blocking Operation**: Training continues without pruning if analysis fails
2. **Comprehensive Validation**: Offline analysis enables complete safety verification
3. **Simple Rollback**: Checkpoint restoration replaces complex runtime rollback
4. **Multi-Level Safety**: Channel → head → layer analysis with escalating safety requirements

## Structured Pruning Implementation

### Phase 1: Validation Mode (Epochs 1-30)
- **Telemetry collection only** with no physical pruning applied
- **Importance score accumulation** for channels, heads, and layers
- **Statistical validation** of importance tracking algorithms
- **Safety threshold calibration** for production pruning phases

### Phase 2: Execution Mode (Epochs 31+)
- **Active structured pruning at checkpoints** with progressive schedules
- **Channel pruning**: 5-15% removal with Taylor expansion importance
- **Attention head pruning**: Redundancy-based removal with 0.95 similarity threshold
- **Layer pruning**: Conservative removal (max 20% layers) with skip connection establishment
- **Safety monitoring and automatic rollback** on 2% accuracy drops

### Progressive Safety Schedule

1. **Preparation (epochs 1-30)**: Analysis only, no modifications
2. **Conservative (epochs 31-60)**: 5% pruning maximum, channel-level only
3. **Moderate (epochs 61-90)**: 10% cumulative pruning, add attention heads
4. **Aggressive (epochs 91-120)**: 20% cumulative pruning, add layer removal
5. **Validation (epochs 121+)**: Monitoring only with stability verification

## System Components

### Component Overview

| Component | Purpose | Details |
|-----------|---------|---------|
| **Importance Tracker** | Multi-level importance calculation using offline analysis | See: 13.1-elesh-importance-tracking.md |
| **Structural Analyzer** | Channel, head, and layer analysis execution | See: 13.2-elesh-analysis-algorithms.md |
| **Safety Validator** | Progressive validation gates and rollback coordination | Enhanced v3.0 with structured pruning |
| **Telemetry Collector** | Aggregates and reports offline analysis metrics | Enhanced with structured pruning metrics |
| **Offline Cache** | Maintains historical importance data across checkpoint cycles | Enhanced with multi-level caching |

### Core Components Summary

**Importance Tracker** (Enhanced v3.0)
- Multi-metric importance calculation: Taylor expansion + magnitude + activation frequency
- Enhanced Count-Min Sketch for offline accuracy (width=100000, depth=10)
- Adaptive weighting based on training phase (early/mid/late)
- Details: 13.1#multi-level-tracking-algorithm

**Structural Analyzer** (NEW v3.0)
- Channel importance with correlation-based redundancy detection
- Attention head analysis using entropy, diversity, and gradient sensitivity
- Layer pruning evaluation with skip connection potential assessment
- Details: 13.2#structured-analysis-protocols

**Safety Validator** (Enhanced v3.0)
- Progressive validation gates with escalating safety requirements
- Multi-level rollback triggers (channel: 2%, head: 3%, layer: 5% accuracy drops)
- Architecture integrity validation with connectivity preservation
- Circuit breakers: 3 consecutive failures triggers pruning shutdown

## Integration Architecture

### Subsystem Dependencies

| Subsystem | Integration Type | Purpose |
|-----------|-----------------|---------|
| Emrakul | Command/Response | Receives analysis parameters and reports structured pruning decisions |
| Kasmina | Checkpoint Pipeline | Imports gradient statistics and applies generated pruning masks |
| Urabrask | Validation Pipeline | Coordinates safety validation for structural modifications |
| Tolaria | Indirect | Checkpoint boundary coordination and progressive phase management |
| Tezzeret | Coordination | Recompilation requests for structurally modified architectures |
| Nissa | Async/Telemetry | Analysis performance monitoring with structured pruning metrics |
| Oona | Async/Message Bus | Analysis coordination and structured pruning status reporting |

### Message Contracts Summary

| Contract | Direction | Purpose |
|----------|-----------|---------|
| StructuralAnalysisRequest | Emrakul → Elesh | Multi-level analysis parameters and performance requirements |
| StructuralPruningDecision | Elesh → Emrakul | Channel/head/layer removal recommendations with safety validation |
| ChannelImportanceData | Kasmina → Elesh | Taylor expansion importance and correlation matrices |
| AttentionHeadAnalysis | Internal | Head redundancy detection and pattern diversity analysis |
| LayerPruningAssessment | Internal | Skip connection evaluation and gradient flow analysis |
| StructuralValidationResult | Elesh → Urabrask | Safety validation results for structured modifications |

## Performance Targets

| Metric | Target | Measurement | Enhancement v3.0 |
|--------|--------|-------------|-------------------|
| Analysis Completion | Within 2-5 minute checkpoint budget | P95 analysis time | Multi-level analysis coordination |
| Channel Analysis Rate | 50K channels/sec | Offline processing throughput | Taylor expansion calculation |
| Head Analysis Rate | 1K heads/sec | Pattern analysis throughput | Entropy and diversity computation |
| Layer Analysis Rate | 100 layers/sec | Skip connection evaluation | Gradient flow assessment |
| Memory Usage During Analysis | <4GB per worker | Peak RSS during offline processing | Enhanced for multi-level analysis |
| Analysis Accuracy | >95% importance estimation | Validation against ground truth | Multi-metric scoring |

**IMPORTANT**: Enhanced structured pruning analysis provides **comprehensive architectural optimization recommendations** for channels, attention heads, and layers with safety validation.

## Configuration

### Key Configuration Parameters

```yaml
elesh:
  # Worker settings (offline)
  analysis_workers_per_gpu: 4    # Workers during checkpoint analysis
  layers_per_worker: 8           # Layers assigned to each worker

  # Enhanced importance tracking (offline) - v3.0
  sketch_width: 100000           # Larger Count-Min Sketch for accuracy (10x runtime)
  sketch_depth: 10               # Enhanced depth for reduced collisions
  importance_history_epochs: 500 # Longer history for better analysis

  # Structured Pruning Configuration - NEW v3.0
  structured_pruning:
    # Channel pruning
    channel_analysis:
      taylor_weight: 0.5         # Taylor expansion importance weight
      magnitude_weight: 0.3      # Weight magnitude importance weight
      frequency_weight: 0.2      # Activation frequency weight
      redundancy_threshold: 0.95 # Channel correlation threshold
      min_channels_per_layer: 16 # Minimum channels to preserve

    # Attention head pruning
    attention_analysis:
      entropy_weight: 0.3        # Attention entropy weight
      diversity_weight: 0.3      # Pattern diversity weight
      gradient_weight: 0.4       # Gradient sensitivity weight
      similarity_threshold: 0.95 # Head redundancy threshold
      min_heads_per_layer: 2     # Minimum heads to preserve

    # Layer pruning
    layer_analysis:
      taylor_threshold: 0.001    # Layer importance threshold
      similarity_threshold: 0.95 # Input/output similarity threshold
      skip_potential_threshold: 0.8  # Skip connection feasibility
      max_layer_removal_ratio: 0.2   # Maximum 20% layer removal

    # Progressive safety schedule
    safety_schedule:
      validation_epochs: 30      # Pure telemetry phase
      conservative_epochs: 30    # 5% pruning maximum
      moderate_epochs: 30        # 10% cumulative pruning
      aggressive_epochs: 30      # 20% cumulative pruning

    # Rollback thresholds
    rollback_triggers:
      channel_accuracy_drop: 0.02    # 2% accuracy drop for channels
      head_accuracy_drop: 0.03       # 3% accuracy drop for heads
      layer_accuracy_drop: 0.05      # 5% accuracy drop for layers

  # Legacy configuration (preserved)
  analysis_timeout: 300          # 5 minutes maximum per layer
  marking_threshold: 1e-8        # Stricter dead parameter threshold
  confidence_threshold: 0.95     # Required confidence for mask generation

  # Performance tuning (offline)
  analysis_cache_size: 1GB       # Larger cache for multi-level analysis
  parallel_analysis_threads: 16  # More threads for structured analysis
  statistical_samples: 10000     # Enhanced sampling for accuracy

  # Enhanced accuracy targets - v3.0
  accuracy_targets:
    channel_importance_ranking:
      metric: "kendall_tau"
      target: 0.90              # Improved channel ranking accuracy
    attention_redundancy_detection:
      metric: "cosine_similarity_precision"
      target: 0.95              # Head redundancy detection accuracy
    layer_skip_potential:
      metric: "gradient_flow_preservation"
      target: 0.85              # Skip connection evaluation accuracy
    structured_pruning_accuracy:
      metric: "MAPE_vs_Taylor"
      target: 0.03              # <3% error in importance estimation
```

## Operational Considerations

### Health Monitoring

- **Health Check Endpoint**: `/health/elesh/{worker_id}/structured_analysis`
- **Key Metrics**: structured_analysis_completion_rate, multi_level_accuracy, peak_memory_usage, analysis_duration_p95
- **SLO Targets**: 95% successful analysis completion, <5 minute analysis time, <4GB memory usage, 90%+ importance ranking accuracy

### Failure Modes

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Structured Analysis Timeout | >5 minutes without completion | Skip multi-level analysis, fall back to basic importance scoring |
| Worker Crash During Analysis | Process termination | Restart worker, retry analysis if time permits within checkpoint window |
| Memory Overflow | >4GB RSS during analysis | Reduce analysis scope to channel-level only, generate conservative recommendations |
| Importance Data Corruption | Multi-level validation failure | Use previous checkpoint's analysis, continue training without structured pruning |
| Safety Validation Failure | Architecture integrity check failure | Reject all pruning decisions, continue with existing architecture |
| Taylor Expansion Calculation Error | Gradient computation failure | Fall back to magnitude-based importance, reduce analysis confidence |

### Scaling Considerations

- **Horizontal Scaling**: Add more analysis workers for larger models (1 worker per 100M parameters recommended)
- **Vertical Scaling**: 16GB RAM per worker for complex multi-level analysis
- **Resource Requirements**: CPU and memory intensive during analysis, no GPU usage required
- **Checkpoint Budget**: Multi-level analysis must complete within 2-5 minute checkpoint save window
- **Worker Scaling**: Enhanced formula W ≥ ceil(P / (R × T × L)) where L = analysis levels (3 for channel/head/layer)

## Security Considerations

- **Authentication**: Internal service mesh activated during checkpoint windows with structured pruning extensions
- **Authorization**: Multi-level analysis triggered only by valid Emrakul coordination with safety verification
- **Data Protection**: Importance statistics and pruning decisions encrypted within checkpoint with architecture fingerprinting
- **Audit**: All structured pruning decisions logged with complete multi-level analysis rationale and safety validation results
- **Idempotency**: All operations are idempotent with replay safety using enhanced idempotency keys: `{epoch}:{model_hash}:{decision_id}:{analysis_level}`

## Migration Notes

> **Migration Status**: Version 3.0 - Enhanced with C-020 structured pruning capabilities
> - **C-020 Conclave** (2025-01-14): Unanimous approval for structured pruning implementation
> - **Multi-Level Analysis**: Channel, attention head, and layer pruning analysis
> - **Enhanced Safety**: Progressive schedules with multi-level rollback thresholds
> - **Performance Optimization**: Taylor expansion importance with correlation analysis
> - **Backward Compatibility**: All legacy analysis capabilities preserved and enhanced

## Future Enhancements

### Phase 3: Advanced Structured Pruning (6-9 months)
- **Description**: Fisher Information-based importance calculation and sparse tensor format optimization
- **Trigger**: After Phase 2 structured pruning validated in production
- **Impact**: More sophisticated importance calculations with optimal sparse representations

### Phase 4: Architecture Search Integration (9-12 months)
- **Description**: Integration with Neural Architecture Search (NAS) for optimal pruned architectures
- **Trigger**: After advanced structured pruning demonstrates consistent performance gains
- **Impact**: Automated discovery of optimal channel/head/layer configurations

### Phase 5: Hardware-Aware Pruning (12-15 months)
- **Description**: Hardware-specific structured pruning with kernel fusion optimization
- **Trigger**: After architecture search integration provides stable baselines
- **Impact**: Hardware-optimized pruning patterns for maximum inference acceleration

## Cross-References

### Subdocuments
- [13.1-elesh-importance-tracking.md]: Enhanced Count-Min Sketch with multi-level importance tracking
- [13.2-elesh-analysis-algorithms.md]: Structured pruning algorithms for channels, heads, and layers

### Related Documents
- [12-emrakul-unified-design.md]: Strategic coordinator with structured pruning coordination (v3.0)
- [02-kasmina-unified-design.md]: Execution layer with gradient statistics export (migration folder)
- [07-urabrask-unified-design.md]: Safety validation with structured pruning gates (migration folder)
- [01-tolaria-unified-design.md]: Phase management and checkpoint coordination (migration folder)
- [00-leyline-shared-contracts.md]: Shared contracts with structured pruning extensions
- [ADR-020-structured-pruning-checkpoint-based.md]: Architecture decision record for structured pruning

## Implementation Status

### Current State (v3.0)
- [x] Architecture Design: Complete (C-020 - Structured Pruning Implementation)
- [x] Structured Pruning Specifications: Complete (channels, heads, layers)
- [x] Message Contracts: Complete (Leyline integration with structured pruning)
- [x] Safety Mechanisms: Complete (progressive schedules with multi-level rollback)
- [ ] Multi-Level Importance Tracker: Design ready, implementation pending
- [ ] Structural Analyzer: Design ready, implementation pending
- [ ] Enhanced Safety Validator: Design ready, implementation pending
- [ ] Checkpoint Integration: Specification complete, implementation pending

### Validation Status
- [ ] Multi-level analysis accuracy validation (channel/head/layer)
- [ ] Checkpoint overhead benchmarking with structured analysis
- [ ] Integration testing with Emrakul v3.0 structured pruning coordination
- [ ] Production readiness review with progressive safety schedules
- [ ] Non-blocking failure testing with multi-level rollback

### C-020 Implementation Completeness
- [x] **Channel Pruning**: Taylor expansion importance with correlation-based redundancy detection
- [x] **Attention Head Pruning**: Pattern entropy and diversity analysis with cosine similarity clustering
- [x] **Layer Pruning**: Skip connection evaluation with gradient flow assessment
- [x] **Progressive Safety**: Multi-phase schedule with escalating safety requirements
- [x] **Message Contracts**: Complete structured pruning message specifications
- [x] **Rollback System**: Multi-level rollback triggers with checkpoint-based recovery

## History & Context

### Version History
- **v1.0** (2025-01-13): Initial runtime distributed worker design from C-019
- **v2.0** (2025-01-13): **Paradigm shift to checkpoint-based analysis** (C-020 Consensus)
- **v2.1** (2025-01-13): All blocking issues resolved for production deployment
- **v3.0** (2025-01-14): **Structured pruning implementation** with multi-level analysis (C-020 Round 6)

### Integration History
- **C-019 Conclave** (2025-01-13): Distributed worker design approved
- **C-020 Conclave** (2025-01-13): **Unanimous approval (6/6) for checkpoint-based pivot**
- **C-020 Round 6** (2025-01-14): **Complete structured pruning specifications** approved
- **Architecture Enhancement** (2025-01-14): Multi-level analysis with channels, heads, and layers
- **Production Readiness** (2025-01-14): Comprehensive safety mechanisms and progressive schedules

### Critical Enhancements Applied (v3.0)
- **Structured Pruning**: Complete channel, attention head, and layer analysis algorithms
- **Multi-Level Safety**: Progressive schedules with escalating rollback thresholds
- **Enhanced Importance**: Taylor expansion calculation with multi-metric scoring
- **Architecture Integration**: Comprehensive message contracts with Leyline extensions
- **Production Safety**: Non-blocking failures with comprehensive validation gates

---

*Last Updated: 2025-01-14 | Next Review: 2025-02-14 | Owner: Innovation Plane Team | Status: PRODUCTION READY | Paradigm: Structured Pruning "Measure Twice, Cut Once"*