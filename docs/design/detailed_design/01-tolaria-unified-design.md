# Tolaria Unified Design Document v3.1 - Enhanced with C-020 Structured Pruning
## Modularized Architecture + Structured Pruning Phase Management

**Version:** 3.1 - Enhanced with C-020 Structured Pruning Phase Management
**Status:** IMPLEMENTATION-READY - Complete Technical Specifications with C-016 + C-020 Integration
**Date:** 2025-01-14
**Author:** System Architecture Team + C-006/C-008/C-016 Integration + C-020 Structured Pruning
**Supersedes:** v3.0 (modularized), v2.2 (monolithic), v2.1, v2.0

---

## Executive Summary

This document presents the core architecture for Tolaria, the Training Orchestrator of the Esper morphogenetic platform. Tolaria serves as the master coordinator that provides the stable temporal framework (epochs and steps) in which the host model lives and evolves. It owns the core training loop, manages the optimizer, executes emergency rollbacks, and acts as the final authority on system stability.

**Modularization Note:** This document has been split from a 2643-line monolithic design into 5 focused documents for maintainability and clarity. Each subdocument handles a specific architectural concern while maintaining complete cross-references.

**C-006 Integration:** Complete algorithms, hardware physics-compliant performance targets, integration protocols, and validation frameworks.

**C-008 Integration:** Two-tier rollback system, four-level emergency protocol, Protocol Buffer integration, and shared infrastructure specifications.

**C-016 Critical Fixes:**
- Assert statements replaced with circuit breakers
- WAL durability semantics with O_DSYNC flags
- UnifiedLRController with exclusive mutation rights
- Memory leak fixes with proper keying and GC
- Timing reality aligned to 18ms epoch boundary

**C-020 Structured Pruning Integration:**
- Checkpoint boundary coordination for structured pruning operations
- Progressive pruning phase management (validation → active pruning)
- Checkpoint storage management for rollback capability
- Zero disruption to main training loop during analysis phases

## 1. Core Architecture Decision

After extensive review and ADR-010 consensus, the architecture for Tolaria is:

### **Performance-Optimized Training Orchestrator with Designed Tight Coupling**

- **Foundation**: Standard PyTorch training loop with morphogenetic extensions
- **Integration Model**: Intentionally tight coupling with Kasmina-Tamiyo triad
- **Authority Model**: Final authority on system stability and rollbacks
- **Service Boundaries**: Respect service boundaries for non-performance-critical paths

## 2. Architectural Principles

### 2.1 Non-Negotiable Requirements

1. **Zero Training Disruption**: Morphogenetic operations must never block training
2. **Performance First**: 18ms epoch boundary operations
3. **Stability Authority**: Tolaria has final say on system health
4. **Synchronized Recovery**: Checkpoint all stateful components together
5. **Optimizer Flexibility**: Dynamic parameter group management
6. **Emergency Response**: Immediate rollback capability
7. **Structured Pruning Coordination**: Zero-disruption checkpoint-based pruning support

### 2.2 Design Principles

1. **Epoch-Driven Architecture**: All major operations synchronized to epoch boundaries
2. **Direct Integration**: Performance-critical paths use direct calls, not messages
3. **State Consistency**: Model, optimizer, and controller states always synchronized
4. **Progressive Enhancement**: Support gradual morphogenetic capability addition
5. **Safety Monitoring**: Independent health checks beyond controller signals
6. **Checkpoint-Based Pruning**: All structural modifications coordinated at checkpoint boundaries

## 3. System Architecture

### 3.1 Core Components

**Training Loop Manager** (Enhanced for C-020)
- Standard PyTorch training and validation loops
- Epoch and step counter management
- Batch processing and data loading
- Loss computation and backpropagation
- **Checkpoint boundary coordination for structured pruning operations**
- **Importance statistics coordination with Kasmina during checkpoint saves**
- **Pruning mask application coordination during checkpoint loads**
- Details: [01.1-tolaria-epoch-lifecycle.md](01.1-tolaria-epoch-lifecycle.md#training-loop)

**[C-016 FIX] UnifiedLRController**
- Exclusive LR mutation rights
- Enhanced parameter group support
- Circuit breaker protection
- Complete state tracking
- Details: [01.3-tolaria-optimizer-lr.md](01.3-tolaria-optimizer-lr.md#unified-lr-controller)

**DynamicOptimizerManager**
- Rebuilds optimizer when parameters added
- Uses UnifiedLRController for all LR operations
- Preserves optimizer state during rebuilds
- Details: [01.3-tolaria-optimizer-lr.md](01.3-tolaria-optimizer-lr.md#dynamic-optimizer)

**Checkpoint System** (Enhanced for C-020)
- Enhanced WAL durability with O_DSYNC
- 256-byte WAL headers with validation
- Execution context in Merkle root
- Emergency restoration procedures
- **Structured pruning metadata storage and retrieval**
- **Checkpoint storage management for pruning rollback capability**
- Details: [01.2-tolaria-rollback-systems.md](01.2-tolaria-rollback-systems.md#checkpoint-system)

**End-of-Epoch Hook** (Enhanced for C-020)
- Assembles SystemStatePacket
- Conservative mode triggers via circuit breakers
- MonotonicTimer validation
- Tamiyo.step() with timeout wrapper
- **Structured pruning phase coordination**
- **Checkpoint boundary pruning trigger coordination**
- Details: [01.1-tolaria-epoch-lifecycle.md](01.1-tolaria-epoch-lifecycle.md#epoch-hook)

**Fast Rollback System**
- Two-tier rollback: 500ms fast, 12s full
- In-memory checkpoint cache
- Four-level emergency protocol
- Shared memory signaling
- Details: [01.2-tolaria-rollback-systems.md](01.2-tolaria-rollback-systems.md#fast-rollback)

**Emergency Rollback System**
- Circuit breaker integration
- Independent stability monitoring
- Distributed rollback coordination
- Quarantine management
- Details: [01.2-tolaria-rollback-systems.md](01.2-tolaria-rollback-systems.md#emergency-rollback)

**Structured Pruning Phase Manager** (New C-020 Component)
- Progressive pruning schedule management
- Two-phase coordination (validation → active pruning)
- Checkpoint boundary timing coordination
- Safety constraint enforcement
- Rollback coordination for failed pruning attempts
- Details: [01.5-tolaria-structured-pruning.md](01.5-tolaria-structured-pruning.md#phase-management)

### 3.2 Training Loop Management (Enhanced for C-020)

**Core Training Loop Responsibilities:**

1. **Standard Training Operations**:
   - Batch processing and forward/backward passes
   - Loss computation and gradient calculation
   - Optimizer step execution and parameter updates
   - Validation loop coordination and metrics collection

2. **Enhanced Checkpoint Boundary Coordination (C-020)**:
   - **Coordinate pruning operations at 2-5 minute checkpoint intervals**
     - Schedule checkpoint saves to align with structured pruning analysis windows
     - Ensure training loop pause coordination during checkpoint operations
     - Manage checkpoint storage requirements for pruning rollback capability (100GB+ storage)
     - Coordinate with Kasmina for importance statistics export during saves

   - **Ensure zero disruption to main training loop**
     - Maintain training throughput during checkpoint boundary coordination
     - Implement asynchronous checkpoint coordination to avoid blocking training
     - Preserve gradient accumulation and batch processing timing
     - Maintain optimizer state consistency across checkpoint boundaries

   - **Manage checkpoint storage for rollback capability**
     - Implement checkpoint rotation with pruning metadata preservation
     - Coordinate checkpoint cleanup with pruning analysis completion
     - Maintain checkpoint integrity for emergency rollback scenarios
     - Manage storage quotas and cleanup policies for large checkpoints

**Implementation Pattern**:
```python
class EnhancedTrainingLoopManager:
    """C-020 Enhanced training loop with structured pruning coordination"""

    def __init__(self, config: TolariaConfig):
        self.config = config
        self.pruning_phase_manager = StructuredPruningPhaseManager(config)
        self.checkpoint_coordinator = CheckpointCoordinator(config)

    async def training_epoch(self, epoch: int, dataloader):
        """Enhanced training epoch with checkpoint boundary coordination"""

        # Check if this epoch requires structured pruning coordination
        pruning_required = self.pruning_phase_manager.is_pruning_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            # Standard training step
            loss = self._training_step(batch, batch_idx)

            # Check for checkpoint boundary
            if self._is_checkpoint_boundary(epoch, batch_idx):
                if pruning_required:
                    # Coordinate structured pruning at checkpoint boundary
                    await self._coordinate_checkpoint_pruning(epoch, batch_idx)
                else:
                    # Standard checkpoint save
                    await self._save_checkpoint(epoch, batch_idx)

        # End-of-epoch coordination
        await self._end_epoch_coordination(epoch)

    async def _coordinate_checkpoint_pruning(self, epoch: int, batch_idx: int):
        """Coordinate structured pruning at checkpoint boundary"""

        # Step 1: Export importance statistics via Kasmina
        importance_stats = await self.kasmina_client.export_importance_statistics()

        # Step 2: Trigger Emrakul analysis coordination
        analysis_request = StructuralAnalysisRequest(
            epoch=epoch,
            importance_stats=importance_stats,
            checkpoint_metadata=self._create_checkpoint_metadata(),
            pruning_phase=self.pruning_phase_manager.current_phase
        )

        # Step 3: Coordinate with Emrakul for offline analysis
        pruning_result = await self.emrakul_client.coordinate_pruning_analysis(analysis_request)

        # Step 4: Apply pruning results if validation passed
        if pruning_result.validation_passed:
            await self._apply_pruning_masks(pruning_result.masks)

        # Step 5: Save checkpoint with pruning metadata
        await self._save_checkpoint_with_pruning_metadata(epoch, batch_idx, pruning_result)
```

### 3.3 Implementation Status

**POST-C-016 + C-020 STATUS: PRODUCTION-READY**

All components have complete technical specifications:
- ✅ Training/validation loops with multi-seed gradient aggregation
- ✅ UnifiedLRController with exclusive mutation rights
- ✅ Enhanced WAL with durability semantics
- ✅ Circuit breaker protection replacing all asserts
- ✅ Dynamic optimizer management with momentum preservation
- ✅ Complete checkpoint/restore with WAL functionality
- ✅ Distributed rollback coordination protocol
- ✅ Memory leak prevention with proper GC
- ✅ Async telemetry aggregation system
- ✅ **Structured pruning phase management and coordination**
- ✅ **Checkpoint boundary coordination for pruning operations**
- ✅ **Progressive pruning schedule implementation**

### 3.4 Shared Contract Dependencies

Tolaria relies on the [Leyline Shared Contracts](00-leyline-shared-contracts.md) virtual subsystem for all cross-subsystem message definitions and shared enums. This ensures binary compatibility across all 13 Esper subsystems (including Elesh).

**Contract Usage:**
- `SystemStatePacket` - Core training state communication with Tamiyo (see [Leyline §3.1](00-leyline-shared-contracts.md#systemstatepacket-optimized))
- `AdaptationCommand` - Control plane commands and responses (see [Leyline §3.2](00-leyline-shared-contracts.md#adaptationcommand-unified))
- `HardwareContext` - Hardware state reporting and monitoring (see [Leyline §3.3](00-leyline-shared-contracts.md#shared-enums-and-constants))
- `StructuralPruningRequest` - Structured pruning coordination messages (C-020)
- `CheckpointMetadata` - Enhanced checkpoint metadata with pruning information (C-020)
- Shared enums: `SeedLifecycleStage`, `HealthStatus`, `CircuitBreakerState`, `MessagePriority`, `PruningPhase`

**Implementation Pattern:**
```python
# Import shared contracts from Leyline
from esper.leyline.contracts import (
    SystemStatePacket,
    AdaptationCommand,
    HardwareContext,
    SeedLifecycleStage,
    HealthStatus,
    StructuralPruningRequest,  # C-020
    CheckpointMetadata,        # C-020
    PruningPhase              # C-020
)
```

This design eliminates local contract definitions, prevents schema drift, and ensures optimal serialization performance (<80μs) through Leyline's performance-first Option B implementation.

## 4. Component Documentation Structure

This unified design has been modularized into focused subdocuments:

### **[01.1-tolaria-epoch-lifecycle.md](01.1-tolaria-epoch-lifecycle.md)**
- Training loop structure and management
- Epoch lifecycle with multi-seed integration
- End-of-epoch hook implementation
- Multi-seed gradient aggregation algorithms
- Timing budgets and performance targets

### **[01.2-tolaria-rollback-systems.md](01.2-tolaria-rollback-systems.md)**
- Two-tier rollback system (500ms fast, 12s full)
- Emergency procedures and four-level protocol
- Distributed rollback coordination
- Checkpoint system with WAL durability
- State management and recovery

### **[01.3-tolaria-optimizer-lr.md](01.3-tolaria-optimizer-lr.md)**
- Dynamic optimizer state preservation
- UnifiedLRController design with exclusive mutation
- Learning rate coordination across seeds
- Parameter group management
- Optimizer rebuild procedures

### **[01.4-tolaria-integration-protocols.md](01.4-tolaria-integration-protocols.md)**
- Shared infrastructure specifications
- Protocol Buffer message serialization
- Hardware physics-compliant performance targets
- Memory management and telemetry systems
- Validation and testing frameworks
- Configuration and API specifications
- Monitoring and observability
- Complete interface definitions

### **[01.5-tolaria-structured-pruning.md](01.5-tolaria-structured-pruning.md)** (New C-020)
- Structured pruning phase management
- Checkpoint boundary coordination protocols
- Progressive schedule implementation
- Safety constraint enforcement
- Rollback coordination for failed pruning attempts

## 5. Structured Pruning Phase Management (C-020)

### 5.1 Progressive Schedule Implementation

**Two-Phase Structured Pruning Schedule:**

```python
class StructuredPruningPhaseManager:
    """C-020 Progressive pruning schedule with safety constraints"""

    def __init__(self, config: TolariaConfig):
        self.config = config
        self.current_phase = PruningPhase.VALIDATION_ONLY
        self.phase_transition_epoch = config.pruning_phase_transition_epoch  # Default: 30

    def get_current_phase(self, epoch: int) -> PruningPhase:
        """Determine current pruning phase based on epoch"""

        if epoch <= self.phase_transition_epoch:
            return PruningPhase.VALIDATION_ONLY
        else:
            return PruningPhase.ACTIVE_PRUNING

    def is_pruning_epoch(self, epoch: int) -> bool:
        """Determine if pruning coordination is required for this epoch"""

        current_phase = self.get_current_phase(epoch)

        if current_phase == PruningPhase.VALIDATION_ONLY:
            # Phase 1: Telemetry collection only, every epoch
            return True
        elif current_phase == PruningPhase.ACTIVE_PRUNING:
            # Phase 2: Active pruning at checkpoint intervals (every 2-5 epochs)
            return epoch % self.config.pruning_checkpoint_interval == 0

        return False
```

**Phase Definitions:**

### 5.2 Phase 1: Validation Mode (Epochs 1-30)

**Purpose**: Pure telemetry collection and importance tracking validation

**Operations**:
- **Telemetry collection only**: Export importance statistics at every checkpoint
- **No physical pruning**: No parameter masking or structural modifications
- **Importance score accumulation**: Build comprehensive importance databases
- **Algorithm validation**: Validate importance tracking accuracy and consistency
- **Safety threshold calibration**: Determine conservative pruning thresholds

**Coordination Flow**:
```
Training Loop → Checkpoint Boundary → Export Importance Stats →
Emrakul Analysis (validation) → Store Results → Continue Training
```

**Performance Impact**: <0.1% training overhead for importance tracking
**Safety Guarantees**: Zero disruption, no structural changes, full rollback capability

### 5.3 Phase 2: Execution Mode (Epochs 31+)

**Purpose**: Active structured pruning with progressive schedule enforcement

**Operations**:
- **Active pruning at checkpoints**: Apply validated pruning masks during checkpoint loads
- **Progressive schedule enforcement**: Conservative → moderate → aggressive thresholds
- **Safety monitoring and rollback**: Continuous accuracy monitoring with automatic rollback
- **Checkpoint storage management**: Maintain rollback capability for failed pruning attempts

**Progressive Thresholds**:
```python
class ProgressivePruningSchedule:
    """Progressive pruning thresholds with safety constraints"""

    def get_pruning_budget(self, epoch: int) -> float:
        """Calculate pruning budget based on training progress"""

        if epoch < 50:
            return 0.01  # Conservative: 1% parameter removal
        elif epoch < 100:
            return 0.03  # Moderate: 3% parameter removal
        else:
            return 0.05  # Aggressive: 5% parameter removal (maximum)

    def get_safety_thresholds(self, epoch: int) -> Dict[str, float]:
        """Get safety thresholds for rollback triggers"""

        return {
            'accuracy_drop_threshold': 0.02,      # 2% accuracy drop triggers rollback
            'gradient_norm_threshold': 5.0,       # 5x gradient norm triggers rollback
            'memory_usage_threshold': 0.90,       # 90% memory usage triggers rollback
            'training_speed_threshold': 0.90      # 90% training speed triggers rollback
        }
```

**Coordination Flow**:
```
Training Loop → Checkpoint Boundary → Export Importance Stats →
Emrakul Analysis → Urabrask Validation → Apply Masks →
Safety Monitoring → [Rollback if needed] → Continue Training
```

**Safety Constraints**:
- **Maximum 5% parameter removal per checkpoint cycle**
- **2% accuracy drop threshold triggers immediate rollback**
- **3 consecutive failed pruning attempts triggers circuit breaker**
- **Comprehensive checkpoint storage for 30-second rollback capability**

### 5.4 Checkpoint Storage Management

**Storage Requirements**:
```yaml
checkpoint_storage:
  # Basic checkpoint requirements
  base_checkpoint_size: 2-10GB        # Model + optimizer state

  # C-020 Enhanced requirements
  importance_metadata_size: 100-500MB  # Importance statistics
  pruning_metadata_size: 50-200MB     # Pruning masks and decisions
  rollback_history_size: 5-15GB       # Last 3 checkpoints for rollback

  # Total storage per checkpoint
  total_checkpoint_size: 7-25GB       # Depends on model size

  # Storage management
  retention_policy:
    last_3_checkpoints: always_keep    # For rollback capability
    historical_checkpoints: 7_day_retention
    importance_statistics: 30_day_retention
```

**Storage Management Implementation**:
```python
class CheckpointStorageManager:
    """C-020 Enhanced checkpoint storage with pruning metadata"""

    def __init__(self, config: TolariaConfig):
        self.config = config
        self.storage_quota = config.checkpoint_storage_quota_gb  # Default: 100GB

    async def save_checkpoint_with_pruning_metadata(self,
                                                   epoch: int,
                                                   checkpoint_data: Dict,
                                                   pruning_metadata: Optional[PruningMetadata] = None):
        """Save checkpoint with enhanced pruning metadata"""

        checkpoint_path = self._generate_checkpoint_path(epoch)

        enhanced_checkpoint = {
            **checkpoint_data,
            'pruning_metadata': pruning_metadata,
            'importance_statistics': self._collect_importance_statistics(),
            'checkpoint_version': 'tolaria_3.1_c020',
            'pruning_phase': self.phase_manager.get_current_phase(epoch),
            'rollback_capability': True
        }

        # Save with atomic operation
        await self._atomic_save(checkpoint_path, enhanced_checkpoint)

        # Manage storage quota
        await self._cleanup_old_checkpoints()

    async def coordinate_emergency_rollback(self, target_epoch: int) -> bool:
        """Coordinate emergency rollback to previous checkpoint"""

        try:
            # Find last known good checkpoint
            rollback_checkpoint = await self._find_rollback_checkpoint(target_epoch)

            if not rollback_checkpoint:
                logger.error(f"No rollback checkpoint found for epoch {target_epoch}")
                return False

            # Restore model, optimizer, and pruning state
            await self._restore_checkpoint(rollback_checkpoint)

            # Reset pruning phase manager state
            self.phase_manager.reset_to_checkpoint(rollback_checkpoint.epoch)

            logger.info(f"Emergency rollback completed to epoch {rollback_checkpoint.epoch}")
            return True

        except Exception as e:
            logger.error(f"Emergency rollback failed: {e}")
            return False
```

## 6. Integration Points

Tolaria integrates tightly with:

**Kasmina** (Execution Layer)
- Direct function calls for morphogenetic layer operations
- Shared memory for tensor operations
- <1ms response requirement
- **Enhanced coordination for importance statistics export**
- **Pruning mask application coordination**

**Emrakul** (Strategic Coordinator)
- **Structured pruning analysis coordination**
- **Checkpoint boundary synchronization**
- **Progressive schedule enforcement**

**Tamiyo** (Strategic Controller)
- End-of-epoch SystemStatePacket submission
- AdaptationCommand processing
- 2-second timeout protection

**Urabrask** (Validation Engine)
- **Structured pruning validation coordination**
- **Safety constraint validation**
- **Emergency rollback trigger coordination**

**Oona** (Message Bus)
- Async telemetry submission
- Non-critical path communications
- Backpressure handling

**Nissa** (Observability)
- Metrics and logging
- Distributed tracing
- Alert generation

## 7. Critical Design Decisions

1. **Tight Coupling with Kasmina-Tamiyo**: Intentional for performance
2. **UnifiedLRController Exclusivity**: Prevents LR mutation conflicts
3. **Two-Tier Rollback**: Balances speed (500ms) with completeness (12s)
4. **Circuit Breakers over Asserts**: Production safety
5. **18ms Epoch Boundary**: Hardware physics reality
6. **Modular Documentation**: Maintainability and focused ownership
7. **Checkpoint-Based Pruning (C-020)**: Zero-disruption structural modifications
8. **Progressive Pruning Schedule (C-020)**: Conservative safety with gradual optimization

## 8. Future Considerations

- Distributed training support (Phase 2)
- Advanced morphogenetic capabilities (Phase 3)
- Multi-model orchestration (Future)
- Federated learning integration (Research)
- **Physical parameter compaction (C-020 Phase 2)**
- **Advanced pruning strategies (Fisher Information, cross-epoch analysis)**

## 9. References

- ADR-010: Execution Plane Architecture Decision
- C-006: Conclave Round 3 Technical Specifications
- C-008: Execution Plane Integration Validation
- C-016: External Review Critical Fixes
- C-017: Tamiyo Restoration Validation
- **C-020: Structured Pruning Checkpoint-Based Design**

## 10. Cross-Reference Index

For detailed implementation:
- Epoch Management → [01.1-tolaria-epoch-lifecycle.md](01.1-tolaria-epoch-lifecycle.md)
- Rollback Systems → [01.2-tolaria-rollback-systems.md](01.2-tolaria-rollback-systems.md)
- Optimizer/LR → [01.3-tolaria-optimizer-lr.md](01.3-tolaria-optimizer-lr.md)
- Integration/API → [01.4-tolaria-integration-protocols.md](01.4-tolaria-integration-protocols.md)
- **Structured Pruning → [01.5-tolaria-structured-pruning.md](01.5-tolaria-structured-pruning.md)** (C-020)

---

**COMPONENT STATUS**: COMPLETE - Enhanced with C-020 Structured Pruning Phase Management
**Production Readiness**: All C-016 critical safety issues addressed + C-020 structured pruning coordination
**Performance Targets**: 18ms epoch boundary maintained with <0.1% pruning overhead
**Structured Pruning**: Progressive two-phase schedule with comprehensive safety constraints and rollback capability
**Architecture**: Clean separation of concerns with focused, maintainable components + structured pruning coordination
**Next Steps**: Individual component refinement and structured pruning optimization