# ADR-002: Checkpoint-Based Pruning Architecture

## Metadata

| Field | Value |
|-------|-------|
| **ADR Number** | ADR-002 |
| **Status** | ACCEPTED |
| **Date** | 2025-01-14 |
| **Author(s)** | System Architecture Team |
| **Supersedes** | None |
| **Superseded By** | None |
| **Tags** | pruning, checkpoints, performance, morphogenetics, core-design |

## Executive Summary

This ADR documents the foundational architectural decision to perform all neural network pruning operations at checkpoint boundaries rather than during runtime training loops. This "measure twice, cut once" approach eliminates CUDA graph recompilation overhead and transforms a distributed real-time problem into a straightforward offline optimization task.

## Context

### Problem Statement

Morphogenetic neural networks require the ability to autonomously remove underutilized parameters during training. However, PyTorch's architectural limitations and GPU performance constraints make runtime pruning operationally impossible for production training workloads.

### Background

The Esper platform enables neural networks to evolve their architecture during training through both addition (grafting) and subtraction (pruning) operations. While parameter addition can be achieved through seed grafting, parameter removal faces fundamental technical barriers:

**PyTorch Limitations:**
- Cannot physically remove parameters from computation graphs
- Only supports parameter masking (setting to zero)
- Masked parameters still consume GPU memory and computation cycles

**CUDA Graph Performance Impact:**
- Runtime topology changes require complete CUDA graph recompilation
- Recompilation overhead: 100-200ms per topology change
- Distributed training amplifies recompilation across all devices

**Morphogenetic Requirements:**
- Continuous architectural evolution during training
- Zero disruption to gradient flow and optimizer state
- Efficient GPU memory utilization
- Maintains training stability and convergence

### Requirements

**Functional Requirements:**
- Enable parameter removal without training disruption
- Preserve gradient coherence across pruning operations
- Support iterative architectural refinement
- Maintain reproducible training outcomes

**Non-Functional Requirements:**
- Zero runtime performance overhead from pruning decisions
- 2-5 minute maximum checkpoint overhead (acceptable)
- Complete rollback capability for failed pruning attempts
- Memory reclamation for removed parameters

## Decision Drivers

1. **Performance Constraints**: Runtime recompilation creates unacceptable training disruption
2. **PyTorch Architecture**: Framework limitations require creative solutions
3. **Training Stability**: Morphogenetic operations cannot compromise gradient flow
4. **Operational Simplicity**: Checkpoint boundaries provide natural synchronization points

Prioritization: Training stability is primary, followed by performance, with operational simplicity ensuring maintainability.

## Decision

**Selected Approach**: Checkpoint-Based Pruning with "Measure Twice, Cut Once" Philosophy

### Rationale

Transform the impossible distributed real-time pruning problem into a straightforward offline optimization task by performing all structural changes at checkpoint boundaries:

1. **Analysis Phase** (During Training):
   - Continuous monitoring of parameter utilization
   - Statistical analysis of gradient magnitudes
   - Identification of pruning candidates
   - No structural modifications to active model

2. **Pruning Phase** (At Checkpoint):
   - Load checkpoint into analysis environment
   - Apply pruning decisions based on collected statistics
   - Create new model topology with parameters physically removed
   - Validate pruned model maintains performance characteristics
   - Save optimized checkpoint or rollback if validation fails

3. **Resume Phase** (Post-Pruning):
   - Resume training from pruned checkpoint
   - Fresh CUDA graph compilation with optimized topology
   - Reclaimed GPU memory available for further evolution

### Technical Implementation

```python
# Continuous analysis during training (zero overhead)
class PruningAnalyzer:
    def monitor_step(self, model, gradients):
        self.gradient_stats.update(gradients)  # Lightweight statistics
        self.utilization_tracker.update(model.parameters())
        # No topology modifications

# Checkpoint-based pruning (offline operation)
class CheckpointPruner:
    def prune_at_checkpoint(self, checkpoint_path):
        model = load_checkpoint(checkpoint_path)

        # Sophisticated analysis with full computational budget
        pruning_plan = self.analyze_full_model(model)

        # Physical parameter removal (not masking)
        pruned_model = self.apply_structural_changes(model, pruning_plan)

        # Validation with safety rollback
        if self.validate_pruned_model(pruned_model):
            save_checkpoint(pruned_model, checkpoint_path)
            return True
        else:
            # Rollback - original checkpoint unchanged
            return False
```

## Consequences

### Positive Consequences

- **Zero Runtime Overhead**: Training loop performance completely unaffected
- **Sophisticated Analysis**: Full computational budget available for pruning decisions
- **Clean Rollback**: Failed pruning attempts don't corrupt training state
- **Memory Reclamation**: Physical parameter removal frees GPU memory
- **Operational Simplicity**: Checkpoint boundaries provide natural coordination points

### Negative Consequences

- **Checkpoint Overhead**: 2-5 minute processing time for large models
- **Delayed Benefits**: Pruning effects only realized at next training resumption
- **Storage Requirements**: Temporary storage needed for analysis and validation
- **Complexity**: Requires robust checkpoint validation and rollback mechanisms

### Neutral Consequences

- **Training Rhythm**: Establishes regular checkpoint-based optimization cycles
- **Analysis Quality**: More thorough analysis possible with offline computational budget

## Implementation

### Action Items

- [x] Implement statistical gradient tracking during training
- [x] Create checkpoint-based pruning pipeline
- [x] Build model validation framework for pruned topologies
- [x] Establish rollback mechanisms for failed pruning attempts
- [x] Integrate with Tolaria checkpoint management

### Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Runtime overhead | 0% impact on training step time | Training loop profiling |
| Checkpoint processing | <5 minutes for 1B parameter model | Pruning pipeline timing |
| Memory reclamation | 15-30% GPU memory freed | CUDA memory profiling |
| Rollback reliability | 100% successful rollback rate | Validation test suite |

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Checkpoint corruption | Low | Critical | Atomic checkpoint operations, backup retention |
| Pruning validation failure | Medium | Medium | Conservative validation thresholds, automatic rollback |
| Storage space exhaustion | Medium | High | Automated cleanup, storage monitoring |
| Analysis phase bugs | Medium | High | Extensive testing, gradual deployment |

## Dependencies

### Technical Dependencies
- PyTorch checkpoint serialization
- CUDA memory management
- Statistical analysis libraries
- File system atomic operations

### Organizational Dependencies
- Training pipeline integration
- Checkpoint management strategy
- Storage infrastructure provisioning

## Related Decisions

- **ADR-001**: [Performance-Optimized Architecture](./ADR-001-performance-optimized-tight-coupling.md)
- **Tolaria Design**: Checkpoint management and rollback authority
- **Kasmina Design**: GPU memory management and execution coordination
- **Tamiyo Design**: Strategic pruning decision policies

## References

### Internal Documentation
- [Tolaria Unified Design](../high-level-design/007-component-specifications.md)
- [Morphogenetic Training Lifecycle](../high-level-design/006-system-design-data-flow.md)

### External Resources
- PyTorch Model Pruning Documentation
- CUDA Graph Best Practices
- Neural Architecture Search Literature

---

## Decision History

| Date | Status | Notes |
|------|--------|-------|
| 2025-01-10 | PROPOSED | Initial analysis of runtime vs checkpoint approaches |
| 2025-01-12 | ANALYZED | Performance testing confirmed CUDA graph overhead |
| 2025-01-14 | ACCEPTED | Conclave C-020 approved checkpoint-based approach |

---

*This decision establishes the foundational pruning architecture for all morphogenetic operations. Changes require performance validation and architecture team approval.*

*Contact: System Architecture Team for questions about this decision*