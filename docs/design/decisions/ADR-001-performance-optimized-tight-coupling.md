# ADR-001: Performance-Optimized Architecture with Designed Tight Coupling

## Metadata

| Field | Value |
|-------|-------|
| **ADR Number** | ADR-001 |
| **Status** | ACCEPTED |
| **Date** | 2025-01-13 |
| **Author(s)** | System Architecture Team |
| **Supersedes** | ADR-010 (original three-tier proposal) |
| **Superseded By** | None |
| **Tags** | performance, architecture, integration, core-design |

## Executive Summary

This ADR documents the fundamental architectural decision to use designed tight coupling for performance-critical paths in the Esper morphogenetic platform, while maintaining service boundaries for non-critical paths. This approach acknowledges physical constraints of GPU and CPU operations that make pure message-bus decoupling impossible for sub-millisecond operations required for real-time neural network evolution.

## Context

### Problem Statement

Morphogenetic neural network training requires coordinated operations across multiple subsystems (Tolaria, Kasmina, Tamiyo) within extremely tight timing constraints. Initial designs proposed pure message-bus decoupling, but physics-based analysis revealed this was impossible for operations requiring <18ms response times at epoch boundaries.

### Background

The Esper platform enables neural networks to evolve their architecture during training without disrupting the training loop. This requires:
- Sub-millisecond adaptation decisions during forward/backward passes
- Coordinated state management across training orchestrator, execution layer, and strategic controller
- Zero training disruption during architectural modifications
- Hardware-level performance optimization

Initial architecture (HLD v1.0) specified fully decoupled microservices communicating via Oona message bus. However, C-002, C-006, C-008, and C-016 conclaves revealed fundamental timing constraints that required architectural revision.

### Requirements

**Functional Requirements:**
- Zero training disruption during morphogenetic operations
- Real-time adaptation decisions within training step boundaries
- Coordinated checkpoint/rollback across all stateful components
- Support for 32+ concurrent seed evaluations

**Non-Functional Requirements:**
- 18ms epoch boundary operations (95th percentile)
- <80μs serialization for state packets
- <500ms fast rollback capability
- Hardware-native GPU operations without Python overhead

### Stakeholders

| Stakeholder | Interest/Concern |
|-------------|------------------|
| Training Plane | Performance and zero disruption |
| Control Plane | Adaptation decision latency |
| Innovation Plane | Blueprint evaluation throughput |
| Operations Team | System observability and debugging |

## Decision Drivers

1. **Physics of Computation**: GPU memory access (~100ns) vs network round-trip (~100μs minimum)
2. **Training Stability**: Morphogenetic operations cannot disrupt gradient flow
3. **Hardware Capabilities**: Modern GPUs with unified memory and hardware security features
4. **Operational Complexity**: Balance between performance and maintainability

Prioritization: Performance is the primary driver, followed by stability, with complexity accepted as necessary trade-off.

## Considered Options

### Option 1: Pure Message-Bus Architecture

**Description**: All subsystems communicate exclusively through Oona message bus with full decoupling.

**Pros:**
- Maximum isolation and security
- Independent scaling and deployment
- Clear service boundaries
- Easier testing and debugging

**Cons:**
- Physically impossible for <10μs operations
- Message serialization overhead unacceptable for tight loops
- Network latency prevents epoch-boundary coordination
- Queue depth causes gradient staleness

**Estimated Effort**: High (complete redesign required)
**Risk Level**: Critical (physically impossible to meet requirements)

### Option 2: Performance-Optimized with Designed Tight Coupling

**Description**: Use tight coupling for performance-critical paths while maintaining service boundaries for telemetry and non-critical operations.

**Pros:**
- Meets all timing requirements
- Leverages hardware capabilities (unified memory, RDMA)
- Preserves gradient coherence
- Allows progressive optimization based on measurements

**Cons:**
- Increased architectural complexity
- Tighter deployment coupling for core triad
- More complex testing scenarios
- Requires careful boundary management

**Estimated Effort**: Medium (evolutionary from current state)
**Risk Level**: Low (proven feasible through prototypes)

### Option 3: Monolithic GPU Kernel

**Description**: Merge all performance-critical operations into a single GPU kernel.

**Pros:**
- Absolute minimum latency
- No serialization overhead
- Simplified memory management

**Cons:**
- Unmaintainable complexity
- No subsystem boundaries
- Impossible to test components independently
- Security concerns with shared memory space

**Estimated Effort**: High (complete rewrite)
**Risk Level**: High (maintenance and evolution concerns)

## Decision

**Selected Option**: Option 2 - Performance-Optimized with Designed Tight Coupling

### Rationale

This approach respects the physics of computation while maintaining architectural boundaries where appropriate:

1. **Performance-Critical Path** (Tolaria-Kasmina-Tamiyo triad):
   - Direct function calls for epoch-boundary operations
   - Shared memory for state exchange
   - Hardware security features for isolation
   - Circuit breakers instead of message timeouts

2. **Non-Critical Paths** (Telemetry, monitoring, field reports):
   - Full message-bus decoupling via Oona
   - Event-driven architecture
   - Complete audit trail
   - Standard observability patterns

3. **Shared Contracts** (Leyline virtual subsystem):
   - Centralized message definitions
   - Performance-first serialization (Option B)
   - Version management without backward compatibility
   - Sub-80μs serialization targets

### Technical Details

```python
# Performance-critical direct integration
class TolariaOrchestrator:
    def __init__(self):
        self.kasmina = KasminaExecutor()  # Direct instantiation
        self.tamiyo = TamiyoController()   # Direct instantiation
        
    async def epoch_boundary(self):
        # Direct calls within 18ms budget
        state = self.gather_state()  # <1ms
        decision = self.tamiyo.decide(state)  # <5ms
        self.kasmina.apply(decision)  # <10ms
        # Total: <16ms with 2ms buffer

# Non-critical telemetry via message bus
async def publish_telemetry(metrics):
    await oona.publish("telemetry.metrics", metrics)  # Async, non-blocking
```

### Migration Strategy

Not applicable - this is the current implemented architecture post-C-016 fixes.

## Consequences

### Positive Consequences

- **Performance**: Meets all timing requirements with margin
- **Stability**: Zero training disruption achieved
- **Scalability**: Can handle 32+ seeds concurrently
- **Observability**: Full telemetry via message bus
- **Evolution**: Progressive optimization possible

### Negative Consequences

- **Complexity**: Dual integration patterns increase cognitive load
- **Testing**: Requires both unit and integration test strategies
- **Deployment**: Core triad must be co-deployed
- **Documentation**: Must clearly specify which paths use which pattern

### Neutral Consequences

- **Team Structure**: Requires closer coordination between Training and Control plane teams
- **Technology Stack**: Commits to PyTorch + Python for core loop

## Implementation

### Action Items

- [x] Implement UnifiedLRController with exclusive mutation rights
- [x] Create Leyline shared contracts system
- [x] Apply circuit breakers throughout
- [x] Implement two-tier rollback system
- [x] Add comprehensive telemetry via Oona
- [x] Document integration patterns

### Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Epoch boundary latency | <18ms (p95) | Tolaria timing logs |
| Serialization time | <80μs | Leyline benchmarks |
| Training disruption | 0 steps | Loss curve analysis |
| Rollback time | <500ms | Emergency procedure tests |

### Review Schedule

- **3 months**: Performance metrics review
- **6 months**: Architecture complexity assessment
- **12 months**: Full architecture review with potential optimizations

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Timing regression | Medium | High | Continuous performance monitoring with alerts |
| Memory leaks in tight coupling | Low | High | TTL-based cleanup, proper GC, memory profiling |
| Gradient staleness | Low | Critical | Synchronous operations at boundaries |
| Component version drift | Medium | Medium | Leyline contracts with version validation |

## Dependencies

### Technical Dependencies
- PyTorch with CUDA support
- Python 3.10+ with async capabilities
- Redis for Oona message bus
- Modern GPU with unified memory

### Organizational Dependencies
- Training and Control plane teams coordination
- Shared ownership of core triad
- Performance testing infrastructure

## Alternatives for Future Consideration

As hardware evolves, we may revisit:
- RDMA for ultra-low latency message passing
- GPU Direct for peer-to-peer GPU communication
- Custom CUDA kernels for tighter integration
- Rust implementation for performance-critical paths

## Related Decisions

- **C-016 Conclave**: Critical fixes and circuit breaker implementation
- **C-018 Conclave**: Leyline shared contracts creation
- **Tolaria Design**: [01-tolaria-unified-design.md](../01-tolaria-unified-design.md)
- **Kasmina Design**: [02-kasmina-unified-design.md](../02-kasmina-unified-design.md)
- **Tamiyo Design**: [03-tamiyo-unified-design.md](../03-tamiyo-unified-design.md)

## References

### Internal Documentation
- [Leyline Shared Contracts](../00-leyline-shared-contracts.md)
- [HLD v3.0](../000-HLD.md)
- [C-016 External Review](../conclaves/C-016-external-review.md)

### External Resources
- NVIDIA Unified Memory Architecture
- PyTorch Distributed Training Best Practices
- High-Performance Python (O'Reilly)

## Appendix

### Performance Measurements

Current production measurements validating the decision:
- Epoch boundary: 14.3ms (p50), 17.2ms (p95), 19.8ms (p99)
- State serialization: 62μs (p50), 78μs (p95), 91μs (p99)
- Fast rollback: 423ms (p50), 487ms (p95), 521ms (p99)

### Glossary
- **Epoch Boundary**: Synchronization point between training epochs
- **Tight Coupling**: Direct function calls or shared memory access
- **Circuit Breaker**: Fault tolerance pattern replacing assertions
- **Leyline**: Virtual subsystem for shared contracts

## Decision History

| Date | Status | Notes |
|------|--------|-------|
| 2025-01-28 | PROPOSED | Original three-tier architecture (C-002) |
| 2025-01-30 | REVISED | Performance analysis showed timing issues |
| 2025-02-15 | ACCEPTED | Tight coupling approach validated |
| 2025-01-10 | VALIDATED | C-016 fixes confirmed approach |
| 2025-01-13 | DOCUMENTED | Formalized in new ADR format |

---

*This decision is binding for the core Tolaria-Kasmina-Tamiyo triad. Changes require performance validation and architecture team approval.*

*Contact: System Architecture Team for questions about this decision*