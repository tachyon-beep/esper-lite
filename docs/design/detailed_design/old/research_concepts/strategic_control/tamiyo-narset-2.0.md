# Tamiyo-Narset 2.0: Strategic-Tactical Control Architecture

## Executive Summary

This proposal defines the architectural separation between Tamiyo (strategic control) and Narset (tactical execution) as distinct subsystems, addressing the critical need for distributed control in large-scale morphogenetic neural networks. Based on C-023 conclave findings and scaling requirements for models with 1000+ regions, this design enables elastic scaling while maintaining clean architectural boundaries.

## Problem Statement

### Current Limitations

1. **Tamiyo Overload**: The strategic controller currently handles both high-level policy and low-level seed coordination
2. **Scaling Bottleneck**: Single controller cannot manage 1000+ regions with 10,000+ seeds effectively
3. **Resource Contention**: Mixing strategic and tactical operations causes CPU/memory pressure
4. **Operational Complexity**: Debugging and monitoring become impossible with mixed responsibilities

### Requirements

- Support for 50-100 tactical controllers in large models
- Each controller managing 10-20 seed banks independently
- Sub-millisecond tactical decisions without blocking strategic planning
- Hierarchical telemetry aggregation to prevent O(N²) message storms
- Clean separation between strategic policies and tactical execution

## Architectural Design

### Subsystem Organization

```
Subsystem 03: Tamiyo (Strategic Controller)
├── Strategic Policy Engine
├── Risk Management System
├── Global Resource Allocation
├── Cross-Subsystem Coordination
└── Epoch-Aligned Decision Making

Subsystem 15: Narset (Tactical Execution Layer)
├── NarsetCoordinator (Singleton)
│   ├── Worker Lifecycle Management
│   ├── Region-to-Worker Mapping
│   ├── Hierarchical Telemetry Aggregation
│   └── Resource Quota Enforcement
└── NarsetWorkers (1-100 instances)
    ├── Local Seed Bank Management (10-20 seeds)
    ├── Tactical Decision Execution
    ├── Regional Metrics Collection
    └── Direct Kasmina Integration
```

### Communication Architecture

```
┌─────────────────────────────────────┐
│     Tamiyo (Strategic Control)      │
│  - Global policies & objectives     │
│  - Resource budgets & constraints   │
│  - Risk thresholds & rollback       │
└──────────────┬──────────────────────┘
               │ Epoch-aligned messages
               │ (Policy & Budgets)
┌──────────────▼──────────────────────┐
│   Narset Coordinator (Singleton)    │
│  - Worker pool management           │
│  - Work distribution algorithm      │
│  - Telemetry aggregation           │
└──────────────┬──────────────────────┘
               │ Direct function calls
               │ (Work assignments)
┌──────────────▼──────────────────────┐
│   Narset Workers (1-100 instances)  │
│  - Tactical seed control            │
│  - Local optimization decisions     │
│  - Regional performance metrics     │
└──────────────┬──────────────────────┘
               │ Direct memory access
               │ (Seed commands)
┌──────────────▼──────────────────────┐
│      Kasmina Seeds (10,000+)        │
│  - Neural network regions           │
│  - Local adaptation agents          │
└─────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Narset as Separate Subsystem

**Rationale**:
- Clean separation of concerns (strategic vs tactical)
- Independent scaling and resource management
- Failure isolation between layers
- Simplified testing and debugging

**Trade-offs**:
- Additional subsystem complexity (+1 subsystem)
- Inter-subsystem communication overhead
- Deployment coordination requirements

#### 2. Worker Pool Pattern

**Benefits**:
- Elastic scaling from 1 to 100+ workers
- Dynamic spawning based on model size
- Resource isolation with per-worker quotas
- Independent failure domains

**Implementation**:
```python
class NarsetCoordinator:
    def __init__(self):
        self.workers = {}  # region_id -> NarsetWorker
        self.worker_pool = WorkerPool(min=1, max=100)
        self.telemetry_aggregator = HierarchicalAggregator()

    def assign_region(self, region_id: str, seed_count: int):
        # Dynamic worker allocation based on load
        if self.needs_new_worker(seed_count):
            worker = self.worker_pool.spawn_worker()
        else:
            worker = self.find_least_loaded_worker()

        worker.assign_region(region_id, seed_count)
        self.workers[region_id] = worker
```

#### 3. Hierarchical Telemetry

**Problem**: 100 workers × 100 metrics × 10Hz = 100,000 messages/sec

**Solution**: Tree-based aggregation
```
Workers (100) → Regional Aggregators (10) → Coordinator (1) → Tamiyo
         O(N)              O(√N)                O(1)         = O(N)
```

**Implementation**:
```python
class HierarchicalAggregator:
    def __init__(self, fan_in=10):
        self.fan_in = fan_in
        self.aggregation_tree = self.build_tree()

    def aggregate(self, worker_metrics):
        # First level: aggregate by region groups
        regional_metrics = self.aggregate_level_1(worker_metrics)
        # Second level: aggregate regions to global
        global_metrics = self.aggregate_level_2(regional_metrics)
        return global_metrics
```

## Integration Points

### Tamiyo → Narset Communication

**Epoch-Aligned Policy Updates** (every 100-1000 steps):
```python
PolicyUpdate = {
    "alpha_schedule": {"warmup": 0.1, "target": 0.3, "decay": 0.99},
    "resource_budget": {"memory_gb": 100, "compute_tflops": 50},
    "risk_thresholds": {"max_loss_spike": 2.0, "min_improvement": 0.001},
    "protected_regions": [1, 5, 10],  # Critical regions not to modify
}
```

### Narset → Kasmina Integration

**Direct Function Calls** (no message bus):
```python
class NarsetWorker:
    def __init__(self, kasmina_ref):
        self.kasmina = kasmina_ref  # Direct reference, same process
        self.seeds = {}

    def execute_tactical_decision(self, region_id, decision):
        # Direct memory manipulation, no serialization
        seed = self.kasmina.get_seed(region_id)
        seed.apply_modification(decision)  # Microsecond operation
```

### Narset → Ral Coordination (if Ral becomes subsystem)

**Compilation Requests** (direct calls for Ral within Kasmina):
```python
class NarsetWorker:
    def request_compilation(self, region_id, alpha):
        # Ral is component within Kasmina, accessed directly
        kernel = self.kasmina.ral.get_kernel(region_id, alpha)
        return kernel  # Sub-millisecond with warm cache
```

## Scaling Analysis

### Small Models (v1.0)
- 1 Narset worker
- 10-50 regions
- 100-500 seeds
- **Overhead**: Minimal, essentially current Tamiyo behavior

### Medium Models (v1.5)
- 5-10 Narset workers
- 100-200 regions
- 1,000-2,000 seeds
- **Benefits**: Parallel tactical execution, reduced Tamiyo load

### Large Models (v2.0)
- 50-100 Narset workers
- 1,000+ regions
- 10,000+ seeds
- **Critical**: Only architecture that can scale to this level

### Massive Models (v3.0)
- Hierarchical Narset with regional coordinators
- 10,000+ regions
- 100,000+ seeds
- **Future-proof**: Tree structure enables arbitrary scaling

## Implementation Phases

### Phase 1: Minimal Viable Separation (2 weeks)
- Extract tactical control from Tamiyo
- Create Narset subsystem with single worker
- Direct Kasmina integration
- Basic telemetry aggregation

### Phase 2: Multi-Worker Support (2 weeks)
- Worker pool implementation
- Dynamic work distribution
- Resource quota enforcement
- Enhanced monitoring

### Phase 3: Elastic Scaling (2 weeks)
- Auto-scaling based on load
- Hierarchical telemetry aggregation
- Advanced failure handling
- Performance optimization

### Phase 4: Production Hardening (2 weeks)
- Comprehensive testing at scale
- Monitoring and alerting
- Documentation and training
- Performance benchmarking

## Risk Mitigation

### Risk 1: Communication Overhead
**Mitigation**: Epoch-aligned updates reduce messages by 100-1000x

### Risk 2: Worker Coordination Complexity
**Mitigation**: Start with single worker, add complexity incrementally

### Risk 3: Resource Contention
**Mitigation**: Strict quotas and cgroup isolation per worker

### Risk 4: Debugging Difficulty
**Mitigation**: Comprehensive telemetry and distributed tracing

## Success Criteria

1. **Performance**: Tactical decisions in <1ms (currently 10-50ms)
2. **Scalability**: Support 100+ workers without degradation
3. **Reliability**: Worker failures don't affect strategic control
4. **Maintainability**: Clear separation enables independent evolution
5. **Observability**: Complete visibility into all control decisions

## Comparison with Alternatives

### Alternative 1: Narset as Tamiyo Component
- ❌ Resource contention within single process
- ❌ Cannot scale to 100 workers
- ❌ Mixing of strategic/tactical code
- ✅ Simpler deployment

### Alternative 2: No Narset (Enhanced Tamiyo)
- ❌ Fundamental scaling limit at ~100 regions
- ❌ Impossible to debug at scale
- ❌ Single point of failure
- ✅ No new subsystem

### Recommended: Separate Narset Subsystem
- ✅ Clean architectural boundaries
- ✅ Elastic scaling to 100+ workers
- ✅ Resource isolation
- ✅ Independent evolution
- ❌ Additional subsystem complexity

## Conclusion

The Tamiyo-Narset 2.0 architecture provides the only viable path to scaling morphogenetic control to production-scale models. By separating strategic control (Tamiyo) from tactical execution (Narset), we achieve:

1. **Unlimited scaling** through the worker pool pattern
2. **Clean separation** of responsibilities
3. **Resource isolation** preventing interference
4. **Evolutionary flexibility** for future enhancements

This architecture has been validated through:
- C-023 conclave consensus on feasibility
- Proven patterns from Elesh (C-019) and Kasmina (C-022)
- Scaling analysis showing viability to 100,000+ seeds

**Recommendation**: Proceed with Phase 1 implementation immediately, targeting initial deployment within 2 weeks.

---

*Document Version*: 2.0
*Date*: 2025-01-15
*Status*: Proposed Architecture
*Author*: System Architect (via C-023 analysis)