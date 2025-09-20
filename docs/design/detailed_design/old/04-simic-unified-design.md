# Simic Unified Design Document v3.0

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 3.0 |
| **Status** | PRODUCTION READY - C-016 Integration Complete |
| **Date** | 2025-01-10 |
| **Author** | System Architecture Team |
| **Component** | Control Plane - Policy Training Environment |
| **Parent** | High-Level Design (HLD) |
| **Subdocuments** | [04.1-simic-rl-algorithms.md](04.1-simic-rl-algorithms.md), [04.2-simic-experience-replay.md](04.2-simic-experience-replay.md) |

## Executive Summary

Simic provides the dedicated, offline training environment for the neural network policies used by Tamiyo (strategic controller) and Karn (generative architect) in the Esper morphogenetic platform. Simic closes the critical learning loop by consuming field reports from production adaptations and generating improved policies that enhance system intelligence over time through conventional gradient descent and policy optimization.

**Important Clarification:** Simic uses STANDARD machine learning techniques (reinforcement learning, imitation learning) to train Tamiyo and Karn. This is conventional ML training, NOT morphogenetic evolution. The innovation is that these conventionally-trained controllers guide the host model's architectural evolution - but the controllers themselves learn through standard ML methods.

Key characteristics:
- **Offline Policy Training**: MLOps pipeline for continuous policy improvement without disrupting production
- **Experience Replay**: Graph-compatible circular buffer with PyTorch Geometric for topology preservation
- **Production Safety**: Complete C-016 fixes including circuit breakers, conservative mode, and chaos engineering

## Core Architecture Decision

### **Offline Policy Training Service with Experience Replay**

- **Foundation**: MLOps pipeline for continuous policy improvement
- **Integration Model**: Event-driven consumption and publication via Oona message bus
- **Authority Model**: Owns policy training, versioning, and validation
- **Deployment Model**: Dedicated training environment with GPU resources

## Architectural Principles

### Non-Negotiable Requirements

1. **Continuous Learning**: Policies improve from real adaptation outcomes
2. **Non-Disruptive Training**: Policy updates never interrupt production
3. **Version Management**: All policies versioned and validated before deployment
4. **Experience Replay**: Efficient learning from historical field reports
5. **Safety Validation**: New policies validated before production deployment
6. **Catastrophic Forgetting Prevention**: EWC and other techniques preserve knowledge

### Design Principles

1. **Offline Training**: Separate from production to prevent disruption
2. **Event-Driven Integration**: Loose coupling via message bus
3. **Experience Management**: Circular buffer for efficient replay
4. **Multi-Algorithm Support**: RL, imitation learning, curriculum learning
5. **Policy Versioning**: Git-like versioning for policy checkpoints

### Production Safety Principles

1. **Circuit Breaker Architecture**: No assert statements in production paths - all replaced with circuit breakers
2. **Conservative Mode**: Graceful degradation when SLOs exceed error budgets
3. **Memory Management**: TTL-based cleanup preventing resource leaks
4. **Chaos Resilience**: Complete failure recovery at all points
5. **Property Invariants**: Mathematically verified system properties maintained

## System Components

### Component Overview

| Component | Purpose | Details |
|-----------|---------|---------|
| **SimicTrainer** | Neural network policy training engine with IMPALA | See: [04.1-simic-rl-algorithms.md](04.1-simic-rl-algorithms.md) |
| **GraphExperienceBuffer** | Graph-compatible circular buffer using PyTorch Geometric | See: [04.2-simic-experience-replay.md](04.2-simic-experience-replay.md) |
| **PolicyManager** | Version control for policy checkpoints | See: Section 5.1 |
| **PolicyValidator** | Comprehensive validation pipeline with chaos testing | See: Section 5.2 |
| **FieldReportProcessor** | Ingests field reports from Tamiyo | See: [04.2-simic-experience-replay.md](04.2-simic-experience-replay.md) |
| **SimicService** | FastAPI service wrapper | See: Section 7.2 |
| **SimicHealth** | Health monitoring with SLO tracking | See: Section 7.2 |

### Core Components Summary

**SimicTrainer**
- Reinforcement learning algorithm: IMPALA with V-trace corrections
- Distributed actor-learner architecture (32 CPU actors, 4 GPU learners)
- Elastic Weight Consolidation (EWC) for memory preservation
- LoRA adapters for efficient updates
- Details: [04.1-simic-rl-algorithms.md#simictrainer](04.1-simic-rl-algorithms.md)

**GraphExperienceBuffer**
- Stores HeteroData objects without topology loss
- Prioritized experience replay with graph-aware sampling
- Memory allocation: 12GB total (6GB experiences, 1.4GB models, 4.6GB overhead)
- TTL-based cleanup every 100 epochs
- Details: [04.2-simic-experience-replay.md#graphexperiencebuffer](04.2-simic-experience-replay.md)

**PolicyValidator**
- Performance, safety, and regression validation
- Chaos engineering validation
- Property-based testing
- Security scanning
- See implementation in Section 5.2

## Integration Architecture

### Subsystem Dependencies

| Subsystem | Integration Type | Purpose |
|-----------|-----------------|---------|
| Tamiyo | Async (Redis Streams) | Field report consumption |
| Karn | Async (EventEnvelope) | Policy updates delivery |
| Jace | Async (EventEnvelope) | Curriculum coordination |
| Oona | Async (Redis/EventEnvelope) | Message bus communication |
| Nissa | Async (TelemetryPacket) | Observability and monitoring |

### Message Contracts

| Contract | Direction | Purpose |
|----------|-----------|---------|
| FieldReport | Tamiyo → Simic | Training data from adaptations |
| PolicyUpdate | Simic → Tamiyo/Karn | Updated policies |
| CurriculumRequest | Jace ↔ Simic | Curriculum coordination |
| TelemetryPacket | Simic → Nissa | Metrics and monitoring |

### Shared Contracts (Leyline)

This subsystem uses the following shared contracts from Leyline:
- `leyline.SystemStatePacket` - System state representation
- `leyline.AdaptationCommand` - Unified adaptation commands
- `leyline.EventEnvelope` - Message bus envelope for Oona
- `leyline.TelemetryPacket` - Observability and metrics
- `leyline.SeedState` - Seed lifecycle management
- `leyline.SeedLifecycleStage` - Seed stage enumeration
- `leyline.HardwareContext` - Hardware state representation
- `leyline.HealthStatus` - Health monitoring
- `leyline.CircuitBreakerState` - Circuit breaker states
- `leyline.MessagePriority` - Message prioritization
- `leyline.TelemetryLevel` - Logging and telemetry levels

For complete contract definitions, see: `/docs/architecture/00-leyline-shared-contracts.md`

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training iteration | <500ms for 32-sample batch | `simic_training_time_ms` histogram |
| Experience replay sampling | <50ms for batch creation | `simic_sampling_time_ms` histogram |
| Policy validation | <1 second for safety checks | `simic_validation_time_ms` histogram |
| Memory usage | 12GB maximum | `simic_memory_usage_gb` gauge |
| GPU utilization | 65-70% during training | `simic_gpu_utilization` gauge |
| Training throughput | 180-250 experiences/second | `simic_throughput_exp_sec` gauge |
| Convergence time | 48-72 hours for daily updates | Manual tracking |
| SLO compliance | >99% adherence to targets | `simic_slo_compliance` gauge |

## Configuration

### Key Configuration Parameters

```yaml
simic:
  # Algorithm selection
  algorithm: "IMPALA"  # NOT PPO

  # Training parameters
  learning_rate: 3e-4
  batch_size: 32
  gamma: 0.99
  max_grad_norm: 0.5

  # V-trace parameters
  rho_bar: 1.0  # Importance sampling truncation
  c_bar: 1.0    # Temporal difference truncation

  # Graph experience replay
  buffer_capacity: 100000  # Reduced for graph data
  graph_batch_method: "PyG_Batch"  # Never torch.stack

  # Memory allocation
  memory_budget_gb: 12.0
  experience_memory_gb: 6.0
  model_memory_gb: 1.4
  overhead_memory_gb: 4.6
  ttl_cleanup_interval_s: 100

  # Distributed training
  num_actors: 32  # CPU actors
  num_learners: 4  # GPU learners

  # Performance targets
  target_throughput_exp_sec: 180
  convergence_time_hours: 48
  epoch_boundary_budget_ms: 18
  training_step_budget_ms: 500

  # Training settings
  min_experiences: 1000
  replay_ratio: 0.25

  # Advanced techniques
  ewc_lambda: 0.1
  ewc_sample_size: 200
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1

  # Curriculum
  curriculum_enabled: true
  curriculum_stages: 4

  # Validation
  min_validation_accuracy: 0.7
  max_validation_loss: 0.5
  min_success_rate: 0.6
  chaos_testing_enabled: true
  property_testing_enabled: true

  # Circuit breakers
  circuit_breaker_failure_threshold: 3
  circuit_breaker_recovery_timeout_s: 30

  # Conservative mode
  conservative_mode_triggers:
    - "high_error_rate"
    - "memory_pressure"
    - "circuit_breaker_open"
    - "slo_violation"
    - "training_instability"
  conservative_batch_size_factor: 0.5
  conservative_timeout_factor: 1.5

  # Resource limits
  max_memory_gb: 8.0
  training_timeout_hours: 24.0

  # Service settings
  health_check_interval: 30
  metrics_port: 9092
```

For detailed configuration options, see subdocuments.

## Policy Versioning and Deployment

### 5.1 Version Management

```python
@dataclass
class PolicyVersionMetadata:
    """Versioned policy metadata with enhanced safety - scheduled for Leyline migration"""
    version_id: str  # e.g., "v1.2.3"
    training_iteration: int
    timestamp: datetime

    # Training context
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    experience_count: int

    # [C-016] Enhanced safety tracking
    safety_score: float              # Validation safety assessment
    chaos_test_passed: bool          # Chaos engineering validation
    property_tests_passed: bool      # Property-based test results

    # Deployment status
    deployment_status: str  # "testing", "staged", "production"
    deployment_timestamp: Optional[datetime]
    rollback_count: int = 0

    # [C-016] Security audit trail
    created_by: str                  # User/system that created version
    approved_by: Optional[str]       # Approval for production deployment
    justification: str               # Reason for policy update

# Migration Note: PolicyVersionMetadata is scheduled for Leyline migration with metadata+reference pattern.
# Currently defined locally, policy weights will be stored in Urza with reference IDs.

class PolicyManager:
    """Manage policy versions with Git-like semantics"""

    def __init__(self, storage_path: str = "/var/lib/simic/policies"):
        self.storage_path = storage_path
        self.current_version = None
        self.version_history = []

    def save_checkpoint(
        self,
        policy_weights: Dict[str, torch.Tensor],
        metadata: PolicyVersionMetadata
    ) -> str:
        """Save policy checkpoint with versioning"""
        version_path = f"{self.storage_path}/{metadata.version_id}"

        # Save weights
        torch.save(policy_weights, f"{version_path}/weights.pth")

        # Save metadata
        with open(f"{version_path}/metadata.json", "w") as f:
            json.dump(asdict(metadata), f)

        # Update version tracking
        self.version_history.append(metadata.version_id)
        self.current_version = metadata.version_id

        logging.info(f"Saved policy checkpoint: {metadata.version_id}")
        return metadata.version_id

    def load_checkpoint(self, version_id: str) -> Tuple[Dict[str, torch.Tensor], PolicyVersionMetadata]:
        """Load specific policy version"""
        version_path = f"{self.storage_path}/{version_id}"

        # Load weights
        weights = torch.load(f"{version_path}/weights.pth")

        # Load metadata
        with open(f"{version_path}/metadata.json", "r") as f:
            metadata_dict = json.load(f)
            metadata = PolicyVersionMetadata(**metadata_dict)

        return weights, metadata

    def rollback(self, target_version: str) -> bool:
        """Rollback to previous policy version"""
        if target_version not in self.version_history:
            logging.error(f"Version {target_version} not found in history")
            return False

        self.current_version = target_version
        logging.info(f"Rolled back to policy version: {target_version}")
        return True
```

### 5.2 Enhanced Validation Pipeline

```python
@dataclass
class ValidationResult:
    """Validation result with comprehensive safety checks - local to Simic"""
    passed: bool
    validations: List[Any]
    ab_config: Optional[Any] = None
    chaos_resistant: bool = False
    properties_maintained: bool = False
    security_clear: bool = False
    failure_reason: Optional[str] = None

# Migration Note: ValidationResult is deferred for future Leyline migration.
# Currently Simic-specific as validation approaches vary by subsystem.

class PolicyValidator:
    """[C-016] Enhanced policy validation with comprehensive safety checks"""

    def __init__(self):
        self.chaos_tester = PolicyChaosTestFramework()
        self.property_tester = PolicyPropertyTester()
        self.security_scanner = PolicySecurityScanner()
        self.validation_circuit_breaker = CircuitBreaker(failure_threshold=3)

    async def validate_policy(self, policy: PolicyVersionMetadata) -> ValidationResult:
        """Comprehensive policy validation with all C-016 requirements"""

        if not self.validation_circuit_breaker.is_closed():
            return ValidationResult(
                passed=False,
                validations=[],
                failure_reason="Validation circuit breaker open"
            )

        try:
            validations = []

            # Performance validation
            perf_result = await self.validate_performance(policy)
            validations.append(perf_result)

            # Safety validation
            safety_result = await self.validate_safety(policy)
            validations.append(safety_result)

            # Regression testing
            regression_result = await self.validate_no_regression(policy)
            validations.append(regression_result)

            # [C-016] Chaos engineering validation
            chaos_result = await self.chaos_tester.validate_policy_chaos_resistance(policy)
            validations.append(chaos_result)

            # [C-016] Property-based testing
            property_result = await self.property_tester.validate_policy_properties(policy)
            validations.append(property_result)

            # [C-016] Security scanning
            security_result = await self.security_scanner.scan_policy(policy)
            validations.append(security_result)

            # Overall assessment
            if all(v.passed for v in validations):
                ab_config = self.generate_ab_test_config(policy)
                self.validation_circuit_breaker.record_success()
                return ValidationResult(
                    passed=True,
                    validations=validations,
                    ab_config=ab_config,
                    chaos_resistant=chaos_result.passed,
                    properties_maintained=property_result.passed,
                    security_clear=security_result.passed
                )

            self.validation_circuit_breaker.record_failure()
            return ValidationResult(
                passed=False,
                validations=validations,
                failure_reason=self.summarize_failures(validations)
            )

        except Exception as e:
            logging.error(f"Policy validation failed: {e}")
            self.validation_circuit_breaker.record_failure()
            return ValidationResult(
                passed=False,
                validations=[],
                failure_reason=str(e)
            )

    async def validate_performance(self, policy: PolicyVersionMetadata) -> ValidationResult:
        """Validate policy meets performance requirements"""
        # Load test environment
        test_env = self.create_test_environment()

        # Run performance tests
        accuracy = await self.test_accuracy(policy, test_env)
        latency = await self.test_latency(policy, test_env)

        passed = (
            accuracy >= self.config.min_validation_accuracy and
            latency <= self.config.max_inference_latency_ms
        )

        return ValidationResult(
            passed=passed,
            validations=[
                {"metric": "accuracy", "value": accuracy, "threshold": self.config.min_validation_accuracy},
                {"metric": "latency_ms", "value": latency, "threshold": self.config.max_inference_latency_ms}
            ]
        )

    async def validate_safety(self, policy: PolicyVersionMetadata) -> ValidationResult:
        """Validate policy safety constraints"""
        # Test adaptation boundaries
        boundary_test = await self.test_adaptation_boundaries(policy)

        # Test rollback capability
        rollback_test = await self.test_rollback_capability(policy)

        # Test resource constraints
        resource_test = await self.test_resource_constraints(policy)

        passed = all([boundary_test, rollback_test, resource_test])

        return ValidationResult(
            passed=passed,
            validations=[
                {"test": "adaptation_boundaries", "passed": boundary_test},
                {"test": "rollback_capability", "passed": rollback_test},
                {"test": "resource_constraints", "passed": resource_test}
            ]
        )

    async def validate_no_regression(self, policy: PolicyVersionMetadata) -> ValidationResult:
        """Ensure no regression from previous version"""
        if not self.policy_manager.current_version:
            # First policy, no regression possible
            return ValidationResult(passed=True, validations=[])

        # Load previous policy
        prev_weights, prev_metadata = self.policy_manager.load_checkpoint(
            self.policy_manager.current_version
        )

        # Compare performance
        regression_found = False
        regressions = []

        for metric, value in policy.validation_metrics.items():
            if metric in prev_metadata.validation_metrics:
                prev_value = prev_metadata.validation_metrics[metric]
                if value < prev_value * 0.95:  # 5% regression threshold
                    regression_found = True
                    regressions.append({
                        "metric": metric,
                        "current": value,
                        "previous": prev_value,
                        "regression": (prev_value - value) / prev_value
                    })

        return ValidationResult(
            passed=not regression_found,
            validations=regressions
        )

    def generate_ab_test_config(self, policy: PolicyVersionMetadata) -> Dict[str, Any]:
        """Generate A/B test configuration for gradual rollout"""
        return {
            "test_id": f"policy_test_{policy.version_id}",
            "control_version": self.policy_manager.current_version,
            "treatment_version": policy.version_id,
            "traffic_split": 0.1,  # Start with 10% traffic
            "success_metrics": ["adaptation_success_rate", "performance_improvement"],
            "duration_hours": 24,
            "auto_rollback": True,
            "rollback_threshold": 0.8  # Rollback if success rate drops below 80%
        }

    def summarize_failures(self, validations: List[ValidationResult]) -> str:
        """Create human-readable failure summary"""
        failures = []
        for validation in validations:
            if not validation.passed:
                if validation.failure_reason:
                    failures.append(validation.failure_reason)
                else:
                    failures.append(f"Validation failed: {validation.validations}")

        return "; ".join(failures)
```

### Chaos Testing Framework

```python
class PolicyChaosTestFramework:
    """[C-016] Chaos engineering for policy validation"""

    def __init__(self):
        self.chaos_scenarios = [
            "network_partition",
            "memory_pressure",
            "gpu_failure",
            "data_corruption",
            "timing_variations",
            "resource_starvation"
        ]

    async def validate_policy_chaos_resistance(
        self,
        policy: PolicyVersionMetadata
    ) -> ValidationResult:
        """Test policy under chaos conditions"""
        results = []

        for scenario in self.chaos_scenarios:
            result = await self.run_chaos_scenario(policy, scenario)
            results.append(result)

        passed = all(r["recovered"] for r in results)

        return ValidationResult(
            passed=passed,
            validations=results,
            chaos_resistant=passed
        )

    async def run_chaos_scenario(
        self,
        policy: PolicyVersionMetadata,
        scenario: str
    ) -> Dict[str, Any]:
        """Execute specific chaos scenario"""
        # Create isolated test environment
        test_env = self.create_chaos_environment(scenario)

        # Inject failure
        await self.inject_failure(test_env, scenario)

        # Test policy behavior
        start_time = time.time()
        recovered = await self.test_recovery(policy, test_env)
        recovery_time = time.time() - start_time

        return {
            "scenario": scenario,
            "recovered": recovered,
            "recovery_time_seconds": recovery_time,
            "max_allowed_seconds": 30
        }
```

## Operational Considerations

### Health Monitoring

- **Health Check Endpoint**: `/health` - Returns comprehensive health status
- **Key Metrics**: Training progress, buffer health, memory usage, SLO compliance
- **SLO Targets**:
  - Policy training latency p95 <= 500ms
  - Policy improvement rate >= 70%
  - Experience buffer memory <= 6GB

### Failure Modes

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Training divergence | Loss increasing over 10 epochs | Restore checkpoint, reduce learning rate |
| Memory exhaustion | Usage > 12GB | TTL cleanup, emergency buffer reduction |
| GPU failure | CUDA errors | Fallback to CPU, alert operators |
| Network partition | Redis disconnection | Local queue, retry with exponential backoff |
| Policy corruption | Validation failure | Reject update, maintain current version |

### Scaling Considerations

- **Horizontal Scaling**: Not supported (single training instance)
- **Vertical Scaling**: Add GPUs for faster training (up to 8 GPUs tested)
- **Resource Requirements**:
  - Minimum: 8GB RAM, 2 GPUs
  - Recommended: 16GB RAM, 4 GPUs
  - Maximum tested: 32GB RAM, 8 GPUs

## Security Considerations

- **Authentication**: Internal service, no external authentication
- **Authorization**: Role-based access for policy deployment
- **Data Protection**: No PII in training data
- **Audit Trail**: All policy updates logged with justification

## Migration Notes

> **Migration Status**: Structures being migrated to Leyline
> - PolicyVersionMetadata: Scheduled for migration with metadata+reference pattern
> - SLOViolation: Scheduled for migration to support system-wide SLO tracking
> - SimicConfig: Approved for immediate migration
> - LRPolicy: Pending migration with UnifiedLRController patterns

## Future Enhancements

### Phase 2: Distributed Training
- **Description**: Multi-node distributed training with data parallelism
- **Trigger**: When single-node training exceeds 72 hours
- **Impact**: 3-5x training speedup

### Phase 3: AutoML Integration
- **Description**: Automated hyperparameter tuning
- **Trigger**: After 6 months of production data
- **Impact**: 20-30% performance improvement

## Cross-References

### Subdocuments
- [04.1-simic-rl-algorithms.md](04.1-simic-rl-algorithms.md): IMPALA implementation, V-trace, EWC, LoRA
- [04.2-simic-experience-replay.md](04.2-simic-experience-replay.md): Buffer management, field report processing

### Related Documents
- [03-tamiyo-unified-design.md](03-tamiyo-unified-design.md): Field report producer
- [05-karn-unified-design.md](05-karn-unified-design.md): Policy consumer
- [00-leyline-shared-contracts.md](00-leyline-shared-contracts.md): Shared contract definitions
- [ADR-011]: Simic Policy Training Architecture

## Implementation Status

> **Esper-Lite prototype note (2025-09-21):** current scaffold includes a
> feature-rich FIFO replay buffer and PPO-style trainer with optional LoRA
> adapters (`src/esper/simic/replay.py`, `src/esper/simic/trainer.py`). The
> implementation extracts structured features from Tamiyo field reports,
> produces multi-head policy updates, exports Simic training metrics, and
> integrates with the demo control loop. Full production features described
> below (IMPALA/V-trace, distributed training, curriculum learning) remain
> future work.

### Current State
- [x] SimicTrainer: IMPALA with V-trace implementation complete
- [x] GraphExperienceBuffer: Circular buffer with TTL cleanup
- [x] PolicyManager: Version control system
- [x] PolicyValidator: Comprehensive validation pipeline
- [x] FieldReportProcessor: Redis stream consumer
- [x] UnifiedLRController: Learning rate invariants enforced
- [x] Conservative Mode: Automatic degradation under stress
- [x] Circuit Breakers: All critical paths protected

### Validation Status
- [x] Unit tests complete (>95% coverage)
- [x] Integration tests complete
- [x] Performance validation (meets all targets)
- [x] Security review passed
- [x] Production readiness review complete

## Monitoring and Observability

### 7.1 Key SLOs and Performance Targets

```yaml
# [C-016] Comprehensive SLO tracking with error budgets
simic_slos:
  # Policy Training Performance
  - name: policy_training_latency_ms
    objective: "p95 <= 500ms"
    error_budget: 0.1%  # 86.4 violations per day
    measurement_window: 24h

  - name: policy_improvement_rate
    objective: "success_rate >= 70%"
    error_budget: 5%    # 5% of policies may not improve
    measurement_window: 7d

  - name: experience_buffer_health
    objective: "memory_usage <= 6GB, TTL cleanup success >= 99%"
    error_budget: 1%
    measurement_window: 24h

  # System Reliability
  - name: conservative_mode_frequency
    objective: "triggers <= 2 per day"
    error_budget: 0%    # Conservative mode should be rare
    measurement_window: 24h

  - name: circuit_breaker_health
    objective: "open_time <= 1% of operational time"
    error_budget: 0%
    measurement_window: 24h
```

### 7.2 Health Monitoring Implementation

```python
@dataclass
class SLOViolation:
    """SLO violation tracking - scheduled for Leyline migration"""
    metric: str
    current_value: float
    threshold: float
    timestamp: float
    subsystem: str = "simic"

# Migration Note: SLOViolation is scheduled for Leyline migration to support
# system-wide SLO tracking by Nissa. Currently defined locally.

class SimicHealth:
    """[C-016] Comprehensive health monitoring with SLO tracking and Leyline integration"""

    def __init__(self):
        from esper.leyline.contracts import HealthStatus

        self.slo_tracker = SLOTracker("simic")
        self.health_checks = {
            'experience_buffer': self.check_buffer_health,
            'training_progress': self.check_training_progress,
            'memory_usage': self.check_memory_usage,
            'policy_quality': self.check_policy_quality,
            'integration_health': self.check_integration_health,
            'circuit_breaker_health': self.check_circuit_breaker_health,
            'conservative_mode_health': self.check_conservative_mode_health,
            'unified_lr_controller': self.check_lr_controller_health
        }

    def get_health_status(self) -> Dict[str, Any]:
        """[C-016] Comprehensive health assessment with SLO tracking"""

        from esper.leyline.contracts import HealthStatus

        checks = {}
        slo_violations = []

        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                checks[check_name] = result.status

                # Track SLO violations
                if result.slo_violation:
                    slo_violations.append(result.slo_violation)

            except Exception as e:
                logging.error(f"Health check {check_name} failed: {e}")
                checks[check_name] = 'ERROR'

        # Update SLO tracking
        for violation in slo_violations:
            self.slo_tracker.record_violation(violation)

        # Determine overall health using Leyline HealthStatus
        if any(check == 'CRITICAL' for check in checks.values()):
            overall_status = HealthStatus.HEALTH_CRITICAL
        elif any(check == 'WARNING' for check in checks.values()):
            overall_status = HealthStatus.HEALTH_DEGRADED
        elif any(check == 'ERROR' for check in checks.values()):
            overall_status = HealthStatus.HEALTH_UNHEALTHY
        else:
            overall_status = HealthStatus.HEALTH_HEALTHY

        return {
            "status": overall_status,
            "checks": checks,
            "slo_violations": slo_violations,
            "error_budget_remaining": self.slo_tracker.get_error_budget_remaining(),
            "timestamp": datetime.now(UTC).isoformat()
        }

    def check_buffer_health(self) -> HealthCheckResult:
        """Check experience buffer health"""
        buffer_size = len(self.experience_buffer.buffer)
        memory_usage_mb = self.experience_buffer._estimate_memory_usage()

        status = 'HEALTHY'
        slo_violation = None

        # Check memory usage
        if memory_usage_mb > 6000:  # 6GB limit
            status = 'CRITICAL'
            slo_violation = SLOViolation(
                metric='experience_buffer_memory',
                current_value=memory_usage_mb,
                threshold=6000,
                timestamp=time.time()
            )
        elif memory_usage_mb > 5000:  # Warning at 5GB
            status = 'WARNING'

        # Check TTL cleanup
        time_since_cleanup = time.time() - self.experience_buffer.last_cleanup
        if time_since_cleanup > 300:  # More than 5 minutes
            status = 'WARNING'

        return HealthCheckResult(
            status=status,
            details={
                'buffer_size': buffer_size,
                'memory_usage_mb': memory_usage_mb,
                'time_since_cleanup_s': time_since_cleanup,
                'cleanup_count': self.experience_buffer.cleanup_count
            },
            slo_violation=slo_violation
        )

    def check_training_progress(self) -> HealthCheckResult:
        """Check policy training progress"""
        if not hasattr(self, 'trainer'):
            return HealthCheckResult(status='UNKNOWN', details={})

        # Check training metrics
        recent_losses = self.trainer.get_recent_losses(window=100)
        if not recent_losses:
            return HealthCheckResult(status='UNKNOWN', details={'reason': 'No recent training'})

        avg_loss = np.mean(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

        status = 'HEALTHY'
        slo_violation = None

        # Check if training is diverging
        if loss_trend > 0.01:  # Loss increasing
            status = 'WARNING'
        if avg_loss > self.config.max_validation_loss * 2:
            status = 'CRITICAL'
            slo_violation = SLOViolation(
                metric='training_loss',
                current_value=avg_loss,
                threshold=self.config.max_validation_loss,
                timestamp=time.time()
            )

        return HealthCheckResult(
            status=status,
            details={
                'avg_loss': avg_loss,
                'loss_trend': loss_trend,
                'training_iterations': self.trainer.iteration_count,
                'time_since_last_update_s': time.time() - self.trainer.last_update_time
            },
            slo_violation=slo_violation
        )

    def check_circuit_breaker_health(self) -> HealthCheckResult:
        """Check circuit breaker states"""
        circuit_breakers = {
            'training': self.trainer.training_circuit_breaker,
            'storage': self.experience_buffer.storage_circuit_breaker,
            'sampling': self.experience_buffer.sampling_circuit_breaker,
            'publication': self.policy_publisher.publication_circuit_breaker
        }

        open_breakers = []
        half_open_breakers = []

        for name, breaker in circuit_breakers.items():
            if breaker.state == CircuitBreakerState.OPEN:
                open_breakers.append(name)
            elif breaker.state == CircuitBreakerState.HALF_OPEN:
                half_open_breakers.append(name)

        status = 'HEALTHY'
        slo_violation = None

        if open_breakers:
            status = 'CRITICAL'
            slo_violation = SLOViolation(
                metric='circuit_breakers_open',
                current_value=len(open_breakers),
                threshold=0,
                timestamp=time.time()
            )
        elif half_open_breakers:
            status = 'WARNING'

        return HealthCheckResult(
            status=status,
            details={
                'open_breakers': open_breakers,
                'half_open_breakers': half_open_breakers,
                'total_breakers': len(circuit_breakers)
            },
            slo_violation=slo_violation
        )
```

### Service Wrapper

```python
class SimicService:
    """FastAPI service wrapper for Simic"""

    def __init__(self, config: SimicConfig):
        self.config = config
        self.app = FastAPI(title="Simic Policy Training Service")

        # Initialize components
        self.lr_controller = UnifiedLRController()
        self.trainer = SimicTrainer(config, self.lr_controller)
        self.experience_buffer = GraphExperienceBuffer(
            capacity=config.buffer_capacity,
            memory_budget_gb=config.experience_memory_gb
        )
        self.policy_manager = PolicyManager()
        self.policy_validator = PolicyValidator()
        self.health_monitor = SimicHealth()

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return self.health_monitor.get_health_status()

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return Response(
                generate_latest(REGISTRY),
                media_type="text/plain"
            )

        @self.app.post("/train")
        async def trigger_training():
            """Manually trigger training iteration"""
            experiences = self.experience_buffer.sample(self.config.batch_size)
            if not experiences:
                return {"status": "insufficient_data"}

            metrics = await self.trainer.train_policy(experiences)
            return {"status": "success", "metrics": metrics}

        @self.app.get("/policy/current")
        async def get_current_policy():
            """Get current policy version"""
            return {
                "version": self.policy_manager.current_version,
                "history": self.policy_manager.version_history[-10:]
            }

        @self.app.post("/policy/rollback/{version}")
        async def rollback_policy(version: str):
            """Rollback to specific policy version"""
            success = self.policy_manager.rollback(version)
            return {"status": "success" if success else "failed"}
```

## History & Context

### Version History
- **v1.0** (2024-12-01): Initial design with basic RL training
- **v2.0** (2025-01-05): C-016 External Review fixes integrated
- **v3.0** (2025-01-10): Modularized architecture with Leyline integration

### Integration History
- **C-016 Integration** (2025-01-10): Complete integration of all critical fixes
- **Leyline Integration** (2025-01-10): Shared contracts adopted
- **UnifiedLRController** (2025-01-10): Learning rate invariants enforced

### Critical Fixes Applied
- **C-016-001**: Circuit breakers replace all assert statements
- **C-016-002**: Conservative mode for graceful degradation
- **C-016-003**: TTL-based memory management
- **C-016-004**: Chaos engineering validation
- **C-016-005**: Property-based testing for invariants

---

*Last Updated: 2025-01-15 | Next Review: 2025-07-15 | Owner: System Architecture Team*

**Note**: This unified design positions Simic as the critical learning engine that enables the Esper platform to continuously improve its morphogenetic capabilities through experience-driven policy optimization, with complete C-016 critical fixes integration ensuring production safety and reliability.
