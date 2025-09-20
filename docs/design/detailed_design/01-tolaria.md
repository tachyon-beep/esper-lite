# Tolaria Combined Design

---
File: docs/design/detailed_design/01-tolaria-unified-design.md
---
# Tolaria Unified Design (v3.1)

## Snapshot
- **Role**: Training orchestrator and stability authority for Esper-Lite
- **Scope**: Core training loop, optimizer governance, rollback orchestration
- **Status**: Implementation-ready (C-016 safety fixes)
- **Key Budgets**: 18 ms epoch boundary, 500 ms fast rollback, 12 s full rollback

## Architectural Spine
- **Epoch-Driven Core**: Training loop synchronises all major work at epoch boundaries; morphogenetic operations must not pause batch processing.
- **Tight Kasmina/Tamiyo Coupling**: Direct calls for low latency in critical paths (no message bus); Tolaria remains final arbiter for rollbacks.
- **Consistent State Surfaces**: Model, optimizer, controller, and checkpoint metadata move in lockstep. UnifiedLRController holds exclusive LR mutation rights.

## Component Map
| Component | Core Responsibility | Notes |
| --- | --- | --- |
| Training Loop Manager | Batch/epoch orchestration, telemetry, Tamiyo handshake | Coordinates checkpoints and telemetry exports. See `01.1`. |
| UnifiedLRController | Single source of truth for all learning-rate changes | Circuit breakers replace asserts; monitors integrity at runtime. See `01.3`. |
| Dynamic Optimizer Manager | Rebuilds optimizers while preserving momentum | Registers all groups with LR controller. |
| Checkpoint + WAL System | Atomic persistence with O_DSYNC semantics | Stores model, optimizer, and telemetry metadata. See `01.2`. |
| Rollback Stack | 500 ms fast path + 12 s full recovery | Shared-memory signalling for CRITICAL events. |
| Integration Layer | Protocol Buffer wiring, telemetry, performance targets | Aligns with Leyline contracts. See `01.4`. |
| Profiling Harness | Capture epoch and batch timings with PyTorch profiler | `scripts/profile_tolaria.py` emits `epoch_time_ms`, optional chrome traces. |

## Control Loop Contract
1. **Train Epoch**: Run batches, aggregate multi-seed gradients, step optimizer via LR controller.
2. **Checkpoint Boundary**: If scheduled, persist a checkpoint, rotate rollback slots, and emit checkpoint metadata for downstream telemetry.
3. **End-of-Epoch Hook** (≤18 ms): Validate model, assemble `SystemStatePacket`, invoke `Tamiyo.step()` with 2 s timeout, process `AdaptationCommand`, schedule async checkpoint.
4. **Stability Enforcement**: Circuit breakers swap Tolaria into conservative mode on timing or integrity violations; rollback stack ready for CRITICAL/SEVERE events.

## Integration Summary
| Subsystem | Interface | Key Contracts |
| --- | --- | --- |
| Tamiyo | End-of-epoch call | `SystemStatePacket`, `AdaptationCommand` |
| Kasmina | Direct kernels & telemetry | Checkpoint metadata, telemetry alerts |
| Oona / Nissa | Telemetry & tracing | `EventEnvelope`, `TelemetryPacket` |

Contracts, enums, and limits originate from Leyline (`leyline.*` namespace) to guarantee binary compatibility and <80 µs serialization.

## Reliability & Performance
- **Epoch boundary**: 3.5 ms state assembly + 12 ms Tamiyo inference + 1.5 ms adaptation processing + 1 ms guard.
- **Rollback**: Fast path uses LRU checkpoint cache, shared-memory signalling; full path uses WAL-protected checkpoints and deterministic subsystem order (Tamiyo → Kasmina → Tolaria → auxiliaries).
- **Monitoring**: Metrics for timing budgets, rollback latency, LR integrity; conservative mode downgrades experimental features when circuit breakers trip.

## Critical Decisions
1. **Keep Kasmina/Tamiyo synchronous** to meet latency targets.
2. **Exclusive LR controller** prevents mutation races and enforces invariants.
3. **Two-tier rollback** balances sub-second containment with full recovery guarantees.
4. **Modular docs**: `01.1`–`01.4` capture detailed algorithms, configs, and APIs; this overview anchors the architecture.

## Forward Look
- Phase 2 distributed training + schema evolution once multi-node runs begin.
- Optional dynamic contract discovery (post-Leyline Phase 3) for heterogenous deployments.

## References
- `docs/design/detailed_design/01.1-tolaria-epoch-lifecycle.md`
- `docs/design/detailed_design/01.2-tolaria-rollback-systems.md`
- `docs/design/detailed_design/01.3-tolaria-optimizer-lr.md`
- `docs/design/detailed_design/01.4-tolaria-integration-protocols.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

---
File: docs/design/detailed_design/01.1-tolaria-epoch-lifecycle.md
---
# Tolaria – Epoch Lifecycle

## Scope
- Owns the core training loop, epoch boundary orchestration, and multi-seed gradient aggregation.
- Version 3.0 components retained; wording trimmed to the behaviours that affect system contracts and timing guarantees.

## Training Epoch Skeleton
```python
class TolariaTrainer:
    def train_epoch(self, epoch: int):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            host_loss, seed_infos = self._forward(batch)
            total_loss, telemetry = self.gradient_aggregator.aggregate_losses(
                host_loss, seed_infos, self.batch_size, epoch, self.total_epochs
            )
            self.gradient_aggregator.backward_with_aggregation(...)
            self.unified_lr_controller.step(epoch, {'host_loss': host_loss.item()})
            self.optimizer.step()
            self.telemetry.collect_step_metrics(...)

            if self._is_checkpoint_boundary(epoch, batch_idx):
                self.checkpoint_coordinator.handle_boundary(
                    epoch, batch_idx
                )

        self.end_of_epoch_hook()
```
- **Non-blocking rule**: telemetry and checkpoint coordination are async to keep batches flowing.
- **LR discipline**: only `UnifiedLRController` mutates learning rates.

## End-of-Epoch Hook (≤18 ms)
```python
def end_of_epoch_hook(self):
    start = perf_counter()
    metrics = self.validate()

    packet = SystemStatePacket(
        current_epoch=self.current_epoch,
        validation_accuracy=metrics['accuracy'],
        validation_loss=metrics['loss'],
        hardware_context=self.hardware_context,
        training_metrics=self.epoch_metrics,
    )
    serialized = self.serializer.serialize(packet)

    try:
        command = self.tamiyo_timeout_wrapper.step(serialized, timeout_override=2.0)
    except TamiyoTimeoutError:
        command = self._fallback_command(packet)

    self._apply_adaptation(command)
    if self.should_checkpoint():
        asyncio.create_task(self.checkpoint_coordinator.save_async())

    MonotonicTimer.within_budget_ms(start, 18.0, lambda d: self._timing_violation('epoch', d))
```
- **Budget split**: 3.5 ms state assembly, 12 ms Tamiyo inference, 1.5 ms command handling, 1 ms guard.
- Circuit breakers flip Tolaria into conservative mode instead of raising asserts.

## Multi-Seed Gradient Aggregation
Mathematics retained to preserve tuning authority.
```python
class MultiSeedGradientAggregator:
    def aggregate_losses(self, host_loss, seed_infos, host_batch_size, epoch, total_epochs):
        total = host_loss.clone()
        telemetry = {'host_loss': host_loss.item(), 'num_active_seeds': 0, 'total_weight': 1.0}

        for seed in seed_infos:
            weight = self._weight(seed, host_batch_size, epoch, total_epochs)
            if weight <= 0:
                continue
            ratio = seed.batch_size / host_batch_size
            total += weight * seed.loss * ratio
            telemetry['num_active_seeds'] += 1
            telemetry['total_weight'] += weight

        return self._stabilise(total, telemetry), telemetry
```
- **State weights**: Dormant 0, Germinated 0.01, Training 0.1, Grafting α(t), Stabilisation 0.5, FineTune 1.0.
- **Grafting α(t)**: `σ((epoch/total_epochs – 0.5) * 2πτ)` smoothing (sigmoid) for gentle blending.
- **Conflict resolution**: PCGrad projections keep gradients orthogonal when conflicts appear.

## Timing & Memory Guards
| Component | Target | Notes |
| --- | --- | --- |
| Epoch boundary | 18 ms | Hard stop; violation flips to conservative mode. |
| Telemetry enqueue | <2 ms | Emergency events bypass queue. |
| Checkpoint boundary | Non-blocking | Coordination handled asynchronously. |
| Gradient aggregation | Batch budget | Pre-allocated buffers for active seeds. |

## Inputs & Outputs
- **Inputs**: PyTorch `DataLoader`, host model with embedded Kasmina layers, seed descriptors, Leyline contracts.
- **Outputs**: `SystemStatePacket` to Tamiyo, telemetry via Oona, checkpoint tasks to rollback stack.
- **Dependencies**: UnifiedLRController (`01.3`), rollback systems (`01.2`), protobuf infrastructure (`01.4`).

The behaviours above remain the contract for downstream teams while stripping the previous template verbosity.

---
File: docs/design/detailed_design/01.2-tolaria-rollback-systems.md
---
# Tolaria – Rollback & Emergency Systems

## Scope
Two-tier rollback (500 ms fast path, 12 s full recovery) backed by durable checkpoints and four-level emergency escalation. Version 3.0 logic retained; wording trimmed around operational guarantees.

## Fast Rollback (≤500 ms)
```python
class FastRollbackCoordinator:
    def __init__(self):
        self.prepare_timeout_ms = 150
        self.commit_timeout_ms = 250
        self.confirm_timeout_ms = 100
        self.checkpoint_cache = LRU(max_items=5)
        self.emergency_signal = SharedMemorySignal('esper_emergency_fast')

    async def initiate_fast_rollback(self, checkpoint_epoch: int, reason: str) -> bool:
        start = perf_counter()
        self.checkpoint_cache.gc_if_needed()
        await self._prepare(checkpoint_epoch)
        await self._commit(checkpoint_epoch)
        await self._confirm()
        elapsed = (perf_counter() - start) * 1000
        MonotonicTimer.within_budget_ms(start, 500.0, lambda d: self._budget_violation(d))
        self.telemetry.publish('fast_rollback_complete', success=True, elapsed_ms=elapsed, reason=reason)
        return True
```
- In-memory checkpoints (last five) enable sub-second restoration.
- Circuit-breaker path escalates to full rollback if any phase fails.

## Four-Level Emergency Protocol
| Level | Response | Deadline | Actions |
| --- | --- | --- | --- |
| CRITICAL | Freeze | 100 ms | Shared-memory broadcast, await core acks |
| SEVERE | Fast rollback | 500 ms | Trigger fast coordinator, alert operators |
| MAJOR | Full rollback | 2 s | Initiate distributed rollback, capture checkpoint |
| MINOR | Tune | 5 s | Log incident, adjust parameters |

Critical path bypasses message bus; uses shared memory handshake with Tolaria/Kasmina/Tamiyo.

## Distributed Rollback Coordinator
```python
class DistributedRollbackCoordinator:
    def initiate_rollback(self, checkpoint_epoch: int, reason: str, severity: str = 'HIGH'):
        checkpoint = self.checkpoints.get(checkpoint_epoch)
        request = RollbackRequest(
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_hash=self._hash(checkpoint),
            severity=severity,
            affected_subsystems={'tolaria', 'kasmina', 'tamiyo'},
            recovery_deadline=utcnow() + timedelta(seconds=30)
        )
        await self._prepare(request)
        await self._commit(request)       # Deterministic order: Tamiyo → Kasmina → Tolaria → auxiliaries
        await self._complete(request)
```
- **Quorum**: 2/3 threshold across core + auxiliary subsystems.
- **Heartbeat**: 500 ms while rollback active.
- **Abort** path restores previous state and raises conservative mode.

## WAL & Checkpoint Durability
```python
class CheckpointWAL:
    WAL_MAGIC = b'ESPERWAL'
    ENTRY_HEADER_SIZE = 256

    def __init__(self, path, filesystem='ext4'):
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, 'O_DSYNC'):  # C-016 fix
            flags |= os.O_DSYNC
        self.fd = os.open(path, flags, 0o644)

    def begin_checkpoint(self, checkpoint_id, epoch):
        self._write_entry('BEGIN', checkpoint_id, epoch, execution_context=...)

    def commit_checkpoint(self, checkpoint_id):
        if not self._verify_integrity(checkpoint_id):
            return self.abort_checkpoint(checkpoint_id)
        self._write_entry('COMMIT', checkpoint_id)
        os.fsync(self.fd)  # Fallback if O_DSYNC absent
        return True
```
- Execution context (model hash, toolchain, hardware fingerprint) stored in WAL header.
- 1 GB chunking supports large tensor writes; WAL rotates at 100 MB.

## Performance & Memory Guardrails
| Item | Target | Notes |
| --- | --- | --- |
| Fast rollback | 500 ms | PREPARE 150 ms • COMMIT 250 ms • CONFIRM 100 ms |
| Full rollback | 12 s | Includes checkpoint load, model/optimizer/Tamiyo restore |
| WAL write | 10 ms | Synchronous writes using O_DSYNC/fsync |
| Checkpoint cache | ≤5 entries | LRU eviction + GC every 5 min |

## Outputs & Telemetry
- `fast_rollback_complete`, `full_rollback_complete`, and circuit-breaker metrics feed Nissa dashboards.
- Emergency events share reason, participants, elapsed time, and success (boolean) for post-mortems.

The condensed spec keeps the exact rollback semantics, timing budgets, and durability requirements needed by engineering and operations teams.

---
File: docs/design/detailed_design/01.3-tolaria-optimizer-lr.md
---
# Tolaria – Optimizer & Learning-Rate Control

## Scope
Covers momentum-preserving optimizer rebuilds and the C-016 UnifiedLRController (exclusive LR authority). Trimmed to core invariants and algorithms required for implementation.

## Dynamic Optimizer Preservation
```python
class DynamicOptimizerStatePreserver:
    def __init__(self, lr_controller: UnifiedLRController):
        self.lr_controller = lr_controller

    def preserve(self, old_optimizer, new_model, old_model, mappings=None):
        state = old_optimizer.state_dict()
        config = {k: v for k, v in state['param_groups'][0].items() if k != 'params'}
        new_opt = type(old_optimizer)(new_model.parameters(), **config)

        self.lr_controller.unregister_optimizer('primary')

        mappings = mappings or self._auto_map(old_model, new_model)
        transformed_state = self._transform_state(state['state'], mappings, type(old_optimizer))
        self._apply_state(new_opt, transformed_state)

        primary_config = GroupConfig(
            name='primary', policy=LRPolicy.HOST_COSINE,
            base_lr=config['lr'], group_id='host_model'
        )
        self.lr_controller.register_optimizer('primary', new_opt, primary_config)
        self._validate(new_opt, old_optimizer)
        return new_opt
```
- Handles expand/contract/reshape scenarios, keeping Adam/RMSprop momentum tensors consistent.
- New parameters initialised via LR controller to respect warm-up and reduced base LR.

## UnifiedLRController (Exclusive LR Entry Point)
```python
class UnifiedLRController:
    def __init__(self, config, enable_guards=True):
        self.optimizers = {}
        self.schedulers = {}
        self.group_configs = {}
        self.last_verified_lrs = {}
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)

    def register_optimizer(self, name, optimizer, config: GroupConfig):
        if name in self.optimizers:
            raise ValueError(f'{name} already registered')
        for idx, group in enumerate(optimizer.param_groups):
            group_id = f"{config.group_id or name}_{idx}"
            self._register_param_group(name, idx, group_id, group)
        self.optimizers[name] = optimizer
        self.group_configs[name] = config
        self.schedulers[name] = self._create_scheduler(optimizer, config)
        self._snapshot_lr_state(name, optimizer)
        self._compute_param_checksums(name, optimizer)

    def step(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, float]:
        start = perf_counter()
        if self.circuit_breaker.is_closed():
            try:
                self._verify_lr_integrity()
            except LRIntegrityError as err:
                self.circuit_breaker.record_failure()
                self._enter_conservative_mode(str(err))

        applied = {}
        for name, optimizer in self.optimizers.items():
            policy = self.group_configs[name].policy
            scheduler = self.schedulers[name]
            if policy == LRPolicy.MORPHO_ADAPTIVE:
                scheduler.step(metrics.get('loss', 0.0))
            elif policy != LRPolicy.FROZEN:
                scheduler.step()
            applied[name] = optimizer.param_groups[0]['lr']
            self._snapshot_lr_state(name, optimizer)

        MonotonicTimer.within_budget_ms(start, 5.0, lambda d: self._timing_violation('lr_step', d))
        self._record_history(epoch, applied)
        return applied
```
- **Invariant L1**: All LR mutations flow through `step()`; schedulers elsewhere are forbidden.
- **Invariant L2**: Optimizer state must survive architecture changes (validated after preservation).
- **Invariant L3**: Runtime integrity guard compares current vs last verified LR with relative epsilon tolerance (prevents false positives below 1e-6).
- Circuit breaker toggles conservative mode (no crash) if integrity fails or timing budgets are exceeded.

## Adding Morphogenetic Parameters
```python
def add_new_parameters(self, optimizer, new_params):
    base = self.group_configs.get('primary').base_lr
    config = GroupConfig(
        name=f'morpho_group_{len(self.group_configs)}',
        policy=LRPolicy.MORPHO_ADAPTIVE,
        base_lr=base * 0.1,
        warmup_epochs=10,
        warmup_factor=0.01,
        group_id='morpho'
    )
    optimizer.add_param_group({'params': new_params, 'lr': config.base_lr * config.warmup_factor})
    self.schedulers[config.name] = self._create_scheduler_for_group(optimizer, config, len(optimizer.param_groups) - 1)
    self.group_configs[config.name] = config
    self._snapshot_lr_state(config.name, optimizer)
```
- Guarantees differential LR + warm-up for new seeds.
- Scheduler choice depends on `LRPolicy` (cosine for host, adaptive for morpho groups, frozen for fossilised parameters).

## Operational Budgets
| Operation | Target | Notes |
| --- | --- | --- |
| LR step (all groups) | ≤5 ms | Includes integrity verification and history snapshot. |
| State preservation | ≤100 ms | Non-critical path; can spill outside batch timeframe. |
| Integrity check | ≤2 ms | Runs each call when circuit breaker closed. |
| Param addition | ≤20 ms | Warning logged if budget exceeded. |

## Outputs & Telemetry
- `applied_lrs`: dict per optimizer for Tamiyo/SRE debugging.
- `lr_history`: stored snapshots (cap 1000) for post-mortem reconstruction.
- `conservative_mode_triggered`: counter increments when guard trips.

This abridged document keeps the full invariant surface and the algorithms teams rely on, minus the template noise.

---
File: docs/design/detailed_design/01.4-tolaria-integration-protocols.md
---
# Tolaria – Integration & Infrastructure

## Scope
Summarises the integration points, performance targets, and shared infrastructure that Tolaria relies on. Derived from v3.0 (C-008 + C-016 updates), trimmed to the contracts and guarantees downstream teams must honour.

## Shared Contracts & Configuration
- All cross-subsystem messages come from Leyline (`leyline.*` namespace):
  - `SystemStatePacket`, `AdaptationCommand`, `TelemetryPacket`, `HardwareContext`, `StructuralPruning*`.
- Leyline also supplies circuit-breaker defaults, timeout ceilings, and delivery guarantees; Tolaria consumes them without local overrides.

## Performance Targets (Hardware-Realistic)
| Operation | Budget | Notes |
| --- | --- | --- |
| Epoch boundary | 18 ms | 12 ms Tamiyo GNN + 3.5 ms state assembly + 1.5 ms command handling + 1 ms guard. |
| Training overhead | ≤7.5 % | Morphogenetic features + telemetry combined. |
| Checkpoint creation (1 B params) | 10 s | Model 4.2 s, optimizer 2.8 s, others 3.0 s. |
| Rollback | 500 ms fast / 12 s full | Matches rollback subsystem spec. |
| Memory overhead | ≤8.5 % | Includes morphogenetic state + telemetry buffers. |
| Loss explosion threshold | 15× | Above this triggers SEVERE response. |

## Serialization & Versioning
```python
class SystemStateSerializer:
    CURRENT_VERSION = 1

    @staticmethod
    def serialize(packet: SystemStatePacket) -> bytes:
        pb = system_state_pb2.SystemStatePacket()
        pb.version = SystemStateSerializer.CURRENT_VERSION
        pb.current_epoch = packet.current_epoch
        pb.validation_accuracy = packet.validation_accuracy
        pb.validation_loss = packet.validation_loss
        pb.timestamp_ns = time.time_ns()
        pb.hardware_context.device_type = packet.hardware_context.device_type
        for key, value in packet.training_metrics.items():
            pb.training_metrics[key] = value
        for seed in packet.seed_states:
            pb_seed = pb.seed_states.add()
            pb_seed.seed_id = seed.seed_id
            pb_seed.gradient_norm = seed.gradient_norm
            pb_seed.learning_rate = seed.learning_rate
        return pb.SerializeToString()
```
- `deserialize()` rejects future versions; Tolaria depends on Leyline governance to bump versions safely.

## Optimizer Rebuild Coordination
```python
class OptimizerRebuildCoordinator:
    def __init__(self, optimizer, model, lr_controller, max_time_ms=10):
        self.optimizer = optimizer
        self.model = model
        self.lr_controller = lr_controller
        self.max_time_ms = max_time_ms

    async def handle_parameter_addition(self, layer_id, old_shape, new_shape, param_name):
        start = perf_counter()
        state = self.optimizer.state_dict()
        new_params = [p for n, p in self.model.named_parameters()
                      if layer_id in n and param_name in n and p.shape == new_shape]
        if not new_params:
            return False
        try:
            self.lr_controller.add_new_parameters(self.optimizer, new_params)
        except Exception:
            self.optimizer.load_state_dict(state)
            raise
        elapsed = (perf_counter() - start) * 1000
        if elapsed > self.max_time_ms:
            logging.warning('Optimizer rebuild exceeded budget %.2fms', elapsed)
        return True
```
- Two-phase commit logic kept: snapshot state → attempt → rollback on failure.

## Memory & Telemetry
```python
class MemoryBudgetManager:
    BUDGETS = {'model': 0.40, 'optimizer': 0.25, 'gradients': 0.15,
               'checkpoints': 0.08, 'telemetry': 0.05, 'morphogenetic': 0.05, 'emergency': 0.02}
    GC_INTERVALS = {'checkpoints': 300, 'telemetry': 60, 'morphogenetic': 180}

    def allocate(self, component):
        return self.total_memory_gb * self.BUDGETS.get(component, 0.0)

    def trigger_gc_if_needed(self, component):
        if component not in self.GC_INTERVALS:
            return False
        if time.time() - self.last_gc[component] > self.GC_INTERVALS[component]:
            self._run_gc(component)
            self.last_gc[component] = time.time()
            return True
        return False
```
```python
class TelemetryCollector:
    def __init__(self, oona_bus, max_queue=10_000):
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.oona_bus = oona_bus
        self.dropped = 0

    async def collect(self, priority, source, event_type, payload):
        message = TelemetryMessage(time.time_ns(), priority, source, event_type, payload)
        if priority == TelemetryPriority.EMERGENCY:
            await self.oona_bus.send_immediate(message)
            return
        try:
            await asyncio.wait_for(self.queue.put(message), timeout=0.001)
        except (asyncio.TimeoutError, asyncio.QueueFull):
            self.dropped += 1
            if self.dropped % 1000 == 0:
                logging.warning('Dropped %d telemetry messages', self.dropped)
```
- Emergency telemetry bypasses queue to meet CRITICAL deadlines.

## Testing Expectations
- Integration tests enforce 18 ms epoch hook and 500 ms fast rollback budgets.
- Property-based tests validate momentum tensor reshaping and LR invariants (see `01.3`).

## Interfaces Consumed
| Subsystem | Channel | Purpose |
| --- | --- | --- |
| Oona | Async queue + immediate path | Telemetry, non-critical messaging |
| Kasmina | Direct calls | Parameter addition, telemetry exchange |
| Tamiyo | Synchronous call | End-of-epoch control loop |

This condensed integration spec keeps the critical contracts, budgets, and helper utilities required for Tolaria’s production footprint.

### Mission-Critical Behaviours (Authoritative Reference)

For full lifecycle logic, consult `docs/design/detailed_design/old/01-tolaria.md`. The following behaviours remain mandatory for Esper-Lite even in the pared-down implementation:

- **Epoch Handshake:** `TolariaTrainer.end_of_epoch_hook()` must assemble the `SystemStatePacket`, invoke `Tamiyo.step()` synchronously, and apply the returned `AdaptationCommand` within the 18 ms budget (see Old §"End-of-Epoch Contract").
- **Checkpoint & WAL Discipline:** `CheckpointCoordinator` maintains the rotating stack (`fast` vs `full` rollback) with WAL-backed atomicity, including optimizer/seed metadata (Old §"Checkpoint Lifecycle").
- **Rollback Stack:** Fast (<500 ms) and full (<12 s) rollback paths remain intact, honouring the deterministic unwind order Tamiyo → Kasmina → Tolaria → auxiliaries (Old §"Rollback Stack").
- **UnifiedLRController:** All optimizer mutations flow through the controller; Tolaria enforces integrity monitors and conservative-mode demotion on breaker events (Old §"Unified LR Governance").
- **Telemetry Surfaces:** Tolaria exports timing/rollback metrics over Oona using Leyline contracts to feed Tamiyo, Simic, and Nissa (Old §"Telemetry Contracts").

These behaviours constitute Tolaria’s “main job” and may not be dropped when creating lightweight builds; optional features (e.g., structured pruning) can remain disabled, but the control loop above is authoritative.
