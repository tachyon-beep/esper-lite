# Tezzeret Combined Design

---
File: docs/design/detailed_design/06-tezzeret-unified-design.md
---
# Tezzeret Unified Design (Esper-Lite)

## Snapshot
- **Role**: Compilation forge that takes predefined blueprints and emits compiled kernel artifacts for Urza. Runs asynchronously; no runtime blueprint generation.
- **Scope**: Poll Urza once at startup (and on demand if new templates are provided), compile using torch.compile pipelines on PyTorch 2.8, persist artifacts with WAL-backed durability, and report status.
- **Status**: Production—keeps C‑016 safety fixes (WAL, circuit breakers, conservative mode, TTL cleanup) while omitting Urabrask integration (not needed for known-good blueprints).

## Responsibilities
1. Discover blueprint definitions in Urza at startup (optionally on manual refresh) and enqueue them for compilation.
2. Run compilation pipelines (Fast/Standard/Aggressive/Emergency) to produce `CompiledKernelArtifact`s.
3. Store artifacts back in Urza with metadata, signing, and versioning.
4. Maintain WAL for crash recovery, circuit breakers for graceful degradation, and telemetry for observability.

## Component Map
| Component | Focus | Reference |
| --- | --- | --- |
| PollingEngine | Blueprint discovery, queue management | `06.1-tezzeret-compilation-internals.md` |
| CompilationCore | torch.compile orchestration, sandboxing | `06.1` |
| PipelineManager | Fast/Standard/Aggressive/Emergency strategy selection | `06.1` |
| WALCheckpointManager | Write-ahead log, crash recovery | `06.1` |
| CircuitBreakerSystem | Replaces asserts, triggers conservative mode | `06.1` |
| TTLMemoryManager | Cache & queue cleanup | `06.1` |

## Simplifications for Esper-Lite
- No Urabrask validation or dynamic blueprint generation. Inputs are trusted static templates already vetted offline.
- Compilation runs primarily during startup; background recompilation can be scheduled manually but default pipeline is idle after initial pass.
- Emergency queue/Tamiyo listeners optional; kept for completeness but disabled by default.

## Workflow
1. **Startup**: Poll Urza for blueprint list → enqueue.
2. **Compile**: For each blueprint, pick pipeline (default Standard) → sandboxed torch.compile → monitor duration & resources.
3. **Persist**: Sign artifact, update Urza catalog, write WAL entry (BEGIN→DATA→COMMIT), emit telemetry.
4. **Recover**: On crash, use WAL to resume in-progress jobs; circuit breakers ensure safe restart.
5. **Conservative Mode**: Triggered by repeated failures or resource pressure; throttles concurrency and switches to Fast pipeline.

## Safety & Operations
- Circuit breakers guard memory usage, GPU utilisation, and compilation timeouts.
- Conservative mode lowers concurrency to 1, disables aggressive pipeline, and lengthens thresholds.
- WAL uses O_DSYNC (or fsync fallback) with 256-byte headers + CRC for data integrity.
- TTL cleanup on caches and queues prevents memory creep; chaos tests verify restart scenarios.

## Performance Targets
| Metric | Target | Notes |
| --- | --- | --- |
| Standard compile time | 75–250 s | First pass at startup. |
| Fast compile time | 20–65 s | Used in conservative mode. |
| Aggressive compile time | 250–960 s | Optional, disabled in conservative mode. |
| Emergency compile time | 4–10 s | CPU-only fallback. |
| Concurrency | ≤2 jobs (1 in conservative mode) | Configurable. |
| GPU utilisation | ≤25 % | Breaker threshold. |
| WAL recovery | <12 s | Tested via chaos suite. |

## Integrations
- **Urza**: Blueprint queries (`BlueprintQuery`), artifact storage (`CompiledKernelArtifact`), metadata updates.
- **Oona/Nissa**: Telemetry events (`TelemetryPacket`), compilation status, breaker states.
- **Configuration**: YAML similar to original—`max_concurrent_jobs`, `default_strategy`, breaker thresholds, WAL path, TTL intervals.

## References
- `docs/design/detailed_design/06.1-tezzeret-compilation-internals.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

---
File: docs/design/detailed_design/06.1-tezzeret-compilation-internals.md
---
# Tezzeret Compilation Internals (Esper-Lite)

## Scope
Implementation details for compiling a fixed set of blueprints into kernel artifacts. Retains WAL durability, circuit breakers, conservative mode, and TTL cleanup; removes Urabrask dependencies. All pipelines assume PyTorch 2.8 as the torch.compile baseline.

## Core Orchestrator
```python
class TezzeretCompilationForge:
    def __init__(self):
        self.polling_engine = PollingEngine()
        self.compilation_core = CompilationCore()
        self.pipeline_manager = PipelineManager()
        self.circuit_breakers = {
            'memory': CircuitBreaker('memory', failure_threshold=3, timeout_ms=30_000),
            'gpu': CircuitBreaker('gpu', failure_threshold=5, timeout_ms=60_000),
            'timeout': CircuitBreaker('timeout', failure_threshold=2, timeout_ms=300_000),
        }
        self.conservative_mode = ConservativeModeController()
        self.wal_checkpoint_manager = WALCheckpointManager()
        self.crash_recovery = CrashRecoveryManager()
        self.resource_monitor = ResourceMonitor()
        self.memory_manager = TTLMemoryManager()
        self.urza_client = UrzaStorageClient()
        self.message_bus = VersionedOonaMessageBus('tezzeret')
        self.compilation_cache = CompilationCacheWithTTL()
        self.deduplication_engine = BlueprintDeduplicator()
```
- `PollingEngine` fetches blueprint list from Urza at startup; refresh interval configurable.
- `CompilationCore` performs sandboxed torch.compile, gathers metrics, signs artifacts.
- `PipelineManager` selects strategy based on blueprint priority and resource state.
- Breakers trigger conservative mode; conservative mode reduces concurrency and selects Fast pipeline.

## WAL & Crash Recovery
- WAL entries use 256-byte headers (magic/version/endian/crc) and O_DSYNC writes; fsync fallback when unavailable.
- Transactions: BEGIN (execution context) → DATA segments → COMMIT/ABORT.
- On restart, crash recovery scans WAL, resumes in-flight jobs, or requeues incomplete ones.

## Compilation Pipelines
| Strategy | Use Case | Notes |
| --- | --- | --- |
| Fast | Conservative mode, emergency rebuilds | CPU-friendly, minimal optimisation. |
| Standard | Default startup compilation | Balanced torch.compile flags. |
| Aggressive | Optional high-optimisation | Disabled when conservative mode active. |
| Emergency | CPU-only fallback | Short jobs when GPU unavailable. |

- Each pipeline records latency (`tezzeret_compilation_duration_ms`) and resource usage.
- Aggressive pipeline optional; configure `enable_aggressive_pipeline` flag.

## Resource Management
- **GPU**: Utilisation tracked; breaker trips >25 % sustained → conservative mode.
- **Memory**: TTL cleanup for caches and queues every 300 s; forced cleanup when budget exceeded.
- **Concurrency**: Max 2 jobs (configurable); conservative mode drops to 1.

## Telemetry & Metrics
- `tezzeret.compilation.duration_ms{strategy}`, `tezzeret.breaker.state`, `tezzeret.conservative_mode_active`, `tezzeret.wal.transactions_total`, `tezzeret.cache.hit_rate`.
- Logging includes blueprint id, strategy, duration, artifact checksum, breaker events.
- Health endpoint surfaces concurrency, queue depth, last WAL checkpoint, conservative mode flag.

## Configuration Notes
```yaml
tezzeret:
  compilation:
    max_concurrent_jobs: 2
    default_strategy: STANDARD
    enable_aggressive_pipeline: false
  wal:
    path: /var/lib/esper/tezzeret/wal
    durability_mode: O_DSYNC
  conservative_mode:
    gpu_threshold_percent: 20
    threshold_duration_minutes: 5
  memory_management:
    cleanup_interval_seconds: 300
  polling:
    initial_full_scan: true
    refresh_interval_minutes: 0  # 0 disables periodic refresh
```

Tezzeret thus acts as a startup compilation service with robust recovery and observability, ensuring Urza holds ready-to-use kernels for Esper-Lite.

