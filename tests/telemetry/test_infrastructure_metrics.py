"""End-to-end tests for infrastructure metrics (TELE-700 to TELE-799).

Verifies infrastructure telemetry metrics flow from their source through to Nissa/Sanctum.

Infrastructure metrics fall into two categories:

1. **Emitted metrics** - Sent via telemetry events (tested end-to-end here):
   - TELE-750: group_id (A/B testing identifier)
   - TELE-760: compile_enabled (torch.compile status)
   - TELE-770: memory_usage_percent (computed from CUDA memory metrics)

2. **Aggregator-computed metrics** - Computed in Sanctum aggregator, not emitted:
   - TELE-701: connected (set on TRAINING_STARTED)
   - TELE-702: staleness_seconds (computed from last event timestamp)
   - TELE-703: training_thread_alive (Thread.is_alive() check)
   - TELE-710/711: epochs_per_second, batches_per_hour (throughput)
   - TELE-720/721: cpu_percent, ram_usage (psutil collection)
   - TELE-730: gpu_memory_usage (torch.cuda collection)
   - TELE-740/741: has_memory_alarm, memory_alarm_devices (computed properties)

This test file covers EMITTED metrics end-to-end. Aggregator-computed metrics
are tested in tests/karn/sanctum/test_aggregator.py and test_schema.py.
"""

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import TrainingStartedPayload, PPOUpdatePayload
from esper.karn.sanctum.schema import (
    SystemVitals,
    GPUStats,
    InfrastructureMetrics,
    TamiyoState,
    SanctumSnapshot,
)
from tests.telemetry.conftest import CaptureHubResult


# =============================================================================
# TELE-750: group_id (A/B Testing Identifier)
# =============================================================================


class TestTELE750GroupId:
    """TELE-750: group_id flows from TelemetryEvent to Sanctum."""

    def test_group_id_in_telemetry_event(self) -> None:
        """TELE-750: TelemetryEvent carries group_id field."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id="A",
            data=_minimal_ppo_payload(),
        )

        assert event.group_id == "A", "TelemetryEvent must carry group_id"

    def test_group_id_default_is_default(self) -> None:
        """TELE-750: group_id defaults to 'default' when not in A/B mode."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=_minimal_ppo_payload(),
        )

        assert event.group_id == "default", "Default group_id should be 'default'"

    def test_group_id_emitted_via_hub(self, capture_hub: CaptureHubResult) -> None:
        """TELE-750: group_id flows through NissaHub to backends."""
        hub, backend = capture_hub

        # Emit event with group_id="B"
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id="B",
            data=_minimal_ppo_payload(),
        )
        hub.emit(event)
        hub.flush()

        # Verify group_id captured
        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].group_id == "B"

    def test_group_id_valid_values(self) -> None:
        """TELE-750: group_id supports A/B/C values per design spec."""
        for group in ["A", "B", "C"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                group_id=group,
                data=_minimal_training_started_payload(),
            )
            assert event.group_id == group


# =============================================================================
# TELE-760: compile_enabled (torch.compile Status)
# =============================================================================


class TestTELE760CompileEnabled:
    """TELE-760: compile_enabled in TrainingStartedPayload."""

    def test_compile_enabled_in_training_started_payload(self) -> None:
        """TELE-760: TrainingStartedPayload has compile_enabled field."""
        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,
            task="cifar10",
            host_params=1_000_000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=5_000_000,
            policy_device="cpu",
            env_devices=("cpu",),
            reward_mode="shaped",
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="reduce-overhead",
        )

        assert payload.compile_enabled is True
        assert payload.compile_backend == "inductor"
        assert payload.compile_mode == "reduce-overhead"

    def test_compile_enabled_default_false(self) -> None:
        """TELE-760: compile_enabled defaults to False (eager mode)."""
        payload = _minimal_training_started_payload()

        assert payload.compile_enabled is False
        assert payload.compile_backend is None
        assert payload.compile_mode is None

    def test_compile_enabled_emitted_in_event(self, capture_hub: CaptureHubResult) -> None:
        """TELE-760: compile_enabled flows through TRAINING_STARTED event."""
        hub, backend = capture_hub

        payload = TrainingStartedPayload(
            n_envs=1,
            max_epochs=10,
            max_batches=10,
            task="test_task",
            host_params=100,
            slot_ids=("r0c0",),
            seed=1,
            n_episodes=10,
            lr=0.001,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=1000,
            policy_device="cpu",
            env_devices=("cpu",),
            reward_mode="shaped",
            compile_enabled=True,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.compile_enabled is True

    def test_infrastructure_metrics_compile_fields(self) -> None:
        """TELE-760: InfrastructureMetrics schema has compile fields."""
        infra = InfrastructureMetrics(
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="max-autotune",
        )

        assert infra.compile_enabled is True
        assert infra.compile_backend == "inductor"
        assert infra.compile_mode == "max-autotune"


# =============================================================================
# TELE-770: memory_usage_percent (GPU Memory Usage Percentage)
# =============================================================================


class TestTELE770MemoryUsagePercent:
    """TELE-770: memory_usage_percent computed property in InfrastructureMetrics."""

    def test_memory_usage_percent_computed_correctly(self) -> None:
        """TELE-770: memory_usage_percent = (allocated / reserved) * 100."""
        infra = InfrastructureMetrics(
            cuda_memory_allocated_gb=4.0,
            cuda_memory_reserved_gb=8.0,
        )

        assert infra.memory_usage_percent == 50.0

    def test_memory_usage_percent_zero_when_reserved_zero(self) -> None:
        """TELE-770: memory_usage_percent returns 0.0 when reserved is 0 (no GPU)."""
        infra = InfrastructureMetrics(
            cuda_memory_allocated_gb=0.0,
            cuda_memory_reserved_gb=0.0,
        )

        assert infra.memory_usage_percent == 0.0

    def test_memory_usage_percent_high_pressure(self) -> None:
        """TELE-770: memory_usage_percent > 90% indicates critical pressure."""
        infra = InfrastructureMetrics(
            cuda_memory_allocated_gb=9.5,
            cuda_memory_reserved_gb=10.0,
        )

        assert infra.memory_usage_percent == 95.0
        assert infra.memory_usage_percent > 90.0  # Critical threshold

    def test_memory_usage_percent_in_tamiyo_state(self) -> None:
        """TELE-770: memory_usage_percent accessible via TamiyoState.infrastructure."""
        tamiyo = TamiyoState()
        tamiyo.infrastructure.cuda_memory_allocated_gb = 6.0
        tamiyo.infrastructure.cuda_memory_reserved_gb = 10.0

        assert tamiyo.infrastructure.memory_usage_percent == 60.0


# =============================================================================
# TELE-740: has_memory_alarm (Memory Alarm Trigger)
# =============================================================================


class TestTELE740HasMemoryAlarm:
    """TELE-740: has_memory_alarm computed property in SystemVitals."""

    def test_has_memory_alarm_false_when_below_threshold(self) -> None:
        """TELE-740: has_memory_alarm is False when all devices < 90%."""
        vitals = SystemVitals(
            ram_used_gb=8.0,
            ram_total_gb=16.0,  # 50% usage
            gpu_memory_used_gb=4.0,
            gpu_memory_total_gb=10.0,  # 40% usage
        )

        assert vitals.has_memory_alarm is False

    def test_has_memory_alarm_true_on_ram_over_90(self) -> None:
        """TELE-740: has_memory_alarm triggers on RAM > 90%."""
        vitals = SystemVitals(
            ram_used_gb=15.0,
            ram_total_gb=16.0,  # 93.75% - triggers alarm
        )

        assert vitals.has_memory_alarm is True

    def test_has_memory_alarm_true_on_gpu_over_90(self) -> None:
        """TELE-740: has_memory_alarm triggers on GPU > 90%."""
        vitals = SystemVitals(
            ram_used_gb=8.0,
            ram_total_gb=16.0,  # 50% - OK
            gpu_stats={
                0: GPUStats(device_id=0, memory_used_gb=9.5, memory_total_gb=10.0),  # 95%
            },
        )

        assert vitals.has_memory_alarm is True

    def test_has_memory_alarm_uses_gpu_stats_dict(self) -> None:
        """TELE-740: has_memory_alarm checks gpu_stats dict for multi-GPU."""
        vitals = SystemVitals(
            gpu_stats={
                0: GPUStats(device_id=0, memory_used_gb=5.0, memory_total_gb=10.0),  # 50% OK
                1: GPUStats(device_id=1, memory_used_gb=9.2, memory_total_gb=10.0),  # 92% ALARM
            },
        )

        assert vitals.has_memory_alarm is True

    def test_has_memory_alarm_fallback_single_gpu_fields(self) -> None:
        """TELE-740: has_memory_alarm uses fallback fields when gpu_stats empty."""
        vitals = SystemVitals(
            gpu_stats={},  # Empty - use fallback fields
            gpu_memory_used_gb=9.5,
            gpu_memory_total_gb=10.0,  # 95% via fallback
        )

        assert vitals.has_memory_alarm is True


# =============================================================================
# TELE-741: memory_alarm_devices (Device List)
# =============================================================================


class TestTELE741MemoryAlarmDevices:
    """TELE-741: memory_alarm_devices list in SystemVitals."""

    def test_memory_alarm_devices_empty_when_healthy(self) -> None:
        """TELE-741: memory_alarm_devices is empty when all devices < 90%."""
        vitals = SystemVitals(
            ram_used_gb=8.0,
            ram_total_gb=16.0,
            gpu_stats={
                0: GPUStats(device_id=0, memory_used_gb=5.0, memory_total_gb=10.0),
            },
        )

        assert vitals.memory_alarm_devices == []

    def test_memory_alarm_devices_includes_ram(self) -> None:
        """TELE-741: memory_alarm_devices includes 'RAM' when RAM > 90%."""
        vitals = SystemVitals(
            ram_used_gb=15.0,
            ram_total_gb=16.0,  # 93.75%
        )

        assert "RAM" in vitals.memory_alarm_devices

    def test_memory_alarm_devices_includes_gpu_device_name(self) -> None:
        """TELE-741: memory_alarm_devices uses 'cuda:N' format for GPUs."""
        vitals = SystemVitals(
            gpu_stats={
                1: GPUStats(device_id=1, memory_used_gb=9.5, memory_total_gb=10.0),  # 95%
            },
        )

        assert "cuda:1" in vitals.memory_alarm_devices

    def test_memory_alarm_devices_multiple_devices(self) -> None:
        """TELE-741: memory_alarm_devices can list multiple alarming devices."""
        vitals = SystemVitals(
            ram_used_gb=15.0,
            ram_total_gb=16.0,  # 93.75% - RAM alarm
            gpu_stats={
                0: GPUStats(device_id=0, memory_used_gb=9.5, memory_total_gb=10.0),  # 95%
                1: GPUStats(device_id=1, memory_used_gb=9.2, memory_total_gb=10.0),  # 92%
            },
        )

        devices = vitals.memory_alarm_devices
        assert "RAM" in devices
        assert "cuda:0" in devices
        assert "cuda:1" in devices

    def test_memory_alarm_devices_fallback_cuda0(self) -> None:
        """TELE-741: memory_alarm_devices uses 'cuda:0' for fallback single-GPU."""
        vitals = SystemVitals(
            gpu_stats={},  # Empty - use fallback
            gpu_memory_used_gb=9.5,
            gpu_memory_total_gb=10.0,  # 95%
        )

        assert "cuda:0" in vitals.memory_alarm_devices


# =============================================================================
# TELE-701: connected (Backend Connection Status)
# =============================================================================


class TestTELE701Connected:
    """TELE-701: connected flag in SanctumSnapshot.

    Note: connected is set by aggregator on TRAINING_STARTED receipt,
    not emitted. These tests verify schema defaults and contract.
    """

    def test_connected_default_false(self) -> None:
        """TELE-701: connected defaults to False before training starts."""
        snapshot = SanctumSnapshot()

        assert snapshot.connected is False

    def test_connected_field_exists_in_snapshot(self) -> None:
        """TELE-701: SanctumSnapshot has connected field."""
        snapshot = SanctumSnapshot(connected=True)

        assert snapshot.connected is True


# =============================================================================
# TELE-702: staleness_seconds (Snapshot Staleness)
# =============================================================================


class TestTELE702StalenessSeconds:
    """TELE-702: staleness_seconds field in SanctumSnapshot.

    Note: staleness_seconds is computed in aggregator from last event timestamp.
    These tests verify schema defaults and contract.
    """

    def test_staleness_default_infinity(self) -> None:
        """TELE-702: staleness_seconds defaults to infinity (disconnected)."""
        snapshot = SanctumSnapshot()

        assert snapshot.staleness_seconds == float('inf')

    def test_staleness_field_settable(self) -> None:
        """TELE-702: staleness_seconds can be set to computed value."""
        snapshot = SanctumSnapshot(staleness_seconds=1.5)

        assert snapshot.staleness_seconds == 1.5


# =============================================================================
# TELE-703: training_thread_alive (Thread Status)
# =============================================================================


class TestTELE703TrainingThreadAlive:
    """TELE-703: training_thread_alive field in SanctumSnapshot.

    Note: training_thread_alive is checked via Thread.is_alive() in SanctumApp.
    These tests verify schema defaults and contract.
    """

    def test_training_thread_alive_default_none(self) -> None:
        """TELE-703: training_thread_alive defaults to None (unknown)."""
        snapshot = SanctumSnapshot()

        assert snapshot.training_thread_alive is None

    def test_training_thread_alive_tri_state(self) -> None:
        """TELE-703: training_thread_alive supports True/False/None."""
        # True = thread running
        snapshot_alive = SanctumSnapshot(training_thread_alive=True)
        assert snapshot_alive.training_thread_alive is True

        # False = thread dead
        snapshot_dead = SanctumSnapshot(training_thread_alive=False)
        assert snapshot_dead.training_thread_alive is False

        # None = unknown
        snapshot_unknown = SanctumSnapshot(training_thread_alive=None)
        assert snapshot_unknown.training_thread_alive is None


# =============================================================================
# TELE-710/711: Throughput Metrics
# =============================================================================


class TestTELE710EpochsPerSecond:
    """TELE-710: epochs_per_second in SystemVitals.

    Note: epochs_per_second is computed in aggregator, not emitted.
    These tests verify schema contract.
    """

    def test_epochs_per_second_default_zero(self) -> None:
        """TELE-710: epochs_per_second defaults to 0.0."""
        vitals = SystemVitals()

        assert vitals.epochs_per_second == 0.0

    def test_epochs_per_second_in_snapshot_vitals(self) -> None:
        """TELE-710: epochs_per_second accessible via snapshot.vitals."""
        vitals = SystemVitals(epochs_per_second=0.85)
        snapshot = SanctumSnapshot(vitals=vitals)

        assert snapshot.vitals.epochs_per_second == 0.85


class TestTELE711BatchesPerHour:
    """TELE-711: batches_per_hour in SystemVitals.

    Note: batches_per_hour is computed in aggregator, not emitted.
    These tests verify schema contract.
    """

    def test_batches_per_hour_default_zero(self) -> None:
        """TELE-711: batches_per_hour defaults to 0.0."""
        vitals = SystemVitals()

        assert vitals.batches_per_hour == 0.0

    def test_batches_per_hour_in_snapshot_vitals(self) -> None:
        """TELE-711: batches_per_hour accessible via snapshot.vitals."""
        vitals = SystemVitals(batches_per_hour=120.5)
        snapshot = SanctumSnapshot(vitals=vitals)

        assert snapshot.vitals.batches_per_hour == 120.5


# =============================================================================
# TELE-720: cpu_percent (CPU Utilization)
# =============================================================================


class TestTELE720CpuPercent:
    """TELE-720: cpu_percent in SystemVitals.

    Note: cpu_percent is collected via psutil in aggregator, not emitted.
    These tests verify schema contract.
    """

    def test_cpu_percent_default_zero(self) -> None:
        """TELE-720: cpu_percent defaults to 0.0."""
        vitals = SystemVitals()

        assert vitals.cpu_percent == 0.0

    def test_cpu_percent_nullable(self) -> None:
        """TELE-720: cpu_percent can be None (collection failure)."""
        vitals = SystemVitals(cpu_percent=None)

        assert vitals.cpu_percent is None


# =============================================================================
# TELE-721: ram_usage (System RAM)
# =============================================================================


class TestTELE721RamUsage:
    """TELE-721: ram_used_gb and ram_total_gb in SystemVitals.

    Note: RAM metrics are collected via psutil in aggregator, not emitted.
    These tests verify schema contract.
    """

    def test_ram_fields_default_zero(self) -> None:
        """TELE-721: RAM fields default to 0.0."""
        vitals = SystemVitals()

        assert vitals.ram_used_gb == 0.0
        assert vitals.ram_total_gb == 0.0

    def test_ram_fields_nullable(self) -> None:
        """TELE-721: RAM fields can be None (collection failure)."""
        vitals = SystemVitals(ram_used_gb=None, ram_total_gb=None)

        assert vitals.ram_used_gb is None
        assert vitals.ram_total_gb is None


# =============================================================================
# TELE-730: gpu_memory_usage (GPU Memory)
# =============================================================================


class TestTELE730GpuMemoryUsage:
    """TELE-730: GPU memory fields in SystemVitals and GPUStats.

    Note: GPU memory is collected via torch.cuda in aggregator, not emitted.
    These tests verify schema contract.
    """

    def test_gpu_stats_dict_structure(self) -> None:
        """TELE-730: gpu_stats is dict[int, GPUStats] for multi-GPU."""
        vitals = SystemVitals(
            gpu_stats={
                0: GPUStats(device_id=0, memory_used_gb=5.0, memory_total_gb=10.0),
                1: GPUStats(device_id=1, memory_used_gb=3.0, memory_total_gb=8.0),
            }
        )

        assert len(vitals.gpu_stats) == 2
        assert vitals.gpu_stats[0].memory_used_gb == 5.0
        assert vitals.gpu_stats[1].memory_total_gb == 8.0

    def test_gpu_stats_default_empty(self) -> None:
        """TELE-730: gpu_stats defaults to empty dict."""
        vitals = SystemVitals()

        assert vitals.gpu_stats == {}

    def test_gpu_fallback_fields(self) -> None:
        """TELE-730: single-GPU fallback fields exist for convenience."""
        vitals = SystemVitals(
            gpu_memory_used_gb=5.0,
            gpu_memory_total_gb=10.0,
        )

        assert vitals.gpu_memory_used_gb == 5.0
        assert vitals.gpu_memory_total_gb == 10.0


# =============================================================================
# Helper Functions
# =============================================================================


def _minimal_ppo_payload() -> PPOUpdatePayload:
    """Create minimal PPOUpdatePayload for tests."""
    return PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        kl_divergence=0.01,
        grad_norm=0.5,
        clip_fraction=0.1,
        nan_grad_count=0,
    )


def _minimal_training_started_payload() -> TrainingStartedPayload:
    """Create minimal TrainingStartedPayload for tests."""
    return TrainingStartedPayload(
        n_envs=1,
        max_epochs=10,
        max_batches=10,
        task="test_task",
        host_params=100,
        slot_ids=("r0c0",),
        seed=42,
        n_episodes=10,
        lr=0.0003,
        clip_ratio=0.2,
        entropy_coef=0.01,
        param_budget=1000,
        policy_device="cpu",
        env_devices=("cpu",),
        reward_mode="shaped",
    )
