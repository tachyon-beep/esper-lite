"""Tests for health monitoring telemetry emission."""


def test_memory_warning_emitted_when_threshold_exceeded():
    """Test that MEMORY_WARNING is emitted when GPU utilization exceeds threshold."""
    from esper.karn.health import HealthMonitor
    from esper.leyline import MemoryWarningPayload, TelemetryEventType

    events = []

    def capture(event):
        events.append(event)

    monitor = HealthMonitor(
        emit_callback=capture,
        memory_warning_threshold=0.8,
    )
    # Simulate 90% GPU utilization
    monitor._check_memory_and_warn(gpu_utilization=0.9, gpu_allocated_gb=10.0, gpu_total_gb=11.1)

    assert len(events) == 1
    assert events[0].event_type == TelemetryEventType.MEMORY_WARNING
    assert isinstance(events[0].data, MemoryWarningPayload)
    assert events[0].data.gpu_utilization == 0.9


def test_no_memory_warning_when_below_threshold():
    """Test that no warning is emitted when GPU utilization is below threshold."""
    from esper.karn.health import HealthMonitor

    events = []

    monitor = HealthMonitor(
        emit_callback=lambda e: events.append(e),
        memory_warning_threshold=0.8,
    )
    monitor._check_memory_and_warn(gpu_utilization=0.7, gpu_allocated_gb=7.0, gpu_total_gb=10.0)

    assert len(events) == 0
