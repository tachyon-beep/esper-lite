"""Shared fixtures for telemetry end-to-end tests.

Provides capture backends and hub setup for verifying telemetry flows
from source to nissa.
"""

import pytest
from typing import NamedTuple

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import NissaHub


class CaptureBackend:
    """Test backend that captures all emitted telemetry events.

    Implements the OutputBackend protocol for use with NissaHub.
    """

    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def start(self) -> None:
        """Called when backend is added to hub."""
        pass

    def close(self) -> None:
        """Called when hub is closed."""
        pass

    def emit(self, event: TelemetryEvent) -> None:
        """Capture event for later inspection."""
        self.events.append(event)

    def find_events(self, event_type: TelemetryEventType) -> list[TelemetryEvent]:
        """Filter captured events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def find_first(self, event_type: TelemetryEventType) -> TelemetryEvent | None:
        """Get first event of type, or None."""
        events = self.find_events(event_type)
        return events[0] if events else None

    def clear(self) -> None:
        """Clear captured events."""
        self.events.clear()


class CaptureHubResult(NamedTuple):
    """Result from capture_hub fixture."""

    hub: NissaHub
    backend: CaptureBackend


@pytest.fixture
def capture_hub() -> CaptureHubResult:
    """Create NissaHub with capture backend for testing.

    Usage:
        def test_something(capture_hub):
            hub, backend = capture_hub
            # ... trigger telemetry emission ...
            events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)

    Yields:
        CaptureHubResult with hub and backend.
    """
    hub = NissaHub()
    backend = CaptureBackend()
    hub.add_backend(backend)
    yield CaptureHubResult(hub, backend)
    hub.close()


@pytest.fixture
def capture_backend() -> CaptureBackend:
    """Standalone capture backend for direct callback injection.

    Usage:
        def test_component(capture_backend):
            component = SomeComponent(telemetry_cb=capture_backend.emit)
            component.do_something()
            events = capture_backend.find_events(...)
    """
    return CaptureBackend()
