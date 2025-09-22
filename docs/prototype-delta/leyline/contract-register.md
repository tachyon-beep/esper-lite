# Leyline Contract Register

## Purpose

This register inventories enums and dataclasses that exist outside the generated Leyline protobuf bindings. It will expand subsystem by subsystem to make hidden cross-coupling visible during the prototype delta review.

## src/esper/core

**Dataclasses**

- `TelemetryMetric` (`src/esper/core/telemetry.py:16`) — Represents a single metric sample added to Leyline telemetry packets; captures name, value, unit, and optional attributes.
- `TelemetryEvent` (`src/esper/core/telemetry.py:26`) — Describes discrete telemetry events with a Leyline `TelemetryLevel`, attributes, and optional event identifier.

**Enums**

- _None defined._

**Notes**

- `EsperSettings` (`src/esper/core/config.py:15`) derives from `pydantic.BaseSettings` rather than `dataclass`, so it is excluded from this register.
- Both dataclasses directly wrap `leyline_pb2.TelemetryPacket` primitives and should eventually be replaced by pure protobuf usage once helper ergonomics are addressed.
