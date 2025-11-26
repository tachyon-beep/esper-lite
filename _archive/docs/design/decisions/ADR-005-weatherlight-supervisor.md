# ADR-005: Weatherlight — Foundation Supervisor Service

 - Status: Accepted
- Date: 2025-09-21
- Driver: Esper‑Lite Working Group
- Deciders: Tech Lead, Infrastructure Owner, Oona/Urza/Kasmina Owners

## Context

Esper‑Lite currently runs subsystems (Oona, Urza, Tezzeret, Kasmina, Tamiyo, Nissa) as disparate components. We have client/worker scaffolds for messaging, prefetch, and policy consumption, plus an end‑to‑end demo script that wires them ad‑hoc. There is no single process that performs deterministic boot, health/backoff, and graceful shutdown across these parts. This raises operational risk and complicates developer workflows.

## Decision

Introduce a small foundation supervisor service named “Weatherlight” that boots and coordinates the core runtime subsystems without changing their internal logic or Leyline contracts.

Weatherlight responsibilities (prototype scope):
- Initialise Oona and ensure consumer groups for required streams.
- Start Urza’s prefetch worker (requests → ready/errors) with backoff.
- Start Kasmina’s prefetch coordinator and route ready/error events to SeedManager.
- Start Tamiyo’s policy‑update consumer loop.
- Periodically publish a foundation telemetry summary (counts, backoff, uptime) to Oona’s telemetry stream.
- Handle graceful shutdown on signals.

Out of scope (prototype):
- Nissa remains its own service (already has an entrypoint).
- No new Leyline messages or enums; Weatherlight uses existing contracts only.

## Consequences

- A single command (or compose file) can bring the prototype up locally.
- Clear ownership for async kernel fetch orchestration and policy consumption.
- Minimal code: primarily composition of existing classes; avoids tech debt in subsystems.
- Future work can migrate Weatherlight into a proper deployment (systemd/K8s) without changing subsystem APIs.

## Alternatives considered

- Keep using the demo script for orchestration (rejected: not a daemon, no backoff/telemetry).
- Push orchestration into one of the subsystems (rejected: violates separation of concerns and tightens coupling).

## References

- Prototype‑Delta package: `docs/prototype-delta/weatherlight/`
- Messaging client: `src/esper/oona/messaging.py`
- Prefetch worker: `src/esper/urza/prefetch.py`
- Kasmina coordinator/manager: `src/esper/kasmina/prefetch.py`, `src/esper/kasmina/seed_manager.py`
- Tamiyo service: `src/esper/tamiyo/service.py`
- Settings: `src/esper/core/config.py`
