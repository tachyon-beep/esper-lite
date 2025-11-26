# Weatherlight — Traceability Map

Design → Implementation (planned) → Evidence

- Oona messaging (contracts, HMAC)
  - Design: `docs/design/detailed_design/09-oona-unified-design.md`, `00-leyline-shared-contracts.md`
  - Impl: `OonaClient` (publish/consume, HMAC), group ensure, retries
  - Evidence: src/esper/oona/messaging.py

- Kernel prefetch (requests → ready/errors)
  - Design: kernel messages in Leyline bundle
  - Impl: `UrzaPrefetchWorker` run loop and checksum
  - Evidence: src/esper/urza/prefetch.py

- Kasmina attach on ready
  - Design: `02-kasmina-unified-design.md` (async attach)
  - Impl: `KasminaPrefetchCoordinator`, `KasminaSeedManager.process_prefetch_ready`
  - Evidence: src/esper/kasmina/prefetch.py, src/esper/kasmina/seed_manager.py

- Tamiyo policy consumer
  - Design: `03-tamiyo` series
  - Impl: `TamiyoService.consume_policy_updates`
  - Evidence: src/esper/tamiyo/service.py

- Observability (Nissa)
  - Design: `11-nissa` series
  - Impl: service runner entrypoint `esper-nissa-service`
  - Evidence: src/esper/nissa/service_runner.py, pyproject.toml

- Foundation orchestration
  - Design: HLD 001 (infrastructure plane), ADR‑001 (tight coupling, safety), ADR‑005 (accepted)
  - Impl: `src/esper/weatherlight/service_runner.py` (supervisor), console `esper-weatherlight-service`, compose at `infra/docker-compose.weatherlight.yml`
  - Evidence: service runner + entrypoint + compose run end‑to‑end
