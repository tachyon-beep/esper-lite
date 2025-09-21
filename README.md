# Esper-Lite Project Scaffold

Esper-Lite is a streamlined morphogenetic control stack centred on a PyTorch 2.8 runtime. This scaffold establishes the shared tooling, package layout, and subsystem contracts needed to begin feature development across Tolaria, Kasmina, Tamiyo, Simic, and the blueprint and infrastructure services.

## Getting Started

1. **Create a virtual environment**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev]
   ```

2. **Run quality gates**

   ```bash
   pytest tests
   pylint --rcfile .codacy/tools-configs/pylint.rc src/esper
   ruff check src tests
   mypy src/esper
   # CI runs the following focused suites as well:
   pytest -m "not integration and not performance"
   pytest -m integration
   pytest tests/leyline/test_contracts.py
   pytest -m "not performance" tests/leyline/test_serialization.py
   ```

3. **Launch local infrastructure (placeholder)**
   - Docker compose definitions live under `infra/`. Use `docker compose -f infra/docker-compose.redis.yml up -d` to start Redis bound to the host.
   - Redis, Prometheus, and Elasticsearch endpoints are documented in `docs/project/implementation_plan.md`.

### Redis for Oona integration tests

Spin up the local Redis instance used by the Oona integration suite:

```bash
docker compose -f infra/docker-compose.redis.yml up -d
pytest -m integration tests/oona/test_messaging_integration.py
docker compose -f infra/docker-compose.redis.yml down
```

Set `REDIS_URL` if you expose Redis on a non-default port.

Or run all integration tests with automatic Redis bring-up/teardown:

```bash
scripts/run_integration.sh
```

### Telemetry & Policy Streams

- `OONA_NORMAL_STREAM`, `OONA_EMERGENCY_STREAM`, `OONA_TELEMETRY_STREAM`, and `OONA_POLICY_STREAM` define the Redis Streams used for Tolaria system state, Tamiyo telemetry, and Simic policy updates. Defaults are provided in `.env.example`.
- Nissa exposes Prometheus metrics via the ASGI helper in `src/esper/nissa/server.py`. Run `uvicorn esper.nissa.server:create_app --factory` with a configured `NissaIngestor` to scrape metrics and feed dashboards/alerts.

### Observability Stack (Prometheus + Grafana + Elasticsearch)

Bring up the local observability stack and wire it to the Nissa runner:

```bash
docker compose -f infra/docker-compose.observability.yml up -d
esper-nissa-service  # in a separate shell
```

The compose file launches Prometheus (host networking), Grafana (port `3000`),
and a single-node Elasticsearch instance at `http://localhost:9200`. The
`esper-nissa-service` entry point (also accessible via
`scripts/run_nissa_service.py`) drains telemetry from Oona and exposes
`/metrics` and `/healthz` on `http://localhost:9100`.

Prometheus is preconfigured via `infra/prometheus/prometheus.yml` to scrape
`localhost:9100/metrics`. Grafana ships with provisioning in `infra/grafana/`
and is available on <http://localhost:3000> (admin/admin). Operational
procedures and teardown steps live in
`docs/project/observability_runbook.md`.

Always-on local infra (recommended)

```bash
# Start all containers (Redis, Prometheus, Grafana, Elasticsearch) and leave them running
scripts/start_all_infra.sh

# When you need to stop them
scripts/stop_all_infra.sh
```

All services are configured with `restart: unless-stopped`, so they’ll auto-restart on reboot.

### Leyline Contracts

- Protobuf definitions live under `contracts/leyline/leyline.proto`.
- Regenerate Python bindings with:

  ```bash
  scripts/generate_leyline.py
  ```

- Validate serialization semantics:

  ```bash
  pytest tests/leyline/test_serialization.py
  ESPER_RUN_PERF_TESTS=1 pytest tests/leyline/test_serialization.py -m performance  # optional
  ```

The generated files are stored in `src/esper/leyline/_generated/` and include
`.pyi` type stubs for static analysis.

### Simic Offline Training

- Field reports collected from Tamiyo populate the replay buffer used by Simic
  (`esper.simic.replay`).
- Run the smoke tests with:

  ```bash
  pytest tests/simic/test_replay.py tests/simic/test_trainer.py
  ```

- `scripts/run_demo.py` demonstrates the full loop: Tolaria → Tamiyo → Kasmina
  followed by Simic PPO training with feature-rich replay vectors (numeric,
  metric sequence, persistent embeddings, attention), optional LoRA adapters,
  and policy hot-reload via Oona. Tamiyo consumes the same shared features to
  produce multi-head actions which are observable through new telemetry fields.
- Simic automatically executes a validation harness after each training run;
  tweak thresholds via `SimicTrainerConfig` (e.g., `validation_min_reward`,
  `validation_max_policy_loss`) to suit your scenario. Policy updates are only
  published if validation succeeds, and failures emit telemetry warnings.
- Nissa now evaluates alert rules (`training_latency_high`,
  `kasmina_isolation_violation`, `oona_queue_depth`,
  `tezzeret_compile_retry_high`) and exposes a `/metrics/summary` endpoint with
  active alerts and SLO burn rates. Feed SLO metrics via telemetry using
  `slo.<name>_actual`/`slo.<name>_objective` pairs.

### Blueprint Catalog

- Karn ships with 50 pre-approved blueprints spanning safe (BP001–BP035),
  experimental (BP036–BP042), and high-risk quarantine (BP043–BP050) tiers.
  Instantiate `KarnCatalog()` to load the full library, or pass
  `load_defaults=False` in tests to start with an empty catalog. Use
  `choose_template(...)` for deterministic selection—context strings hash to
  stable templates and conservative mode restricts results to the safe pool.

- Kasmina uses those same blueprint ids when grafting seeds. The seed manager
  requests kernels from Urza via `UrzaRuntime.fetch_kernel`, enforces a
  10 ms latency budget, and automatically falls back to a conservative
  blueprint (`BP001` by default) or an identity kernel if the fetch fails.

### Tezzeret Compilation Forge

- `TezzeretForge` orchestrates blueprint compilation at startup. Provide a
  `KarnCatalog`, `UrzaLibrary`, and `TezzeretCompiler`—it will queue every
  blueprint, compile missing artifacts, and resume from interruptions via a
  simple JSON WAL. Use `CompileJobConfig(max_retries=...)` to control retry
  behaviour; failures leave the WAL intact for the next run.

### Urza Catalog

- `UrzaLibrary` now stores artifacts in SQLite with an LRU cache and WAL-based
  crash recovery. Save operations copy artifacts into the Urza root, log a WAL
  entry, upsert the catalog row, and then clear the WAL. On restart any
  residual WAL entry is replayed automatically. Use `fetch_by_tier`/`get` for
  low-latency lookups or run TezzeretForge to populate missing artifacts.

### Tamiyo Blueprint Awareness

- When Tamiyo issues a seed command, it fetches blueprint metadata from Urza
  (tier, risk, stage, quarantine flag). High-risk or quarantined blueprints
  automatically trigger a pause command, and telemetry emits
  `tamiyo.blueprint.*` metrics for Nissa. Metadata lookups are cached for
  five minutes and respect Urza’s persisted risk data.

### Field Report Persistence

- Tamiyo writes every generated field report to `var/tamiyo/field_reports.log`
  using a WAL-style binary log. Entries are retained for 24 hours by default
  (override with `TAMIYO_FIELD_REPORT_RETENTION_HOURS`) and are reloaded on
  startup so Simic always has a consistent replay source after a restart.
  Delete the log file if you need a clean slate for local testing.

## Repository Layout

- `src/esper/` — Python packages for Esper subsystems, organised by lifecycle phase.
- `tests/` — Pytest suites mirroring the source tree (`tests/tolaria`, `tests/kasmina`, etc.).
- `docs/` — Design references, implementation plans, and backlog materials.
- `.codacy/` — Shared linting and CI bootstrap configuration.

## Subsystem Overview

Each subsystem module exposes a narrow public API under `src/esper/<subsystem>/__init__.py` and includes placeholder classes that cite the authoritative detailed design section:

- **Tolaria** (`docs/design/detailed_design/01-tolaria.md`) — Training orchestrator and epoch control loop.
- **Kasmina** (`docs/design/detailed_design/02-kasmina.md`) — Seed lifecycle manager and kernel executor.
- **Tamiyo** (`docs/design/detailed_design/03-tamiyo.md`) — Strategic controller, risk governance, and telemetry hub.
- **Simic** (`docs/design/detailed_design/04-simic.md`) — Offline policy trainer using PPO + LoRA on PyTorch 2.8.
- **Karn, Tezzeret, Urza** (`docs/design/detailed_design/05-karn.md`, `06-tezzeret.md`, `08-urza.md`) — Blueprint catalog, compiler, and artifact library.
- **Leyline, Oona, Nissa** (`docs/design/detailed_design/00-leyline.md`, `09-oona.md`, `10-nissa.md`) — Contracts, messaging fabric, and observability stack.

## Next Steps

The backlog in `docs/project/backlog.md` decomposes the first implementation sprints. Slice 0 focuses on CI/tooling and Leyline contracts. Slice 1 establishes the control loop with Tolaria, Kasmina, and Tamiyo, followed by blueprint management, observability, and offline learning in subsequent slices.

Refer to `docs/project/implementation_plan.md` for full sequencing and ownership recommendations.
