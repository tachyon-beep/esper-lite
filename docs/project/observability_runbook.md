# Observability Runbook

Document status: draft (aligned with `docs/design/detailed_design/10-nissa.md`).

## Overview

The observability stack combines Prometheus, Grafana, and a single-node
Elasticsearch instance to visualise telemetry emitted from the control loop.
`scripts/run_nissa_service.py` (also exposed as the `esper-nissa-service`
console entry point) is responsible for ingesting telemetry from Oona and
serving metrics at `http://localhost:9100/metrics`.

If Elasticsearch is unavailable the service automatically falls back to an
in-memory stub. Telemetry ingestion continues and Prometheus/Grafana remain
operational, but indexed documents are only retained for the lifetime of the
process. The service will log a warning when the stub is activated.

## Local bring-up

1. Ensure the Python environment is activated and dependencies installed.
2. Start the observability stack components:

   ```bash
   docker compose -f infra/docker-compose.observability.yml up -d
   ```

   This launches Prometheus (host networking), Grafana (port `3000` mapped
   from the container), and a single-node Elasticsearch instance listening on
   `http://localhost:9200`.

3. Launch the Nissa service runner:

   ```bash
   esper-nissa-service
   ```

   The runner creates the required Oona consumer groups with bounded retries,
   establishes telemetry ingestion, and exposes the `/metrics` and `/healthz`
   endpoints. Logs are emitted with timestamps to aid troubleshooting.

4. Run the demo workflow (`python scripts/run_demo.py`) or any workload that
   produces telemetry. Prometheus should report the `nissa` scrape target as
   `up` via `http://localhost:9090/api/v1/targets`, and Grafana will auto-load
   the `Nissa Observability` dashboard located at `/var/lib/grafana/dashboards`.

## Shutdown

1. Stop the Nissa service with `Ctrl+C` or by terminating the process.
2. Tear down the containers:

   ```bash
   docker compose -f infra/docker-compose.observability.yml down
   ```

3. Remove the Elasticsearch data volume if a clean slate is required:

   ```bash
   docker volume rm infra_elasticsearch-data
   ```

## Operational notes

- The Grafana provisioning files live under `infra/grafana/` and are mounted
  read-only. Changes through the UI must be exported back into version control
  to persist across restarts.
- Elasticsearch memory usage is capped at 512 MiB (`ES_JAVA_OPTS`) to avoid
  exhausting local developer machines. Adjust the limit if the index grows.
- The Nissa service exposes a `/healthz` endpoint suitable for container or
  systemd health probes. The ingest loop logs and retries on Redis/Oona
  interruptions without exiting the process.
