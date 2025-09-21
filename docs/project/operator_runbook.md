# Esper-Lite Operator Runbook

Document status: draft (aligned with legacy detailed designs in `docs/design/detailed_design/old/`).

This runbook captures the day-to-day procedures for bringing up, validating,
monitoring, and recovering Esper-Lite’s subsystems: Tolaria, Kasmina, Tamiyo,
Oona, Nissa, Urza/Tezzeret, and Simic. Use it alongside the observability
runbook (`docs/project/observability_runbook.md`) and the authoritative design
references (`docs/design/detailed_design/old/`).

## 1. Environment & Dependencies

1. Create and activate the virtual environment; install tooling:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev]
   ```

2. Start shared infrastructure. Redis is mandatory for Oona; Prometheus,
   Grafana, and Elasticsearch back the observability stack.

   ```bash
   docker compose -f infra/docker-compose.redis.yml up -d
   docker compose -f infra/docker-compose.observability.yml up -d  # optional but recommended
   ```

3. Export any custom stream names or service URLs via `.env` (see
   `.env.example`). Defaults follow the backlog/design docs.

## 2. Subsystem Quick Reference

| Subsystem | Start/Stop | Health Checks | Key Telemetry/Logs |
| --- | --- | --- | --- |
| Tolaria | `python scripts/run_demo.py` (training loop) | Telemetry metrics (`tolaria.training.*`); review stdout logs | Training loss/latency, command annotations |
| Kasmina | Driven by Tolaria/Tamiyo via `run_demo.py` | Telemetry metrics (`kasmina.*`); `metrics_snapshot()` from Oona to watch queue depth | Isolation violations, kernel fallback usage |
| Tamiyo | `python scripts/run_demo.py` or integrate via service API | Telemetry (`tamiyo.*`), field reports at `var/tamiyo/field_reports.log` | Risk events (`pause_triggered`, `bp_quarantine`) |
| Oona | Redis Streams (`docker compose … redis.yml`), client via `esper.oona` | `redis-cli` stream lengths, `metrics_snapshot()` counters | Queue depth, rerouted/dropped messages |
| Nissa | `esper-nissa-service` (`scripts/run_nissa_service.py`) | `http://localhost:9100/healthz`, Prometheus `/metrics`, Grafana dashboards | Alert summary, scrape status |
| Urza / Tezzeret | `python -m esper.tezzeret.runner` (custom script) or run via demo | Check `urza_forge_wal.json`, artifact directory | Blueprint metadata persisted, WAL empty after success |
| Simic | `python scripts/run_demo.py` (Simic PPO cycle) | Telemetry (`simic.*`), policy updates on Oona policy stream | PPO reward/loss, validation pass flag |

## 3. Startup Procedures

### 3.1 Tolaria + Kasmina + Tamiyo (Control Loop)

1. Ensure Redis (Oona) is running.
2. Launch the demo control loop which seeds Tolaria, Tamiyo, and Kasmina with
   stub data and hooks:

   ```bash
   python scripts/run_demo.py
   ```

   - The demo performs three epochs, streams telemetry through Oona, and
     triggers Tamiyo → Kasmina actions.

3. Verify Tolaria telemetry metrics in Prometheus:

   - `tolaria.training.loss`
   - `tolaria.training.latency_ms`
   - `tolaria.seeds.active`

4. For Kasmina, confirm no isolation violations:

   ```bash
   redis-cli XLEN oona.emergency
   # Should remain low unless emergency backpressure triggered
   ```

   Telemetry `kasmina.isolation.violations` > 0 indicates seeds quarantined –
   see troubleshooting.

5. For Tamiyo, tail field reports and check the run log:

   ```bash
   tail -f var/tamiyo/field_reports.log
   ```

   Telemetry events should include `pause_triggered` or `bp_quarantine` when the
   risk engine engages.

6. Enum policy (Leyline): All lifecycle and outcome enums are defined and enforced
   by Leyline. Subsystems must use Leyline enums directly; no internal variants.
   Operational conditions (e.g., degraded or isolated) should be assessed via
   `TelemetryPacket.system_health` and events, not lifecycle stages.

### 3.2 Oona Message Bus

1. Redis is handled via Docker compose. Optional environment variables:
   `OONA_NORMAL_STREAM`, `OONA_EMERGENCY_STREAM`, `OONA_TELEMETRY_STREAM`,
   `OONA_POLICY_STREAM`.

2. Before producers connect, create consumer groups:

   ```python
   from esper.oona import OonaClient, StreamConfig
   import asyncio

   async def bootstrap():
       client = OonaClient(
           "redis://localhost",
           StreamConfig(
               normal_stream="oona.normal",
               emergency_stream="oona.emergency",
               group="ops",
               emergency_threshold=4000,
               backpressure_drop_threshold=8000,
           ),
       )
       await client.ensure_consumer_group()
       await client.close()

   asyncio.run(bootstrap())
   ```

3. Inspect queue depth/backpressure metrics:

   ```python
   async def snapshot():
       client = ...
       print(await client.metrics_snapshot())
   ```

   - `publish_rerouted` increases when NORMAL stream is congested and messages
     spill into the EMERGENCY stream.
   - `publish_dropped` increases only if the drop threshold is exceeded; treat
     as an incident.

### 3.3 Nissa Observability Stack

1. With infrastructure running, start the service runner:

   ```bash
   esper-nissa-service
   ```

2. Check liveness:

   ```bash
   curl http://localhost:9100/healthz
   curl http://localhost:9100/metrics | head
   ```

3. Use Prometheus (`http://localhost:9090`) and Grafana (`http://localhost:3000`)
   to confirm scrapes.

### 3.4 Tezzeret Forge + Urza Library

1. Use the Python REPL to run the forge (compiles default blueprint catalog):

   ```python
   from pathlib import Path
   from esper.karn import KarnCatalog
   from esper.urza import UrzaLibrary
   from esper.tezzeret import TezzeretCompiler, CompileJobConfig, TezzeretForge

   catalog = KarnCatalog()
   library = UrzaLibrary(root=Path("var/urza"))
   compiler = TezzeretCompiler(CompileJobConfig(artifact_dir=Path("var/artifacts"), max_retries=2))
   forge = TezzeretForge(catalog, library, compiler, wal_path=Path("var/urza/forge_wal.json"))
   forge.run()
   ```

2. Ensure `forge_wal.json` and compiler WAL files are absent after successful
   run; if present, examine the pending job list (see troubleshooting).

## 4. Health Checks & Monitoring

### Tolaria

- Telemetry: `tolaria.training.loss`, `tolaria.training.latency_ms`,
  `tolaria.seeds.active`.
- Logs: console output from training loop.
- Alert: `training_latency_high` (Nissa) – triggered after three consecutive
  epochs >18 ms. Response: pause training (Tamiyo conservative mode), inspect
  GPU utilisation.

### Kasmina

- Telemetry: `kasmina.isolation.violations`, `kasmina.kernel.fetch_latency_ms`.
- Alert: `kasmina_isolation_violation` (PagerDuty). Response: retire affected
  seed via Tamiyo command or `KasminaSeedManager`, inspect kernel fetch latency.

### Tamiyo

- Telemetry: `tamiyo.validation_loss`, `tamiyo.loss_delta`,
  `tamiyo.conservative_mode`, `tamiyo.blueprint.risk`.
- Field reports: `var/tamiyo/field_reports.log`.
- Alert path: when Tamiyo enters conservative mode or quarantines a blueprint,
  expect `pause_triggered` or `bp_quarantine` events. Response: verify blueprint
  metadata in Urza, ensure risk score accurate, manually toggle conservative
  mode if necessary via service API (`TamiyoService.set_conservative_mode`).

### Oona

- Metrics snapshot or Redis `XLEN` to track queue depth.
- Alert: `oona_queue_depth` (Slack). Response:
  1. Check metrics snapshot for `publish_rerouted`/`publish_dropped`.
  2. If drops occur, enable conservative mode for publishers (pause non-critical
     traffic) and scale consumers.
  3. After depth < threshold, verify `publish_dropped` is stable.

### Nissa

- Endpoints: `/healthz`, `/metrics`, `/metrics/summary`.
- Alert: Any of the core alerts (`training_latency_high`,
  `kasmina_isolation_violation`, `oona_queue_depth`,
  `tezzeret_compile_retry_high`). Response: follow subsystem-specific actions,
  acknowledge alert in routing stub (Slack/PagerDuty/email).

### Tezzeret / Urza

- Check `var/urza/forge_wal.json` after runs; non-empty WAL indicates an
  unfinished compile job. Inspect JSON for `pending` blueprint ids.
- Alert: `tezzeret_compile_retry_high` ( email ). Response: rerun forge,
  inspect compiler logs, consider raising `max_retries` temporarily.

### Simic

- Telemetry: `simic.training.reward`, `simic.training.loss`,
  `simic.validation.pass`.
- Policy updates delivered over Oona policy stream – monitor `publish_total`
  and the policy stream length.
- If validation fails (`simic.validation.pass` = 0), inspect `Tamiyo` telemetry
  for pause events and review validation reasons emitted in Simic telemetry
  events.

## 5. Troubleshooting & Recovery

### Oona Queue Depth Spike

1. Run `metrics_snapshot()`; note `publish_rerouted` and `publish_dropped`.
2. If drops >0, temporarily reduce Tolaria/Tamiyo publish rate or toggle
   conservative mode (pause non-critical producers).
3. Validate `redis-cli XLEN oona.normal` decreases; when under control, resume
   normal operations.

### Tezzeret Forge Crash Recovery

1. Inspect `var/urza/forge_wal.json` for `pending` blueprint ids.
2. Re-run `forge.run()`; it will resume from the WAL.
3. If the compiler WAL (`tezzeret_wal.json`) persists, delete the corrupt
   artifact and re-run.

### Tamiyo Conservative Mode Locked

1. Telemetry `tamiyo.conservative_mode`=1 indicates manual or automatic pause.
2. Check `tamiyo.loss_delta` and blueprint risk metrics.
3. When stable, toggle via API:

   ```python
   service.set_conservative_mode(False)
   ```

4. Monitor for new `pause_triggered` events.

### Kasmina Isolation Breach

1. Telemetry `kasmina.isolation.violations` >0 and Nissa alert triggered.
2. Use Tamiyo to issue a `SEED_OP_CULL` command for the affected seed.
3. Confirm telemetry returns to zero; rerun forge if fallback kernels were
   loaded and need refreshing.

### Simic Validation Failure

1. Inspect telemetry event `Simic policy validation failed` for reasons.
2. Review `var/tamiyo/field_reports.log` for recent reports; ensure data quality.
3. Re-run training with adjusted thresholds or revert to previous policy (Tamiyo
   retains last known-good policy in memory).

## 6. Shutdown Procedures

1. Gracefully stop application scripts (`Ctrl+C` for `run_demo.py`,
   `esper-nissa-service`).
2. If TezzeretForge is mid-run, allow it to finish; otherwise expect WAL to
   contain remaining jobs for future resumption.
3. Tear down infrastructure:

   ```bash
   docker compose -f infra/docker-compose.observability.yml down
   docker compose -f infra/docker-compose.redis.yml down
   ```

4. Optionally archive logs (`var/tamiyo`, `var/urza`) for post-mortem analysis.

## 7. Appendix – Useful Commands

- Redis stream depth:

  ```bash
  redis-cli XLEN oona.normal
  redis-cli XLEN oona.emergency
  ```

- Oona metrics snapshot (Python REPL):

  ```python
  import asyncio
  from esper.oona import OonaClient, StreamConfig

  async def main():
      client = OonaClient("redis://localhost", StreamConfig(
          normal_stream="oona.normal",
          emergency_stream="oona.emergency",
          group="ops",
      ))
      await client.ensure_consumer_group()
      print(await client.metrics_snapshot())
      await client.close()

  asyncio.run(main())
  ```

- Tamiyo field report tail:

  ```bash
  tail -f var/tamiyo/field_reports.log
  ```

- Prometheus scrape check:

  ```bash
  curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {scrapeUrl, health}'
  ```

- Forge pending jobs (JSON):

  ```bash
  cat var/urza/forge_wal.json | jq .
  ```

Keep this runbook updated as subsystems evolve. For any behaviour not covered
here, refer to the authoritative legacy design documents under
`docs/design/detailed_design/old/`.
