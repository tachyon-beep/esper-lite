# Urabrask / Tamiyo BSDS — Dashboard Notes (Prototype)

This guide outlines suggested Grafana panels and Prometheus metrics to monitor BSDS hazard signals and provenance in the prototype.

## Data Sources
- Prometheus: Nissa exposes counters and time series
  - `esper_tamiyo_bsds_hazard_high_total{provenance}`
  - `esper_tamiyo_bsds_hazard_critical_total{provenance}`
  - Optional alert signal metrics (instant):
    - `tamiyo.bsds.hazard_high_signal` (transient)
    - `tamiyo.bsds.hazard_critical_signal` (transient)
- Weatherlight Telemetry (indicators in health payload)
  - `bsds_provenance`: last seen provenance (e.g., `urabrask`, `heuristic`)
  - `bsds_hazard`: last seen hazard band (e.g., `HIGH`, `CRITICAL`)

## Panels
- BSDS Hazard Breakdown (Bar/Stacked)
  - Query: `sum by (provenance) (increase(esper_tamiyo_bsds_hazard_critical_total[24h]))`
  - Query: `sum by (provenance) (increase(esper_tamiyo_bsds_hazard_high_total[24h]))`
  - Combine in a stacked graph for a 24h window.

- Hazard Trend (Time Series)
  - Query (Critical): `increase(esper_tamiyo_bsds_hazard_critical_total[1h])`
  - Query (High): `increase(esper_tamiyo_bsds_hazard_high_total[1h])`

- Last BSDS Provenance (Stat)
  - Source: Weatherlight `bsds_provenance` indicator (extracted from latest telemetry document).
  - Display as lowercase tag (e.g., `heuristic`, `urabrask`).

- Last BSDS Hazard (Stat)
  - Source: Weatherlight `bsds_hazard` indicator (uppercased), e.g., `HIGH`, `CRITICAL`.

- Alert Table (Optional)
  - Reflect active alerts from Nissa for BSDS:
    - `tamiyo_bsds_critical` (PagerDuty)
    - `tamiyo_bsds_high` (Slack)

## Benchmarks (Prototype)

Urabrask Bench v1 persists a JSON mirror under `extras["benchmarks"]` as a list of per‑profile dicts:
`[{name, batch_size, in_shape, dtype, p50_latency_ms, p95_latency_ms, throughput_samples_per_s, provenance}]`.

- Data source: Weatherlight → Elasticsearch (telemetry documents). If ES is unavailable, panels can be doc‑only for now.
- Suggested panels:
  - Benchmark p50 by profile (Time Series)
    - Query ES for latest telemetry where extras contains `benchmarks.name=<profile>` and plot `p50_latency_ms`.
  - Benchmark p95 by profile (Time Series)
    - Same as above using `p95_latency_ms`.
  - Throughput (samples/sec) by profile (Time Series)
  - Last Benchmark Provenance (Stat)
    - Extract `provenance` for each profile; display `runtime` vs `fallback`.
  - Device/Torch Version (Table)
    - From the `BlueprintBenchmark` proto fields via the ingestion job (optional).

Notes
- Profiles default to `batch16_f32` and `batch32_f32`; CUDA hosts may include an optional `batch32_bf16` profile.
- Keep budget small (≤100 ms per profile) and cadence low if later wired into a worker.

### Bench Telemetry (Weatherlight)

Weatherlight exposes Urabrask bench worker counters in its telemetry packet. These are not Prometheus metrics by default; index Weatherlight telemetry into Elasticsearch to chart them.

- Metrics (names as emitted in telemetry)
  - `urabrask.bench.profiles_total` — cumulative number of profiles attached
  - `urabrask.bench.failures_total` — cumulative worker failures/timeouts
  - `urabrask.bench.last_duration_ms` — duration of the last bench cycle
  - `urabrask.bench.last_processed` — number of blueprints processed in last cycle

- Grafana (ES datasource) example panels
  - Profiles Attached (Rate)
    - Query (KQL): `metrics.name: "urabrask.bench.profiles_total" and source: "weatherlight"`
    - Aggregations: Max per time bucket → Derivative → Positive Only → Moving Avg (optional) → Line
  - Bench Cycle Duration (ms)
    - Query (KQL): `metrics.name: "urabrask.bench.last_duration_ms" and source: "weatherlight"`
    - Aggregation: Avg per time bucket → Line
  - Blueprints Processed (Last Cycle)
    - Query (KQL): `metrics.name: "urabrask.bench.last_processed" and source: "weatherlight"`
    - Aggregation: Max per time bucket → Bar/Line

## Crucible Hazards (Prototype)

Crucible v1 emits additional hazard keys in the BSDS mirror (extras["bsds"]["hazards"]):
- `memory_watermark`: ok | high
- `oom_risk`: ok | risk

Suggested panels (ES datasource; doc‑only until Nissa indexes hazards):
- Memory Watermark Incidents (Table/Bar)
  - Count occurrences of `hazards.memory_watermark: "high"` over time windows.
- OOM Risk Flags (Table/Bar)
  - Count occurrences of `hazards.oom_risk: "risk"` over time windows.

Operational flags:
- `URABRASK_CRUCIBLE_MEMORY_WATERMARK_MB` — threshold in MB (default 64.0)
- `URABRASK_CRUCIBLE_ALLOW_OOM` — enable OOM probe (default false)
- `URABRASK_CRUCIBLE_SIMULATE_OOM` — CI‑safe risk flag without real allocation (default false)

## BSDS Events (Prototype)

If Oona eventing is enabled (`URABRASK_OONA_PUBLISH_ENABLED=true`), Urabrask publishes:
- `BSDSIssued` with the canonical BSDS
- `BSDSFailed` with `blueprint_id` and `reason`

Suggested panels (ES datasource; requires Oona events indexing):
- BSDS Issued per 1h (Bar/Line)
  - Query: type:BSDSIssued (or index-based filter), aggregate count over time buckets.
- BSDS Failed per 1h (Bar/Line)
  - Query: type:BSDSFailed; break down by reason where available.
- Last 20 BSDS Events (Table)
  - Columns: timestamp, type, blueprint_id, hazard (from bsds), reason (on failure)

## WAL Integrity (Doc‑Only)

If WAL entries are indexed in ES (optional ops script):
- Append Rate (Time Series)
  - Count of WAL entries ingested per time bucket.
- Integrity Check Results (Stat/Table)
  - Count of integrity errors (hash‑chain mismatches). In prototype, provide via an ops job that parses WAL and emits a heartbeat document.

## Urabrask Integrity & WAL Metrics

Weatherlight telemetry includes process‑local counters under the `urabrask.*` namespace:

- `urabrask.wal.append_errors_total` — WAL append failures (best‑effort; fail‑open)
- `urabrask.integrity_failures` — signature verification failures observed by Tamiyo

Suggested panels (Prometheus/ES via Weatherlight telemetry ingestion):
- WAL Append Errors (Rate)
  - Query: `increase(urabrask.wal.append_errors_total[10m])`
  - Alert if non‑zero for 3 consecutive intervals.
- Integrity Failures (Rate)
  - Query: `increase(urabrask.integrity_failures[10m])`
  - Alert if non‑zero for 3 consecutive intervals.


## Weatherlight Producer Stats (Telemetry)
- Weatherlight emits `urabrask.*` metrics in its telemetry packet (not Prometheus native unless ingested):
  - `urabrask.produced_total` — total BSDS attachments made by the producer
  - `urabrask.failures_total` — total failures/timeouts
  - `urabrask.last_duration_ms` — duration of the last producer cycle
  - `urabrask.last_processed` — number of blueprints processed in the last cycle
- Suggested usage:
  - Index Weatherlight telemetry to Elasticsearch (via Nissa), and chart these fields as time series in Grafana using the ES datasource.
  - Correlate `urabrask.produced_total` growth with Tamiyo BSDS hazard events and provenance trends.

## Operational Notes
- Counters are cumulative; prefer `increase()` for windowed views.
- `tamiyo.bsds.*_signal` are transient, suitable for alert engines, not for long‑term graphs.
- Provenance should trend toward `urabrask` as the crucible comes online; divergence (e.g., many `heuristic`) indicates producer paths still active.

## Next Steps
- When direct BSDS protobuf transport is enabled, add panels for Oona event rates (`BSDSIssued`, `BSDSFailed`) and correlate with Tamiyo hazard events.
- Add a drill‑down linking from Weatherlight indicators to the latest Tamiyo telemetry documents in Elasticsearch.
