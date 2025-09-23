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

## Operational Notes
- Counters are cumulative; prefer `increase()` for windowed views.
- `tamiyo.bsds.*_signal` are transient, suitable for alert engines, not for long‑term graphs.
- Provenance should trend toward `urabrask` as the crucible comes online; divergence (e.g., many `heuristic`) indicates producer paths still active.

## Next Steps
- When direct BSDS protobuf transport is enabled, add panels for Oona event rates (`BSDSIssued`, `BSDSFailed`) and correlate with Tamiyo hazard events.
- Add a drill‑down linking from Weatherlight indicators to the latest Tamiyo telemetry documents in Elasticsearch.

