#!/usr/bin/env python3
"""Run Nissa fault drills to exercise alert and SLO pipelines."""

from __future__ import annotations

from esper.core.config import EsperSettings
from esper.nissa import (
    NissaIngestor,
    NissaIngestorConfig,
    run_all_drills,
)


def main() -> None:
    settings = EsperSettings()
    config = NissaIngestorConfig(
        prometheus_gateway=settings.prometheus_pushgateway,
        elasticsearch_url=settings.elasticsearch_url,
    )
    ingestor = NissaIngestor(config)
    results = run_all_drills(ingestor)
    for name, info in results.items():
        alert = info.get("alert")
        cleared = info.get("cleared")
        print(f"Drill {name}: triggered={bool(alert)} cleared={cleared}")
        if alert:
            print(f"  routes={alert.routes} value={alert.value}")


if __name__ == "__main__":
    main()

