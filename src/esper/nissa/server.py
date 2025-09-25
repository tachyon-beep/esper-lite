"""Nissa ASGI application exposing Prometheus metrics."""

from __future__ import annotations

import os

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from esper.core.config import EsperSettings

from .observability import NissaIngestor, NissaIngestorConfig


def create_app(ingestor: NissaIngestor) -> FastAPI:
    """Create a FastAPI application bound to the provided Nissa ingestor."""

    app = FastAPI(title="Nissa Telemetry API")

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:  # pragma: no cover - thin wrapper
        data = generate_latest(ingestor.registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pragma: no cover - thin wrapper
        return {"status": "ok"}

    @app.get("/metrics/summary")
    async def metrics_summary() -> dict[str, object]:
        summary = {
            metric: {
                "total": status.total,
                "violations": status.violations,
                "burn_rate": status.burn_rate,
            }
            for metric, status in ingestor.slo_summary().items()
        }
        alerts = {
            name: {
                "metric": event.metric,
                "value": event.value,
                "routes": event.routes,
            }
            for name, event in ingestor.active_alerts.items()
        }
        return {"slo": summary, "alerts": alerts}

    return app


def create_default_app() -> FastAPI:
    """Factory compatible with uvicorn --factory, using environment settings."""

    settings = EsperSettings()

    # Read NISSA_* environment overrides
    def _get_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _get_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, "")) if os.getenv(name) is not None else default
        except ValueError:
            return default

    def _get_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "")) if os.getenv(name) is not None else default
        except ValueError:
            return default

    def _get_list(name: str) -> tuple[str, ...] | None:
        raw = os.getenv(name)
        if not raw:
            return None
        return tuple(x.strip() for x in raw.split(",") if x.strip())

    config = NissaIngestorConfig(
        prometheus_gateway=settings.prometheus_pushgateway,
        elasticsearch_url=settings.elasticsearch_url,
        alerts_enabled=_get_bool("NISSA_ALERTS_ENABLED", True),
        coverage_alert_threshold=_get_float("NISSA_COVERAGE_ALERT_THRESHOLD", 0.7),
        coverage_alert_consecutive=_get_int("NISSA_COVERAGE_ALERT_CONSECUTIVE", 3),
        bsds_elevated_risk_threshold=_get_float("NISSA_BSDS_ELEVATED_RISK_THRESHOLD", 0.8),
        coverage_feature_keys=_get_list("NISSA_COVERAGE_FEATURE_KEYS"),
    )
    ingestor = NissaIngestor(config)
    return create_app(ingestor)


__all__ = ["create_app", "create_default_app"]
