"""Nissa ASGI application exposing Prometheus metrics."""

from __future__ import annotations

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

    return app


def create_default_app() -> FastAPI:
    """Factory compatible with uvicorn --factory, using environment settings."""

    settings = EsperSettings()
    config = NissaIngestorConfig(
        prometheus_gateway=settings.prometheus_pushgateway,
        elasticsearch_url=settings.elasticsearch_url,
    )
    ingestor = NissaIngestor(config)
    return create_app(ingestor)


__all__ = ["create_app", "create_default_app"]
