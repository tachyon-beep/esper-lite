"""Nissa ASGI application exposing Prometheus metrics."""

from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .observability import NissaIngestor

app = FastAPI(title="Nissa Telemetry API")


def create_app(ingestor: NissaIngestor) -> FastAPI:
    """Create a FastAPI app bound to the provided Nissa ingestor."""

    metrics_app = FastAPI()

    @metrics_app.get("/metrics")
    async def metrics_endpoint() -> Response:  # pragma: no cover - simple wrapper
        data = generate_latest(ingestor.registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    app.mount("", metrics_app)
    return app


__all__ = ["app", "create_app"]
