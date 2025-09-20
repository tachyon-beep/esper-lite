"""Service runner for the Nissa observability pipeline.

Implements the long-lived process described in
`docs/design/detailed_design/10-nissa.md` and the operational playbook in
`docs/project/observability_runbook.md`. The runner performs three tasks:

1. Bootstrap the Oona consumer group with bounded retries.
2. Continuously drain telemetry packets into the Nissa metrics ingestor.
3. Expose Prometheus metrics via FastAPI/uvicorn on ``0.0.0.0:9100``.

When the configured Elasticsearch endpoint is unavailable the runner falls
back to an in-memory stub so local demos can proceed without the full stack.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

import uvicorn

from esper.core import EsperSettings
from esper.nissa import NissaIngestor, NissaIngestorConfig
from esper.nissa.server import create_app
from esper.oona import OonaClient, StreamConfig

logger = logging.getLogger(__name__)


class MemoryElasticsearch:
    """In-memory Elasticsearch stub for local development."""

    def __init__(self) -> None:
        self.documents: list[tuple[str, dict[str, Any]]] = []

    def index(self, index: str, document: dict[str, Any]) -> None:
        self.documents.append((index, document))


def _build_es_client(url: str) -> Any:
    """Return a working Elasticsearch client or the stub."""

    try:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(hosts=[url])
        if client.ping():  # pragma: no cover - integration behaviour
            logger.info("Connected to Elasticsearch at %s", url)
            return client
        logger.warning("Elasticsearch ping to %s failed, using in-memory stub", url)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Unable to initialise Elasticsearch client: %s", exc)
    return MemoryElasticsearch()


async def _ensure_oona_group(oona: OonaClient, *, retries: int = 5) -> None:
    """Create the Oona consumer group with bounded retries."""

    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            await oona.ensure_consumer_group()
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            if attempt == retries:
                logger.error("Failed to initialise Oona after %s attempts", attempt)
                raise
            logger.warning(
                "Oona ensure_consumer_group failed (attempt %s/%s): %s",
                attempt,
                retries,
                exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)


async def _ingest_loop(
    ingestor: NissaIngestor,
    client: OonaClient,
    stop_event: asyncio.Event,
) -> None:
    """Continuously drain telemetry from Oona into Nissa."""

    while not stop_event.is_set():
        try:
            await ingestor.consume_from_oona(
                client,
                stream=client.telemetry_stream,
                count=100,
                block_ms=1000,
            )
            await ingestor.consume_from_oona(
                client,
                stream=client.normal_stream,
                count=50,
                block_ms=500,
            )
        except asyncio.CancelledError:  # pragma: no cover - graceful shutdown
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Nissa ingestion loop error: %s", exc)
            await asyncio.sleep(1)


async def run_service(settings: EsperSettings | None = None) -> None:
    """Bootstrap and run the Nissa observability service."""

    settings = settings or EsperSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    stream_config = StreamConfig(
        normal_stream=settings.oona_normal_stream,
        emergency_stream=settings.oona_emergency_stream,
        telemetry_stream=settings.oona_telemetry_stream,
        policy_stream=settings.oona_policy_stream,
        group="nissa-service",
    )
    oona = OonaClient(settings.redis_url, config=stream_config)
    await _ensure_oona_group(oona)

    ingestor_config = NissaIngestorConfig(
        prometheus_gateway=settings.prometheus_pushgateway,
        elasticsearch_url=settings.elasticsearch_url,
    )
    ingestor = NissaIngestor(
        ingestor_config,
        es_client=_build_es_client(settings.elasticsearch_url),
    )

    app = create_app(ingestor)
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=9100,
        log_level=settings.log_level.lower(),
        loop="asyncio",
    )
    server = uvicorn.Server(uvicorn_config)
    server.install_signal_handlers = False

    stop_event = asyncio.Event()
    ingest_task = asyncio.create_task(_ingest_loop(ingestor, oona, stop_event))

    try:
        await server.serve()
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        raise
    except KeyboardInterrupt:  # pragma: no cover - console exit
        logger.info("Signal received, shutting down Nissa service")
    finally:
        stop_event.set()
        ingest_task.cancel()
        with suppress(asyncio.CancelledError):
            await ingest_task
        await oona.close()


def main() -> None:
    """Console entry point for ``esper-nissa-service``."""

    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:  # pragma: no cover - console exit
        logger.info("Nissa service interrupted")


__all__ = ["run_service", "main", "MemoryElasticsearch"]
