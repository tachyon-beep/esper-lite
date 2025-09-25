"""Blueprint compilation pipeline tying Karn, Tezzeret, and Urza together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from esper.karn import BlueprintDescriptor, KarnCatalog
from esper.leyline import leyline_pb2
from esper.tezzeret import TezzeretCompiler
from esper.urza import UrzaLibrary


@dataclass(slots=True)
class BlueprintRequest:
    blueprint_id: str
    parameters: dict[str, float]
    training_run_id: str


@dataclass(slots=True)
class BlueprintResponse:
    metadata: BlueprintDescriptor
    artifact_path: str
    catalog_update: leyline_pb2.KernelCatalogUpdate | None = None


class BlueprintPipeline:
    """Coordinates blueprint lookup, compilation, and storage."""

    def __init__(
        self,
        catalog: KarnCatalog,
        compiler: TezzeretCompiler,
        library: UrzaLibrary,
        *,
        catalog_notifier: (
            Callable[[leyline_pb2.KernelCatalogUpdate], Awaitable[None]] | None
        ) = None,
    ) -> None:
        self._catalog = catalog
        self._compiler = compiler
        self._library = library
        self._catalog_notifier = catalog_notifier

    async def handle_request(self, request: BlueprintRequest) -> BlueprintResponse:
        """Compile the requested blueprint and persist the artifact."""

        metadata = self._catalog.validate_request(request.blueprint_id, request.parameters)
        artifact_path = self._compiler.compile(metadata, parameters=request.parameters)
        update = self._compiler.latest_catalog_update()
        result = self._compiler.latest_result()
        extras = None
        if result is not None:
            extras = {
                "guard_spec": result.guard_spec,
                "guard_digest": result.guard_digest,
                "guard_spec_summary": list(result.guard_summary),
                "compile_ms": result.compile_ms,
                "prewarm_ms": result.prewarm_ms,
                "compile_strategy": result.compile_strategy,
                "eager_fallback": result.eager_fallback,
                "inductor_cache_dir": result.inductor_cache_dir,
            }
        self._library.save(
            metadata,
            artifact_path,
            catalog_update=update,
            extras=extras,
        )
        if update and self._catalog_notifier is not None:
            await self._catalog_notifier(update)
        return BlueprintResponse(
            metadata=metadata,
            artifact_path=str(artifact_path),
            catalog_update=update,
        )


__all__ = ["BlueprintPipeline", "BlueprintRequest", "BlueprintResponse"]
