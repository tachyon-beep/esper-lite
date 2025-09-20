"""Blueprint compilation pipeline tying Karn, Tezzeret, and Urza together."""

from __future__ import annotations

from dataclasses import dataclass

from esper.karn import BlueprintMetadata, KarnCatalog
from esper.tezzeret import TezzeretCompiler
from esper.urza import UrzaLibrary


@dataclass(slots=True)
class BlueprintRequest:
    blueprint_id: str
    parameters: dict[str, float]
    training_run_id: str


@dataclass(slots=True)
class BlueprintResponse:
    metadata: BlueprintMetadata
    artifact_path: str


class BlueprintPipeline:
    """Coordinates blueprint lookup, compilation, and storage."""

    def __init__(
        self,
        catalog: KarnCatalog,
        compiler: TezzeretCompiler,
        library: UrzaLibrary,
    ) -> None:
        self._catalog = catalog
        self._compiler = compiler
        self._library = library

    def handle_request(self, request: BlueprintRequest) -> BlueprintResponse:
        """Compile the requested blueprint and persist the artifact."""

        metadata = self._catalog.validate_request(request.blueprint_id, request.parameters)
        artifact_path = self._compiler.compile(metadata, parameters=request.parameters)
        self._library.save(metadata, artifact_path)
        return BlueprintResponse(metadata=metadata, artifact_path=str(artifact_path))


__all__ = ["BlueprintPipeline", "BlueprintRequest", "BlueprintResponse"]
