"""Annotation validation helpers for Kasmina command handling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping

from esper.leyline import leyline_pb2 as pb


@dataclass(slots=True)
class AnnotationValidationResult:
    """Represents the outcome of validating command annotations."""

    accepted: bool
    reason: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SeedCommandPayload:
    """Minimal payload describing a seed command for validation."""

    seed_id: str
    blueprint_id: str
    training_run_id: str
    operation: int
    annotations: Mapping[str, str]


class CommandAnnotationValidator:
    """Validates Tamiyo → Kasmina command annotations before execution."""

    def validate(self, command_type: int, payload: object | None = None) -> AnnotationValidationResult:
        if command_type == pb.COMMAND_SEED and isinstance(payload, SeedCommandPayload):
            return self._validate_seed(payload)
        return AnnotationValidationResult(True)

    def _validate_seed(self, payload: SeedCommandPayload) -> AnnotationValidationResult:
        seed_id = payload.seed_id.strip()
        blueprint_id = payload.blueprint_id.strip()
        training_run_id = payload.training_run_id.strip()
        if not seed_id:
            return AnnotationValidationResult(
                False,
                "missing_seed_id",
                {"seed_id": payload.seed_id, "blueprint_id": blueprint_id},
            )

        if not blueprint_id:
            return AnnotationValidationResult(
                False,
                "missing_blueprint_id",
                {"seed_id": seed_id},
            )

        if not training_run_id:
            return AnnotationValidationResult(
                False,
                "missing_training_run_id",
                {"seed_id": seed_id, "blueprint_id": blueprint_id},
            )

        annotations = payload.annotations
        requires_mesh = payload.operation not in (
            pb.SEED_OP_CULL,
            pb.SEED_OP_CANCEL,
        )
        mesh_layers = annotations.get("mesh_host_layers") or annotations.get("mesh_layers")
        if requires_mesh:
            if not mesh_layers:
                return AnnotationValidationResult(
                    False,
                    "missing_mesh_layers",
                    {"seed_id": seed_id, "blueprint_id": blueprint_id},
                )
            try:
                self._parse_mesh_layers(str(mesh_layers))
            except ValueError as exc:
                return AnnotationValidationResult(
                    False,
                    "invalid_mesh_layers",
                    {
                        "seed_id": seed_id,
                        "blueprint_id": blueprint_id,
                        "error": str(exc),
                    },
                )

        requires_logits = annotations.get("confidence_logits_required", "").lower() == "true"
        if requires_logits and not annotations.get("confidence_logits"):
            return AnnotationValidationResult(
                False,
                "missing_confidence_logits",
                {"seed_id": seed_id, "blueprint_id": blueprint_id},
            )

        return AnnotationValidationResult(True)

    @staticmethod
    def _parse_mesh_layers(payload: str) -> tuple[str, ...]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = [item.strip() for item in payload.split(",")]
        else:
            if isinstance(data, str):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("mesh_layers_not_list")
        entries: list[str] = []
        for entry in data:
            if entry is None:
                continue
            name = str(entry).strip()
            if name:
                entries.append(name)
        unique = list(dict.fromkeys(entries))
        if not unique:
            raise ValueError("mesh_layers_empty")
        return tuple(unique)


__all__ = [
    "AnnotationValidationResult",
    "CommandAnnotationValidator",
    "SeedCommandPayload",
]
