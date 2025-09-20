"""Leyline registry scaffolding.

Responsible for tracking contract bundle metadata and schema revisions as laid out
in `docs/design/detailed_design/00-leyline.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ContractBundle:
    name: str
    version: str
    schema_dir: Path


class ContractRegistry:
    """In-memory registry for Leyline protobuf bundles."""

    def __init__(self) -> None:
        self._bundles: dict[str, ContractBundle] = {}

    def register(self, bundle: ContractBundle) -> None:
        self._bundles[bundle.name] = bundle

    def get(self, name: str) -> ContractBundle | None:
        return self._bundles.get(name)

    def list_all(self) -> dict[str, ContractBundle]:
        return dict(self._bundles)


__all__ = ["ContractRegistry", "ContractBundle"]
