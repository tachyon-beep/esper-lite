"""Persistent embedding registries for Simic identifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(slots=True)
class EmbeddingRegistryConfig:
    path: Path
    max_size: int = 4096


class EmbeddingRegistry:
    def __init__(self, config: EmbeddingRegistryConfig) -> None:
        self._config = config
        self._path = config.path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._table: Dict[str, int] = {}
        self._load()

    def get(self, key: str) -> int:
        if not key:
            return 0
        if key in self._table:
            return self._table[key]
        index = len(self._table) + 1
        if index >= self._config.max_size:
            return hash(key) % self._config.max_size
        self._table[key] = index
        self._save()
        return index

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            self._table = {str(k): int(v) for k, v in data.items()}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._table), encoding="utf-8")

    # --- Extensions for parity/validation ---
    def snapshot(self) -> Dict[str, int]:
        """Return a copy of the registry mapping."""
        return dict(self._table)

    def digest(self) -> str:
        """Stable content digest for checkpoint parity.

        Computes a SHA-256 over the sorted JSON representation of key→index.
        """
        try:
            import hashlib
        except Exception:
            return ""
        items = sorted(self._table.items(), key=lambda kv: kv[0])
        payload = json.dumps(items, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["EmbeddingRegistry", "EmbeddingRegistryConfig"]
