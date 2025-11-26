"""GPU kernel caching helpers for Kasmina.

The prototype does not perform real GPU residency management, but this cache
mirrors the behaviour described in the proto delta: bounded storage with
LRU-style eviction and lightweight statistics for telemetry.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, MutableMapping


@dataclass(slots=True)
class KernelCacheStats:
    """Expose cache diagnostics for telemetry."""

    size: int
    capacity: int
    hit_rate: float
    evictions: int


class KasminaKernelCache:
    """Bounded cache with LRU eviction semantics."""

    def __init__(self, *, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._store: MutableMapping[Hashable, object] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> object | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        # Move to end to preserve LRU ordering.
        self._store.move_to_end(key)
        self._hits += 1
        return entry

    def set(self, key: Hashable, value: object) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return

        if len(self._store) >= self._capacity:
            self._store.popitem(last=False)
            self._evictions += 1
        self._store[key] = value

    def delete(self, key: Hashable) -> None:
        self._store.pop(key, None)

    def stats(self) -> KernelCacheStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return KernelCacheStats(
            size=len(self._store),
            capacity=self._capacity,
            hit_rate=hit_rate,
            evictions=self._evictions,
        )


__all__ = ["KasminaKernelCache", "KernelCacheStats"]
